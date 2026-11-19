"""EvidenceGraph-20K selection, materialization, and leakage validation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .benchmark_loaders import load_fever
from .schema import ClaimEpisode, ProvenanceFamily, SourceEvidence
from .wiki_index import WikiIndex


SPLIT_TARGETS = {
    "train": {"SUPPORTS": 7000, "REFUTES": 7000},
    "dev": {"SUPPORTS": 1500, "REFUTES": 1500},
    "test": {"SUPPORTS": 1500, "REFUTES": 1500},
}
CORRUPTIONS = ("duplication", "paraphrase_flood", "support_refute_mix", "hidden_best_evidence")
_CONTENT_TOKEN = re.compile(r"[a-z0-9]{4,}")
_DISTRACTOR_STOP = {
    "about",
    "after",
    "before",
    "from",
    "have",
    "into",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "were",
    "which",
    "with",
}


def normalize_wiki_text(text: str) -> str:
    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
        "``": '"',
        "''": '"',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(value: str, seed: int = 103) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def evidence_references(record: dict[str, Any]) -> list[tuple[str, int]]:
    references = {
        (unicodedata.normalize("NFC", str(item[2])), int(item[3]))
        for group in record.get("evidence", [])
        for item in group
        if len(item) >= 4 and item[2] is not None and item[3] is not None
    }
    return sorted(references)


def canonical_evidence_references(record: dict[str, Any]) -> list[tuple[str, int]]:
    """Choose one complete FEVER evidence group, never the union of alternatives."""
    groups: list[list[tuple[str, int]]] = []
    for raw_group in record.get("evidence", []):
        group: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for item in raw_group:
            if len(item) < 4 or item[2] is None or item[3] is None:
                continue
            reference = (unicodedata.normalize("NFC", str(item[2])), int(item[3]))
            if reference not in seen:
                group.append(reference)
                seen.add(reference)
        if group:
            groups.append(group)
    if not groups:
        return []
    return min(groups, key=lambda group: (len(group), tuple(group)))


def topic_for(record: dict[str, Any]) -> str:
    references = canonical_evidence_references(record)
    if not references:
        raise ValueError("verifiable record has no evidence reference")
    return min(page_id for page_id, _ in references)


def topic_split(topic_id: str, seed: int = 103) -> str:
    bucket = stable_hash(topic_id, seed=seed) % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "dev"
    return "test"


def select_claims(fever_path: Path, seed: int = 103) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in load_fever(fever_path):
        label = str(record["label"])
        if label not in {"SUPPORTS", "REFUTES"} or not evidence_references(record):
            continue
        topic_id = topic_for(record)
        split = topic_split(topic_id, seed=seed)
        candidates[(split, label)].append(record)
    selected: list[dict[str, Any]] = []
    for split, label_targets in SPLIT_TARGETS.items():
        for label, target in label_targets.items():
            pool = sorted(
                candidates[(split, label)],
                key=lambda record: stable_hash(str(record["id"]), seed=seed),
            )
            if len(pool) < target:
                raise ValueError(f"insufficient {split}/{label}: {len(pool)} < {target}")
            for record in pool[:target]:
                selected.append(_selection_record(record, split, label, seed))
    return sorted(selected, key=lambda record: (record["split"], record["claim_id"]))


def _selection_record(
    record: dict[str, Any], split: str, label: str, seed: int
) -> dict[str, Any]:
    claim_id = f"fever-{record['id']}"
    return {
        "claim_id": claim_id,
        "source_record_id": str(record["id"]),
        "claim": record["claim"],
        "label": label == "SUPPORTS",
        "fever_label": label,
        "topic_id": topic_for(record),
        "split": split,
        "corruption": CORRUPTIONS[stable_hash(claim_id, seed=seed) % len(CORRUPTIONS)],
        "evidence": [
            list(reference) for reference in canonical_evidence_references(record)
        ],
    }


def select_reserve_claims(
    fever_path: Path,
    selected: list[dict[str, Any]],
    *,
    per_stratum: int = 100,
    seed: int = 103,
) -> list[dict[str, Any]]:
    """Select unused replacements within split/label/corruption strata."""
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    strata = reserve_candidate_strata(fever_path, selected, seed=seed)
    return choose_reserve_claims(
        strata,
        {key: per_stratum for key in strata},
        seed=seed,
    )


def reserve_candidate_strata(
    fever_path: Path,
    selected: list[dict[str, Any]],
    *,
    seed: int = 103,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Return every unused eligible FEVER claim grouped by frozen stratum."""

    selected_ids = {str(record["claim_id"]) for record in selected}
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in load_fever(fever_path):
        label = str(record["label"])
        if label not in {"SUPPORTS", "REFUTES"} or not canonical_evidence_references(record):
            continue
        split = topic_split(topic_for(record), seed=seed)
        candidate = _selection_record(record, split, label, seed)
        if candidate["claim_id"] in selected_ids:
            continue
        key = (split, label, candidate["corruption"])
        strata[key].append(candidate)
    expected_keys = {
        (split, label, corruption)
        for split in SPLIT_TARGETS
        for label in ("SUPPORTS", "REFUTES")
        for corruption in CORRUPTIONS
    }
    if set(strata) != expected_keys:
        raise ValueError("reserve candidates do not cover every stratum")
    return dict(strata)


def choose_reserve_claims(
    strata: dict[tuple[str, str, str], list[dict[str, Any]]],
    targets: dict[tuple[str, str, str], int],
    *,
    seed: int = 103,
) -> list[dict[str, Any]]:
    """Choose deterministic, non-uniform reserve counts from known strata."""

    unknown = set(targets) - set(strata)
    if unknown:
        raise ValueError(f"reserve targets contain unknown strata: {sorted(unknown)}")
    reserves = []
    for key in sorted(targets):
        target = int(targets[key])
        if target < 0:
            raise ValueError(f"negative reserve target for {key}")
        pool = sorted(
            strata[key],
            key=lambda record: stable_hash(str(record["claim_id"]), seed=seed),
        )
        if len(pool) < target:
            raise ValueError(
                f"insufficient reserve claims in {key}: need {target}, have {len(pool)}"
            )
        reserves.extend(pool[:target])
    if len({str(record["claim_id"]) for record in reserves}) != len(reserves):
        raise ValueError("reserve selection contains duplicate claim IDs")
    return reserves


def select_reserve_claims_by_stratum(
    fever_path: Path,
    selected: list[dict[str, Any]],
    targets: dict[tuple[str, str, str], int],
    *,
    seed: int = 103,
) -> list[dict[str, Any]]:
    """Select a demand-sized replacement pool under the frozen strata."""

    return choose_reserve_claims(
        reserve_candidate_strata(fever_path, selected, seed=seed),
        targets,
        seed=seed,
    )


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    topics: dict[str, set[str]] = defaultdict(set)
    for record in records:
        topics[record["split"]].add(record["topic_id"])
    leakage = {
        f"{left}_{right}": len(topics[left] & topics[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    return {
        "episodes": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "label_counts": dict(
            Counter(f"{record['split']}:{record['fever_label']}" for record in records)
        ),
        "corruption_counts": dict(Counter(record["corruption"] for record in records)),
        "topic_counts": {split: len(values) for split, values in topics.items()},
        "topic_leakage": leakage,
    }


def stratified_sample(
    records: list[dict[str, Any]], limit: int, seed: int = 103
) -> list[dict[str, Any]]:
    """Deterministically round-robin over split, label, and corruption strata."""
    if limit < 1 or limit > len(records):
        raise ValueError("sample limit must be within the record count")
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["split"], record["fever_label"], record["corruption"])
        strata[key].append(record)
    for key in strata:
        strata[key].sort(
            key=lambda record: stable_hash(str(record["claim_id"]), seed=seed)
        )
    positions = {key: 0 for key in strata}
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(strata[key]):
                selected.append(strata[key][position])
                positions[key] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            raise ValueError("could not fill stratified sample")
    return selected


def _source(
    *,
    source_id: str,
    text: str,
    family: ProvenanceFamily,
    stance: float,
    domain: str,
    origin_id: str,
    cited_source_ids: tuple[str, ...] = (),
    synthetic: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SourceEvidence:
    return SourceEvidence(
        source_id=source_id,
        text=text,
        family=family,
        stance=stance,
        stance_confidence=0.95 if stance else 0.5,
        domain=domain,
        origin_id=origin_id,
        cited_source_ids=cited_source_ids,
        synthetic=synthetic,
        metadata=metadata or {},
    )


def _evidence_bundle(record: dict[str, Any], index: WikiIndex) -> tuple[str, str]:
    sentences: list[str] = []
    for page_id, line_number in record["evidence"]:
        sentence = index.get_sentence(str(page_id), int(line_number))
        if sentence:
            sentences.append(normalize_wiki_text(sentence))
    if not sentences:
        raise ValueError(f"no evidence text found for {record['claim_id']}")
    reference_key = "|".join(f"{page_id}:{line_number}" for page_id, line_number in record["evidence"])
    return f"fever-bundle:{reference_key}", "\n".join(sentences)


def _content_tokens(text: str) -> set[str]:
    return set(_CONTENT_TOKEN.findall(text.casefold())) - _DISTRACTOR_STOP


def _deterministic_distractors(
    record: dict[str, Any],
    index: WikiIndex,
    evidence_pages: set[str],
    *,
    requested: int,
    max_attempts_per_source: int = 256,
) -> list[dict[str, str | int]]:
    """Choose unique-page unrelated Wikipedia sentences without an FTS scan."""
    if requested < 0:
        raise ValueError("requested distractor count must be non-negative")
    if requested == 0:
        return []
    count = index.sentence_count()
    claim_tokens = _content_tokens(str(record["claim"]))
    selected: list[dict[str, str | int]] = []
    used_pages = set(evidence_pages)
    used_texts: set[str] = set()
    for attempt in range(max_attempts_per_source * requested):
        rowid = 1 + stable_hash(f"distractor:{record['claim_id']}:{attempt}") % count
        candidate = index.get_sentence_by_rowid(rowid)
        if candidate is None or str(candidate["page_id"]) in used_pages:
            continue
        text = normalize_wiki_text(str(candidate["text"]))
        if (
            text not in used_texts
            and len(_content_tokens(text)) >= 3
            and not (claim_tokens & _content_tokens(text))
        ):
            selected.append({**candidate, "text": text, "rowid": rowid})
            used_pages.add(str(candidate["page_id"]))
            used_texts.add(text)
            if len(selected) == requested:
                return selected
    raise ValueError(
        f"could not construct {requested} unrelated distractors for {record['claim_id']}"
    )


def build_episode(
    record: dict[str, Any],
    index: WikiIndex,
    paraphrases: list[str],
    wikidata_entities: list[dict[str, Any]] | None = None,
) -> ClaimEpisode:
    origin_id, evidence_text = _evidence_bundle(record, index)
    polarity = 1.0 if record["label"] else -1.0
    evidence_pages = {str(page_id) for page_id, _ in record["evidence"]}
    wikidata_entities = wikidata_entities or []
    if wikidata_entities:
        mapped_entities = {
            str(entity.get("entity_id", "")) for entity in wikidata_entities
        }
        if any(
            not str(entity.get("entity_id", "")).startswith("Q")
            or entity.get("snapshot_date") != "2017-02-20"
            or entity.get("license") != "CC0-1.0"
            or not isinstance(entity.get("statement_property_ids"), list)
            for entity in wikidata_entities
        ):
            raise ValueError("invalid date-aligned Wikidata provenance")
        if len(mapped_entities) != len(wikidata_entities):
            raise ValueError("duplicate Wikidata evidence-page provenance")
    sources: list[SourceEvidence] = [
        _source(
            source_id=f"{record['claim_id']}-origin-0",
            text=evidence_text,
            family=ProvenanceFamily.INDEPENDENT,
            stance=polarity,
            domain="wikipedia.org",
            origin_id=origin_id,
            metadata={
                "evidence_references": record["evidence"],
                "page_ids": sorted(evidence_pages),
                "wikidata_entities": wikidata_entities,
            },
        )
    ]
    origin = sources[0]
    mirror_count = 5 if record["corruption"] == "duplication" else 1
    for copy_index in range(mirror_count):
        sources.append(
            _source(
                source_id=f"{record['claim_id']}-mirror-{copy_index}",
                text=origin.text,
                family=ProvenanceFamily.MIRROR,
                stance=polarity,
                domain=f"mirror-{copy_index}.synthetic.invalid",
                origin_id=origin.origin_id,
                synthetic=True,
            )
        )
    sources.append(
        _source(
            source_id=f"{record['claim_id']}-citation-0",
            text=f"This report cites the underlying Wikipedia source: {origin.text}",
            family=ProvenanceFamily.CITATION,
            stance=polarity,
            domain="citation.synthetic.invalid",
            origin_id=origin.origin_id,
            cited_source_ids=(origin.source_id,),
            synthetic=True,
        )
    )
    paraphrase_count = 4 if record["corruption"] == "paraphrase_flood" else 1
    if len(paraphrases) < paraphrase_count:
        raise ValueError(f"insufficient paraphrases for {record['claim_id']}")
    for paraphrase_index, text in enumerate(paraphrases[:paraphrase_count]):
        sources.append(
            _source(
                source_id=f"{record['claim_id']}-paraphrase-{paraphrase_index}",
                text=text,
                family=ProvenanceFamily.PARAPHRASE,
                stance=polarity,
                domain=f"paraphrase-{paraphrase_index}.synthetic.invalid",
                origin_id=origin.origin_id,
                synthetic=True,
            )
        )
    if record["corruption"] == "support_refute_mix":
        opposing = (
            f"Synthetic controlled counter-source: it is not the case that {record['claim']}"
            if record["label"]
            else f"Synthetic controlled counter-source: {record['claim']}"
        )
        # One genuinely independent counter-source balances the one effective
        # origin cluster without making document count itself decide the label.
        for source_index in range(1):
            sources.append(
                _source(
                    source_id=f"{record['claim_id']}-counter-{source_index}",
                    text=opposing,
                    family=ProvenanceFamily.INDEPENDENT,
                    stance=-polarity,
                    domain=f"counter-{source_index}.synthetic.invalid",
                    origin_id=f"fabricated:{record['claim_id']}:{source_index}",
                    synthetic=True,
                    metadata={"controlled_fabrication": True},
                )
            )
    # Deterministic, lexically unrelated, non-gold Wikipedia sentences fill a
    # claim-hashed 5--30 source target without expensive per-episode FTS scans.
    # Hidden-best episodes need a neutral, dependence-dominated initial top-5
    # plus three additional same-origin decoys immediately after it. The latter
    # makes the matched no-active-seek intervention causal: a dependence-aware
    # acquisition can skip the decoy and reach the hidden gold cluster, while
    # the passive next-rank policy cannot receive the same source by accident.
    # Other conditions may use their complete constructed source set at the
    # lower endpoint, preserving the registered 5--30 range overall.
    minimum_target = len(sources) + (
        8 if record["corruption"] == "hidden_best_evidence" else 0
    )
    target_count = minimum_target + stable_hash(
        f"source-count:{record['claim_id']}"
    ) % (31 - minimum_target)
    candidates = _deterministic_distractors(
        record,
        index,
        evidence_pages,
        requested=target_count - len(sources),
    )
    for distractor_index, candidate in enumerate(candidates):
        sources.append(
            _source(
                source_id=f"{record['claim_id']}-distractor-{distractor_index}",
                text=str(candidate["text"]),
                family=ProvenanceFamily.INDEPENDENT,
                stance=0.0,
                domain="wikipedia.org",
                origin_id=f"wiki:{candidate['page_id']}",
                metadata={
                    "page_id": candidate["page_id"],
                    "line_number": candidate["line_number"],
                    "selection": "stable-rowid-no-content-token-overlap-unique-page",
                },
            )
        )
    if record["corruption"] == "hidden_best_evidence":
        # Put the entire shared-origin evidence cluster after an eight-source
        # neutral decoy cluster. The initial top-5 is therefore uninformative
        # and dependence dominated; ranks six through eight are more copies.
        # A find_indep action can skip them for the first independent gold
        # source, whereas the matched passive policy exhausts its three-step
        # budget on decoys. Moving the whole gold cluster prevents a mirror,
        # citation, or paraphrase leak into the initial context.
        gold_cluster = [source for source in sources if source.origin_id == origin.origin_id]
        others = [source for source in sources if source.origin_id != origin.origin_id]
        if len(others) < 8 or any(source.stance != 0.0 for source in others[:8]):
            raise RuntimeError("hidden-best construction lacks eight neutral decoys")
        decoy_origin = others[0]
        decoy_cluster = [decoy_origin]
        for copy_index in range(7):
            decoy_cluster.append(
                _source(
                    source_id=f"{record['claim_id']}-hidden-decoy-{copy_index}",
                    text=decoy_origin.text,
                    family=ProvenanceFamily.MIRROR,
                    stance=0.0,
                    domain=f"hidden-decoy-{copy_index}.synthetic.invalid",
                    origin_id=decoy_origin.origin_id,
                    synthetic=True,
                    metadata={
                        "controlled_intervention": "dependence-dominated-neutral-prefix",
                        "copied_source_id": decoy_origin.source_id,
                    },
                )
            )
        sources = decoy_cluster + gold_cluster + others[8:]
    return ClaimEpisode(
        claim_id=record["claim_id"],
        claim=record["claim"],
        label=bool(record["label"]),
        topic_id=record["topic_id"],
        split=record["split"],
        corruption=record["corruption"],
        sources=tuple(sources),
        source_dataset="FEVER+Wikidata" if wikidata_entities else "FEVER",
        source_record_id=record["source_record_id"],
        metadata={
            "protocol": (
                "evidencegraph-20k-v5-wikidata-budget-separated-decoy"
                if wikidata_entities
                else "evidencegraph-20k-v4-budget-separated-decoy"
            ),
            "source_target_count": target_count,
        },
    )


def validate_episodes(episodes: list[ClaimEpisode], expected_count: int = 20_000) -> dict[str, Any]:
    if len(episodes) != expected_count:
        raise ValueError(f"expected {expected_count} episodes, found {len(episodes)}")
    claim_ids = [episode.claim_id for episode in episodes]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("duplicate claim IDs")
    topics: dict[str, set[str]] = defaultdict(set)
    family_counts = Counter()
    protocols = {str(episode.metadata.get("protocol", "")) for episode in episodes}
    allowed_protocols = {
        "evidencegraph-20k-v4-budget-separated-decoy",
        "evidencegraph-20k-v5-wikidata-budget-separated-decoy",
    }
    if len(protocols) != 1 or not protocols <= allowed_protocols:
        raise ValueError("episodes mix or omit the frozen construction protocol")
    wikidata_links: set[str] = set()
    hidden_best_interventions = 0
    for episode in episodes:
        topics[episode.split].add(episode.topic_id)
        families = {source.family for source in episode.sources}
        if families != set(ProvenanceFamily):
            raise ValueError(f"{episode.claim_id} does not contain all provenance families")
        family_counts.update(source.family.value for source in episode.sources)
        if episode.corruption == "hidden_best_evidence":
            hidden_best_interventions += 1
            gold = [
                source
                for source in episode.sources
                if source.source_id.endswith("-origin-0")
            ]
            if len(gold) != 1:
                raise ValueError(
                    f"{episode.claim_id} hidden-best episode omits one gold origin"
                )
            gold_origin = gold[0].origin_id
            prefix = list(episode.sources[:8])
            gold_positions = [
                index
                for index, source in enumerate(episode.sources)
                if source.origin_id == gold_origin
            ]
            if (
                len(episode.sources) < 12
                or any(source.stance != 0.0 for source in prefix)
                or len({source.origin_id for source in prefix}) != 1
                or {source.family for source in prefix}
                != {ProvenanceFamily.INDEPENDENT, ProvenanceFamily.MIRROR}
                or gold_positions != list(range(8, 12))
            ):
                raise ValueError(
                    f"{episode.claim_id} violates the hidden-best active/passive intervention"
                )
        if next(iter(protocols)) == (
            "evidencegraph-20k-v5-wikidata-budget-separated-decoy"
        ):
            if episode.source_dataset != "FEVER+Wikidata":
                raise ValueError("Wikidata protocol omits the combined source dataset")
            origins = [
                source for source in episode.sources if source.source_id.endswith("-origin-0")
            ]
            entities = origins[0].metadata.get("wikidata_entities") if len(origins) == 1 else None
            if not isinstance(entities, list) or not entities:
                raise ValueError("Wikidata protocol omits episode provenance")
            wikidata_links.update(str(entity["entity_id"]) for entity in entities)
    if any(topics[left] & topics[right] for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("topic leakage across splits")
    return {
        "protocol": next(iter(protocols)),
        "episodes": len(episodes),
        "sources": sum(len(episode.sources) for episode in episodes),
        "minimum_sources_per_episode": min(len(episode.sources) for episode in episodes),
        "maximum_sources_per_episode": max(len(episode.sources) for episode in episodes),
        "family_counts": dict(family_counts),
        "split_counts": dict(Counter(episode.split for episode in episodes)),
        "label_counts": dict(Counter(str(episode.label) for episode in episodes)),
        "wikidata_unique_entities": len(wikidata_links),
        "hidden_best_intervention_episodes": hidden_best_interventions,
    }
