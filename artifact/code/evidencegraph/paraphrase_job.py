"""Append-only local paraphrase generation with strict schema checks."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
from typing import Any

import requests

from .dataset_builder import normalize_wiki_text, read_jsonl, stable_hash
from .local_llm import OllamaClient
from .semantic_validation import (
    PARAPHRASE_NLI_MODEL,
    PARAPHRASE_NLI_PROTOCOL,
    PARAPHRASE_NLI_REVISION,
    MutualEntailmentValidator,
    protected_spans,
)
from .wiki_index import WikiIndex


PROMPT_PROFILES = {
    "faithful_v11": "faithful-segmented-overgenerate-v11-keyed-schema",
    "lexically_anchored_v12": "lexically-anchored-reordering-v12-keyed-schema",
    "structure_checklist_v13": "structure-checklist-reordering-v13-keyed-schema",
    "token_preserving_preorder_v16": (
        "token-preserving-syntactic-preordering-v16-keyed-schema"
    ),
}
BACKTRANSLATION_PROMPT_VERSION = (
    "lexically-constrained-roundtrip-en-de-v14"
)
PROTECTED_BACKTRANSLATION_PROMPT_VERSION = (
    "protected-span-roundtrip-en-de-v15"
)
DETERMINISTIC_PREORDER_PROMPT_VERSION = (
    "deterministic-constituent-preordering-v17"
)
DETERMINISTIC_COORDINATION_PROMPT_VERSION = (
    "deterministic-constituent-coordination-v18"
)
DETERMINISTIC_PSEUDOCLEFT_PROMPT_VERSION = (
    "deterministic-copular-pseudocleft-v19"
)
DETERMINISTIC_EXPANDED_SYNTAX_PROMPT_VERSION = (
    "deterministic-dependency-genitive-reordering-v20"
)
DEFAULT_PROMPT_PROFILE = "faithful_v11"
PROMPT_VERSION = PROMPT_PROFILES[DEFAULT_PROMPT_PROFILE]
SUPPORTED_PROMPT_VERSIONS = frozenset(
    [
        *PROMPT_PROFILES.values(),
        BACKTRANSLATION_PROMPT_VERSION,
        PROTECTED_BACKTRANSLATION_PROMPT_VERSION,
        DETERMINISTIC_PREORDER_PROMPT_VERSION,
        DETERMINISTIC_COORDINATION_PROMPT_VERSION,
        DETERMINISTIC_PSEUDOCLEFT_PROMPT_VERSION,
        DETERMINISTIC_EXPANDED_SYNTAX_PROMPT_VERSION,
    ]
)
CANDIDATES_PER_ITEM = 8
REQUIRED_VALIDATED = 4


@contextmanager
def exclusive_output_lock(output_path: Path):
    """Prevent two append-only generators from snapshotting the same cache."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another paraphrase generator holds the output lock: {output_path}"
            ) from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def candidate_requirements(record: dict[str, Any]) -> tuple[int, int]:
    """Return (overgenerated candidates, required validated) for an episode."""
    return (8, 4) if record.get("corruption") == "paraphrase_flood" else (2, 1)


def load_completed(
    path: Path, *, expected_prompt_version: str | None = None
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = read_jsonl(path)
    if expected_prompt_version is not None:
        mismatched = [
            record
            for record in records
            if record.get("prompt_version") != expected_prompt_version
        ]
        if mismatched:
            raise ValueError("generation output mixes prompt protocols")
    return {str(record["claim_id"]): record for record in records if record.get("status") == "generated"}


def select_pending(
    selection: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Freeze a limit-defined cohort before removing completed records."""
    cohort = selection if limit is None else selection[:limit]
    return [record for record in cohort if record["claim_id"] not in completed]


def source_segments(record: dict[str, Any], index: WikiIndex) -> list[str]:
    sentences = []
    for page_id, line_number in record["evidence"]:
        sentence = index.get_sentence(str(page_id), int(line_number))
        if sentence:
            sentences.append(normalize_wiki_text(sentence))
    if not sentences:
        raise ValueError(f"missing source evidence for {record['claim_id']}")
    return sentences


def source_text(record: dict[str, Any], index: WikiIndex) -> str:
    return "\n".join(source_segments(record, index))


def _prompt(
    items: list[tuple[str, str]],
    candidates_per_item: int,
    *,
    prompt_version: str = PROMPT_VERSION,
    item_constraints: dict[str, Any] | None = None,
) -> str:
    payload = []
    for claim_id, text in items:
        item = {
            "id": claim_id,
            "original": text,
            "protected_spans_copy_exactly": sorted(protected_spans(text)),
        }
        if item_constraints is not None:
            if claim_id not in item_constraints:
                raise ValueError(f"missing structural checklist for {claim_id}")
            item["required_structure_checklist"] = item_constraints[claim_id]
        payload.append(item)
    if prompt_version == PROMPT_PROFILES["faithful_v11"]:
        protocol = (
            "Do not change, substitute, add, or remove any named entity, number, place, date, "
            "relation, role, negation, comparison, or factual claim. Before writing, silently "
            "make a checklist of every factual clause and verify every output against it. If a "
            "role or relation occurs more than once, retain every occurrence and its attachment "
            "to the correct entity (for example, CEO of A and CEO of B are two separate facts). "
            "Never summarize or shorten a compound original. "
        )
    elif prompt_version == PROMPT_PROFILES["lexically_anchored_v12"]:
        protocol = (
            "Use every original content word and preserve its lemma: retain every noun, proper "
            "noun, verb, adjective, and adverb rather than substituting a synonym. Preserve the "
            "original tense, aspect, modality, negation, comparison, named entities, numbers, "
            "places, dates, bracketed pronunciation spans, and every subject/object relation. "
            "Only reorder clauses, alter function words or punctuation, or split/join sentences. "
            "Active/passive reordering is allowed only when the same agents and patients remain "
            "attached to the same predicates. Do not return the original string unchanged, and "
            "never summarize or shorten a compound original. "
        )
    elif prompt_version == PROMPT_PROFILES["structure_checklist_v13"]:
        if item_constraints is None:
            raise ValueError("structure-checklist prompt requires item constraints")
        protocol = (
            "Each item contains a parser-derived required_structure_checklist. Before returning "
            "an output, verify every listed content lemma count, relation marker, aspect/modal "
            "marker, semantic role, temporal attachment, and phonetic span. Preserve the listed "
            "predicate agents and patients exactly and do not substitute a synonym for a listed "
            "lemma. The checklist is for silent verification only; do not print it. Obtain surface "
            "diversity only by reordering compatible clauses, changing function words or "
            "punctuation, or splitting/joining sentences. Do not return the original string "
            "unchanged, summarize it, or borrow content from another item. "
        )
    elif prompt_version == PROMPT_PROFILES["token_preserving_preorder_v16"]:
        if item_constraints is None:
            raise ValueError("token-preserving prompt requires item constraints")
        protocol = (
            "Each item contains a parser-derived required_structure_checklist. Preserve every "
            "listed content lemma, predicate agent and patient, relation marker, auxiliary, "
            "aspect/modal marker, temporal attachment, and phonetic span. Copy every original "
            "noun, proper noun, verb, adjective, adverb, number, and named-entity token without "
            "synonym substitution, deletion, or insertion. Preserve each predicate's original "
            "voice, subject, object, auxiliaries, and modifier attachment. Create diversity only "
            "by moving an intact temporal, locative, adverbial, parenthetical, or subordinate-"
            "clause constituent to another grammatical position and adjusting punctuation or "
            "the capitalization of a moved function word. Never convert active voice to passive "
            "voice or passive to active. Never invert subject and object, turn an object into a "
            "prepositional phrase, split a named entity, replace a content word, summarize, or "
            "copy the original string unchanged. The checklist is silent and must not appear in "
            "the output. "
        )
    else:
        raise ValueError(f"unknown paraphrase prompt protocol: {prompt_version}")
    diversity = (
        "Use two different safe constituent placements; never use voice conversion or lexical "
        "substitution. "
        if prompt_version == PROMPT_PROFILES["token_preserving_preorder_v16"]
        else "Use distinct syntax such as active voice, passive voice, reordered clauses, and "
        "sentence splitting, while preserving the same facts. "
    )
    return (
        f"For each item, produce exactly {candidates_per_item} meaning-preserving paraphrases. "
        + protocol
        + "For each item, every string in "
        "protected_spans_copy_exactly must occur verbatim in every output. All outputs must "
        "be unique after case-folding. "
        + diversity
        + "Faithfulness is more important than diversity. "
        "Return one JSON object with exactly one key named items. Its value must be an "
        "object keyed by each exact input id. Each id value must be an array of "
        f"exactly {candidates_per_item} strings. Input: "
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse(
    payload: Any,
    expected_ids: list[str],
    candidates_per_item: int,
) -> dict[str, list[str]]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], dict):
        raise ValueError("response must contain exactly one keyed items object")
    if set(payload["items"]) != set(expected_ids):
        raise ValueError("response IDs do not match the request")
    result: dict[str, list[str]] = {}
    for expected_id in expected_ids:
        values = payload["items"][expected_id]
        if (
            not isinstance(values, list)
            or len(values) != candidates_per_item
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError(
                f"each item must contain exactly {candidates_per_item} string paraphrases"
            )
        result[expected_id] = values
    return result


def response_schema(expected_ids: list[str], candidates_per_item: int) -> dict[str, Any]:
    """Constrain Ollama output cardinality before applying strict semantic parsing."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": expected_ids,
                "properties": {
                    expected_id: {
                        "type": "array",
                        "minItems": candidates_per_item,
                        "maxItems": candidates_per_item,
                        "items": {"type": "string", "minLength": 1},
                    }
                    for expected_id in expected_ids
                },
            }
        },
    }


def generation_batches(
    records: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    """Group by candidate requirement, preserving order within each tier."""
    batches: list[list[dict[str, Any]]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {
        (2, 1): [],
        (8, 4): [],
    }
    for record in records:
        grouped[candidate_requirements(record)].append(record)
    for requirement in ((2, 1), (8, 4)):
        tier = grouped[requirement]
        batches.extend(
            tier[offset : offset + batch_size]
            for offset in range(0, len(tier), batch_size)
        )
    return batches


def generation_token_budget(
    items: list[tuple[str, str]],
    candidates_per_item: int,
    *,
    scale: float = 1.0,
) -> int:
    """Budget output from requested copy volume, with a bounded context guard.

    Faithful outputs are approximately as long as their inputs. Two characters
    per token is deliberately conservative for names, punctuation, and JSON
    structure; the cap preserves room in the 32K local context window.
    """
    if not items or candidates_per_item < 1 or scale < 1.0:
        raise ValueError("token budgeting requires source items and candidates")
    source_characters = sum(len(text) for _, text in items)
    estimated = math.ceil(source_characters * candidates_per_item / 2.0)
    estimated += 100 * len(items) + 200
    estimated = math.ceil(estimated * scale)
    return min(16_000, max(math.ceil(500 * scale), estimated))


def unique_casefolded(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.casefold().strip()
        if normalized not in seen:
            unique.append(value)
            seen.add(normalized)
    return unique


def validate_generated_record_shape(
    record: dict[str, Any], *, expected_prompt_version: str | None = None
) -> None:
    """Reject malformed generated rows even when the generation audit is skipped."""
    claim_id = str(record.get("claim_id", "")).strip()
    model = str(record.get("model", "")).strip()
    source_record_id = str(record.get("source_record_id", "")).strip()
    original = str(record.get("original", "")).strip()
    if not claim_id or not model or not source_record_id or not original:
        raise ValueError("generated record is missing identity, model, or original text")
    prompt_version = record.get("prompt_version")
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"generated record uses the wrong prompt protocol: {claim_id}")
    if expected_prompt_version is not None and prompt_version != expected_prompt_version:
        raise ValueError(f"generated record uses the wrong prompt protocol: {claim_id}")
    candidates_requested = int(record.get("candidates_requested", -1))
    required_validated = int(record.get("required_validated", -1))
    if (candidates_requested, required_validated) not in {(2, 1), (8, 4)}:
        raise ValueError(f"invalid generated candidate requirement for {claim_id}")
    paraphrases = record.get("paraphrases")
    if (
        not isinstance(paraphrases, list)
        or len(paraphrases) != candidates_requested
        or any(not isinstance(value, str) or not value.strip() for value in paraphrases)
    ):
        raise ValueError(f"invalid generated candidate content for {claim_id}")


def combine_segment_paraphrases(
    parsed: dict[str, list[str]],
    segment_ids: list[str],
    candidates_per_item: int,
) -> list[str]:
    return [
        "\n".join(parsed[segment_id][index] for segment_id in segment_ids)
        for index in range(candidates_per_item)
    ]


def _generate_unlocked(
    selection_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    model: str,
    batch_size: int = 4,
    limit: int | None = None,
    seed: int = 103,
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
) -> dict[str, int | float]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"unknown paraphrase prompt profile: {prompt_profile}")
    prompt_version = PROMPT_PROFILES[prompt_profile]
    selection = read_jsonl(selection_path)
    completed = load_completed(
        output_path, expected_prompt_version=prompt_version
    )
    pending = select_pending(selection, completed, limit)
    client = OllamaClient()
    index = WikiIndex(index_path)
    constraint_gate = None
    if prompt_profile in {
        "structure_checklist_v13",
        "token_preserving_preorder_v16",
    }:
        from .structural_validation import SpacyLexicallyAnchoredGate

        constraint_gate = SpacyLexicallyAnchoredGate("en_core_web_sm")
    generated = failures = prompt_tokens = completion_tokens = 0
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for batch in generation_batches(pending, batch_size):
                candidates_per_item, required_validated = candidate_requirements(batch[0])
                record_items: list[
                    tuple[dict[str, Any], list[str], list[str]]
                ] = []
                items: list[tuple[str, str]] = []
                for record in batch:
                    segments = source_segments(record, index)
                    segment_ids = [
                        f"{record['claim_id']}::segment-{segment_index}"
                        for segment_index in range(len(segments))
                    ]
                    items.extend(zip(segment_ids, segments))
                    record_items.append((record, segments, segment_ids))
                max_tokens = generation_token_budget(
                    items,
                    candidates_per_item,
                    scale=(
                        4.0
                        if prompt_profile == "token_preserving_preorder_v16"
                        else 1.0
                    ),
                )
                try:
                    expected_ids = [claim_id for claim_id, _ in items]
                    item_constraints = (
                        {
                            claim_id: constraint_gate.prompt_constraints(text)
                            for claim_id, text in items
                        }
                        if constraint_gate is not None
                        else None
                    )
                    payload, response = client.chat_json(
                        model,
                        [
                            {
                                "role": "system",
                                "content": "You are a precision paraphraser. Return valid JSON only.",
                            },
                            {
                                "role": "user",
                                "content": _prompt(
                                    items,
                                    candidates_per_item,
                                    prompt_version=prompt_version,
                                    item_constraints=item_constraints,
                                ),
                            },
                        ],
                        temperature=0.0,
                        seed=seed + stable_hash(str(batch[0]["claim_id"]), seed=seed) % 1_000_000,
                        max_tokens=max_tokens,
                        json_schema=response_schema(expected_ids, candidates_per_item),
                    )
                    parsed = _parse(
                        payload,
                        expected_ids,
                        candidates_per_item,
                    )
                    prompt_tokens += response.prompt_tokens
                    completion_tokens += response.completion_tokens
                    for record, segments, segment_ids in record_items:
                        claim_id = str(record["claim_id"])
                        original = "\n".join(segments)
                        result = {
                            "claim_id": claim_id,
                            "source_record_id": record["source_record_id"],
                            "model": response.model,
                            "prompt_version": prompt_version,
                            "candidates_requested": candidates_per_item,
                            "required_validated": required_validated,
                            "max_tokens_requested": max_tokens,
                            "original": original,
                            "paraphrases": combine_segment_paraphrases(
                                parsed, segment_ids, candidates_per_item
                            ),
                            "status": "generated",
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                            "batch_wall_seconds": response.wall_seconds,
                        }
                        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        generated += 1
                except (
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                    requests.RequestException,
                ) as error:
                    failure = {
                        "claim_ids": [record["claim_id"] for record in batch],
                        "model": model,
                        "prompt_version": prompt_version,
                        "status": "schema_failure",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:300],
                        "max_tokens_requested": max_tokens,
                    }
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
                    handle.flush()
                    failures += len(batch)
    finally:
        index.close()
    return {
        "generated": generated,
        "schema_failures": failures,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def generate(
    selection_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    model: str,
    batch_size: int = 4,
    limit: int | None = None,
    seed: int = 103,
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
) -> dict[str, int | float]:
    """Generate while holding an exclusive lock for the append-only cache."""

    with exclusive_output_lock(output_path):
        return _generate_unlocked(
            selection_path,
            index_path,
            output_path,
            model=model,
            batch_size=batch_size,
            limit=limit,
            seed=seed,
            prompt_profile=prompt_profile,
        )


def validate_cache(
    raw_path: Path,
    output_path: Path,
    *,
    model_name: str = PARAPHRASE_NLI_MODEL,
    model_revision: str = PARAPHRASE_NLI_REVISION,
    validator_protocol: str = PARAPHRASE_NLI_PROTOCOL,
    threshold: float = 0.80,
    device: str = "cpu",
    limit: int | None = None,
    record_batch_size: int = 16,
    pair_batch_size: int = 32,
) -> dict[str, int]:
    if record_batch_size < 1 or pair_batch_size < 1:
        raise ValueError("validation batch sizes must be positive")
    completed_ids: set[str] = set()
    if output_path.exists():
        for record in read_jsonl(output_path):
            claim_id = str(record["claim_id"])
            if claim_id in completed_ids:
                raise ValueError(f"duplicate validation record for {claim_id}")
            if record.get("status") not in {"validated", "semantic_failure"}:
                raise ValueError(f"invalid validation status for {claim_id}")
            if (
                record.get("validator_model") != model_name
                or record.get("validator_revision") != model_revision
                or record.get("validator_protocol") != validator_protocol
                or float(record.get("threshold", -1.0)) != threshold
            ):
                raise ValueError(
                    "validation output mixes models, revisions, protocols, or thresholds"
                )
            completed_ids.add(claim_id)

    generated: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(raw_path):
        if record.get("status") != "generated":
            continue
        validate_generated_record_shape(record)
        claim_id = str(record["claim_id"])
        if claim_id in generated:
            raise ValueError(f"duplicate generated claim in raw cache: {claim_id}")
        generated[claim_id] = record
    absent = completed_ids - set(generated)
    if absent:
        raise ValueError("validation output contains claims absent from its raw cache")
    pending = [record for claim_id, record in generated.items() if claim_id not in completed_ids]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return {"validated": 0, "semantic_failures": 0}
    validator = MutualEntailmentValidator(
        model_name=model_name,
        revision=model_revision,
        threshold=threshold,
        device=device,
    )
    validated = failures = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), record_batch_size):
            batch = pending[offset : offset + record_batch_size]
            batch_results = validator.validate_records(
                [(record["original"], record["paraphrases"]) for record in batch],
                batch_size=pair_batch_size,
            )
            for record, results in zip(batch, batch_results):
                required_validated = int(record.get("required_validated", REQUIRED_VALIDATED))
                accepted = unique_casefolded([
                    candidate
                    for candidate, result in zip(record["paraphrases"], results)
                    if result.accepted
                ])
                output = {
                    "claim_id": record["claim_id"],
                    "source_record_id": record["source_record_id"],
                    "generator_model": record["model"],
                    "prompt_version": record["prompt_version"],
                    "validator_model": model_name,
                    "validator_revision": model_revision,
                    "validator_protocol": validator_protocol,
                    "threshold": threshold,
                    "required_validated": required_validated,
                    "original": record["original"],
                    "paraphrases": accepted[:required_validated],
                    "n_candidates": len(record["paraphrases"]),
                    "n_accepted": len(accepted),
                    "candidate_results": [
                        {
                            "candidate": candidate,
                            "accepted": result.accepted,
                            "forward_entailment": result.forward_entailment,
                            "reverse_entailment": result.reverse_entailment,
                            "reasons": list(result.reasons),
                        }
                        for candidate, result in zip(record["paraphrases"], results)
                    ],
                    "status": (
                        "validated"
                        if len(accepted) >= required_validated
                        else "semantic_failure"
                    ),
                }
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                if output["status"] == "validated":
                    validated += 1
                else:
                    failures += 1
    return {"validated": validated, "semantic_failures": failures}
