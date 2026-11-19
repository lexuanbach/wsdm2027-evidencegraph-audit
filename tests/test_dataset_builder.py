from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.controller import oracle_action
from evidencegraph.controller_rollout import rollout_controller
from evidencegraph.controller_training import (
    constructed_dependence,
    constructed_nli_probabilities,
)
from evidencegraph.dataset_builder import (
    build_episode,
    canonical_evidence_references,
    choose_reserve_claims,
    normalize_wiki_text,
    selection_statistics,
    stable_hash,
    stratified_sample,
    topic_split,
    validate_episodes,
)


class _OracleController:
    def predict(self, features):
        return np.asarray([oracle_action(row) for row in features])


def test_hash_and_topic_split_are_deterministic() -> None:
    assert stable_hash("Topic") == stable_hash("Topic")
    assert topic_split("Topic") == topic_split("Topic")


def test_selection_statistics_detects_no_topic_overlap() -> None:
    records = [
        {"split": "train", "topic_id": "a", "fever_label": "SUPPORTS", "corruption": "duplication"},
        {"split": "dev", "topic_id": "b", "fever_label": "REFUTES", "corruption": "paraphrase_flood"},
        {"split": "test", "topic_id": "c", "fever_label": "SUPPORTS", "corruption": "hidden_best_evidence"},
    ]
    stats = selection_statistics(records)
    assert stats["topic_leakage"] == {"train_dev": 0, "train_test": 0, "dev_test": 0}


def test_normalize_wiki_text_removes_fever_bracket_tokens() -> None:
    assert normalize_wiki_text("Person -LRB- born 1976 -RRB- , writer .") == "Person (born 1976), writer."


def test_stratified_sample_covers_available_strata() -> None:
    records = []
    for split in ("train", "dev"):
        for label in ("SUPPORTS", "REFUTES"):
            for corruption in ("duplication", "paraphrase_flood"):
                records.append(
                    {
                        "claim_id": f"{split}-{label}-{corruption}",
                        "split": split,
                        "fever_label": label,
                        "corruption": corruption,
                    }
                )
    sample = stratified_sample(records, limit=len(records))
    assert len(sample) == 8
    assert len(
        {(r["split"], r["fever_label"], r["corruption"]) for r in sample}
    ) == 8


def test_canonical_evidence_uses_one_shortest_complete_group() -> None:
    record = {
        "evidence": [
            [[1, 1, "Page_A", 0], [1, 1, "Page_B", 2]],
            [[2, 2, "Page_C", 3]],
        ]
    }
    assert canonical_evidence_references(record) == [("Page_C", 3)]


def test_canonical_evidence_normalizes_unicode_page_ids() -> None:
    record = {"evidence": [[[1, 1, "Jose\u0301_Mari\u0301a_Chaco\u0301n", 0]]]}

    assert canonical_evidence_references(record) == [("José_María_Chacón", 0)]


def test_choose_reserve_claims_supports_nonuniform_demand() -> None:
    first = ("train", "SUPPORTS", "duplication")
    second = ("test", "REFUTES", "paraphrase_flood")
    strata = {
        first: [{"claim_id": f"a-{index}"} for index in range(4)],
        second: [{"claim_id": f"b-{index}"} for index in range(3)],
    }

    selected = choose_reserve_claims(strata, {first: 3, second: 1})

    assert len(selected) == 4
    assert sum(str(record["claim_id"]).startswith("a-") for record in selected) == 3
    assert sum(str(record["claim_id"]).startswith("b-") for record in selected) == 1


def test_choose_reserve_claims_fails_before_truncating_demand() -> None:
    key = ("dev", "REFUTES", "hidden_best_evidence")

    with pytest.raises(ValueError, match="need 2, have 1"):
        choose_reserve_claims({key: [{"claim_id": "only"}]}, {key: 2})


class _EpisodeIndex:
    def get_sentence(self, page_id, line_number):
        return "Gold evidence supports the claim."

    def sentence_count(self):
        return 10_000

    def get_sentence_by_rowid(self, rowid):
        return {
            "page_id": f"Unrelated_Page_{rowid}",
            "line_number": 0,
            "text": f"Zebra quasar meadow lantern {rowid}.",
        }


def test_episode_claim_hashed_source_count_is_within_registered_range() -> None:
    record = {
        "claim_id": "fever-1",
        "source_record_id": "1",
        "claim": "A gold claim about an entity.",
        "label": True,
        "topic_id": "Gold_Page",
        "split": "train",
        "corruption": "duplication",
        "evidence": [["Gold_Page", 0]],
    }

    episode = build_episode(record, _EpisodeIndex(), ["Faithful paraphrase."])

    assert 5 <= len(episode.sources) <= 30
    assert len(episode.sources) == episode.metadata["source_target_count"]
    distractors = [source for source in episode.sources if "-distractor-" in source.source_id]
    assert len({source.origin_id for source in distractors}) == len(distractors)


def test_episode_binds_date_aligned_wikidata_without_changing_sources() -> None:
    record = {
        "claim_id": "fever-wikidata",
        "source_record_id": "wikidata",
        "claim": "A gold claim about an entity.",
        "label": True,
        "topic_id": "Gold_Page",
        "split": "train",
        "corruption": "duplication",
        "evidence": [["Gold_Page", 0]],
    }
    provenance = [{
        "entity_id": "Q123",
        "enwiki_title": "Gold Page",
        "lastrevid": 10,
        "modified": "2017-02-19T00:00:00Z",
        "statement_property_ids": ["P31"],
        "snapshot_date": "2017-02-20",
        "license": "CC0-1.0",
    }]

    plain = build_episode(record, _EpisodeIndex(), ["Faithful paraphrase."])
    enriched = build_episode(
        record,
        _EpisodeIndex(),
        ["Faithful paraphrase."],
        wikidata_entities=provenance,
    )

    assert len(enriched.sources) == len(plain.sources)
    assert enriched.source_dataset == "FEVER+Wikidata"
    assert enriched.metadata["protocol"] == (
        "evidencegraph-20k-v5-wikidata-budget-separated-decoy"
    )
    origin = next(source for source in enriched.sources if source.source_id.endswith("-origin-0"))
    assert origin.metadata["wikidata_entities"] == provenance


def test_episode_hashes_cover_both_source_count_endpoints() -> None:
    counts = set()
    for record_id in range(200):
        record = {
            "claim_id": f"fever-{record_id}",
            "source_record_id": str(record_id),
            "claim": "A gold claim about an entity.",
            "label": True,
            "topic_id": "Gold_Page",
            "split": "train",
            "corruption": "support_refute_mix",
            "evidence": [["Gold_Page", 0]],
        }
        counts.add(len(build_episode(record, _EpisodeIndex(), ["Faithful paraphrase."]).sources))

    assert min(counts) == 5
    assert max(counts) == 30


def test_hidden_best_separates_active_seek_from_passive_next_rank() -> None:
    record = {
        "claim_id": "fever-hidden",
        "source_record_id": "hidden",
        "claim": "A gold claim about an entity.",
        "label": True,
        "topic_id": "Gold_Page",
        "split": "test",
        "corruption": "hidden_best_evidence",
        "evidence": [["Gold_Page", 0]],
    }

    episode = build_episode(record, _EpisodeIndex(), ["Faithful paraphrase."])

    assert len(episode.sources) >= 12
    assert all(source.stance == 0.0 for source in episode.sources[:5])
    decoy_origin = episode.sources[0].origin_id
    assert all(source.origin_id == decoy_origin for source in episode.sources[:8])
    assert {source.family.value for source in episode.sources[:8]} == {
        "independent",
        "mirror",
    }
    origin_ids = {
        source.origin_id
        for source in episode.sources
        if source.source_id.endswith("-origin-0")
    }
    assert len(origin_ids) == 1
    origin_id = next(iter(origin_ids))
    assert all(source.origin_id != origin_id for source in episode.sources[:8])
    assert all(source.origin_id == origin_id for source in episode.sources[8:12])
    assert {source.family.value for source in episode.sources if source.origin_id == origin_id} == {
        "independent",
        "mirror",
        "citation",
        "paraphrase",
    }

    records = [source.to_dict() for source in episode.sources]
    probabilities = [constructed_nli_probabilities(source) for source in records]
    dependence = constructed_dependence(records)
    active = rollout_controller(
        probabilities,
        dependence,
        _OracleController(),
        initial_sources=5,
        action_budget=3,
    )
    passive = rollout_controller(
        probabilities,
        dependence,
        _OracleController(),
        initial_sources=5,
        action_budget=3,
        enable_independent_seek=False,
    )

    assert active["actions"] == ["find_indep", "answer"]
    assert active["selected_source_indices"] == [0, 1, 2, 3, 4, 8]
    assert passive["selected_source_indices"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert all(
        episode.sources[index].origin_id != origin_id
        for index in passive["selected_source_indices"]
    )
    summary = validate_episodes([episode], expected_count=1)
    assert summary["hidden_best_intervention_episodes"] == 1
