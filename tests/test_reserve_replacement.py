from __future__ import annotations

import pytest

from evidencegraph.reserve_replacement import (
    finalize_selection,
    reserve_retry_selection,
)


def _selection(claim_id: str, split: str, label: str, corruption: str, topic: str) -> dict:
    return {
        "claim_id": claim_id,
        "split": split,
        "fever_label": label,
        "corruption": corruption,
        "topic_id": topic,
    }


def _validation(claim_id: str, status: str) -> dict:
    return {"claim_id": claim_id, "status": status, "paraphrases": ["rewrite"]}


def test_reserve_retry_selection_matches_failed_strata() -> None:
    primary = [
        _selection("p1", "train", "SUPPORTS", "duplication", "t1"),
        _selection("p2", "dev", "REFUTES", "paraphrase_flood", "t2"),
    ]
    reserves = [
        _selection(f"r{index}", "train", "SUPPORTS", "duplication", f"rt{index}")
        for index in range(5)
    ]
    validation = [_validation("p1", "semantic_failure"), _validation("p2", "validated")]

    selected = reserve_retry_selection(
        primary, reserves, validation, per_failure=2, round_index=0
    )

    assert len(selected) == 2
    assert all(record["split"] == "train" for record in selected)


def test_finalize_selection_preserves_strata_and_records_manifest() -> None:
    primary = [
        _selection("p1", "train", "SUPPORTS", "duplication", "t1"),
        _selection("p2", "dev", "REFUTES", "paraphrase_flood", "t2"),
        _selection("p3", "test", "SUPPORTS", "hidden_best_evidence", "t3"),
    ]
    reserves = [
        _selection("r1", "train", "SUPPORTS", "duplication", "rt1"),
        _selection("r2", "train", "SUPPORTS", "duplication", "rt2"),
    ]
    validation = [
        _validation("p1", "semantic_failure"),
        _validation("p2", "validated"),
        _validation("p3", "validated"),
        _validation("r1", "semantic_failure"),
        _validation("r2", "validated"),
    ]

    final, replacements, statistics = finalize_selection(primary, reserves, validation)

    assert {record["claim_id"] for record in final} == {"r2", "p2", "p3"}
    assert replacements == [
        {
            "primary_claim_id": "p1",
            "reserve_claim_id": "r2",
            "split": "train",
            "fever_label": "SUPPORTS",
            "corruption": "duplication",
        }
    ]
    assert statistics["replacement_count"] == 1
    assert not any(statistics["topic_leakage"].values())


def test_finalize_selection_rejects_incomplete_primary_validation() -> None:
    primary = [_selection("p1", "train", "SUPPORTS", "duplication", "t1")]

    with pytest.raises(ValueError, match="incomplete primary validation"):
        finalize_selection(primary, [], [])


def test_finalize_selection_replaces_provenance_ineligible_primary() -> None:
    primary = [_selection("p1", "train", "SUPPORTS", "duplication", "t1")]
    reserves = [
        _selection("r1", "train", "SUPPORTS", "duplication", "rt1"),
        _selection("r2", "train", "SUPPORTS", "duplication", "rt2"),
    ]
    validation = [
        _validation("p1", "validated"),
        _validation("r1", "validated"),
        _validation("r2", "validated"),
    ]

    final, replacements, statistics = finalize_selection(
        primary,
        reserves,
        validation,
        eligible_claim_ids={"r2"},
    )

    assert [record["claim_id"] for record in final] == ["r2"]
    assert replacements[0]["primary_claim_id"] == "p1"
    assert statistics["provenance_coverage_replacement_count"] == 1
