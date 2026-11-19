from __future__ import annotations

import pytest

from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from run_paraphrase_reserve_recovery import (
    freeze_combined_validation,
    provenance_eligible_claim_ids,
    replacement_demand,
    reserve_shortfalls,
    select_initial_coverage_aware_reserves,
)


def _selection(
    claim_id: str,
    split: str = "train",
    label: str = "SUPPORTS",
    corruption: str = "duplication",
) -> dict:
    return {
        "claim_id": claim_id,
        "split": split,
        "fever_label": label,
        "corruption": corruption,
        "evidence": [[claim_id + "_page", 0]],
    }


def _validation(claim_id: str, status: str) -> dict:
    return {"claim_id": claim_id, "status": status}


def test_reserve_shortfall_uses_validated_supply_in_same_stratum() -> None:
    primary = [_selection("p1"), _selection("p2"), _selection("p3")]
    reserves = [_selection("r1"), _selection("r2"), _selection("r3")]
    validation = [
        _validation("p1", "semantic_failure"),
        _validation("p2", "semantic_failure"),
        _validation("p3", "validated"),
        _validation("r1", "validated"),
        _validation("r2", "semantic_failure"),
        _validation("r3", "semantic_failure"),
    ]

    assert reserve_shortfalls(primary, reserves, validation) == {
        ("train", "SUPPORTS", "duplication"): 1
    }


def test_reserve_shortfall_is_empty_when_supply_meets_demand() -> None:
    primary = [_selection("p1")]
    reserves = [_selection("r1"), _selection("r2")]
    validation = [
        _validation("p1", "semantic_failure"),
        _validation("r1", "validated"),
        _validation("r2", "semantic_failure"),
    ]

    assert reserve_shortfalls(primary, reserves, validation) == {}


def test_reserve_shortfall_counts_coverage_failure_and_filters_supply() -> None:
    primary = [_selection("p1")]
    reserves = [_selection("r1"), _selection("r2")]
    validation = [
        _validation("p1", "validated"),
        _validation("r1", "validated"),
        _validation("r2", "validated"),
    ]

    assert reserve_shortfalls(
        primary,
        reserves,
        validation,
        eligible_claim_ids={"r2"},
    ) == {}


def test_coverage_aware_initial_selection_uses_only_eligible_same_stratum() -> None:
    primary = [_selection("p1")]
    validation = [_validation("p1", "validated")]
    pool = [_selection("r1"), _selection("r2"), _selection("r3")]
    eligible = provenance_eligible_claim_ids(
        primary + pool, {"r2_page", "r3_page"}
    )
    demand = replacement_demand(
        primary, validation, eligible_claim_ids=eligible
    )

    selected = select_initial_coverage_aware_reserves(
        pool,
        demand,
        eligible_claim_ids=eligible,
        per_failure=2,
    )

    assert {record["claim_id"] for record in selected} == {"r2", "r3"}


@pytest.mark.parametrize(
    "validation",
    [
        [_validation("p1", "semantic_failure")],
        [
            _validation("p1", "semantic_failure"),
            _validation("r1", "validated"),
            _validation("extra", "validated"),
        ],
    ],
)
def test_reserve_shortfall_requires_exact_combined_coverage(
    validation: list[dict],
) -> None:
    with pytest.raises(ValueError, match="exact combined validation coverage"):
        reserve_shortfalls([_selection("p1")], [_selection("r1")], validation)


def test_final_release_handoffs_have_stable_canonical_names() -> None:
    assert FINAL_COMBINED_VALIDATION.name == "final_combined_validation.jsonl"
    assert FINAL_COMBINED_VALIDATION.parent.name == "production_nfc_reserve_recovery"
    assert FINAL_WIKIDATA_TARGETS.name == (
        "wikidata20170220_final_target_entities.jsonl"
    )
    assert FINAL_WIKIDATA_TERMS.name == (
        "wikidata20170220_final_reference_terms.jsonl"
    )
    assert FINAL_WIKIDATA_AUDIT.name == "wikidata20170220_final_snapshot_audit.json"


def test_freeze_combined_validation_writes_exact_stable_handoff(
    tmp_path,
) -> None:
    output = tmp_path / "final_combined_validation.jsonl"
    records = [{"claim_id": "p1", "status": "validated"}]

    assert freeze_combined_validation(records, output) == output
    assert freeze_combined_validation(records, output) == output
    assert output.read_text(encoding="utf-8") == (
        '{"claim_id": "p1", "status": "validated"}\n'
    )
