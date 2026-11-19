from __future__ import annotations

from pathlib import Path

import pytest

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.paraphrase_structural_filter import STRUCTURE_PROTOCOL
from evidencegraph.semantic_validation import (
    PARAPHRASE_NLI_MODEL,
    PARAPHRASE_NLI_PROTOCOL,
    PARAPHRASE_NLI_REVISION,
)
from run_paraphrase_recovery_ladder import (
    STAGES,
    _write_jsonl_exact,
    exact_failure_selection,
    stage_paths,
    validate_initial_structural_provenance,
)


def _selection(claim_id: str) -> dict:
    return {"claim_id": claim_id, "source_record_id": claim_id}


def _validation(claim_id: str, status: str) -> dict:
    return {
        "claim_id": claim_id,
        "status": status,
        "threshold": 0.8,
        "validator_revision": PARAPHRASE_NLI_REVISION,
        "validator_protocol": PARAPHRASE_NLI_PROTOCOL,
        "nli_validator_model": PARAPHRASE_NLI_MODEL,
        "nli_validator_revision": PARAPHRASE_NLI_REVISION,
        "nli_validator_protocol": PARAPHRASE_NLI_PROTOCOL,
        "structural_gate_protocol": STRUCTURE_PROTOCOL,
        "structural_parser_model": "en_core_web_sm",
        "structural_parser_model_version": "3.8.0",
        "spacy_version": "3.8.7",
    }


def test_failure_selection_is_exact_and_preserves_primary_order() -> None:
    selection = [_selection("a"), _selection("b"), _selection("c")]
    validation = [
        _validation("c", "semantic_failure"),
        _validation("a", "semantic_failure"),
        _validation("b", "validated"),
    ]

    failures = exact_failure_selection(selection, validation)

    assert [record["claim_id"] for record in failures] == ["a", "c"]


@pytest.mark.parametrize(
    "validation",
    [
        [_validation("a", "validated")],
        [
            _validation("a", "validated"),
            _validation("b", "validated"),
            _validation("extra", "validated"),
        ],
        [_validation("a", "validated"), _validation("a", "validated")],
    ],
)
def test_failure_selection_rejects_inexact_coverage(validation: list[dict]) -> None:
    with pytest.raises(ValueError):
        exact_failure_selection([_selection("a"), _selection("b")], validation)


def test_initial_validation_requires_frozen_structural_provenance() -> None:
    selection = [_selection("a")]
    valid = [_validation("a", "validated")]
    validate_initial_structural_provenance(selection, valid)

    invalid = [dict(valid[0], structural_gate_protocol="old")]
    with pytest.raises(ValueError, match="frozen composite gate"):
        validate_initial_structural_provenance(selection, invalid)


def test_exact_writer_resumes_match_and_refuses_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "selection.jsonl"
    records = [_selection("a")]
    _write_jsonl_exact(output, records)
    original_bytes = output.read_bytes()
    _write_jsonl_exact(output, records)
    assert output.read_bytes() == original_bytes
    assert read_jsonl(output) == records

    with pytest.raises(ValueError, match="refusing to replace"):
        _write_jsonl_exact(output, [_selection("b")])
    assert output.read_bytes() == original_bytes


def test_stage_paths_are_disjoint_and_stable(tmp_path: Path) -> None:
    all_paths: list[Path] = []
    for stage in STAGES:
        paths = stage_paths(tmp_path, stage)
        assert paths.raw.name == f"{stage.name}_raw.jsonl"
        assert paths.cumulative_validation.name == (
            f"primary_through_{stage.name}_validated.jsonl"
        )
        values = list(vars(paths).values())
        if stage.kind == "local":
            assert paths.failure_selection == paths.eligible_selection
            values.remove(paths.eligible_selection)
        all_paths.extend(values)
    assert len(all_paths) == len(set(all_paths))
