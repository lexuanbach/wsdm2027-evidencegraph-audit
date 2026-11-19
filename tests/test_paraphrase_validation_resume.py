from __future__ import annotations

from pathlib import Path

import pytest

from evidencegraph.dataset_builder import write_jsonl
from evidencegraph.paraphrase_job import PROMPT_VERSION, validate_cache


def _raw(claim_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "source_record_id": claim_id,
        "model": "generator",
        "prompt_version": PROMPT_VERSION,
        "candidates_requested": 2,
        "required_validated": 1,
        "original": "Original.",
        "paraphrases": ["Rewritten.", "Restated."],
        "status": "generated",
    }


def _validated(
    claim_id: str,
    threshold: float = 0.8,
    revision: str = "revision",
    protocol: str = "protocol",
) -> dict:
    return {
        "claim_id": claim_id,
        "validator_model": "validator",
        "validator_revision": revision,
        "validator_protocol": protocol,
        "threshold": threshold,
        "status": "validated",
    }


def test_complete_validation_resume_does_not_load_model(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "validated.jsonl"
    write_jsonl([_raw("c1")], raw)
    write_jsonl([_validated("c1")], output)

    assert validate_cache(
        raw,
        output,
        model_name="validator",
        model_revision="revision",
        validator_protocol="protocol",
        threshold=0.8,
    ) == {
        "validated": 0,
        "semantic_failures": 0,
    }


def test_validation_resume_rejects_configuration_change(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "validated.jsonl"
    write_jsonl([_raw("c1")], raw)
    write_jsonl([_validated("c1", threshold=0.7)], output)

    with pytest.raises(ValueError, match="models, revisions, protocols, or thresholds"):
        validate_cache(
            raw,
            output,
            model_name="validator",
            model_revision="revision",
            validator_protocol="protocol",
            threshold=0.8,
        )


@pytest.mark.parametrize(
    ("record", "revision", "protocol"),
    [
        (_validated("c1", revision="old"), "new", "protocol"),
        (_validated("c1", protocol="old"), "revision", "new"),
    ],
)
def test_validation_resume_rejects_revision_or_protocol_change(
    tmp_path: Path, record: dict, revision: str, protocol: str
) -> None:
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "validated.jsonl"
    write_jsonl([_raw("c1")], raw)
    write_jsonl([record], output)

    with pytest.raises(ValueError, match="models, revisions, protocols, or thresholds"):
        validate_cache(
            raw,
            output,
            model_name="validator",
            model_revision=revision,
            validator_protocol=protocol,
            threshold=0.8,
        )


def test_validation_rejects_duplicate_raw_claims(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "validated.jsonl"
    write_jsonl([_raw("c1"), _raw("c1")], raw)

    with pytest.raises(ValueError, match="duplicate generated claim"):
        validate_cache(
            raw,
            output,
            model_name="validator",
            model_revision="revision",
            validator_protocol="protocol",
            threshold=0.8,
        )


def test_validation_rejects_blank_raw_candidate_before_loading_model(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "validated.jsonl"
    record = _raw("c1")
    record["paraphrases"][1] = " "
    write_jsonl([record], raw)

    with pytest.raises(ValueError, match="invalid generated candidate content"):
        validate_cache(
            raw,
            output,
            model_name="validator",
            model_revision="revision",
            validator_protocol="protocol",
            threshold=0.8,
        )
