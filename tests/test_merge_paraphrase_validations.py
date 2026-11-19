import pytest

from pathlib import Path

from evidencegraph.dataset_builder import write_jsonl
from merge_paraphrase_validations import merge_records


def _record(paraphrases: list[str], model: str) -> dict:
    return {
        "claim_id": "c1",
        "source_record_id": "1",
        "original": "Original evidence.",
        "paraphrases": paraphrases,
        "generator_model": model,
        "prompt_version": "test",
        "n_candidates": 8,
        "n_accepted": len(paraphrases),
        "required_validated": 4,
        "validator_model": "validator",
        "validator_revision": "revision",
        "validator_protocol": "protocol",
        "threshold": 0.8,
        "status": "validated" if len(paraphrases) >= 4 else "semantic_failure",
    }


def test_merge_pools_unique_candidates_across_attempts(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl([_record(["One.", "Two."], "small")], first)
    write_jsonl([_record(["two.", "Three.", "Four."], "large")], second)

    merged = merge_records([first, second])

    assert len(merged) == 1
    assert merged[0]["status"] == "validated"
    assert merged[0]["n_unique_accepted"] == 4
    assert merged[0]["n_unique_accepted_lower_bound"] == 4
    assert merged[0]["n_unique_accepted_is_lower_bound"] is False
    assert merged[0]["paraphrases"] == ["One.", "Two.", "Three.", "Four."]
    assert [attempt["generator_model"] for attempt in merged[0]["attempts"]] == [
        "small",
        "large",
    ]


def test_merge_rejects_mixed_validation_configurations(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl([_record(["One."], "small")], first)
    changed = _record(["Two."], "large")
    changed["threshold"] = 0.7
    write_jsonl([changed], second)

    try:
        merge_records([first, second])
    except ValueError as error:
        assert "different validator" in str(error)
    else:
        raise AssertionError("mixed validation thresholds must be rejected")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_record_id", "other", "conflicting source record IDs"),
        ("required_validated", 1, "conflicting paraphrase requirements"),
    ],
)
def test_merge_rejects_cross_attempt_identity_mismatch(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl([_record(["One."], "small")], first)
    changed = _record(["Two."], "large")
    changed[field] = value
    if field == "required_validated":
        changed["status"] = "validated"
    write_jsonl([changed], second)

    with pytest.raises(ValueError, match=message):
        merge_records([first, second])


def test_merge_rejects_duplicate_input_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    write_jsonl([_record(["One."], "small")], first)

    with pytest.raises(ValueError, match="non-empty and unique"):
        merge_records([first, first])


def test_merge_preserves_structural_attempt_provenance(tmp_path: Path) -> None:
    validation = tmp_path / "validation.jsonl"
    record = _record(["One."], "small")
    record.update(
        {
            "nli_validator_model": "nli",
            "nli_validator_revision": "nli-revision",
            "nli_validator_protocol": "nli-protocol",
            "structural_gate_protocol": "structure-v3",
            "structural_parser_model": "parser",
            "structural_parser_model_version": "1",
            "spacy_version": "2",
            "source_validation_sha256": "a" * 64,
        }
    )
    write_jsonl([record], validation)

    attempt = merge_records([validation])[0]["attempts"][0]

    assert attempt["nli_validator_model"] == "nli"
    assert attempt["nli_validator_revision"] == "nli-revision"
    assert attempt["nli_validator_protocol"] == "nli-protocol"
    assert attempt["structural_gate_protocol"] == "structure-v3"
    assert attempt["source_validation_sha256"] == "a" * 64


def test_merge_accepts_pooled_records_without_claiming_hidden_union_exactness(
    tmp_path: Path,
) -> None:
    pooled_path = tmp_path / "pooled.jsonl"
    new_path = tmp_path / "new.jsonl"
    pooled = {
        "claim_id": "c1",
        "source_record_id": "1",
        "original": "Original evidence.",
        "paraphrases": ["One."],
        "attempts": [
            {
                "generator_model": "small",
                "prompt_version": "old",
                "n_candidates": 3,
                "n_accepted": 2,
                "validator_model": "validator",
                "validator_revision": "revision",
                "validator_protocol": "protocol",
                "threshold": 0.8,
            }
        ],
        "required_validated": 1,
        "validator_model": "validator",
        "validator_revision": "revision",
        "validator_protocol": "protocol",
        "threshold": 0.8,
        "n_unique_accepted": 2,
        "status": "validated",
    }
    write_jsonl([pooled], pooled_path)
    new_record = _record(["Three."], "large")
    new_record["required_validated"] = 1
    new_record["status"] = "validated"
    write_jsonl([new_record], new_path)

    merged = merge_records([pooled_path, new_path])[0]

    assert merged["status"] == "validated"
    assert merged["n_unique_accepted_lower_bound"] == 2
    assert merged["n_unique_accepted_is_lower_bound"] is True
    assert "n_unique_accepted" not in merged
    assert len(merged["attempts"]) == 2


def test_merge_rejects_unpinned_pooled_attempt(tmp_path: Path) -> None:
    pooled_path = tmp_path / "pooled.jsonl"
    record = _record(["One."], "small")
    attempt = {
        key: record[key]
        for key in (
            "generator_model",
            "prompt_version",
            "n_candidates",
            "n_accepted",
            "validator_model",
            "validator_protocol",
            "threshold",
        )
    }
    pooled = {
        **record,
        "attempts": [attempt],
        "n_unique_accepted": 1,
    }
    write_jsonl([pooled], pooled_path)

    with pytest.raises(ValueError, match="pooled attempt validator"):
        merge_records([pooled_path])
