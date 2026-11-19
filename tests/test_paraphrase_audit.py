from __future__ import annotations

from pathlib import Path

import pytest

from evidencegraph.dataset_builder import write_jsonl
from evidencegraph.paraphrase_audit import audit_generation_cache
from evidencegraph.paraphrase_job import PROMPT_VERSION


def _selection(claim_id: str, corruption: str = "duplication") -> dict:
    return {
        "claim_id": claim_id,
        "source_record_id": claim_id.removeprefix("c"),
        "corruption": corruption,
    }


def _generated(claim_id: str, candidates: int = 2, required: int = 1) -> dict:
    return {
        "claim_id": claim_id,
        "source_record_id": claim_id.removeprefix("c"),
        "model": "small",
        "prompt_version": PROMPT_VERSION,
        "candidates_requested": candidates,
        "required_validated": required,
        "original": "Original evidence.",
        "paraphrases": [f"Rewrite {index}." for index in range(candidates)],
        "status": "generated",
    }


def test_generation_audit_accepts_recovered_failure(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    raw = tmp_path / "raw.jsonl"
    write_jsonl([_selection("c1"), _selection("c2", "paraphrase_flood")], selection)
    failure = {
        "claim_ids": ["c2"],
        "model": "small",
        "prompt_version": PROMPT_VERSION,
        "status": "schema_failure",
    }
    write_jsonl([_generated("c1"), failure, _generated("c2", 8, 4)], raw)

    summary = audit_generation_cache(selection, raw, expected_model="small")

    assert summary["complete"]
    assert summary["generated_count"] == 2
    assert summary["recovered_failure_ids"] == 1
    assert summary["unresolved_failure_ids"] == 0


def test_generation_audit_rejects_missing_claim(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    raw = tmp_path / "raw.jsonl"
    write_jsonl([_selection("c1"), _selection("c2")], selection)
    write_jsonl([_generated("c1")], raw)

    with pytest.raises(ValueError, match="1 claims are missing"):
        audit_generation_cache(selection, raw, expected_model="small")


def test_generation_audit_rejects_duplicates_and_mixed_models(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    raw = tmp_path / "raw.jsonl"
    write_jsonl([_selection("c1")], selection)
    write_jsonl([_generated("c1"), _generated("c1")], raw)
    with pytest.raises(ValueError, match="duplicate generated claim"):
        audit_generation_cache(selection, raw, expected_model="small")

    wrong_model = _generated("c1")
    wrong_model["model"] = "large"
    write_jsonl([wrong_model], raw)
    with pytest.raises(ValueError, match="mixes models"):
        audit_generation_cache(selection, raw, expected_model="small")
