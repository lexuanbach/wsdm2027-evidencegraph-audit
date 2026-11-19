from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_local_benchmark_matrix import (
    PYTHON,
    PROJECT_ROOT,
    cell_paths,
    generation_command,
    prediction_progress,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_cell_paths_match_registry_template_contract() -> None:
    paths = cell_paths("averitec_dev", "fixed_rag")

    assert paths["quality"] == PROJECT_ROOT / (
        "artifact/runs/benchmarks/summaries/"
        "averitec_dev_fixed_rag_qwen14_quality.json"
    )
    assert paths["cost"] == PROJECT_ROOT / (
        "artifact/runs/benchmarks/summaries/"
        "averitec_dev_fixed_rag_qwen14_cost.json"
    )


def test_child_python_preserves_virtual_environment_path() -> None:
    assert PYTHON == Path(__import__("sys").executable)


def test_generation_command_uses_frozen_retrieval_and_model() -> None:
    command = generation_command("hover_dev", "fixed_rag")

    assert "qwen2.5:14b" in command
    assert str(
        PROJECT_ROOT
        / "artifact/runs/benchmarks/hover_dev_retrieval_v7_finite.jsonl"
    ) in command


def test_prediction_progress_counts_recovered_failures(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        cases,
        [
            {"dataset": "HoVer", "case_id": "1"},
            {"dataset": "HoVer", "case_id": "2"},
        ],
    )
    _write_jsonl(
        predictions,
        [
            {"dataset": "HoVer", "case_id": "1", "status": "schema_failure"},
            {"dataset": "HoVer", "case_id": "1", "status": "complete"},
        ],
    )

    assert prediction_progress(cases, predictions) == (1, 1)


def test_prediction_progress_rejects_duplicate_completion(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(cases, [{"dataset": "HoVer", "case_id": "1"}])
    _write_jsonl(
        predictions,
        [
            {"dataset": "HoVer", "case_id": "1", "status": "complete"},
            {"dataset": "HoVer", "case_id": "1", "status": "complete"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate completions"):
        prediction_progress(cases, predictions)
