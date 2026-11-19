from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_evidencegraph_benchmark_matrix import (
    DATASETS,
    MODEL,
    answer_command,
    endpoint_audit_command,
    selection_command,
    selection_path,
    selection_progress,
    verify_local_success_gate,
)
from evidencegraph.result_registry import source_descriptor


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_graph_commands_use_learned_selection_and_expertqa_runner() -> None:
    hover_selection = selection_path("hover_dev")
    assert str(hover_selection) in selection_command("hover_dev")
    assert any(
        item.endswith("run_benchmark_answering.py")
        for item in answer_command("hover_dev")
    )
    assert str(hover_selection) in answer_command("hover_dev")
    assert any(
        item.endswith("run_expertqa_evidencegraph.py")
        for item in answer_command("expertqa_domain_test")
    )
    with pytest.raises(ValueError, match="verdict datasets"):
        selection_path("expertqa_domain_test")
    audit = endpoint_audit_command()
    assert audit.count("--comparison") == 4


def test_selection_progress_preserves_recovered_failure(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    output = tmp_path / "selection.jsonl"
    _write_jsonl(
        cases,
        [
            {"dataset": "HoVer", "case_id": "1"},
            {"dataset": "HoVer", "case_id": "2"},
        ],
    )
    identity = {
        "dataset": "HoVer",
        "case_id": "1",
        "query": "claim",
        "selection_method": "evidencegraph",
        "selection_config": {"protocol": "test"},
    }
    _write_jsonl(
        output,
        [
            {**identity, "status": "selection_failure"},
            {**identity, "status": "complete"},
        ],
    )

    assert selection_progress(cases, output) == (1, 1)


def test_paid_transfer_gate_rehashes_all_local_success_inputs(tmp_path: Path) -> None:
    records = []
    comparisons = {}
    for dataset in DATASETS:
        fields = ("predictions", "quality", "cost", "comparison")
        paths = {field: tmp_path / f"{dataset}-{field}.json" for field in fields}
        if dataset != "expertqa_domain_test":
            paths["selection"] = tmp_path / f"{dataset}-selection.jsonl"
        if dataset == "fever_dev":
            paths["fever_official"] = tmp_path / f"{dataset}-official.json"
        for path in paths.values():
            path.write_text("{}\n", encoding="utf-8")
        descriptors = {
            field: source_descriptor(path, tmp_path) for field, path in paths.items()
        }
        records.append({"dataset": dataset, **descriptors})
        comparisons[dataset] = descriptors["comparison"]

    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "protocol": "evidencegraph-paired-endpoint-audit-v3-dataset-primary",
                "promotion_allowed": True,
                "negative_endpoints": [],
                "claim_gates": {
                    "quality_replication_met": True,
                    "calibration_replication_met": True,
                },
                "inputs": comparisons,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "protocol": "local-evidencegraph-benchmark-matrix-v1",
                "model": MODEL,
                "endpoint_audit": source_descriptor(endpoint, tmp_path),
                "datasets": records,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert len(
        verify_local_success_gate(summary, endpoint, project_root=tmp_path)["datasets"]
    ) == 4
    (tmp_path / "hover_dev-comparison.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="descriptor mismatch"):
        verify_local_success_gate(summary, endpoint, project_root=tmp_path)
