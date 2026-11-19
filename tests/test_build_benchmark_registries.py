import json
from pathlib import Path

import pytest

from build_benchmark_registries import (
    PROJECT_ROOT,
    DATASETS,
    METHODS,
    TRANSFER_METHODS,
    _benchmark_input_paths,
    _load_endpoint_audit,
    _verify_raw_benchmark_cell,
    cost_transfer_registry,
    main_registry,
    reliability_registry,
)
from evidencegraph.result_registry import source_descriptor


def _metric(n: int) -> dict:
    record = {"n_cases": n, "reliability_bins": []}
    for metric in ("accuracy", "macro_f1", "ece"):
        record[metric] = 0.8
        record[f"{metric}_low"] = 0.7
        record[f"{metric}_high"] = 0.9
    for index in range(10):
        record["reliability_bins"].append(
            {
                "bin_lower": index / 10,
                "bin_upper": (index + 1) / 10,
                "mean_confidence": (index + 0.5) / 10,
                "empirical_accuracy": (index + 0.4) / 10,
                "count": n // 10 + int(index < n % 10),
            }
        )
    return record


def _quality(method_keys) -> dict:
    output = {}
    for dataset, n in DATASETS.items():
        output[dataset] = {}
        for method in method_keys:
            if dataset == "ExpertQA":
                citation = {
                    "citation_source_f1": 0.75,
                    "citation_source_precision": 0.8,
                    "citation_source_precision_low": 0.7,
                    "citation_source_precision_high": 0.9,
                    "citation_source_recall": 0.7,
                    "citation_source_recall_low": 0.6,
                    "citation_source_recall_high": 0.8,
                }
                output[dataset][method] = {"expertqa": citation}
            else:
                output[dataset][method] = {
                    "verdicts": {"datasets": {dataset: _metric(n)}}
                }
    return output


def _cost(method_keys) -> dict:
    result = {}
    for dataset, n in DATASETS.items():
        result[dataset] = {}
        for method in method_keys:
            record = {"n_cases": n}
            for metric, value in (
                ("searches", 1.0),
                ("documents_opened", 5.0),
                ("prompt_tokens", 100.0),
                ("completion_tokens", 20.0),
                ("end_to_end_wall_seconds", 2.0),
            ):
                record[f"mean_{metric}"] = value
                record[f"mean_{metric}_low"] = value * 0.9
                record[f"mean_{metric}_high"] = value * 1.1
            result[dataset][method] = {"cost": {"datasets": {dataset: record}}}
    return result


def test_main_and_reliability_registries_have_exact_denominators() -> None:
    methods = tuple(key for key, _ in METHODS)
    quality = _quality(methods)
    fever = {
        method: {
            "label_accuracy": 0.8,
            "label_accuracy_low": 0.7,
            "label_accuracy_high": 0.9,
            "evidence_f1": 0.75,
            "evidence_f1_low": 0.65,
            "evidence_f1_high": 0.85,
        }
        for method in methods
    }
    main = main_registry(quality, fever)
    assert len(main["rows"]) == 24
    assert next(
        column for column in main["columns"] if column["key"] == "macro_f1"
    )["header"] == "Macro F1 [95\\% CI]"
    assert main["rows"][0]["macro_f1"] == "80.0 [70.0, 90.0]"
    reliability = reliability_registry(quality)
    assert sum(row[2] for row in reliability["EvidenceGraph"]) == 24_498


def test_reliability_rejects_offsetting_per_dataset_denominator_errors() -> None:
    methods = tuple(key for key, _ in TRANSFER_METHODS)
    quality = _quality(methods)
    fever_bins = quality["FEVER"]["fixed_rag"]["verdicts"]["datasets"][
        "FEVER"
    ]["reliability_bins"]
    hover_bins = quality["HoVer"]["fixed_rag"]["verdicts"]["datasets"][
        "HoVer"
    ]["reliability_bins"]
    fever_bins[0]["count"] += 1
    hover_bins[0]["count"] -= 1

    with pytest.raises(ValueError, match="FEVER/fixed_rag reliability denominator"):
        reliability_registry(quality)


def test_raw_cell_verification_resolves_append_only_attempts(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    cases.write_text(
        json.dumps({"dataset": "FEVER", "case_id": "c1"}) + "\n",
        encoding="utf-8",
    )
    identity = {
        "dataset": "FEVER",
        "case_id": "c1",
        "method": "fixed_rag",
        "model_requested": "model-x",
        "model": "model-x",
        "seed": 103,
        "answer_config": {"temperature": 0.0},
    }
    attempts = [
        {**identity, "status": "backend_failure"},
        {**identity, "status": "complete"},
    ]
    predictions.write_text(
        "".join(json.dumps(record) + "\n" for record in attempts),
        encoding="utf-8",
    )
    prediction_descriptor = source_descriptor(predictions, PROJECT_ROOT)
    summary = {
        "inputs": {
            "cases": [source_descriptor(cases, PROJECT_ROOT)],
            "predictions": prediction_descriptor,
        },
        "run": {
            "method": "fixed_rag",
            "model_requested": "model-x",
            "model_resolved": "model-x",
            "seed": 103,
            "configuration": {"temperature": 0.0},
            "n_predictions": 1,
            "predictions_sha256": prediction_descriptor["sha256"],
        },
    }

    assert _verify_raw_benchmark_cell(
        summary,
        dataset="FEVER",
        method="fixed_rag",
        model_requested="model-x",
        expected_count=1,
    ) == [cases, predictions]
    summary["run"]["configuration"] = {"temperature": 0.7}
    with pytest.raises(ValueError, match="summary disagrees with raw predictions"):
        _verify_raw_benchmark_cell(
            summary,
            dataset="FEVER",
            method="fixed_rag",
            model_requested="model-x",
            expected_count=1,
        )


def test_cost_transfer_registry_compares_two_methods_on_two_backbones() -> None:
    methods = tuple(key for key, _ in TRANSFER_METHODS)
    quality = _quality(methods)
    costs = _cost(methods)
    averitec_fixed = quality["AVeriTeC"]["fixed_rag"]["verdicts"]["datasets"][
        "AVeriTeC"
    ]
    averitec_graph = quality["AVeriTeC"]["evidencegraph"]["verdicts"]["datasets"][
        "AVeriTeC"
    ]
    averitec_fixed.update({"accuracy": 0.9, "macro_f1": 0.3})
    averitec_graph.update({"accuracy": 0.1, "macro_f1": 0.4})
    table = cost_transfer_registry(
        [("Qwen", quality, costs), ("Haiku", quality, costs)]
    )
    assert len(table["rows"]) == 8
    assert "method" not in {column["key"] for column in table["columns"]}
    assert next(
        column for column in table["columns"] if column["key"] == "quality_delta"
    )["header"] == "$\\Delta$ quality (pp)"
    assert set(table["rows"][0]) == {
        "backbone",
        "dataset",
        "quality_delta",
        "search_delta",
        "document_delta",
        "token_delta",
        "latency_delta",
    }
    averitec_row = next(
        row
        for row in table["rows"]
        if row["backbone"] == "Qwen" and row["dataset"] == "AVeriTeC"
    )
    assert averitec_row["quality_delta"] == "+10.0"


def test_benchmark_summary_inputs_bind_cases_and_predictions(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    cases.write_text(json.dumps({"case_id": "c1"}) + "\n", encoding="utf-8")
    predictions.write_text(
        json.dumps({"case_id": "c1", "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    prediction_descriptor = source_descriptor(predictions, PROJECT_ROOT)
    summary = {
        "inputs": {
            "cases": [source_descriptor(cases, PROJECT_ROOT)],
            "predictions": prediction_descriptor,
        },
        "run": {"predictions_sha256": prediction_descriptor["sha256"]},
    }

    assert _benchmark_input_paths(summary) == [cases, predictions]
    summary["run"]["predictions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="disagrees with run hash"):
        _benchmark_input_paths(summary)


def test_endpoint_audit_is_bound_to_paired_run_identity(tmp_path: Path) -> None:
    inputs = {}
    for slug in (
        "fever_dev",
        "hover_dev",
        "averitec_dev",
        "expertqa_domain_test",
    ):
        comparison = tmp_path / f"{slug}.json"
        comparison.write_text(
            json.dumps(
                {
                    "protocol": "paired-benchmark-answering-v2-official-fever-evidence",
                    "candidate_run": {
                        "method": "evidencegraph",
                        "model_requested": "model-x",
                    },
                    "baseline_run": {
                        "method": "fixed_rag",
                        "model_requested": "model-x",
                    },
                    **(
                        {
                            "official_fever_scorer": {
                                "commit": "4801615100fbf6327f8e99b5dbaefe5dd890e869",
                                "max_evidence": 5,
                            }
                        }
                        if slug == "fever_dev"
                        else {}
                    ),
                }
            ),
            encoding="utf-8",
        )
        inputs[slug] = source_descriptor(comparison, PROJECT_ROOT)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "protocol": "evidencegraph-paired-endpoint-audit-v3-dataset-primary",
                "promotion_allowed": True,
                "negative_endpoints": [],
                "claim_gates": {
                    "quality_replication_met": True,
                    "calibration_replication_met": True,
                },
                "inputs": inputs,
            }
        ),
        encoding="utf-8",
    )

    sources = _load_endpoint_audit(
        {"endpoint_audit": str(audit_path), "model_requested": "model-x"}
    )
    assert sources[0] == audit_path
    assert len(sources) == 5

    first = tmp_path / "averitec_dev.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["candidate_run"]["model_requested"] = "wrong-model"
    first.write_text(json.dumps(payload), encoding="utf-8")
    inputs["averitec_dev"] = source_descriptor(first, PROJECT_ROOT)
    audit_path.write_text(
        json.dumps(
            {
                "protocol": "evidencegraph-paired-endpoint-audit-v3-dataset-primary",
                "promotion_allowed": True,
                "negative_endpoints": [],
                "claim_gates": {
                    "quality_replication_met": True,
                    "calibration_replication_met": True,
                },
                "inputs": inputs,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wrong run identity"):
        _load_endpoint_audit(
            {"endpoint_audit": str(audit_path), "model_requested": "model-x"}
        )
