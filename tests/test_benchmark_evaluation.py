from __future__ import annotations

import pytest

from evidencegraph.benchmark_evaluation import (
    benchmark_run_metadata,
    join_benchmark_predictions,
    resolve_benchmark_attempts,
    summarize_expertqa_citations,
    summarize_verdict_predictions,
)


def test_benchmark_run_metadata_rejects_mixed_methods() -> None:
    records = [
        {
            "dataset": "HoVer",
            "case_id": str(index),
            "status": "complete",
            "method": method,
            "model_requested": "qwen",
            "model": "qwen:resolved",
            "seed": 103,
            "answer_config": {"protocol": "v1"},
        }
        for index, method in enumerate(("fixed_rag", "evidencegraph"))
    ]

    with pytest.raises(ValueError, match="mix"):
        benchmark_run_metadata(records)


def test_benchmark_run_metadata_preserves_exact_config() -> None:
    records = [
        {
            "dataset": "HoVer",
            "case_id": str(index),
            "status": "complete",
            "method": "fixed_rag",
            "model_requested": "qwen",
            "model": "qwen:resolved",
            "seed": 103,
            "answer_config": {"protocol": "v1", "top_k": 5},
        }
        for index in range(2)
    ]

    metadata = benchmark_run_metadata(records)

    assert metadata["configuration"] == {"protocol": "v1", "top_k": 5}
    assert metadata["n_predictions"] == 2


def test_verdict_summary_reports_accuracy_and_calibration() -> None:
    cases = [
        {"dataset": "HoVer", "case_id": "1", "gold_label": "SUPPORTED"},
        {"dataset": "HoVer", "case_id": "2", "gold_label": "NOT_SUPPORTED"},
    ]
    predictions = [
        {
            "dataset": "HoVer",
            "case_id": "1",
            "status": "complete",
            "method": "fixed_rag",
            "model_requested": "qwen",
            "seed": 103,
            "answer_config": {"protocol": "v1"},
            "label": "SUPPORTED",
            "probabilities": {"SUPPORTED": 0.8, "NOT_SUPPORTED": 0.2},
        },
        {
            "dataset": "HoVer",
            "case_id": "2",
            "status": "complete",
            "method": "fixed_rag",
            "model_requested": "qwen",
            "seed": 103,
            "answer_config": {"protocol": "v1"},
            "label": "NOT_SUPPORTED",
            "probabilities": {"SUPPORTED": 0.1, "NOT_SUPPORTED": 0.9},
        },
    ]
    summary = summarize_verdict_predictions(
        join_benchmark_predictions(cases, predictions), n_resamples=100
    )["datasets"]["HoVer"]
    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["brier"] > 0.0
    assert sum(item["count"] for item in summary["reliability_bins"]) == 2
    assert summary["reliability_bins"][0]["mean_confidence"] == 0.8
    assert summary["reliability_bins"][0]["empirical_accuracy"] == 1.0


def test_expertqa_citation_summary_uses_complete_support_gold_ids() -> None:
    cases = [
        {"dataset": "ExpertQA", "case_id": "1", "gold_evidence": ["a", "b"]},
        {"dataset": "ExpertQA", "case_id": "2", "gold_evidence": []},
    ]
    predictions = [
        {
            "dataset": "ExpertQA",
            "case_id": "1",
            "status": "complete",
            "method": "fixed_rag",
            "model_requested": "qwen",
            "seed": 103,
            "answer_config": {"protocol": "v1"},
            "claims": [{"claim": "Fact", "cited_source_ids": ["a", "x"]}],
        },
        {
            "dataset": "ExpertQA",
            "case_id": "2",
            "status": "complete",
            "method": "fixed_rag",
            "model_requested": "qwen",
            "seed": 103,
            "answer_config": {"protocol": "v1"},
            "abstained": True,
            "claims": [],
        },
    ]
    summary = summarize_expertqa_citations(
        join_benchmark_predictions(cases, predictions), n_resamples=100
    )
    assert summary["citation_source_precision"] == 0.75
    assert summary["citation_source_recall"] == 0.5
    assert summary["n_cases_with_gold"] == 1


def test_prediction_join_rejects_extra_cases() -> None:
    cases = [{"dataset": "HoVer", "case_id": "1", "gold_label": "SUPPORTED"}]
    predictions = [
        {
            "dataset": "HoVer",
            "case_id": case_id,
            "status": "complete",
            "method": "fixed_rag",
            "model_requested": "qwen",
            "seed": 103,
            "answer_config": {"protocol": "v1"},
        }
        for case_id in ("1", "2")
    ]

    with pytest.raises(ValueError, match="extra cases"):
        join_benchmark_predictions(cases, predictions)


def test_append_only_attempt_resolution_accepts_recovered_failure() -> None:
    base = {
        "dataset": "HoVer",
        "case_id": "1",
        "method": "fixed_rag",
        "model_requested": "qwen",
        "seed": 103,
        "answer_config": {"protocol": "v1"},
    }
    records = [
        {**base, "status": "schema_failure"},
        {**base, "status": "complete", "label": "SUPPORTED"},
    ]
    assert resolve_benchmark_attempts(records) == [records[1]]


def test_append_only_attempt_resolution_rejects_unresolved_or_duplicate_completion() -> None:
    base = {
        "dataset": "HoVer",
        "case_id": "1",
        "method": "fixed_rag",
        "model_requested": "qwen",
        "seed": 103,
        "answer_config": {"protocol": "v1"},
    }
    with pytest.raises(ValueError, match="0 completed"):
        resolve_benchmark_attempts([{**base, "status": "schema_failure"}])
    with pytest.raises(ValueError, match="2 completed"):
        resolve_benchmark_attempts(
            [{**base, "status": "complete"}, {**base, "status": "complete"}]
        )
