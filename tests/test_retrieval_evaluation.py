from __future__ import annotations

import pytest

from evidencegraph.retrieval_evaluation import (
    compare_retrieval_cases,
    retrieval_case_records,
    retrieval_run_metadata,
    resolve_retrieval_attempts,
    resolve_selection_attempts,
    summarize_retrieval_cases,
)


def _attempt(status: str) -> dict:
    return {
        "status": status,
        "dataset": "FEVER",
        "case_id": "1",
        "query": "Claim.",
        "retrieval_protocol": "lexical",
        "retrieval_config": {"top_k": 20},
        "results": [],
    }


def test_retrieval_attempts_allow_failure_then_one_completion() -> None:
    complete = _attempt("complete")
    assert resolve_retrieval_attempts(
        [_attempt("retrieval_failure"), complete]
    ) == [complete]


def test_retrieval_attempts_reject_duplicate_completions() -> None:
    with pytest.raises(ValueError, match="duplicate completions"):
        resolve_retrieval_attempts([_attempt("complete"), _attempt("complete")])


def test_retrieval_attempts_allow_unresolved_failure_during_resume() -> None:
    assert resolve_retrieval_attempts(
        [_attempt("retrieval_failure")], require_complete=False
    ) == []


def test_selection_attempts_allow_failure_then_one_completion() -> None:
    base = {
        "dataset": "FEVER",
        "case_id": "1",
        "query": "Claim.",
        "selection_method": "evidencegraph",
        "selection_config": {"protocol": "selection-v1"},
    }
    complete = {**base, "status": "complete"}
    assert resolve_selection_attempts(
        [{**base, "status": "selection_failure"}, complete]
    ) == [complete]


def test_selection_attempts_reject_failure_after_completion() -> None:
    base = {
        "dataset": "FEVER",
        "case_id": "1",
        "query": "Claim.",
        "selection_method": "evidencegraph",
        "selection_config": {"protocol": "selection-v1"},
    }
    with pytest.raises(ValueError, match="failures after completion"):
        resolve_selection_attempts(
            [{**base, "status": "complete"}, {**base, "status": "selection_failure"}]
        )


def test_retrieval_run_metadata_rejects_mixed_configs() -> None:
    records = [
        {
            "status": "complete",
            "dataset": "HoVer",
            "case_id": str(top_k),
            "query": "Claim.",
            "retrieval_protocol": "lexical",
            "retrieval_config": {"top_k": top_k},
        }
        for top_k in (5, 20)
    ]

    with pytest.raises(ValueError, match="mix"):
        retrieval_run_metadata(records)


def test_fever_alternative_evidence_groups_use_complete_group_hit() -> None:
    cases = [
        {
            "dataset": "FEVER",
            "case_id": "1",
            "gold_evidence": ["A:0", "B:0", "C:0"],
            "metadata": {"evidence_groups": [["A:0"], ["B:0", "C:0"]]},
        }
    ]
    retrievals = [
        {
            "dataset": "FEVER",
            "case_id": "1",
            "query": "Claim.",
            "status": "complete",
            "retrieval_protocol": "lexical",
            "retrieval_config": {"top_k": 20},
            "results": [{"source_id": "A:0"}, {"source_id": "X:0"}],
        }
    ]
    record = retrieval_case_records(cases, retrievals)[0]

    assert record["item_recall"] == pytest.approx(1 / 3)
    assert record["complete_group_hit"] == 1
    assert record["reciprocal_rank"] == 1.0


def test_retrieval_summary_excludes_cases_without_gold_from_metric_denominator() -> None:
    records = [
        {
            "dataset": "ExpertQA",
            "case_id": "1",
            "gold_items": 1,
            "retrieved_items": 2,
            "item_recall": 1.0,
            "complete_group_hit": 1,
            "reciprocal_rank": 0.5,
        },
        {
            "dataset": "ExpertQA",
            "case_id": "2",
            "gold_items": 0,
            "retrieved_items": 0,
            "item_recall": None,
            "complete_group_hit": None,
            "reciprocal_rank": None,
        },
    ]
    summary = summarize_retrieval_cases(records, n_resamples=100)["datasets"]["ExpertQA"]

    assert summary["n_cases"] == 2
    assert summary["n_cases_with_gold"] == 1
    assert summary["item_recall"] == 1.0


def test_retrieval_bootstrap_is_finite_for_expertqa_sized_denominator() -> None:
    records = [
        {
            "dataset": "ExpertQA",
            "case_id": str(index),
            "gold_items": 1,
            "retrieved_items": 20,
            "item_recall": 1.0,
            "complete_group_hit": 1,
            "reciprocal_rank": 1.0 / (1 + index % 5),
        }
        for index in range(210)
    ]

    summary = summarize_retrieval_cases(records, n_resamples=100)["datasets"][
        "ExpertQA"
    ]

    assert 0.0 < summary["mrr_low"] <= summary["mrr"] <= summary["mrr_high"] <= 1.0


def test_retrieval_join_rejects_extra_cases() -> None:
    cases = [{"dataset": "FEVER", "case_id": "1", "gold_evidence": []}]
    retrievals = [
        {**_attempt("complete"), "case_id": "1"},
        {**_attempt("complete"), "case_id": "2"},
    ]

    with pytest.raises(ValueError, match="extra cases"):
        retrieval_case_records(cases, retrievals)


def test_retrieval_join_rejects_missing_cases() -> None:
    cases = [
        {"dataset": "FEVER", "case_id": "1", "gold_evidence": []},
        {"dataset": "FEVER", "case_id": "2", "gold_evidence": []},
    ]
    retrievals = [{**_attempt("complete"), "case_id": "1"}]

    with pytest.raises(ValueError, match="missing retrieval for case 2"):
        retrieval_case_records(cases, retrievals)


def test_paired_retrieval_comparison_uses_aligned_claim_differences() -> None:
    candidate = [
        {
            "dataset": "HoVer",
            "case_id": str(index),
            "gold_items": 2,
            "item_recall": item_recall,
            "complete_group_hit": complete,
            "reciprocal_rank": reciprocal_rank,
        }
        for index, (item_recall, complete, reciprocal_rank) in enumerate(
            [(1.0, 1, 1.0), (0.5, 0, 0.5)]
        )
    ]
    baseline = [
        {
            "dataset": "HoVer",
            "case_id": str(index),
            "gold_items": 2,
            "item_recall": item_recall,
            "complete_group_hit": complete,
            "reciprocal_rank": reciprocal_rank,
        }
        for index, (item_recall, complete, reciprocal_rank) in enumerate(
            [(0.5, 0, 0.5), (0.0, 0, 0.0)]
        )
    ]

    result = compare_retrieval_cases(candidate, baseline, n_resamples=100)
    summary = result["datasets"]["HoVer"]

    assert summary["n_cases_with_gold"] == 2
    assert summary["item_recall_difference"] == pytest.approx(0.5)
    assert summary["complete_group_recall_difference"] == pytest.approx(0.5)
    assert summary["mrr_difference"] == pytest.approx(0.5)


def test_paired_retrieval_comparison_rejects_mismatched_cases() -> None:
    record = {
        "dataset": "HoVer",
        "case_id": "1",
        "gold_items": 1,
        "item_recall": 1.0,
        "complete_group_hit": 1,
        "reciprocal_rank": 1.0,
    }

    with pytest.raises(ValueError, match="identical case identities"):
        compare_retrieval_cases([record], [{**record, "case_id": "2"}])


def test_paired_retrieval_comparison_rejects_mismatched_gold_counts() -> None:
    record = {
        "dataset": "HoVer",
        "case_id": "1",
        "gold_items": 1,
        "item_recall": 1.0,
        "complete_group_hit": 1,
        "reciprocal_rank": 1.0,
    }

    with pytest.raises(ValueError, match="mismatched gold counts"):
        compare_retrieval_cases([record], [{**record, "gold_items": 2}])
