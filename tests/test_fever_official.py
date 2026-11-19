from __future__ import annotations

import pytest

from evidencegraph.fever_official import (
    align_official_fever_records,
    parse_fever_source_id,
    summarize_official_fever,
)


def _actual() -> list[dict]:
    return [
        {
            "id": 1,
            "label": "SUPPORTS",
            "evidence": [[[1, 1, "Category:Page", 2]]],
        },
        {"id": 2, "label": "NOT ENOUGH INFO", "evidence": []},
    ]


def _predictions() -> list[dict]:
    records = [
        {
            "dataset": "FEVER",
            "case_id": "2",
            "status": "complete",
            "label": "NOT ENOUGH INFO",
            "cited_source_ids": [],
        },
        {
            "dataset": "FEVER",
            "case_id": "1",
            "status": "complete",
            "label": "SUPPORTS",
            "cited_source_ids": ["Category:Page:2"],
        },
    ]
    for record in records:
        record.update(
            {
                "method": "fixed_rag",
                "model_requested": "qwen",
                "seed": 103,
                "answer_config": {"protocol": "v1"},
            }
        )
    return records


def test_parse_fever_source_id_preserves_colons_in_page_title() -> None:
    assert parse_fever_source_id("Category:Living_people:7") == [
        "Category:Living_people",
        7,
    ]


def test_align_official_fever_records_uses_official_order_and_citations() -> None:
    predictions, actual = align_official_fever_records(_actual(), _predictions())

    assert [record["id"] for record in predictions] == ["1", "2"]
    assert predictions[0]["predicted_evidence"] == [["Category:Page", 2]]
    assert [record["id"] for record in actual] == [1, 2]


def test_align_official_fever_records_rejects_incomplete_coverage() -> None:
    with pytest.raises(ValueError, match="missing=1"):
        align_official_fever_records(_actual(), _predictions()[:1])


def test_summarize_official_fever_uses_score_function_for_point_and_bootstrap() -> None:
    predictions, actual = align_official_fever_records(_actual(), _predictions())
    calls = []

    def fake_score(predicted, gold, max_evidence=5):
        calls.append((len(predicted), len(gold), max_evidence))
        assert all("evidence" not in record for record in predicted)
        return 1.0, 1.0, 1.0, 1.0, 1.0

    summary = summarize_official_fever(
        predictions, actual, fake_score, n_resamples=100
    )

    assert summary["n_cases"] == 2
    assert summary["fever_score"] == 1.0
    assert summary["evidence_f1_low"] == 1.0
    assert len(calls) == 101
