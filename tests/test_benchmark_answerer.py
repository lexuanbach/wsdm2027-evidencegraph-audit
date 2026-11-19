from __future__ import annotations

import pytest

from evidencegraph.benchmark_answerer import (
    BenchmarkPrediction,
    LocalBenchmarkAnswerer,
    apply_graph_calibrated_probability,
    answer_context_results,
    cited_evidence_records,
    deterministic_no_evidence_verdict,
    verdict_schema,
)
from evidencegraph.local_llm import LocalResponse


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, *args, **kwargs):
        return self.payload, LocalResponse(
            content="{}",
            model="fake",
            wall_seconds=0.1,
            prompt_tokens=10,
            completion_tokens=5,
        )


def test_verdict_schema_requires_every_dataset_label() -> None:
    schema = verdict_schema(("A", "B"))
    probabilities = schema["properties"]["probabilities"]
    assert probabilities["required"] == ["A", "B"]
    assert probabilities["additionalProperties"] is False


def test_local_benchmark_answerer_validates_probability_and_citations() -> None:
    payload = {
        "label": "SUPPORTED",
        "probabilities": {"SUPPORTED": 0.8, "NOT_SUPPORTED": 0.2},
        "answer": "Supported by the source.",
        "cited_source_ids": ["s1"],
    }
    prediction = LocalBenchmarkAnswerer("fake", FakeClient(payload)).predict_verdict(
        "HoVer", "Claim", [{"source_id": "s1", "text": "Evidence."}]
    )
    assert prediction.label == "SUPPORTED"
    assert prediction.cited_source_ids == ("s1",)


def test_local_benchmark_answerer_rejects_non_argmax_label() -> None:
    payload = {
        "label": "NOT_SUPPORTED",
        "probabilities": {"SUPPORTED": 0.8, "NOT_SUPPORTED": 0.2},
        "answer": "Answer.",
        "cited_source_ids": [],
    }
    with pytest.raises(ValueError, match="argmax"):
        LocalBenchmarkAnswerer("fake", FakeClient(payload)).predict_verdict(
            "HoVer", "Claim", [{"source_id": "s1", "text": "Evidence."}]
        )


def test_cited_answer_preserves_atomic_claim_source_mapping() -> None:
    payload = {
        "answer": "A detailed answer.",
        "claims": [{"claim": "Fact one.", "cited_source_ids": ["s1"]}],
    }
    answer = LocalBenchmarkAnswerer("fake", FakeClient(payload)).generate_cited_answer(
        "Why?", [{"source_id": "s1", "text": "Fact one."}]
    )

    assert answer.claims[0]["claim"] == "Fact one."
    assert answer.claims[0]["cited_source_ids"] == ["s1"]


def test_closed_book_verdict_requires_empty_citations() -> None:
    payload = {
        "label": "SUPPORTED",
        "probabilities": {"SUPPORTED": 0.6, "NOT_SUPPORTED": 0.4},
        "answer": "Supported from parametric knowledge.",
        "cited_source_ids": [],
    }

    prediction = LocalBenchmarkAnswerer("fake", FakeClient(payload)).predict_closed_book(
        "HoVer", "Claim"
    )

    assert prediction.cited_source_ids == ()


def test_closed_book_long_form_claims_have_no_fabricated_citations() -> None:
    payload = {"answer": "Answer.", "claims": ["Atomic fact."]}

    answer = LocalBenchmarkAnswerer(
        "fake", FakeClient(payload)
    ).generate_closed_book_answer("Why?")

    assert answer.claims == ({"claim": "Atomic fact.", "cited_source_ids": []},)


def test_claim_draft_has_no_preassigned_citations() -> None:
    payload = {"answer": "Draft.", "claims": ["Atomic fact."]}

    answer = LocalBenchmarkAnswerer("fake", FakeClient(payload)).draft_atomic_claims(
        "Why?", [{"source_id": "s1", "text": "Evidence."}]
    )

    assert answer.claims == ({"claim": "Atomic fact.", "cited_source_ids": []},)


def test_supported_answer_preserves_graph_fixed_claims() -> None:
    payload = {"answer": "Atomic fact [s1]."}
    claims = [{"claim": "Atomic fact.", "cited_source_ids": ["s1"]}]

    answer = LocalBenchmarkAnswerer("fake", FakeClient(payload)).compose_supported_answer(
        "Why?", claims, [{"source_id": "s1", "text": "Evidence."}]
    )

    assert answer.claims == tuple(claims)


def test_cited_evidence_records_preserve_exact_passage_text_once() -> None:
    sources = [
        {"source_id": "s1", "text": "First passage."},
        {"source_id": "s2", "text": "Second passage."},
    ]
    prediction = {
        "claims": [
            {"claim": "A.", "cited_source_ids": ["s2"]},
            {"claim": "B.", "cited_source_ids": ["s2", "s1"]},
        ]
    }

    assert cited_evidence_records(sources, prediction) == [
        {"source_id": "s2", "text": "Second passage."},
        {"source_id": "s1", "text": "First passage."},
    ]


@pytest.mark.parametrize(
    ("dataset", "label"),
    [
        ("FEVER", "NOT ENOUGH INFO"),
        ("HoVer", "NOT_SUPPORTED"),
        ("AVeriTeC", "Not Enough Evidence"),
    ],
)
def test_no_evidence_verdict_is_dataset_defined_abstention(dataset, label) -> None:
    record = deterministic_no_evidence_verdict(dataset)

    assert record["label"] == label
    assert record["probabilities"][label] == 1.0
    assert record["cited_source_ids"] == []
    assert record["prompt_tokens"] == 0


def test_graph_answer_context_retains_active_acquisitions() -> None:
    results = [{"source_id": f"s{index}"} for index in range(8)]

    assert len(answer_context_results("fixed_rag", results, 5)) == 5
    assert len(answer_context_results("evidencegraph", results, 5)) == 8


def _prediction() -> BenchmarkPrediction:
    return BenchmarkPrediction(
        label="SUPPORTS",
        probabilities={"SUPPORTS": 0.8, "REFUTES": 0.15, "NOT ENOUGH INFO": 0.05},
        answer="The claim is supported.",
        cited_source_ids=("s1",),
        model="fake",
        prompt_tokens=10,
        completion_tokens=5,
        wall_seconds=0.1,
    )


def test_graph_calibration_overrides_llm_with_best_non_support_label() -> None:
    result = apply_graph_calibrated_probability(_prediction(), "FEVER", 0.2)

    assert result.label == "REFUTES"
    assert result.probabilities == {
        "SUPPORTS": 0.2,
        "REFUTES": 0.8,
        "NOT ENOUGH INFO": 0.0,
    }
    assert "refutes" in result.answer
    assert result.cited_source_ids == ("s1",)


def test_graph_calibration_uses_graph_probability_for_support() -> None:
    result = apply_graph_calibrated_probability(_prediction(), "FEVER", 0.7)

    assert result.label == "SUPPORTS"
    assert result.probabilities["SUPPORTS"] == pytest.approx(0.7)
    assert result.probabilities["REFUTES"] == pytest.approx(0.3)
    assert result.probabilities["NOT ENOUGH INFO"] == 0.0


def test_graph_calibration_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="graph probability"):
        apply_graph_calibrated_probability(_prediction(), "FEVER", float("nan"))
