from __future__ import annotations

from evidencegraph.benchmark_cost import summarize_benchmark_cost


def test_benchmark_cost_sums_all_latency_stages() -> None:
    joined = [
        {
            "case": {"dataset": "FEVER", "case_id": "1"},
            "prediction": {
                "searches": 1,
                "documents_opened": 5,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "llm_calls": 1,
                "wall_seconds": 2.0,
                "retrieval_wall_seconds": 1.0,
                "edge_and_selection_wall_seconds": 0.5,
            },
        },
        {
            "case": {"dataset": "FEVER", "case_id": "2"},
            "prediction": {
                "searches": 3,
                "documents_opened": 7,
                "prompt_tokens": 200,
                "completion_tokens": 40,
                "llm_calls": 2,
                "wall_seconds": 4.0,
                "retrieval_wall_seconds": 2.0,
                "edge_and_selection_wall_seconds": 1.0,
            },
        },
    ]

    summary = summarize_benchmark_cost(joined, n_resamples=100)["datasets"]["FEVER"]

    assert summary["mean_searches"] == 2.0
    assert summary["mean_prompt_tokens"] == 150.0
    assert summary["mean_end_to_end_wall_seconds"] == 5.25
