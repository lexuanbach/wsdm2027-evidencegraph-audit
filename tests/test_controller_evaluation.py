from __future__ import annotations

from itertools import combinations

from evidencegraph.controller_evaluation import (
    dependence_quartile_summary,
    rollout_dataset_split,
)


def test_dependence_quartiles_are_balanced_and_report_mechanism_metrics() -> None:
    records = [
        {
            "claim_id": f"c{index}",
            "dependence_confidence": index / 39,
            "label": index % 2,
            "probability_true": 0.8 if index % 2 else 0.2,
            "find_indep_count": int(index >= 20),
        }
        for index in range(40)
    ]
    summary = dependence_quartile_summary(records, n_resamples=100)

    assert summary["n_claims"] == [10, 10, 10, 10]
    assert summary["accuracy_mean"] == [1.0, 1.0, 1.0, 1.0]
    assert summary["find_indep_rate_mean"] == [0.0, 0.0, 1.0, 1.0]
    assert summary["confidence_range"][0][1] < summary["confidence_range"][-1][0]
    assert "2 * abs" in summary["bucketing"]


def test_dependence_bucket_feature_uses_only_initial_observed_graph() -> None:
    source_ids = [f"s{index}" for index in range(6)]
    episode = {
        "claim_id": "c1",
        "split": "test",
        "label": True,
        "corruption": "duplication",
        "sources": [{"source_id": source_id} for source_id in source_ids],
    }
    nli = {
        ("c1", source_id): {"support": 0.9, "neutral": 0.05, "refute": 0.05}
        for source_id in source_ids
    }
    dependence = {
        ("c1", *sorted((left, right))): (1.0 if "s5" in (left, right) else 0.5)
        for left, right in combinations(source_ids, 2)
    }

    record = rollout_dataset_split(
        [episode], nli, dependence, None, split="test", method="fixed_top5"
    )[0]

    assert record["dependence_confidence"] == 0.0
    assert record["dependence_scope"] == "initial_observed_graph"
