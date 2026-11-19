from __future__ import annotations

import pytest

from evidencegraph.episode_evaluation import score_dataset_split, summarize_episode_predictions


def _fixtures():
    episodes = [
        {
            "claim_id": "c1",
            "split": "test",
            "label": True,
            "corruption": "duplication",
            "sources": [{"source_id": "s1"}, {"source_id": "s2"}],
        }
    ]
    nli = [
        {
            "claim_id": "c1",
            "source_id": source,
            "status": "complete",
            "probabilities": {"support": 0.9, "neutral": 0.05, "refute": 0.05},
        }
        for source in ("s1", "s2")
    ]
    dependence = [
        {
            "claim_id": "c1",
            "left_id": "s2",
            "right_id": "s1",
            "probabilities": {"103": 0.9, "104": 0.8, "105": 0.85},
        }
    ]
    return episodes, nli, dependence


def test_score_dataset_split_joins_edges_in_episode_source_order() -> None:
    records = score_dataset_split(*_fixtures(), split="test")

    assert records[0]["claim_id"] == "c1"
    assert records[0]["label"] == 1
    assert records[0]["mean_dependence"] == pytest.approx(0.85)
    assert records[0]["evidence_logit"] > 0.0


def test_score_dataset_split_fails_closed_on_missing_pair() -> None:
    episodes, nli, _ = _fixtures()
    with pytest.raises(ValueError, match="index is empty"):
        score_dataset_split(episodes, nli, [], split="test")


def test_score_dataset_split_rejects_incomplete_seed_set() -> None:
    episodes, nli, dependence = _fixtures()
    dependence[0]["probabilities"].pop("105")
    with pytest.raises(ValueError, match="exactly seeds"):
        score_dataset_split(episodes, nli, dependence, split="test")


def test_episode_summary_reports_corruption_and_reliability() -> None:
    records = [
        {
            "claim_id": f"c{index}",
            "label": index % 2,
            "probability_true": 0.8 if index % 2 else 0.2,
            "corruption": "duplication" if index < 10 else "paraphrase_flood",
        }
        for index in range(20)
    ]
    summary = summarize_episode_predictions(records, n_resamples=100)

    assert summary["groups"]["overall"]["accuracy"] == pytest.approx(1.0)
    assert summary["groups"]["duplication"]["n_claims"] == 10
    assert summary["groups"]["paraphrase_flood"]["reliability_bins"]
