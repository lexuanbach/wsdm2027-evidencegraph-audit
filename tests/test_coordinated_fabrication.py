from __future__ import annotations

from evidencegraph.coordinated_fabrication import stress_records, summarize_stress
from evidencegraph.episode_inference import LogisticCalibrator


def _episode(claim_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "split": "test",
        "label": False,
        "corruption": "support_refute_mix",
        "sources": [
            {
                "source_id": f"{claim_id}-counter-0",
                "metadata": {"controlled_fabrication": True},
            }
        ],
    }


def _nli(claim_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "source_id": f"{claim_id}-counter-0",
        "status": "complete",
        "probabilities": {"support": 0.95, "neutral": 0.03, "refute": 0.02},
    }


def test_independent_looking_fabrication_grows_effective_evidence() -> None:
    records = stress_records(
        [_episode("c1"), _episode("c2")],
        [_nli("c1"), _nli("c2")],
        LogisticCalibrator(scale=1.0, intercept=0.0),
        source_counts=(1, 5),
    )
    by_key = {
        (record["claim_id"], record["scenario"], record["source_count"]): record
        for record in records
    }

    assert by_key[("c1", "independent_looking", 5)]["effective_evidence"] > (
        by_key[("c1", "oracle_shared_campaign", 5)]["effective_evidence"]
    )
    assert by_key[("c1", "independent_looking", 5)][
        "probability_false_claim_true"
    ] > by_key[("c1", "oracle_shared_campaign", 5)][
        "probability_false_claim_true"
    ]


def test_fabrication_summary_uses_paired_complete_grid() -> None:
    records = stress_records(
        [_episode("c1"), _episode("c2")],
        [_nli("c1"), _nli("c2")],
        LogisticCalibrator(scale=1.0, intercept=0.0),
        source_counts=(1, 5),
    )

    summary = summarize_stress(records, n_resamples=100)

    assert summary["n_claims"] == 2
    assert set(summary["cells"]) == {
        "independent_looking",
        "oracle_shared_campaign",
    }
    assert set(summary["paired_differences"]) == {"1", "5"}
