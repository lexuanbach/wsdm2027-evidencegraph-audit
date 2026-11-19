"""Held-out boundary stress for independent-looking coordinated false sources."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .episode_evaluation import index_nli_records
from .episode_inference import LogisticCalibrator, score_episode


SCENARIOS = ("independent_looking", "oracle_shared_campaign")
DEFAULT_SOURCE_COUNTS = (1, 2, 5, 10, 20)


def stress_records(
    episodes: list[dict[str, Any]],
    nli_records: list[dict[str, Any]],
    calibrator: LogisticCalibrator,
    *,
    source_counts: Iterable[int] = DEFAULT_SOURCE_COUNTS,
) -> list[dict[str, Any]]:
    """Repeat each held-out fabricated counter-source under two dependence views.

    The independent-looking condition is the explicit false-negative boundary:
    the detector assigns zero pairwise dependence.  The oracle control assigns
    0.95 common-campaign dependence.  Both conditions reuse the same frozen NLI
    edge, so only the assumed source relation changes.
    """

    counts = tuple(int(value) for value in source_counts)
    if not counts or len(set(counts)) != len(counts) or any(value < 1 for value in counts):
        raise ValueError("fabrication source counts must be unique positive integers")
    nli_index = index_nli_records(nli_records)
    selected = [
        episode
        for episode in episodes
        if episode.get("split") == "test"
        and not bool(episode.get("label"))
        and episode.get("corruption") == "support_refute_mix"
    ]
    if not selected:
        raise ValueError("fabrication stress has no held-out false support/refute claims")

    output: list[dict[str, Any]] = []
    for episode in selected:
        claim_id = str(episode["claim_id"])
        counter_ids = [
            str(source["source_id"])
            for source in episode.get("sources", [])
            if str(source.get("source_id", "")).endswith("-counter-0")
            and bool(source.get("metadata", {}).get("controlled_fabrication"))
        ]
        if len(counter_ids) != 1:
            raise ValueError(f"claim {claim_id} lacks exactly one fabricated counter-source")
        key = (claim_id, counter_ids[0])
        if key not in nli_index:
            raise ValueError(f"claim {claim_id} lacks its fabricated-source NLI edge")
        edge = nli_index[key]
        for count in counts:
            probabilities = [edge] * count
            for scenario in SCENARIOS:
                dependence = np.eye(count, dtype=np.float64)
                if scenario == "oracle_shared_campaign":
                    dependence.fill(0.95)
                    np.fill_diagonal(dependence, 1.0)
                score = score_episode(probabilities, dependence)
                probability = float(calibrator.predict([score.evidence_logit])[0])
                output.append(
                    {
                        "claim_id": claim_id,
                        "split": "test",
                        "label": 0,
                        "scenario": scenario,
                        "source_count": count,
                        "counter_source_id": counter_ids[0],
                        "nli_probabilities": edge,
                        **score.to_dict(),
                        "probability_false_claim_true": probability,
                        "false_claim_accepted": probability >= 0.5,
                    }
                )
    expected = len(selected) * len(counts) * len(SCENARIOS)
    if len(output) != expected:
        raise RuntimeError("fabrication stress row count drifted")
    return output


def summarize_stress(
    records: list[dict[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 103,
) -> dict[str, Any]:
    """Summarize each stress cell with paired claim-bootstrap intervals."""

    if not records or n_resamples < 100:
        raise ValueError("fabrication summary requires rows and >=100 resamples")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (str(record["scenario"]), int(record["source_count"]))
        claim_id = str(record["claim_id"])
        if claim_id in grouped[key]:
            raise ValueError(f"duplicate fabrication stress row for {key}/{claim_id}")
        grouped[key][claim_id] = record
    expected_keys = {
        (scenario, count)
        for scenario in SCENARIOS
        for count in sorted({int(record["source_count"]) for record in records})
    }
    if set(grouped) != expected_keys:
        raise ValueError("fabrication stress does not cover the scenario/count grid")
    claim_sets = [set(rows) for rows in grouped.values()]
    if any(claims != claim_sets[0] for claims in claim_sets[1:]):
        raise ValueError("fabrication stress cells do not share the same claims")
    claims = sorted(claim_sets[0])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(claims), size=(n_resamples, len(claims)))

    cells: dict[str, dict[str, Any]] = {scenario: {} for scenario in SCENARIOS}
    for scenario, count in sorted(grouped):
        rows = grouped[(scenario, count)]
        probabilities = np.asarray(
            [float(rows[claim]["probability_false_claim_true"]) for claim in claims]
        )
        accepted = np.asarray(
            [float(bool(rows[claim]["false_claim_accepted"])) for claim in claims]
        )
        probability_bootstrap = np.mean(probabilities[samples], axis=1)
        acceptance_bootstrap = np.mean(accepted[samples], axis=1)
        p_low, p_high = np.quantile(probability_bootstrap, [0.025, 0.975])
        a_low, a_high = np.quantile(acceptance_bootstrap, [0.025, 0.975])
        cells[scenario][str(count)] = {
            "n_claims": len(claims),
            "mean_false_claim_probability": float(np.mean(probabilities)),
            "mean_false_claim_probability_low": float(p_low),
            "mean_false_claim_probability_high": float(p_high),
            "false_acceptance": float(np.mean(accepted)),
            "false_acceptance_low": float(a_low),
            "false_acceptance_high": float(a_high),
        }

    paired: dict[str, Any] = {}
    for count in sorted(int(value) for value in cells[SCENARIOS[0]]):
        independent = grouped[("independent_looking", count)]
        oracle = grouped[("oracle_shared_campaign", count)]
        probability_delta = np.asarray(
            [
                float(independent[claim]["probability_false_claim_true"])
                - float(oracle[claim]["probability_false_claim_true"])
                for claim in claims
            ]
        )
        acceptance_delta = np.asarray(
            [
                float(bool(independent[claim]["false_claim_accepted"]))
                - float(bool(oracle[claim]["false_claim_accepted"]))
                for claim in claims
            ]
        )
        p_bootstrap = np.mean(probability_delta[samples], axis=1)
        a_bootstrap = np.mean(acceptance_delta[samples], axis=1)
        paired[str(count)] = {
            "mean_probability_delta_independent_minus_oracle": float(
                np.mean(probability_delta)
            ),
            "probability_delta_low": float(np.quantile(p_bootstrap, 0.025)),
            "probability_delta_high": float(np.quantile(p_bootstrap, 0.975)),
            "false_acceptance_delta_independent_minus_oracle": float(
                np.mean(acceptance_delta)
            ),
            "false_acceptance_delta_low": float(np.quantile(a_bootstrap, 0.025)),
            "false_acceptance_delta_high": float(np.quantile(a_bootstrap, 0.975)),
        }
    return {
        "n_claims": len(claims),
        "bootstrap_resamples": n_resamples,
        "cells": cells,
        "paired_differences": paired,
    }
