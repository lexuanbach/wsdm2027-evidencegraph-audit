"""Controller rollout evaluation and dependence-stratified diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .controller_rollout import Controller, rollout_controller
from .episode_evaluation import episode_inputs
from .episode_inference import LogisticCalibrator, score_episode


ROLLOUT_METHODS = (
    "fixed_top5",
    "evidencegraph_active",
    "no_dependence",
    "no_active_seek",
    "no_contradiction",
    "no_effective_evidence",
    "hard_dependence",
)
METHOD_OPTIONS: dict[str, dict[str, Any]] = {
    "fixed_top5": {},
    "evidencegraph_active": {},
    "no_dependence": {"use_dependence": False},
    "no_active_seek": {},
    "no_contradiction": {"use_contradiction": False},
    "no_effective_evidence": {"use_effective_evidence": False},
    "hard_dependence": {"hard_dependence_threshold": 0.5},
}


def rollout_dataset_split(
    episodes: list[dict[str, Any]],
    nli_index: dict[tuple[str, str], dict[str, float]],
    dependence_index: dict[tuple[str, str, str], float],
    controller: Controller,
    *,
    split: str,
    method: str,
    initial_sources: int = 5,
    action_budget: int = 3,
) -> list[dict[str, Any]]:
    if method not in ROLLOUT_METHODS:
        raise ValueError(f"unknown rollout method: {method}")
    records = []
    for episode in episodes:
        if episode["split"] != split:
            continue
        probabilities, dependence = episode_inputs(
            episode, nli_index, dependence_index
        )
        initial = list(range(min(initial_sources, len(probabilities))))
        initial_dependence = dependence[np.ix_(initial, initial)]
        off_diagonal = initial_dependence[
            np.triu_indices(len(initial_dependence), k=1)
        ]
        shared = {
            "claim_id": str(episode["claim_id"]),
            "split": split,
            "label": int(bool(episode["label"])),
            "corruption": str(episode["corruption"]),
            "method": method,
            "mean_dependence": float(np.mean(off_diagonal)),
            "dependence_confidence": float(
                np.mean(np.abs(off_diagonal - 0.5) * 2.0)
            ),
            "dependence_scope": "initial_observed_graph",
        }
        if method == "fixed_top5":
            selected = initial
            subset = np.ix_(selected, selected)
            score = score_episode(
                [probabilities[index] for index in selected],
                dependence[subset],
                use_dependence=False,
            )
            records.append(
                {
                    **shared,
                    **score.to_dict(),
                    "selected_source_indices": selected,
                    "actions": [],
                    "find_indep_count": 0,
                    "search_count": 0,
                    "documents_opened": len(selected),
                }
            )
            continue
        rollout = rollout_controller(
            probabilities,
            dependence,
            controller,
            initial_sources=initial_sources,
            action_budget=action_budget,
            enable_independent_seek=method != "no_active_seek",
            score_options=METHOD_OPTIONS[method],
        )
        records.append({**shared, **rollout})
    if not records:
        raise ValueError(f"no rollout records for split {split}")
    return records


def calibrate_rollouts(
    records: list[dict[str, Any]], calibrator: LogisticCalibrator
) -> list[dict[str, Any]]:
    probabilities = calibrator.predict([record["evidence_logit"] for record in records])
    return [
        {**record, "probability_true": float(probability)}
        for record, probability in zip(records, probabilities)
    ]


def dependence_quartile_summary(
    records: list[dict[str, Any]], *, n_resamples: int = 1000, seed: int = 103
) -> dict[str, Any]:
    if len(records) < 4 or n_resamples < 100:
        raise ValueError("dependence summary requires four records and 100 bootstraps")
    if len({str(record["claim_id"]) for record in records}) != len(records):
        raise ValueError("dependence summary claim IDs must be unique")
    order = np.argsort(
        np.asarray([float(record["dependence_confidence"]) for record in records]),
        kind="stable",
    )
    buckets = np.array_split(order, 4)
    output: dict[str, list[Any]] = {
        "buckets": ["Q1 (least certain)", "Q2", "Q3", "Q4 (most certain)"],
        "n_claims": [],
    }
    for metric in ("accuracy", "false_acceptance", "find_indep_rate"):
        output[f"{metric}_mean"] = []
        output[f"{metric}_low"] = []
        output[f"{metric}_high"] = []
    output["confidence_range"] = []
    for bucket_index, indices in enumerate(buckets):
        selected = [records[index] for index in indices]
        y = np.asarray([int(record["label"]) for record in selected])
        p = np.asarray([float(record["probability_true"]) for record in selected])
        find = np.asarray([int(record["find_indep_count"] > 0) for record in selected])
        confidence = np.asarray(
            [float(record["dependence_confidence"]) for record in selected]
        )

        def values(sample: np.ndarray) -> tuple[float, float, float]:
            sy, sp = y[sample], p[sample]
            negative = sy == 0
            return (
                float(np.mean((sp >= 0.5) == sy)),
                float(np.mean(sp[negative] >= 0.5)) if np.any(negative) else float("nan"),
                float(np.mean(find[sample])),
            )

        complete = np.arange(len(selected))
        point = values(complete)
        rng = np.random.default_rng(seed + bucket_index)
        boot = [[], [], []]
        for _ in range(n_resamples):
            sample = rng.integers(0, len(selected), len(selected))
            for metric_index, value in enumerate(values(sample)):
                if np.isfinite(value):
                    boot[metric_index].append(value)
        output["n_claims"].append(len(selected))
        output["confidence_range"].append(
            [float(np.min(confidence)), float(np.max(confidence))]
        )
        for metric_index, metric in enumerate(
            ("accuracy", "false_acceptance", "find_indep_rate")
        ):
            low, high = np.quantile(np.asarray(boot[metric_index]), [0.025, 0.975])
            output[f"{metric}_mean"].append(point[metric_index])
            output[f"{metric}_low"].append(float(low))
            output[f"{metric}_high"].append(float(high))
    output["bootstrap_resamples"] = n_resamples
    output["bucketing"] = (
        "test-set rank quartiles of mean pairwise dependence confidence "
        "(2 * abs(probability - 0.5))"
    )
    return output
