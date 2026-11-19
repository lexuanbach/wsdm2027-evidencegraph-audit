"""Claim-level benchmark resource accounting with bootstrap intervals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .bootstrap import multinomial_weight_batches


COST_METRICS = (
    "searches",
    "documents_opened",
    "prompt_tokens",
    "completion_tokens",
    "llm_calls",
    "end_to_end_wall_seconds",
)


def _cost_row(prediction: dict[str, Any]) -> list[float]:
    required = {
        "searches",
        "documents_opened",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
        "wall_seconds",
        "retrieval_wall_seconds",
        "edge_and_selection_wall_seconds",
    }
    missing = sorted(required - set(prediction))
    if missing:
        raise ValueError(f"benchmark prediction lacks cost fields: {missing}")
    values = [
        float(prediction["searches"]),
        float(prediction["documents_opened"]),
        float(prediction["prompt_tokens"]),
        float(prediction["completion_tokens"]),
        float(prediction["llm_calls"]),
        float(prediction["wall_seconds"])
        + float(prediction["retrieval_wall_seconds"])
        + float(prediction["edge_and_selection_wall_seconds"]),
    ]
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("benchmark prediction has invalid cost fields")
    return values


def summarize_benchmark_cost(
    joined: list[dict[str, Any]], *, n_resamples: int = 1000, seed: int = 103
) -> dict[str, Any]:
    if not joined or n_resamples < 100:
        raise ValueError("cost summary requires predictions and 100 bootstraps")
    datasets = sorted({str(row["case"]["dataset"]) for row in joined})
    groups = [(dataset, [row for row in joined if row["case"]["dataset"] == dataset]) for dataset in datasets]
    groups.append(("ALL", joined))
    output = {}
    for group_index, (dataset, rows) in enumerate(groups):
        values = np.asarray(
            [_cost_row(row["prediction"]) for row in rows], dtype=np.float64
        )
        rng = np.random.default_rng(seed + group_index)
        boot = np.concatenate(
            [
                np.einsum(
                    "ij,jk->ik",
                    weights.astype(np.float64),
                    values,
                    optimize=False,
                )
                / len(rows)
                for weights in multinomial_weight_batches(
                    rng, len(rows), n_resamples
                )
            ],
            axis=0,
        )
        summary: dict[str, Any] = {"n_cases": len(rows)}
        for metric_index, metric in enumerate(COST_METRICS):
            low, high = np.quantile(boot[:, metric_index], [0.025, 0.975])
            summary[f"mean_{metric}"] = float(np.mean(values[:, metric_index]))
            summary[f"mean_{metric}_low"] = float(low)
            summary[f"mean_{metric}_high"] = float(high)
        output[dataset] = summary
    return {"bootstrap_resamples": n_resamples, "datasets": output}
