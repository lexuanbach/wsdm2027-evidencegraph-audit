"""Strict joins from raw edge predictions to episode-level experiment records."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .episode_inference import EpisodeScore, score_episode
from .metrics import accuracy, brier_score, expected_calibration_error, reliability_bins


VARIANT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "evidencegraph": {},
    "no_dependence": {"use_dependence": False},
    "no_contradiction": {"use_contradiction": False},
    "no_effective_evidence": {"use_effective_evidence": False},
    "hard_dependence": {"hard_dependence_threshold": 0.5},
}


def index_nli_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    index: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        if record.get("status") != "complete":
            raise ValueError("episode evaluation requires complete NLI records")
        key = (str(record["claim_id"]), str(record["source_id"]))
        if key in index:
            raise ValueError(f"duplicate NLI key: {key}")
        index[key] = {
            label: float(record["probabilities"][label])
            for label in ("support", "neutral", "refute")
        }
    if not index:
        raise ValueError("NLI record index is empty")
    return index


def index_dependence_records(
    records: list[dict[str, Any]], *, expected_seeds: tuple[int, ...] = (103, 104, 105)
) -> dict[tuple[str, str, str], float]:
    seed_keys = {str(seed) for seed in expected_seeds}
    index: dict[tuple[str, str, str], float] = {}
    for record in records:
        left, right = sorted((str(record["left_id"]), str(record["right_id"])))
        key = (str(record["claim_id"]), left, right)
        if key in index:
            raise ValueError(f"duplicate dependence key: {key}")
        probabilities = record.get("probabilities", {})
        if set(probabilities) != seed_keys:
            raise ValueError(
                f"dependence key {key} does not contain exactly seeds {sorted(seed_keys)}"
            )
        values = np.asarray([float(probabilities[seed]) for seed in sorted(seed_keys)])
        if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"invalid dependence probabilities for {key}")
        index[key] = float(np.mean(values))
    if not index:
        raise ValueError("dependence record index is empty")
    return index


def episode_inputs(
    episode: dict[str, Any],
    nli_index: dict[tuple[str, str], dict[str, float]],
    dependence_index: dict[tuple[str, str, str], float],
) -> tuple[list[dict[str, float]], np.ndarray]:
    claim_id = str(episode["claim_id"])
    source_ids = [str(source["source_id"]) for source in episode["sources"]]
    if len(source_ids) != len(set(source_ids)) or not source_ids:
        raise ValueError(f"episode {claim_id} has invalid source IDs")
    missing_nli = [source_id for source_id in source_ids if (claim_id, source_id) not in nli_index]
    if missing_nli:
        raise ValueError(f"episode {claim_id} lacks {len(missing_nli)} NLI edges")
    probabilities = [nli_index[(claim_id, source_id)] for source_id in source_ids]
    dependence = np.eye(len(source_ids), dtype=np.float64)
    missing_pairs = []
    positions = {source_id: index for index, source_id in enumerate(source_ids)}
    for left, right in combinations(source_ids, 2):
        key = (claim_id, *sorted((left, right)))
        if key not in dependence_index:
            missing_pairs.append(key)
            continue
        i, j = positions[left], positions[right]
        dependence[i, j] = dependence[j, i] = dependence_index[key]
    if missing_pairs:
        raise ValueError(f"episode {claim_id} lacks {len(missing_pairs)} dependence edges")
    return probabilities, dependence


def score_dataset_split(
    episodes: list[dict[str, Any]],
    nli_records: list[dict[str, Any]],
    dependence_records: list[dict[str, Any]],
    *,
    split: str,
    variant: str = "evidencegraph",
) -> list[dict[str, Any]]:
    nli_index = index_nli_records(nli_records)
    dependence_index = index_dependence_records(dependence_records)
    return score_dataset_split_indexed(
        episodes,
        nli_index,
        dependence_index,
        split=split,
        variant=variant,
    )


def score_dataset_split_indexed(
    episodes: list[dict[str, Any]],
    nli_index: dict[tuple[str, str], dict[str, float]],
    dependence_index: dict[tuple[str, str, str], float],
    *,
    split: str,
    variant: str = "evidencegraph",
) -> list[dict[str, Any]]:
    if variant not in VARIANT_ARGUMENTS:
        raise ValueError(f"unknown episode variant: {variant}")
    output = []
    for episode in episodes:
        if episode["split"] != split:
            continue
        probabilities, dependence = episode_inputs(episode, nli_index, dependence_index)
        score: EpisodeScore = score_episode(
            probabilities,
            dependence,
            **VARIANT_ARGUMENTS[variant],
        )
        off_diagonal = dependence[np.triu_indices(len(dependence), k=1)]
        output.append(
            {
                "claim_id": str(episode["claim_id"]),
                "split": split,
                "label": int(bool(episode["label"])),
                "corruption": str(episode["corruption"]),
                "variant": variant,
                "mean_dependence": float(np.mean(off_diagonal)),
                "dependence_confidence": float(np.mean(np.abs(off_diagonal - 0.5) * 2.0)),
                **score.to_dict(),
            }
        )
    if not output:
        raise ValueError(f"dataset has no episodes for split {split}")
    return output


def summarize_episode_predictions(
    records: list[dict[str, Any]], *, n_resamples: int = 1000, seed: int = 103
) -> dict[str, Any]:
    if not records or n_resamples < 100:
        raise ValueError("summary requires records and at least 100 bootstrap resamples")
    keys = [str(record["claim_id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("episode prediction claim IDs must be unique")
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    probabilities = np.asarray(
        [float(record["probability_true"]) for record in records], dtype=np.float64
    )
    if not set(np.unique(labels)).issubset({0, 1}) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("invalid episode labels or calibrated probabilities")
    groups: dict[str, np.ndarray] = {"overall": np.ones(len(records), dtype=bool)}
    for corruption in sorted({str(record["corruption"]) for record in records}):
        groups[corruption] = np.asarray(
            [record["corruption"] == corruption for record in records], dtype=bool
        )
    output: dict[str, Any] = {}
    for group_index, (name, mask) in enumerate(groups.items()):
        y, p = labels[mask], probabilities[mask]
        if len(y) == 0:
            raise ValueError(f"empty episode summary group: {name}")

        def metrics(sample_y: np.ndarray, sample_p: np.ndarray) -> dict[str, float]:
            negative = sample_y == 0
            return {
                "accuracy": accuracy(sample_y, sample_p),
                "brier": brier_score(sample_y, sample_p),
                "ece": expected_calibration_error(sample_y, sample_p),
                "false_acceptance": float(np.mean(sample_p[negative] >= 0.5))
                if np.any(negative)
                else float("nan"),
            }

        point = metrics(y, p)
        rng = np.random.default_rng(seed + group_index)
        bootstrap: dict[str, list[float]] = {key: [] for key in point}
        for _ in range(n_resamples):
            sample = rng.integers(0, len(y), len(y))
            for metric, value in metrics(y[sample], p[sample]).items():
                if np.isfinite(value):
                    bootstrap[metric].append(value)
        group_summary: dict[str, Any] = {
            "n_claims": int(len(y)),
            "n_false_claims": int(np.sum(y == 0)),
            **point,
            "reliability_bins": reliability_bins(y, p, n_bins=10),
        }
        for metric, values in bootstrap.items():
            if values:
                low, high = np.quantile(np.asarray(values), [0.025, 0.975])
                group_summary[f"{metric}_low"] = float(low)
                group_summary[f"{metric}_high"] = float(high)
        output[name] = group_summary
    return {"bootstrap_resamples": n_resamples, "groups": output}
