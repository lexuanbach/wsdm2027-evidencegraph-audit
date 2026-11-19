"""Pair construction and claim-clustered evaluation for dependence edges."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .dependence_features import (
    FEATURE_NAMES,
    cache_source_features,
    pair_features_from_cache,
    pair_label,
)
from .metrics import expected_calibration_error
from .schema import ProvenanceFamily, SourceEvidence


FAMILY_NAMES = tuple(family.value for family in ProvenanceFamily)


def source_from_record(record: dict[str, Any]) -> SourceEvidence:
    return SourceEvidence(
        source_id=str(record["source_id"]),
        text=str(record["text"]),
        family=ProvenanceFamily(str(record["family"])),
        stance=float(record["stance"]),
        stance_confidence=float(record["stance_confidence"]),
        domain=str(record["domain"]),
        origin_id=str(record["origin_id"]),
        author=record.get("author"),
        timestamp=record.get("timestamp"),
        cited_source_ids=tuple(record.get("cited_source_ids", [])),
        synthetic=bool(record.get("synthetic", False)),
        metadata=dict(record.get("metadata", {})),
    )


def dependence_examples(
    episodes: list[dict[str, Any]], split: str, *, include_metadata: bool = True
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Build all within-episode source pairs for one leakage-safe split."""
    selected = [episode for episode in episodes if episode["split"] == split]
    pair_count = sum(
        len(episode["sources"]) * (len(episode["sources"]) - 1) // 2
        for episode in selected
    )
    if pair_count < 1:
        raise ValueError(f"no dependence pairs for split {split}")
    features = np.empty((pair_count, len(FEATURE_NAMES)), dtype=np.float64)
    labels = np.empty(pair_count, dtype=np.int64)
    metadata: list[dict[str, str]] = []
    row = 0
    for episode in selected:
        sources = [source_from_record(record) for record in episode["sources"]]
        caches = [cache_source_features(source) for source in sources]
        for left_index, right_index in combinations(range(len(sources)), 2):
            left, right = sources[left_index], sources[right_index]
            features[row] = pair_features_from_cache(
                left,
                right,
                caches[left_index],
                caches[right_index],
            )
            labels[row] = pair_label(left, right)
            if include_metadata:
                metadata.append(
                    {
                        "claim_id": str(episode["claim_id"]),
                        "left_id": left.source_id,
                        "right_id": right.source_id,
                        "left_family": left.family.value,
                        "right_family": right.family.value,
                    }
                )
            row += 1
    if row != pair_count or (include_metadata and len(metadata) != pair_count):
        raise RuntimeError("dependence pair preallocation count drifted")
    return features, labels, metadata


def _point_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": expected_calibration_error(labels, probabilities, n_bins=10),
        "auroc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else float("nan"),
    }


def _claim_sufficient_statistics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    claim_ids: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Return additive claim statistics for fast cluster bootstrapping."""
    bins = np.minimum((probabilities * 10).astype(int), 9)
    predictions = probabilities >= 0.5
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, claim_id in enumerate(claim_ids):
        grouped[str(claim_id)].append(index)
    names = sorted(grouped)
    # n, correct, tp, fp, fn, brier_sum, then 10*(count, probability_sum, label_sum)
    stats = np.zeros((len(names), 36), dtype=np.float64)
    for row, claim_id in enumerate(names):
        index = np.asarray(grouped[claim_id], dtype=int)
        y = labels[index]
        p = probabilities[index]
        pred = predictions[index]
        stats[row, 0] = len(index)
        stats[row, 1] = np.sum(pred == y)
        stats[row, 2] = np.sum(pred & (y == 1))
        stats[row, 3] = np.sum(pred & (y == 0))
        stats[row, 4] = np.sum(~pred & (y == 1))
        stats[row, 5] = np.sum((p - y) ** 2)
        for bin_index in range(10):
            mask = bins[index] == bin_index
            offset = 6 + 3 * bin_index
            stats[row, offset] = np.sum(mask)
            stats[row, offset + 1] = np.sum(p[mask])
            stats[row, offset + 2] = np.sum(y[mask])
    return names, stats


def _metrics_from_sufficient(stats: np.ndarray) -> dict[str, float]:
    total = stats[0]
    correct, tp, fp, fn, brier_sum = stats[1:6]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ece = 0.0
    for bin_index in range(10):
        count, probability_sum, label_sum = stats[6 + 3 * bin_index : 9 + 3 * bin_index]
        if count:
            ece += count * abs(probability_sum / count - label_sum / count)
    return {
        "accuracy": float(correct / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(brier_sum / total),
        "ece": float(ece / total),
    }


def clustered_metric_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: list[dict[str, str]],
    *,
    n_resamples: int = 1000,
    seed: int = 103,
) -> dict[str, Any]:
    """Report overall/family metrics with claim-cluster bootstrap intervals."""
    if labels.shape != probabilities.shape or labels.ndim != 1 or len(metadata) != len(labels):
        raise ValueError("labels, probabilities, and metadata must align")
    if n_resamples < 100:
        raise ValueError("use at least 100 bootstrap resamples")
    claim_ids = np.asarray([record["claim_id"] for record in metadata], dtype=object)
    output: dict[str, Any] = {}
    groups = {"overall": np.ones(len(labels), dtype=bool)}
    for family in FAMILY_NAMES:
        groups[family] = np.asarray(
            [
                record["left_family"] == family or record["right_family"] == family
                for record in metadata
            ],
            dtype=bool,
        )
    for group_index, (name, mask) in enumerate(groups.items()):
        y, p, claims = labels[mask], probabilities[mask], claim_ids[mask]
        if len(y) == 0 or len(np.unique(y)) < 2:
            raise ValueError(f"group {name} does not contain both dependence labels")
        point = _point_metrics(y, p)
        claim_names, sufficient = _claim_sufficient_statistics(y, p, claims)
        rng = np.random.default_rng(seed + group_index)
        weights = rng.multinomial(
            len(claim_names),
            np.full(len(claim_names), 1.0 / len(claim_names)),
            size=n_resamples,
        )
        aggregates = np.einsum(
            "ij,jk->ik", weights.astype(np.float64), sufficient, optimize=False
        )
        bootstrap: dict[str, list[float]] = defaultdict(list)
        for aggregate in aggregates:
            values = _metrics_from_sufficient(aggregate)
            for metric, value in values.items():
                bootstrap[metric].append(value)
        record: dict[str, Any] = {
            "n_pairs": int(len(y)),
            "n_claims": len(claim_names),
            **point,
        }
        for metric, values in bootstrap.items():
            low, high = np.quantile(np.asarray(values), [0.025, 0.975])
            record[f"{metric}_low"] = float(low)
            record[f"{metric}_high"] = float(high)
        output[name] = record
    return {
        "feature_names": list(FEATURE_NAMES),
        "bootstrap_resamples": n_resamples,
        "groups": output,
    }


def grouped_point_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: list[dict[str, str]],
) -> dict[str, Any]:
    """Point estimates for individual seeds; CIs belong to the three-seed mean."""
    if labels.shape != probabilities.shape or labels.ndim != 1 or len(metadata) != len(labels):
        raise ValueError("labels, probabilities, and metadata must align")
    groups = {"overall": np.ones(len(labels), dtype=bool)}
    for family in FAMILY_NAMES:
        groups[family] = np.asarray(
            [
                record["left_family"] == family or record["right_family"] == family
                for record in metadata
            ],
            dtype=bool,
        )
    output = {}
    for name, mask in groups.items():
        if not np.any(mask) or len(np.unique(labels[mask])) < 2:
            raise ValueError(f"group {name} does not contain both dependence labels")
        output[name] = {
            "n_pairs": int(np.sum(mask)),
            "n_claims": len({metadata[index]["claim_id"] for index in np.flatnonzero(mask)}),
            **_point_metrics(labels[mask], probabilities[mask]),
        }
    return {"feature_names": list(FEATURE_NAMES), "groups": output}
