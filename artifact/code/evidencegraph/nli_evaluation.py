"""Claim-clustered metrics for three-way NLI source edges."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

import numpy as np

from .schema import ProvenanceFamily


LABELS = ("support", "neutral", "refute")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
SCORING_PROTOCOL = "nli-source-edge-v2-content-bound"


def nli_input_sha256(claim: str, source_text: str) -> str:
    """Hash the exact ordered text pair presented to the NLI checkpoint."""

    serialized = json.dumps(
        {"claim": str(claim), "source_text": str(source_text)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def gold_nli_label(stance: float) -> str:
    if stance > 0.0:
        return "support"
    if stance < 0.0:
        return "refute"
    return "neutral"


def validate_nli_resume(
    records: list[dict[str, Any]],
    expected: dict[tuple[str, str], tuple[str, str, str]],
    *,
    split: str,
    model: str,
    revision: str,
    dataset_sha256: str,
) -> set[tuple[str, str]]:
    """Validate append-only scorer rows against the frozen dataset/configuration."""
    completed: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record["claim_id"]), str(record["source_id"]))
        if key in completed:
            raise ValueError(f"duplicate NLI source key: {key}")
        if key not in expected:
            raise ValueError(f"unexpected NLI source key: {key}")
        expected_family, expected_gold, expected_input_sha256 = expected[key]
        if (
            record.get("status") != "complete"
            or record.get("split") != split
            or record.get("model") != model
            or record.get("model_revision") != revision
            or record.get("scoring_protocol") != SCORING_PROTOCOL
            or record.get("family") != expected_family
            or record.get("gold_label") != expected_gold
            or record.get("dataset_sha256") != dataset_sha256
            or record.get("input_sha256") != expected_input_sha256
        ):
            raise ValueError("NLI output conflicts with the frozen run configuration")
        completed.add(key)
    return completed


def nli_run_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require one checkpoint/protocol while permitting explicit dev+test splits."""
    if not records or any(record.get("status") != "complete" for record in records):
        raise ValueError("NLI metadata requires complete records")
    configurations = {
        (
            str(record["model"]),
            str(record["model_revision"]),
            int(record["model_parameters"]),
            str(record["scoring_protocol"]),
            str(record["dataset_sha256"]),
        )
        for record in records
    }
    if len(configurations) != 1:
        raise ValueError("NLI records mix model configurations")
    model, revision, parameters, protocol, dataset_sha256 = configurations.pop()
    splits = sorted({str(record["split"]) for record in records})
    return {
        "model": model,
        "model_revision": revision,
        "model_parameters": parameters,
        "scoring_protocol": protocol,
        "dataset_sha256": dataset_sha256,
        "splits": splits,
        "records": len(records),
    }


def validate_nli_dataset(
    records: list[dict[str, Any]], dataset_sha256: str
) -> dict[str, Any]:
    """Require a complete NLI run to bind the dataset being evaluated."""

    metadata = nli_run_metadata(records)
    if metadata["dataset_sha256"] != dataset_sha256:
        raise ValueError("NLI records are bound to a different dataset checksum")
    return metadata


def _point_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    classes = np.unique(labels)
    precisions, recalls, f1s = [], [], []
    for class_index in classes:
        tp = np.sum((predictions == class_index) & (labels == class_index))
        fp = np.sum((predictions == class_index) & (labels != class_index))
        fn = np.sum((predictions != class_index) & (labels == class_index))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if np.any(mask):
            ece += np.sum(mask) * abs(np.mean(confidence[mask]) - np.mean(correct[mask]))
    one_hot = np.eye(len(LABELS), dtype=float)[labels]
    return {
        "accuracy": float(np.mean(correct)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(ece / len(labels)),
        "gold_classes": [LABELS[index] for index in classes],
    }


def _claim_statistics(
    labels: np.ndarray, probabilities: np.ndarray, claim_ids: np.ndarray
) -> tuple[list[str], np.ndarray]:
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    bins = np.minimum((confidence * 10).astype(int), 9)
    one_hot = np.eye(len(LABELS), dtype=float)[labels]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, claim_id in enumerate(claim_ids):
        grouped[str(claim_id)].append(index)
    names = sorted(grouped)
    # n, correct, brier_sum, 3x3 confusion, 10*(count, confidence_sum, correct_sum)
    stats = np.zeros((len(names), 42), dtype=np.float64)
    for row, claim_id in enumerate(names):
        index = np.asarray(grouped[claim_id], dtype=int)
        y, pred, p = labels[index], predictions[index], probabilities[index]
        stats[row, 0] = len(index)
        stats[row, 1] = np.sum(pred == y)
        stats[row, 2] = np.sum(np.sum((p - one_hot[index]) ** 2, axis=1))
        for gold in range(3):
            for predicted in range(3):
                stats[row, 3 + 3 * gold + predicted] = np.sum(
                    (y == gold) & (pred == predicted)
                )
        for bin_index in range(10):
            mask = bins[index] == bin_index
            offset = 12 + 3 * bin_index
            stats[row, offset] = np.sum(mask)
            stats[row, offset + 1] = np.sum(confidence[index][mask])
            stats[row, offset + 2] = np.sum(correct[index][mask])
    return names, stats


def _metrics_from_stats(stats: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    n, correct, brier_sum = stats[:3]
    confusion = stats[3:12].reshape(3, 3)
    precisions, recalls, f1s = [], [], []
    for class_index in classes:
        tp = confusion[class_index, class_index]
        fp = np.sum(confusion[:, class_index]) - tp
        fn = np.sum(confusion[class_index, :]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    ece = 0.0
    for bin_index in range(10):
        count, confidence_sum, correct_sum = stats[12 + 3 * bin_index : 15 + 3 * bin_index]
        if count:
            ece += count * abs(confidence_sum / count - correct_sum / count)
    return {
        "accuracy": float(correct / n),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "brier": float(brier_sum / n),
        "ece": float(ece / n),
    }


def summarize_nli_records(
    records: list[dict[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 103,
) -> dict[str, Any]:
    complete = [record for record in records if record.get("status") == "complete"]
    if len(complete) != len(records) or not complete:
        raise ValueError("NLI records must be non-empty and complete")
    keys = [(record["claim_id"], record["source_id"]) for record in complete]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate NLI source keys")
    metadata_records = [
        {**record, "scoring_protocol": record.get("scoring_protocol", SCORING_PROTOCOL)}
        for record in complete
    ]
    run = nli_run_metadata(metadata_records)
    labels = np.asarray([LABEL_INDEX[str(record["gold_label"])] for record in complete])
    probabilities = np.asarray(
        [
            [float(record["probabilities"][label]) for label in LABELS]
            for record in complete
        ],
        dtype=np.float64,
    )
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-5):
        raise ValueError("NLI probabilities do not sum to one")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("NLI probabilities must be finite and in [0, 1]")
    claim_ids = np.asarray([str(record["claim_id"]) for record in complete], dtype=object)
    groups = {"overall": np.ones(len(complete), dtype=bool)}
    for family in ProvenanceFamily:
        groups[family.value] = np.asarray(
            [record["family"] == family.value for record in complete], dtype=bool
        )
    output = {}
    for group_index, (name, mask) in enumerate(groups.items()):
        if not np.any(mask):
            raise ValueError(f"empty NLI family group {name}")
        y, p, claims = labels[mask], probabilities[mask], claim_ids[mask]
        point = _point_metrics(y, p)
        classes = np.unique(y)
        claim_names, sufficient = _claim_statistics(y, p, claims)
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
            for metric, value in _metrics_from_stats(aggregate, classes).items():
                bootstrap[metric].append(value)
        record = {
            "n_sources": int(np.sum(mask)),
            "n_claims": len(claim_names),
            **point,
        }
        for metric, values in bootstrap.items():
            low, high = np.quantile(np.asarray(values), [0.025, 0.975])
            record[f"{metric}_low"] = float(low)
            record[f"{metric}_high"] = float(high)
        output[name] = record
    return {
        "model": run["model"],
        "model_revision": run["model_revision"],
        "model_parameters": run["model_parameters"],
        "scoring_protocol": run["scoring_protocol"],
        "bootstrap_resamples": n_resamples,
        "groups": output,
    }
