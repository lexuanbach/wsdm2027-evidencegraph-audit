"""End-to-end benchmark quality, calibration, and citation-source metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .benchmark_answerer import BENCHMARK_LABELS
from .bootstrap import multinomial_weight_batches


_ATTEMPT_STATUSES = frozenset(
    {"complete", "schema_failure", "protocol_failure", "backend_failure"}
)


def resolve_benchmark_attempts(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve append-only retries to exactly one completion per attempted case."""
    if not predictions:
        raise ValueError("benchmark attempt resolution requires predictions")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in predictions:
        status = str(record.get("status", ""))
        if status not in _ATTEMPT_STATUSES:
            raise ValueError(f"unknown benchmark attempt status: {status!r}")
        key = (str(record.get("dataset", "")), str(record.get("case_id", "")))
        if not all(key):
            raise ValueError("benchmark attempt lacks dataset or case ID")
        groups.setdefault(key, []).append(record)
    resolved = []
    for key, attempts in groups.items():
        identities = {
            (
                str(record.get("method", "")),
                str(record.get("model_requested", "")),
                int(record.get("seed", -1)),
            )
            for record in attempts
        }
        configurations = set()
        for record in attempts:
            available = [
                record[name]
                for name in ("answer_config", "react_config")
                if name in record
            ]
            if len(available) != 1 or not isinstance(available[0], dict):
                raise ValueError(f"benchmark attempt {key} lacks one run configuration")
            configurations.add(
                json.dumps(available[0], sort_keys=True, separators=(",", ":"))
            )
        if len(identities) != 1 or len(configurations) != 1:
            raise ValueError(f"benchmark attempts mix run identity for {key}")
        complete = [record for record in attempts if record["status"] == "complete"]
        if len(complete) != 1:
            raise ValueError(
                f"benchmark case {key} has {len(complete)} completed attempts"
            )
        completion_index = attempts.index(complete[0])
        if any(
            record["status"] != "complete"
            for record in attempts[completion_index + 1 :]
        ):
            raise ValueError(f"benchmark case {key} has failures after completion")
        resolved.append(complete[0])
    return resolved


def benchmark_run_metadata(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Require one method/model/seed/config before producing a run summary."""
    predictions = resolve_benchmark_attempts(predictions)
    identities = {
        (
            str(record.get("method", "")),
            str(record.get("model_requested", "")),
            int(record.get("seed", -1)),
        )
        for record in predictions
    }
    identity = next(iter(identities))
    if len(identities) != 1 or not identity[0] or not identity[1]:
        raise ValueError("benchmark predictions mix method, model, or seed identities")
    resolved_models = {
        str(record["model"]) for record in predictions if record.get("model")
    }
    if len(resolved_models) > 1:
        raise ValueError("benchmark predictions mix resolved model identities")
    configs = []
    for record in predictions:
        available = [
            record[key]
            for key in ("answer_config", "react_config")
            if key in record
        ]
        if len(available) != 1 or not isinstance(available[0], dict):
            raise ValueError("benchmark prediction lacks one exact run configuration")
        configs.append(json.dumps(available[0], sort_keys=True, separators=(",", ":")))
    if len(set(configs)) != 1:
        raise ValueError("benchmark predictions contain mixed run configurations")
    method, model_requested, seed = identity
    return {
        "method": method,
        "model_requested": model_requested,
        "model_resolved": next(iter(resolved_models), None),
        "seed": seed,
        "configuration": json.loads(configs[0]),
        "n_predictions": len(predictions),
    }


def join_benchmark_predictions(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    predictions = resolve_benchmark_attempts(predictions)
    index = {}
    for record in predictions:
        key = (str(record["dataset"]), str(record["case_id"]))
        if key in index:
            raise ValueError(f"duplicate benchmark prediction: {key}")
        index[key] = record
    joined = []
    case_keys = set()
    for case in cases:
        key = (str(case["dataset"]), str(case["case_id"]))
        if key in case_keys:
            raise ValueError(f"duplicate benchmark case: {key}")
        case_keys.add(key)
        if key not in index:
            raise ValueError(f"missing benchmark prediction: {key}")
        joined.append({"case": case, "prediction": index[key]})
    extra = sorted(set(index) - case_keys)
    if extra:
        raise ValueError(f"benchmark predictions contain {len(extra)} extra cases")
    return joined


def _metrics_from_sufficient(stats: np.ndarray, n_classes: int) -> dict[str, float]:
    n, correct, brier_sum = stats[:3]
    confusion = stats[3 : 3 + n_classes * n_classes].reshape(n_classes, n_classes)
    f1s = []
    for class_index in range(n_classes):
        tp = confusion[class_index, class_index]
        fp = np.sum(confusion[:, class_index]) - tp
        fn = np.sum(confusion[class_index, :]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    calibration_offset = 3 + n_classes * n_classes
    ece = 0.0
    for bin_index in range(10):
        count, confidence_sum, correct_sum = stats[
            calibration_offset + 3 * bin_index : calibration_offset + 3 * (bin_index + 1)
        ]
        if count:
            ece += count * abs(confidence_sum / count - correct_sum / count)
    return {
        "accuracy": float(correct / n),
        "macro_f1": float(np.mean(f1s)),
        "brier": float(brier_sum / n),
        "ece": float(ece / n),
    }


def _reliability_bins(stats: np.ndarray, n_classes: int) -> list[dict[str, float | int]]:
    """Return non-empty top-label calibration bins from the same ECE sufficient stats."""
    calibration_offset = 3 + n_classes * n_classes
    bins = []
    for bin_index in range(10):
        count, confidence_sum, correct_sum = stats[
            calibration_offset + 3 * bin_index : calibration_offset + 3 * (bin_index + 1)
        ]
        if not count:
            continue
        bins.append(
            {
                "bin_lower": bin_index / 10.0,
                "bin_upper": (bin_index + 1) / 10.0,
                "mean_confidence": float(confidence_sum / count),
                "empirical_accuracy": float(correct_sum / count),
                "count": int(count),
            }
        )
    return bins


def summarize_verdict_predictions(
    joined: list[dict[str, Any]], *, n_resamples: int = 1000, seed: int = 103
) -> dict[str, Any]:
    if not joined or n_resamples < 100:
        raise ValueError("verdict summary requires records and 100 bootstraps")
    datasets = sorted(
        {row["case"]["dataset"] for row in joined if row["case"]["dataset"] != "ExpertQA"}
    )
    output = {}
    for dataset_index, dataset in enumerate(datasets):
        rows = [row for row in joined if row["case"]["dataset"] == dataset]
        labels = BENCHMARK_LABELS[dataset]
        label_index = {label: index for index, label in enumerate(labels)}
        n_classes = len(labels)
        width = 3 + n_classes * n_classes + 30
        sufficient = np.zeros((len(rows), width), dtype=np.float64)
        for row_index, row in enumerate(rows):
            case, prediction = row["case"], row["prediction"]
            if set(prediction["probabilities"]) != set(labels):
                raise ValueError(f"{dataset} prediction has the wrong label probability keys")
            probability = np.asarray(
                [float(prediction["probabilities"][label]) for label in labels]
            )
            if np.any((probability < 0.0) | (probability > 1.0)) or not np.isclose(
                np.sum(probability), 1.0, atol=1e-5
            ):
                raise ValueError(f"{dataset} prediction has invalid probabilities")
            gold = label_index[str(case["gold_label"])]
            predicted = label_index[str(prediction["label"])]
            if predicted != int(np.argmax(probability)):
                raise ValueError(f"{dataset} prediction label is not the probability argmax")
            one_hot = np.eye(n_classes)[gold]
            sufficient[row_index, 0] = 1
            sufficient[row_index, 1] = int(predicted == gold)
            sufficient[row_index, 2] = np.sum((probability - one_hot) ** 2)
            sufficient[row_index, 3 + n_classes * gold + predicted] = 1
            confidence = float(np.max(probability))
            bin_index = min(int(confidence * 10), 9)
            offset = 3 + n_classes * n_classes + 3 * bin_index
            sufficient[row_index, offset : offset + 3] = [1, confidence, int(predicted == gold)]
        aggregate = np.sum(sufficient, axis=0)
        point = _metrics_from_sufficient(aggregate, n_classes)
        rng = np.random.default_rng(seed + dataset_index)
        boot = np.concatenate(
            [
                np.einsum(
                    "ij,jk->ik",
                    weights.astype(np.float64),
                    sufficient,
                    optimize=False,
                )
                for weights in multinomial_weight_batches(
                    rng, len(rows), n_resamples
                )
            ],
            axis=0,
        )
        bootstrap = {metric: [] for metric in point}
        for stats in boot:
            for metric, value in _metrics_from_sufficient(stats, n_classes).items():
                bootstrap[metric].append(value)
        record: dict[str, Any] = {
            "n_cases": len(rows),
            **point,
            "reliability_bins": _reliability_bins(aggregate, n_classes),
        }
        for metric, values in bootstrap.items():
            low, high = np.quantile(np.asarray(values), [0.025, 0.975])
            record[f"{metric}_low"] = float(low)
            record[f"{metric}_high"] = float(high)
        output[dataset] = record
    return {"bootstrap_resamples": n_resamples, "datasets": output}


def summarize_expertqa_citations(
    joined: list[dict[str, Any]], *, n_resamples: int = 1000, seed: int = 103
) -> dict[str, Any]:
    rows = [row for row in joined if row["case"]["dataset"] == "ExpertQA"]
    if not rows or n_resamples < 100:
        raise ValueError("ExpertQA summary requires records and 100 bootstraps")
    values = []
    abstentions = 0
    for row in rows:
        gold = set(map(str, row["case"]["gold_evidence"]))
        prediction = row["prediction"]
        cited = {
            str(source_id)
            for claim in prediction.get("claims", [])
            for source_id in claim.get("cited_source_ids", [])
        }
        abstentions += int(bool(prediction.get("abstained", False)))
        precision = len(cited & gold) / len(cited) if cited else float(not gold)
        recall = len(cited & gold) / len(gold) if gold else float("nan")
        values.append([precision, recall])
    array = np.asarray(values, dtype=np.float64)
    precision = array[:, 0]
    recall = array[np.isfinite(array[:, 1]), 1]
    rng = np.random.default_rng(seed)
    precision_boot = np.mean(
        precision[rng.integers(0, len(precision), size=(n_resamples, len(precision)))], axis=1
    )
    recall_boot = np.mean(
        recall[rng.integers(0, len(recall), size=(n_resamples, len(recall)))], axis=1
    )
    p_low, p_high = np.quantile(precision_boot, [0.025, 0.975])
    r_low, r_high = np.quantile(recall_boot, [0.025, 0.975])
    p_mean, r_mean = float(np.mean(precision)), float(np.mean(recall))
    f1 = 2 * p_mean * r_mean / (p_mean + r_mean) if p_mean + r_mean else 0.0
    return {
        "n_cases": len(rows),
        "n_cases_with_gold": len(recall),
        "abstentions": abstentions,
        "citation_source_precision": p_mean,
        "citation_source_precision_low": float(p_low),
        "citation_source_precision_high": float(p_high),
        "citation_source_recall": r_mean,
        "citation_source_recall_low": float(r_low),
        "citation_source_recall_high": float(r_high),
        "citation_source_f1": float(f1),
        "bootstrap_resamples": n_resamples,
    }
