"""Evaluate the reduced-scale API-backbone benchmark matrix honestly.

Each (dataset, method) cell is an append-only prediction file.  Cases that
have no completed attempt for *every* method in a dataset are excluded from
that dataset's comparison, and the exclusion counts are reported, so every
paired difference is computed on exactly the same claims.  Metrics: accuracy,
macro-F1 over the dataset label set, ECE on the maximum-probability confidence
(10 bins), Brier score over the one-hot label vector, and the false-support
rate (probability mass assigned to the support label wins on a non-support
claim).  Claim bootstrap (1,000 resamples) gives 95% intervals and paired
differences against ``fixed_rag``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidencegraph.benchmark_answerer import BENCHMARK_LABELS, SUPPORT_LABEL
from evidencegraph.benchmark_evaluation import resolve_benchmark_attempts
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "haiku-subset-matrix-evaluation-v1"
DATASET_NAMES = {"fever_dev": "FEVER", "hover_dev": "HoVer", "averitec_dev": "AVeriTeC"}


def _metrics(
    gold: np.ndarray, pred: np.ndarray, probs: np.ndarray, labels: tuple[str, ...], support_index: int
) -> dict[str, float]:
    n = len(gold)
    correct = pred == gold
    f1s = []
    for index in range(len(labels)):
        tp = np.sum((pred == index) & (gold == index))
        fp = np.sum((pred == index) & (gold != index))
        fn = np.sum((pred != index) & (gold == index))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    confidence = np.max(probs, axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        mask = bins == b
        if np.any(mask):
            ece += np.sum(mask) * abs(np.mean(confidence[mask]) - np.mean(correct[mask]))
    one_hot = np.eye(len(labels))[gold]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    non_support = gold != support_index
    false_support = (
        float(np.mean(pred[non_support] == support_index)) if np.any(non_support) else float("nan")
    )
    return {
        "accuracy": float(np.mean(correct)),
        "macro_f1": float(np.mean(f1s)),
        "ece": float(ece / n),
        "brier": brier,
        "false_support_rate": false_support,
    }


def evaluate_dataset(
    dataset: str,
    cases: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    n_resamples: int,
    seed: int,
    reference: str = "fixed_rag",
) -> dict[str, Any]:
    name = DATASET_NAMES[dataset]
    labels = BENCHMARK_LABELS[name]
    label_index = {label: index for index, label in enumerate(labels)}
    support_index = label_index[SUPPORT_LABEL[name]]
    resolved = {}
    for method, rows in predictions.items():
        # Cases whose every attempt failed are dropped and reported in
        # ``excluded_cases`` rather than crashing the summary; cases with a
        # completion still go through the strict single-identity resolver.
        by_case: dict[str, list[dict[str, Any]]] = {}
        for record in rows:
            by_case.setdefault(str(record["case_id"]), []).append(record)
        attempted = [
            record
            for attempts in by_case.values()
            if any(a.get("status") == "complete" for a in attempts)
            for record in attempts
        ]
        complete = {}
        if attempted:
            for record in resolve_benchmark_attempts(attempted):
                complete[str(record["case_id"])] = record
        resolved[method] = complete
    case_ids = [str(c["case_id"]) for c in cases]
    shared = [cid for cid in case_ids if all(cid in resolved[m] for m in resolved)]
    gold_by_case = {str(c["case_id"]): str(c["gold_label"]) for c in cases}
    output: dict[str, Any] = {
        "dataset": name,
        "labels": list(labels),
        "n_cases_in_subset": len(case_ids),
        "n_cases_compared": len(shared),
        "excluded_case_counts": {
            method: len(set(case_ids) - set(resolved[method])) for method in resolved
        },
        "excluded_cases": {
            method: sorted(set(case_ids) - set(resolved[method])) for method in resolved
        },
        "methods": {},
        "paired_versus_reference": {},
        "reference": reference,
    }
    gold = np.asarray([label_index[gold_by_case[cid]] for cid in shared])
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, complete in resolved.items():
        pred = np.asarray([label_index[str(complete[cid]["label"])] for cid in shared])
        probs = np.asarray(
            [[float(complete[cid]["probabilities"][label]) for label in labels] for cid in shared]
        )
        probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)
        arrays[method] = (pred, probs)
        point = _metrics(gold, pred, probs, labels, support_index)
        rng = np.random.default_rng(seed)
        boots: dict[str, list[float]] = {k: [] for k in point}
        for _ in range(n_resamples):
            sample = rng.integers(0, len(shared), len(shared))
            for key, value in _metrics(gold[sample], pred[sample], probs[sample], labels, support_index).items():
                if np.isfinite(value):
                    boots[key].append(value)
        entry: dict[str, Any] = dict(point)
        for key, values in boots.items():
            if values:
                low, high = np.quantile(np.asarray(values), [0.025, 0.975])
                entry[f"{key}_low"], entry[f"{key}_high"] = float(low), float(high)
        entry["mean_prompt_tokens"] = float(np.mean([complete[cid].get("prompt_tokens", 0) for cid in shared]))
        entry["mean_completion_tokens"] = float(np.mean([complete[cid].get("completion_tokens", 0) for cid in shared]))
        entry["mean_documents_opened"] = float(np.mean([complete[cid].get("documents_opened", 0) for cid in shared]))
        entry["mean_wall_seconds"] = float(np.mean([complete[cid].get("wall_seconds", 0.0) for cid in shared]))
        entry["abstention_rate"] = float(np.mean([bool(complete[cid].get("abstained", False)) for cid in shared]))
        entry["repair_rate"] = float(np.mean([int(complete[cid].get("repair_attempts", 0)) > 0 for cid in shared]))
        output["methods"][method] = entry
    if reference in arrays:
        ref_pred, ref_probs = arrays[reference]
        for method, (pred, probs) in arrays.items():
            if method == reference:
                continue
            point = {
                k: _metrics(gold, pred, probs, labels, support_index)[k]
                - _metrics(gold, ref_pred, ref_probs, labels, support_index)[k]
                for k in ("accuracy", "macro_f1", "ece", "brier", "false_support_rate")
            }
            rng = np.random.default_rng(seed + 1)
            boots = {k: [] for k in point}
            for _ in range(n_resamples):
                sample = rng.integers(0, len(shared), len(shared))
                a = _metrics(gold[sample], pred[sample], probs[sample], labels, support_index)
                b = _metrics(gold[sample], ref_pred[sample], ref_probs[sample], labels, support_index)
                for k in boots:
                    if np.isfinite(a[k]) and np.isfinite(b[k]):
                        boots[k].append(a[k] - b[k])
            entry = {}
            for k, value in point.items():
                if boots[k]:
                    low, high = np.quantile(np.asarray(boots[k]), [0.025, 0.975])
                    entry[k] = {"difference": float(value), "low": float(low), "high": float(high)}
            output["paired_versus_reference"][method] = entry
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=PROJECT_ROOT / "data/benchmarks/subset")
    parser.add_argument(
        "--predictions-dir", type=Path, default=PROJECT_ROOT / "artifact/runs/benchmarks/haiku_subset"
    )
    parser.add_argument("--model-tag", default="haiku45")
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--reference",
        default="fixed_rag",
        help="Method that the paired difference rows are computed against.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = args.dataset or ["fever_dev", "hover_dev", "averitec_dev"]
    methods = args.method or ["closed_book", "fixed_rag", "evidencegraph"]
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "model_tag": args.model_tag,
        "reference": args.reference,
        "bootstrap_resamples": args.bootstrap_resamples,
        "inputs": {},
        "datasets": {},
    }
    for dataset in datasets:
        cases_path = args.cases_dir / f"{dataset}_n500.jsonl"
        cases = read_jsonl(cases_path)
        predictions = {}
        inputs = {"cases": source_descriptor(cases_path, PROJECT_ROOT)}
        for method in methods:
            path = args.predictions_dir / f"{dataset}_{method}_{args.model_tag}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            if not rows:
                continue
            predictions[method] = rows
            inputs[method] = source_descriptor(path, PROJECT_ROOT)
        if not predictions:
            continue
        summary["inputs"][dataset] = inputs
        summary["datasets"][dataset] = evaluate_dataset(
            dataset,
            cases,
            predictions,
            n_resamples=args.bootstrap_resamples,
            seed=103,
            reference=args.reference,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for dataset, block in summary["datasets"].items():
        print(dataset, "n_compared", block["n_cases_compared"])
        for method, entry in block["methods"].items():
            print(
                f"  {method:14s} acc {entry['accuracy']:.3f} [{entry.get('accuracy_low', float('nan')):.3f},{entry.get('accuracy_high', float('nan')):.3f}]"
                f" f1 {entry['macro_f1']:.3f} ece {entry['ece']:.3f} fsr {entry['false_support_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
