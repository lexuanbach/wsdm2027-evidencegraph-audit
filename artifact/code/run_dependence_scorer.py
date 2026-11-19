"""Train and evaluate the dependence estimator with three frozen seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.dependence_model import DependenceEstimator
from evidencegraph.dependence_features import DEPENDENCE_SCORER_PROTOCOL, FEATURE_NAMES
from evidencegraph.edge_evaluation import (
    clustered_metric_summary,
    dependence_examples,
    grouped_point_summary,
)
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_predictions_atomic(
    output: Path,
    seeds: tuple[int, ...],
    groups: list[
        tuple[str, np.ndarray, list[dict[str, str]], list[np.ndarray]]
    ],
) -> int:
    """Stream raw pair probabilities without retaining a second million-row list."""

    partial = output.with_name(f".{output.name}.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with partial.open("w", encoding="utf-8") as handle:
        for split, labels, metadata, predictions in groups:
            if len(labels) != len(metadata) or any(
                len(probability) != len(labels) for probability in predictions
            ):
                raise ValueError("dependence prediction rows do not align")
            if len(predictions) != len(seeds):
                raise ValueError("dependence predictions do not cover the frozen seeds")
            for index, record in enumerate(metadata):
                row = {
                    **record,
                    "split": split,
                    "label": int(labels[index]),
                    "probabilities": {
                        str(seed): float(probability[index])
                        for seed, probability in zip(seeds, predictions)
                    },
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    partial.replace(output)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/edge_scorers/dependence",
    )
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--expected-split-sizes",
        default='{"train": 14000, "dev": 3000, "test": 3000}',
        help=(
            "JSON object of exact train/dev/test claim counts the dataset must "
            "have; the default is the frozen EvidenceGraph-20K release contract. "
            "Subset runs must pass their own denominators explicitly and are "
            "recorded as such in the summary."
        ),
    )
    args = parser.parse_args()
    expected_split_sizes = {
        str(key): int(value)
        for key, value in json.loads(args.expected_split_sizes).items()
    }
    if set(expected_split_sizes) != {"train", "dev", "test"} or any(
        value < 1 for value in expected_split_sizes.values()
    ):
        raise ValueError("expected split sizes must give positive train/dev/test counts")
    seeds = tuple(args.seed or [103, 104, 105])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("dependence evaluation requires exactly three distinct seeds")
    episodes = read_jsonl(args.dataset)
    train_x, train_y, _ = dependence_examples(
        episodes, "train", include_metadata=False
    )
    dev_x, dev_y, dev_metadata = dependence_examples(episodes, "dev")
    test_x, test_y, test_metadata = dependence_examples(episodes, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[str, list[np.ndarray]] = {"dev": [], "test": []}
    seed_summaries = []
    model_paths = []
    for seed in seeds:
        estimator = DependenceEstimator(seed=seed).fit(train_x, train_y)
        model_path = args.output_dir / f"dependence_seed{seed}.joblib"
        estimator.save(model_path)
        model_paths.append(model_path)
        dev_probability = estimator.predict_proba(dev_x)
        test_probability = estimator.predict_proba(test_x)
        all_predictions["dev"].append(dev_probability)
        all_predictions["test"].append(test_probability)
        seed_summary = {
            "seed": seed,
            "dev": grouped_point_summary(dev_y, dev_probability, dev_metadata),
            "test": grouped_point_summary(test_y, test_probability, test_metadata),
        }
        seed_summaries.append(seed_summary)
    mean_dev = np.mean(np.vstack(all_predictions["dev"]), axis=0)
    mean_test = np.mean(np.vstack(all_predictions["test"]), axis=0)
    train_pairs = int(len(train_y))
    del train_x, train_y
    predictions_path = args.output_dir / "predictions.jsonl"
    written = write_predictions_atomic(
        predictions_path,
        seeds,
        [
            ("dev", dev_y, dev_metadata, all_predictions["dev"]),
            ("test", test_y, test_metadata, all_predictions["test"]),
        ],
    )
    if written != len(dev_y) + len(test_y):
        raise RuntimeError("dependence prediction writer lost rows")
    split_claims = {
        split: sum(1 for episode in episodes if episode["split"] == split)
        for split in ("train", "dev", "test")
    }
    if split_claims != expected_split_sizes:
        raise ValueError(
            f"unexpected dataset split sizes: {split_claims} "
            f"(expected {expected_split_sizes})"
        )
    summary = {
        "protocol": DEPENDENCE_SCORER_PROTOCOL,
        "expected_split_sizes": expected_split_sizes,
        "is_release_denominator": expected_split_sizes
        == {"train": 14000, "dev": 3000, "test": 3000},
        "feature_names": list(FEATURE_NAMES),
        "seeds": list(seeds),
        "train_pairs": train_pairs,
        "dev_pairs": int(len(dev_y)),
        "test_pairs": int(len(test_y)),
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "split_claims": split_claims,
            "predictions": source_descriptor(predictions_path, PROJECT_ROOT),
            "models": [source_descriptor(path, PROJECT_ROOT) for path in model_paths],
        },
        "seed_summaries": seed_summaries,
        "three_seed_mean": {
            "dev": clustered_metric_summary(
                dev_y,
                mean_dev,
                dev_metadata,
                n_resamples=args.bootstrap_resamples,
                seed=103,
            ),
            "test": clustered_metric_summary(
                test_y,
                mean_test,
                test_metadata,
                n_resamples=args.bootstrap_resamples,
                seed=103,
            ),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "train_pairs": train_pairs,
                "dev_pairs": len(dev_y),
                "test_pairs": len(test_y),
                "seeds": list(seeds),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
