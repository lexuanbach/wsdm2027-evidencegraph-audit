"""Evaluate active independent-source acquisition and its matched ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.controller import BehaviorClonedController
from evidencegraph.controller_evaluation import (
    ROLLOUT_METHODS,
    calibrate_rollouts,
    dependence_quartile_summary,
    rollout_dataset_split,
)
from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.episode_evaluation import (
    index_dependence_records,
    index_nli_records,
    summarize_episode_predictions,
)
from evidencegraph.episode_inference import fit_logistic_calibrator
from evidencegraph.nli_evaluation import validate_nli_dataset
from evidencegraph.provenance import sha256
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument("--nli", type=Path, action="append", required=True)
    parser.add_argument("--dependence", type=Path, required=True)
    parser.add_argument(
        "--controller",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/controller/controller_seed103.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/controller_evaluation",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    episodes = read_jsonl(args.dataset)
    nli_records = [record for path in args.nli for record in read_jsonl(path)]
    nli_metadata = validate_nli_dataset(nli_records, sha256(args.dataset))
    nli_index = index_nli_records(nli_records)
    dependence_index = index_dependence_records(read_jsonl(args.dependence))
    controller = BehaviorClonedController.load(args.controller)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "candidate-pool-active-seek-v1",
        "calibration_split": "dev",
        "evaluation_split": "test",
        "initial_sources": 5,
        "action_budget": 3,
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "nli": [source_descriptor(path, PROJECT_ROOT) for path in args.nli],
            "nli_run": nli_metadata,
            "dependence": source_descriptor(args.dependence, PROJECT_ROOT),
            "controller": source_descriptor(args.controller, PROJECT_ROOT),
        },
        "methods": {},
    }
    all_predictions = []
    active_test = []
    for method in ROLLOUT_METHODS:
        dev = rollout_dataset_split(
            episodes,
            nli_index,
            dependence_index,
            controller,
            split="dev",
            method=method,
        )
        test = rollout_dataset_split(
            episodes,
            nli_index,
            dependence_index,
            controller,
            split="test",
            method=method,
        )
        calibrator = fit_logistic_calibrator(
            [record["evidence_logit"] for record in dev],
            [record["label"] for record in dev],
        )
        dev = calibrate_rollouts(dev, calibrator)
        test = calibrate_rollouts(test, calibrator)
        all_predictions.extend(dev)
        all_predictions.extend(test)
        summary["methods"][method] = {
            "calibrator": calibrator.to_dict(),
            "dev": summarize_episode_predictions(
                dev, n_resamples=args.bootstrap_resamples, seed=103
            ),
            "test": summarize_episode_predictions(
                test, n_resamples=args.bootstrap_resamples, seed=103
            ),
        }
        if method == "evidencegraph_active":
            active_test = test
    summary["dependence_quartiles"] = dependence_quartile_summary(
        active_test, n_resamples=args.bootstrap_resamples, seed=103
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(all_predictions, predictions_path)
    summary["predictions"] = source_descriptor(predictions_path, PROJECT_ROOT)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {"methods": list(summary["methods"]), "predictions": len(all_predictions)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
