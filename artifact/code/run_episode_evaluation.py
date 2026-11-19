"""Join frozen edge outputs, calibrate on dev, and evaluate episode variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.episode_evaluation import (
    VARIANT_ARGUMENTS,
    index_dependence_records,
    index_nli_records,
    score_dataset_split_indexed,
    summarize_episode_predictions,
)
from evidencegraph.episode_inference import fit_logistic_calibrator
from evidencegraph.nli_evaluation import validate_nli_dataset
from evidencegraph.provenance import sha256


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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/episode_evaluation",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    episodes = read_jsonl(args.dataset)
    nli_records = [record for path in args.nli for record in read_jsonl(path)]
    dataset_sha256 = sha256(args.dataset)
    nli_metadata = validate_nli_dataset(nli_records, dataset_sha256)
    dependence_records = read_jsonl(args.dependence)
    nli_index = index_nli_records(nli_records)
    dependence_index = index_dependence_records(dependence_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    summary = {
        "protocol": "evidencegraph-episode-inference-v1",
        "calibration_split": "dev",
        "evaluation_split": "test",
        "inputs": {
            "dataset": {"path": str(args.dataset), "sha256": dataset_sha256},
            "nli": [
                {"path": str(path), "sha256": sha256(path)} for path in args.nli
            ],
            "nli_run": nli_metadata,
            "dependence": {
                "path": str(args.dependence),
                "sha256": sha256(args.dependence),
            },
        },
        "variants": {},
    }
    for variant in VARIANT_ARGUMENTS:
        dev = score_dataset_split_indexed(
            episodes,
            nli_index,
            dependence_index,
            split="dev",
            variant=variant,
        )
        test = score_dataset_split_indexed(
            episodes,
            nli_index,
            dependence_index,
            split="test",
            variant=variant,
        )
        calibrator = fit_logistic_calibrator(
            [record["evidence_logit"] for record in dev],
            [record["label"] for record in dev],
        )
        calibrated_dev = [
            {**record, **calibrated_record_from_dict(record, calibrator)} for record in dev
        ]
        calibrated_test = [
            {**record, **calibrated_record_from_dict(record, calibrator)} for record in test
        ]
        all_predictions.extend(calibrated_dev)
        all_predictions.extend(calibrated_test)
        summary["variants"][variant] = {
            "calibrator": calibrator.to_dict(),
            "dev": summarize_episode_predictions(
                calibrated_dev,
                n_resamples=args.bootstrap_resamples,
                seed=103,
            ),
            "test": summarize_episode_predictions(
                calibrated_test,
                n_resamples=args.bootstrap_resamples,
                seed=103,
            ),
        }
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(all_predictions, predictions_path)
    summary["predictions"] = {
        "path": str(predictions_path),
        "sha256": sha256(predictions_path),
        "records": len(all_predictions),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "variants": list(summary["variants"]),
                "predictions": len(all_predictions),
            },
            sort_keys=True,
        )
    )


def calibrated_record_from_dict(record, calibrator):
    """Use the canonical calibrator without reconstructing an EpisodeScore."""
    probability = float(calibrator.predict([record["evidence_logit"]])[0])
    return {"probability_true": probability}


if __name__ == "__main__":
    main()
