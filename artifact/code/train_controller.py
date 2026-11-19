"""Train the fixed small controller from train-only construction demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.controller_training import train_and_evaluate_controller
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.provenance import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/controller/controller_seed103.joblib",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/controller/summary.json",
    )
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument(
        "--allow-missing-actions",
        action="store_true",
        help=(
            "Train on the actions the frozen teacher actually demonstrated and "
            "record any action it never chose, instead of failing closed."
        ),
    )
    args = parser.parse_args()
    controller, summary = train_and_evaluate_controller(
        read_jsonl(args.dataset),
        seed=args.seed,
        allow_missing_actions=args.allow_missing_actions,
    )
    controller.save(args.model_output)
    summary["inputs"] = {
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)}
    }
    summary["model"] = {
        "path": str(args.model_output),
        "sha256": sha256(args.model_output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
