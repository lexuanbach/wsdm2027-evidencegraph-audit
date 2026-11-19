"""Fail-closed summary for the large and small NLI edge scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.nli_evaluation import (
    nli_run_metadata,
    nli_input_sha256,
    summarize_nli_records,
    validate_nli_resume,
)
from evidencegraph.provenance import sha256
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="TAG=JSONL")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-test-claims",
        type=int,
        default=3000,
        help=(
            "Exact number of test claims the dataset must contain (release "
            "contract: 3,000). Subset runs pass their own denominator explicitly."
        ),
    )
    args = parser.parse_args()
    dataset_sha256 = sha256(args.dataset)
    episodes = read_jsonl(args.dataset)
    split_episodes = [episode for episode in episodes if episode["split"] == args.split]
    if args.split == "test" and len(split_episodes) != args.expected_test_claims:
        raise ValueError(
            f"test NLI summary requires exactly {args.expected_test_claims:,} claims, "
            f"found {len(split_episodes):,}"
        )
    expected_records = {}
    for episode in split_episodes:
        for source in episode["sources"]:
            key = (str(episode["claim_id"]), str(source["source_id"]))
            if key in expected_records:
                raise ValueError(f"duplicate dataset NLI source key: {key}")
            stance = float(source["stance"])
            label = "support" if stance > 0.0 else "refute" if stance < 0.0 else "neutral"
            expected_records[key] = (
                str(source["family"]),
                label,
                nli_input_sha256(str(episode["claim"]), str(source["text"])),
            )
    expected = len(expected_records)
    summaries = {}
    raw_inputs = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError("--run must be TAG=JSONL")
        tag, raw_path = spec.split("=", 1)
        if not tag or tag in summaries:
            raise ValueError(f"duplicate or empty NLI run tag: {tag!r}")
        path = Path(raw_path)
        records = read_jsonl(path)
        if len(records) != expected:
            raise ValueError(f"incomplete NLI run {tag}: {len(records)} of {expected}")
        metadata = nli_run_metadata(records)
        validate_nli_resume(
            records,
            expected_records,
            split=args.split,
            model=metadata["model"],
            revision=metadata["model_revision"],
            dataset_sha256=dataset_sha256,
        )
        summaries[tag] = summarize_nli_records(
            records,
            n_resamples=args.bootstrap_resamples,
            seed=103,
        )
        raw_inputs[tag] = {
            **source_descriptor(path, PROJECT_ROOT),
            "model": metadata,
        }
    output = {
        "protocol": "nli-source-edge-v2-content-bound",
        "split": args.split,
        "n_claims": len(split_episodes),
        "expected_test_claims": args.expected_test_claims,
        "is_release_denominator": args.expected_test_claims == 3000,
        "expected_sources": expected,
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "runs": raw_inputs,
        },
        "models": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"models": sorted(summaries), "sources": expected}, sort_keys=True))


if __name__ == "__main__":
    main()
