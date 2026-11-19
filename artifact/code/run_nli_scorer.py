"""Append-only source-to-claim NLI scoring on a materialized dataset split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.nli import NLIVerifier
from evidencegraph.nli_evaluation import (
    SCORING_PROTOCOL,
    gold_nli_label,
    nli_input_sha256,
    validate_nli_resume,
)
from evidencegraph.provenance import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--record-batch-size", type=int, default=64)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.record_batch_size < 1 or args.model_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    dataset_sha256 = sha256(args.dataset)
    episodes = read_jsonl(args.dataset)
    expected = {}
    source_lookup = {}
    for episode in episodes:
        if episode["split"] != args.split:
            continue
        for source in episode["sources"]:
            key = (str(episode["claim_id"]), str(source["source_id"]))
            if key in expected:
                raise ValueError(f"duplicate dataset NLI source key: {key}")
            expected[key] = (
                str(source["family"]),
                gold_nli_label(float(source["stance"])),
                nli_input_sha256(str(episode["claim"]), str(source["text"])),
            )
            source_lookup[key] = (episode, source)
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = validate_nli_resume(
        existing,
        expected,
        split=args.split,
        model=args.model,
        revision=args.revision,
        dataset_sha256=dataset_sha256,
    )
    pending = [source_lookup[key] for key in expected if key not in completed]
    if not pending:
        print(json.dumps({"complete": len(completed), "new": 0}, sort_keys=True))
        return
    verifier = NLIVerifier(args.model, revision=args.revision, device=args.device)
    parameter_count = sum(parameter.numel() for parameter in verifier.model.parameters())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.record_batch_size):
            batch = pending[offset : offset + args.record_batch_size]
            started = time.monotonic()
            scores = verifier.score(
                [str(source["text"]) for _, source in batch],
                [str(episode["claim"]) for episode, _ in batch],
                batch_size=args.model_batch_size,
            )
            wall_seconds = time.monotonic() - started
            for (episode, source), score in zip(batch, scores):
                probabilities = {
                    "support": score.support,
                    "neutral": score.neutral,
                    "refute": score.refute,
                }
                output = {
                    "claim_id": str(episode["claim_id"]),
                    "source_id": str(source["source_id"]),
                    "split": args.split,
                    "family": str(source["family"]),
                    "gold_label": gold_nli_label(float(source["stance"])),
                    "probabilities": probabilities,
                    "prediction": max(probabilities, key=probabilities.get),
                    "model": args.model,
                    "model_revision": args.revision,
                    "model_parameters": parameter_count,
                    "scoring_protocol": SCORING_PROTOCOL,
                    "dataset_sha256": dataset_sha256,
                    "input_sha256": nli_input_sha256(
                        str(episode["claim"]), str(source["text"])
                    ),
                    "batch_wall_seconds": wall_seconds,
                    "status": "complete",
                }
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                written += 1
    print(json.dumps({"complete": len(completed) + written, "new": written}, sort_keys=True))


if __name__ == "__main__":
    main()
