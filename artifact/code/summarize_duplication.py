"""Fail-closed bootstrap summary for the duplication intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.summarize import (
    duplication_holds,
    summarize_duplication,
    summarize_duplication_multiseed,
    validate_duplication_protocol,
)
from run_duplication import (
    COPY_COUNTS,
    METHODS,
    PROJECT_ROOT,
    PROTOCOL_VERSION,
    SELECTION_SEED,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--expected-claims", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, action="append")
    parser.add_argument("--expected-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [(path, read_jsonl(path)) for path in args.input]
    records = [record for _, path_records in inputs for record in path_records]
    run = validate_duplication_protocol(
        records,
        expected_model=args.expected_model,
        expected_protocol=PROTOCOL_VERSION,
        expected_selection_seed=SELECTION_SEED,
    )
    if args.expected_seed:
        summary = summarize_duplication_multiseed(
            records,
            expected_claims=args.expected_claims,
            expected_seeds=tuple(args.expected_seed),
            methods=METHODS,
            copy_counts=COPY_COUNTS,
        )
    else:
        summary = summarize_duplication(
            records,
            expected_claims=args.expected_claims,
            methods=METHODS,
            copy_counts=COPY_COUNTS,
        )
    summary["diagnostic"] = duplication_holds(summary)
    summary["run"] = run
    summary["sources"] = [
        {"path": str(path), "sha256": _sha256(path), "records": len(path_records)}
        for path, path_records in inputs
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary["diagnostic"], sort_keys=True))


if __name__ == "__main__":
    main()
