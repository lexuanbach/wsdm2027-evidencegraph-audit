"""Select same-stratum reserve candidates for persistent paraphrase failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.reserve_replacement import reserve_retry_selection


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
    )
    parser.add_argument(
        "--reserves",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_reserve.jsonl",
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--per-failure", type=int, default=2)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = reserve_retry_selection(
        read_jsonl(args.primary),
        read_jsonl(args.reserves),
        read_jsonl(args.validation),
        per_failure=args.per_failure,
        round_index=args.round,
    )
    write_jsonl(selected, args.output)
    print(json.dumps({"reserve_candidates": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
