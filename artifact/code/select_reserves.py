"""Create same-stratum replacement candidates for persistent generation failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evidencegraph.dataset_builder import (
    read_jsonl,
    select_reserve_claims,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_reserve.jsonl",
    )
    args = parser.parse_args()
    selected = read_jsonl(
        PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl"
    )
    reserves = select_reserve_claims(
        PROJECT_ROOT / "data/raw/fever_train.jsonl",
        selected,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    write_jsonl(reserves, args.output)
    counts = Counter(
        f"{record['split']}:{record['fever_label']}:{record['corruption']}"
        for record in reserves
    )
    print(json.dumps({"reserves": len(reserves), "strata": dict(counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
