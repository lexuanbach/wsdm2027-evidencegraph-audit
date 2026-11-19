"""Build the frozen FTS5 index for the official AVeriTeC dev store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.averitec_store import create_averitec_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        type=Path,
        default=PROJECT_ROOT / "data/raw/averitec_dev_knowledge_store",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/averitec_dev.sqlite",
    )
    args = parser.parse_args()
    def report(record):
        print(json.dumps(record, sort_keys=True), flush=True)

    print(
        json.dumps(
            create_averitec_index(args.store, args.output, progress=report), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
