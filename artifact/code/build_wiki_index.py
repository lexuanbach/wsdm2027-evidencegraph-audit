"""Build the immutable FEVER Wikipedia FTS5 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.wiki_index import create_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-dir", type=Path, default=PROJECT_ROOT / "data/raw/fever_wiki"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/processed/fever_wiki.sqlite"
    )
    args = parser.parse_args()
    counts = create_index(args.wiki_dir, args.output)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
