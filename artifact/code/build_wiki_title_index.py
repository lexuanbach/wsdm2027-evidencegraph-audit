"""Build and audit the compact FEVER Wikipedia page-title FTS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.wiki_title_index import create_title_index, write_title_index_audit


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-database",
        type=Path,
        default=PROJECT_ROOT / "data/processed/fever_wiki.sqlite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/fever_wiki_titles.sqlite",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/wiki_title_index_audit.json",
    )
    args = parser.parse_args()
    summary = create_title_index(args.wiki_database, args.output)
    write_title_index_audit(summary, args.audit_output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
