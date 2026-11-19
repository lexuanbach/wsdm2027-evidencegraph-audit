"""Generate constrained local round-trip paraphrases for persistent failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.backtranslation_job import generate_backtranslations


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    summary = generate_backtranslations(
        args.selection,
        PROJECT_ROOT / "data/processed/fever_wiki.sqlite",
        args.output,
        device=args.device,
        limit=args.limit,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
