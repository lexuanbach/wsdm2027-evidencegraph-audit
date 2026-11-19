"""Generate EvidenceGraph paraphrases locally with append-only resumption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.paraphrase_job import DEFAULT_PROMPT_PROFILE, PROMPT_PROFILES, generate


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--prompt-profile",
        choices=sorted(PROMPT_PROFILES),
        default=DEFAULT_PROMPT_PROFILE,
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifact/runs/paraphrases/raw.jsonl"
    )
    args = parser.parse_args()
    summary = generate(
        args.selection,
        PROJECT_ROOT / "data/processed/fever_wiki.sqlite",
        args.output,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        prompt_profile=args.prompt_profile,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
