"""Apply lexical and bidirectional-NLI faithfulness gates to generated rewrites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.paraphrase_job import validate_cache
from evidencegraph.semantic_validation import (
    PARAPHRASE_NLI_MODEL,
    PARAPHRASE_NLI_PROTOCOL,
    PARAPHRASE_NLI_REVISION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=PROJECT_ROOT / "artifact/runs/paraphrases/raw.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/paraphrases/validated.jsonl",
    )
    parser.add_argument("--model", default=PARAPHRASE_NLI_MODEL)
    parser.add_argument("--revision", default=PARAPHRASE_NLI_REVISION)
    parser.add_argument("--protocol", default=PARAPHRASE_NLI_PROTOCOL)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--record-batch-size", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    args = parser.parse_args()
    summary = validate_cache(
        args.raw,
        args.output,
        model_name=args.model,
        model_revision=args.revision,
        validator_protocol=args.protocol,
        threshold=args.threshold,
        device=args.device,
        limit=args.limit,
        record_batch_size=args.record_batch_size,
        pair_batch_size=args.pair_batch_size,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
