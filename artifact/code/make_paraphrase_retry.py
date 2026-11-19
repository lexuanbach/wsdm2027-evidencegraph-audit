"""Create a deterministic selection subset for semantic-generation retries."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.paraphrase_job import validate_generated_record_shape


def excluded_claim_ids(paths: list[Path]) -> set[str]:
    return {
        str(record["claim_id"])
        for path in paths
        for record in read_jsonl(path)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help="exclude claims already attempted in a separate retry tier",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = read_jsonl(args.validation)
    validated_ids = {
        str(record["claim_id"])
        for record in validation
        if record.get("status") == "validated"
    }
    failed_ids = {
        str(record["claim_id"])
        for record in validation
        if record.get("status") == "semantic_failure"
    }
    if args.raw is not None:
        generated_ids: set[str] = set()
        schema_failure_ids: set[str] = set()
        invalid_generated_ids: set[str] = set()
        for record in read_jsonl(args.raw):
            if record.get("status") == "schema_failure":
                schema_failure_ids.update(
                    str(value) for value in record.get("claim_ids", [])
                )
            elif record.get("status") == "generated":
                claim_id = str(record["claim_id"])
                try:
                    validate_generated_record_shape(record)
                except ValueError:
                    invalid_generated_ids.add(claim_id)
                else:
                    generated_ids.add(claim_id)
        failed_ids.update(schema_failure_ids - generated_ids)
        failed_ids.update(invalid_generated_ids)
    failed_ids -= validated_ids
    failed_ids -= excluded_claim_ids(args.exclude_selection)
    records = [
        record for record in read_jsonl(args.selection) if record["claim_id"] in failed_ids
    ]
    if len(records) != len(failed_ids):
        raise ValueError("some semantic-failure IDs are absent from the selection")
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} retry claims to {args.output}")


if __name__ == "__main__":
    main()
