"""Sample aligned selection and validation rows for a disjoint protocol gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl, stratified_sample, write_jsonl


def sample_aligned_rows(
    selection: list[dict],
    validation: list[dict],
    *,
    excluded_ids: set[str],
    limit: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    by_id: dict[str, dict] = {}
    for record in validation:
        claim_id = str(record["claim_id"])
        if claim_id in by_id:
            raise ValueError(f"duplicate validation claim ID: {claim_id}")
        by_id[claim_id] = record
    eligible = [
        record
        for record in selection
        if str(record["claim_id"]) in by_id
        and str(record["claim_id"]) not in excluded_ids
    ]
    sampled_selection = stratified_sample(eligible, limit, seed=seed)
    sampled_validation = [by_id[str(record["claim_id"])] for record in sampled_selection]
    return sampled_selection, sampled_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    args = parser.parse_args()
    excluded_ids = {
        str(record["claim_id"])
        for path in args.exclude
        for record in read_jsonl(path)
    }
    sampled_selection, sampled_validation = sample_aligned_rows(
        read_jsonl(args.selection),
        read_jsonl(args.validation),
        excluded_ids=excluded_ids,
        limit=args.limit,
        seed=args.seed,
    )
    write_jsonl(sampled_selection, args.output_selection)
    write_jsonl(sampled_validation, args.output_validation)
    strata = {
        (record["split"], record["fever_label"], record["corruption"])
        for record in sampled_selection
    }
    print(json.dumps({"episodes": len(sampled_selection), "strata": len(strata)}))


if __name__ == "__main__":
    main()
