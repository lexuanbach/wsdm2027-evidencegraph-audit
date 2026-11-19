"""Size same-stratum reserves from observed complete primary validation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from evidencegraph.dataset_builder import (
    choose_reserve_claims,
    read_jsonl,
    reserve_candidate_strata,
    write_jsonl,
)
from evidencegraph.reserve_replacement import selection_stratum
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def adaptive_targets(
    primary: list[dict], validation_records: list[dict], *, per_failure: int
) -> tuple[Counter[tuple[str, str, str]], dict[tuple[str, str, str], int]]:
    if per_failure < 1:
        raise ValueError("per_failure must be positive")
    validation: dict[str, dict] = {}
    for record in validation_records:
        claim_id = str(record["claim_id"])
        if claim_id in validation:
            raise ValueError(f"duplicate validation record for {claim_id}")
        if record.get("status") not in {"validated", "semantic_failure"}:
            raise ValueError(f"invalid validation status for {claim_id}")
        validation[claim_id] = record
    primary_ids = {str(record["claim_id"]) for record in primary}
    if len(primary_ids) != len(primary):
        raise ValueError("primary selection contains duplicate claim IDs")
    missing = primary_ids - set(validation)
    extra = set(validation) - primary_ids
    if missing or extra:
        raise ValueError(
            "adaptive reserve sizing requires exactly the complete primary validation"
        )
    failures: Counter[tuple[str, str, str]] = Counter(
        selection_stratum(record)
        for record in primary
        if validation[str(record["claim_id"])]["status"] == "semantic_failure"
    )
    targets = {key: count * per_failure for key, count in failures.items()}
    return failures, targets


def _render(counts: dict[tuple[str, str, str], int]) -> dict[str, int]:
    return {"/".join(key): int(value) for key, value in sorted(counts.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
    )
    parser.add_argument(
        "--fever",
        type=Path,
        default=PROJECT_ROOT / "data/raw/fever_train.jsonl",
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--per-failure", type=int, default=2)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evidencegraph20k_adaptive_reserve.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/evidencegraph20k_adaptive_reserve_audit.json",
    )
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("refusing to replace an adaptive reserve output or audit")

    primary = read_jsonl(args.primary)
    validation = read_jsonl(args.validation)
    failures, targets = adaptive_targets(
        primary, validation, per_failure=args.per_failure
    )
    strata = reserve_candidate_strata(args.fever, primary, seed=args.seed)
    capacities = {key: len(value) for key, value in strata.items()}
    insufficient = {
        key: {"target": target, "available": capacities.get(key, 0)}
        for key, target in targets.items()
        if target > capacities.get(key, 0)
    }
    if insufficient:
        raise ValueError(f"insufficient adaptive reserve capacity: {insufficient}")
    reserves = choose_reserve_claims(strata, targets, seed=args.seed)
    write_jsonl(reserves, args.output)
    audit = {
        "schema_version": 1,
        "protocol": "observed-failure-sized-same-stratum-reserves-v1",
        "seed": args.seed,
        "per_failure": args.per_failure,
        "primary_count": len(primary),
        "primary_failure_count": sum(failures.values()),
        "reserve_count": len(reserves),
        "failure_counts": _render(failures),
        "target_counts": _render(targets),
        "available_counts": _render(capacities),
        "sources": [
            source_descriptor(args.primary, PROJECT_ROOT),
            source_descriptor(args.validation, PROJECT_ROOT),
            source_descriptor(args.fever, PROJECT_ROOT),
        ],
        "output": source_descriptor(args.output, PROJECT_ROOT),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
