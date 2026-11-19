"""Deterministic same-stratum replacement for rejected benchmark claims."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .dataset_builder import selection_statistics, stable_hash


Stratum = tuple[str, str, str]


def selection_stratum(record: dict[str, Any]) -> Stratum:
    return (
        str(record["split"]),
        str(record["fever_label"]),
        str(record["corruption"]),
    )


def _validation_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        claim_id = str(record["claim_id"])
        if claim_id in index:
            raise ValueError(f"duplicate validation record for {claim_id}")
        if record.get("status") not in {"validated", "semantic_failure"}:
            raise ValueError(f"invalid validation status for {claim_id}")
        index[claim_id] = record
    return index


def _require_primary_validation(
    primary: list[dict[str, Any]], validation: dict[str, dict[str, Any]]
) -> None:
    missing = [str(record["claim_id"]) for record in primary if str(record["claim_id"]) not in validation]
    if missing:
        raise ValueError(
            f"incomplete primary validation: {len(missing)} claims are missing"
        )


def reserve_retry_selection(
    primary: list[dict[str, Any]],
    reserves: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    per_failure: int = 2,
    round_index: int = 0,
) -> list[dict[str, Any]]:
    """Choose a deterministic reserve window for each failed primary stratum."""
    if per_failure < 1 or round_index < 0:
        raise ValueError("per_failure must be positive and round_index non-negative")
    validation = _validation_index(validation_records)
    _require_primary_validation(primary, validation)
    failures = Counter(
        selection_stratum(record)
        for record in primary
        if validation[str(record["claim_id"])]["status"] != "validated"
    )
    reserve_by_stratum: dict[Stratum, list[dict[str, Any]]] = defaultdict(list)
    for record in reserves:
        reserve_by_stratum[selection_stratum(record)].append(record)
    selected: list[dict[str, Any]] = []
    for stratum, failure_count in sorted(failures.items()):
        pool = sorted(
            reserve_by_stratum[stratum],
            key=lambda record: stable_hash(str(record["claim_id"])),
        )
        window = failure_count * per_failure
        start = round_index * window
        stop = start + window
        if stop > len(pool):
            raise ValueError(
                f"insufficient reserves for {stratum}: need indices {start}:{stop}, "
                f"have {len(pool)}"
            )
        selected.extend(pool[start:stop])
    return selected


def finalize_selection(
    primary: list[dict[str, Any]],
    reserves: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    eligible_claim_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replace semantically failed or provenance-ineligible primary claims."""
    validation = _validation_index(validation_records)
    _require_primary_validation(primary, validation)
    primary_ids = {str(record["claim_id"]) for record in primary}
    reserve_ids = {str(record["claim_id"]) for record in reserves}
    if primary_ids & reserve_ids:
        raise ValueError("primary and reserve claim IDs overlap")

    accepted_reserves: dict[Stratum, list[dict[str, Any]]] = defaultdict(list)
    for record in reserves:
        claim_id = str(record["claim_id"])
        result = validation.get(claim_id)
        coverage_ok = eligible_claim_ids is None or claim_id in eligible_claim_ids
        if result is not None and result["status"] == "validated" and coverage_ok:
            accepted_reserves[selection_stratum(record)].append(record)
    for stratum in accepted_reserves:
        accepted_reserves[stratum].sort(
            key=lambda record: stable_hash(str(record["claim_id"]))
        )

    final: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    offsets: Counter[Stratum] = Counter()
    for record in primary:
        claim_id = str(record["claim_id"])
        coverage_ok = eligible_claim_ids is None or claim_id in eligible_claim_ids
        if validation[claim_id]["status"] == "validated" and coverage_ok:
            final.append(record)
            continue
        stratum = selection_stratum(record)
        position = offsets[stratum]
        pool = accepted_reserves[stratum]
        if position >= len(pool):
            raise ValueError(f"no validated reserve remains for {stratum}")
        replacement = pool[position]
        offsets[stratum] += 1
        final.append(replacement)
        replacements.append(
            {
                "primary_claim_id": claim_id,
                "reserve_claim_id": str(replacement["claim_id"]),
                "split": stratum[0],
                "fever_label": stratum[1],
                "corruption": stratum[2],
            }
        )

    if len({str(record["claim_id"]) for record in final}) != len(final):
        raise ValueError("final selection contains duplicate claim IDs")
    before = Counter(selection_stratum(record) for record in primary)
    after = Counter(selection_stratum(record) for record in final)
    if before != after:
        raise ValueError("same-stratum replacement changed dataset composition")
    statistics = selection_statistics(final)
    if any(statistics["topic_leakage"].values()):
        raise ValueError("replacement introduced topic leakage")
    statistics["replacement_count"] = len(replacements)
    statistics["semantic_failure_replacement_count"] = sum(
        validation[str(record["claim_id"])]["status"] == "semantic_failure"
        for record in primary
    )
    statistics["provenance_coverage_replacement_count"] = sum(
        eligible_claim_ids is not None
        and str(record["claim_id"]) not in eligible_claim_ids
        for record in primary
    )
    statistics["original_episode_count"] = len(primary)
    return final, replacements, statistics
