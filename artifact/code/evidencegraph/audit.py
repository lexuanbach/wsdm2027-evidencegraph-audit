"""Fail-closed validation for independent claim- and source-level audits."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


CLAIM_COLUMNS = ("claim_id", "rater_id", "gold_verdict", "timestamp_utc")
SOURCE_COLUMNS = (
    "claim_id",
    "source_id",
    "rater_id",
    "family_label",
    "timestamp_utc",
)
VERDICT_LABELS = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}
FAMILY_LABELS = {"independent", "mirror", "citation", "paraphrase"}
KAPPA_TARGET = 0.8


def meets_kappa_target(
    metrics: dict[str, float | int], target: float = KAPPA_TARGET
) -> bool:
    """Return whether both preregistered inter-rater agreement gates pass."""

    if not 0.0 <= target <= 1.0:
        raise ValueError("kappa target must lie in [0, 1]")
    if not isinstance(metrics, dict):
        return False
    return all(
        math.isfinite(float(metrics.get(key, float("nan"))))
        and float(metrics[key]) >= target
        for key in ("verdict_kappa", "family_kappa")
    )


def _read(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{path.name} columns must be exactly {columns}")
        rows = list(reader)
    if any(any(not row[column].strip() for column in columns) for row in rows):
        raise ValueError(f"{path.name} may not contain blank ratings")
    for row in rows:
        raw_timestamp = row["timestamp_utc"]
        if not raw_timestamp.endswith("Z"):
            raise ValueError(f"{path.name} timestamps must use UTC Z notation")
        try:
            timestamp = datetime.fromisoformat(raw_timestamp[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError(f"{path.name} contains an invalid timestamp") from error
        if timestamp.tzinfo != timezone.utc:
            raise ValueError(f"{path.name} timestamps must be UTC")
    return rows


def _paired_labels(
    rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
    label_field: str,
    expected_keys: set[tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected_keys:
        missing, extra = expected_keys - set(grouped), set(grouped) - expected_keys
        raise ValueError(f"audit IDs disagree: {len(missing)} missing, {len(extra)} extra")
    first, second = [], []
    for key, item_rows in grouped.items():
        if len(item_rows) != 2 or len({row["rater_id"] for row in item_rows}) != 2:
            raise ValueError(f"{key} must have exactly two distinct raters")
        ordered = sorted(item_rows, key=lambda row: row["rater_id"])
        first.append(ordered[0][label_field])
        second.append(ordered[1][label_field])
    return first, second


def validate_audit(
    claim_path: Path,
    source_path: Path,
    expected_claim_ids: set[str],
    expected_source_ids: set[tuple[str, str]],
) -> dict[str, float | int]:
    claim_rows = _read(claim_path, CLAIM_COLUMNS)
    source_rows = _read(source_path, SOURCE_COLUMNS)
    if any(row["gold_verdict"] not in VERDICT_LABELS for row in claim_rows):
        raise ValueError(f"claim labels must be exactly {sorted(VERDICT_LABELS)}")
    if any(row["family_label"] not in FAMILY_LABELS for row in source_rows):
        raise ValueError(f"family labels must be exactly {sorted(FAMILY_LABELS)}")
    claim_raters = {row["rater_id"] for row in claim_rows}
    source_raters = {row["rater_id"] for row in source_rows}
    if len(claim_raters) != 2 or claim_raters != source_raters:
        raise ValueError("audit requires the same two global raters for both tasks")
    claim_first, claim_second = _paired_labels(
        claim_rows,
        ("claim_id",),
        "gold_verdict",
        {(claim_id,) for claim_id in expected_claim_ids},
    )
    family_first, family_second = _paired_labels(
        source_rows,
        ("claim_id", "source_id"),
        "family_label",
        expected_source_ids,
    )
    verdict_kappa = float(cohen_kappa_score(claim_first, claim_second))
    family_kappa = float(cohen_kappa_score(family_first, family_second))
    if not math.isfinite(verdict_kappa) or not math.isfinite(family_kappa):
        raise ValueError("audit kappa is undefined for the submitted labels")
    return {
        "n_claims": len(expected_claim_ids),
        "n_sources": len(expected_source_ids),
        "n_claim_ratings": len(claim_rows),
        "n_source_ratings": len(source_rows),
        "rater_count": len(claim_raters),
        "verdict_raw_agreement": sum(
            left == right for left, right in zip(claim_first, claim_second)
        )
        / len(claim_first),
        "family_raw_agreement": sum(
            left == right for left, right in zip(family_first, family_second)
        )
        / len(family_first),
        "verdict_kappa": verdict_kappa,
        "family_kappa": family_kappa,
    }
