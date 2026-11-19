from __future__ import annotations

import csv

import pytest

from evidencegraph.audit import (
    CLAIM_COLUMNS,
    SOURCE_COLUMNS,
    meets_kappa_target,
    validate_audit,
)


def test_kappa_target_requires_both_audit_tasks() -> None:
    assert meets_kappa_target({"verdict_kappa": 0.8, "family_kappa": 0.9})
    assert not meets_kappa_target({"verdict_kappa": 0.79, "family_kappa": 0.9})
    assert not meets_kappa_target({"verdict_kappa": 0.9, "family_kappa": 0.79})
    assert not meets_kappa_target({"verdict_kappa": 0.9})


def _write(path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_complete_two_rater_audit(tmp_path) -> None:
    claims, sources = tmp_path / "claims.csv", tmp_path / "sources.csv"
    _write(
        claims,
        CLAIM_COLUMNS,
        [
            ["c1", "a", "SUPPORTED", "2026-01-01T00:00:00Z"],
            ["c1", "b", "SUPPORTED", "2026-01-01T00:01:00Z"],
            ["c2", "a", "REFUTED", "2026-01-01T00:02:00Z"],
            ["c2", "b", "REFUTED", "2026-01-01T00:03:00Z"],
        ],
    )
    _write(
        sources,
        SOURCE_COLUMNS,
        [
            ["c1", "s1", "a", "mirror", "2026-01-01T00:00:00Z"],
            ["c1", "s1", "b", "mirror", "2026-01-01T00:01:00Z"],
            ["c2", "s2", "a", "independent", "2026-01-01T00:02:00Z"],
            ["c2", "s2", "b", "independent", "2026-01-01T00:03:00Z"],
        ],
    )
    result = validate_audit(
        claims, sources, {"c1", "c2"}, {("c1", "s1"), ("c2", "s2")}
    )
    assert result["verdict_kappa"] == 1.0
    assert result["family_kappa"] == 1.0
    assert result["verdict_raw_agreement"] == 1.0
    assert result["family_raw_agreement"] == 1.0
    assert result["rater_count"] == 2


def test_audit_rejects_invalid_labels_or_more_than_two_global_raters(tmp_path) -> None:
    claims, sources = tmp_path / "claims.csv", tmp_path / "sources.csv"
    _write(
        claims,
        CLAIM_COLUMNS,
        [
            ["c1", "a", "TRUE", "2026-01-01T00:00:00Z"],
            ["c1", "b", "TRUE", "2026-01-01T00:01:00Z"],
        ],
    )
    _write(
        sources,
        SOURCE_COLUMNS,
        [
            ["c1", "s1", "a", "mirror", "2026-01-01T00:00:00Z"],
            ["c1", "s1", "b", "mirror", "2026-01-01T00:01:00Z"],
        ],
    )
    with pytest.raises(ValueError, match="claim labels"):
        validate_audit(claims, sources, {"c1"}, {("c1", "s1")})
