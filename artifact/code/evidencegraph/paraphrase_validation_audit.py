"""Descriptive, non-promotable audit of a paraphrase-validation tranche."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .dataset_builder import read_jsonl
from .result_registry import source_descriptor


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 6),
        "p10": round(_quantile(values, 0.10), 6),
        "median": round(statistics.median(values), 6),
        "p90": round(_quantile(values, 0.90), 6),
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _length_bucket(words: int) -> str:
    for upper in (20, 30, 40, 60):
        if words <= upper:
            return f"<= {upper}"
    return "> 60"


def _reason_category(reason: str) -> str:
    """Collapse instance-specific reasons into stable audit categories."""
    if ":missing:" in reason:
        return "lexical_anchor_missing"
    structural_categories = {
        "missing_content_lemma:": "structural_content_lemma_missing",
        "missing_phonetic_span:": "structural_phonetic_span_missing",
        "missing_relation_marker:": "structural_relation_marker_missing",
        "changed_semantic_role:": "structural_semantic_role_changed",
        "changed_temporal_attachment:": "structural_temporal_attachment_changed",
        "changed_aspect_modal_marker:": "structural_aspect_modal_changed",
        "empty_or_identical": "structural_empty_or_identical",
        "segment_count_mismatch": "structural_segment_count_mismatch",
    }
    for marker, category in structural_categories.items():
        if marker in reason:
            return category
    return reason


def audit_validation_tranche(
    selection_path: Path,
    validation_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    selection: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(selection_path):
        claim_id = str(record["claim_id"])
        if claim_id in selection:
            raise ValueError(f"duplicate selection claim ID: {claim_id}")
        selection[claim_id] = record

    rows = read_jsonl(validation_path)
    seen: set[str] = set()
    validator_models: set[str] = set()
    validator_revisions: set[str] = set()
    validator_protocols: set[str] = set()
    thresholds: set[float] = set()
    for record in rows:
        claim_id = str(record["claim_id"])
        if claim_id in seen:
            raise ValueError(f"duplicate validation claim ID: {claim_id}")
        if claim_id not in selection:
            raise ValueError(f"validation contains unselected claim ID: {claim_id}")
        if record.get("status") not in {"validated", "semantic_failure"}:
            raise ValueError(f"invalid validation status for {claim_id}")
        seen.add(claim_id)
        validator_models.add(str(record.get("validator_model", "")).strip())
        validator_revisions.add(str(record.get("validator_revision", "")).strip())
        validator_protocols.add(str(record.get("validator_protocol", "")).strip())
        thresholds.add(float(record.get("threshold", -1.0)))
    if not rows:
        raise ValueError("validation tranche is empty")
    if (
        len(validator_models) != 1
        or len(validator_revisions) != 1
        or len(validator_protocols) != 1
        or len(thresholds) != 1
        or not next(iter(validator_models))
        or not next(iter(validator_revisions))
        or not next(iter(validator_protocols))
    ):
        raise ValueError(
            "validation tranche mixes or omits validator identity or threshold"
        )

    def grouped(
        key: Callable[[dict[str, Any], dict[str, Any]], Any]
    ) -> dict[str, dict[str, float | int]]:
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for record in rows:
            source = selection[str(record["claim_id"])]
            group = str(key(record, source))
            counts[group][0] += int(record["status"] == "validated")
            counts[group][1] += 1
        return {
            group: {
                "passed": passed,
                "total": total,
                "yield": round(passed / total, 6),
            }
            for group, (passed, total) in sorted(counts.items())
        }

    reason_categories: Counter[str] = Counter()
    exact_reasons: Counter[str] = Counter()
    accepted_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    for record in rows:
        for candidate in record.get("candidate_results", []):
            target = accepted_candidates if candidate.get("accepted") else rejected_candidates
            target.append(candidate)
            if candidate.get("accepted"):
                continue
            categories: set[str] = set()
            reasons = [
                *candidate.get("reasons", []),
                *candidate.get("structural_reasons", []),
            ]
            for reason in reasons:
                exact_reasons[str(reason)] += 1
                categories.add(_reason_category(str(reason)))
            reason_categories.update(categories)

    def candidate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "forward_entailment": _distribution(
                [float(candidate["forward_entailment"]) for candidate in candidates]
            ),
            "reverse_entailment": _distribution(
                [float(candidate["reverse_entailment"]) for candidate in candidates]
            ),
            "word_count": _distribution(
                [float(_word_count(str(candidate["candidate"]))) for candidate in candidates]
            ),
        }

    passed = sum(record["status"] == "validated" for record in rows)
    return {
        "schema_version": 1,
        "promotable": False,
        "scope": "partial_validation_tranche",
        "sources": [
            source_descriptor(selection_path, project_root),
            source_descriptor(validation_path, project_root),
        ],
        "validator_model": next(iter(validator_models)),
        "validator_revision": next(iter(validator_revisions)),
        "validator_protocol": next(iter(validator_protocols)),
        "threshold": next(iter(thresholds)),
        "selection_count": len(selection),
        "validated_record_count": len(rows),
        "complete_selection": len(rows) == len(selection),
        "overall": {
            "passed": passed,
            "failed": len(rows) - passed,
            "total": len(rows),
            "yield": round(passed / len(rows), 6) if rows else 0.0,
        },
        "yield_by_corruption": grouped(lambda _record, source: source["corruption"]),
        "yield_by_split": grouped(lambda _record, source: source["split"]),
        "yield_by_label": grouped(lambda _record, source: source["label"]),
        "yield_by_required_validated": grouped(
            lambda record, _source: record["required_validated"]
        ),
        "yield_by_original_word_count": grouped(
            lambda record, _source: _length_bucket(_word_count(str(record["original"])))
        ),
        "yield_by_segment_count": grouped(
            lambda record, _source: len(str(record["original"]).splitlines())
        ),
        "original_word_count": {
            status: _distribution(
                [
                    float(_word_count(str(record["original"])))
                    for record in rows
                    if record["status"] == status
                ]
            )
            for status in ("validated", "semantic_failure")
        },
        "candidate_scores": {
            "accepted": candidate_summary(accepted_candidates),
            "rejected": candidate_summary(rejected_candidates),
        },
        "rejected_candidate_count": len(rejected_candidates),
        "rejection_category_candidate_counts": dict(reason_categories.most_common()),
        "top_exact_reasons": dict(exact_reasons.most_common(20)),
    }
