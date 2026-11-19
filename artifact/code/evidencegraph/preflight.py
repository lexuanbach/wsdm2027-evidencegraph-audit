"""Fail-closed evidence-reference preflight for selected FEVER cohorts."""

from __future__ import annotations

from typing import Any

from .wiki_index import WikiIndex


def preflight_evidence_references(
    selections: dict[str, list[dict[str, Any]]], index: WikiIndex
) -> dict[str, Any]:
    if not selections or any(not records for records in selections.values()):
        raise ValueError("preflight requires non-empty selection cohorts")
    seen_claim_ids: set[str] = set()
    cohort_summaries: dict[str, dict[str, int]] = {}
    missing: list[tuple[str, str, str, int]] = []
    total_records = 0
    total_references = 0
    for cohort, records in selections.items():
        references = 0
        for record in records:
            claim_id = str(record["claim_id"])
            if claim_id in seen_claim_ids:
                raise ValueError(f"duplicate claim ID across cohorts: {claim_id}")
            seen_claim_ids.add(claim_id)
            evidence = record.get("evidence")
            if not evidence:
                raise ValueError(f"{claim_id} has no canonical evidence references")
            for page_id, line_number in evidence:
                references += 1
                if index.get_sentence(str(page_id), int(line_number)) is None:
                    missing.append((cohort, claim_id, str(page_id), int(line_number)))
        cohort_summaries[cohort] = {
            "records": len(records),
            "references": references,
        }
        total_records += len(records)
        total_references += references
    if missing:
        examples = ", ".join(
            f"{cohort}/{claim_id}/{page_id}:{line_number}"
            for cohort, claim_id, page_id, line_number in missing[:5]
        )
        raise ValueError(
            f"{len(missing)} evidence references do not resolve; examples: {examples}"
        )
    return {
        "status": "complete",
        "cohorts": cohort_summaries,
        "total_records": total_records,
        "total_references": total_references,
        "missing_references": 0,
    }
