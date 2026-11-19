"""Fail-closed coverage and provenance audit for paraphrase generation caches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .dataset_builder import read_jsonl
from .paraphrase_job import PROMPT_VERSION, SUPPORTED_PROMPT_VERSIONS, candidate_requirements


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_generation_cache(
    selection_path: Path,
    raw_path: Path,
    *,
    expected_model: str,
    expected_prompt_version: str = PROMPT_VERSION,
    require_complete: bool = True,
) -> dict[str, Any]:
    if expected_prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError("unsupported expected prompt protocol")
    selection = read_jsonl(selection_path)
    selected: dict[str, dict[str, Any]] = {}
    for record in selection:
        claim_id = str(record["claim_id"])
        if claim_id in selected:
            raise ValueError(f"duplicate selection claim ID: {claim_id}")
        selected[claim_id] = record

    generated: dict[str, dict[str, Any]] = {}
    failed_ids: set[str] = set()
    failure_events = 0
    failure_claim_events = 0
    raw_records = read_jsonl(raw_path)
    for record in raw_records:
        status = record.get("status")
        if record.get("model") != expected_model:
            raise ValueError("generation cache mixes models or uses the wrong model")
        if record.get("prompt_version") != expected_prompt_version:
            raise ValueError("generation cache mixes prompt protocols")
        if status == "schema_failure":
            claim_ids = [str(value) for value in record.get("claim_ids", [])]
            if not claim_ids or len(claim_ids) != len(set(claim_ids)):
                raise ValueError("schema-failure row has invalid claim IDs")
            unknown = set(claim_ids) - set(selected)
            if unknown:
                raise ValueError("schema-failure row contains unselected claim IDs")
            failed_ids.update(claim_ids)
            failure_events += 1
            failure_claim_events += len(claim_ids)
            continue
        if status != "generated":
            raise ValueError(f"unexpected generation status: {status}")

        claim_id = str(record["claim_id"])
        if claim_id not in selected:
            raise ValueError(f"generated unselected claim: {claim_id}")
        if claim_id in generated:
            raise ValueError(f"duplicate generated claim: {claim_id}")
        source = selected[claim_id]
        expected_candidates, expected_required = candidate_requirements(source)
        paraphrases = record.get("paraphrases")
        if (
            int(record.get("candidates_requested", -1)) != expected_candidates
            or int(record.get("required_validated", -1)) != expected_required
            or not isinstance(paraphrases, list)
            or len(paraphrases) != expected_candidates
            or any(not isinstance(value, str) or not value.strip() for value in paraphrases)
        ):
            raise ValueError(f"invalid generated candidate cardinality for {claim_id}")
        if str(record.get("source_record_id")) != str(source["source_record_id"]):
            raise ValueError(f"source record mismatch for {claim_id}")
        if not str(record.get("original", "")).strip():
            raise ValueError(f"missing original evidence for {claim_id}")
        generated[claim_id] = record

    missing = sorted(set(selected) - set(generated))
    if require_complete and missing:
        raise ValueError(f"incomplete generation cache: {len(missing)} claims are missing")
    recovered = failed_ids & set(generated)
    unresolved = failed_ids - set(generated)
    return {
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "raw": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "expected_model": expected_model,
        "prompt_version": expected_prompt_version,
        "selection_count": len(selected),
        "raw_record_count": len(raw_records),
        "generated_count": len(generated),
        "missing_count": len(missing),
        "schema_failure_events": failure_events,
        "schema_failure_claim_events": failure_claim_events,
        "recovered_failure_ids": len(recovered),
        "unresolved_failure_ids": len(unresolved),
        "complete": not missing and not unresolved,
    }
