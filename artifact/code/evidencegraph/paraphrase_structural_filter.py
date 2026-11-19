"""Apply the conservative structural gate after mutual-NLI validation."""

from __future__ import annotations

from typing import Any, Protocol

from .structural_validation import StructuralGateResult


STRUCTURE_PROTOCOL = "lexically-balanced-structure-and-surface-v5"


class StructureGate(Protocol):
    def validate(self, original: str, candidate: str) -> StructuralGateResult: ...


def _unique_casefolded(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        normalized = value.casefold().strip()
        if normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def filter_validation_record(
    record: dict[str, Any],
    gate: StructureGate,
    *,
    composite_validator_id: str,
    parser_model: str,
    parser_model_version: str,
    spacy_version: str,
    source_validation_sha256: str,
) -> dict[str, Any]:
    """Recompute accepted candidates while preserving the frozen NLI scores."""

    if record.get("status") not in {"validated", "semantic_failure"}:
        raise ValueError("invalid base validation status")
    original = str(record.get("original", ""))
    candidates = list(record.get("candidate_results", []))
    if not original or len(candidates) != int(record.get("n_candidates", -1)):
        raise ValueError("malformed base validation record")
    updated_candidates = []
    accepted_texts = []
    for candidate in candidates:
        text = str(candidate.get("candidate", ""))
        if not text.strip():
            raise ValueError("blank candidate in base validation record")
        base_accepted = candidate.get("accepted") is True
        structural_result = (
            gate.validate(original, text)
            if base_accepted
            else StructuralGateResult(False, ("not_evaluated_base_nli_rejection",))
        )
        accepted = base_accepted and structural_result.accepted
        updated_candidates.append(
            {
                **candidate,
                "base_nli_accepted": base_accepted,
                "structural_accepted": structural_result.accepted if base_accepted else None,
                "structural_reasons": list(structural_result.reasons),
                "accepted": accepted,
            }
        )
        if accepted:
            accepted_texts.append(text)
    accepted_texts = _unique_casefolded(accepted_texts)
    required = int(record.get("required_validated", -1))
    if required < 1:
        raise ValueError("invalid required paraphrase count")
    nli_model = str(record.get("validator_model", "")).strip()
    nli_revision = str(record.get("validator_revision", "")).strip()
    nli_protocol = str(record.get("validator_protocol", "")).strip()
    if not nli_model or not nli_revision or not nli_protocol:
        raise ValueError("missing base NLI validator provenance")
    output = {
        **record,
        "nli_validator_model": nli_model,
        "nli_validator_revision": nli_revision,
        "nli_validator_protocol": nli_protocol,
        "validator_model": composite_validator_id,
        "structural_gate_protocol": STRUCTURE_PROTOCOL,
        "structural_parser_model": parser_model,
        "structural_parser_model_version": parser_model_version,
        "spacy_version": spacy_version,
        "source_validation_sha256": source_validation_sha256,
        "candidate_results": updated_candidates,
        "paraphrases": accepted_texts[:required],
        "n_accepted": len(accepted_texts),
        "status": "validated" if len(accepted_texts) >= required else "semantic_failure",
    }
    return output
