"""Pool unique validated paraphrases across append-only generation attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.paraphrase_job import REQUIRED_VALIDATED


def _common_record_fields(record: dict, claim_id: str) -> tuple[str, str, int, list[str]]:
    source_record_id = str(record.get("source_record_id", "")).strip()
    original = str(record.get("original", "")).strip()
    validator_model = str(record.get("validator_model", "")).strip()
    validator_revision = str(record.get("validator_revision", "")).strip()
    validator_protocol = str(record.get("validator_protocol", "")).strip()
    paraphrases = record.get("paraphrases")
    required = int(record.get("required_validated", REQUIRED_VALIDATED))
    if not source_record_id or not original:
        raise ValueError(f"missing source identity or original for {claim_id}")
    if not validator_model or not validator_revision or not validator_protocol:
        raise ValueError(f"missing model or prompt provenance for {claim_id}")
    if required < 1:
        raise ValueError(f"invalid paraphrase requirement for {claim_id}")
    if (
        not isinstance(paraphrases, list)
        or any(not isinstance(value, str) or not value.strip() for value in paraphrases)
        or len({value.casefold().strip() for value in paraphrases}) != len(paraphrases)
        or len(paraphrases) > required
    ):
        raise ValueError(f"invalid accepted paraphrases for {claim_id}")
    return source_record_id, original, required, paraphrases


def _validate_attempt(
    attempt: dict,
    claim_id: str,
    validator_config: tuple[str, str, str, float],
) -> None:
    if not isinstance(attempt, dict):
        raise ValueError(f"malformed pooled attempt for {claim_id}")
    generator_model = str(attempt.get("generator_model", "")).strip()
    prompt_version = str(attempt.get("prompt_version", "")).strip()
    n_candidates = int(attempt.get("n_candidates", -1))
    n_accepted = int(attempt.get("n_accepted", -1))
    attempt_validator = (
        str(attempt.get("validator_model", "")).strip(),
        str(attempt.get("validator_revision", "")).strip(),
        str(attempt.get("validator_protocol", "")).strip(),
        float(attempt.get("threshold", -1.0)),
    )
    if not generator_model or not prompt_version:
        raise ValueError(f"missing pooled attempt generator provenance for {claim_id}")
    if n_candidates < 1 or not 0 <= n_accepted <= n_candidates:
        raise ValueError(f"invalid pooled attempt cardinality for {claim_id}")
    if attempt_validator != validator_config:
        raise ValueError(f"pooled attempt validator disagrees with root for {claim_id}")


def _record_attempts_and_floor(
    record: dict,
    claim_id: str,
    required: int,
    paraphrases: list[str],
    validator_config: tuple[str, str, str, float],
) -> tuple[list[dict], int, bool]:
    """Return attempts, a certified unique-count floor, and exactness."""

    if "attempts" in record:
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"missing pooled validation attempts for {claim_id}")
        for attempt in attempts:
            _validate_attempt(attempt, claim_id, validator_config)
        unique_count = int(
            record.get(
                "n_unique_accepted_lower_bound",
                record.get("n_unique_accepted", -1),
            )
        )
        if unique_count < len(paraphrases):
            raise ValueError(f"invalid pooled unique count for {claim_id}")
        expected_status = "validated" if unique_count >= required else "semantic_failure"
        if record.get("status") != expected_status:
            raise ValueError(f"pooled status disagrees with counts for {claim_id}")
        exact = (
            unique_count == len(paraphrases)
            and record.get("n_unique_accepted_is_lower_bound") is not True
        )
        return [dict(attempt) for attempt in attempts], unique_count, exact

    generator_model = str(record.get("generator_model", "")).strip()
    prompt_version = str(record.get("prompt_version", "")).strip()
    n_candidates = int(record.get("n_candidates", -1))
    n_accepted = int(record.get("n_accepted", -1))
    if not generator_model or not prompt_version:
        raise ValueError(f"missing model or prompt provenance for {claim_id}")
    if n_candidates < 1 or not 0 <= n_accepted <= n_candidates:
        raise ValueError(f"invalid validation cardinality for {claim_id}")
    if len(paraphrases) > n_accepted:
        raise ValueError(f"invalid accepted paraphrases for {claim_id}")
    expected_status = "validated" if n_accepted >= required else "semantic_failure"
    if record.get("status") != expected_status:
        raise ValueError(f"validation status disagrees with counts for {claim_id}")
    attempt = {
        "generator_model": generator_model,
        "prompt_version": prompt_version,
        "n_candidates": n_candidates,
        "n_accepted": n_accepted,
        "validator_model": validator_config[0],
        "validator_revision": validator_config[1],
        "validator_protocol": validator_config[2],
        "threshold": validator_config[3],
    }
    for field in (
        "nli_validator_model",
        "nli_validator_revision",
        "nli_validator_protocol",
        "structural_gate_protocol",
        "structural_parser_model",
        "structural_parser_model_version",
        "spacy_version",
        "source_validation_sha256",
    ):
        if field in record:
            attempt[field] = record[field]
    return [attempt], len(paraphrases), n_accepted == len(paraphrases)


def merge_records(paths: list[Path]) -> list[dict]:
    resolved_paths = [path.resolve() for path in paths]
    if not paths or len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("validation inputs must be non-empty and unique")
    pooled: dict[str, dict] = {}
    order: list[str] = []
    validator_config: tuple[str, str, str, float] | None = None
    for path in paths:
        seen_in_path: set[str] = set()
        for record in read_jsonl(path):
            claim_id = str(record["claim_id"])
            if claim_id in seen_in_path:
                raise ValueError(f"duplicate validation record in {path}: {claim_id}")
            seen_in_path.add(claim_id)
            if record.get("status") not in {"validated", "semantic_failure"}:
                raise ValueError(f"invalid validation status for {claim_id}")
            source_record_id, original, required, paraphrases = _common_record_fields(
                record, claim_id
            )
            current_validator = (
                str(record["validator_model"]),
                str(record["validator_revision"]),
                str(record["validator_protocol"]),
                float(record["threshold"]),
            )
            if validator_config is None:
                validator_config = current_validator
            elif current_validator != validator_config:
                raise ValueError(
                    "cannot merge different validator models, revisions, protocols, or thresholds"
                )
            attempts, accepted_floor, exact_count = _record_attempts_and_floor(
                record,
                claim_id,
                required,
                paraphrases,
                current_validator,
            )
            if claim_id not in pooled:
                pooled[claim_id] = {
                    "claim_id": claim_id,
                    "source_record_id": source_record_id,
                    "original": original,
                    "paraphrases": [],
                    "attempts": [],
                    "required_validated": required,
                    "validator_model": current_validator[0],
                    "validator_revision": current_validator[1],
                    "validator_protocol": current_validator[2],
                    "threshold": current_validator[3],
                    "accepted_count_floor": 0,
                    "accepted_count_exact": True,
                }
                order.append(claim_id)
            target = pooled[claim_id]
            if target["original"] != original:
                raise ValueError(f"conflicting originals for {claim_id}")
            if target["source_record_id"] != source_record_id:
                raise ValueError(f"conflicting source record IDs for {claim_id}")
            if target["required_validated"] != required:
                raise ValueError(f"conflicting paraphrase requirements for {claim_id}")
            target["attempts"].extend(attempts)
            target["accepted_count_floor"] = max(
                target["accepted_count_floor"], accepted_floor
            )
            target["accepted_count_exact"] = (
                target["accepted_count_exact"] and exact_count
            )
            seen = {text.casefold().strip() for text in target["paraphrases"]}
            for candidate in paraphrases:
                normalized = candidate.casefold().strip()
                if normalized not in seen:
                    target["paraphrases"].append(candidate)
                    seen.add(normalized)

    outputs = []
    for claim_id in order:
        output = pooled[claim_id]
        visible_count = len(output["paraphrases"])
        accepted_floor = max(output.pop("accepted_count_floor"), visible_count)
        accepted_count_exact = output.pop("accepted_count_exact")
        output["n_unique_accepted_lower_bound"] = accepted_floor
        output["n_unique_accepted_is_lower_bound"] = not accepted_count_exact
        if accepted_count_exact:
            output["n_unique_accepted"] = accepted_floor
        output["paraphrases"] = output["paraphrases"][: output["required_validated"]]
        output["status"] = (
            "validated"
            if accepted_floor >= output["required_validated"]
            else "semantic_failure"
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = merge_records(args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "semantic_failures": sum(record["status"] != "validated" for record in records),
        "validated": sum(record["status"] == "validated" for record in records),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
