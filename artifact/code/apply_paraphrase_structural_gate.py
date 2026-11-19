"""Append-only structural post-filter for mutual-NLI validation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.paraphrase_structural_filter import (
    STRUCTURE_PROTOCOL,
    filter_validation_record,
)
from evidencegraph.structural_validation import SpacyLexicallyAnchoredGate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parser-model", default="en_core_web_sm")
    args = parser.parse_args()

    rows = read_jsonl(args.validation)
    source_sha = sha256(args.validation)
    completed: set[str] = set()
    existing = read_jsonl(args.output) if args.output.exists() else []
    for record in existing:
        claim_id = str(record["claim_id"])
        if claim_id in completed:
            raise ValueError(f"duplicate structural validation record: {claim_id}")
        if record.get("source_validation_sha256") != source_sha:
            raise ValueError("structural output was derived from different validation bytes")
        if record.get("structural_gate_protocol") != STRUCTURE_PROTOCOL:
            raise ValueError("structural output mixes protocol versions")
        completed.add(claim_id)
    input_ids = [str(record["claim_id"]) for record in rows]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("base validation contains duplicate claim IDs")
    if not completed.issubset(input_ids):
        raise ValueError("structural output contains claims absent from base validation")
    pending = [record for record in rows if str(record["claim_id"]) not in completed]
    if not pending:
        print(json.dumps({"filtered": 0, "semantic_failures": 0, "validated": 0}))
        return

    gate = SpacyLexicallyAnchoredGate(args.parser_model)
    base_configs = {
        (
            str(record.get("validator_model", "")).strip(),
            str(record.get("validator_revision", "")).strip(),
            str(record.get("validator_protocol", "")).strip(),
        )
        for record in rows
    }
    if len(base_configs) != 1 or not all(next(iter(base_configs))):
        raise ValueError(
            "base validation mixes or omits NLI model, revision, or protocol"
        )
    base_model, base_revision, base_protocol = next(iter(base_configs))
    composite_id = (
        f"{base_model}@{base_revision}|{base_protocol}|{STRUCTURE_PROTOCOL}|"
        f"spacy-{gate.spacy_version}:{gate.model_name}-{gate.model_version}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    validated = failures = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for record in pending:
            output = filter_validation_record(
                record,
                gate,
                composite_validator_id=composite_id,
                parser_model=gate.model_name,
                parser_model_version=gate.model_version,
                spacy_version=gate.spacy_version,
                source_validation_sha256=source_sha,
            )
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if output["status"] == "validated":
                validated += 1
            else:
                failures += 1
    print(
        json.dumps(
            {"filtered": len(pending), "semantic_failures": failures, "validated": validated},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
