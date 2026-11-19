"""Build and verify the blinded EvidenceGraph-20K human-audit packet."""

from __future__ import annotations

import hashlib
from typing import Any


AUDIT_PACKET_PROTOCOL = "two-independent-raters-v2-blinded-aliases"
SOURCE_TASK_FIELDS = frozenset(
    {"source_id", "text", "author", "timestamp", "cited_source_ids"}
)


def audit_source_alias(claim_id: str, source_id: str) -> str:
    payload = f"evidencegraph-audit-source-v2\0{claim_id}\0{source_id}"
    return "src-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_audit_package(
    test_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    claim_ids: set[str] = set()
    tasks = []
    source_key = []
    ordered = sorted(
        test_records,
        key=lambda record: hashlib.sha256(
            str(record["claim_id"]).encode("utf-8")
        ).hexdigest(),
    )
    for record in ordered:
        claim_id = str(record["claim_id"])
        if claim_id in claim_ids:
            raise ValueError(f"duplicate audit claim ID: {claim_id}")
        claim_ids.add(claim_id)
        sources = list(record.get("sources", []))
        actual_ids = [str(source["source_id"]) for source in sources]
        if not sources or len(actual_ids) != len(set(actual_ids)):
            raise ValueError(f"invalid audit source IDs for {claim_id}")
        aliases = {
            source_id: audit_source_alias(claim_id, source_id)
            for source_id in actual_ids
        }
        if len(set(aliases.values())) != len(aliases):
            raise ValueError(f"audit source alias collision for {claim_id}")
        visible_sources = []
        for source in sources:
            source_id = str(source["source_id"])
            cited = [str(value) for value in source.get("cited_source_ids", [])]
            if any(value not in aliases for value in cited):
                raise ValueError(f"audit citation target is absent for {claim_id}")
            visible = {
                "source_id": aliases[source_id],
                "text": str(source["text"]),
                "cited_source_ids": [aliases[value] for value in cited],
            }
            if source.get("author"):
                visible["author"] = str(source["author"])
            if source.get("timestamp"):
                visible["timestamp"] = str(source["timestamp"])
            visible_sources.append(visible)
            source_key.append(
                {
                    "claim_id": claim_id,
                    "audit_source_id": aliases[source_id],
                    "source_id": source_id,
                }
            )
        visible_sources.sort(key=lambda source: str(source["source_id"]))
        tasks.append(
            {
                "claim_id": claim_id,
                "claim": str(record["claim"]),
                "sources": visible_sources,
            }
        )
    source_key.sort(key=lambda row: (row["claim_id"], row["audit_source_id"]))
    return tasks, source_key


def validate_audit_package(
    test_records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    source_key: list[dict[str, str]],
) -> tuple[set[str], set[tuple[str, str]]]:
    expected_tasks, expected_key = build_audit_package(test_records)
    if tasks != expected_tasks:
        raise ValueError("blinded audit tasks do not match the frozen dataset")
    if source_key != expected_key:
        raise ValueError("audit source key does not match the frozen dataset")
    for task in tasks:
        for source in task["sources"]:
            if not set(source).issubset(SOURCE_TASK_FIELDS):
                raise ValueError("blinded audit task exposes a forbidden source field")
    claim_ids = {str(task["claim_id"]) for task in tasks}
    source_ids = {
        (str(row["claim_id"]), str(row["audit_source_id"])) for row in source_key
    }
    return claim_ids, source_ids

