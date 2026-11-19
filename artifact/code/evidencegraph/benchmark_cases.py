"""Normalize official benchmark releases into immutable evaluation cases."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_loaders import (
    load_averitec,
    load_expertqa,
    load_expertqa_lfqa,
    load_fever,
    load_hover,
)


@dataclass(frozen=True)
class BenchmarkCase:
    dataset: str
    case_id: str
    query: str
    gold_label: str | None
    gold_evidence: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["gold_evidence"] = list(self.gold_evidence)
        return record


_FEVER_TITLE_ESCAPES = {
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LSB-",
    "]": "-RSB-",
    "{": "-LCB-",
    "}": "-RCB-",
    ":": "-COLON-",
}


def fever_wiki_page_id(title: str) -> str:
    """Map an official human-readable title to the FEVER snapshot page-ID form."""
    normalized = unicodedata.normalize("NFC", html.unescape(title)).replace(" ", "_")
    return "".join(_FEVER_TITLE_ESCAPES.get(character, character) for character in normalized)


def fever_cases(path: Path) -> list[BenchmarkCase]:
    cases = []
    for record in load_fever(path):
        evidence_groups = []
        for raw_group in record["evidence"]:
            group = []
            for item in raw_group:
                if len(item) >= 4 and item[2] is not None and item[3] is not None:
                    page_id = unicodedata.normalize("NFC", str(item[2]))
                    group.append(f"{page_id}:{int(item[3])}")
            if group:
                evidence_groups.append(sorted(set(group)))
        cases.append(
            BenchmarkCase(
                dataset="FEVER",
                case_id=str(record["id"]),
                query=str(record["claim"]),
                gold_label=str(record["label"]),
                gold_evidence=tuple(sorted({item for group in evidence_groups for item in group})),
                metadata={
                    "verifiable": str(record["verifiable"]),
                    "evidence_groups": evidence_groups,
                },
            )
        )
    return cases


def hover_cases(path: Path) -> list[BenchmarkCase]:
    cases = []
    for record in load_hover(path):
        supporting = tuple(
            sorted(
                f"{fever_wiki_page_id(str(page))}:{int(line)}"
                for page, line in record.get("supporting_facts", [])
            )
        )
        cases.append(
            BenchmarkCase(
                dataset="HoVer",
                case_id=str(record["uid"]),
                query=str(record["claim"]),
                gold_label=record.get("label"),
                gold_evidence=supporting,
                metadata={
                    "num_hops": record.get("num_hops"),
                    "hpqa_id": record.get("hpqa_id"),
                },
            )
        )
    return cases


def averitec_cases(path: Path) -> list[BenchmarkCase]:
    cases = []
    for claim_id, record in enumerate(load_averitec(path)):
        gold_urls = {
            str(answer["source_url"])
            for question in record["questions"]
            for answer in question.get("answers", [])
            if answer.get("source_url")
        }
        cases.append(
            BenchmarkCase(
                dataset="AVeriTeC",
                case_id=str(claim_id),
                query=str(record["claim"]),
                gold_label=str(record["label"]),
                gold_evidence=tuple(sorted(gold_urls)),
                metadata={
                    "claim_date": record.get("claim_date"),
                    "speaker": record.get("speaker"),
                    "justification": record.get("justification"),
                },
            )
        )
    return cases


_EVIDENCE_PREFIX = re.compile(
    r"^\s*\[(\d+)\]\s+(https?://\S+)(?:\s*\n+\s*)?", re.IGNORECASE | re.MULTILINE
)


def parse_expertqa_evidence(value: str) -> tuple[str, str]:
    """Return a stable evidence ID and passage from either official release form.

    Most ExpertQA evidence items start with a numbered source URL.  The official
    all-passage release also contains passage-only items for which no reliable
    per-passage URL mapping is retained.  Those receive content-addressed IDs so
    that the passage remains usable without fabricating provenance.
    """
    match = _EVIDENCE_PREFIX.match(value)
    if match:
        citation_id, url = match.group(1), match.group(2)
        passage = value[match.end() :].strip()
        if not passage:
            raise ValueError("ExpertQA numbered evidence lacks passage text")
        return f"[{citation_id}] {url}", passage
    passage = value.strip()
    if not passage:
        raise ValueError("ExpertQA evidence is empty")
    if re.match(r"^\s*\[\d+\]", value):
        raise ValueError("ExpertQA numbered evidence lacks a valid HTTP(S) URL")
    digest = hashlib.sha256(passage.encode("utf-8")).hexdigest()
    return f"passage:sha256:{digest}", passage


def expertqa_cases(annotated_path: Path, lfqa_test_path: Path) -> list[BenchmarkCase]:
    annotated = {record["question"]: record for record in load_expertqa(annotated_path)}
    requested = load_expertqa_lfqa(lfqa_test_path)
    missing = [record["question"] for record in requested if record["question"] not in annotated]
    if missing:
        raise ValueError(f"{len(missing)} ExpertQA test questions lack annotated evidence")
    cases = []
    for test_record in requested:
        record = annotated[test_record["question"]]
        if len(record["answers"]) != 1:
            raise ValueError("ExpertQA annotated release must have one sampled answer per question")
        system, answer = next(iter(record["answers"].items()))
        evidence: dict[str, str] = {}
        gold_ids: set[str] = set()
        claims = []
        for claim in answer.get("claims", []):
            claim_record = {
                "claim": claim.get("claim_string", ""),
                "support": claim.get("support"),
                "evidence_ids": [],
            }
            for raw_evidence in claim.get("evidence") or []:
                evidence_id, passage = parse_expertqa_evidence(str(raw_evidence))
                if evidence_id in evidence and evidence[evidence_id] != passage:
                    raise ValueError(
                        f"ExpertQA evidence ID maps to conflicting passages: {evidence_id}"
                    )
                evidence[evidence_id] = passage
                claim_record["evidence_ids"].append(evidence_id)
                # ExpertQA's official AutoAIS/human-correlation code treats
                # Complete as positive and every other support category as
                # negative. Preserve all evidence above, but expose only fully
                # supporting evidence as gold.
                if claim.get("support") == "Complete":
                    gold_ids.add(evidence_id)
            claims.append(claim_record)
        cases.append(
            BenchmarkCase(
                dataset="ExpertQA",
                case_id=str(test_record.get("example_id", len(cases))),
                query=str(record["question"]),
                gold_label=None,
                gold_evidence=tuple(sorted(gold_ids)),
                metadata={
                    "reference_system": system,
                    "reference_answer": answer.get("answer_string", ""),
                    "evidence_passages": evidence,
                    "claims": claims,
                    "field": record.get("metadata", {}).get("field"),
                },
            )
        )
    return cases
