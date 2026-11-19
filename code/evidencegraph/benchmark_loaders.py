"""Strict loaders for the pinned official benchmark releases."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"expected a list of records in {path}")
    return value


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def load_fever(path: Path) -> list[dict[str, Any]]:
    records = list(_jsonl(path))
    required = {"id", "verifiable", "label", "claim", "evidence"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("FEVER record is missing a required field")
    return records


def load_hover(path: Path) -> list[dict[str, Any]]:
    records = _json_list(path)
    # The official blind test release intentionally contains only IDs/claims.
    required = {"uid", "claim"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("HoVer record is missing a required field")
    annotation_fields = {"supporting_facts", "label", "num_hops"}
    if any(annotation_fields & set(record) and not annotation_fields.issubset(record) for record in records):
        raise ValueError("HoVer annotated record has only a partial annotation group")
    return records


def load_averitec(path: Path) -> list[dict[str, Any]]:
    records = _json_list(path)
    required = {"claim", "label", "questions", "justification"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("AVeriTeC record is missing a required field")
    return records


def load_expertqa_lfqa(path: Path) -> list[dict[str, Any]]:
    """The official LFQA files use JSON Lines despite their .json suffix."""
    records = list(_jsonl(path))
    if any("question" not in record for record in records):
        raise ValueError("ExpertQA LFQA record is missing question")
    return records


def load_expertqa(path: Path) -> list[dict[str, Any]]:
    """Load the evidence-annotated ExpertQA release used for citation metrics."""
    records = list(_jsonl(path))
    required = {"question", "annotator_id", "answers", "metadata"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("ExpertQA annotated record is missing a required field")
    if any(not isinstance(record["answers"], dict) for record in records):
        raise ValueError("ExpertQA answers must be a model-keyed object")
    return records


def benchmark_inventory(project_root: Path) -> dict[str, dict[str, int]]:
    external = project_root / ".external"
    raw = project_root / "data" / "raw"
    return {
        "FEVER": {
            "train": len(load_fever(raw / "fever_train.jsonl")),
            "dev": len(load_fever(raw / "fever_dev.jsonl")),
        },
        "HoVer": {
            split: len(
                load_hover(
                    external
                    / "hover-official"
                    / "data"
                    / "hover"
                    / f"hover_{split}_release_v1.1.json"
                )
            )
            for split in ("train", "dev", "test")
        },
        "AVeriTeC": {
            split: len(load_averitec(external / "averitec-official" / "data" / f"{split}.json"))
            for split in ("train", "dev")
        },
        "ExpertQA": {
            "annotated": len(
                load_expertqa(
                    external / "expertqa-official" / "data" / "r2_compiled_anon.jsonl"
                )
            ),
            "all_passage_evidence": len(
                load_expertqa(
                    external
                    / "expertqa-official"
                    / "data"
                    / "r2_compiled_all_evidences_autoais_anon.jsonl"
                )
            ),
            **{
                f"lfqa_{split}": len(
                    load_expertqa_lfqa(
                        external
                        / "expertqa-official"
                        / "data"
                        / "lfqa"
                        / f"domain_lfqa_{split}.json"
                    )
                )
                for split in ("train", "val", "test")
            },
        },
    }
