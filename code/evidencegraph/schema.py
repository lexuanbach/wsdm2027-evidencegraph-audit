"""Canonical records shared by dataset construction and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ProvenanceFamily(str, Enum):
    INDEPENDENT = "independent"
    MIRROR = "mirror"
    CITATION = "citation"
    PARAPHRASE = "paraphrase"


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    text: str
    family: ProvenanceFamily
    stance: float
    stance_confidence: float
    domain: str
    origin_id: str
    author: str | None = None
    timestamp: str | None = None
    cited_source_ids: tuple[str, ...] = ()
    synthetic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.text or not self.origin_id:
            raise ValueError("source_id, text, and origin_id must be non-empty")
        if not -1.0 <= self.stance <= 1.0:
            raise ValueError("stance must be in [-1, 1]")
        if not 0.0 <= self.stance_confidence <= 1.0:
            raise ValueError("stance_confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["family"] = self.family.value
        record["cited_source_ids"] = list(self.cited_source_ids)
        return record


@dataclass(frozen=True)
class ClaimEpisode:
    claim_id: str
    claim: str
    label: bool
    topic_id: str
    split: str
    corruption: str
    sources: tuple[SourceEvidence, ...]
    source_dataset: str
    source_record_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if not 5 <= len(self.sources) <= 30:
            raise ValueError("an episode must contain 5-30 candidate sources")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique within an episode")

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["sources"] = [source.to_dict() for source in self.sources]
        return record
