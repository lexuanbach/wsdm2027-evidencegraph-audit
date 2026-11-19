"""Deterministic source-pair features for shared-origin estimation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from .schema import SourceEvidence


FEATURE_NAMES = (
    "token_embedding_cosine",
    "token_jaccard",
    "char_4gram_jaccard",
    "shared_rare_token_rate",
    "same_domain",
    "explicit_citation",
    "same_author",
    "known_temporal_order",
    "log_time_gap_days",
)
DEPENDENCE_SCORER_PROTOCOL = "dependence-logistic-calibrated-v3-token-cosine"


@dataclass(frozen=True)
class SourceFeatureCache:
    """Source-local text/metadata features reused across within-episode pairs."""

    token_counts: Counter[str]
    token_set: frozenset[str]
    char_4grams: frozenset[str]
    timestamp: datetime | None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard(
    left: set[str] | frozenset[str], right: set[str] | frozenset[str]
) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    """Cosine similarity of deterministic sparse token-count embeddings."""

    if not left or not right:
        return 0.0
    dot = sum(float(count) * float(right.get(token, 0)) for token, count in left.items())
    left_norm = math.sqrt(sum(float(count) ** 2 for count in left.values()))
    right_norm = math.sqrt(sum(float(count) ** 2 for count in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    normalized = " ".join(_tokens(text))
    return {normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1))}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def cache_source_features(source: SourceEvidence) -> SourceFeatureCache:
    tokens = _tokens(source.text)
    return SourceFeatureCache(
        token_counts=Counter(tokens),
        token_set=frozenset(tokens),
        char_4grams=frozenset(_char_ngrams(source.text)),
        timestamp=_parse_time(source.timestamp),
    )


def pair_features_from_cache(
    left: SourceEvidence,
    right: SourceEvidence,
    left_cache: SourceFeatureCache,
    right_cache: SourceFeatureCache,
) -> NDArray[np.float64]:
    left_set, right_set = left_cache.token_set, right_cache.token_set
    shared = left_set & right_set
    rare_shared = sum(
        1
        for token in shared
        if left_cache.token_counts[token] + right_cache.token_counts[token] <= 2
    )
    rare_rate = rare_shared / max(1, min(len(left_set), len(right_set)))
    explicit_citation = float(
        left.source_id in right.cited_source_ids or right.source_id in left.cited_source_ids
    )
    left_time, right_time = left_cache.timestamp, right_cache.timestamp
    if left_time is None or right_time is None:
        temporal_order = 0.0
        log_gap = 0.0
    else:
        temporal_order = 1.0
        log_gap = math.log1p(abs((right_time - left_time).total_seconds()) / 86400.0)
    return np.asarray(
        [
            _counter_cosine(left_cache.token_counts, right_cache.token_counts),
            _jaccard(left_set, right_set),
            _jaccard(left_cache.char_4grams, right_cache.char_4grams),
            rare_rate,
            float(left.domain.casefold() == right.domain.casefold()),
            explicit_citation,
            float(bool(left.author) and left.author == right.author),
            temporal_order,
            log_gap,
        ],
        dtype=np.float64,
    )


def pair_features(left: SourceEvidence, right: SourceEvidence) -> NDArray[np.float64]:
    return pair_features_from_cache(
        left,
        right,
        cache_source_features(left),
        cache_source_features(right),
    )


def pair_label(left: SourceEvidence, right: SourceEvidence) -> int:
    """Construction label only; audited records must override disputed origins."""
    return int(left.origin_id == right.origin_id)
