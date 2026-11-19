"""Convert retrieved passages and learned edge scores into a controller-ready graph."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from .dependence_features import (
    FEATURE_NAMES,
    cache_source_features,
    pair_features_from_cache,
)
from .schema import ProvenanceFamily, SourceEvidence


class ProbabilityEstimator(Protocol):
    def predict_proba(self, features: Any) -> np.ndarray: ...


NEAR_DUPLICATE_THRESHOLD = 0.80


def source_from_retrieval(
    result: dict[str, Any], probabilities: dict[str, float]
) -> SourceEvidence:
    source_id = str(result["source_id"])
    url = str(result.get("url", ""))
    domain = urlparse(url).netloc.casefold() if url else "wikipedia.org"
    stance = float(probabilities["support"] - probabilities["refute"])
    confidence = float(probabilities["support"] + probabilities["refute"])
    author = result.get("author")
    timestamp = result.get("timestamp")
    return SourceEvidence(
        source_id=source_id,
        text=str(result["text"]),
        family=ProvenanceFamily.INDEPENDENT,
        stance=stance,
        stance_confidence=confidence,
        domain=domain or "unknown.invalid",
        origin_id=f"unobserved:{source_id}",
        author=None if author in (None, "") else str(author),
        timestamp=None if timestamp in (None, "") else str(timestamp),
        cited_source_ids=tuple(map(str, result.get("cited_source_ids") or [])),
        metadata={"retrieval_rank": int(result["rank"]), "url": url or None},
    )


def source_nodes(sources: list[SourceEvidence]) -> list[dict[str, Any]]:
    """Serialize compact source-node identities without duplicating passage text."""

    return [
        {
            "source_id": source.source_id,
            "kind": "source",
            "domain": source.domain,
            "retrieval_rank": int(source.metadata["retrieval_rank"]),
            "url": source.metadata.get("url"),
        }
        for source in sources
    ]


def predicted_dependence_matrix(
    sources: list[SourceEvidence], estimators: list[ProbabilityEstimator]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(sources) < 2 or not estimators:
        raise ValueError("dependence prediction requires sources and estimators")
    pairs = list(combinations(range(len(sources)), 2))
    caches = [cache_source_features(source) for source in sources]
    features = np.vstack(
        [
            pair_features_from_cache(
                sources[left], sources[right], caches[left], caches[right]
            )
            for left, right in pairs
        ]
    )
    seed_predictions = [np.asarray(estimator.predict_proba(features)) for estimator in estimators]
    if any(prediction.shape != (len(pairs),) for prediction in seed_predictions):
        raise ValueError("dependence estimator returned an invalid prediction shape")
    values = np.mean(np.vstack(seed_predictions), axis=0)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("dependence estimator returned invalid probabilities")
    matrix = np.eye(len(sources), dtype=np.float64)
    records = []
    for (left, right), probability in zip(pairs, values):
        matrix[left, right] = matrix[right, left] = float(probability)
        records.append(
            {
                "left_id": sources[left].source_id,
                "right_id": sources[right].source_id,
                "probability": float(probability),
            }
        )
    return matrix, records


def typed_source_pair_edges(
    sources: list[SourceEvidence],
    *,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, list[dict[str, Any]]]:
    """Materialize citation and thresholded near-duplicate graph edges."""

    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    known = {source.source_id for source in sources}
    caches = [cache_source_features(source) for source in sources]
    cosine_index = FEATURE_NAMES.index("token_embedding_cosine")
    ngram_index = FEATURE_NAMES.index("char_4gram_jaccard")
    citation_edges: list[dict[str, Any]] = []
    near_duplicate_edges: list[dict[str, Any]] = []
    for source in sources:
        for target_id in sorted(set(source.cited_source_ids)):
            if target_id in known and target_id != source.source_id:
                citation_edges.append(
                    {
                        "edge_type": "citation",
                        "source_id": source.source_id,
                        "target_id": target_id,
                    }
                )
    for left, right in combinations(range(len(sources)), 2):
        features = pair_features_from_cache(
            sources[left], sources[right], caches[left], caches[right]
        )
        cosine = float(features[cosine_index])
        ngram = float(features[ngram_index])
        score = 0.5 * (cosine + ngram)
        if score >= near_duplicate_threshold:
            near_duplicate_edges.append(
                {
                    "edge_type": "near_duplicate",
                    "left_id": sources[left].source_id,
                    "right_id": sources[right].source_id,
                    "token_embedding_cosine": cosine,
                    "char_4gram_jaccard": ngram,
                    "score": score,
                    "threshold": near_duplicate_threshold,
                }
            )
    return {
        "citation_edges": citation_edges,
        "near_duplicate_edges": near_duplicate_edges,
    }


def representative_indices(
    probabilities: list[dict[str, float]],
    dependence: np.ndarray,
    selected_indices: list[int],
    *,
    threshold: float = 0.5,
) -> list[int]:
    """Keep the strongest semantic edge from each predicted origin cluster."""
    n_sources = len(probabilities)
    if not 0.0 < threshold < 1.0:
        raise ValueError("dependence cluster threshold must be in (0, 1)")
    if dependence.shape != (n_sources, n_sources):
        raise ValueError("dependence matrix and source probabilities must align")
    if not selected_indices or len(selected_indices) != len(set(selected_indices)):
        raise ValueError("selected source indices must be non-empty and unique")
    if min(selected_indices) < 0 or max(selected_indices) >= n_sources:
        raise ValueError("selected source index is out of bounds")
    def strength_key(index: int) -> tuple[float, int]:
        return (
            -max(
                float(probabilities[index]["support"]),
                float(probabilities[index]["refute"]),
            ),
            index,
        )

    remaining = set(selected_indices)
    representatives = []
    while remaining:
        frontier = [min(remaining)]
        component = set()
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            remaining.discard(current)
            frontier.extend(
                candidate
                for candidate in list(remaining)
                if float(dependence[current, candidate]) >= threshold
            )
        representatives.append(min(component, key=strength_key))
    return sorted(representatives, key=strength_key)
