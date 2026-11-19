from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.retrieval_graph import (
    predicted_dependence_matrix,
    representative_indices,
    source_from_retrieval,
    typed_source_pair_edges,
)


class ConstantEstimator:
    def __init__(self, value):
        self.value = value

    def predict_proba(self, features):
        return np.full(len(features), self.value)


def test_retrieval_source_preserves_domain_and_nli_stance() -> None:
    source = source_from_retrieval(
        {
            "source_id": "s1",
            "text": "Evidence.",
            "rank": 1,
            "url": "https://News.Example/path",
            "author": "Reporter",
            "timestamp": "2017-02-01T00:00:00Z",
            "cited_source_ids": ["s2"],
        },
        {"support": 0.8, "neutral": 0.1, "refute": 0.1},
    )
    assert source.domain == "news.example"
    assert source.stance == pytest.approx(0.7)
    assert source.stance_confidence == pytest.approx(0.9)
    assert source.author == "Reporter"
    assert source.timestamp == "2017-02-01T00:00:00Z"
    assert source.cited_source_ids == ("s2",)


def test_retrieval_source_normalizes_null_citations() -> None:
    source = source_from_retrieval(
        {
            "source_id": "s1",
            "text": "Evidence.",
            "rank": 1,
            "cited_source_ids": None,
        },
        {"support": 0.8, "neutral": 0.1, "refute": 0.1},
    )

    assert source.cited_source_ids == ()


def test_retrieval_source_ignores_gold_and_construction_metadata() -> None:
    shared = {
        "source_id": "s1",
        "text": "Evidence.",
        "rank": 1,
        "url": "https://news.example/path",
    }
    probabilities = {"support": 0.8, "neutral": 0.1, "refute": 0.1}
    left = source_from_retrieval(
        {
            **shared,
            "label": "SUPPORTED",
            "gold_label": "support",
            "family": "mirror",
            "origin_id": "gold-origin",
            "corruption": "duplication",
            "stance": -1.0,
        },
        probabilities,
    )
    right = source_from_retrieval(
        {
            **shared,
            "label": "REFUTED",
            "gold_label": "refute",
            "family": "independent",
            "origin_id": "another-origin",
            "corruption": "hidden_best_evidence",
            "stance": 1.0,
        },
        probabilities,
    )

    assert left == right
    assert left.origin_id == "unobserved:s1"


def test_dependence_matrix_averages_three_estimators() -> None:
    sources = [
        source_from_retrieval(
            {"source_id": source_id, "text": text, "rank": rank},
            {"support": 0.8, "neutral": 0.1, "refute": 0.1},
        )
        for rank, (source_id, text) in enumerate(
            [("a", "same text"), ("b", "same text")], start=1
        )
    ]
    matrix, records = predicted_dependence_matrix(
        sources, [ConstantEstimator(0.6), ConstantEstimator(0.8), ConstantEstimator(1.0)]
    )
    assert matrix[0, 1] == pytest.approx(0.8)
    assert records[0]["probability"] == pytest.approx(0.8)


def test_typed_source_pair_edges_materialize_citation_and_near_duplicate() -> None:
    probabilities = {"support": 0.8, "neutral": 0.1, "refute": 0.1}
    sources = [
        source_from_retrieval(
            {
                "source_id": "a",
                "text": "The same evidence appears here.",
                "rank": 1,
                "cited_source_ids": ["b"],
            },
            probabilities,
        ),
        source_from_retrieval(
            {"source_id": "b", "text": "The same evidence appears here.", "rank": 2},
            probabilities,
        ),
    ]

    edges = typed_source_pair_edges(sources)

    assert edges["citation_edges"] == [
        {"edge_type": "citation", "source_id": "a", "target_id": "b"}
    ]
    assert edges["near_duplicate_edges"][0]["score"] == pytest.approx(1.0)


def test_representatives_keep_strongest_semantic_edge_per_cluster() -> None:
    probabilities = [
        {"support": 0.8, "neutral": 0.1, "refute": 0.1},
        {"support": 0.95, "neutral": 0.03, "refute": 0.02},
        {"support": 0.7, "neutral": 0.2, "refute": 0.1},
    ]
    dependence = np.asarray(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]]
    )

    assert representative_indices(probabilities, dependence, [0, 1, 2]) == [1, 2]


def test_representative_clusters_use_transitive_connected_components() -> None:
    probabilities = [
        {"support": value, "neutral": 1.0 - value, "refute": 0.0}
        for value in (0.8, 0.9, 0.7)
    ]
    dependence = np.asarray(
        [[1.0, 0.8, 0.1], [0.8, 1.0, 0.8], [0.1, 0.8, 1.0]]
    )

    assert representative_indices(probabilities, dependence, [0, 1, 2]) == [1]
