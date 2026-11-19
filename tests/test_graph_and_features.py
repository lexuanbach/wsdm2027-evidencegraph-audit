from __future__ import annotations

import pytest

from evidencegraph.dependence_features import FEATURE_NAMES, pair_features, pair_label
from evidencegraph.graph import EvidenceGraph
from evidencegraph.schema import ProvenanceFamily, SourceEvidence


def source(
    source_id: str,
    text: str,
    origin: str,
    confidence: float = 0.9,
    domain: str | None = None,
) -> SourceEvidence:
    return SourceEvidence(
        source_id=source_id,
        text=text,
        family=ProvenanceFamily.INDEPENDENT if source_id == origin else ProvenanceFamily.MIRROR,
        stance=1.0,
        stance_confidence=confidence,
        domain=domain or f"{source_id}.example",
        origin_id=origin,
    )


def test_pair_features_detect_same_origin_and_copy_similarity() -> None:
    left = source("origin", "A rare quokka observation happened in 2024.", "origin")
    right = source("copy", "A rare quokka observation happened in 2024.", "origin")
    features = pair_features(left, right)
    assert pair_label(left, right) == 1
    assert features[0] == pytest.approx(1.0)
    assert features[1] == pytest.approx(1.0)


def test_pair_features_include_deterministic_token_embedding_cosine() -> None:
    left = source("left", "alpha alpha beta", "left")
    right = source("right", "alpha beta beta", "right")

    features = pair_features(left, right)

    assert FEATURE_NAMES[0] == "token_embedding_cosine"
    assert features[0] == pytest.approx(4.0 / 5.0)


def test_same_domain_is_observable_without_implying_shared_origin() -> None:
    left = source(
        "page-a",
        "Independent evidence from one Wikipedia page.",
        "wiki:page-a",
        domain="wikipedia.org",
    )
    right = source(
        "page-b",
        "Different evidence from another Wikipedia page.",
        "wiki:page-b",
        domain="wikipedia.org",
    )

    features = pair_features(left, right)

    assert features[FEATURE_NAMES.index("same_domain")] == pytest.approx(1.0)
    assert pair_label(left, right) == 0


def test_graph_reports_one_representative_for_exact_copy_cluster() -> None:
    graph = EvidenceGraph("A claim")
    graph.add_source(source("origin", "same text", "origin", 0.95))
    graph.add_source(source("copy", "same text", "origin", 0.80))
    graph.set_dependence("origin", "copy", 1.0)
    assert graph.representative_source_ids() == ["origin"]


def test_graph_uses_transitive_dependence_components() -> None:
    graph = EvidenceGraph("A claim")
    graph.add_source(source("a", "first", "a", 0.70))
    graph.add_source(source("b", "second", "b", 0.95))
    graph.add_source(source("c", "third", "c", 0.80))
    graph.set_dependence("a", "b", 0.90)
    graph.set_dependence("b", "c", 0.90)
    graph.set_dependence("a", "c", 0.10)
    assert graph.representative_source_ids() == ["b"]


def test_graph_rejects_invalid_representative_threshold() -> None:
    graph = EvidenceGraph("A claim")
    graph.add_source(source("origin", "same text", "origin"))
    with pytest.raises(ValueError):
        graph.representative_source_ids(threshold=1.0)


def test_graph_rejects_unknown_dependence_edge() -> None:
    graph = EvidenceGraph("A claim")
    graph.add_source(source("origin", "same text", "origin"))
    with pytest.raises(KeyError):
        graph.set_dependence("origin", "missing", 0.5)
