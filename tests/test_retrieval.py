from __future__ import annotations

import json

import numpy as np
import pytest

from evidencegraph.retrieval import (
    aggregate_averitec_documents,
    finite_cross_scores,
    finite_dense_scores,
    finite_multiquery_dense_scores,
    lexical_overlap_order,
    local_lexical_results,
    maximal_marginal_relevance,
    merge_query_candidates,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_union,
    select_averitec_url_maxp,
)


def test_reciprocal_rank_fusion_rewards_consensus() -> None:
    identifiers = ["a", "b", "c"]
    order = reciprocal_rank_fusion(identifiers, [0, 1, 2], [1, 0, 2])
    assert order[-1] == 2
    assert set(order) == {0, 1, 2}


def test_reciprocal_rank_fusion_union_keeps_new_items_and_consensus() -> None:
    fused = reciprocal_rank_fusion_union(
        [["lexical-only", "shared"], ["shared", "dense-only"]]
    )

    assert [identifier for identifier, _score in fused] == [
        "shared",
        "lexical-only",
        "dense-only",
    ]
    json.dumps(fused)


def test_mmr_exhausts_unique_sources_before_duplicate_passages() -> None:
    selected = maximal_marginal_relevance(
        [0.9, 0.89, 0.8],
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        [0, 1, 2],
        limit=3,
        relevance_weight=1.0,
        group_ids=["source-a", "source-a", "source-b"],
    )

    assert [index for index, _score in selected] == [0, 2, 1]


def test_mmr_normalizes_numpy_indices_for_json_output() -> None:
    selected = maximal_marginal_relevance(
        [0.9, 0.8],
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        list(np.argsort(-np.asarray([0.9, 0.8]))),
        limit=2,
    )

    assert all(type(index) is int for index, _score in selected)
    json.dumps(selected)


def test_case_local_lexical_order_prefers_query_overlap() -> None:
    order = lexical_overlap_order(
        "solar energy policy",
        ["unrelated cooking notes", "solar policy and clean energy", "solar"],
    )
    assert order[0] == 1
    assert order[-1] == 0


def test_local_lexical_results_preserve_source_ids() -> None:
    results = local_lexical_results(
        "solar policy",
        {"s1": "cooking", "s2": "solar energy policy"},
        limit=1,
    )

    assert results == [
        {
            "source_id": "s2",
            "text": "solar energy policy",
            "rank": 1,
            "lexical_rank": 1,
        }
    ]


def test_finite_dense_scores_uses_query_first() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]])
    assert finite_dense_scores(embeddings).tolist() == [0.5, -1.0]


def test_finite_dense_scores_rejects_nonfinite_embeddings() -> None:
    with pytest.raises(RuntimeError, match="non-finite embeddings"):
        finite_dense_scores([[1.0, 0.0], [np.nan, 1.0]])


def test_finite_cross_scores_rejects_bad_values_and_cardinality() -> None:
    assert finite_cross_scores([0.25, -0.5], 2).tolist() == [0.25, -0.5]
    with pytest.raises(RuntimeError, match="wrong score cardinality"):
        finite_cross_scores([0.25], 2)
    with pytest.raises(RuntimeError, match="non-finite scores"):
        finite_cross_scores([np.inf], 1)


def test_finite_multiquery_scores_choose_best_query_per_passage() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.1, 0.9]]
    )
    scores, queries = finite_multiquery_dense_scores(embeddings, 2)

    assert scores.tolist() == pytest.approx([0.8, 0.9])
    assert queries.tolist() == [0, 1]


def test_merge_query_candidates_preserves_query_specific_ranks() -> None:
    first = {
        "claim_id": "7",
        "document_index": 2,
        "passage_index": 0,
        "text": "shared",
    }
    second = {
        "claim_id": "7",
        "document_index": 3,
        "passage_index": 0,
        "text": "new",
    }
    merged = merge_query_candidates([[first], [second, first]])

    assert [record["text"] for record in merged] == ["shared", "new"]
    assert merged[0]["query_lexical_ranks"] == [1, 2]
    assert merged[1]["query_lexical_ranks"] == [None, 1]


def test_averitec_document_aggregation_deduplicates_urls_and_text() -> None:
    passages = [
        {
            "url": "https://example.test/a",
            "text": "first",
            "document_index": 1,
            "passage_index": 0,
        },
        {
            "url": "https://example.test/b",
            "text": "other",
            "document_index": 2,
            "passage_index": 0,
        },
        {
            "url": "https://example.test/a",
            "text": "second",
            "document_index": 1,
            "passage_index": 1,
        },
        {
            "url": "https://example.test/a",
            "text": "first",
            "document_index": 3,
            "passage_index": 0,
        },
    ]
    documents = aggregate_averitec_documents(passages)

    assert [record["url"] for record in documents] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
    assert documents[0]["text"] == "first second"
    assert documents[0]["lexical_rank"] == 1
    assert documents[0]["document_passage_count"] == 2


def test_averitec_url_maxp_selects_best_passage_and_stable_ties() -> None:
    passages = [
        {"url": "https://a.example", "text": "A first."},
        {"url": "https://b.example", "text": "B first."},
        {"url": "https://a.example", "text": "A best."},
        {"url": "https://b.example", "text": "B tied."},
    ]

    assert select_averitec_url_maxp(passages, [0.1, 0.4, 0.8, 0.4]) == [2, 1]


def test_averitec_url_maxp_rejects_malformed_scores() -> None:
    passages = [{"url": "https://a.example", "text": "A."}]

    with pytest.raises(ValueError, match="aligned finite"):
        select_averitec_url_maxp(passages, [float("nan")])
