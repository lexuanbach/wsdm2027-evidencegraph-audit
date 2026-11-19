"""Frozen lexical+dense retrieval and cross-encoder reranking."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .averitec_store import AVERITEC_QUERY_PROTOCOL, AveritecIndex
from .wiki_index import WIKI_QUERY_PROTOCOL, WikiIndex


DENSE_MODEL = "intfloat/e5-large-v2"
DENSE_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
AVERITEC_MXBAI_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
AVERITEC_MXBAI_REVISION = "b33106f585b9ce46904ad7443a3b52b7a63e231c"
AVERITEC_MXBAI_QUERY_PROMPT = (
    "Represent this sentence for searching relevant passages: "
)


def finite_dense_scores(embeddings: Any) -> np.ndarray:
    """Compute query--passage dot products without Apple BLAS matmul faults."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("dense embeddings must contain one query and passages")
    if not np.isfinite(matrix).all():
        raise RuntimeError("dense encoder returned non-finite embeddings")
    scores = np.einsum("ij,j->i", matrix[1:], matrix[0], optimize=False)
    if scores.shape != (matrix.shape[0] - 1,) or not np.isfinite(scores).all():
        raise RuntimeError("dense similarity produced non-finite scores")
    return scores


def finite_cross_scores(scores: Any, expected_count: int) -> np.ndarray:
    """Normalize reranker output and reject malformed or non-finite scores."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if expected_count < 1 or values.shape != (expected_count,):
        raise RuntimeError("cross-encoder returned the wrong score cardinality")
    if not np.isfinite(values).all():
        raise RuntimeError("cross-encoder returned non-finite scores")
    return values


def finite_multiquery_dense_scores(
    embeddings: Any, query_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return each passage's best finite score and matching query index."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if (
        matrix.ndim != 2
        or query_count < 1
        or matrix.shape[0] <= query_count
        or matrix.shape[1] < 1
    ):
        raise ValueError("multi-query embeddings must contain queries and passages")
    if not np.isfinite(matrix).all():
        raise RuntimeError("multi-query encoder returned non-finite embeddings")
    similarities = np.einsum(
        "qd,pd->qp", matrix[:query_count], matrix[query_count:], optimize=False
    )
    if not np.isfinite(similarities).all():
        raise RuntimeError("multi-query similarity produced non-finite scores")
    best_queries = np.argmax(similarities, axis=0).astype(np.int64)
    best_scores = np.max(similarities, axis=0)
    return best_scores, best_queries


def merge_query_candidates(
    rankings: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Union query-specific AVeriTeC hits without losing per-query ranks."""

    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    query_count = len(rankings)
    if query_count < 1:
        raise ValueError("at least one query ranking is required")
    for query_index, ranking in enumerate(rankings):
        seen_in_query: set[str] = set()
        for rank, record in enumerate(ranking, start=1):
            identifier = (
                f"{record.get('claim_id')}:{record.get('document_index')}:"
                f"{record.get('passage_index')}"
            )
            if identifier in seen_in_query:
                raise ValueError("query ranking contains duplicate passage IDs")
            seen_in_query.add(identifier)
            if identifier not in positions:
                copied = dict(record)
                copied["query_lexical_ranks"] = [None] * query_count
                copied["query_lexical_ranks"][query_index] = rank
                positions[identifier] = len(merged)
                merged.append(copied)
            else:
                existing = merged[positions[identifier]]
                existing["query_lexical_ranks"][query_index] = rank
    return merged


def aggregate_averitec_documents(
    passages: list[dict[str, Any]],
    *,
    maximum_passages: int = 3,
    maximum_characters: int = 2400,
) -> list[dict[str, Any]]:
    """Aggregate ranked passages into deterministic URL-level document blocks."""

    if maximum_passages < 1 or maximum_characters < 100:
        raise ValueError("invalid AVeriTeC document aggregation limits")
    documents: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    seen_texts: dict[str, set[str]] = {}
    for lexical_rank, passage in enumerate(passages, start=1):
        url = str(passage.get("url", "")).strip()
        text = str(passage.get("text", "")).strip()
        if not url or not text:
            raise ValueError("AVeriTeC passage lacks URL or text")
        if url not in positions:
            record = dict(passage)
            record.update(
                {
                    "text": "",
                    "lexical_rank": lexical_rank,
                    "document_indices": [],
                    "passage_indices": [],
                    "document_passage_count": 0,
                }
            )
            positions[url] = len(documents)
            documents.append(record)
            seen_texts[url] = set()
        document = documents[positions[url]]
        if (
            document["document_passage_count"] >= maximum_passages
            or text.casefold() in seen_texts[url]
        ):
            continue
        proposed = f"{document['text']} {text}".strip()
        if document["text"] and len(proposed) > maximum_characters:
            continue
        document["text"] = proposed[:maximum_characters]
        document["document_passage_count"] += 1
        document["document_indices"].append(int(passage["document_index"]))
        document["passage_indices"].append(int(passage["passage_index"]))
        seen_texts[url].add(text.casefold())
    return documents


def select_averitec_url_maxp(
    passages: list[dict[str, Any]], scores: Any
) -> list[int]:
    """Select each URL's highest-scoring passage with stable lexical ties."""

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(passages) != len(values) or not np.isfinite(values).all():
        raise ValueError("AVeriTeC MaxP requires aligned finite passage scores")
    best: dict[str, int] = {}
    for index, passage in enumerate(passages):
        url = str(passage.get("url", "")).strip()
        text = str(passage.get("text", "")).strip()
        if not url or not text:
            raise ValueError("AVeriTeC MaxP passage lacks URL or text")
        current = best.get(url)
        if current is None or float(values[index]) > float(values[current]):
            best[url] = index
    return list(best.values())


def lexical_overlap_order(query: str, passages: list[str]) -> list[int]:
    """Deterministic lexical order for small case-local candidate pools."""
    query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
    scores = []
    for index, passage in enumerate(passages):
        passage_tokens = re.findall(r"[a-z0-9]+", passage.casefold())
        overlap = sum(token in query_tokens for token in passage_tokens)
        scores.append(overlap / max(1.0, np.sqrt(len(passage_tokens))))
    return sorted(range(len(passages)), key=lambda index: (-scores[index], index))


def local_lexical_results(
    query: str, passages: dict[str, str], *, limit: int
) -> list[dict[str, Any]]:
    """Rank an immutable case-local passage pool without loading neural models."""
    if limit < 1:
        raise ValueError("limit must be positive")
    identifiers = list(passages)
    texts = [str(passages[identifier]) for identifier in identifiers]
    order = lexical_overlap_order(query, texts)[:limit]
    return [
        {
            "source_id": identifiers[index],
            "text": texts[index],
            "rank": rank,
            "lexical_rank": rank,
        }
        for rank, index in enumerate(order, start=1)
    ]


def reciprocal_rank_fusion(
    identifiers: list[str],
    lexical_order: list[int],
    dense_order: list[int],
    constant: int = 60,
) -> list[int]:
    if set(lexical_order) != set(dense_order) or set(lexical_order) != set(range(len(identifiers))):
        raise ValueError("rankings must be permutations over every identifier")
    lexical_rank = {item: rank for rank, item in enumerate(lexical_order, start=1)}
    dense_rank = {item: rank for rank, item in enumerate(dense_order, start=1)}
    return sorted(
        range(len(identifiers)),
        key=lambda item: -(
            1.0 / (constant + lexical_rank[item]) + 1.0 / (constant + dense_rank[item])
        ),
    )


def reciprocal_rank_fusion_union(
    rankings: list[list[str]], constant: int = 60
) -> list[tuple[str, float]]:
    """Fuse partially overlapping rankings while retaining unseen evidence."""

    if constant < 1 or not rankings:
        raise ValueError("RRF union requires rankings and a positive constant")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("RRF union rankings cannot contain duplicate IDs")
        for rank, identifier in enumerate(ranking, start=1):
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("RRF union IDs must be non-empty strings")
            if identifier not in first_seen:
                first_seen[identifier] = seen_counter
                seen_counter += 1
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (
                constant + rank
            )
    return sorted(
        scores.items(), key=lambda item: (-item[1], first_seen[item[0]])
    )


def maximal_marginal_relevance(
    query_scores: Any,
    passage_embeddings: Any,
    candidate_order: list[int],
    *,
    limit: int,
    relevance_weight: float = 0.75,
    group_ids: list[str] | None = None,
) -> list[tuple[int, float]]:
    """Select relevant, nonredundant candidates with deterministic ties."""

    scores = np.asarray(query_scores, dtype=np.float32).reshape(-1)
    embeddings = np.asarray(passage_embeddings, dtype=np.float32)
    # ``np.argsort`` yields NumPy integer scalars.  Normalize the public
    # boundary here so ranks returned by this function remain JSON-safe.
    candidate_order = [int(index) for index in candidate_order]
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != scores.shape[0]
        or not np.isfinite(embeddings).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("MMR requires aligned finite scores and embeddings")
    if not 0.0 <= relevance_weight <= 1.0 or limit < 1:
        raise ValueError("invalid MMR weight or limit")
    if len(candidate_order) != len(set(candidate_order)) or any(
        index < 0 or index >= len(scores) for index in candidate_order
    ):
        raise ValueError("MMR candidate order contains invalid indices")
    if group_ids is not None and len(group_ids) != len(scores):
        raise ValueError("MMR group IDs must align with scores")

    selected: list[int] = []
    selected_groups: set[str] = set()
    remaining = list(candidate_order)
    outputs: list[tuple[int, float]] = []
    while remaining and len(selected) < min(limit, len(candidate_order)):
        eligible = (
            [index for index in remaining if group_ids[index] not in selected_groups]
            if group_ids is not None
            else list(remaining)
        )
        # Preserve the requested result depth if the prefetch contains fewer
        # unique sources than passages, but exhaust unique sources first.
        if not eligible:
            eligible = list(remaining)
        ranked: list[tuple[float, int, int]] = []
        for index in eligible:
            redundancy = (
                max(
                    float(np.dot(embeddings[index], embeddings[chosen]))
                    for chosen in selected
                )
                if selected
                else 0.0
            )
            mmr_score = relevance_weight * float(scores[index]) - (
                1.0 - relevance_weight
            ) * redundancy
            ranked.append((mmr_score, -candidate_order.index(index), -index))
        best_score, _negative_rank, negative_index = max(ranked)
        best = -negative_index
        selected.append(best)
        outputs.append((best, best_score))
        remaining.remove(best)
        if group_ids is not None:
            selected_groups.add(group_ids[best])
    return outputs


class RetrievalCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrievals (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                config_json TEXT NOT NULL,
                results_json TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def key(query: str, config: dict[str, Any]) -> str:
        payload = json.dumps({"query": query, "config": config}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, query: str, config: dict[str, Any]) -> list[dict[str, Any]] | None:
        row = self.connection.execute(
            "SELECT results_json FROM retrievals WHERE cache_key = ?", (self.key(query, config),)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, query: str, config: dict[str, Any], results: list[dict[str, Any]]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO retrievals VALUES (?, ?, ?, ?)",
                (
                    self.key(query, config),
                    query,
                    json.dumps(config, sort_keys=True),
                    json.dumps(results, ensure_ascii=False, sort_keys=True),
                ),
            )

    def close(self) -> None:
        self.connection.close()


class HybridRetriever:
    def __init__(
        self,
        wiki_index: Path,
        cache_path: Path,
        dense_model: str = DENSE_MODEL,
        dense_revision: str = DENSE_REVISION,
        reranker_model: str = RERANKER_MODEL,
        reranker_revision: str = RERANKER_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.index = WikiIndex(wiki_index)
        self.cache = RetrievalCache(cache_path)
        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.reranker_model_name = reranker_model
        self.reranker_revision = reranker_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )
        self.reranker = CrossEncoder(
            reranker_model,
            revision=reranker_revision,
            device=device,
            local_files_only=True,
        )

    def close(self) -> None:
        self.index.close()
        self.cache.close()

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        lexical_candidates: int = 100,
        rerank_candidates: int = 30,
    ) -> list[dict[str, Any]]:
        config = {
            "version": 4,
            "lexical_query_protocol": WIKI_QUERY_PROTOCOL,
            "dense_model": self.dense_model_name,
            "dense_revision": self.dense_revision,
            "reranker_model": self.reranker_model_name,
            "reranker_revision": self.reranker_revision,
            "limit": limit,
            "lexical_candidates": lexical_candidates,
            "rerank_candidates": rerank_candidates,
        }
        cached = self.cache.get(query, config)
        if cached is not None:
            return cached
        lexical = self.index.search(query, limit=lexical_candidates)
        if not lexical:
            return []
        texts = [str(record["text"]) for record in lexical]
        embeddings = self.encoder.encode(
            [f"query: {query}"] + [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores = finite_dense_scores(embeddings)
        dense_order = list(np.argsort(-dense_scores))
        lexical_order = list(range(len(lexical)))
        identifiers = [f"{record['page_id']}:{record['line_number']}" for record in lexical]
        fused = reciprocal_rank_fusion(identifiers, lexical_order, dense_order)
        candidates = fused[: min(rerank_candidates, len(fused))]
        cross_scores = finite_cross_scores(self.reranker.predict(
            [(query, texts[index]) for index in candidates], show_progress_bar=False
        ), len(candidates))
        candidate_scores = {
            index: float(score) for index, score in zip(candidates, cross_scores)
        }
        final_order = sorted(
            candidates, key=lambda index: (-candidate_scores[index], index)
        )[:limit]
        results = []
        for rank, index in enumerate(final_order, start=1):
            record = dict(lexical[index])
            record.update(
                {
                    "rank": rank,
                    "dense_score": float(dense_scores[index]),
                    "cross_score": candidate_scores[index],
                    "lexical_rank": index + 1,
                }
            )
            results.append(record)
        self.cache.put(query, config, results)
        return results


class AveritecHybridRetriever:
    """Claim-scoped hybrid retrieval over the official AVeriTeC web store."""

    def __init__(
        self,
        store_index: Path,
        cache_path: Path,
        dense_model: str = DENSE_MODEL,
        dense_revision: str = DENSE_REVISION,
        reranker_model: str = RERANKER_MODEL,
        reranker_revision: str = RERANKER_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.index = AveritecIndex(store_index)
        self.cache = RetrievalCache(cache_path)
        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.reranker_model_name = reranker_model
        self.reranker_revision = reranker_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )
        self.reranker = CrossEncoder(
            reranker_model,
            revision=reranker_revision,
            device=device,
            local_files_only=True,
        )

    def close(self) -> None:
        self.index.close()
        self.cache.close()

    def search(
        self,
        claim_id: str,
        query: str,
        *,
        limit: int = 20,
        lexical_candidates: int = 100,
        rerank_candidates: int = 30,
    ) -> list[dict[str, Any]]:
        config = {
            "version": 4,
            "dataset": "averitec_dev_store",
            "claim_id": str(claim_id),
            "lexical_query_protocol": AVERITEC_QUERY_PROTOCOL,
            "dense_model": self.dense_model_name,
            "dense_revision": self.dense_revision,
            "reranker_model": self.reranker_model_name,
            "reranker_revision": self.reranker_revision,
            "limit": limit,
            "lexical_candidates": lexical_candidates,
            "rerank_candidates": rerank_candidates,
        }
        cached = self.cache.get(query, config)
        if cached is not None:
            return cached
        lexical = self.index.search(str(claim_id), query, limit=lexical_candidates)
        if not lexical:
            return []
        texts = [str(record["text"]) for record in lexical]
        embeddings = self.encoder.encode(
            [f"query: {query}"] + [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores = finite_dense_scores(embeddings)
        dense_order = list(np.argsort(-dense_scores))
        lexical_order = list(range(len(lexical)))
        identifiers = [
            f"averitec:{claim_id}:{record['document_index']}:{record['passage_index']}"
            for record in lexical
        ]
        fused = reciprocal_rank_fusion(identifiers, lexical_order, dense_order)
        candidates = fused[: min(rerank_candidates, len(fused))]
        cross_scores = finite_cross_scores(self.reranker.predict(
            [(query, texts[index]) for index in candidates], show_progress_bar=False
        ), len(candidates))
        candidate_scores = {
            index: float(score) for index, score in zip(candidates, cross_scores)
        }
        final_order = sorted(
            candidates, key=lambda index: (-candidate_scores[index], index)
        )[:limit]
        results = []
        for rank, index in enumerate(final_order, start=1):
            record = dict(lexical[index])
            record.update(
                {
                    "source_id": identifiers[index],
                    "rank": rank,
                    "dense_score": float(dense_scores[index]),
                    "cross_score": candidate_scores[index],
                    "lexical_rank": index + 1,
                }
            )
            results.append(record)
        self.cache.put(query, config, results)
        return results


class AveritecMmrRetriever:
    """Claim-scoped Mixedbread dense retrieval with source-diverse MMR."""

    def __init__(
        self,
        store_index: Path,
        cache_path: Path,
        dense_model: str = AVERITEC_MXBAI_MODEL,
        dense_revision: str = AVERITEC_MXBAI_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self.index = AveritecIndex(store_index)
        self.cache = RetrievalCache(cache_path)
        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )

    def close(self) -> None:
        self.index.close()
        self.cache.close()

    def search(
        self,
        claim_id: str,
        query: str,
        *,
        limit: int = 20,
        lexical_candidates: int = 1000,
        rerank_candidates: int = 40,
    ) -> list[dict[str, Any]]:
        config = {
            "version": 5,
            "dataset": "averitec_dev_store",
            "claim_id": str(claim_id),
            "lexical_query_protocol": AVERITEC_QUERY_PROTOCOL,
            "dense_model": self.dense_model_name,
            "dense_revision": self.dense_revision,
            "query_prompt": AVERITEC_MXBAI_QUERY_PROMPT,
            "reranker": "source-diverse-mmr",
            "mmr_lambda": 0.75,
            "limit": limit,
            "lexical_candidates": lexical_candidates,
            "rerank_candidates": rerank_candidates,
        }
        cached = self.cache.get(query, config)
        if cached is not None:
            return cached
        lexical = self.index.search(str(claim_id), query, limit=lexical_candidates)
        if not lexical:
            return []
        texts = [str(record["text"]) for record in lexical]
        embeddings = self.encoder.encode(
            [AVERITEC_MXBAI_QUERY_PROMPT + query] + texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores = finite_dense_scores(embeddings)
        dense_order = list(np.argsort(-dense_scores))
        candidates = dense_order[: min(rerank_candidates, len(dense_order))]
        groups = [str(record["url"]) for record in lexical]
        selected = maximal_marginal_relevance(
            dense_scores,
            embeddings[1:],
            candidates,
            limit=limit,
            relevance_weight=0.75,
            group_ids=groups,
        )
        results = []
        for rank, (index, mmr_score) in enumerate(selected, start=1):
            record = dict(lexical[index])
            record.update(
                {
                    "source_id": (
                        f"averitec:{claim_id}:{record['document_index']}:"
                        f"{record['passage_index']}"
                    ),
                    "rank": rank,
                    "dense_score": float(dense_scores[index]),
                    "mmr_score": float(mmr_score),
                    "lexical_rank": index + 1,
                }
            )
            results.append(record)
        self.cache.put(query, config, results)
        return results


class AveritecMultiQueryMmrRetriever:
    """Claim plus generated-question retrieval with local Mixedbread and MMR."""

    def __init__(
        self,
        store_index: Path,
        cache_path: Path,
        dense_model: str = AVERITEC_MXBAI_MODEL,
        dense_revision: str = AVERITEC_MXBAI_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self.index = AveritecIndex(store_index)
        self.cache = RetrievalCache(cache_path)
        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )

    def close(self) -> None:
        self.index.close()
        self.cache.close()

    def search(
        self,
        claim_id: str,
        claim: str,
        questions: list[str],
        *,
        limit: int = 20,
        per_query_candidates: int = 200,
        rerank_candidates: int = 100,
        query_generation_protocol: str,
    ) -> list[dict[str, Any]]:
        if (
            not questions
            or len(questions) != len({question.casefold() for question in questions})
            or limit < 1
            or per_query_candidates < limit
            or rerank_candidates < limit
        ):
            raise ValueError("invalid multi-query retrieval inputs")
        queries = [claim, *questions]
        config = {
            "version": 6,
            "dataset": "averitec_dev_store",
            "claim_id": str(claim_id),
            "questions": questions,
            "query_generation_protocol": query_generation_protocol,
            "lexical_query_protocol": AVERITEC_QUERY_PROTOCOL,
            "dense_model": self.dense_model_name,
            "dense_revision": self.dense_revision,
            "query_prompt": AVERITEC_MXBAI_QUERY_PROMPT,
            "reranker": "multi-query-source-diverse-mmr",
            "mmr_lambda": 0.75,
            "limit": limit,
            "per_query_candidates": per_query_candidates,
            "rerank_candidates": rerank_candidates,
        }
        cached = self.cache.get(claim, config)
        if cached is not None:
            return cached
        query_rankings = [
            self.index.search(str(claim_id), query, limit=per_query_candidates)
            for query in queries
        ]
        candidates = merge_query_candidates(query_rankings)
        if not candidates:
            return []
        texts = [str(record["text"]) for record in candidates]
        embeddings = self.encoder.encode(
            [AVERITEC_MXBAI_QUERY_PROMPT + query for query in queries] + texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores, best_queries = finite_multiquery_dense_scores(
            embeddings, len(queries)
        )
        dense_order = [int(index) for index in np.argsort(-dense_scores)]
        prefetch = dense_order[: min(rerank_candidates, len(dense_order))]
        groups = [str(record["url"]) for record in candidates]
        selected = maximal_marginal_relevance(
            dense_scores,
            embeddings[len(queries) :],
            prefetch,
            limit=limit,
            relevance_weight=0.75,
            group_ids=groups,
        )
        results = []
        for rank, (index, mmr_score) in enumerate(selected, start=1):
            record = dict(candidates[index])
            query_ranks = list(record.pop("query_lexical_ranks"))
            record.update(
                {
                    "source_id": (
                        f"averitec:{claim_id}:{record['document_index']}:"
                        f"{record['passage_index']}"
                    ),
                    "rank": rank,
                    "dense_score": float(dense_scores[index]),
                    "mmr_score": float(mmr_score),
                    "matched_query_index": int(best_queries[index]),
                    "query_lexical_ranks": query_ranks,
                    "lexical_rank": min(value for value in query_ranks if value is not None),
                }
            )
            results.append(record)
        self.cache.put(claim, config, results)
        return results


class AveritecDocumentMmrRetriever:
    """Document-first AVeriTeC retrieval with local Mixedbread and MMR."""

    def __init__(
        self,
        store_index: Path,
        cache_path: Path,
        dense_model: str = AVERITEC_MXBAI_MODEL,
        dense_revision: str = AVERITEC_MXBAI_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self.index = AveritecIndex(store_index)
        self.cache = RetrievalCache(cache_path)
        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )

    def close(self) -> None:
        self.index.close()
        self.cache.close()

    def search(
        self,
        claim_id: str,
        query: str,
        *,
        limit: int = 20,
        lexical_candidates: int = 1000,
        rerank_candidates: int = 100,
    ) -> list[dict[str, Any]]:
        config = {
            "version": 7,
            "dataset": "averitec_dev_store",
            "claim_id": str(claim_id),
            "lexical_query_protocol": AVERITEC_QUERY_PROTOCOL,
            "dense_model": self.dense_model_name,
            "dense_revision": self.dense_revision,
            "query_prompt": AVERITEC_MXBAI_QUERY_PROMPT,
            "reranker": "url-document-source-diverse-mmr",
            "mmr_lambda": 0.75,
            "document_passages": 3,
            "document_characters": 2400,
            "limit": limit,
            "lexical_candidates": lexical_candidates,
            "rerank_candidates": rerank_candidates,
        }
        cached = self.cache.get(query, config)
        if cached is not None:
            return cached
        lexical = self.index.search(str(claim_id), query, limit=lexical_candidates)
        if not lexical:
            return []
        documents = aggregate_averitec_documents(lexical)
        texts = [str(record["text"]) for record in documents]
        embeddings = self.encoder.encode(
            [AVERITEC_MXBAI_QUERY_PROMPT + query] + texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores = finite_dense_scores(embeddings)
        dense_order = [int(index) for index in np.argsort(-dense_scores)]
        candidates = dense_order[: min(rerank_candidates, len(dense_order))]
        selected = maximal_marginal_relevance(
            dense_scores,
            embeddings[1:],
            candidates,
            limit=limit,
            relevance_weight=0.75,
        )
        results = []
        for rank, (index, mmr_score) in enumerate(selected, start=1):
            record = dict(documents[index])
            record.update(
                {
                    "source_id": f"averitec:{claim_id}:url:{index}",
                    "rank": rank,
                    "dense_score": float(dense_scores[index]),
                    "mmr_score": float(mmr_score),
                }
            )
            results.append(record)
        self.cache.put(query, config, results)
        return results


class AveritecDocumentMaxpMmrRetriever:
    """URL-level MaxP retrieval retaining each winning passage before MMR."""

    def __init__(
        self,
        store_index: Path,
        cache_path: Path,
        dense_model: str = AVERITEC_MXBAI_MODEL,
        dense_revision: str = AVERITEC_MXBAI_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self.index = AveritecIndex(store_index)
        self.cache = RetrievalCache(cache_path)
        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )

    def close(self) -> None:
        self.index.close()
        self.cache.close()

    def search(
        self,
        claim_id: str,
        query: str,
        *,
        limit: int = 20,
        lexical_candidates: int = 1000,
        rerank_candidates: int = 100,
    ) -> list[dict[str, Any]]:
        config = {
            "version": 8,
            "dataset": "averitec_dev_store",
            "claim_id": str(claim_id),
            "lexical_query_protocol": AVERITEC_QUERY_PROTOCOL,
            "dense_model": self.dense_model_name,
            "dense_revision": self.dense_revision,
            "query_prompt": AVERITEC_MXBAI_QUERY_PROMPT,
            "reranker": "url-maxp-source-diverse-mmr",
            "mmr_lambda": 0.75,
            "limit": limit,
            "lexical_candidates": lexical_candidates,
            "rerank_candidates": rerank_candidates,
        }
        cached = self.cache.get(query, config)
        if cached is not None:
            return cached
        lexical = self.index.search(str(claim_id), query, limit=lexical_candidates)
        if not lexical:
            return []
        texts = [str(record["text"]) for record in lexical]
        embeddings = self.encoder.encode(
            [AVERITEC_MXBAI_QUERY_PROMPT + query] + texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        passage_scores = finite_dense_scores(embeddings)
        best_indices = select_averitec_url_maxp(lexical, passage_scores)
        document_scores = passage_scores[best_indices]
        document_embeddings = np.asarray(embeddings[1:])[best_indices]
        dense_order = [int(index) for index in np.argsort(-document_scores)]
        candidates = dense_order[: min(rerank_candidates, len(dense_order))]
        selected = maximal_marginal_relevance(
            document_scores,
            document_embeddings,
            candidates,
            limit=limit,
            relevance_weight=0.75,
        )
        results = []
        for rank, (document_index, mmr_score) in enumerate(selected, start=1):
            passage_index = best_indices[document_index]
            record = dict(lexical[passage_index])
            record.update(
                {
                    "source_id": (
                        f"averitec:{claim_id}:url-maxp:{record['document_index']}:"
                        f"{record['passage_index']}"
                    ),
                    "rank": rank,
                    "dense_score": float(document_scores[document_index]),
                    "mmr_score": float(mmr_score),
                    "lexical_rank": passage_index + 1,
                }
            )
            results.append(record)
        self.cache.put(query, config, results)
        return results


class LocalPassageRetriever:
    """Hybrid ranking for ExpertQA's official per-question evidence pool."""

    def __init__(
        self,
        dense_model: str = DENSE_MODEL,
        dense_revision: str = DENSE_REVISION,
        reranker_model: str = RERANKER_MODEL,
        reranker_revision: str = RERANKER_REVISION,
        device: str = "cpu",
    ):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.dense_model_name = dense_model
        self.dense_revision = dense_revision
        self.reranker_model_name = reranker_model
        self.reranker_revision = reranker_revision
        self.encoder = SentenceTransformer(
            dense_model, revision=dense_revision, device=device, local_files_only=True
        )
        self.reranker = CrossEncoder(
            reranker_model,
            revision=reranker_revision,
            device=device,
            local_files_only=True,
        )

    def search(
        self,
        query: str,
        passages: dict[str, str],
        *,
        limit: int = 20,
        rerank_candidates: int = 30,
    ) -> list[dict[str, Any]]:
        identifiers = list(passages)
        texts = [str(passages[identifier]) for identifier in identifiers]
        if not texts:
            return []
        lexical_order = lexical_overlap_order(query, texts)
        embeddings = self.encoder.encode(
            [f"query: {query}"] + [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores = finite_dense_scores(embeddings)
        dense_order = list(np.argsort(-dense_scores))
        fused = reciprocal_rank_fusion(identifiers, lexical_order, dense_order)
        candidates = fused[: min(rerank_candidates, len(fused))]
        cross_scores = finite_cross_scores(self.reranker.predict(
            [(query, texts[index]) for index in candidates], show_progress_bar=False
        ), len(candidates))
        candidate_scores = {
            index: float(score) for index, score in zip(candidates, cross_scores)
        }
        final_order = sorted(
            candidates, key=lambda index: (-candidate_scores[index], index)
        )[:limit]
        return [
            {
                "source_id": identifiers[index],
                "text": texts[index],
                "rank": rank,
                "lexical_rank": lexical_order.index(index) + 1,
                "dense_score": float(dense_scores[index]),
                "cross_score": candidate_scores[index],
            }
            for rank, index in enumerate(final_order, start=1)
        ]
