"""Append-only frozen retrieval for all four official benchmark manifests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evidencegraph.averitec_store import AVERITEC_QUERY_PROTOCOL
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.retrieval import (
    AveritecDocumentMaxpMmrRetriever,
    AveritecDocumentMmrRetriever,
    AveritecHybridRetriever,
    AveritecMmrRetriever,
    AVERITEC_MXBAI_MODEL,
    AVERITEC_MXBAI_REVISION,
    DENSE_MODEL,
    DENSE_REVISION,
    HybridRetriever,
    LocalPassageRetriever,
    RERANKER_MODEL,
    RERANKER_REVISION,
)
from evidencegraph.retrieval_evaluation import resolve_retrieval_attempts
from evidencegraph.wiki_index import WIKI_QUERY_PROTOCOL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_FILES = {
    "fever_dev": "fever_dev.jsonl",
    "hover_dev": "hover_dev.jsonl",
    "averitec_dev": "averitec_dev.jsonl",
    "expertqa_domain_test": "expertqa_domain_test.jsonl",
}


def resolve_candidate_depths(
    dataset: str,
    *,
    top_k: int,
    lexical_candidates: int | None,
    rerank_candidates: int,
) -> tuple[int, int]:
    lexical = lexical_candidates or (1000 if dataset == "averitec_dev" else 100)
    if top_k < 1 or lexical < top_k or rerank_candidates < top_k:
        raise ValueError("candidate depths must be at least positive top-k")
    return lexical, rerank_candidates


def select_case_window(
    cases: list[dict[str, object]], *, offset: int, limit: int | None
) -> list[dict[str, object]]:
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("case offset must be nonnegative and limit positive")
    stop = None if limit is None else offset + limit
    return cases[offset:stop]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CASE_FILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lexical-candidates", type=int)
    parser.add_argument("--rerank-candidates", type=int, default=30)
    parser.add_argument("--dense-model", default=DENSE_MODEL)
    parser.add_argument("--dense-revision", default=DENSE_REVISION)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--reranker-revision", default=RERANKER_REVISION)
    parser.add_argument("--averitec-mmr", action="store_true")
    parser.add_argument("--averitec-document-mmr", action="store_true")
    parser.add_argument("--averitec-document-maxp-mmr", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    lexical_candidates, rerank_candidates = resolve_candidate_depths(
        args.dataset,
        top_k=args.top_k,
        lexical_candidates=args.lexical_candidates,
        rerank_candidates=args.rerank_candidates,
    )
    averitec_special_modes = sum(
        (
            args.averitec_mmr,
            args.averitec_document_mmr,
            args.averitec_document_maxp_mmr,
        )
    )
    if averitec_special_modes and args.dataset != "averitec_dev":
        parser.error("AVeriTeC MMR flags are only valid for averitec_dev")
    if averitec_special_modes > 1:
        parser.error("choose only one AVeriTeC MMR protocol")
    if args.averitec_mmr and args.rerank_candidates == 30:
        rerank_candidates = 40
    if args.averitec_document_mmr and args.rerank_candidates == 30:
        rerank_candidates = 100
    if args.averitec_document_maxp_mmr and args.rerank_candidates == 30:
        rerank_candidates = 100
    use_averitec_mxbai = bool(averitec_special_modes)
    dense_model = AVERITEC_MXBAI_MODEL if use_averitec_mxbai else args.dense_model
    dense_revision = (
        AVERITEC_MXBAI_REVISION if use_averitec_mxbai else args.dense_revision
    )
    retrieval_config = {
        "protocol": (
            "bm25-mxbai-source-diverse-mmr-v5-finite-dot"
            if args.averitec_mmr
            else "bm25-mxbai-document-maxp-mmr-v8-finite-dot"
            if args.averitec_document_maxp_mmr
            else "bm25-mxbai-document-mmr-v7-finite-dot"
            if args.averitec_document_mmr
            else
            "lexical-candidate-e5-rrf-crossencoder-v4-finite-dot"
            if args.dataset in {"fever_dev", "hover_dev", "averitec_dev"}
            else "lexical-candidate-e5-rrf-crossencoder-v3-finite-dot"
        ),
        "dataset": args.dataset,
        "lexical_query_protocol": (
            AVERITEC_QUERY_PROTOCOL
            if args.dataset == "averitec_dev"
            else WIKI_QUERY_PROTOCOL
            if args.dataset in {"fever_dev", "hover_dev"}
            else None
        ),
        "top_k": args.top_k,
        "lexical_candidates": lexical_candidates,
        "rerank_candidates": rerank_candidates,
        "dense_model": dense_model,
        "dense_revision": dense_revision,
        "reranker_model": None if use_averitec_mxbai else args.reranker_model,
        "reranker_revision": None if use_averitec_mxbai else args.reranker_revision,
        "device": args.device,
    }
    if use_averitec_mxbai:
        retrieval_config["mmr_lambda"] = 0.75
    if args.averitec_document_mmr:
        retrieval_config["document_passages"] = 3
        retrieval_config["document_characters"] = 2400
    if args.offset:
        retrieval_config["case_offset"] = args.offset
    all_cases = read_jsonl(PROJECT_ROOT / "data/benchmarks" / CASE_FILES[args.dataset])
    cases = select_case_window(all_cases, offset=args.offset, limit=args.limit)
    completed = set()
    if args.output.exists():
        previous = read_jsonl(args.output)
        expected_cases = {str(case["case_id"]): case for case in all_cases}
        incompatible = [
            record
            for record in previous
            if record.get("retrieval_config") != retrieval_config
            or str(record.get("case_id", "")) not in expected_cases
            or record.get("dataset")
            != expected_cases.get(str(record.get("case_id", "")), {}).get("dataset")
            or record.get("query")
            != expected_cases.get(str(record.get("case_id", "")), {}).get("query")
        ]
        if incompatible:
            raise ValueError(
                "retrieval output contains completions from a different configuration"
            )
        completed = {
            str(record["case_id"])
            for record in resolve_retrieval_attempts(
                previous, require_complete=False
            )
        }
    pending = [case for case in cases if str(case["case_id"]) not in completed]
    if not pending:
        print(json.dumps({"complete": len(completed), "new": 0}, sort_keys=True))
        return
    if args.dataset in {"fever_dev", "hover_dev"}:
        retriever = HybridRetriever(
            PROJECT_ROOT / "data/processed/fever_wiki.sqlite",
            PROJECT_ROOT / "data/processed/retrieval_wikipedia.sqlite",
            dense_model=args.dense_model,
            dense_revision=args.dense_revision,
            reranker_model=args.reranker_model,
            reranker_revision=args.reranker_revision,
            device=args.device,
        )
    elif args.dataset == "averitec_dev":
        retriever_class = (
            AveritecDocumentMaxpMmrRetriever
            if args.averitec_document_maxp_mmr
            else AveritecDocumentMmrRetriever
            if args.averitec_document_mmr
            else AveritecMmrRetriever
            if args.averitec_mmr
            else AveritecHybridRetriever
        )
        retriever_kwargs = {
            "dense_model": dense_model,
            "dense_revision": dense_revision,
            "device": args.device,
        }
        if not use_averitec_mxbai:
            retriever_kwargs.update(
                {
                    "reranker_model": args.reranker_model,
                    "reranker_revision": args.reranker_revision,
                }
            )
        retriever = retriever_class(
            PROJECT_ROOT / "data/processed/averitec_dev.sqlite",
            PROJECT_ROOT / "data/processed/retrieval_averitec.sqlite",
            **retriever_kwargs,
        )
    else:
        retriever = LocalPassageRetriever(
            dense_model=args.dense_model,
            dense_revision=args.dense_revision,
            reranker_model=args.reranker_model,
            reranker_revision=args.reranker_revision,
            device=args.device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = failures = 0
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for case in pending:
                started = time.monotonic()
                try:
                    if args.dataset == "averitec_dev":
                        results = retriever.search(
                            str(case["case_id"]),
                            str(case["query"]),
                            limit=args.top_k,
                            lexical_candidates=lexical_candidates,
                            rerank_candidates=rerank_candidates,
                        )
                    elif args.dataset == "expertqa_domain_test":
                        results = retriever.search(
                            str(case["query"]),
                            dict(case["metadata"]["evidence_passages"]),
                            limit=args.top_k,
                        )
                    else:
                        results = retriever.search(
                            str(case["query"]),
                            limit=args.top_k,
                            lexical_candidates=lexical_candidates,
                            rerank_candidates=rerank_candidates,
                        )
                        for result in results:
                            result["source_id"] = (
                                f"{result['page_id']}:{result['line_number']}"
                            )
                    output = {
                        "dataset": case["dataset"],
                        "case_id": str(case["case_id"]),
                        "query": str(case["query"]),
                        "retrieval_protocol": retrieval_config["protocol"],
                        "retrieval_config": retrieval_config,
                        "results": results,
                        "retrieved": len(results),
                        "wall_seconds": time.monotonic() - started,
                        "status": "complete",
                    }
                    written += 1
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    output = {
                        "dataset": case["dataset"],
                        "case_id": str(case["case_id"]),
                        "query": str(case["query"]),
                        "retrieval_protocol": retrieval_config["protocol"],
                        "retrieval_config": retrieval_config,
                        "status": "retrieval_failure",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:500],
                    }
                    failures += 1
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
    finally:
        close = getattr(retriever, "close", None)
        if close is not None:
            close()
    print(
        json.dumps(
            {"complete": len(completed) + written, "new": written, "failures": failures},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
