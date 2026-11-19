"""Append-only deterministic lexical retrieval baseline for all benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evidencegraph.averitec_store import AVERITEC_QUERY_PROTOCOL, AveritecIndex
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.retrieval import local_lexical_results
from evidencegraph.retrieval_evaluation import resolve_retrieval_attempts
from evidencegraph.wiki_index import WIKI_QUERY_PROTOCOL, WikiIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_FILES = {
    "fever_dev": "fever_dev.jsonl",
    "hover_dev": "hover_dev.jsonl",
    "averitec_dev": "averitec_dev.jsonl",
    "expertqa_domain_test": "expertqa_domain_test.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CASE_FILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    config = {"protocol": "frozen-lexical-only-v1", "top_k": args.top_k}
    if args.dataset == "averitec_dev":
        config = {
            "protocol": "frozen-lexical-only-v2",
            "top_k": args.top_k,
            "lexical_query_protocol": AVERITEC_QUERY_PROTOCOL,
        }
    elif args.dataset in {"fever_dev", "hover_dev"}:
        config = {
            "protocol": "frozen-lexical-only-v2",
            "top_k": args.top_k,
            "lexical_query_protocol": WIKI_QUERY_PROTOCOL,
        }
    all_cases = read_jsonl(PROJECT_ROOT / "data/benchmarks" / CASE_FILES[args.dataset])
    cases = all_cases
    if args.limit is not None:
        cases = cases[: args.limit]
    completed = set()
    if args.output.exists():
        previous = read_jsonl(args.output)
        expected_cases = {str(case["case_id"]): case for case in all_cases}
        if any(
            record.get("retrieval_config") != config
            or str(record.get("case_id", "")) not in expected_cases
            or record.get("dataset")
            != expected_cases.get(str(record.get("case_id", "")), {}).get("dataset")
            or record.get("query")
            != expected_cases.get(str(record.get("case_id", "")), {}).get("query")
            for record in previous
        ):
            raise ValueError(
                "lexical retrieval output contains a different configuration"
            )
        completed = {
            str(record["case_id"])
            for record in resolve_retrieval_attempts(
                previous, require_complete=False
            )
        }
    if args.dataset in {"fever_dev", "hover_dev"}:
        index = WikiIndex(PROJECT_ROOT / "data/processed/fever_wiki.sqlite")
    elif args.dataset == "averitec_dev":
        index = AveritecIndex(PROJECT_ROOT / "data/processed/averitec_dev.sqlite")
    else:
        index = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = failures = 0
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for case in cases:
                case_id = str(case["case_id"])
                if case_id in completed:
                    continue
                started = time.monotonic()
                base = {
                    "dataset": str(case["dataset"]),
                    "case_id": case_id,
                    "query": str(case["query"]),
                    "retrieval_protocol": config["protocol"],
                    "retrieval_config": config,
                }
                try:
                    if args.dataset in {"fever_dev", "hover_dev"}:
                        assert isinstance(index, WikiIndex)
                        results = index.search(str(case["query"]), limit=args.top_k)
                        for rank, result in enumerate(results, start=1):
                            result["source_id"] = (
                                f"{result['page_id']}:{result['line_number']}"
                            )
                            result["rank"] = rank
                            result["lexical_rank"] = rank
                    elif args.dataset == "averitec_dev":
                        assert isinstance(index, AveritecIndex)
                        results = index.search(
                            case_id, str(case["query"]), limit=args.top_k
                        )
                        for rank, result in enumerate(results, start=1):
                            result["source_id"] = (
                                f"averitec:{case_id}:{result['document_index']}:"
                                f"{result['passage_index']}"
                            )
                            result["rank"] = rank
                            result["lexical_rank"] = rank
                    else:
                        results = local_lexical_results(
                            str(case["query"]),
                            dict(case["metadata"]["evidence_passages"]),
                            limit=args.top_k,
                        )
                    output = {
                        **base,
                        "results": results,
                        "retrieved": len(results),
                        "wall_seconds": time.monotonic() - started,
                        "status": "complete",
                    }
                    written += 1
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    output = {
                        **base,
                        "status": "retrieval_failure",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:500],
                    }
                    failures += 1
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
    finally:
        if index is not None:
            index.close()
    print(json.dumps({"complete": written, "failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
