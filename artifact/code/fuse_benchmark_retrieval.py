"""Fuse a complete lexical run with a source-diverse neural retrieval run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.provenance import sha256
from evidencegraph.retrieval import reciprocal_rank_fusion_union


PROTOCOL = "bm25-mxbai-source-diverse-mmr-rrf-v6"


def _ranked_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("status") != "complete":
        raise ValueError(f"case {record.get('case_id')} is not complete")
    results = sorted(record.get("results", []), key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in results]
    if ranks != list(range(1, len(results) + 1)):
        raise ValueError(f"case {record.get('case_id')} has malformed ranks")
    identifiers = [str(item.get("source_id", "")) for item in results]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError(f"case {record.get('case_id')} has malformed source IDs")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--constant", type=int, default=60)
    args = parser.parse_args()
    if args.top_k < 1 or args.constant < 1:
        raise ValueError("top-k and RRF constant must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    lexical_rows = read_jsonl(args.lexical)
    neural_rows = read_jsonl(args.neural)
    lexical_by_id = {str(row.get("case_id", "")): row for row in lexical_rows}
    if len(lexical_by_id) != len(lexical_rows):
        raise ValueError("lexical input has duplicate or blank case IDs")

    config = {
        "protocol": PROTOCOL,
        "fusion": "reciprocal-rank-fusion-union",
        "fusion_constant": args.constant,
        "top_k": args.top_k,
        "lexical_input_sha256": sha256(args.lexical),
        "neural_input_sha256": sha256(args.neural),
    }
    output_rows = []
    seen_cases: set[str] = set()
    for neural in neural_rows:
        case_id = str(neural.get("case_id", ""))
        if not case_id or case_id in seen_cases or case_id not in lexical_by_id:
            raise ValueError(f"invalid neural case ID: {case_id!r}")
        seen_cases.add(case_id)
        lexical = lexical_by_id[case_id]
        if (lexical.get("dataset"), lexical.get("query")) != (
            neural.get("dataset"),
            neural.get("query"),
        ):
            raise ValueError(f"case identity mismatch for {case_id}")
        lexical_results = _ranked_results(lexical)
        neural_results = _ranked_results(neural)
        lexical_ids = [str(item["source_id"]) for item in lexical_results]
        neural_ids = [str(item["source_id"]) for item in neural_results]
        lexical_rank = {identifier: rank for rank, identifier in enumerate(lexical_ids, 1)}
        neural_rank = {identifier: rank for rank, identifier in enumerate(neural_ids, 1)}
        by_id = {str(item["source_id"]): item for item in lexical_results}
        by_id.update({str(item["source_id"]): item for item in neural_results})
        fused = reciprocal_rank_fusion_union(
            [lexical_ids, neural_ids], constant=args.constant
        )[: args.top_k]
        results = []
        for rank, (identifier, score) in enumerate(fused, start=1):
            result = dict(by_id[identifier])
            result.update(
                {
                    "rank": rank,
                    "fusion_score": float(score),
                    "lexical_union_rank": lexical_rank.get(identifier),
                    "neural_union_rank": neural_rank.get(identifier),
                }
            )
            results.append(result)
        output_rows.append(
            {
                "dataset": neural["dataset"],
                "case_id": case_id,
                "query": neural["query"],
                "retrieval_protocol": PROTOCOL,
                "retrieval_config": {
                    **config,
                    "lexical_protocol": lexical.get("retrieval_protocol"),
                    "neural_protocol": neural.get("retrieval_protocol"),
                },
                "results": results,
                "retrieved": len(results),
                "status": "complete",
            }
        )

    write_jsonl(output_rows, args.output)
    print(
        json.dumps(
            {"cases": len(output_rows), "output": str(args.output), **config},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
