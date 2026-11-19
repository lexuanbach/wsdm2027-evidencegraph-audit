"""Append-only frozen answering from completed benchmark retrievals."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import requests

from evidencegraph.benchmark_answerer import (
    LocalBenchmarkAnswerer,
    NO_EVIDENCE_LABEL,
    apply_graph_calibrated_probability,
    answer_context_results,
    cited_evidence_records,
    deterministic_no_evidence_verdict,
)
from evidencegraph.baseline_agents import ChainOfVerificationAgent
from evidencegraph.bedrock_llm import BedrockConverseClient, BedrockInferenceError
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.local_llm import OllamaClient
from evidencegraph.result_registry import source_descriptor
from evidencegraph.retrieval_evaluation import (
    resolve_retrieval_attempts,
    resolve_selection_attempts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--retrievals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--backend", choices=("ollama", "bedrock"), default="ollama")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--method",
        choices=("closed_book", "lexical_rag", "fixed_rag", "cove", "evidencegraph"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("EVIDENCEGRAPH_LLM_WORKERS", "1")),
        help="Ordered concurrent Ollama requests; Bedrock remains serial.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.backend != "ollama" and args.workers != 1:
        raise ValueError("concurrent workers are restricted to the local Ollama backend")
    if args.backend == "bedrock" and (
        args.model == "qwen2.5:14b" or args.model.startswith("REPLACE_WITH")
    ):
        raise ValueError(
            "Bedrock runs require --model with the exact verified model/profile ID"
        )
    answer_config = {
        "protocol": (
            ChainOfVerificationAgent.PROMPT_VERSION
            if args.method == "cove"
            else (
                "graph-calibrated-structured-answering-v2-argmax-repair"
                if args.method == "evidencegraph"
                else "strict-structured-answering-v3-argmax-repair"
            )
        ),
        "method": args.method,
        "model_requested": args.model,
        "backend": args.backend,
        "region": args.region if args.backend == "bedrock" else None,
        "seed_control": (
            "ollama-seed"
            if args.backend == "ollama"
            else "temperature-zero-no-api-seed"
        ),
        "seed": args.seed,
        "top_k": None if args.method == "evidencegraph" else args.top_k,
        "context_policy": (
            "all-graph-selected" if args.method == "evidencegraph" else "fixed-top-k"
        ),
        "probability_policy": (
            "evidencegraph20k-active-dev-calibrated-support-v1"
            if args.method == "evidencegraph"
            else "frozen-llm-structured-probabilities"
        ),
        "output_protocol": "exact-cited-passage-audit-v1",
        "cases": source_descriptor(args.cases, PROJECT_ROOT),
        "retrievals": (
            source_descriptor(args.retrievals, PROJECT_ROOT)
            if args.retrievals is not None
            else None
        ),
    }
    cases = read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.method == "evidencegraph" and any(
        case.get("dataset") == "ExpertQA" for case in cases
    ):
        raise ValueError(
            "ExpertQA EvidenceGraph answers require run_expertqa_evidencegraph.py"
        )
    if args.method == "closed_book":
        if args.retrievals is not None:
            raise ValueError("closed-book evaluation must not receive retrievals")
        retrievals = {}
    else:
        if args.retrievals is None:
            raise ValueError(f"{args.method} evaluation requires --retrievals")
        attempts = read_jsonl(args.retrievals)
        resolved = (
            resolve_selection_attempts(attempts)
            if args.method == "evidencegraph"
            else resolve_retrieval_attempts(attempts)
        )
        retrievals = {str(record["case_id"]): record for record in resolved}
    completed = set()
    if args.output.exists():
        previous = read_jsonl(args.output)
        incompatible = [
            record
            for record in previous
            if record.get("answer_config") != answer_config
        ]
        if incompatible:
            raise ValueError(
                "answer output contains completions from a different configuration"
            )
        completed = {
            str(record["case_id"])
            for record in previous
            if record.get("status") == "complete"
        }
    client = (
        OllamaClient(timeout=300)
        if args.backend == "ollama"
        else BedrockConverseClient(region=args.region, timeout=300)
    )
    answerer = LocalBenchmarkAnswerer(args.model, client=client)
    cove = (
        ChainOfVerificationAgent(args.model, client=answerer.client, answerer=answerer)
        if args.method == "cove"
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = failures = 0

    def process_case(case: dict) -> tuple[dict, bool]:
        case_id = str(case["case_id"])
        retrieval = retrievals.get(case_id)
        if args.method != "closed_book" and retrieval is None:
            raise ValueError(f"case {case_id} lacks a completed retrieval")
        if (
            args.method == "evidencegraph"
            and retrieval is not None
            and retrieval.get("selection_method") != "evidencegraph"
        ):
            raise ValueError(
                "EvidenceGraph answering requires a graph-selected retrieval manifest"
            )
        context_results = (
            []
            if retrieval is None
            else answer_context_results(
                args.method, list(retrieval["results"]), args.top_k
            )
        )
        sources = [
            {"source_id": str(result["source_id"]), "text": str(result["text"])}
            for result in context_results
        ]
        graph_probability_true = (
            None
            if args.method != "evidencegraph" or retrieval is None
            else retrieval.get("graph_probability_true")
        )
        if args.method == "evidencegraph" and sources:
            if graph_probability_true is None:
                raise ValueError(
                    "EvidenceGraph selection lacks its calibrated graph probability"
                )
            graph_probability_true = float(graph_probability_true)
            if not 0.0 <= graph_probability_true <= 1.0:
                raise ValueError("EvidenceGraph selection has an invalid graph probability")
        base = {
            "dataset": str(case["dataset"]),
            "case_id": case_id,
            "method": args.method,
            "model_requested": args.model,
            "seed": args.seed,
            "top_k": answer_config["top_k"],
            "answer_config": answer_config,
            "documents_opened": (
                len(sources)
                if retrieval is None
                else int(retrieval.get("documents_opened", len(sources)))
            ),
            "searches": 0 if retrieval is None else int(retrieval.get("searches", 0)),
            "llm_calls": 0 if not sources and args.method != "closed_book" else 1,
            "retrieval_wall_seconds": (
                0.0
                if retrieval is None
                else float(
                    retrieval.get(
                        "retrieval_wall_seconds", retrieval.get("wall_seconds", 0.0)
                    )
                )
            ),
            "edge_and_selection_wall_seconds": (
                0.0
                if retrieval is None
                else float(retrieval.get("edge_and_selection_wall_seconds", 0.0))
            ),
            **(
                {
                    "graph_probability_true": graph_probability_true,
                    "graph_evidence_logit": (
                        None if retrieval is None else retrieval.get("evidence_logit")
                    ),
                    "controller_documents_opened": (
                        0
                        if retrieval is None
                        else int(retrieval.get("controller_documents_opened", 0))
                    ),
                    "candidate_documents_scored": (
                        0
                        if retrieval is None
                        else int(retrieval.get("candidate_documents_scored", 0))
                    ),
                }
                if args.method == "evidencegraph"
                else {}
            ),
        }
        fatal_backend_failure = False
        try:
            if args.method == "closed_book" and case["dataset"] == "ExpertQA":
                result = answerer.generate_closed_book_answer(
                    str(case["query"]), seed=args.seed
                )
                output = {
                    **base,
                    "abstained": False,
                    **result.to_dict(),
                    "status": "complete",
                }
            elif args.method == "closed_book":
                result = answerer.predict_closed_book(
                    str(case["dataset"]), str(case["query"]), seed=args.seed
                )
                output = {
                    **base,
                    "abstained": False,
                    **result.to_dict(),
                    "status": "complete",
                }
            elif not sources and case["dataset"] == "ExpertQA":
                output = {
                    **base,
                    "abstained": True,
                    "answer": "",
                    "claims": [],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_seconds": 0.0,
                    "status": "complete",
                }
            elif not sources:
                output = {
                    **base,
                    "abstained": True,
                    **deterministic_no_evidence_verdict(str(case["dataset"])),
                    "status": "complete",
                }
            elif args.method == "cove" and case["dataset"] == "ExpertQA":
                assert cove is not None
                result = cove.run_cited_answer(
                    str(case["query"]), sources, seed=args.seed
                )
                output = {
                    **base,
                    "abstained": False,
                    **result,
                    "status": "complete",
                }
            elif args.method == "cove":
                assert cove is not None
                result = cove.run_verdict(
                    str(case["dataset"]),
                    str(case["query"]),
                    sources,
                    seed=args.seed,
                )
                output = {
                    **base,
                    "abstained": False,
                    **result,
                    "status": "complete",
                }
            elif case["dataset"] == "ExpertQA":
                result = answerer.generate_cited_answer(
                    str(case["query"]),
                    sources,
                    seed=args.seed,
                    policy=args.method,
                )
                output = {
                    **base,
                    "abstained": False,
                    **result.to_dict(),
                    "status": "complete",
                }
            else:
                raw_result = answerer.predict_verdict(
                    str(case["dataset"]),
                    str(case["query"]),
                    sources,
                    seed=args.seed,
                    policy=args.method,
                    additional_context=(
                        "The frozen EvidenceGraph-20K development calibrator assigns "
                        f"probability {float(graph_probability_true):.8f} that the claim "
                        "is supported after dependence discounting. Treat this as the "
                        "support-versus-non-support confidence; use the passages to choose "
                        "the best dataset-specific non-support alternative."
                        if args.method == "evidencegraph"
                        else None
                    ),
                )
                result = (
                    apply_graph_calibrated_probability(
                        raw_result,
                        str(case["dataset"]),
                        float(graph_probability_true),
                    )
                    if args.method == "evidencegraph"
                    else raw_result
                )
                output = {
                    **base,
                    "abstained": bool(
                        args.method == "evidencegraph"
                        and result.label == NO_EVIDENCE_LABEL[str(case["dataset"])]
                    ),
                    **result.to_dict(),
                    **(
                        {
                            "raw_llm_label": raw_result.label,
                            "raw_llm_probabilities": raw_result.probabilities,
                        }
                        if args.method == "evidencegraph"
                        else {}
                    ),
                    "status": "complete",
                }
            output["cited_evidence"] = cited_evidence_records(sources, output)
        except BedrockInferenceError as error:
            output = {
                **base,
                "status": "backend_failure",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
            fatal_backend_failure = True
        except (
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            json.JSONDecodeError,
            requests.RequestException,
        ) as error:
            output = {
                **base,
                "status": "schema_failure",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
        return output, fatal_backend_failure

    pending = [case for case in cases if str(case["case_id"]) not in completed]
    with args.output.open("a", encoding="utf-8") as handle:
        executor = (
            ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        )
        try:
            results = (
                executor.map(process_case, pending)
                if executor is not None
                else map(process_case, pending)
            )
            for output, fatal_backend_failure in results:
                if output["status"] == "complete":
                    written += 1
                else:
                    failures += 1
                handle.write(
                    json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                if fatal_backend_failure:
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    print(json.dumps({"complete": written, "schema_failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
