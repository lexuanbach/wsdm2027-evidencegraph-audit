"""Re-summarise the reduced-scale API-backbone matrix by frozen-retrieval recall stratum.

Checks whether the end-to-end conclusions survive when retrieval actually
finds the annotated evidence.  This script answers that
without a single model call: every arm's per-case verdict and probability vector
is already stored, and the stratifying variable -- whether any annotated gold
evidence is present in the *shared frozen 20-slot pool* -- is a pre-treatment
covariate identical for every arm.  Conditioning on a method's own selected
context would be post-treatment and would bias the comparison, so we never do
that.

Strata (per dataset), all computed on the shared pool:

``all``                  the same all-method intersection the headline table uses
``gold_in_pool``         >=1 annotated gold item inside the 20-slot pool
``gold_missing``         gold annotated but no gold item reached the pool
``no_gold_annotation``   no gold evidence is annotated (FEVER NEI claims)
``complete_group_in_pool``  a *complete* annotated evidence group is in the pool
``gold_in_top5``         >=1 gold item inside the fixed top-5 window

Metrics, bootstrap protocol, exclusion handling and the paired differences
against ``fixed_rag`` are reused verbatim from ``evaluate_haiku_subset`` so the
``all`` stratum reproduces ``evaluation_summary.json`` exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_haiku_subset import DATASET_NAMES, evaluate_dataset
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.result_registry import source_descriptor
from evidencegraph.retrieval_evaluation import resolve_retrieval_attempts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "haiku-subset-recall-stratified-evaluation-v1"
POOL_SIZE = 20
TOP_K = 5
STRATA = (
    "all",
    "gold_in_pool",
    "gold_missing",
    "no_gold_annotation",
    "complete_group_in_pool",
    "gold_in_top5",
)


def ranked_identifiers(dataset_name: str, record: dict[str, Any]) -> list[str]:
    """Identifiers of the frozen ranked pool, in the same space as gold evidence."""
    if dataset_name == "AVeriTeC":
        return [str(result["url"]) for result in record["results"]]
    return [str(result["source_id"]) for result in record["results"]]


def case_strata(case: dict[str, Any], record: dict[str, Any]) -> dict[str, bool]:
    ranked = ranked_identifiers(str(case["dataset"]), record)
    pool = set(ranked[:POOL_SIZE])
    top5 = set(ranked[:TOP_K])
    gold = {str(item) for item in case.get("gold_evidence", [])}
    groups = [
        {str(item) for item in group}
        for group in case.get("metadata", {}).get("evidence_groups", [])
    ]
    if not groups and gold:
        groups = [gold]
    annotated = bool(gold)
    in_pool = annotated and bool(gold & pool)
    return {
        "all": True,
        "gold_in_pool": in_pool,
        "gold_missing": annotated and not in_pool,
        "no_gold_annotation": not annotated,
        "complete_group_in_pool": bool(groups) and any(group <= pool for group in groups),
        "gold_in_top5": annotated and bool(gold & top5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=PROJECT_ROOT / "data/benchmarks/subset")
    parser.add_argument(
        "--predictions-dir", type=Path, default=PROJECT_ROOT / "artifact/runs/benchmarks/haiku_subset"
    )
    parser.add_argument("--model-tag", default="haiku45")
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--reference", default="fixed_rag")
    parser.add_argument(
        "--secondary-reference",
        default="evidencegraph_llm_verdict",
        help=(
            "Second reference arm.  Differences against it isolate the verdict "
            "override, because the selection-only arm is the identical frozen "
            "generation re-scored on the model's own verdict."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = args.dataset or ["fever_dev", "hover_dev", "averitec_dev"]
    methods = args.method or [
        "closed_book",
        "fixed_rag",
        "evidencegraph",
        "evidencegraph_llm_verdict",
    ]
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "model_tag": args.model_tag,
        "reference": args.reference,
        "secondary_reference": args.secondary_reference,
        "bootstrap_resamples": args.bootstrap_resamples,
        "pool_size": POOL_SIZE,
        "top_k": TOP_K,
        "strata": list(STRATA),
        "inputs": {},
        "datasets": {},
    }
    for dataset in datasets:
        cases_path = args.cases_dir / f"{dataset}_n500.jsonl"
        retrieval_path = args.predictions_dir / f"{dataset}_retrieval_subset.jsonl"
        cases = read_jsonl(cases_path)
        retrievals = resolve_retrieval_attempts(read_jsonl(retrieval_path))
        retrieval_index = {str(record["case_id"]): record for record in retrievals}
        predictions: dict[str, list[dict[str, Any]]] = {}
        inputs = {
            "cases": source_descriptor(cases_path, PROJECT_ROOT),
            "retrievals": source_descriptor(retrieval_path, PROJECT_ROOT),
        }
        for method in methods:
            path = args.predictions_dir / f"{dataset}_{method}_{args.model_tag}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            if not rows:
                continue
            predictions[method] = rows
            inputs[method] = source_descriptor(path, PROJECT_ROOT)
        if not predictions:
            continue
        summary["inputs"][dataset] = inputs
        # The all-method intersection is computed once, on the full subset, and
        # every stratum is a subset of it: stratifying first and intersecting
        # afterwards would let a different exclusion set enter each stratum.
        overall = evaluate_dataset(
            dataset,
            cases,
            predictions,
            n_resamples=args.bootstrap_resamples,
            seed=103,
            reference=args.reference,
        )
        excluded = set()
        for dropped in overall["excluded_cases"].values():
            excluded.update(str(item) for item in dropped)
        shared_cases = [case for case in cases if str(case["case_id"]) not in excluded]
        assert len(shared_cases) == overall["n_cases_compared"], (
            dataset,
            len(shared_cases),
            overall["n_cases_compared"],
        )
        membership = {
            str(case["case_id"]): case_strata(case, retrieval_index[str(case["case_id"])])
            for case in cases
        }
        block: dict[str, Any] = {
            "dataset": DATASET_NAMES[dataset],
            "n_cases_in_subset": len(cases),
            "n_cases_compared": overall["n_cases_compared"],
            "excluded_cases": overall["excluded_cases"],
            "stratum_counts_full_subset": {
                stratum: sum(1 for flags in membership.values() if flags[stratum])
                for stratum in STRATA
            },
            "strata": {},
        }
        for stratum in STRATA:
            selected = [
                case for case in shared_cases if membership[str(case["case_id"])][stratum]
            ]
            if not selected:
                block["strata"][stratum] = {"n_cases_compared": 0, "empty": True}
                continue
            result = (
                overall
                if stratum == "all"
                else evaluate_dataset(
                    dataset,
                    selected,
                    predictions,
                    n_resamples=args.bootstrap_resamples,
                    seed=103,
                    reference=args.reference,
                )
            )
            result.pop("excluded_cases", None)
            result.pop("excluded_case_counts", None)
            if args.secondary_reference and args.secondary_reference in predictions:
                secondary = evaluate_dataset(
                    dataset,
                    selected,
                    predictions,
                    n_resamples=args.bootstrap_resamples,
                    seed=103,
                    reference=args.secondary_reference,
                )
                result["secondary_reference"] = args.secondary_reference
                result["paired_versus_secondary_reference"] = secondary[
                    "paired_versus_reference"
                ]
            result["gold_label_counts"] = {
                label: sum(1 for case in selected if str(case["gold_label"]) == label)
                for label in sorted({str(case["gold_label"]) for case in selected})
            }
            block["strata"][stratum] = result
        summary["datasets"][dataset] = block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for dataset, block in summary["datasets"].items():
        print(f"== {block['dataset']}  compared={block['n_cases_compared']}")
        for stratum, result in block["strata"].items():
            if result.get("empty"):
                print(f"  {stratum:24s} EMPTY")
                continue
            print(f"  {stratum:24s} n={result['n_cases_compared']}")
            for method, entry in sorted(result["methods"].items()):
                print(
                    f"    {method:26s} acc {entry['accuracy']:.3f}"
                    f" [{entry.get('accuracy_low', float('nan')):.3f},{entry.get('accuracy_high', float('nan')):.3f}]"
                    f" ece {entry['ece']:.3f} fsr {entry['false_support_rate']:.3f}"
                )


if __name__ == "__main__":
    main()
