"""Build main, pooled-reliability, and cost/transfer measured registries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evidencegraph.benchmark_evaluation import (
    benchmark_run_metadata,
    join_benchmark_predictions,
)
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.result_registry import verified_source_path, write_complete_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "FEVER": 19_998,
    "HoVer": 4_000,
    "AVeriTeC": 500,
    "ExpertQA": 243,
}
METHODS = (
    ("closed_book", "Closed-book"),
    ("lexical_rag", "Lexical RAG"),
    ("fixed_rag", "Fixed top-5 RAG"),
    ("react", "ReAct search"),
    ("cove", "CoVe"),
    ("evidencegraph", "EvidenceGraph"),
)
TRANSFER_METHODS = (
    ("fixed_rag", "Fixed top-5 RAG"),
    ("evidencegraph", "EvidenceGraph"),
)
FEVER_SCORER_COMMIT = "4801615100fbf6327f8e99b5dbaefe5dd890e869"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _unique_paths(paths: list[Path]) -> list[Path]:
    output = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            output.append(path)
            seen.add(resolved)
    return output


def _benchmark_input_paths(summary: dict[str, Any]) -> list[Path]:
    inputs = summary.get("inputs", {})
    cases = inputs.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark summary is missing case descriptors")
    prediction = inputs.get("predictions")
    if not isinstance(prediction, dict):
        raise ValueError("benchmark summary is missing its prediction descriptor")
    prediction_path = verified_source_path(prediction, PROJECT_ROOT)
    if prediction.get("sha256") != summary.get("run", {}).get("predictions_sha256"):
        raise ValueError("benchmark prediction descriptor disagrees with run hash")
    return [
        *(verified_source_path(descriptor, PROJECT_ROOT) for descriptor in cases),
        prediction_path,
    ]


def _verify_raw_benchmark_cell(
    summary: dict[str, Any],
    *,
    dataset: str,
    method: str,
    model_requested: str,
    expected_count: int,
) -> list[Path]:
    """Re-resolve checksum-bound attempts and cases before promoting a cell."""

    paths = _benchmark_input_paths(summary)
    cases = [record for path in paths[:-1] for record in read_jsonl(path)]
    predictions = read_jsonl(paths[-1])
    joined = join_benchmark_predictions(cases, predictions)
    if (
        len(joined) != expected_count
        or {str(row["case"].get("dataset", "")) for row in joined} != {dataset}
    ):
        raise ValueError(f"{dataset}/{method} raw benchmark denominator is incomplete")
    metadata = benchmark_run_metadata(predictions)
    expected_metadata = {
        "method": method,
        "model_requested": model_requested,
        "seed": 103,
        "n_predictions": expected_count,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"{dataset}/{method} raw predictions violate the run identity")
    run = summary.get("run", {})
    if any(run.get(key) != value for key, value in metadata.items()):
        raise ValueError(f"{dataset}/{method} summary disagrees with raw predictions")
    return paths


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}"


def _interval(record: dict[str, Any], metric: str) -> str:
    return (
        f"{_pct(record[metric])} "
        f"[{_pct(record[f'{metric}_low'])}, {_pct(record[f'{metric}_high'])}]"
    )


def _load_matrix(section: dict[str, Any], methods: tuple[str, ...]) -> tuple[dict, list[Path]]:
    if set(section.get("quality", {})) != set(DATASETS):
        raise ValueError("benchmark manifest must cover exactly four datasets")
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    sources = []
    case_descriptors: dict[str, str] = {}
    expected_model = str(section["model_requested"])
    for dataset, expected_count in DATASETS.items():
        entries = section["quality"][dataset]
        if set(entries) != set(methods):
            raise ValueError(f"{dataset} benchmark method matrix is incomplete")
        summaries[dataset] = {}
        for method in methods:
            path = _resolve(str(entries[method]))
            summary = _read(path)
            run = summary.get("run", {})
            if (
                run.get("method") != method
                or run.get("model_requested") != expected_model
                or run.get("n_predictions") != expected_count
                or run.get("seed") != 103
            ):
                raise ValueError(f"{dataset}/{method} violates the frozen run identity")
            if dataset == "ExpertQA":
                if (
                    summary.get("expertqa", {}).get("n_cases") != expected_count
                    or summary["expertqa"].get("bootstrap_resamples") != 1000
                ):
                    raise ValueError(f"{dataset}/{method} has an incomplete citation summary")
            else:
                verdicts = summary.get("verdicts", {})
                if (
                    verdicts.get("bootstrap_resamples") != 1000
                    or set(verdicts.get("datasets", {})) != {dataset}
                    or verdicts["datasets"][dataset].get("n_cases") != expected_count
                ):
                    raise ValueError(f"{dataset}/{method} has an incomplete verdict summary")
            raw_paths = _verify_raw_benchmark_cell(
                summary,
                dataset=dataset,
                method=method,
                model_requested=expected_model,
                expected_count=expected_count,
            )
            serialized_cases = json.dumps(
                summary.get("inputs", {}).get("cases"),
                sort_keys=True,
                separators=(",", ":"),
            )
            if dataset in case_descriptors and case_descriptors[dataset] != serialized_cases:
                raise ValueError(f"{dataset} methods use different benchmark cases")
            case_descriptors[dataset] = serialized_cases
            summaries[dataset][method] = summary
            sources.extend([path, *raw_paths])
    return summaries, sources


def _load_fever_official(
    section: dict[str, Any],
    quality: dict[str, dict[str, dict[str, Any]]],
    methods: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    entries = section.get("fever_official", {})
    if set(entries) != set(methods):
        raise ValueError("official FEVER scorer matrix is incomplete")
    summaries = {}
    sources = []
    for method in methods:
        path = _resolve(str(entries[method]))
        summary = _read(path)
        inputs = summary.get("inputs", {})
        actual_path = verified_source_path(inputs.get("actual"), PROJECT_ROOT)
        prediction_path = verified_source_path(
            inputs.get("predictions"), PROJECT_ROOT
        )
        if (
            summary.get("n_cases") != DATASETS["FEVER"]
            or summary.get("bootstrap_resamples") != 1000
            or summary.get("max_evidence") != 5
            or summary.get("official_scorer", {}).get("commit") != FEVER_SCORER_COMMIT
            or inputs.get("predictions")
            != quality["FEVER"][method].get("inputs", {}).get("predictions")
        ):
            raise ValueError(f"FEVER/{method} official score violates the scorer contract")
        summaries[method] = summary
        sources.extend([path, actual_path, prediction_path])
    return summaries, sources


def main_registry(
    quality: dict[str, dict[str, dict[str, Any]]],
    fever: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for dataset in DATASETS:
        for method, display in METHODS:
            summary = quality[dataset][method]
            if dataset == "ExpertQA":
                citation = summary["expertqa"]
                accuracy = macro_f1 = ece = "--"
                attribution = (
                    f"P {_interval(citation, 'citation_source_precision')} / "
                    f"R {_interval(citation, 'citation_source_recall')}"
                )
            else:
                record = summary["verdicts"]["datasets"][dataset]
                accuracy = _interval(record, "accuracy")
                macro_f1 = _interval(record, "macro_f1")
                ece = _pct(record["ece"])
                attribution = "--"
                if dataset == "FEVER":
                    accuracy = _interval(fever[method], "label_accuracy")
                    attribution = _interval(fever[method], "evidence_f1")
            rows.append(
                {
                    "dataset": dataset,
                    "method": display,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "attribution": attribution,
                    "ece": ece,
                }
            )
    return {
        "caption": "End-to-end quality on the complete benchmark evaluation splits.",
        "columns": [
            {"key": "dataset", "header": "Dataset"},
            {"key": "method", "header": "Method"},
            {"key": "accuracy", "header": "Accuracy [95\\% CI]"},
            {"key": "macro_f1", "header": "Macro F1 [95\\% CI]"},
            {"key": "attribution", "header": "Evidence / citation [95\\% CI]"},
            {"key": "ece", "header": "ECE"},
        ],
        "rows": rows,
    }


def reliability_registry(
    quality: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, list[list[float | int]]]:
    output = {}
    for method, display in TRANSFER_METHODS:
        bins = {index: [0, 0.0, 0.0] for index in range(10)}
        for dataset in ("FEVER", "HoVer", "AVeriTeC"):
            records = quality[dataset][method]["verdicts"]["datasets"][dataset][
                "reliability_bins"
            ]
            if not isinstance(records, list) or not records:
                raise ValueError(f"{dataset}/{method} reliability bins are missing")
            seen_bins: set[int] = set()
            dataset_count = 0
            for record in records:
                lower = float(record["bin_lower"])
                upper = float(record["bin_upper"])
                confidence = float(record["mean_confidence"])
                accuracy = float(record["empirical_accuracy"])
                index = round(lower * 10)
                count = record["count"]
                if (
                    not all(math.isfinite(value) for value in (lower, upper, confidence, accuracy))
                    or index not in range(10)
                    or index in seen_bins
                    or not math.isclose(lower, index / 10.0, abs_tol=1e-12)
                    or not math.isclose(upper, (index + 1) / 10.0, abs_tol=1e-12)
                    or not lower <= confidence <= upper
                    or not 0.0 <= accuracy <= 1.0
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise ValueError(f"{dataset}/{method} has invalid reliability bins")
                seen_bins.add(index)
                dataset_count += count
                bins[index][0] += count
                bins[index][1] += count * confidence
                bins[index][2] += count * accuracy
            if dataset_count != DATASETS[dataset]:
                raise ValueError(
                    f"{dataset}/{method} reliability denominator is incomplete"
                )
        pooled = []
        for count, confidence_sum, correct_sum in bins.values():
            if count:
                pooled.append([confidence_sum / count, correct_sum / count, count])
        if sum(int(row[2]) for row in pooled) != 24_498:
            raise ValueError("pooled reliability denominator is incomplete")
        output[display] = pooled
    return output


def _load_costs(
    section: dict[str, Any],
    quality: dict[str, dict[str, dict[str, Any]]],
    methods: tuple[str, ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[Path]]:
    if set(section.get("cost", {})) != set(DATASETS):
        raise ValueError("cost manifest must cover exactly four datasets")
    output = {}
    sources = []
    for dataset in DATASETS:
        entries = section["cost"][dataset]
        if set(entries) != set(methods):
            raise ValueError(f"{dataset} cost matrix is incomplete")
        output[dataset] = {}
        for method in methods:
            path = _resolve(str(entries[method]))
            summary = _read(path)
            if (
                summary.get("run") != quality[dataset][method]["run"]
                or summary.get("cost", {}).get("bootstrap_resamples") != 1000
                or set(summary["cost"].get("datasets", {})) != {dataset, "ALL"}
                or summary["cost"]["datasets"][dataset].get("n_cases")
                != DATASETS[dataset]
                or summary["cost"]["datasets"]["ALL"].get("n_cases")
                != DATASETS[dataset]
            ):
                raise ValueError(f"{dataset}/{method} cost summary is incomplete")
            if summary.get("inputs") != quality[dataset][method].get("inputs"):
                raise ValueError(f"{dataset}/{method} cost inputs differ from quality inputs")
            output[dataset][method] = summary
            sources.extend([path, *_benchmark_input_paths(summary)])
    return output, sources


def _load_endpoint_audit(section: dict[str, Any]) -> list[Path]:
    path = _resolve(str(section.get("endpoint_audit", "")))
    audit = _read(path)
    gates = audit.get("claim_gates", {})
    if (
        audit.get("protocol")
        != "evidencegraph-paired-endpoint-audit-v3-dataset-primary"
        or audit.get("promotion_allowed") is not True
        or audit.get("negative_endpoints") != []
        or gates.get("quality_replication_met") is not True
        or gates.get("calibration_replication_met") is not True
    ):
        raise ValueError("benchmark endpoint audit does not permit promotion")
    inputs = audit.get("inputs", {})
    expected_slugs = {
        "fever_dev",
        "hover_dev",
        "averitec_dev",
        "expertqa_domain_test",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_slugs:
        raise ValueError("benchmark endpoint audit inputs are incomplete")
    sources = [path]
    for slug in sorted(inputs):
        comparison_path = verified_source_path(inputs[slug], PROJECT_ROOT)
        comparison = _read(comparison_path)
        candidate = comparison.get("candidate_run", {})
        baseline = comparison.get("baseline_run", {})
        if (
            comparison.get("protocol")
            != "paired-benchmark-answering-v2-official-fever-evidence"
            or candidate.get("model_requested") != section.get("model_requested")
            or baseline.get("model_requested") != section.get("model_requested")
            or candidate.get("method") != "evidencegraph"
            or baseline.get("method") != "fixed_rag"
        ):
            raise ValueError("endpoint audit comparison uses the wrong run identity")
        if slug == "fever_dev" and comparison.get("official_fever_scorer") != {
            "commit": "4801615100fbf6327f8e99b5dbaefe5dd890e869",
            "max_evidence": 5,
        }:
            raise ValueError("endpoint audit FEVER comparison is not officially bound")
        sources.append(comparison_path)
    return sources


def cost_transfer_registry(
    sections: list[
        tuple[
            str,
            dict[str, dict[str, dict[str, Any]]],
            dict[str, dict[str, dict[str, Any]]],
        ]
    ]
) -> dict[str, Any]:
    rows = []
    for backbone, quality, costs in sections:
        for dataset in DATASETS:
            if dataset == "ExpertQA":
                fixed_quality = quality[dataset]["fixed_rag"]["expertqa"][
                    "citation_source_f1"
                ]
                graph_quality = quality[dataset]["evidencegraph"]["expertqa"][
                    "citation_source_f1"
                ]
            elif dataset == "AVeriTeC":
                fixed_quality = quality[dataset]["fixed_rag"]["verdicts"][
                    "datasets"
                ][dataset]["macro_f1"]
                graph_quality = quality[dataset]["evidencegraph"]["verdicts"][
                    "datasets"
                ][dataset]["macro_f1"]
            else:
                fixed_quality = quality[dataset]["fixed_rag"]["verdicts"][
                    "datasets"
                ][dataset]["accuracy"]
                graph_quality = quality[dataset]["evidencegraph"]["verdicts"][
                    "datasets"
                ][dataset]["accuracy"]
            fixed_cost = costs[dataset]["fixed_rag"]["cost"]["datasets"][dataset]
            graph_cost = costs[dataset]["evidencegraph"]["cost"]["datasets"][
                dataset
            ]
            fixed_tokens = (
                fixed_cost["mean_prompt_tokens"]
                + fixed_cost["mean_completion_tokens"]
            )
            graph_tokens = (
                graph_cost["mean_prompt_tokens"]
                + graph_cost["mean_completion_tokens"]
            )
            rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "quality_delta": f"{100.0 * (graph_quality - fixed_quality):+.1f}",
                    "search_delta": (
                        f"{graph_cost['mean_searches'] - fixed_cost['mean_searches']:+.2f}"
                    ),
                    "document_delta": (
                        f"{graph_cost['mean_documents_opened'] - fixed_cost['mean_documents_opened']:+.2f}"
                    ),
                    "token_delta": f"{graph_tokens - fixed_tokens:+.0f}",
                    "latency_delta": (
                        f"{graph_cost['mean_end_to_end_wall_seconds'] - fixed_cost['mean_end_to_end_wall_seconds']:+.2f}"
                    ),
                }
            )
    return {
        "caption": (
            "EvidenceGraph gains and incremental resource use relative to fixed "
            "top-5 RAG across frozen answer backbones."
        ),
        "columns": [
            {"key": "backbone", "header": "Backbone"},
            {"key": "dataset", "header": "Dataset"},
            {"key": "quality_delta", "header": "$\\Delta$ quality (pp)"},
            {"key": "search_delta", "header": "$\\Delta$ searches"},
            {"key": "document_delta", "header": "$\\Delta$ docs"},
            {"key": "token_delta", "header": "$\\Delta$ tokens"},
            {"key": "latency_delta", "header": "$\\Delta$ seconds"},
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifact/config/benchmark_registry_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/results/registries",
    )
    parser.add_argument(
        "--available",
        action="store_true",
        help="build complete canonical main/reliability outputs before transfer exists",
    )
    args = parser.parse_args()
    manifest = _read(args.manifest)
    if manifest.get("protocol") != "benchmark-registry-manifest-v1":
        raise ValueError("unknown benchmark registry manifest protocol")
    if not args.available and manifest.get("status") != "complete_matrix_manifest":
        raise ValueError("final transfer registry requires a complete matrix manifest")
    method_keys = tuple(key for key, _ in METHODS)
    transfer_keys = tuple(key for key, _ in TRANSFER_METHODS)
    canonical, canonical_quality_sources = _load_matrix(
        manifest["canonical"], method_keys
    )
    canonical_audit_sources = _load_endpoint_audit(manifest["canonical"])
    fever, fever_sources = _load_fever_official(
        manifest["canonical"], canonical, method_keys
    )
    canonical_cost, canonical_cost_sources = _load_costs(
        manifest["canonical"], canonical, method_keys
    )
    main_sources = _unique_paths(
        [*canonical_quality_sources, *fever_sources, *canonical_audit_sources]
    )
    main_path = write_complete_registry(
        "main",
        main_registry(canonical, fever),
        main_sources,
        args.output_dir,
        project_root=PROJECT_ROOT,
    )
    reliability_sources = []
    for dataset in ("FEVER", "HoVer", "AVeriTeC"):
        for method, _ in TRANSFER_METHODS:
            summary_path = _resolve(
                manifest["canonical"]["quality"][dataset][method]
            )
            reliability_sources.extend(
                [summary_path, *_benchmark_input_paths(canonical[dataset][method])]
            )
    reliability_sources = _unique_paths(
        [*reliability_sources, *canonical_audit_sources]
    )
    reliability_path = write_complete_registry(
        "reliability",
        reliability_registry(canonical),
        reliability_sources,
        args.output_dir,
        project_root=PROJECT_ROOT,
    )
    outputs = [main_path, reliability_path]
    if not args.available:
        transfer, transfer_quality_sources = _load_matrix(
            manifest["transfer"], transfer_keys
        )
        transfer_audit_sources = _load_endpoint_audit(manifest["transfer"])
        transfer_cost, transfer_cost_sources = _load_costs(
            manifest["transfer"], transfer, transfer_keys
        )
        transfer_path = write_complete_registry(
            "cost_transfer",
            cost_transfer_registry(
                [
                    (manifest["canonical"]["display_name"], canonical, canonical_cost),
                    (manifest["transfer"]["display_name"], transfer, transfer_cost),
                ]
            ),
            _unique_paths([
                args.manifest,
                *canonical_quality_sources,
                *canonical_cost_sources,
                *transfer_quality_sources,
                *transfer_cost_sources,
                *canonical_audit_sources,
                *transfer_audit_sources,
            ]),
            args.output_dir,
            project_root=PROJECT_ROOT,
        )
        outputs.append(transfer_path)
    print(
        json.dumps(
            {"registries": [str(path) for path in outputs]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
