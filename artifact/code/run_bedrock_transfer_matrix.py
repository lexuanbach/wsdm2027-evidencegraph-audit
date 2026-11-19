"""Run the frozen fixed-RAG/EvidenceGraph Haiku transfer after local success."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any

import boto3

from evidencegraph.audit import KAPPA_TARGET, meets_kappa_target
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.result_registry import source_descriptor, verified_source_path
from run_evidencegraph_benchmark_matrix import (
    canonical_outputs_ready,
    selection_path,
    selection_progress,
    verify_local_success_gate,
)
from run_local_benchmark_matrix import (
    DATASETS,
    PROJECT_ROOT,
    PYTHON,
    _retrieval_complete,
    _run,
    prediction_progress,
)
from write_benchmark_registry_template import HAIKU_45_PROFILE_ID, template


REGION = "us-east-1"
MODEL = HAIKU_45_PROFILE_ID
METHODS = ("fixed_rag", "evidencegraph")
AUDIT_SUMMARY = PROJECT_ROOT / "data/audit/audit_summary.json"
DATASET_REGISTRY = PROJECT_ROOT / "artifact/results/registries/dataset.json"
DATASET_RELEASE_MANIFEST = (
    PROJECT_ROOT / "release/evidencegraph20k-v1/release_manifest.json"
)


def transfer_paths(dataset: str, method: str) -> dict[str, Path]:
    if dataset not in DATASETS or method not in METHODS:
        raise ValueError("unknown Bedrock transfer cell")
    prefix = f"{dataset}_{method}_haiku45"
    return {
        "cases": PROJECT_ROOT / f"data/benchmarks/{dataset}.jsonl",
        "predictions": PROJECT_ROOT / f"artifact/runs/benchmarks/{prefix}.jsonl",
        "quality": PROJECT_ROOT
        / f"artifact/runs/benchmarks/summaries/{prefix}_quality.json",
        "cost": PROJECT_ROOT
        / f"artifact/runs/benchmarks/summaries/{prefix}_cost.json",
        "fever_official": PROJECT_ROOT
        / f"artifact/runs/benchmarks/summaries/{prefix}_fever_official.json",
    }


def transfer_comparison_path(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError("unknown Bedrock transfer dataset")
    return PROJECT_ROOT / (
        "artifact/runs/benchmarks/summaries/"
        f"{dataset}_evidencegraph_vs_fixed_rag_haiku45_paired.json"
    )


def transfer_comparison_command(dataset: str) -> list[str]:
    return [
        str(PYTHON),
        "artifact/code/compare_benchmark_answering.py",
        "--cases",
        str(transfer_paths(dataset, "fixed_rag")["cases"]),
        "--candidate-predictions",
        str(transfer_paths(dataset, "evidencegraph")["predictions"]),
        "--baseline-predictions",
        str(transfer_paths(dataset, "fixed_rag")["predictions"]),
        "--output",
        str(transfer_comparison_path(dataset)),
        "--bootstrap-resamples",
        "10000",
    ]


def transfer_command(dataset: str, method: str) -> list[str]:
    paths = transfer_paths(dataset, method)
    common = [
        "--backend",
        "bedrock",
        "--region",
        REGION,
        "--model",
        MODEL,
        "--cases",
        str(paths["cases"]),
        "--output",
        str(paths["predictions"]),
    ]
    if dataset == "expertqa_domain_test" and method == "evidencegraph":
        return [
            str(PYTHON),
            "artifact/code/run_expertqa_evidencegraph.py",
            *common,
            "--retrievals",
            str(PROJECT_ROOT / DATASETS[dataset]["fixed"]),
        ]
    retrievals = (
        selection_path(dataset)
        if method == "evidencegraph"
        else PROJECT_ROOT / DATASETS[dataset]["fixed"]
    )
    return [
        str(PYTHON),
        "artifact/code/run_benchmark_answering.py",
        *common,
        "--method",
        method,
        "--retrievals",
        str(retrievals),
    ]


def unresolved_backend_failure(path: Path) -> bool:
    if not path.exists():
        return False
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(path):
        grouped[(str(record.get("dataset", "")), str(record.get("case_id", "")))].append(
            record
        )
    return any(
        not any(record.get("status") == "complete" for record in records)
        and any(record.get("status") == "backend_failure" for record in records)
        for records in grouped.values()
    )


def verify_bedrock_connectivity(
    catalog: Any | None = None, runtime: Any | None = None
) -> dict[str, Any]:
    catalog = catalog or boto3.client("bedrock", region_name=REGION)
    runtime = runtime or boto3.client("bedrock-runtime", region_name=REGION)
    models = catalog.list_foundation_models(byProvider="Amazon")
    active_amazon = sum(
        model.get("modelLifecycle", {}).get("status") == "ACTIVE"
        for model in models.get("modelSummaries", [])
    )
    profiles = catalog.list_inference_profiles(
        maxResults=100, typeEquals="SYSTEM_DEFINED"
    )
    profile = next(
        (
            item
            for item in profiles.get("inferenceProfileSummaries", [])
            if item.get("inferenceProfileId") == MODEL
        ),
        None,
    )
    if profile is None or profile.get("status") != "ACTIVE":
        raise RuntimeError("the frozen Haiku 4.5 inference profile is not active")
    response = runtime.converse(
        modelId="amazon.nova-micro-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "Reply with exactly: OK"}],
            }
        ],
        inferenceConfig={"maxTokens": 8, "temperature": 0},
    )
    text = response["output"]["message"]["content"][0]["text"].strip()
    result = {
        "region": REGION,
        "catalog_http_status": models["ResponseMetadata"]["HTTPStatusCode"],
        "active_amazon_models": active_amazon,
        "nova_model": "amazon.nova-micro-v1:0",
        "nova_http_status": response["ResponseMetadata"]["HTTPStatusCode"],
        "nova_exact_ok": text == "OK",
        "haiku_profile": MODEL,
        "haiku_profile_status": profile.get("status"),
    }
    if (
        result["catalog_http_status"] != 200
        or result["nova_http_status"] != 200
        or not result["nova_exact_ok"]
    ):
        raise RuntimeError("Bedrock preflight did not return the frozen success result")
    return result


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_dataset_release_gate(
    audit_summary_path: Path = AUDIT_SUMMARY,
    dataset_registry_path: Path = DATASET_REGISTRY,
    release_manifest_path: Path = DATASET_RELEASE_MANIFEST,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Rehash the genuine audit and release before any paid model call."""

    audit = _read_object(audit_summary_path)
    metrics = audit.get("metrics", {})
    audit_inputs = audit.get("inputs", {})
    expected_audit_inputs = {
        "dataset",
        "tasks",
        "source_key",
        "claim_ratings",
        "source_ratings",
        "audit_protocol",
    }
    if (
        audit.get("status") != "complete"
        or audit.get("protocol") != "two-independent-raters-v2-blinded-aliases"
        or audit.get("target_kappa") != KAPPA_TARGET
        or audit.get("target_met") is not True
        or not meets_kappa_target(metrics)
        or metrics.get("n_claims") != 3_000
        or metrics.get("n_claim_ratings") != 6_000
        or metrics.get("rater_count") != 2
        or not isinstance(audit_inputs, dict)
        or set(audit_inputs) != expected_audit_inputs
    ):
        raise ValueError("Bedrock transfer requires the complete genuine audit")
    audited_inputs = {
        name: verified_source_path(descriptor, project_root)
        for name, descriptor in audit_inputs.items()
    }

    registry = _read_object(dataset_registry_path)
    registry_sources = registry.get("sources")
    if (
        registry.get("status") != "complete"
        or registry.get("display_id") != "dataset"
        or not isinstance(registry.get("registry"), dict)
        or not registry["registry"]
        or not isinstance(registry_sources, list)
        or not registry_sources
    ):
        raise ValueError("Bedrock transfer requires the complete dataset registry")
    verified_registry_sources = [
        verified_source_path(descriptor, project_root)
        for descriptor in registry_sources
    ]
    registry_source_set = {path.resolve() for path in verified_registry_sources}
    if (
        audit_summary_path.resolve() not in registry_source_set
        or audited_inputs["dataset"].resolve() not in registry_source_set
    ):
        raise ValueError("dataset registry is not bound to the audit and corpus")

    release = _read_object(release_manifest_path)
    release_inputs = release.get("inputs")
    release_outputs = release.get("outputs")
    validation = release.get("validation", {})
    release_root = release_manifest_path.parent
    if (
        release.get("schema_version") != 1
        or release.get("protocol") != "evidencegraph20k-audited-release-v1"
        or release.get("status") != "complete"
        or release.get("release_name") != "evidencegraph20k-v1"
        or validation.get("episodes") != 20_000
        or validation.get("test_claims") != 3_000
        or not isinstance(release_inputs, list)
        or not release_inputs
        or not isinstance(release_outputs, list)
        or not release_outputs
    ):
        raise ValueError("Bedrock transfer requires the complete dataset release")
    verified_release_inputs = [
        verified_source_path(descriptor, project_root)
        for descriptor in release_inputs
    ]
    release_input_set = {path.resolve() for path in verified_release_inputs}
    if (
        audit_summary_path.resolve() not in release_input_set
        or dataset_registry_path.resolve() not in release_input_set
        or audited_inputs["dataset"].resolve() not in release_input_set
    ):
        raise ValueError("dataset release is not bound to the audit, registry, and corpus")
    verified_release_outputs = [
        verified_source_path(descriptor, release_root)
        for descriptor in release_outputs
    ]
    relative_outputs = {
        str(path.resolve().relative_to(release_root.resolve()))
        for path in verified_release_outputs
    }
    required_outputs = {
        "data/evidencegraph20k.jsonl",
        "metadata/dataset_registry.json",
        "audit/audit_summary.json",
    }
    if not required_outputs <= relative_outputs:
        raise ValueError("dataset release omits an audited required output")
    return {
        "audit_summary": source_descriptor(audit_summary_path, project_root),
        "dataset_registry": source_descriptor(dataset_registry_path, project_root),
        "release_manifest": source_descriptor(release_manifest_path, project_root),
        "n_claims": metrics["n_claims"],
        "rater_count": metrics["rater_count"],
    }


def _evaluate(dataset: str, method: str) -> None:
    paths = transfer_paths(dataset, method)
    common = [
        "--cases",
        str(paths["cases"]),
        "--predictions",
        str(paths["predictions"]),
        "--bootstrap-resamples",
        "1000",
    ]
    _run(
        [
            str(PYTHON),
            "artifact/code/evaluate_benchmark_answering.py",
            *common,
            "--output",
            str(paths["quality"]),
        ]
    )
    _run(
        [
            str(PYTHON),
            "artifact/code/evaluate_benchmark_cost.py",
            *common,
            "--output",
            str(paths["cost"]),
        ]
    )
    if dataset == "fever_dev":
        _run(
            [
                str(PYTHON),
                "artifact/code/evaluate_fever_official.py",
                "--predictions",
                str(paths["predictions"]),
                "--output",
                str(paths["fever_official"]),
                "--bootstrap-resamples",
                "1000",
            ]
        )


def transfer_outputs_ready() -> bool:
    for dataset in DATASETS:
        for method in METHODS:
            paths = transfer_paths(dataset, method)
            if not paths["quality"].is_file() or not paths["cost"].is_file():
                return False
        if not transfer_comparison_path(dataset).is_file():
            return False
    return True


def transfer_endpoint_audit_command(output: Path) -> list[str]:
    command = [
        str(PYTHON),
        "artifact/code/audit_evidencegraph_endpoints.py",
        "--output",
        str(output),
    ]
    for dataset in DATASETS:
        command.extend(
            ["--comparison", dataset, str(transfer_comparison_path(dataset))]
        )
    return command


def write_complete_manifest() -> Path:
    output = PROJECT_ROOT / "artifact/config/benchmark_registry_manifest.json"
    payload = template()
    payload["status"] = "complete_matrix_manifest"
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--retry-rounds", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.retry_rounds < 1 or args.poll_seconds < 1:
        raise ValueError("retry rounds and poll interval must be positive")
    datasets = args.dataset or list(DATASETS)
    methods = args.method or list(METHODS)
    if not canonical_outputs_ready():
        raise RuntimeError("Bedrock transfer requires the complete canonical matrix")
    local_summary = (
        PROJECT_ROOT / "artifact/runs/benchmarks/evidencegraph_matrix_summary.json"
    )
    endpoint_audit = (
        PROJECT_ROOT / "artifact/runs/benchmarks/evidencegraph_endpoint_audit.json"
    )
    if not local_summary.is_file() or not endpoint_audit.is_file():
        raise RuntimeError("Bedrock transfer requires the local success and endpoint audit")
    verify_local_success_gate(local_summary, endpoint_audit)
    dataset_release = verify_dataset_release_gate()
    preflight = verify_bedrock_connectivity()
    print(
        json.dumps(
            {"dataset_release_gate": dataset_release, "bedrock_preflight": preflight},
            sort_keys=True,
        ),
        flush=True,
    )

    lock_path = PROJECT_ROOT / "artifact/runs/benchmarks/.bedrock_transfer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Bedrock transfer matrix is active") from error
        for dataset in datasets:
            while not _retrieval_complete(dataset, "fixed_rag"):
                print(f"waiting for complete transfer retrieval: {dataset}", flush=True)
                time.sleep(args.poll_seconds)
            if dataset != "expertqa_domain_test" and "evidencegraph" in methods:
                complete, _failures = selection_progress(
                    PROJECT_ROOT / f"data/benchmarks/{dataset}.jsonl",
                    selection_path(dataset),
                )
                if complete != int(DATASETS[dataset]["count"]):
                    raise RuntimeError(f"{dataset} graph selection is incomplete")
            for method in methods:
                paths = transfer_paths(dataset, method)
                expected = int(DATASETS[dataset]["count"])
                complete, failures = prediction_progress(
                    paths["cases"], paths["predictions"]
                )
                for attempt in range(args.retry_rounds):
                    if complete == expected:
                        break
                    if unresolved_backend_failure(paths["predictions"]):
                        raise RuntimeError(
                            f"{dataset}/{method} has an unresolved Bedrock failure; "
                            "refresh the credential and resume explicitly"
                        )
                    print(
                        f"running Haiku {dataset}/{method}: {complete}/{expected} "
                        f"complete, {failures} preserved failures, retry {attempt + 1}",
                        flush=True,
                    )
                    _run(transfer_command(dataset, method))
                    complete, failures = prediction_progress(
                        paths["cases"], paths["predictions"]
                    )
                if complete != expected:
                    raise RuntimeError(
                        f"{dataset}/{method} Haiku transfer remains incomplete: "
                        f"{complete}/{expected}"
                    )
                _evaluate(dataset, method)
                print(f"completed Haiku transfer: {dataset}/{method}", flush=True)
            if set(methods) == set(METHODS):
                _run(transfer_comparison_command(dataset))

    if set(datasets) == set(DATASETS) and set(methods) == set(METHODS):
        if not transfer_outputs_ready():
            raise RuntimeError("Haiku transfer summaries are incomplete")
        endpoint_audit = PROJECT_ROOT / (
            "artifact/runs/benchmarks/haiku45_endpoint_audit.json"
        )
        _run(transfer_endpoint_audit_command(endpoint_audit))
        with endpoint_audit.open(encoding="utf-8") as handle:
            audit = json.load(handle)
        if not audit.get("promotion_allowed"):
            raise RuntimeError(
                "Haiku transfer failed the frozen replicated-gain endpoint audit; "
                "preserve the paid runs and execute negative-result recovery"
            )
        manifest = write_complete_manifest()
        _run(
            [
                str(PYTHON),
                "artifact/code/build_benchmark_registries.py",
                "--manifest",
                str(manifest),
            ]
        )
        _run([str(PYTHON), "artifact/code/make_tables.py", "--available"])
        _run([str(PYTHON), "artifact/code/make_figures.py", "--available"])
        summary_path = (
            PROJECT_ROOT / "artifact/runs/benchmarks/bedrock_transfer_summary.json"
        )
        payload = {
            "protocol": "bedrock-haiku45-transfer-v1",
            "preflight": preflight,
            "dataset_release_gate": dataset_release,
            "model": MODEL,
            "manifest": source_descriptor(manifest, PROJECT_ROOT),
            "endpoint_audit": source_descriptor(endpoint_audit, PROJECT_ROOT),
            "cells": [
                {
                    "dataset": dataset,
                    "method": method,
                    "predictions": source_descriptor(
                        transfer_paths(dataset, method)["predictions"], PROJECT_ROOT
                    ),
                    "quality": source_descriptor(
                        transfer_paths(dataset, method)["quality"], PROJECT_ROOT
                    ),
                    "cost": source_descriptor(
                        transfer_paths(dataset, method)["cost"], PROJECT_ROOT
                    ),
                    "comparison": source_descriptor(
                        transfer_comparison_path(dataset), PROJECT_ROOT
                    ),
                }
                for dataset in DATASETS
                for method in METHODS
            ],
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({"summary": str(summary_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
