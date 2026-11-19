"""Run learned EvidenceGraph selection, answering, evaluation, and promotion."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.result_registry import source_descriptor, verified_source_path
from evidencegraph.retrieval_evaluation import resolve_selection_attempts
from run_local_benchmark_matrix import (
    DATASETS,
    MODEL,
    PROJECT_ROOT,
    PYTHON,
    _evaluate,
    _retrieval_complete,
    _run,
    cell_paths,
    prediction_progress,
)


METHOD = "evidencegraph"


def selection_path(dataset: str) -> Path:
    if dataset not in DATASETS or dataset == "expertqa_domain_test":
        raise ValueError("graph-selection paths apply only to verdict datasets")
    return (
        PROJECT_ROOT
        / f"artifact/runs/benchmarks/{dataset}_evidencegraph_selection.jsonl"
    )


def selection_progress(
    cases_path: Path, selections_path: Path
) -> tuple[int, int]:
    cases = read_jsonl(cases_path)
    expected = {(str(row["dataset"]), str(row["case_id"])) for row in cases}
    if len(expected) != len(cases):
        raise ValueError("benchmark cases contain duplicate identities")
    if not selections_path.exists():
        return 0, 0
    attempts = read_jsonl(selections_path)
    completed = resolve_selection_attempts(attempts, require_complete=False)
    complete_keys = {
        (str(record.get("dataset", "")), str(record.get("case_id", "")))
        for record in completed
    }
    if not complete_keys <= expected:
        raise ValueError("graph selection contains an unknown case")
    failures = Counter(str(record.get("status", "")) for record in attempts)[
        "selection_failure"
    ]
    return len(complete_keys), failures


def selection_command(dataset: str) -> list[str]:
    return [
        str(PYTHON),
        "artifact/code/run_graph_selection.py",
        "--retrievals",
        str(PROJECT_ROOT / DATASETS[dataset]["fixed"]),
        "--output",
        str(selection_path(dataset)),
    ]


def answer_command(dataset: str) -> list[str]:
    paths = cell_paths(dataset, METHOD)
    if dataset == "expertqa_domain_test":
        return [
            str(PYTHON),
            "artifact/code/run_expertqa_evidencegraph.py",
            "--cases",
            str(paths["cases"]),
            "--retrievals",
            str(PROJECT_ROOT / DATASETS[dataset]["fixed"]),
            "--model",
            MODEL,
            "--output",
            str(paths["predictions"]),
        ]
    return [
        str(PYTHON),
        "artifact/code/run_benchmark_answering.py",
        "--cases",
        str(paths["cases"]),
        "--retrievals",
        str(selection_path(dataset)),
        "--method",
        METHOD,
        "--model",
        MODEL,
        "--output",
        str(paths["predictions"]),
    ]


def comparison_path(dataset: str) -> Path:
    return PROJECT_ROOT / (
        "artifact/runs/benchmarks/summaries/"
        f"{dataset}_evidencegraph_vs_fixed_rag_qwen14_paired.json"
    )


def comparison_command(dataset: str) -> list[str]:
    return [
        str(PYTHON),
        "artifact/code/compare_benchmark_answering.py",
        "--cases",
        str(cell_paths(dataset, METHOD)["cases"]),
        "--candidate-predictions",
        str(cell_paths(dataset, METHOD)["predictions"]),
        "--baseline-predictions",
        str(cell_paths(dataset, "fixed_rag")["predictions"]),
        "--output",
        str(comparison_path(dataset)),
        "--bootstrap-resamples",
        "10000",
    ]


def canonical_outputs_ready() -> bool:
    for dataset in DATASETS:
        for method in (
            "closed_book",
            "lexical_rag",
            "fixed_rag",
            "react",
            "cove",
            "evidencegraph",
        ):
            paths = cell_paths(dataset, method)
            if not paths["quality"].is_file() or not paths["cost"].is_file():
                return False
            if dataset == "fever_dev" and not paths["fever_official"].is_file():
                return False
    return True


def endpoint_audit_command() -> list[str]:
    command = [
        str(PYTHON),
        "artifact/code/audit_evidencegraph_endpoints.py",
        "--output",
        "artifact/runs/benchmarks/evidencegraph_endpoint_audit.json",
    ]
    for dataset in DATASETS:
        command.extend(["--comparison", dataset, str(comparison_path(dataset))])
    return command


def _write_success_summary(datasets: list[str]) -> Path:
    output = PROJECT_ROOT / "artifact/runs/benchmarks/evidencegraph_matrix_summary.json"
    endpoint_audit = (
        PROJECT_ROOT / "artifact/runs/benchmarks/evidencegraph_endpoint_audit.json"
    )
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        paths = cell_paths(dataset, METHOD)
        record: dict[str, Any] = {
            "dataset": dataset,
            "predictions": source_descriptor(paths["predictions"], PROJECT_ROOT),
            "quality": source_descriptor(paths["quality"], PROJECT_ROOT),
            "cost": source_descriptor(paths["cost"], PROJECT_ROOT),
            "comparison": source_descriptor(comparison_path(dataset), PROJECT_ROOT),
        }
        if dataset != "expertqa_domain_test":
            record["selection"] = source_descriptor(
                selection_path(dataset), PROJECT_ROOT
            )
        if dataset == "fever_dev":
            record["fever_official"] = source_descriptor(
                paths["fever_official"], PROJECT_ROOT
            )
        records.append(record)
    payload = {
        "schema_version": 1,
        "protocol": "local-evidencegraph-benchmark-matrix-v1",
        "model": MODEL,
        "endpoint_audit": source_descriptor(endpoint_audit, PROJECT_ROOT),
        "datasets": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def verify_local_success_gate(
    summary_path: Path,
    endpoint_audit_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Re-verify every raw local artifact before paid transfer is permitted."""

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    records = summary.get("datasets")
    if (
        summary.get("protocol") != "local-evidencegraph-benchmark-matrix-v1"
        or summary.get("model") != MODEL
        or not isinstance(records, list)
        or len(records) != len(DATASETS)
        or summary.get("endpoint_audit")
        != source_descriptor(endpoint_audit_path, project_root)
    ):
        raise ValueError("local EvidenceGraph success summary violates its contract")
    by_dataset: dict[str, dict[str, Any]] = {}
    for record in records:
        dataset = str(record.get("dataset", ""))
        if dataset not in DATASETS or dataset in by_dataset:
            raise ValueError("local success summary has duplicate or unknown datasets")
        expected_fields = {
            "dataset",
            "predictions",
            "quality",
            "cost",
            "comparison",
        }
        if dataset != "expertqa_domain_test":
            expected_fields.add("selection")
        if dataset == "fever_dev":
            expected_fields.add("fever_official")
        if set(record) != expected_fields:
            raise ValueError(f"local success record is incomplete for {dataset}")
        for field in expected_fields - {"dataset"}:
            verified_source_path(record[field], project_root)
        by_dataset[dataset] = record
    if set(by_dataset) != set(DATASETS):
        raise ValueError("local success summary does not cover the canonical matrix")

    with endpoint_audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    gates = audit.get("claim_gates", {})
    inputs = audit.get("inputs")
    if (
        audit.get("protocol")
        != "evidencegraph-paired-endpoint-audit-v3-dataset-primary"
        or audit.get("promotion_allowed") is not True
        or audit.get("negative_endpoints") != []
        or gates.get("quality_replication_met") is not True
        or gates.get("calibration_replication_met") is not True
        or not isinstance(inputs, dict)
        or set(inputs) != set(DATASETS)
    ):
        raise ValueError("local endpoint audit does not permit paid transfer")
    for dataset, descriptor in inputs.items():
        verified_source_path(descriptor, project_root)
        if descriptor != by_dataset[dataset]["comparison"]:
            raise ValueError(
                f"local endpoint audit and success summary disagree for {dataset}"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    parser.add_argument("--retry-rounds", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.retry_rounds < 1 or args.poll_seconds < 1:
        raise ValueError("retry rounds and poll interval must be positive")
    datasets = args.dataset or list(DATASETS)
    lock_path = PROJECT_ROOT / "artifact/runs/benchmarks/.evidencegraph_matrix.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another EvidenceGraph benchmark matrix is active") from error
        for dataset in datasets:
            while not _retrieval_complete(dataset, "fixed_rag"):
                print(f"waiting for complete retrieval: {dataset}", flush=True)
                time.sleep(args.poll_seconds)
            paths = cell_paths(dataset, METHOD)
            expected = int(DATASETS[dataset]["count"])
            if dataset != "expertqa_domain_test":
                complete, failures = selection_progress(
                    paths["cases"], selection_path(dataset)
                )
                for attempt in range(args.retry_rounds):
                    if complete == expected:
                        break
                    print(
                        f"selecting {dataset}: {complete}/{expected} complete, "
                        f"{failures} preserved failures, retry {attempt + 1}",
                        flush=True,
                    )
                    _run(selection_command(dataset))
                    complete, failures = selection_progress(
                        paths["cases"], selection_path(dataset)
                    )
                if complete != expected:
                    raise RuntimeError(
                        f"{dataset} graph selection remains incomplete: "
                        f"{complete}/{expected}"
                    )
            complete, failures = prediction_progress(
                paths["cases"], paths["predictions"]
            )
            for attempt in range(args.retry_rounds):
                if complete == expected:
                    break
                print(
                    f"answering {dataset}: {complete}/{expected} complete, "
                    f"{failures} preserved failures, retry {attempt + 1}",
                    flush=True,
                )
                _run(answer_command(dataset))
                complete, failures = prediction_progress(
                    paths["cases"], paths["predictions"]
                )
            if complete != expected:
                raise RuntimeError(
                    f"{dataset} EvidenceGraph answers remain incomplete: "
                    f"{complete}/{expected}"
                )
            _evaluate(dataset, METHOD)
            _run(comparison_command(dataset))
            print(f"completed EvidenceGraph endpoint: {dataset}", flush=True)

        if set(datasets) == set(DATASETS):
            if not canonical_outputs_ready():
                raise RuntimeError("canonical matrix is incomplete after graph evaluation")
            _run(endpoint_audit_command())
            endpoint_audit = PROJECT_ROOT / (
                "artifact/runs/benchmarks/evidencegraph_endpoint_audit.json"
            )
            with endpoint_audit.open(encoding="utf-8") as handle:
                audit = json.load(handle)
            if not audit.get("promotion_allowed"):
                raise RuntimeError(
                    "paired EvidenceGraph endpoint audit found material harm or "
                    "failed the frozen replicated-gain gates; preserve the runs "
                    "and execute the negative-result recovery protocol before "
                    "registry promotion"
                )
            _run(
                [
                    str(PYTHON),
                    "artifact/code/build_benchmark_registries.py",
                    "--manifest",
                    "artifact/config/benchmark_registry_manifest.template.json",
                    "--available",
                ]
            )
            _run([str(PYTHON), "artifact/code/make_tables.py", "--available"])
            _run([str(PYTHON), "artifact/code/make_figures.py", "--available"])
            summary = _write_success_summary(datasets)
            print(json.dumps({"summary": str(summary)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
