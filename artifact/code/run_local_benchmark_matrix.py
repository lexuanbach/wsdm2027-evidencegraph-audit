"""Run and evaluate the frozen local-Qwen benchmark baseline matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.retrieval_evaluation import resolve_retrieval_attempts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Keep the venv path itself: resolving its symlink would make child commands
# bypass the environment's pyvenv.cfg and installed packages.
PYTHON = Path(sys.executable)
MODEL = "qwen2.5:14b"
DATASETS: dict[str, dict[str, Any]] = {
    "averitec_dev": {
        "count": 500,
        "fixed": "artifact/runs/benchmarks/averitec_dev_retrieval_document_mmr.jsonl",
        "lexical": "artifact/runs/benchmarks/averitec_dev_lexical.jsonl",
    },
    "expertqa_domain_test": {
        "count": 243,
        "fixed": "artifact/runs/benchmarks/expertqa_domain_test_retrieval_v7_finite.jsonl",
        "lexical": "artifact/runs/benchmarks/expertqa_domain_test_lexical.jsonl",
    },
    "hover_dev": {
        "count": 4_000,
        "fixed": "artifact/runs/benchmarks/hover_dev_retrieval_v7_finite.jsonl",
        "lexical": "artifact/runs/benchmarks/hover_dev_lexical.jsonl",
    },
    "fever_dev": {
        "count": 19_998,
        "fixed": "artifact/runs/benchmarks/fever_dev_retrieval_v7_finite.jsonl",
        "lexical": "artifact/runs/benchmarks/fever_dev_lexical.jsonl",
    },
}
METHODS = ("fixed_rag", "closed_book", "lexical_rag", "cove", "react")
CELL_METHODS = (*METHODS, "evidencegraph")


def cell_paths(dataset: str, method: str) -> dict[str, Path]:
    if dataset not in DATASETS or method not in CELL_METHODS:
        raise ValueError("unknown local benchmark cell")
    prefix = f"{dataset}_{method}_qwen14"
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


def generation_command(dataset: str, method: str) -> list[str]:
    if method not in METHODS:
        raise ValueError("EvidenceGraph cells require the graph-matrix driver")
    paths = cell_paths(dataset, method)
    if method == "react":
        return [
            str(PYTHON),
            "artifact/code/run_react_benchmark.py",
            "--dataset",
            dataset,
            "--model",
            MODEL,
            "--output",
            str(paths["predictions"]),
        ]
    command = [
        str(PYTHON),
        "artifact/code/run_benchmark_answering.py",
        "--cases",
        str(paths["cases"]),
        "--method",
        method,
        "--model",
        MODEL,
        "--output",
        str(paths["predictions"]),
    ]
    if method != "closed_book":
        retrieval_key = "lexical" if method == "lexical_rag" else "fixed"
        command.extend(
            ["--retrievals", str(PROJECT_ROOT / DATASETS[dataset][retrieval_key])]
        )
    return command


def prediction_progress(cases_path: Path, predictions_path: Path) -> tuple[int, int]:
    cases = read_jsonl(cases_path)
    expected = {(str(row["dataset"]), str(row["case_id"])) for row in cases}
    if len(expected) != len(cases):
        raise ValueError("benchmark cases contain duplicate identities")
    if not predictions_path.exists():
        return 0, 0
    complete: Counter[tuple[str, str]] = Counter()
    failures = 0
    for record in read_jsonl(predictions_path):
        key = (str(record.get("dataset", "")), str(record.get("case_id", "")))
        if key not in expected:
            raise ValueError("benchmark output contains an unknown case")
        status = str(record.get("status", ""))
        if status == "complete":
            complete[key] += 1
        elif status in {"schema_failure", "protocol_failure", "backend_failure"}:
            failures += 1
        else:
            raise ValueError(f"unknown benchmark output status: {status!r}")
    duplicate = [key for key, count in complete.items() if count != 1]
    if duplicate:
        raise ValueError("benchmark output contains duplicate completions")
    return len(complete), failures


def _retrieval_complete(dataset: str, method: str) -> bool:
    if method in {"closed_book", "react"}:
        return True
    retrieval_key = "lexical" if method == "lexical_rag" else "fixed"
    path = PROJECT_ROOT / DATASETS[dataset][retrieval_key]
    if not path.exists():
        return False
    completed = resolve_retrieval_attempts(read_jsonl(path), require_complete=False)
    return len(completed) == int(DATASETS[dataset]["count"])


def _run(command: list[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "artifact/code")
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _evaluate(dataset: str, method: str) -> None:
    paths = cell_paths(dataset, method)
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


def _wait_for_pid(pid: int, poll_seconds: int) -> None:
    while pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--retry-rounds", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if args.retry_rounds < 1 or args.poll_seconds < 1:
        raise ValueError("retry rounds and poll interval must be positive")
    datasets = args.dataset or list(DATASETS)
    methods = args.method or list(METHODS)
    if args.wait_pid is not None:
        _wait_for_pid(args.wait_pid, args.poll_seconds)

    lock_path = PROJECT_ROOT / "artifact/runs/benchmarks/.local_qwen14_matrix.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another local Qwen benchmark matrix is active") from error
        for dataset in datasets:
            for method in methods:
                while not _retrieval_complete(dataset, method):
                    print(f"waiting for complete retrieval: {dataset}/{method}", flush=True)
                    time.sleep(args.poll_seconds)
                paths = cell_paths(dataset, method)
                expected = int(DATASETS[dataset]["count"])
                complete, failures = prediction_progress(
                    paths["cases"], paths["predictions"]
                )
                for attempt in range(args.retry_rounds):
                    if complete == expected:
                        break
                    print(
                        f"running {dataset}/{method}: {complete}/{expected} complete, "
                        f"{failures} preserved failures, retry {attempt + 1}",
                        flush=True,
                    )
                    _run(generation_command(dataset, method))
                    complete, failures = prediction_progress(
                        paths["cases"], paths["predictions"]
                    )
                if complete != expected:
                    raise RuntimeError(
                        f"{dataset}/{method} remains incomplete after "
                        f"{args.retry_rounds} retry rounds: {complete}/{expected}"
                    )
                _evaluate(dataset, method)
                print(f"completed and evaluated {dataset}/{method}", flush=True)


if __name__ == "__main__":
    main()
