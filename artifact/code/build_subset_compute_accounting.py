"""Measured compute accounting for the runs this paper reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "artifact/runs"


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json", type=Path, default=RUNS / "subset_compute_accounting.json"
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=PROJECT_ROOT / "manuscript/generated/compute_accounting.tex",
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=RUNS / "benchmarks/haiku_subset/evaluation_summary.json",
        help="End-to-end evaluation summary that defines which cells the paper reports.",
    )
    args = parser.parse_args()
    matrix_summary = args.matrix_summary
    blocks: dict[str, Any] = {}
    sources: list[Path] = []

    local_calls = local_prompt = local_completion = 0
    local_seconds = 0.0
    for path in sorted((RUNS / "duplication").glob("*.jsonl")):
        if not (path.name.startswith("qwen3b-v6") or path.name.startswith("scale-qwen15b")
                or path.name == "qwen3b-v5-seed103-limit40-normalized.jsonl"):
            continue
        sources.append(path)
        for record in _jsonl(path):
            if record.get("status") != "complete":
                continue
            local_calls += 1
            local_prompt += int(record.get("prompt_tokens", 0))
            local_completion += int(record.get("completion_tokens", 0))
            local_seconds += float(record.get("wall_seconds", 0.0))
    blocks["local_duplication"] = {
        "backbones": ["qwen2.5:3b-instruct", "qwen2.5:1.5b"],
        "calls": local_calls,
        "prompt_tokens": local_prompt,
        "completion_tokens": local_completion,
        "wall_hours": local_seconds / 3600.0,
    }

    # Scope the API block to exactly the (dataset, method) cells the end-to-end
    # table reports, read from that table's own evaluation summary, so the
    # accounting can never drift from the paper by matching a file the table
    # does not print.
    with matrix_summary.open(encoding="utf-8") as handle:
        matrix = json.load(handle)
    reported_cells: list[Path] = []
    reported_datasets: set[str] = set()
    for dataset, inputs in matrix["inputs"].items():
        if dataset not in matrix["datasets"]:
            continue
        for method in matrix["datasets"][dataset]["methods"]:
            descriptor = inputs.get(method)
            if descriptor is None:
                raise ValueError(f"reported cell {dataset}/{method} has no input descriptor")
            reported_cells.append(PROJECT_ROOT / descriptor["path"])
            reported_datasets.add(dataset)

    api: dict[str, Any] = {}
    api_calls = api_prompt = api_completion = 0
    api_seconds = 0.0
    derived_cells: list[str] = []
    for path in sorted(reported_cells):
        first = next(_jsonl(path), None)
        if first is not None and "rescoring_protocol" in first.get("answer_config", {}):
            # A re-scored arm is a deterministic function of another cell's
            # frozen generations. Counting its tokens would bill the same
            # inference twice.
            derived_cells.append(path.name)
            continue
        sources.append(path)
        calls = prompt = completion = 0
        seconds = 0.0
        failures = 0
        for record in _jsonl(path):
            if record.get("status") != "complete":
                failures += 1
                continue
            calls += int(record.get("llm_calls", 1))
            prompt += int(record.get("prompt_tokens", 0))
            completion += int(record.get("completion_tokens", 0))
            seconds += float(record.get("wall_seconds", 0.0))
        api[path.stem] = {
            "completed_cases": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "wall_hours": seconds / 3600.0,
            "non_complete_attempts": failures,
        }
        api_calls += calls
        api_prompt += prompt
        api_completion += completion
        api_seconds += seconds
    blocks["api_backbone"] = {
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "scope": "exactly the cells reported by the end-to-end table",
        "matrix_summary": source_descriptor(matrix_summary, PROJECT_ROOT),
        "reported_cells": sorted(path.name for path in reported_cells),
        "derived_cells_excluded_from_cost": sorted(derived_cells),
        "cells": api,
        "total_calls": api_calls,
        "total_prompt_tokens": api_prompt,
        "total_completion_tokens": api_completion,
        "total_wall_hours": api_seconds / 3600.0,
    }

    scorer_seconds = 0.0
    scorer_rows = 0
    for path in sorted((RUNS / "subset_edge_episode/nli").glob("*.jsonl")):
        sources.append(path)
        seen: set[float] = set()
        for record in _jsonl(path):
            scorer_rows += 1
            value = float(record.get("batch_wall_seconds", 0.0))
            key = (value, record.get("model"))
            if key not in seen:
                seen.add(key)
                scorer_seconds += value
    blocks["nli_scoring"] = {
        "edges_scored": scorer_rows,
        "wall_hours": scorer_seconds / 3600.0,
        "device": "Apple-silicon MPS, float32",
    }

    # Graph selection is a cost of the EvidenceGraph cells only, so count it for
    # exactly the datasets whose EvidenceGraph cell the table reports.
    selection_seconds = 0.0
    selection_cases = 0
    selection_paths = [
        RUNS / f"benchmarks/haiku_subset/{dataset}_evidencegraph_selection.jsonl"
        for dataset in sorted(reported_datasets)
        if "evidencegraph" in matrix["datasets"][dataset]["methods"]
    ]
    for path in selection_paths:
        sources.append(path)
        for record in _jsonl(path):
            if record.get("status") == "complete":
                selection_cases += 1
                selection_seconds += float(record.get("edge_and_selection_wall_seconds", 0.0))
    blocks["graph_selection"] = {
        "cases": selection_cases,
        "wall_hours": selection_seconds / 3600.0,
    }

    summary = {
        "protocol": "subset-compute-accounting-v1",
        "blocks": blocks,
        "sources": [source_descriptor(path, PROJECT_ROOT) for path in sources],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    api_block = blocks["api_backbone"]
    tex = (
        "Measured totals for the runs reported here. Local inference on one "
        "Apple-silicon workstation: {calls:,} frozen-backbone verification calls "
        "for the duplication intervention ({ptok:,} prompt and {ctok:,} completion "
        "tokens, {lhrs:.1f} GPU-hours) and {edges:,} entailment edges scored on MPS "
        "({ehrs:.1f} hours), plus {sel:,} graph selections ({shrs:.1f} hours). "
        "Hosted inference used one model at temperature zero: {acalls:,} completed "
        "answer calls, {aptok:,} prompt and {actok:,} completion tokens, "
        "{ahrs:.1f} wall-hours. We do not extrapolate energy or carbon without a "
        "power meter.\n"
    ).format(
        calls=blocks["local_duplication"]["calls"],
        ptok=blocks["local_duplication"]["prompt_tokens"],
        ctok=blocks["local_duplication"]["completion_tokens"],
        lhrs=blocks["local_duplication"]["wall_hours"],
        edges=blocks["nli_scoring"]["edges_scored"],
        ehrs=blocks["nli_scoring"]["wall_hours"],
        sel=blocks["graph_selection"]["cases"],
        shrs=blocks["graph_selection"]["wall_hours"],
        acalls=api_block["total_calls"],
        aptok=api_block["total_prompt_tokens"],
        actok=api_block["total_completion_tokens"],
        ahrs=api_block["total_wall_hours"],
    )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    main()
