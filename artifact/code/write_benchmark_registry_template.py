"""Write the exact expected benchmark-summary matrix without claiming completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_benchmark_registries import DATASETS, METHODS, TRANSFER_METHODS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLUGS = {
    "FEVER": "fever_dev",
    "HoVer": "hover_dev",
    "AVeriTeC": "averitec_dev",
    "ExpertQA": "expertqa_domain_test",
}
HAIKU_45_PROFILE_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _paths(methods, backbone: str, kind: str) -> dict:
    return {
        dataset: {
            method: (
                "artifact/runs/benchmarks/summaries/"
                f"{SLUGS[dataset]}_{method}_{backbone}_{kind}.json"
            )
            for method, _ in methods
        }
        for dataset in DATASETS
    }


def template() -> dict:
    canonical = {
        "display_name": "Qwen2.5-14B-Instruct",
        "model_requested": "qwen2.5:14b",
        "endpoint_audit": (
            "artifact/runs/benchmarks/evidencegraph_endpoint_audit.json"
        ),
        "quality": _paths(METHODS, "qwen14", "quality"),
        "cost": _paths(METHODS, "qwen14", "cost"),
        "fever_official": {
            method: (
                "artifact/runs/benchmarks/summaries/"
                f"fever_dev_{method}_qwen14_fever_official.json"
            )
            for method, _ in METHODS
        },
    }
    transfer = {
        "display_name": "Claude Haiku 4.5",
        "model_requested": HAIKU_45_PROFILE_ID,
        "endpoint_audit": (
            "artifact/runs/benchmarks/haiku45_endpoint_audit.json"
        ),
        "quality": _paths(TRANSFER_METHODS, "haiku45", "quality"),
        "cost": _paths(TRANSFER_METHODS, "haiku45", "cost"),
    }
    return {
        "protocol": "benchmark-registry-manifest-v1",
        "status": "template_not_results",
        "canonical": canonical,
        "transfer": transfer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/config/benchmark_registry_manifest.template.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(template(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
