"""Measure the expected independent-looking coordinated-fabrication boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.coordinated_fabrication import stress_records, summarize_stress
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.episode_inference import LogisticCalibrator
from evidencegraph.nli_evaluation import validate_nli_dataset
from evidencegraph.provenance import sha256
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_json_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl_atomic(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _render_snippet(summary: dict) -> str:
    first = summary["results"]["cells"]["independent_looking"]["1"]
    last = summary["results"]["cells"]["independent_looking"]["20"]
    oracle = summary["results"]["cells"]["oracle_shared_campaign"]["20"]
    return (
        "\\paragraph{Coordinated-fabrication boundary.}\n"
        f"On {summary['results']['n_claims']:,} held-out false claims, the frozen "
        "NLI edge and dev calibrator were stressed with one through 20 "
        "coordinated supporting sources. When their dependence relation was "
        "missed and they appeared independent, mean false-claim confidence "
        f"changed from {first['mean_false_claim_probability']:.3f} to "
        f"{last['mean_false_claim_probability']:.3f}, and false acceptance "
        f"changed from {100 * first['false_acceptance']:.1f}\\% to "
        f"{100 * last['false_acceptance']:.1f}\\%. With the same NLI evidence "
        "but oracle 0.95 shared-campaign dependence, the 20-source values were "
        f"{oracle['mean_false_claim_probability']:.3f} and "
        f"{100 * oracle['false_acceptance']:.1f}\\%. This is an expected "
        "dependence-false-negative boundary, not a robustness success or a "
        "promotion endpoint.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument("--nli", type=Path, action="append", required=True)
    parser.add_argument(
        "--episode-summary",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/episode_evaluation/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/coordinated_fabrication",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--snippet",
        type=Path,
        default=PROJECT_ROOT
        / "manuscript/generated/coordinated_fabrication_result.tex",
    )
    args = parser.parse_args()
    episodes = read_jsonl(args.dataset)
    nli_records = [record for path in args.nli for record in read_jsonl(path)]
    dataset_hash = sha256(args.dataset)
    validate_nli_dataset(nli_records, dataset_hash)
    with args.episode_summary.open(encoding="utf-8") as handle:
        episode_summary = json.load(handle)
    if (
        episode_summary.get("protocol") != "evidencegraph-episode-inference-v1"
        or episode_summary.get("calibration_split") != "dev"
        or episode_summary.get("evaluation_split") != "test"
        or episode_summary.get("inputs", {}).get("dataset", {}).get("sha256")
        != dataset_hash
    ):
        raise ValueError("fabrication stress requires the matching frozen episode run")
    expected_nli = [sha256(path) for path in args.nli]
    if [row.get("sha256") for row in episode_summary["inputs"].get("nli", [])] != expected_nli:
        raise ValueError("fabrication stress NLI inputs do not match episode calibration")
    parameters = episode_summary["variants"]["evidencegraph"]["calibrator"]
    calibrator = LogisticCalibrator(
        scale=float(parameters["scale"]), intercept=float(parameters["intercept"])
    )
    records = stress_records(episodes, nli_records, calibrator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    _write_jsonl_atomic(records, predictions)
    summary = {
        "schema_version": 1,
        "protocol": "coordinated-fabrication-false-negative-boundary-v1",
        "status": "complete",
        "interpretation": (
            "expected limitation: coordinated sources receive independent-looking "
            "dependence in the stress arm; no promotion claim is permitted"
        ),
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "nli": [source_descriptor(path, PROJECT_ROOT) for path in args.nli],
            "episode_summary": source_descriptor(args.episode_summary, PROJECT_ROOT),
        },
        "predictions": source_descriptor(predictions, PROJECT_ROOT),
        "calibrator": parameters,
        "results": summarize_stress(
            records, n_resamples=args.bootstrap_resamples, seed=103
        ),
    }
    summary_path = args.output_dir / "summary.json"
    _write_json_atomic(summary, summary_path)
    args.snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet_text = _render_snippet(summary)
    temporary = args.snippet.with_name(f".{args.snippet.name}.partial")
    temporary.write_text(snippet_text, encoding="utf-8")
    temporary.replace(args.snippet)
    print(json.dumps({"claims": summary["results"]["n_claims"], "rows": len(records)}))


if __name__ == "__main__":
    main()
