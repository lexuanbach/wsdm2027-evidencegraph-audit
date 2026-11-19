"""Run the complete local EvidenceGraph edge and episode evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from evidencegraph.controller_evaluation import ROLLOUT_METHODS
from evidencegraph.controller import CONTROLLER_PROTOCOL
from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.dependence_features import DEPENDENCE_SCORER_PROTOCOL, FEATURE_NAMES
from evidencegraph.episode_evaluation import VARIANT_ARGUMENTS
from evidencegraph.result_registry import source_descriptor, verified_source_path
from run_paraphrase_recovery_ladder import (
    CODE_DIR,
    PROJECT_ROOT,
    _run,
    _sha256,
    _write_json_exact,
)


LARGE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
LARGE_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
SMALL_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
SMALL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"


def nli_run_specs(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "tag": tag,
            "split": split,
            "model": model,
            "revision": revision,
            "output": output_dir / f"{tag}_{split}.jsonl",
        }
        for tag, model, revision in (
            ("deberta_large", LARGE_MODEL, LARGE_REVISION),
            ("minilm", SMALL_MODEL, SMALL_REVISION),
        )
        for split in ("dev", "test")
    ]


def _read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    return value


def dependence_denominators(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Derive exact claim and within-episode pair counts for every split."""

    split_claims = {split: 0 for split in ("train", "dev", "test")}
    split_pairs = {split: 0 for split in split_claims}
    seen: set[str] = set()
    for episode in episodes:
        split = str(episode.get("split", ""))
        claim_id = str(episode.get("claim_id", ""))
        sources = episode.get("sources")
        if (
            split not in split_claims
            or not claim_id
            or claim_id in seen
            or not isinstance(sources, list)
            or len(sources) < 2
        ):
            raise ValueError("dependence denominators require valid unique episodes")
        seen.add(claim_id)
        split_claims[split] += 1
        split_pairs[split] += len(sources) * (len(sources) - 1) // 2
    if split_claims != {"train": 14_000, "dev": 3_000, "test": 3_000}:
        raise ValueError(f"unexpected EvidenceGraph-20K split sizes: {split_claims}")
    return split_claims, split_pairs


def _dependence_complete(
    path: Path,
    dataset: Path,
    split_claims: dict[str, int],
    split_pairs: dict[str, int],
) -> bool:
    summary = _read_summary(path)
    if summary is None:
        return False
    if (
        summary.get("protocol") != DEPENDENCE_SCORER_PROTOCOL
        or summary.get("feature_names") != list(FEATURE_NAMES)
        or summary.get("seeds") != [103, 104, 105]
        or summary.get("inputs", {}).get("split_claims") != split_claims
        or {
            split: int(summary.get(f"{split}_pairs", -1))
            for split in ("train", "dev", "test")
        }
        != split_pairs
        or summary.get("inputs", {}).get("dataset")
        != source_descriptor(dataset, PROJECT_ROOT)
    ):
        raise ValueError("existing dependence summary violates the frozen contract")
    inputs = summary["inputs"]
    verified_source_path(inputs["predictions"], PROJECT_ROOT)
    if inputs["predictions"].get("rows") != split_pairs["dev"] + split_pairs["test"]:
        raise ValueError("existing dependence predictions have a partial denominator")
    models = inputs.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("existing dependence summary omits three models")
    for descriptor in models:
        verified_source_path(descriptor, PROJECT_ROOT)
    return True


def _controller_complete(summary_path: Path, model_path: Path, dataset: Path) -> bool:
    summary = _read_summary(summary_path)
    if summary is None:
        return False
    if (
        summary.get("protocol")
        != CONTROLLER_PROTOCOL
        or summary.get("seed") != 103
        or summary.get("inputs", {}).get("dataset", {}).get("sha256")
        != _sha256(dataset)
        or summary.get("model", {}).get("sha256") != _sha256(model_path)
    ):
        raise ValueError("existing controller summary violates the frozen contract")
    return True


def _episode_complete(
    summary_path: Path,
    *,
    dataset: Path,
    nli_paths: list[Path],
    dependence_path: Path,
) -> bool:
    summary = _read_summary(summary_path)
    if summary is None:
        return False
    inputs = summary.get("inputs", {})
    if (
        summary.get("protocol") != "evidencegraph-episode-inference-v1"
        or summary.get("calibration_split") != "dev"
        or summary.get("evaluation_split") != "test"
        or set(summary.get("variants", {})) != set(VARIANT_ARGUMENTS)
        or inputs.get("dataset", {}).get("sha256") != _sha256(dataset)
        or [record.get("sha256") for record in inputs.get("nli", [])]
        != [_sha256(path) for path in nli_paths]
        or inputs.get("dependence", {}).get("sha256") != _sha256(dependence_path)
    ):
        raise ValueError("existing episode summary violates the frozen contract")
    predictions = summary.get("predictions", {})
    prediction_path = Path(str(predictions.get("path", "")))
    if not prediction_path.is_absolute():
        prediction_path = PROJECT_ROOT / prediction_path
    if (
        not prediction_path.is_file()
        or predictions.get("sha256") != _sha256(prediction_path)
        or int(predictions.get("records", -1))
        != len(VARIANT_ARGUMENTS) * 2 * 3_000
    ):
        raise ValueError("existing episode predictions violate the frozen contract")
    for variant in VARIANT_ARGUMENTS:
        calibrator = summary["variants"][variant].get("calibrator", {})
        if (
            not math.isfinite(float(calibrator.get("scale", float("nan"))))
            or not math.isfinite(float(calibrator.get("intercept", float("nan"))))
            or float(calibrator["scale"]) < 0.0
        ):
            raise ValueError("existing episode summary has an invalid calibrator")
        for split in ("dev", "test"):
            split_summary = summary["variants"][variant].get(split, {})
            group = split_summary.get("groups", {}).get("overall", {})
            if (
                split_summary.get("bootstrap_resamples") != 1000
                or group.get("n_claims") != 3000
            ):
                raise ValueError(
                    "existing episode summary has a partial calibration/evaluation denominator"
                )
    return True


def _rollout_complete(
    summary_path: Path,
    *,
    dataset: Path,
    nli_paths: list[Path],
    dependence_path: Path,
    controller_path: Path,
) -> bool:
    summary = _read_summary(summary_path)
    if summary is None:
        return False
    if (
        summary.get("protocol") != "candidate-pool-active-seek-v1"
        or set(summary.get("methods", {})) != set(ROLLOUT_METHODS)
        or summary.get("inputs", {}).get("dataset")
        != source_descriptor(dataset, PROJECT_ROOT)
        or summary.get("inputs", {}).get("nli")
        != [source_descriptor(path, PROJECT_ROOT) for path in nli_paths]
        or summary.get("inputs", {}).get("dependence")
        != source_descriptor(dependence_path, PROJECT_ROOT)
        or summary.get("inputs", {}).get("controller")
        != source_descriptor(controller_path, PROJECT_ROOT)
    ):
        raise ValueError("existing rollout summary violates the frozen contract")
    predictions = summary.get("predictions", {})
    verified_source_path(predictions, PROJECT_ROOT)
    if predictions.get("rows") != len(ROLLOUT_METHODS) * 2 * 3_000:
        raise ValueError("existing rollout predictions have a partial denominator")
    for method in ROLLOUT_METHODS:
        calibrator = summary["methods"][method].get("calibrator", {})
        if (
            not math.isfinite(float(calibrator.get("scale", float("nan"))))
            or not math.isfinite(float(calibrator.get("intercept", float("nan"))))
            or float(calibrator["scale"]) < 0.0
        ):
            raise ValueError("existing rollout summary has an invalid calibrator")
        for split in ("dev", "test"):
            split_summary = summary["methods"][method].get(split, {})
            group = split_summary.get("groups", {}).get("overall", {})
            if (
                group.get("n_claims") != 3000
                or split_summary.get("bootstrap_resamples") != 1000
            ):
                raise ValueError(
                    "existing rollout summary has a partial calibration/evaluation denominator"
                )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--record-batch-size", type=int, default=64)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/edge_episode_pipeline",
    )
    args = parser.parse_args()
    episodes = read_jsonl(args.dataset)
    split_counts, split_pairs = dependence_denominators(episodes)

    nli_dir = PROJECT_ROOT / "artifact/runs/edge_scorers/nli"
    specs = nli_run_specs(nli_dir)
    for spec in specs:
        _run(
            [
                sys.executable,
                str(CODE_DIR / "run_nli_scorer.py"),
                "--dataset",
                str(args.dataset),
                "--split",
                spec["split"],
                "--model",
                spec["model"],
                "--revision",
                spec["revision"],
                "--device",
                args.device,
                "--record-batch-size",
                str(args.record_batch_size),
                "--model-batch-size",
                str(args.model_batch_size),
                "--output",
                str(spec["output"]),
            ],
            offline=True,
        )

    nli_summary = nli_dir / "summary_test.json"
    _run(
        [
            sys.executable,
            str(CODE_DIR / "summarize_nli_scorers.py"),
            "--dataset",
            str(args.dataset),
            "--run",
            f"DeBERTa-large={nli_dir / 'deberta_large_test.jsonl'}",
            "--run",
            f"MiniLM={nli_dir / 'minilm_test.jsonl'}",
            "--split",
            "test",
            "--bootstrap-resamples",
            "1000",
            "--output",
            str(nli_summary),
        ]
    )

    dependence_dir = PROJECT_ROOT / "artifact/runs/edge_scorers/dependence"
    dependence_summary = dependence_dir / "summary.json"
    if not _dependence_complete(
        dependence_summary,
        args.dataset,
        split_counts,
        split_pairs,
    ):
        _run(
            [
                sys.executable,
                str(CODE_DIR / "run_dependence_scorer.py"),
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(dependence_dir),
                "--bootstrap-resamples",
                "1000",
            ]
        )
    dependence_predictions = dependence_dir / "predictions.jsonl"

    controller_model = PROJECT_ROOT / "artifact/runs/controller/controller_seed103.joblib"
    controller_summary = PROJECT_ROOT / "artifact/runs/controller/summary.json"
    if not _controller_complete(controller_summary, controller_model, args.dataset):
        _run(
            [
                sys.executable,
                str(CODE_DIR / "train_controller.py"),
                "--dataset",
                str(args.dataset),
                "--model-output",
                str(controller_model),
                "--summary-output",
                str(controller_summary),
                "--seed",
                "103",
            ]
        )

    large_nli = [
        nli_dir / "deberta_large_dev.jsonl",
        nli_dir / "deberta_large_test.jsonl",
    ]
    small_nli = [nli_dir / "minilm_dev.jsonl", nli_dir / "minilm_test.jsonl"]
    episode_dir = PROJECT_ROOT / "artifact/runs/episode_evaluation"
    if not _episode_complete(
        episode_dir / "summary.json",
        dataset=args.dataset,
        nli_paths=large_nli,
        dependence_path=dependence_predictions,
    ):
        command = [
            sys.executable,
            str(CODE_DIR / "run_episode_evaluation.py"),
            "--dataset",
            str(args.dataset),
            "--dependence",
            str(dependence_predictions),
            "--output-dir",
            str(episode_dir),
            "--bootstrap-resamples",
            "1000",
        ]
        for path in large_nli:
            command.extend(["--nli", str(path)])
        _run(command)

    fabrication_summary = (
        PROJECT_ROOT / "artifact/runs/coordinated_fabrication/summary.json"
    )
    _run(
        [
            sys.executable,
            str(CODE_DIR / "run_coordinated_fabrication_stress.py"),
            "--dataset",
            str(args.dataset),
            "--nli",
            str(large_nli[0]),
            "--nli",
            str(large_nli[1]),
            "--episode-summary",
            str(episode_dir / "summary.json"),
            "--bootstrap-resamples",
            "1000",
        ]
    )

    rollout_outputs = []
    for nli_paths, output_dir in (
        (
            large_nli,
            PROJECT_ROOT / "artifact/runs/controller_evaluation",
        ),
        (
            small_nli,
            PROJECT_ROOT / "artifact/runs/controller_evaluation_minilm",
        ),
    ):
        summary_path = output_dir / "summary.json"
        if not _rollout_complete(
            summary_path,
            dataset=args.dataset,
            nli_paths=nli_paths,
            dependence_path=dependence_predictions,
            controller_path=controller_model,
        ):
            command = [
                sys.executable,
                str(CODE_DIR / "run_controller_evaluation.py"),
                "--dataset",
                str(args.dataset),
                "--dependence",
                str(dependence_predictions),
                "--controller",
                str(controller_model),
                "--output-dir",
                str(output_dir),
                "--bootstrap-resamples",
                "1000",
            ]
            for path in nli_paths:
                command.extend(["--nli", str(path)])
            _run(command)
        rollout_outputs.append(summary_path)

    ablation_audit = (
        PROJECT_ROOT / "artifact/runs/controller_evaluation/ablation_audit.json"
    )
    _run(
        [
            sys.executable,
            str(CODE_DIR / "audit_controller_ablations.py"),
            "--predictions",
            str(PROJECT_ROOT / "artifact/runs/controller_evaluation/predictions.jsonl"),
            "--summary",
            str(rollout_outputs[0]),
            "--output",
            str(ablation_audit),
            "--bootstrap-resamples",
            "1000",
        ]
    )
    with ablation_audit.open(encoding="utf-8") as handle:
        ablation_result = json.load(handle)
    if not ablation_result.get("promotion_allowed"):
        raise RuntimeError(
            "targeted paired ablations did not confirm every frozen component; "
            "preserve the run and execute negative-result recovery"
        )

    _run(
        [
            sys.executable,
            str(CODE_DIR / "build_episode_registries.py"),
            "--controller-summary",
            str(rollout_outputs[0]),
            "--minilm-controller-summary",
            str(rollout_outputs[1]),
            "--nli-summary",
            str(nli_summary),
            "--dependence-summary",
            str(dependence_summary),
            "--ablation-audit",
            str(ablation_audit),
        ]
    )
    _run([sys.executable, str(CODE_DIR / "make_tables.py"), "--available"])
    _run([sys.executable, str(CODE_DIR / "make_figures.py"), "--available"])

    summary = {
        "schema_version": 1,
        "protocol": "local-edge-episode-pipeline-v1",
        "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
        "nli_summary": source_descriptor(nli_summary, PROJECT_ROOT),
        "dependence_summary": source_descriptor(dependence_summary, PROJECT_ROOT),
        "controller_summary": source_descriptor(controller_summary, PROJECT_ROOT),
        "episode_summary": source_descriptor(
            episode_dir / "summary.json", PROJECT_ROOT
        ),
        "coordinated_fabrication_summary": source_descriptor(
            fabrication_summary, PROJECT_ROOT
        ),
        "rollout_summaries": [
            source_descriptor(path, PROJECT_ROOT) for path in rollout_outputs
        ],
        "ablation_audit": source_descriptor(ablation_audit, PROJECT_ROOT),
        "registries": [
            source_descriptor(
                PROJECT_ROOT / "artifact/results/registries" / f"{name}.json",
                PROJECT_ROOT,
            )
            for name in (
                "edge_scorers",
                "ablations",
                "robustness",
                "dependence_buckets",
            )
        ],
    }
    _write_json_exact(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
