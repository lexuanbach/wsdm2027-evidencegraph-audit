"""Post-hoc bounded-precision re-analysis of frozen rollout/episode scores.

Observed on the development split: a handful of learned covariance matrices
are near-singular with heterogeneous proxy variances, and generalised least
squares then *extrapolates* (negative weights, stance outside [-1, 1]) and
reports precisions of 1e6--1e9, i.e. evidence logits of order 1e4.  Two such
development episodes (0.6%) were enough to drive one ablation's two-parameter
calibrator to a zero slope.  This script re-scores the *same frozen rollouts*
with two bounds that follow from the model itself: the GLS mean is clipped to
the stance range [-1, 1], and the precision is capped at n/0.05 (the maximum n
independent sources can contribute at the variance floor).  Dev calibrators
are refitted and test metrics recomputed for every method, so the effect of
the bound is visible method by method.  It is a labelled post-hoc repair, not
the preregistered protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.episode_evaluation import summarize_episode_predictions
from evidencegraph.episode_inference import fit_logistic_calibrator
from evidencegraph.metrics import expected_calibration_error
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "bounded-precision-reanalysis-v1"
VARIANCE_FLOOR = 0.05


NO_SCALING = frozenset({"no_effective_evidence"})


def bounded_logit(
    record: dict[str, Any], count_key: str, *, variant: str = ""
) -> tuple[float, bool]:
    """Recompute the evidence logit under the model's own bounds.

    ``no_effective_evidence`` deliberately omits the $\sqrt{w_{eff}}$ factor,
    so its bounded logit is the clipped GLS mean alone; re-applying the scale
    here would silently turn that ablation into the full method.
    """

    stance = float(record["stance_estimate"])
    precision = float(record["effective_evidence"])
    n = int(record[count_key])
    cap = n / VARIANCE_FLOOR
    bounded = abs(stance) > 1.0 or precision > cap
    stance = float(np.clip(stance, -1.0, 1.0))
    precision = min(precision, cap)
    if variant in NO_SCALING:
        return stance, bounded
    return stance * float(np.sqrt(max(precision, 0.0))), bounded


def _paired(
    active: list[dict[str, Any]],
    other: list[dict[str, Any]],
    *,
    metric: str,
    corruptions: tuple[str, ...],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    index = {r["claim_id"]: r for r in other}
    pairs = [(r, index[r["claim_id"]]) for r in active if not corruptions or r["corruption"] in corruptions]
    y = np.asarray([int(a["label"]) for a, _ in pairs])
    pa = np.asarray([float(a["probability_true"]) for a, _ in pairs])
    pb = np.asarray([float(b["probability_true"]) for _, b in pairs])

    def m(sample: np.ndarray) -> float:
        sy, sa, sb = y[sample], pa[sample], pb[sample]
        if metric == "accuracy":
            return float(np.mean((sa >= 0.5) == sy) - np.mean((sb >= 0.5) == sy))
        if metric == "false_acceptance":
            neg = sy == 0
            return float(np.mean(sa[neg] >= 0.5) - np.mean(sb[neg] >= 0.5))
        return float(expected_calibration_error(sy, sa) - expected_calibration_error(sy, sb))

    full = np.arange(len(pairs))
    point = m(full)
    rng = np.random.default_rng(seed)
    boots = [m(rng.integers(0, len(pairs), len(pairs))) for _ in range(n_resamples)]
    low, high = np.quantile(np.asarray(boots), [0.025, 0.975])
    return {"n_claims": len(pairs), "difference": point, "low": float(low), "high": float(high)}


def reanalyze(
    records: list[dict[str, Any]], *, group_key: str, count_key: str, n_resamples: int
) -> dict[str, Any]:
    methods = sorted({str(r[group_key]) for r in records})
    output: dict[str, Any] = {"methods": {}, "calibrated_test": {}}
    for method in methods:
        dev = [r for r in records if r[group_key] == method and r["split"] == "dev"]
        test = [r for r in records if r[group_key] == method and r["split"] == "test"]
        dev_z = []
        dev_bounded = 0
        for r in dev:
            z, bounded = bounded_logit(r, count_key, variant=method)
            dev_z.append(z)
            dev_bounded += int(bounded)
        test_z = []
        test_bounded = 0
        for r in test:
            z, bounded = bounded_logit(r, count_key, variant=method)
            test_z.append(z)
            test_bounded += int(bounded)
        calibrator = fit_logistic_calibrator(dev_z, [r["label"] for r in dev])
        cal_dev = [
            {**r, "probability_true": float(calibrator.predict([z])[0])} for r, z in zip(dev, dev_z)
        ]
        cal_test = [
            {**r, "probability_true": float(calibrator.predict([z])[0])} for r, z in zip(test, test_z)
        ]
        output["methods"][method] = {
            "calibrator_bounded": calibrator.to_dict(),
            "dev_episodes_bounded": dev_bounded,
            "dev_episodes": len(dev),
            "test_episodes_bounded": test_bounded,
            "test_episodes": len(test),
            "dev": summarize_episode_predictions(cal_dev, n_resamples=n_resamples, seed=103),
            "test": summarize_episode_predictions(cal_test, n_resamples=n_resamples, seed=103),
        }
        output["calibrated_test"][method] = cal_test
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    rollouts = read_jsonl(args.rollouts)
    episodes = read_jsonl(args.episodes)
    roll = reanalyze(
        rollouts, group_key="method", count_key="documents_opened", n_resamples=args.bootstrap_resamples
    )
    epis = reanalyze(
        episodes, group_key="variant", count_key="raw_source_count", n_resamples=args.bootstrap_resamples
    )
    specs = (
        ("no_dependence", "false_acceptance", ("duplication", "paraphrase_flood")),
        ("no_active_seek", "accuracy", ("hidden_best_evidence",)),
        ("no_contradiction", "accuracy", ("support_refute_mix",)),
        ("no_effective_evidence", "ece", ("duplication", "paraphrase_flood")),
        ("hard_dependence", "ece", ()),
        ("fixed_top5", "accuracy", ()),
        ("fixed_top5", "false_acceptance", ()),
        ("fixed_top5", "ece", ()),
        ("no_dependence", "accuracy", ()),
        ("no_dependence", "ece", ()),
    )
    paired = []
    active = roll["calibrated_test"]["evidencegraph_active"]
    for index, (method, metric, corruptions) in enumerate(specs):
        if method not in roll["calibrated_test"]:
            continue
        paired.append(
            {
                "ablation": method,
                "metric": metric,
                "corruptions": list(corruptions) or ["overall"],
                **_paired(
                    active,
                    roll["calibrated_test"][method],
                    metric=metric,
                    corruptions=corruptions,
                    n_resamples=args.bootstrap_resamples,
                    seed=403 + index,
                ),
            }
        )
    summary = {
        "protocol": PROTOCOL,
        "variance_floor": VARIANCE_FLOOR,
        "bounds": "stance clipped to [-1,1]; precision capped at n/variance_floor",
        "inputs": {
            "rollouts": source_descriptor(args.rollouts, PROJECT_ROOT),
            "episodes": source_descriptor(args.episodes, PROJECT_ROOT),
        },
        "rollouts": {k: v for k, v in roll.items() if k != "calibrated_test"},
        "episodes": {k: v for k, v in epis.items() if k != "calibrated_test"},
        "rollout_paired_active_minus_ablation": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for method, block in roll["methods"].items():
        g = block["test"]["groups"]["overall"]
        print(
            f"{method:22s} bounded-dev {block['dev_episodes_bounded']}/{block['dev_episodes']} "
            f"acc {g['accuracy']:.3f} ece {g['ece']:.3f} fa {g['false_acceptance']:.3f} "
            f"cal {block['calibrator_bounded']}"
        )
    for row in paired:
        print(row)


if __name__ == "__main__":
    main()
