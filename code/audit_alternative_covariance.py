"""Reviewer Q3: alternative covariance models on the frozen episode data.

Three questions, all answered by re-analysis of frozen NLI + dependence
predictions.  No model is called and no test label is ever used for fitting.

1. **Shared-scorer term.**  ``Sigma' = sigma^2 J + Sigma_prov``.  This is a
   rank-one update along ``1``.  By Sherman-Morrison
   ``Sigma'^{-1} 1 = Sigma^{-1} 1 / (1 + sigma^2 w)`` with ``w = 1^T Sigma^+ 1``,
   so the *normalised* GLS weights -- and therefore ``theta_hat`` -- are
   unchanged, and ``w_eff -> w / (1 + sigma^2 w)``, saturating at ``1/sigma^2``.
   The audit verifies both claims numerically on every real episode.

2. **Does the confidence rescaling help?**  ``sigma^2`` is fitted on dev only
   (grid, dev log-loss and dev ECE selection reported separately) and the frozen
   two-parameter logistic calibrator is refitted on dev at every grid point.
   Test metrics follow.

3. **Ridge.**  Unlike ``sigma^2 J``, ridge moves ``theta_hat``.  Two forms are
   scored: ``(Sigma + lambda I)^+`` (Tikhonov on the covariance) and
   ``Sigma^+ + lambda I``.  ``lambda`` is tuned on dev only.  The headline is the
   ``support_refute_mix`` cell, where the frozen system loses to plain counting.
   A uniformity diagnostic reports how far the ridge weights have moved toward
   uniform, i.e. how much of any gain is just "ridge turned the model off".

Run::

    PYTHONPATH=artifact/code .venv/bin/python artifact/code/audit_alternative_covariance.py \
      --output-dir artifact/runs/revision/P1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.effective_evidence import (
    aggregate_stances,
    covariance_from_dependence,
    gls_weights,
    shared_scorer_effective_evidence,
)
from evidencegraph.episode_evaluation import (
    episode_inputs,
    index_dependence_records,
    index_nli_records,
    summarize_episode_predictions,
)
from evidencegraph.episode_inference import (
    _probability_matrix,
    edge_variances,
    fit_logistic_calibrator,
)
from evidencegraph.metrics import expected_calibration_error
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "alternative-covariance-audit-v1"
SHRINKAGE = 0.05
SCORER_VARIANCE_GRID = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    0.01,
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
    2.0,
    5.0,
)
RIDGE_GRID = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
)
RIDGE_MODES = ("covariance", "precision")


# ---------------------------------------------------------------------------
# Frozen per-episode inputs
# ---------------------------------------------------------------------------


def prepare_episodes(
    dataset: Path, nli: list[Path], dependence: Path
) -> list[dict[str, Any]]:
    episodes = read_jsonl(dataset)
    nli_index = index_nli_records(
        [record for path in nli for record in read_jsonl(path)]
    )
    dependence_index = index_dependence_records(read_jsonl(dependence))
    prepared: list[dict[str, Any]] = []
    for episode in episodes:
        split = str(episode["split"])
        if split not in ("dev", "test"):
            continue
        probabilities, learned = episode_inputs(episode, nli_index, dependence_index)
        matrix = _probability_matrix(probabilities)
        stances = matrix[:, 0] - matrix[:, 2]
        dep = np.asarray(learned, dtype=np.float64).copy()
        np.fill_diagonal(dep, 1.0)
        prepared.append(
            {
                "claim_id": str(episode["claim_id"]),
                "split": split,
                "label": int(bool(episode["label"])),
                "corruption": str(episode["corruption"]),
                "stances": stances,
                "dependence": dep,
                "variances": edge_variances(stances),
                "n_sources": int(len(stances)),
            }
        )
    if not prepared:
        raise ValueError("no dev/test episodes prepared")
    return prepared


def _logit(estimate: float, effective: float) -> float:
    return float(estimate * np.sqrt(effective))


def score_all(
    prepared: list[dict[str, Any]],
    *,
    use_dependence: bool = True,
    scorer_variance: float = 0.0,
    ridge: float = 0.0,
    ridge_mode: str = "covariance",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prepared:
        n = item["n_sources"]
        dep = item["dependence"] if use_dependence else np.eye(n, dtype=np.float64)
        covariance = covariance_from_dependence(
            dep,
            variances=item["variances"],
            shrinkage=SHRINKAGE if use_dependence else 0.0,
            scorer_variance=scorer_variance,
        )
        estimate, effective = aggregate_stances(
            item["stances"], covariance, ridge=ridge, ridge_mode=ridge_mode
        )
        rows.append(
            {
                "claim_id": item["claim_id"],
                "split": item["split"],
                "label": item["label"],
                "corruption": item["corruption"],
                "stance_estimate": float(estimate),
                "effective_evidence": float(effective),
                "evidence_logit": _logit(estimate, effective),
                "raw_source_count": n,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Dev-only calibration and evaluation
# ---------------------------------------------------------------------------


def _split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def calibrate(rows: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    dev, test = _split(rows, "dev"), _split(rows, "test")
    calibrator = fit_logistic_calibrator(
        [row["evidence_logit"] for row in dev], [row["label"] for row in dev]
    )
    dev = [
        {**row, "probability_true": float(calibrator.predict([row["evidence_logit"]])[0])}
        for row in dev
    ]
    test = [
        {**row, "probability_true": float(calibrator.predict([row["evidence_logit"]])[0])}
        for row in test
    ]
    return calibrator, dev, test


def _log_loss(rows: list[dict[str, Any]]) -> float:
    y = np.asarray([row["label"] for row in rows], dtype=np.float64)
    p = np.clip(
        np.asarray([row["probability_true"] for row in rows], dtype=np.float64),
        1e-12,
        1.0 - 1e-12,
    )
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _point_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    p = np.asarray([row["probability_true"] for row in rows], dtype=np.float64)
    negative = y == 0
    positive = y == 1
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "ece": float(expected_calibration_error(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": _log_loss(rows),
        "false_acceptance": float(np.mean(p[negative] >= 0.5)) if np.any(negative) else float("nan"),
        "true_claim_accuracy": float(np.mean(p[positive] >= 0.5)) if np.any(positive) else float("nan"),
    }


def cell_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {"overall": _point_metrics(rows)}
    for corruption in sorted({row["corruption"] for row in rows}):
        subset = [row for row in rows if row["corruption"] == corruption]
        out[corruption] = _point_metrics(subset)
    return out


def paired_difference(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    *,
    group: str | None,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired claim bootstrap of ``a - b`` on one corruption cell (or overall)."""

    a = {row["claim_id"]: row for row in a_rows}
    b = {row["claim_id"]: row for row in b_rows}
    if set(a) != set(b):
        raise ValueError("paired comparison requires identical claim sets")
    claims = sorted(
        c for c in a if group is None or a[c]["corruption"] == group
    )
    y = np.asarray([a[c]["label"] for c in claims], dtype=np.int64)
    pa = np.asarray([a[c]["probability_true"] for c in claims], dtype=np.float64)
    pb = np.asarray([b[c]["probability_true"] for c in claims], dtype=np.float64)

    def metrics(idx: np.ndarray) -> dict[str, float]:
        sy, sa, sb = y[idx], pa[idx], pb[idx]
        neg = sy == 0
        out = {
            "accuracy": float(np.mean((sa >= 0.5) == sy) - np.mean((sb >= 0.5) == sy)),
            "ece": float(
                expected_calibration_error(sy, sa) - expected_calibration_error(sy, sb)
            ),
            "brier": float(np.mean((sa - sy) ** 2) - np.mean((sb - sy) ** 2)),
        }
        if np.any(neg):
            out["false_acceptance"] = float(
                np.mean(sa[neg] >= 0.5) - np.mean(sb[neg] >= 0.5)
            )
        return out

    base = np.arange(len(claims))
    point = metrics(base)
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(n_resamples):
        sample = rng.integers(0, len(base), len(base))
        for key, value in metrics(sample).items():
            if key in boots and np.isfinite(value):
                boots[key].append(value)
    entry: dict[str, Any] = {"n_claims": int(len(claims))}
    for key, value in point.items():
        arr = np.asarray(boots[key])
        low, high = np.quantile(arr, [0.025, 0.975])
        entry[key] = {
            "difference": value,
            "low": float(low),
            "high": float(high),
            "excludes_zero": bool(low > 0.0 or high < 0.0),
        }
    return entry


# ---------------------------------------------------------------------------
# Part 1: Sherman-Morrison verification on real episodes
# ---------------------------------------------------------------------------


def verify_sherman_morrison(
    prepared: list[dict[str, Any]], grid: tuple[float, ...]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    test_items = [item for item in prepared if item["split"] == "test"]
    for scorer_variance in grid:
        if scorer_variance == 0.0:
            continue
        theta_abs: list[float] = []
        theta_rel: list[float] = []
        conditioned_theta_abs: list[float] = []
        violating: list[dict[str, Any]] = []
        weight_abs: list[float] = []
        weff_abs: list[float] = []
        weff_rel: list[float] = []
        saturation: list[float] = []
        for item in test_items:
            base_cov = covariance_from_dependence(
                item["dependence"], variances=item["variances"], shrinkage=SHRINKAGE
            )
            updated_cov = covariance_from_dependence(
                item["dependence"],
                variances=item["variances"],
                shrinkage=SHRINKAGE,
                scorer_variance=scorer_variance,
            )
            base_estimate, base_weff = aggregate_stances(item["stances"], base_cov)
            new_estimate, new_weff = aggregate_stances(item["stances"], updated_cov)
            predicted = shared_scorer_effective_evidence(base_weff, scorer_variance)
            delta_theta = abs(new_estimate - base_estimate)
            eigenvalues = np.linalg.eigvalsh((base_cov + base_cov.T) / 2.0)
            condition = float(
                np.max(eigenvalues) / max(float(np.min(eigenvalues)), 1e-300)
            )
            if condition < 1e8:
                conditioned_theta_abs.append(delta_theta)
            elif delta_theta > 1e-9:
                violating.append(
                    {
                        "claim_id": item["claim_id"],
                        "corruption": item["corruption"],
                        "n_sources": item["n_sources"],
                        "condition_number": condition,
                        "theta_base": float(base_estimate),
                        "theta_shared_scorer": float(new_estimate),
                        "weff_base": float(base_weff),
                        "weff_shared_scorer": float(new_weff),
                    }
                )
            theta_abs.append(delta_theta)
            theta_rel.append(
                abs(new_estimate - base_estimate) / max(abs(base_estimate), 1e-12)
            )
            weight_abs.append(
                float(
                    np.max(
                        np.abs(gls_weights(updated_cov) - gls_weights(base_cov))
                    )
                )
            )
            weff_abs.append(abs(new_weff - predicted))
            weff_rel.append(abs(new_weff - predicted) / max(predicted, 1e-12))
            saturation.append(new_weff)
        checks[f"scorer_variance_{scorer_variance:g}"] = {
            "n_well_conditioned_episodes": len(conditioned_theta_abs),
            "n_rank_deficient_episodes_with_theta_change": len(violating),
            "max_abs_theta_change_well_conditioned": float(
                np.max(conditioned_theta_abs)
            ),
            "rank_deficient_episodes": violating,
            "max_abs_theta_change": float(np.max(theta_abs)),
            "median_abs_theta_change": float(np.median(theta_abs)),
            "max_rel_theta_change": float(np.max(theta_rel)),
            "max_abs_gls_weight_change": float(np.max(weight_abs)),
            "max_abs_weff_vs_sherman_morrison": float(np.max(weff_abs)),
            "max_rel_weff_vs_sherman_morrison": float(np.max(weff_rel)),
            "saturation_bound_one_over_s2": float(1.0 / scorer_variance),
            "max_observed_weff": float(np.max(saturation)),
            "mean_observed_weff": float(np.mean(saturation)),
        }
    checks["n_test_episodes_checked"] = len(test_items)
    return checks


# ---------------------------------------------------------------------------
# Weight-uniformity diagnostic for ridge
# ---------------------------------------------------------------------------


def uniformity(
    prepared: list[dict[str, Any]], *, ridge: float, ridge_mode: str
) -> dict[str, float]:
    distances: list[float] = []
    max_weights: list[float] = []
    for item in prepared:
        if item["split"] != "test":
            continue
        n = item["n_sources"]
        covariance = covariance_from_dependence(
            item["dependence"], variances=item["variances"], shrinkage=SHRINKAGE
        )
        weights = gls_weights(covariance, ridge=ridge, ridge_mode=ridge_mode)
        uniform = np.full(n, 1.0 / n)
        distances.append(float(np.linalg.norm(weights - uniform, ord=1)))
        max_weights.append(float(np.max(weights)))
    return {
        "mean_l1_distance_from_uniform": float(np.mean(distances)),
        "median_l1_distance_from_uniform": float(np.median(distances)),
        "mean_max_weight": float(np.mean(max_weights)),
    }


# ---------------------------------------------------------------------------


def limit_arm_rows(prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The lambda -> infinity limit of ridge: uniform weights, w_eff = n.

    For large lambda, ``(Sigma + lambda I)^+ -> I/lambda``, so the GLS weights
    become uniform and ``w_eff -> n/lambda``.  The dev-fitted calibrator absorbs
    the constant ``1/sqrt(lambda)``, so the limiting arm is exactly
    ``mean(stance) * sqrt(n)``: an unweighted average vote that uses neither the
    dependence matrix nor the inverse-variance edge weights.
    """

    rows: list[dict[str, Any]] = []
    for item in prepared:
        estimate = float(np.mean(item["stances"]))
        effective = float(item["n_sources"])
        rows.append(
            {
                "claim_id": item["claim_id"],
                "split": item["split"],
                "label": item["label"],
                "corruption": item["corruption"],
                "stance_estimate": estimate,
                "effective_evidence": effective,
                "evidence_logit": _logit(estimate, effective),
                "raw_source_count": item["n_sources"],
            }
        )
    return rows


def dependence_influence(
    prepared: list[dict[str, Any]], *, ridge: float, ridge_mode: str
) -> dict[str, Any]:
    """Does the learned dependence matrix still do anything at this lambda?

    Scores the same episodes with the learned ``Sigma_prov`` and with the
    dependence structure removed (identity correlation, same variances), holding
    ``lambda`` fixed.  If the two agree, the ridge arm's behaviour is not
    dependence-aware aggregation: ridge has switched the model off.
    """

    with_dep = score_all(prepared, ridge=ridge, ridge_mode=ridge_mode)
    without_dep = score_all(
        prepared, use_dependence=False, ridge=ridge, ridge_mode=ridge_mode
    )
    a = np.asarray([row["evidence_logit"] for row in with_dep])
    b = np.asarray([row["evidence_logit"] for row in without_dep])
    theta_a = np.asarray([row["stance_estimate"] for row in with_dep])
    theta_b = np.asarray([row["stance_estimate"] for row in without_dep])
    _, _, test_a = calibrate(with_dep)
    _, _, test_b = calibrate(without_dep)
    pa = np.asarray([row["probability_true"] for row in test_a])
    pb = np.asarray([row["probability_true"] for row in test_b])
    return {
        "ridge": float(ridge),
        "ridge_mode": ridge_mode,
        "logit_pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "max_abs_theta_difference": float(np.max(np.abs(theta_a - theta_b))),
        "median_abs_theta_difference": float(np.median(np.abs(theta_a - theta_b))),
        "test_decision_agreement": float(np.mean((pa >= 0.5) == (pb >= 0.5))),
        "test_metrics_with_dependence": _point_metrics(test_a),
        "test_metrics_without_dependence": _point_metrics(test_b),
    }


def sweep(
    prepared: list[dict[str, Any]],
    *,
    kind: str,
    grid: tuple[float, ...],
    ridge_mode: str = "covariance",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in grid:
        kwargs = (
            {"scorer_variance": value}
            if kind == "scorer_variance"
            else {"ridge": value, "ridge_mode": ridge_mode}
        )
        rows = score_all(prepared, **kwargs)
        calibrator, dev, test = calibrate(rows)
        records.append(
            {
                "kind": kind,
                "ridge_mode": ridge_mode if kind == "ridge" else None,
                "value": float(value),
                "calibrator": calibrator.to_dict(),
                "dev": cell_metrics(dev),
                "test": cell_metrics(test),
                "_dev_rows": dev,
                "_test_rows": test,
            }
        )
    return records


def select(records: list[dict[str, Any]], criterion: str) -> dict[str, Any]:
    return min(records, key=lambda record: record["dev"]["overall"][criterion])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_validated_subset.jsonl",
    )
    parser.add_argument(
        "--nli",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--dependence",
        type=Path,
        default=PROJECT_ROOT
        / "runs/subset_edge_episode/dependence/predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs/revision/P1"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    nli_paths = args.nli or [
        PROJECT_ROOT / "runs/subset_edge_episode/nli/deberta_large_dev.jsonl",
        PROJECT_ROOT / "runs/subset_edge_episode/nli/deberta_large_test.jsonl",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else p.name
               for p in [args.dataset, *nli_paths, args.dependence] if not p.is_file()]
    if missing:
        parser.error("Missing frozen inputs: " + ", ".join(missing) +
                     ". Recover checksum-matched episodes and predictions; frozen weights alone are insufficient.")
    from sys import path as import_path
    import_path.insert(0, str(PROJECT_ROOT / "revision"))
    from input_preflight import validate_splits
    validate_splits(read_jsonl(args.dataset))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_episodes(args.dataset, nli_paths, args.dependence)

    # Reference arms: frozen soft aggregator and plain counting.
    frozen_rows = score_all(prepared)
    _, frozen_dev, frozen_test = calibrate(frozen_rows)
    count_rows = score_all(prepared, use_dependence=False)
    _, count_dev, count_test = calibrate(count_rows)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "calibration_split": "dev",
        "evaluation_split": "test",
        "shrinkage": SHRINKAGE,
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "nli": [source_descriptor(path, PROJECT_ROOT) for path in nli_paths],
            "dependence": source_descriptor(args.dependence, PROJECT_ROOT),
        },
        "n_episodes": {
            "dev": sum(1 for item in prepared if item["split"] == "dev"),
            "test": sum(1 for item in prepared if item["split"] == "test"),
        },
        "sherman_morrison_verification": verify_sherman_morrison(
            prepared, SCORER_VARIANCE_GRID
        ),
        "reference": {
            "learned_soft_dependence": {
                "dev": cell_metrics(frozen_dev),
                "test": cell_metrics(frozen_test),
            },
            "count_sum": {"dev": cell_metrics(count_dev), "test": cell_metrics(count_test)},
        },
    }

    # --- shared scorer sweep -------------------------------------------------
    scorer_records = sweep(
        prepared, kind="scorer_variance", grid=SCORER_VARIANCE_GRID
    )
    payload["scorer_variance_sweep"] = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in scorer_records
    ]
    payload["scorer_variance_selected"] = {}
    for criterion in ("log_loss", "ece"):
        chosen = select(scorer_records, criterion)
        payload["scorer_variance_selected"][criterion] = {
            "value": chosen["value"],
            "calibrator": chosen["calibrator"],
            "dev": chosen["dev"],
            "test": chosen["test"],
            "paired_vs_frozen_overall": paired_difference(
                chosen["_test_rows"],
                frozen_test,
                group=None,
                n_resamples=args.bootstrap_resamples,
                seed=203,
            ),
            "paired_vs_frozen_support_refute_mix": paired_difference(
                chosen["_test_rows"],
                frozen_test,
                group="support_refute_mix",
                n_resamples=args.bootstrap_resamples,
                seed=207,
            ),
        }

    # --- ridge sweep ---------------------------------------------------------
    payload["ridge_sweep"] = {}
    payload["ridge_selected"] = {}
    for mode in RIDGE_MODES:
        ridge_records = sweep(prepared, kind="ridge", grid=RIDGE_GRID, ridge_mode=mode)
        payload["ridge_sweep"][mode] = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in ridge_records
        ]
        payload["ridge_selected"][mode] = {}
        for criterion in ("log_loss", "ece", "accuracy_negated"):
            if criterion == "accuracy_negated":
                chosen = max(
                    ridge_records, key=lambda record: record["dev"]["overall"]["accuracy"]
                )
            else:
                chosen = select(ridge_records, criterion)
            payload["ridge_selected"][mode][criterion] = {
                "value": chosen["value"],
                "calibrator": chosen["calibrator"],
                "dev": chosen["dev"],
                "test": chosen["test"],
                "weight_uniformity": uniformity(
                    prepared, ridge=chosen["value"], ridge_mode=mode
                ),
                "paired_vs_frozen_overall": paired_difference(
                    chosen["_test_rows"],
                    frozen_test,
                    group=None,
                    n_resamples=args.bootstrap_resamples,
                    seed=211,
                ),
                "paired_vs_frozen_support_refute_mix": paired_difference(
                    chosen["_test_rows"],
                    frozen_test,
                    group="support_refute_mix",
                    n_resamples=args.bootstrap_resamples,
                    seed=213,
                ),
                "paired_vs_count_sum_overall": paired_difference(
                    chosen["_test_rows"],
                    count_test,
                    group=None,
                    n_resamples=args.bootstrap_resamples,
                    seed=217,
                ),
                "paired_vs_count_sum_support_refute_mix": paired_difference(
                    chosen["_test_rows"],
                    count_test,
                    group="support_refute_mix",
                    n_resamples=args.bootstrap_resamples,
                    seed=219,
                ),
            }
        # per-episode rows for the dev-log-loss pick
        best = select(ridge_records, "log_loss")
        write_jsonl(
            best["_dev_rows"] + best["_test_rows"],
            args.output_dir / f"ridge_{mode}_predictions.jsonl",
        )

    # --- is the ridge gain just "ridge switched the model off"? -------------
    limit_rows = limit_arm_rows(prepared)
    _, limit_dev, limit_test = calibrate(limit_rows)
    payload["ridge_degeneracy"] = {
        "uniform_mean_stance_limit_arm": {
            "description": (
                "theta = mean(stance), w_eff = n; the lambda -> infinity limit of "
                "ridge. Uses neither the dependence matrix nor the "
                "inverse-variance edge weights."
            ),
            "dev": cell_metrics(limit_dev),
            "test": cell_metrics(limit_test),
            "paired_vs_count_sum_overall": paired_difference(
                limit_test,
                count_test,
                group=None,
                n_resamples=args.bootstrap_resamples,
                seed=223,
            ),
            "paired_vs_count_sum_support_refute_mix": paired_difference(
                limit_test,
                count_test,
                group="support_refute_mix",
                n_resamples=args.bootstrap_resamples,
                seed=227,
            ),
            "paired_vs_frozen_support_refute_mix": paired_difference(
                limit_test,
                frozen_test,
                group="support_refute_mix",
                n_resamples=args.bootstrap_resamples,
                seed=229,
            ),
        },
        "dependence_influence": [
            dependence_influence(prepared, ridge=value, ridge_mode=mode)
            for mode in RIDGE_MODES
            for value in (0.0, 0.1, 1.0, 10.0, 100.0)
        ],
    }
    write_jsonl(
        limit_dev + limit_test, args.output_dir / "uniform_mean_stance_predictions.jsonl"
    )

    payload["weight_uniformity_reference"] = {
        "ridge_zero": uniformity(prepared, ridge=0.0, ridge_mode="covariance"),
    }

    # bootstrapped CIs on the two reference arms in the mixture cell
    payload["reference"]["mixture_bootstrap"] = {
        "learned_soft_dependence": summarize_episode_predictions(
            frozen_test, n_resamples=args.bootstrap_resamples, seed=103
        )["groups"]["support_refute_mix"],
        "count_sum": summarize_episode_predictions(
            count_test, n_resamples=args.bootstrap_resamples, seed=103
        )["groups"]["support_refute_mix"],
    }

    output = args.output_dir / "alternative_covariance.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["sherman_morrison_verification"], indent=2, sort_keys=True))
    print(json.dumps(payload["scorer_variance_selected"]["log_loss"]["test"]["overall"], indent=2, sort_keys=True))
    for mode in RIDGE_MODES:
        entry = payload["ridge_selected"][mode]["log_loss"]
        print(
            json.dumps(
                {
                    "ridge_mode": mode,
                    "lambda": entry["value"],
                    "test_overall": entry["test"]["overall"],
                    "test_support_refute_mix": entry["test"]["support_refute_mix"],
                    "weight_uniformity": entry["weight_uniformity"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
