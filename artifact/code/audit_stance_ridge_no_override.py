"""Stance-conditional effective evidence + ridge-regularized covariance (no override).

Wave-2 candidate.  Combines three measured Wave-1 findings:

  (1) no calibrated-probability override -- the aggregator emits a calibrated
      probability and never replaces a downstream verdict;
  (2) stance-conditional effective evidence -- a separate GLS mean and a
      separate ``w_eff`` for the support side and the refute side, combined
      through a dev-fitted comparative logistic;
  (4) ridge on the covariance/precision -- ``(Sigma + lambda I)^+`` (or
      ``Sigma^+ + lambda I``), which unlike the shared-scorer rank-one term
      actually moves theta-hat as well as ``w_eff``.

Why ridge is the right dial.  For a side that is one perfectly-copied cluster
of ``k`` sources with edge variance ``sigma^2``, ``1' (Sigma + lambda I)^+ 1 =
k / (k sigma^2 + lambda)``: at ``lambda -> 0`` it is ``1/sigma^2`` (full
deduplication, one origin = one vote) and at ``lambda -> infinity`` it is
``k/lambda`` (counting).  Ridge therefore interpolates continuously between the
paper's aggregator and the count baseline that beats it, and dev picks where on
that line to sit.

EVERY hyperparameter (tau, lambda, ridge mode, comparative form) is selected on
the dev split only, by out-of-fold cross-validated log-loss inside dev.  The
test split is never read by this script.

No model call is made; this is re-analysis of frozen NLI and dependence
predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.effective_evidence import (
    covariance_from_dependence,
    nearest_psd,
)
from evidencegraph.episode_evaluation import (
    episode_inputs,
    index_dependence_records,
    index_nli_records,
    summarize_episode_predictions,
)
from evidencegraph.episode_inference import (
    edge_variances,
    fit_logistic_calibrator,
    score_episode,
)
from evidencegraph.provenance import sha256
from evidencegraph.result_registry import source_descriptor
from evidencegraph.stance_conditional import (
    COMPARATIVE_FORMS,
    comparative_features,
    fit_comparative_model,
    log_loss,
)

import run_episode_baselines as baselines


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "stance-conditional-ridge-no-override-v1"

TAU_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
RIDGE_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0)
RIDGE_MODES = ("covariance", "precision")
FORMS = COMPARATIVE_FORMS  # difference_2p, asymmetric_3p, generous_5p
SHRINKAGE = 0.05
RCOND = 1e-10
N_FOLDS = 5
FOLD_SEED = 20260824
CELLS = ("overall", "support_refute_mix", "duplication", "hidden_best_evidence")
NLI_LABELS = ("support", "neutral", "refute")


# --------------------------------------------------------------------- eigen
class SideSpectrum:
    """Eigendecomposition of one side's covariance, reusable over lambda.

    ``nearest_psd`` and ``_psd_pseudoinverse`` both act in this eigenbasis, so
    every ridge value is obtained from a single decomposition.  ``verify``
    checks the fast path against the frozen library routine bit-closely.
    """

    __slots__ = ("eigenvalues", "u", "v", "n")

    def __init__(self, covariance: np.ndarray, values: np.ndarray) -> None:
        matrix = nearest_psd(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        self.eigenvalues = eigenvalues
        self.u = eigenvectors.T @ np.ones(matrix.shape[0])
        self.v = eigenvectors.T @ np.asarray(values, dtype=np.float64)
        self.n = matrix.shape[0]

    def stats(self, ridge: float, mode: str) -> tuple[float, float]:
        d = self.eigenvalues
        if mode == "covariance":
            shifted = d + ridge
            cutoff = RCOND * max(1.0, float(np.max(shifted)))
            inverse = np.where(shifted > cutoff, 1.0 / np.where(shifted > cutoff, shifted, 1.0), 0.0)
        elif mode == "precision":
            cutoff = RCOND * max(1.0, float(np.max(d)))
            inverse = np.where(d > cutoff, 1.0 / np.where(d > cutoff, d, 1.0), 0.0) + ridge
        else:
            raise ValueError(f"unknown ridge mode: {mode}")
        w = float(np.sum(inverse * self.u * self.u))
        if w <= 0.0:
            raise ArithmeticError("side covariance has no estimable shared mean")
        theta = float(np.sum(inverse * self.u * self.v) / w)
        return theta, max(0.0, w)


EMPTY = None


def _probability_matrix(probabilities: list[dict[str, float]]) -> np.ndarray:
    return np.asarray(
        [[float(r[label]) for label in NLI_LABELS] for r in probabilities],
        dtype=np.float64,
    )


def side_spectra(
    probabilities: list[dict[str, float]], dependence: np.ndarray, tau: float
) -> dict[str, SideSpectrum | None]:
    """Support-side and refute-side spectra at informativeness threshold ``tau``."""

    matrix = _probability_matrix(probabilities)
    stances = matrix[:, 0] - matrix[:, 2]
    magnitudes = np.abs(stances)
    cut = max(tau, np.nextafter(0.0, 1.0))
    dep = np.clip(np.asarray(dependence, dtype=np.float64).copy(), 0.0, 1.0)
    np.fill_diagonal(dep, 1.0)
    out: dict[str, SideSpectrum | None] = {}
    for name, index in (
        ("support", np.flatnonzero(stances >= cut)),
        ("refute", np.flatnonzero(-stances >= cut)),
    ):
        if index.size == 0:
            out[name] = None
            continue
        values = magnitudes[index]
        sub = dep[np.ix_(index, index)]
        covariance = covariance_from_dependence(
            sub, variances=edge_variances(values), shrinkage=SHRINKAGE
        )
        out[name] = SideSpectrum(covariance, values)
    out["support_count"] = int(np.sum(stances >= cut))  # type: ignore[assignment]
    out["refute_count"] = int(np.sum(-stances >= cut))  # type: ignore[assignment]
    out["n_sources"] = int(stances.size)  # type: ignore[assignment]
    return out


def score_record(spectra: dict[str, Any], ridge: float, mode: str) -> dict[str, float]:
    record: dict[str, float] = {}
    for name in ("support", "refute"):
        spectrum = spectra[name]
        if spectrum is None:
            theta, w = 0.0, 0.0
        else:
            theta, w = spectrum.stats(ridge, mode)
        record[f"{name}_stance"] = theta
        record[f"{name}_effective_evidence"] = w
        record[f"{name}_evidence"] = float(theta * np.sqrt(w))
    record["evidence_difference"] = record["support_evidence"] - record["refute_evidence"]
    record["support_count"] = float(spectra["support_count"])
    record["refute_count"] = float(spectra["refute_count"])
    record["dropped_count"] = float(
        spectra["n_sources"] - spectra["support_count"] - spectra["refute_count"]
    )
    return record


# ------------------------------------------------------------------- folding
def stratified_folds(items: list[dict[str, Any]], n_folds: int, seed: int) -> np.ndarray:
    """Fold assignment stratified by (label, corruption)."""

    rng = np.random.default_rng(seed)
    folds = np.zeros(len(items), dtype=np.int64)
    strata: dict[tuple[int, str], list[int]] = {}
    for index, item in enumerate(items):
        strata.setdefault((int(item["label"]), str(item["corruption"])), []).append(index)
    for key in sorted(strata):
        indices = np.asarray(strata[key])
        order = rng.permutation(indices.size)
        for position, index in enumerate(indices[order]):
            folds[index] = position % n_folds
    return folds


def oof_comparative(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    folds: np.ndarray,
    form: str,
) -> np.ndarray:
    design = np.asarray([comparative_features(r, form) for r in records])
    out = np.zeros(len(records), dtype=np.float64)
    for fold in np.unique(folds):
        train = folds != fold
        held = folds == fold
        if len(set(labels[train].tolist())) < 2:
            raise ValueError("a training fold lost a label")
        model = fit_comparative_model(design[train], labels[train], form=form)
        out[held] = model.predict(design[held])
    return out


def oof_scalar(logits: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(logits), dtype=np.float64)
    for fold in np.unique(folds):
        train = folds != fold
        held = folds == fold
        calibrator = fit_logistic_calibrator(logits[train], labels[train])
        out[held] = calibrator.predict(logits[held])
    return out


def build_records(
    items: list[dict[str, Any]], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": item["claim_id"],
            "split": item["split"],
            "label": item["label"],
            "corruption": item["corruption"],
            "probability_true": float(p),
        }
        for item, p in zip(items, probabilities)
    ]


def cell_accuracy(items, probabilities, cell: str) -> float:
    mask = [i for i, item in enumerate(items) if item["corruption"] == cell]
    if not mask:
        return float("nan")
    return float(
        np.mean([(probabilities[i] >= 0.5) == bool(items[i]["label"]) for i in mask])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--nli", type=Path, action="append", required=True)
    parser.add_argument("--dependence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    episodes = read_jsonl(args.dataset)
    nli_index = index_nli_records(
        [record for path in args.nli for record in read_jsonl(path)]
    )
    dependence_index = index_dependence_records(read_jsonl(args.dependence))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- DEV only
    items: list[dict[str, Any]] = []
    for episode in episodes:
        if str(episode["split"]) != "dev":
            continue
        probabilities, learned = episode_inputs(episode, nli_index, dependence_index)
        items.append(
            {
                "claim_id": str(episode["claim_id"]),
                "split": "dev",
                "label": int(bool(episode["label"])),
                "corruption": str(episode["corruption"]),
                "probabilities": probabilities,
                "learned": learned,
            }
        )
    if not items:
        raise SystemExit("no dev episodes found")
    labels = np.asarray([item["label"] for item in items], dtype=np.int64)
    folds = stratified_folds(items, N_FOLDS, FOLD_SEED)

    # ------------------------------------------------------- fast-path check
    from evidencegraph.effective_evidence import aggregate_stances

    checked = 0
    rng = np.random.default_rng(7)
    for index in rng.choice(len(items), size=min(12, len(items)), replace=False):
        item = items[int(index)]
        spectra = side_spectra(item["probabilities"], item["learned"], 0.05)
        matrix = _probability_matrix(item["probabilities"])
        stances = matrix[:, 0] - matrix[:, 2]
        cut = max(0.05, np.nextafter(0.0, 1.0))
        for name, mask in (
            ("support", stances >= cut),
            ("refute", -stances >= cut),
        ):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            values = np.abs(stances)[idx]
            dep = np.clip(item["learned"].copy(), 0.0, 1.0)
            np.fill_diagonal(dep, 1.0)
            covariance = covariance_from_dependence(
                dep[np.ix_(idx, idx)], variances=edge_variances(values), shrinkage=SHRINKAGE
            )
            for ridge in (0.0, 0.3, 10.0):
                for mode in RIDGE_MODES:
                    want = aggregate_stances(values, covariance, ridge=ridge, ridge_mode=mode)
                    got = spectra[name].stats(ridge, mode)
                    if not np.allclose(want, got, rtol=1e-9, atol=1e-11):
                        raise AssertionError(
                            f"fast path disagrees: {want} vs {got} ({name}, {ridge}, {mode})"
                        )
                    checked += 1
    print(f"fast-path eigen check: {checked} comparisons matched the frozen routine")

    # ------------------------------------------------------------ references
    references: dict[str, np.ndarray] = {}
    reference_logits: dict[str, np.ndarray] = {}
    for name, kwargs in (
        ("count_sum", {"use_dependence": False}),
        ("learned_soft_dependence", {}),
    ):
        logits = np.asarray(
            [
                score_episode(item["probabilities"], item["learned"], **kwargs).evidence_logit
                for item in items
            ]
        )
        reference_logits[name] = logits
        references[name] = oof_scalar(logits, labels, folds)

    # ------------------------------------------------------------- dev sweep
    # ``learned``  -- the paper's dependence matrix.
    # ``identity``  -- the decisive control: dependence modelling DELETED, ridge
    #                  and the stance split kept.  If the identity arm matches
    #                  the learned arm, the covariance is contributing nothing.
    dependence_sources = {
        "learned": [item["learned"] for item in items],
        "identity": [np.eye(len(item["probabilities"])) for item in items],
    }
    spectra_cache = {
        (source, tau): [
            side_spectra(item["probabilities"], dependence, tau)
            for item, dependence in zip(items, matrices)
        ]
        for source, matrices in dependence_sources.items()
        for tau in TAU_GRID
    }
    sweep: list[dict[str, Any]] = []
    oof_cache: dict[tuple[str, float, float, str, str], np.ndarray] = {}
    for source in dependence_sources:
        for tau in TAU_GRID:
            cache = spectra_cache[(source, tau)]
            for ridge in RIDGE_GRID:
                for mode in RIDGE_MODES:
                    if ridge == 0.0 and mode == "precision":
                        continue  # identical to ridge=0 covariance mode
                    scored = [score_record(s, ridge, mode) for s in cache]
                    for form in FORMS:
                        key = (source, tau, ridge, mode, form)
                        probabilities = oof_comparative(scored, labels, folds, form)
                        oof_cache[key] = probabilities
                        sweep.append(
                            {
                                "dependence": source,
                                "tau": tau,
                                "ridge": ridge,
                                "ridge_mode": mode,
                                "form": form,
                                "oof_log_loss": log_loss(labels, probabilities),
                                "oof_accuracy": float(np.mean((probabilities >= 0.5) == labels)),
                                "oof_brier": float(np.mean((probabilities - labels) ** 2)),
                                "oof_mix_accuracy": cell_accuracy(
                                    items, probabilities, "support_refute_mix"
                                ),
                            }
                        )
    sweep.sort(key=lambda row: row["oof_log_loss"])
    learned_sweep = [row for row in sweep if row["dependence"] == "learned"]
    identity_sweep = [row for row in sweep if row["dependence"] == "identity"]
    best = learned_sweep[0]
    print("dev-selected configuration:", json.dumps(best, indent=2, sort_keys=True))

    # -------------------------------------------- signed (no stance split) arms
    signed_rows: list[dict[str, Any]] = []
    signed_oof: dict[tuple[str, float, str], np.ndarray] = {}
    signed_logits_cache: dict[tuple[str, float, str], np.ndarray] = {}
    for source in ("learned", "identity"):
        for ridge in RIDGE_GRID:
            for mode in RIDGE_MODES:
                if ridge == 0.0 and mode == "precision":
                    continue
                logits = np.asarray(
                    [
                        score_episode(
                            item["probabilities"],
                            item["learned"],
                            use_dependence=(source == "learned"),
                            ridge=ridge,
                            ridge_mode=mode,
                        ).evidence_logit
                        for item in items
                    ]
                )
                probabilities = oof_scalar(logits, labels, folds)
                signed_oof[(source, ridge, mode)] = probabilities
                signed_logits_cache[(source, ridge, mode)] = logits
                signed_rows.append(
                    {
                        "dependence": source,
                        "tau": None,
                        "ridge": ridge,
                        "ridge_mode": mode,
                        "form": "signed_scalar_2p",
                        "oof_log_loss": log_loss(labels, probabilities),
                        "oof_accuracy": float(np.mean((probabilities >= 0.5) == labels)),
                        "oof_brier": float(np.mean((probabilities - labels) ** 2)),
                        "oof_mix_accuracy": cell_accuracy(
                            items, probabilities, "support_refute_mix"
                        ),
                    }
                )
    signed_rows.sort(key=lambda row: row["oof_log_loss"])

    def pick(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
        candidates = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in filters.items())
        ]
        if not candidates:
            raise ValueError(f"no candidate for {filters}")
        return min(candidates, key=lambda row: row["oof_log_loss"])

    # Parsimony tie-break (dev-only, stated before test is touched): among
    # configurations whose OOF log-loss is within TOLERANCE of the argmin, take
    # the smallest ridge.  The log-loss surface is flat over three decades of
    # lambda, so the argmin alone is not a meaningful choice of lambda.
    TOLERANCE = 0.005
    near = [
        row
        for row in learned_sweep
        if row["oof_log_loss"] <= learned_sweep[0]["oof_log_loss"] + TOLERANCE
    ]
    parsimonious = min(near, key=lambda row: (row["ridge"], row["oof_log_loss"]))

    # Closed-form lambda -> infinity limit of the whole ridge family:
    # (Sigma + lambda I)^+ -> I / lambda, so theta -> the UNIFORM mean stance and
    # w_eff -> n / lambda.  The evidence logit is therefore sum_i s_i / sqrt(n),
    # which contains neither the dependence covariance nor the inverse-variance
    # edge weighting.  If this parameter-free arm matches the tuned ones, the
    # ridge family's gain is entirely the removal of those two ingredients.
    limit_logits = np.asarray(
        [
            float(
                np.sum(
                    _probability_matrix(item["probabilities"])[:, 0]
                    - _probability_matrix(item["probabilities"])[:, 2]
                )
                / np.sqrt(len(item["probabilities"]))
            )
            for item in items
        ]
    )
    limit_oof = oof_scalar(limit_logits, labels, folds)
    limit_calibrator = fit_logistic_calibrator(limit_logits, labels)

    # Every arm below is selected on dev out-of-fold log-loss alone.
    selections: dict[str, dict[str, Any]] = {
        # the candidate: stance split + learned dependence + ridge
        "stance_ridge_no_override": best,
        # decisive control A: same design, dependence DELETED (identity)
        "stance_ridge_identity_dependence": pick(identity_sweep),
        # decisive control B: ridge only, no stance split, learned dependence
        "signed_ridge_learned": pick(signed_rows, dependence="learned"),
        # decisive control C: ridge only, no stance split, no dependence
        "signed_ridge_identity_dependence": pick(signed_rows, dependence="identity"),
        # finding-2 alone: stance split, learned dependence, ridge OFF
        "stance_split_no_ridge": pick(learned_sweep, ridge=0.0),
        # the candidate under the parsimony tie-break
        "stance_ridge_parsimonious": parsimonious,
    }
    ablations = dict(selections)

    def realize(name: str, choice: dict[str, Any]) -> dict[str, np.ndarray]:
        if choice["form"] == "signed_scalar_2p":
            key = (choice["dependence"], choice["ridge"], choice["ridge_mode"])
            logits = signed_logits_cache[key]
            calibrator = fit_logistic_calibrator(logits, labels)
            return {
                "oof": signed_oof[key],
                "insample": calibrator.predict(logits),
            }
        key = (
            choice["dependence"],
            choice["tau"],
            choice["ridge"],
            choice["ridge_mode"],
            choice["form"],
        )
        records = [
            score_record(s, choice["ridge"], choice["ridge_mode"])
            for s in spectra_cache[(choice["dependence"], choice["tau"])]
        ]
        design = np.asarray([comparative_features(r, choice["form"]) for r in records])
        fitted = fit_comparative_model(design, labels, form=choice["form"])
        if name == "stance_ridge_no_override":
            selected_model.append(fitted.to_dict())
        return {"oof": oof_cache[key], "insample": fitted.predict(design)}

    selected_model: list[dict[str, Any]] = []
    arms: dict[str, dict[str, np.ndarray]] = {
        name: realize(name, choice) for name, choice in selections.items()
    }
    for name in ("count_sum", "learned_soft_dependence"):
        logits = reference_logits[name]
        calibrator = fit_logistic_calibrator(logits, labels)
        arms[name] = {
            "oof": references[name],
            "insample": calibrator.predict(logits),
        }
    arms["uniform_mean_sqrt_n_limit"] = {
        "oof": limit_oof,
        "insample": limit_calibrator.predict(limit_logits),
    }
    tau, ridge, mode, form = best["tau"], best["ridge"], best["ridge_mode"], best["form"]
    model_dict = selected_model[0] if selected_model else None

    # -------------------------------------------------------------- reporting
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "selection_split": "dev",
        "evaluation_split": "dev",
        "test_split_touched": False,
        "n_dev_episodes": len(items),
        "folds": {"n_folds": N_FOLDS, "seed": FOLD_SEED, "stratified_by": "label x corruption"},
        "grids": {
            "tau": list(TAU_GRID),
            "ridge": list(RIDGE_GRID),
            "ridge_mode": list(RIDGE_MODES),
            "form": list(FORMS),
        },
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "nli": [source_descriptor(path, PROJECT_ROOT) for path in args.nli],
            "dependence": source_descriptor(args.dependence, PROJECT_ROOT),
        },
        "selected": {
            "tau": tau,
            "ridge": ridge,
            "ridge_mode": mode,
            "form": form,
            "model": model_dict,
            **{k: best[k] for k in ("oof_log_loss", "oof_accuracy", "oof_brier", "oof_mix_accuracy")},
        },
        "ablations": ablations,
        "parsimonious_selection": {"tolerance_log_loss": TOLERANCE, **parsimonious},
        "sweep_top_20": sweep[:20],
        "identity_sweep_top_10": identity_sweep[:10],
        "signed_ridge_sweep": signed_rows,
        "arms": {},
        "paired_vs_count_sum": {},
    }

    predictions: list[dict[str, Any]] = []
    for name, values in arms.items():
        entry: dict[str, Any] = {}
        for mode_name in ("oof", "insample"):
            records = build_records(items, values[mode_name])
            stats = summarize_episode_predictions(
                records, n_resamples=args.bootstrap_resamples, seed=911
            )
            entry[mode_name] = {
                cell: {
                    metric: stats["groups"][cell][metric]
                    for metric in (
                        "accuracy",
                        "accuracy_low",
                        "accuracy_high",
                        "brier",
                        "brier_low",
                        "brier_high",
                        "ece",
                        "false_acceptance",
                        "n_claims",
                    )
                }
                for cell in CELLS
                if cell in stats["groups"]
            }
            entry[mode_name]["log_loss"] = log_loss(labels, values[mode_name])
            if mode_name == "oof":
                for item, p in zip(items, values[mode_name]):
                    predictions.append(
                        {
                            "claim_id": item["claim_id"],
                            "split": "dev",
                            "label": item["label"],
                            "corruption": item["corruption"],
                            "variant": name,
                            "estimate": "out_of_fold",
                            "probability_true": float(p),
                        }
                    )
            else:
                for item, p in zip(items, values[mode_name]):
                    predictions.append(
                        {
                            "claim_id": item["claim_id"],
                            "split": "dev",
                            "label": item["label"],
                            "corruption": item["corruption"],
                            "variant": name,
                            "estimate": "dev_fitted",
                            "probability_true": float(p),
                        }
                    )
        summary["arms"][name] = entry

    for name, values in arms.items():
        if name == "count_sum":
            continue
        summary["paired_vs_count_sum"][name] = {
            mode_name: baselines._paired(
                build_records(items, values[mode_name]),
                build_records(items, arms["count_sum"][mode_name]),
                n_resamples=args.bootstrap_resamples,
                seed=4242,
            )
            for mode_name in ("oof", "insample")
        }

    # Does the candidate's extra machinery earn its keep?  Paired dev bootstrap
    # of the candidate against every simpler arm in the same family.
    summary["paired_candidate_vs"] = {
        name: {
            mode_name: baselines._paired(
                build_records(items, arms["stance_ridge_no_override"][mode_name]),
                build_records(items, values[mode_name]),
                n_resamples=args.bootstrap_resamples,
                seed=8686,
            )
            for mode_name in ("oof", "insample")
        }
        for name, values in arms.items()
        if name != "stance_ridge_no_override"
    }

    # Fold-seed robustness: re-run the OOF estimate of every *already selected*
    # configuration under four further fold partitions.  Selection is not
    # repeated; this measures how much of the reported margin is fold noise.
    robustness: dict[str, Any] = {}
    for seed in (11, 23, 47, 91):
        alt_folds = stratified_folds(items, N_FOLDS, seed)
        entry: dict[str, Any] = {}
        count_p = oof_scalar(reference_logits["count_sum"], labels, alt_folds)
        entry["count_sum"] = {
            "accuracy": float(np.mean((count_p >= 0.5) == labels)),
            "brier": float(np.mean((count_p - labels) ** 2)),
            "mix_accuracy": cell_accuracy(items, count_p, "support_refute_mix"),
        }
        for name, choice in selections.items():
            if choice["form"] == "signed_scalar_2p":
                logits = signed_logits_cache[
                    (choice["dependence"], choice["ridge"], choice["ridge_mode"])
                ]
                probabilities = oof_scalar(logits, labels, alt_folds)
            else:
                records = [
                    score_record(s, choice["ridge"], choice["ridge_mode"])
                    for s in spectra_cache[(choice["dependence"], choice["tau"])]
                ]
                probabilities = oof_comparative(records, labels, alt_folds, choice["form"])
            entry[name] = {
                "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
                "brier": float(np.mean((probabilities - labels) ** 2)),
                "mix_accuracy": cell_accuracy(items, probabilities, "support_refute_mix"),
                "accuracy_minus_count_sum": float(
                    np.mean((probabilities >= 0.5) == labels)
                    - np.mean((count_p >= 0.5) == labels)
                ),
            }
        limit_p = oof_scalar(limit_logits, labels, alt_folds)
        entry["uniform_mean_sqrt_n_limit"] = {
            "accuracy": float(np.mean((limit_p >= 0.5) == labels)),
            "brier": float(np.mean((limit_p - labels) ** 2)),
            "mix_accuracy": cell_accuracy(items, limit_p, "support_refute_mix"),
            "accuracy_minus_count_sum": float(
                np.mean((limit_p >= 0.5) == labels) - np.mean((count_p >= 0.5) == labels)
            ),
        }
        robustness[f"fold_seed_{seed}"] = entry
    summary["fold_seed_robustness"] = robustness

    predictions_path = args.output_dir / "dev_predictions.jsonl"
    write_jsonl(predictions, predictions_path)
    summary["predictions"] = source_descriptor(predictions_path, PROJECT_ROOT)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    headline = {
        name: {
            mode_name: {
                "accuracy": round(entry[mode_name]["overall"]["accuracy"], 4),
                "brier": round(entry[mode_name]["overall"]["brier"], 4),
                "log_loss": round(entry[mode_name]["log_loss"], 4),
                "mix_accuracy": round(
                    entry[mode_name]["support_refute_mix"]["accuracy"], 4
                ),
            }
            for mode_name in ("oof", "insample")
        }
        for name, entry in summary["arms"].items()
    }
    for name, block in summary["paired_vs_count_sum"].items():
        for mode_name in ("oof", "insample"):
            for cell in ("overall", "support_refute_mix"):
                headline.setdefault(f"{name}_minus_count_sum", {})[f"{mode_name}_{cell}_accuracy"] = {
                    key: round(block[mode_name][cell]["accuracy"][key], 4)
                    for key in ("difference", "low", "high")
                }
    with (args.output_dir / "headline.json").open("w", encoding="utf-8") as handle:
        json.dump(headline, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
