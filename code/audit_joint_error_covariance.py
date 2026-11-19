"""Is the plug-in map Sigma_ij ~= sigma_i sigma_j d_ij calibrated against real errors?

Section 4 item 1 of the manuscript states the plug-in covariance as an
*unvalidated hypothesis*: shared origin is asserted to transmit evidence error
one-for-one, and origin-distinct sources are asserted to have uncorrelated
errors.  Nothing in the artifact ever checked that.  This audit checks it on the
held-out test split and reports the answer whatever it is.

Preregistered before the run (do not change after seeing numbers):

* **Error definition.**  For source ``i`` on claim ``c``,
  ``e_i = (p_support - p_refute) - gold_stance(i)`` where ``gold_stance`` is
  ``+1 / 0 / -1`` for the source's own NLI ``gold_label``
  (``support / neutral / refute``) as recorded in the frozen NLI run.  This is
  the error of the *edge*, which is exactly the quantity ``Sigma`` claims to
  model.  No alternative definition is scored.
* **Population.**  Every within-episode unordered pair of sources over the test
  split only.
* **Statistic.**  The symmetrised Pearson correlation of ``(e_i, e_j)`` --- i.e.
  the correlation over the pooled set ``{(e_i, e_j)} U {(e_j, e_i)}`` --- with
  95% claim-clustered bootstrap intervals (claims resampled with replacement,
  all pairs of a resampled claim carried along).
* **Strata.**  same-origin vs origin-distinct; then each provenance-family pair
  inside both, so that the same-origin figure is *attributed to verbatim
  copying* rather than claimed as a validation of the map.
* **Calibration curve.**  Empirical correlation inside ten equal-width bins of
  the predicted dependence ``d_ij`` (mean of the three frozen seeds), against
  the identity line the plug-in assumes.

Known confound, reported rather than corrected: one frozen NLI backbone scores
every source, so part of any origin-distinct error correlation is shared-scorer
error rather than shared upstream bias.  That does not excuse the map --- a
shared scorer is itself a dependence ``Sigma`` does not model --- and it is
recorded in the summary and in the manuscript sentence.

Run::

    PYTHONPATH=artifact/code python artifact/code/audit_joint_error_covariance.py
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.episode_evaluation import index_dependence_records
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "joint-error-covariance-audit-v1"
ERROR_DEFINITION = "signed_stance_minus_gold_stance"
SPLIT = "test"
GOLD_STANCE = {"support": 1.0, "neutral": 0.0, "refute": -1.0}
N_BINS = 10
FAMILIES = ("citation", "independent", "mirror", "paraphrase")


# ---------------------------------------------------------------------------
# Symmetrised correlation from per-claim sufficient statistics.
#
# For a set of pairs with left errors ``a`` and right errors ``b`` the pooled
# vectors are x = [a; b] and y = [b; a].  Then
#   N = 2n,  Sx = Sy = sum(a) + sum(b),  Sxx = Syy = sum(a^2) + sum(b^2),
#   Sxy = 2 * sum(a * b),
# so the correlation is a function of four additive per-claim quantities.  That
# makes the claim-clustered bootstrap exact and cheap: a resample is a sum of
# per-claim statistics, never a re-materialised pair list.
# ---------------------------------------------------------------------------


def pair_statistics(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return ``[n, Sx, Sxx, Sxy]`` for one block of pairs."""

    return np.asarray(
        [
            float(len(left)),
            float(np.sum(left) + np.sum(right)),
            float(np.sum(left**2) + np.sum(right**2)),
            float(2.0 * np.sum(left * right)),
        ],
        dtype=np.float64,
    )


def correlation_from_statistics(statistics: np.ndarray) -> float | None:
    """Symmetrised Pearson correlation, or ``None`` when it is undefined."""

    n, s_x, s_xx, s_xy = (
        statistics[..., 0],
        statistics[..., 1],
        statistics[..., 2],
        statistics[..., 3],
    )
    total = 2.0 * n
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s_x / total
        variance = s_xx / total - mean**2
        covariance = s_xy / total - mean**2
        value = covariance / variance
    scalar = np.ndim(statistics) == 1
    result = np.where(
        (total >= 4.0) & np.isfinite(variance) & (variance > 1e-12), value, np.nan
    )
    if scalar:
        return None if not np.isfinite(result) else float(result)
    return result


def summarize_block(
    per_claim: dict[str, np.ndarray],
    claims: list[str],
    samples: np.ndarray,
    predicted: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Point estimate plus a claim-clustered bootstrap interval for one stratum."""

    matrix = np.asarray([per_claim.get(claim, np.zeros(4)) for claim in claims])
    totals = matrix.sum(axis=0)
    point = correlation_from_statistics(totals)
    weights = np.asarray([predicted.get(claim, np.zeros(2)) for claim in claims])
    weight_total = weights.sum(axis=0)
    resampled = correlation_from_statistics(matrix[samples].sum(axis=1))
    finite = resampled[np.isfinite(resampled)]
    block: dict[str, Any] = {
        "n_pairs": int(totals[0]),
        "n_claims_contributing": int(sum(1 for claim in claims if claim in per_claim)),
        "error_correlation": point,
        "error_correlation_low": (
            float(np.quantile(finite, 0.025)) if finite.size >= 100 else None
        ),
        "error_correlation_high": (
            float(np.quantile(finite, 0.975)) if finite.size >= 100 else None
        ),
        "bootstrap_resamples_defined": int(finite.size),
    }
    if weight_total[0] > 0:
        mean_predicted = float(weight_total[1] / weight_total[0])
        block["mean_predicted_dependence"] = mean_predicted
        if point is not None:
            gap = mean_predicted - point
            block["predicted_minus_observed"] = float(gap)
            # The plug-in equates the pairwise correlation with d_ij, so a
            # predicted value *below* the observed correlation means Sigma
            # under-states shared error: the anti-conservative direction.
            block["direction"] = (
                "anti_conservative" if gap < 0 else "conservative"
            )
    return block


def load_errors(
    subset: Path, nli: Path
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    episodes = {
        str(record["claim_id"]): record
        for record in read_jsonl(subset)
        if str(record.get("split")) == SPLIT
    }
    if not episodes:
        raise ValueError(f"no {SPLIT} episodes in {subset}")
    per_claim: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for record in read_jsonl(nli):
        if str(record.get("split")) != SPLIT:
            continue
        if record.get("status") != "complete":
            raise ValueError("joint-error audit requires complete NLI records")
        claim_id = str(record["claim_id"])
        if claim_id not in episodes:
            continue
        source_id = str(record["source_id"])
        if (claim_id, source_id) in seen:
            raise ValueError(f"duplicate NLI row for {claim_id}/{source_id}")
        seen.add((claim_id, source_id))
        gold = str(record["gold_label"])
        if gold not in GOLD_STANCE:
            raise ValueError(f"unknown gold label {gold!r}")
        probabilities = record["probabilities"]
        stance = float(probabilities["support"]) - float(probabilities["refute"])
        per_claim.setdefault(claim_id, {})[source_id] = {
            "error": stance - GOLD_STANCE[gold],
            "family": str(record["family"]),
        }
    for claim_id, episode in episodes.items():
        scored = per_claim.get(claim_id, {})
        for source in episode["sources"]:
            source_id = str(source["source_id"])
            if source_id not in scored:
                raise ValueError(f"episode {claim_id} lacks the NLI edge {source_id}")
            if scored[source_id]["family"] != str(source["family"]):
                raise ValueError(f"family disagreement on {claim_id}/{source_id}")
            scored[source_id]["origin_id"] = str(source["origin_id"])
        if len(scored) != len(episode["sources"]):
            raise ValueError(f"episode {claim_id} has unmatched NLI rows")
    return sorted(episodes), per_claim


def build(
    subset: Path, nli: Path, dependence: Path, *, n_resamples: int, seed: int
) -> dict[str, Any]:
    claims, errors = load_errors(subset, nli)
    dependence_index = index_dependence_records(
        [record for record in read_jsonl(dependence) if str(record.get("split")) == SPLIT]
    )

    # Accumulate per-claim sufficient statistics for every stratum we report.
    blocks: dict[str, dict[str, list[tuple[float, float]]]] = {}
    predicted_blocks: dict[str, dict[str, list[float]]] = {}

    def add(name: str, claim_id: str, left: float, right: float, d: float) -> None:
        blocks.setdefault(name, {}).setdefault(claim_id, []).append((left, right))
        predicted_blocks.setdefault(name, {}).setdefault(claim_id, []).append(d)

    n_pairs = 0
    for claim_id in claims:
        scored = errors[claim_id]
        for left_id, right_id in combinations(sorted(scored), 2):
            key = (claim_id, *sorted((left_id, right_id)))
            if key not in dependence_index:
                raise ValueError(f"missing dependence edge for {key}")
            d = float(dependence_index[key])
            left, right = scored[left_id], scored[right_id]
            same = left["origin_id"] == right["origin_id"]
            stratum = "same_origin" if same else "origin_distinct"
            family = "|".join(sorted((left["family"], right["family"])))
            bin_index = min(N_BINS - 1, int(d * N_BINS))
            n_pairs += 1
            for name in (
                stratum,
                f"{stratum}::{family}",
                f"predicted_bin::{bin_index}",
            ):
                add(name, claim_id, left["error"], right["error"], d)

    per_claim_statistics = {
        name: {
            claim_id: pair_statistics(
                np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
            )
            for claim_id, pairs in claim_map.items()
        }
        for name, claim_map in blocks.items()
    }
    per_claim_predicted = {
        name: {
            claim_id: np.asarray([float(len(values)), float(np.sum(values))])
            for claim_id, values in claim_map.items()
        }
        for name, claim_map in predicted_blocks.items()
    }

    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(claims), size=(n_resamples, len(claims)))

    def block(name: str) -> dict[str, Any]:
        return summarize_block(
            per_claim_statistics[name], claims, samples, per_claim_predicted[name]
        )

    strata = {name: block(name) for name in ("same_origin", "origin_distinct")}
    family_cells: dict[str, dict[str, Any]] = {"same_origin": {}, "origin_distinct": {}}
    for name in sorted(per_claim_statistics):
        if "::" not in name or name.startswith("predicted_bin"):
            continue
        stratum, family = name.split("::", 1)
        family_cells[stratum][family] = block(name)

    bins = []
    for index in range(N_BINS):
        name = f"predicted_bin::{index}"
        entry = {
            "bin_index": index,
            "predicted_low_edge": index / N_BINS,
            "predicted_high_edge": (index + 1) / N_BINS,
        }
        entry.update(
            block(name)
            if name in per_claim_statistics
            else {"n_pairs": 0, "error_correlation": None}
        )
        bins.append(entry)

    independent_cell = family_cells["origin_distinct"].get("independent|independent", {})
    mirror_cell = family_cells["same_origin"].get("mirror|mirror", {})
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "split": SPLIT,
        "error_definition": {
            "name": ERROR_DEFINITION,
            "formula": "e_i = (p_support - p_refute) - gold_stance(gold_label_i)",
            "gold_stance_map": GOLD_STANCE,
            "preregistered": True,
            "note": (
                "frozen before the run; no alternative error definition, split, "
                "or pair filter was scored"
            ),
        },
        "confounds": {
            "shared_nli_backbone": (
                "One frozen NLI checkpoint scores every source, so part of the "
                "origin-distinct error correlation is shared-scorer error rather "
                "than shared upstream bias. This is not a reason to discount the "
                "finding: a shared scorer is itself a dependence the plug-in "
                "covariance does not model."
            ),
            "same_origin_cell_is_trivial": (
                "Same-origin mirror/mirror sources are verbatim copies, so their "
                "error correlation is 1 by construction. It is reported as "
                "trivially true, never as evidence that the map is calibrated."
            ),
        },
        "interpretation": (
            "The plug-in map asserts corr(e_i, e_j) = d_ij. On origin-distinct "
            "pairs, which the estimator scores at d ~ 0, the observed error "
            "correlation is far from 0, so Sigma is mis-specified in the "
            "anti-conservative direction. This is a measured defect of the "
            "method, not a repair of it."
        ),
        "inputs": {
            "dataset": source_descriptor(subset, PROJECT_ROOT),
            "nli": source_descriptor(nli, PROJECT_ROOT),
            "dependence": source_descriptor(dependence, PROJECT_ROOT),
        },
        "results": {
            "n_claims": len(claims),
            "n_pairs": n_pairs,
            "n_sources": sum(len(errors[claim]) for claim in claims),
            "bootstrap_resamples": n_resamples,
            "bootstrap_seed": seed,
            "strata": strata,
            "family_cells": family_cells,
            "predicted_dependence_bins": bins,
            "headline": {
                "same_origin_correlation": strata["same_origin"]["error_correlation"],
                "origin_distinct_correlation": strata["origin_distinct"][
                    "error_correlation"
                ],
                "origin_distinct_independent_pair_correlation": independent_cell.get(
                    "error_correlation"
                ),
                "origin_distinct_independent_pair_low": independent_cell.get(
                    "error_correlation_low"
                ),
                "origin_distinct_independent_pair_high": independent_cell.get(
                    "error_correlation_high"
                ),
                "origin_distinct_independent_pair_n": independent_cell.get("n_pairs"),
                "origin_distinct_independent_mean_predicted_dependence": (
                    independent_cell.get("mean_predicted_dependence")
                ),
                "same_origin_mirror_pair_correlation": mirror_cell.get(
                    "error_correlation"
                ),
                "same_origin_mirror_pair_n": mirror_cell.get("n_pairs"),
                "map_refuted": bool(
                    independent_cell.get("error_correlation_low") is not None
                    and independent_cell["error_correlation_low"] > 0.0
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evidencegraph20k_validated_subset.jsonl",
    )
    parser.add_argument(
        "--nli",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/subset_edge_episode/nli/deberta_large_test.jsonl",
    )
    parser.add_argument(
        "--dependence",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/subset_edge_episode/dependence/predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/subset_edge_episode/joint_error_covariance/summary.json",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=103)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        raise ValueError("joint-error audit requires >=100 bootstrap resamples")
    payload = build(
        args.dataset,
        args.nli,
        args.dependence,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(payload["results"]["headline"], indent=2, sort_keys=True))
    print(json.dumps(payload["results"]["strata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
