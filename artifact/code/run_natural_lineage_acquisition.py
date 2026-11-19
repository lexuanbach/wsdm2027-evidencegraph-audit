"""Acquisition under a fixed budget on *observed* Web lineage, not constructed copies.

The paper's one surviving positive result --- spend an acquisition step on a
predicted-independent source --- was measured on a construction that deliberately
hides gold evidence behind eight same-origin decoys. This experiment re-asks the
same question where lineage is *observed* rather than generated, and where the
endpoint is evidence gathering rather than a constructed label:

* AVeriTeC: candidates are real Web pages. Observed origin is the registered
  domain of the URL, after unwrapping ``web.archive.org/web/<ts>/<url>``
  snapshots, so syndicated copies and repeated snapshots of one outlet collapse
  to one origin.
* FEVER and HoVer: candidates are Wikipedia sentences. Observed origin is the
  page, so several sentences of one article count once; HoVer additionally
  *requires* evidence from multiple pages, making origin diversity a real need
  rather than a stylistic preference.

Every policy starts from the same frozen top-5 context and spends the same
three-step budget over the same frozen top-20 candidate pool. The endpoint is
gold-evidence recall of the final selected set, which needs no model call, so
the comparison is exact and fully offline.

Note that on natural lineage the observed-origin policy *is* an oracle that
happens to be free: provenance is readable from the URL or page id. That is
precisely the comparison the learned estimator has to justify itself against.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.dependence_features import (
    cache_source_features,
    pair_features_from_cache,
    FEATURE_NAMES,
)
from evidencegraph.dependence_model import DependenceEstimator
from evidencegraph.effective_evidence import covariance_from_dependence, effective_evidence
from evidencegraph.result_registry import source_descriptor
from evidencegraph.retrieval_evaluation import resolve_retrieval_attempts
from evidencegraph.retrieval_graph import NEAR_DUPLICATE_THRESHOLD


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# v1 wrote an empty ``domain`` on every shim source, pinning the estimator's
# ``same_domain`` feature to a constant 1.0.  v2 writes the observed origin.
PROTOCOL = "natural-lineage-acquisition-v2"
PROTOCOL_BLANK_DOMAIN = "natural-lineage-acquisition-v1"
POLICIES = (
    "passive_rank",
    "mmr_text",
    "neardup_diverse",
    "observed_origin_diverse",
    "learned_hard_dependence",
    "learned_soft_find_indep",
    "oracle_gold_ceiling",
)
ARCHIVE = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(?P<target>https?://.+)$")


def observed_origin(result: dict[str, Any], dataset: str) -> str:
    """Provenance that is *readable off the record*, never a construction label."""

    if dataset == "AVeriTeC":
        url = str(result.get("url", ""))
        match = ARCHIVE.match(url)
        if match:
            url = match.group("target")
        host = urlparse(url).netloc.casefold()
        host = host[4:] if host.startswith("www.") else host
        return host or f"unknown:{result.get('source_id')}"
    page = result.get("page_id")
    if page:
        return f"wiki:{page}"
    return str(result.get("source_id", "")).rsplit(":", 1)[0]


def gold_identity(result: dict[str, Any], dataset: str) -> str:
    return str(result["url"]) if dataset == "AVeriTeC" else str(result["source_id"])


def _similarity(caches, texts, i: int, j: int) -> tuple[float, float]:
    features = pair_features_from_cache(texts[i], texts[j], caches[i], caches[j])
    cosine = float(features[FEATURE_NAMES.index("token_embedding_cosine")])
    ngram = float(features[FEATURE_NAMES.index("char_4gram_jaccard")])
    return cosine, 0.5 * (cosine + ngram)


def acquire(
    policy: str,
    *,
    n: int,
    initial: list[int],
    budget: int,
    origins: list[str],
    dependence: np.ndarray,
    sim: np.ndarray,
    neardup: np.ndarray,
    mmr_lambda: float,
    gold_flags: np.ndarray,
) -> list[int]:
    selected = list(initial)
    remaining = [index for index in range(n) if index not in set(initial)]
    for _ in range(budget):
        if not remaining:
            break
        if policy == "passive_rank":
            choice = remaining[0]
        elif policy == "oracle_gold_ceiling":
            # Non-deployable ceiling: spend the budget on gold evidence if the
            # pool contains any that is still unselected. It bounds what *any*
            # acquisition policy could gain over passive rank.
            choice = next(
                (candidate for candidate in remaining if gold_flags[candidate]),
                remaining[0],
            )
        elif policy == "mmr_text":
            # Relevance is the frozen retrieval order, normalised to [0, 1].
            best, best_score = None, -np.inf
            for candidate in remaining:
                relevance = 1.0 - candidate / max(1, n - 1)
                redundancy = max(sim[candidate, index] for index in selected)
                score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
                if score > best_score:
                    best, best_score = candidate, score
            choice = best
        elif policy == "neardup_diverse":
            choice = next(
                (
                    candidate
                    for candidate in remaining
                    if all(not neardup[candidate, index] for index in selected)
                ),
                remaining[0],
            )
        elif policy == "observed_origin_diverse":
            seen = {origins[index] for index in selected}
            choice = next(
                (candidate for candidate in remaining if origins[candidate] not in seen),
                remaining[0],
            )
        elif policy == "learned_hard_dependence":
            choice = next(
                (
                    candidate
                    for candidate in remaining
                    if max(dependence[candidate, index] for index in selected) < 0.5
                ),
                remaining[0],
            )
        elif policy == "learned_soft_find_indep":
            # The deployed mechanism: maximise the marginal gain in effective
            # evidence, under unit marginal variances so the quantity is a
            # source count rather than a variance-scaled precision.
            base = effective_evidence(
                covariance_from_dependence(dependence[np.ix_(selected, selected)])
            )
            best, best_gain = None, -np.inf
            for candidate in remaining:
                subset = selected + [candidate]
                gain = (
                    effective_evidence(
                        covariance_from_dependence(dependence[np.ix_(subset, subset)])
                    )
                    - base
                )
                if gain > best_gain or (gain == best_gain and best is not None and candidate < best):
                    best, best_gain = candidate, gain
            choice = best
        else:
            raise ValueError(f"unknown acquisition policy: {policy}")
        selected.append(choice)
        remaining.remove(choice)
    return selected


def _bootstrap(
    values: dict[str, np.ndarray], reference: str, *, n_resamples: int, seed: int
) -> dict[str, Any]:
    n = len(next(iter(values.values())))
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, n, n) for _ in range(n_resamples)]
    output: dict[str, Any] = {}
    for policy, array in values.items():
        boots = np.asarray([float(np.mean(array[s])) for s in samples])
        low, high = np.quantile(boots, [0.025, 0.975])
        entry = {"mean": float(np.mean(array)), "low": float(low), "high": float(high)}
        if policy != reference:
            delta = array - values[reference]
            dboots = np.asarray([float(np.mean(delta[s])) for s in samples])
            dlow, dhigh = np.quantile(dboots, [0.025, 0.975])
            entry["difference_vs_reference"] = {
                "difference": float(np.mean(delta)),
                "low": float(dlow),
                "high": float(dhigh),
            }
        output[policy] = entry
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="name=cases=retrievals")
    parser.add_argument("--dependence-model", type=Path, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--initial-sources", type=int, default=5)
    parser.add_argument("--action-budget", type=int, default=3)
    parser.add_argument("--mmr-lambda", type=float, default=0.75)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--source-domain",
        choices=("observed", "blank"),
        default="observed",
        help=(
            "Provenance written into each shim source's `domain` field, which is "
            "what the estimator's `same_domain` feature compares. `observed` "
            "(default, correct) uses observed_origin(); `blank` reproduces the "
            "frozen v1 defect where every domain was the empty string, pinning "
            "same_domain to a constant 1.0."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.dependence_model) != 3:
        raise ValueError("natural-lineage acquisition uses the three frozen estimators")
    estimators = [DependenceEstimator.load(path) for path in args.dependence_model]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "protocol": PROTOCOL if args.source_domain == "observed" else PROTOCOL_BLANK_DOMAIN,
        "policies": list(POLICIES),
        "reference": "passive_rank",
        "initial_sources": args.initial_sources,
        "action_budget": args.action_budget,
        "top_k": args.top_k,
        "mmr_lambda": args.mmr_lambda,
        "source_domain": args.source_domain,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "inputs": {
            "dependence_models": [
                source_descriptor(path, PROJECT_ROOT) for path in args.dependence_model
            ]
        },
        "datasets": {},
    }
    all_rows: list[dict[str, Any]] = []
    for spec in args.dataset:
        name, cases_path, retrieval_path = spec.split("=", 2)
        cases = {str(c["case_id"]): c for c in read_jsonl(Path(cases_path))}
        retrievals = resolve_retrieval_attempts(read_jsonl(Path(retrieval_path)))
        dataset_label = None
        per_case: dict[str, dict[str, Any]] = {policy: {} for policy in POLICIES}
        origins_seen: list[int] = []
        eligible: list[str] = []
        for record in retrievals:
            case_id = str(record["case_id"])
            if case_id not in cases:
                continue
            case = cases[case_id]
            dataset_label = str(case["dataset"])
            gold = {str(g) for g in case.get("gold_evidence", [])}
            if not gold:
                continue
            groups = [
                {str(g) for g in group}
                for group in case.get("metadata", {}).get("evidence_groups", [])
            ] or [gold]
            results = list(record["results"])[: args.top_k]
            if len(results) < args.initial_sources + 1:
                continue
            n = len(results)
            origins = [observed_origin(r, dataset_label) for r in results]
            # The frozen v1 run left ``domain`` empty on every shim source, which
            # makes ``same_domain`` (dependence_features.py) a constant 1.0 --- a
            # defect, not a modelling choice.  ``--source-domain observed`` gives
            # each source the provenance that is readable off its own record (the
            # registered domain for AVeriTeC, the wiki page for FEVER/HoVer);
            # ``blank`` reproduces the frozen v1 behaviour for the disclosed
            # ablation.  The retrieval records carry no author, no timestamp and
            # no citation edges, so those three fields stay unpopulated rather
            # than invented; they were constant-zero in training as well.
            domains = origins if args.source_domain == "observed" else [""] * n
            texts = [
                type("S", (), {"text": str(r["text"]), "source_id": str(r["source_id"]),
                               "cited_source_ids": (), "domain": domain, "author": None,
                               "timestamp": None})()
                for r, domain in zip(results, domains)
            ]
            caches = [cache_source_features(t) for t in texts]
            sim = np.eye(n)
            neardup = np.zeros((n, n), dtype=bool)
            for i, j in combinations(range(n), 2):
                cosine, score = _similarity(caches, texts, i, j)
                sim[i, j] = sim[j, i] = cosine
                flag = score >= NEAR_DUPLICATE_THRESHOLD
                neardup[i, j] = neardup[j, i] = flag
            features = np.vstack(
                [
                    pair_features_from_cache(texts[i], texts[j], caches[i], caches[j])
                    for i, j in combinations(range(n), 2)
                ]
            )
            probs = np.mean(
                np.vstack([e.predict_proba(features) for e in estimators]), axis=0
            )
            dependence = np.eye(n)
            for (i, j), p in zip(combinations(range(n), 2), probs):
                dependence[i, j] = dependence[j, i] = float(p)
            identities = [gold_identity(r, dataset_label) for r in results]
            gold_flags = np.asarray([identity in gold for identity in identities])
            initial = list(range(args.initial_sources))
            eligible.append(case_id)
            origins_seen.append(len({origins[i] for i in range(n)}))
            for policy in POLICIES:
                selected = acquire(
                    policy,
                    n=n,
                    initial=initial,
                    budget=args.action_budget,
                    origins=origins,
                    dependence=dependence,
                    sim=sim,
                    neardup=neardup,
                    mmr_lambda=args.mmr_lambda,
                    gold_flags=gold_flags,
                )
                chosen = {identities[i] for i in selected}
                per_case[policy][case_id] = {
                    "item_recall": len(gold & chosen) / len(gold),
                    "complete_group_hit": float(
                        any(group <= chosen for group in groups)
                    ),
                    "distinct_origins": float(len({origins[i] for i in selected})),
                    "new_gold_after_initial": float(
                        len(gold & chosen)
                        - len(gold & {identities[i] for i in initial})
                    ),
                }
                all_rows.append(
                    {
                        "dataset": dataset_label,
                        "case_id": case_id,
                        "policy": policy,
                        "selected_ranks": [i + 1 for i in selected],
                        **per_case[policy][case_id],
                    }
                )
        if not eligible:
            raise ValueError(f"no eligible cases with gold evidence for {name}")
        metrics: dict[str, Any] = {
            "n_cases": len(eligible),
            "mean_distinct_origins_in_pool": float(np.mean(origins_seen)),
        }
        for metric in ("item_recall", "complete_group_hit", "distinct_origins", "new_gold_after_initial"):
            metrics[metric] = _bootstrap(
                {
                    policy: np.asarray([per_case[policy][c][metric] for c in eligible])
                    for policy in POLICIES
                },
                "passive_rank",
                n_resamples=args.bootstrap_resamples,
                seed=103,
            )
        summary["datasets"][name] = {
            "dataset": dataset_label,
            "inputs": {
                "cases": source_descriptor(Path(cases_path), PROJECT_ROOT),
                "retrievals": source_descriptor(Path(retrieval_path), PROJECT_ROOT),
            },
            **metrics,
        }
        print(
            f"{name} ({dataset_label}, n={len(eligible)}): "
            + ", ".join(
                f"{policy} {100 * metrics['item_recall'][policy]['mean']:.1f}"
                for policy in POLICIES
            )
        )
    rows_path = args.output_dir / "cases.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary["cases"] = source_descriptor(rows_path, PROJECT_ROOT)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
