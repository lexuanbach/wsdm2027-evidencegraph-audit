"""Duplicate-aware and oracle baselines on the same frozen episode inputs.

Tests whether *soft* learned dependence adds value over hard rules.
This runner scores every held-out episode under matched alternatives that see
exactly the same NLI edges:

* ``count_sum``                -- no dependence at all (count-based aggregation);
* ``domain_dedup_vote``        -- one representative per domain (URL/domain dedup);
* ``near_dup_cluster_vote``    -- one representative per textual near-duplicate
                                  cluster (mean of token cosine and char-4-gram
                                  Jaccard at the frozen 0.80 threshold);
* ``learned_hard_dependence``  -- learned pairwise probabilities thresholded at 0.5;
* ``learned_soft_dependence``  -- the EvidenceGraph aggregator (soft $w_{eff}$);
* ``oracle_origin_vote``       -- one representative per *known* origin;
* ``oracle_soft_dependence``   -- soft aggregator with the known-origin matrix;
* ``conflict_aware_vote``      -- SURE-RAG-style transparent aggregation: explicit
                                  conflict, coverage and uncertainty features, no
                                  covariance and no probability override;
* ``conflict_aware_xbackbone`` -- the same rule with a free cross-backbone
                                  disagreement feature from the secondary NLI run;
* ``semantic_dedup_vote``      -- SemDeDup-style semantic deduplication: one
                                  representative per connected component of the
                                  cosine-similarity graph over cached sentence
                                  embeddings, thresholded at a dev-selected tau;
* ``semantic_lexical_union_vote`` -- the union of the semantic and lexical
                                  near-duplicate graphs, i.e. the strongest
                                  purely *hard* independence prior available
                                  without covariance estimation.

Each variant gets its own two-parameter logistic calibrator fitted on the dev
split only; test metrics use 1,000 claim bootstraps, and paired claim
bootstraps compare every variant against the learned soft aggregator (and,
with ``--secondary-reference``, against a second named variant).  Oracle
variants read construction ``origin_id`` fields and are reference points, not
deployable systems.

The conflict-aware arms obey the same frozen-calibration contract as every
other arm: the aggregation rule itself is fixed and parameter-free, and the
only quantities fitted on dev are the two ``LogisticCalibrator`` parameters.
See ``conflict_aware_logit`` for the rule and its exponents.

The semantic-dedup arms obey the same contract.  Their only free quantity is
the cosine threshold ``tau`` (and the E5 prefix), fixed on the dev split by
``analyze_semantic_dedup_arm.py`` and frozen here as ``SEMANTIC_THRESHOLD`` /
``SEMANTIC_PREFIX``; the aggregation itself is the same parameter-free
representative vote that ``near_dup_cluster_vote`` uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.edge_evaluation import source_from_record
from evidencegraph.episode_evaluation import (
    episode_inputs,
    index_dependence_records,
    index_nli_records,
    summarize_episode_predictions,
)
from evidencegraph.episode_inference import (
    EpisodeScore,
    fit_logistic_calibrator,
    score_episode,
)
from evidencegraph.metrics import expected_calibration_error
from evidencegraph.nli_evaluation import validate_nli_dataset
from evidencegraph.provenance import sha256
from evidencegraph.result_registry import source_descriptor
from evidencegraph.retrieval_graph import (
    NEAR_DUPLICATE_THRESHOLD,
    representative_indices,
    typed_source_pair_edges,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "episode-duplicate-aware-baselines-v1"
VARIANTS = (
    "count_sum",
    "domain_dedup_vote",
    "near_dup_cluster_vote",
    "learned_hard_dependence",
    "learned_soft_dependence",
    "oracle_origin_vote",
    "oracle_soft_dependence",
    "conflict_aware_vote",
    "conflict_aware_xbackbone",
    "semantic_dedup_vote",
    "semantic_lexical_union_vote",
)
REFERENCE = "learned_soft_dependence"
SECONDARY_NLI_VARIANTS = ("conflict_aware_xbackbone",)
SEMANTIC_VARIANTS = ("semantic_dedup_vote", "semantic_lexical_union_vote")
# Cosine threshold and E5 prefix of the semantic-dedup arm, selected on the dev
# split only (see ``artifact/runs/revision/P3/threshold_selection.json``).
SEMANTIC_THRESHOLD = 0.999
SEMANTIC_PREFIX = "query"
# Exponents of the transparent conflict-aware rule, selected on the dev split
# only (see ``artifact/runs/revision/P2/form_selection.json``): conflict
# discount CONFLICT_EXPONENT, coverage COVERAGE_EXPONENT, uncertainty penalty
# UNCERTAINTY_EXPONENT, cross-backbone penalty CROSS_BACKBONE_EXPONENT.
CONFLICT_EXPONENT = 2.0
COVERAGE_EXPONENT = 1.0
UNCERTAINTY_EXPONENT = 1.0
CROSS_BACKBONE_EXPONENT = 1.0


def _group_matrix(keys: list[str]) -> np.ndarray:
    return np.asarray(
        [[1.0 if left == right else 0.0 for right in keys] for left in keys],
        dtype=np.float64,
    )


def _near_duplicate_matrix(sources: list[Any]) -> np.ndarray:
    n = len(sources)
    matrix = np.eye(n, dtype=np.float64)
    position = {source.source_id: index for index, source in enumerate(sources)}
    edges = typed_source_pair_edges(
        sources, near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD
    )["near_duplicate_edges"]
    for edge in edges:
        i, j = position[edge["left_id"]], position[edge["right_id"]]
        matrix[i, j] = matrix[j, i] = 1.0
    return matrix


def load_semantic_embeddings(path: Path, prefix: str) -> dict[str, np.ndarray]:
    """Load the cached unit-norm passage embeddings keyed by SHA-256 of the text."""

    with np.load(path, allow_pickle=False) as payload:
        field = f"embeddings_{prefix}"
        if field not in payload:
            raise ValueError(f"embedding cache {path} has no prefix {prefix!r}")
        keys = [str(key) for key in payload["keys"]]
        matrix = np.asarray(payload[field], dtype=np.float64)
    if matrix.shape[0] != len(keys):
        raise ValueError("embedding cache keys and vectors disagree")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embedding cache must contain unit-norm vectors")
    return dict(zip(keys, matrix))


def _semantic_matrix(
    sources: list[Any], embeddings: dict[str, np.ndarray], threshold: float
) -> np.ndarray:
    """Binary SemDeDup-style adjacency: cosine(v_i, v_j) >= tau.

    The vectors are unit-norm, so the cosine is a plain dot product.  The
    resulting 0/1 matrix is handed to the same ``representative_indices``
    connected-component routine that the lexical near-duplicate arm uses, which
    is why the two arms differ *only* in the similarity that defines an edge.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("semantic dedup threshold must be in (0, 1]")
    n = len(sources)
    vectors = np.vstack(
        [embeddings[hashlib.sha256(source.text.encode("utf-8")).hexdigest()] for source in sources]
    )
    cosine = vectors @ vectors.T
    matrix = np.where(cosine >= threshold, 1.0, 0.0)
    np.fill_diagonal(matrix, 1.0)
    return np.asarray(matrix, dtype=np.float64)


def _representative_subset(
    probabilities: list[dict[str, float]], grouping: np.ndarray
) -> list[int]:
    return representative_indices(
        probabilities, grouping, list(range(len(probabilities))), threshold=0.5
    )


def conflict_aware_features(
    probabilities: list[dict[str, float]],
    secondary: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    """Transparent conflict / coverage / uncertainty features (no covariance).

    Every quantity is a closed-form function of the frozen per-edge NLI
    posteriors; nothing here is fitted on any split.  Writing
    ``s_i = p_support - p_refute`` and re-using the *same* inverse-variance
    edge weights ``w_i = 1 / max(0.05, 1 - |s_i|)`` that ``count_sum`` uses:

    * ``support_mass``   ``A = sum_i w_i max(s_i, 0)``
    * ``refute_mass``    ``R = sum_i w_i max(-s_i, 0)``
    * ``coverage``       ``C = A + R`` -- total informative evidence mass
    * ``uncertainty``    ``U = sum_i w_i (1 - |s_i|) / sum_i w_i`` -- weighted
      mean neutral mass, in ``[0, 1]``; it reaches 1 only when every edge is
      exactly neutral, and the aggregator then returns a zero logit
    * ``conflict``       ``K = 2 min(A, R) / C`` -- 0 when the pool is
      unanimous, 1 when support and refute mass cancel exactly.  A pool with
      no informative mass at all (``C = 0``) is the limit of ``A = R``, so it
      takes ``K = 1``; the ``(1 - U)`` factor zeroes its logit regardless.
    * ``agreement``      ``N = (A - R) / C`` -- signed net agreement; note the
      identity ``|N| = 1 - K`` holds exactly (including at ``C = 0``), so a
      conflict discount ``(1 - K)^k`` is the same object as ``|N|^k``
    * ``cross_backbone`` ``X = mean_i |s_i - s'_i|`` -- disagreement against a
      second NLI backbone, when one is supplied (an uncertainty feature that
      costs no new model call).
    """
    matrix = np.asarray(
        [[float(r["support"]), float(r["neutral"]), float(r["refute"])] for r in probabilities],
        dtype=np.float64,
    )
    stances = matrix[:, 0] - matrix[:, 2]
    weights = 1.0 / np.maximum(0.05, 1.0 - np.abs(stances))
    support_mass = float(np.sum(weights * np.maximum(stances, 0.0)))
    refute_mass = float(np.sum(weights * np.maximum(-stances, 0.0)))
    coverage = support_mass + refute_mass
    uncertainty = float(np.sum(weights * (1.0 - np.abs(stances))) / np.sum(weights))
    conflict = 2.0 * min(support_mass, refute_mass) / coverage if coverage > 0.0 else 1.0
    agreement = (support_mass - refute_mass) / coverage if coverage > 0.0 else 0.0
    features = {
        "support_mass": support_mass,
        "refute_mass": refute_mass,
        "coverage": coverage,
        "uncertainty": uncertainty,
        "conflict": conflict,
        "agreement": agreement,
        "cross_backbone": 0.0,
    }
    if secondary is not None:
        other = np.asarray(
            [[float(r["support"]), float(r["neutral"]), float(r["refute"])] for r in secondary],
            dtype=np.float64,
        )
        if other.shape != matrix.shape:
            raise ValueError("secondary NLI probabilities must align with the primary run")
        features["cross_backbone"] = float(np.mean(np.abs(stances - (other[:, 0] - other[:, 2]))))
    return features


def conflict_aware_logit(
    features: dict[str, float], *, use_cross_backbone: bool = False
) -> float:
    """Combine the transparent features into one signed evidence logit.

    ``logit = sign(N) |N|^a * C^b * (1 - U)^g * (1 - X/2)^d``

    The exponents are frozen module constants chosen on the dev split alone.
    There is no covariance term and no probability override: the value is fed
    to the same two-parameter dev-fitted ``LogisticCalibrator`` that every
    other variant uses.
    """
    agreement = features["agreement"]
    magnitude = abs(agreement) ** CONFLICT_EXPONENT
    value = (
        math.copysign(magnitude, agreement)
        * features["coverage"] ** COVERAGE_EXPONENT
        * (1.0 - features["uncertainty"]) ** UNCERTAINTY_EXPONENT
    )
    if use_cross_backbone:
        value *= (1.0 - features["cross_backbone"] / 2.0) ** CROSS_BACKBONE_EXPONENT
    return float(value)


def score_variant(
    variant: str,
    probabilities: list[dict[str, float]],
    learned: np.ndarray,
    sources: list[Any],
    secondary: list[dict[str, float]] | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> dict[str, Any]:
    n = len(probabilities)
    if variant == "count_sum":
        score = score_episode(probabilities, np.eye(n), use_dependence=False)
        kept = list(range(n))
    elif variant == "domain_dedup_vote":
        grouping = _group_matrix([source.domain.casefold() for source in sources])
        kept = _representative_subset(probabilities, grouping)
        score = score_episode(
            [probabilities[i] for i in kept], np.eye(len(kept)), use_dependence=False
        )
    elif variant == "near_dup_cluster_vote":
        grouping = _near_duplicate_matrix(sources)
        kept = _representative_subset(probabilities, grouping)
        score = score_episode(
            [probabilities[i] for i in kept], np.eye(len(kept)), use_dependence=False
        )
    elif variant == "learned_hard_dependence":
        score = score_episode(probabilities, learned, hard_dependence_threshold=0.5)
        kept = list(range(n))
    elif variant == "learned_soft_dependence":
        score = score_episode(probabilities, learned)
        kept = list(range(n))
    elif variant == "oracle_origin_vote":
        grouping = _group_matrix([source.origin_id for source in sources])
        kept = _representative_subset(probabilities, grouping)
        score = score_episode(
            [probabilities[i] for i in kept], np.eye(len(kept)), use_dependence=False
        )
    elif variant == "oracle_soft_dependence":
        grouping = _group_matrix([source.origin_id for source in sources])
        score = score_episode(probabilities, grouping)
        kept = list(range(n))
    elif variant in SEMANTIC_VARIANTS:
        if embeddings is None:
            raise ValueError(f"{variant} requires --semantic-embeddings")
        grouping = _semantic_matrix(sources, embeddings, semantic_threshold)
        if variant == "semantic_lexical_union_vote":
            grouping = np.maximum(grouping, _near_duplicate_matrix(sources))
        kept = _representative_subset(probabilities, grouping)
        score = score_episode(
            [probabilities[i] for i in kept], np.eye(len(kept)), use_dependence=False
        )
    elif variant in ("conflict_aware_vote", "conflict_aware_xbackbone"):
        cross = variant == "conflict_aware_xbackbone"
        if cross and secondary is None:
            raise ValueError(f"{variant} requires --nli-secondary probabilities")
        features = conflict_aware_features(probabilities, secondary if cross else None)
        logit = conflict_aware_logit(features, use_cross_backbone=cross)
        # ``stance_estimate``/``effective_evidence`` keep the shared schema:
        # net agreement plays the role of the aggregated stance and the
        # informative evidence mass plays the role of w_eff.
        score = EpisodeScore(
            stance_estimate=features["agreement"],
            effective_evidence=features["coverage"],
            evidence_logit=logit,
            raw_source_count=n,
        )
        return {**score.to_dict(), "sources_kept": n, **features}
    else:
        raise ValueError(f"unknown baseline variant: {variant}")
    return {**score.to_dict(), "sources_kept": len(kept)}


def _paired(
    reference: list[dict[str, Any]],
    other: list[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    ref = {record["claim_id"]: record for record in reference}
    oth = {record["claim_id"]: record for record in other}
    if set(ref) != set(oth):
        raise ValueError("paired comparison requires identical claim sets")
    claims = sorted(ref)
    y = np.asarray([int(ref[c]["label"]) for c in claims])
    a = np.asarray([float(ref[c]["probability_true"]) for c in claims])
    b = np.asarray([float(oth[c]["probability_true"]) for c in claims])
    corruption = np.asarray([str(ref[c]["corruption"]) for c in claims])

    def metrics(sample: np.ndarray) -> dict[str, float]:
        sy, sa, sb = y[sample], a[sample], b[sample]
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
        pos = sy == 1
        if np.any(pos):
            out["true_claim_accuracy"] = float(
                np.mean(sa[pos] >= 0.5) - np.mean(sb[pos] >= 0.5)
            )
        return out

    groups = {"overall": np.ones(len(claims), dtype=bool)}
    for name in sorted(set(corruption)):
        groups[name] = corruption == name
    output: dict[str, Any] = {}
    for group_index, (name, mask) in enumerate(groups.items()):
        indices = np.flatnonzero(mask)
        point = metrics(indices)
        rng = np.random.default_rng(seed + group_index)
        boots: dict[str, list[float]] = {key: [] for key in point}
        for _ in range(n_resamples):
            sample = rng.choice(indices, len(indices), replace=True)
            for key, value in metrics(sample).items():
                if key in boots:
                    boots[key].append(value)
        entry: dict[str, Any] = {"n_claims": int(len(indices))}
        for key, value in point.items():
            low, high = np.quantile(np.asarray(boots[key]), [0.025, 0.975])
            entry[key] = {"difference": value, "low": float(low), "high": float(high)}
        output[name] = entry
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--nli", type=Path, action="append", required=True)
    parser.add_argument(
        "--nli-secondary",
        type=Path,
        action="append",
        default=None,
        help="second NLI backbone, used only for the cross-backbone disagreement feature",
    )
    parser.add_argument("--dependence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(VARIANTS),
        help="comma-separated subset of the registered variants to score",
    )
    parser.add_argument(
        "--semantic-embeddings",
        type=Path,
        default=None,
        help="cached passage embeddings (.npz) from run_semantic_dedup_embeddings.py",
    )
    parser.add_argument(
        "--semantic-prefix",
        type=str,
        default=SEMANTIC_PREFIX,
        choices=("query", "passage"),
        help="which cached E5 prefix to use for passage-to-passage cosine",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=SEMANTIC_THRESHOLD,
        help="cosine threshold tau for the semantic-dedup arms (dev-selected)",
    )
    parser.add_argument(
        "--secondary-reference",
        type=str,
        default=None,
        help="extra variant to run paired claim bootstraps against (e.g. count_sum)",
    )
    args = parser.parse_args()
    variants = tuple(name.strip() for name in args.variants.split(",") if name.strip())
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown baseline variants: {unknown}")
    if REFERENCE not in variants:
        raise SystemExit(f"--variants must include the reference {REFERENCE}")
    if args.secondary_reference is not None and args.secondary_reference not in variants:
        raise SystemExit("--secondary-reference must name a scored variant")
    needs_secondary = [name for name in variants if name in SECONDARY_NLI_VARIANTS]
    if needs_secondary and not args.nli_secondary:
        raise SystemExit(f"variants {needs_secondary} require --nli-secondary")
    needs_embeddings = [name for name in variants if name in SEMANTIC_VARIANTS]
    if needs_embeddings and args.semantic_embeddings is None:
        raise SystemExit(f"variants {needs_embeddings} require --semantic-embeddings")
    episodes = read_jsonl(args.dataset)
    dataset_sha256 = sha256(args.dataset)
    nli_records = [record for path in args.nli for record in read_jsonl(path)]
    nli_metadata = validate_nli_dataset(nli_records, dataset_sha256)
    nli_index = index_nli_records(nli_records)
    secondary_index: dict[tuple[str, str], dict[str, float]] | None = None
    secondary_metadata: dict[str, Any] | None = None
    if args.nli_secondary:
        secondary_records = [
            record for path in args.nli_secondary for record in read_jsonl(path)
        ]
        secondary_metadata = validate_nli_dataset(secondary_records, dataset_sha256)
        secondary_index = index_nli_records(secondary_records)
    dependence_index = index_dependence_records(read_jsonl(args.dependence))
    embeddings: dict[str, np.ndarray] | None = None
    embedding_metadata: dict[str, Any] | None = None
    if args.semantic_embeddings is not None:
        embeddings = load_semantic_embeddings(args.semantic_embeddings, args.semantic_prefix)
        sidecar = args.semantic_embeddings.with_suffix(".json")
        if sidecar.exists():
            embedding_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if str(embedding_metadata.get("dataset_sha256")) != dataset_sha256:
                raise SystemExit("embedding cache was built from a different dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_variant: dict[str, dict[str, list[dict[str, Any]]]] = {
        variant: {"dev": [], "test": []} for variant in variants
    }
    for episode in episodes:
        split = str(episode["split"])
        if split not in ("dev", "test"):
            continue
        probabilities, learned = episode_inputs(episode, nli_index, dependence_index)
        sources = [source_from_record(record) for record in episode["sources"]]
        secondary = None
        if secondary_index is not None:
            claim_id = str(episode["claim_id"])
            secondary = [
                secondary_index[(claim_id, str(record["source_id"]))]
                for record in episode["sources"]
            ]
        for variant in variants:
            scored = score_variant(
                variant,
                probabilities,
                learned,
                sources,
                secondary,
                embeddings=embeddings,
                semantic_threshold=args.semantic_threshold,
            )
            per_variant[variant][split].append(
                {
                    "claim_id": str(episode["claim_id"]),
                    "split": split,
                    "label": int(bool(episode["label"])),
                    "corruption": str(episode["corruption"]),
                    "variant": variant,
                    **scored,
                }
            )
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "calibration_split": "dev",
        "evaluation_split": "test",
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "inputs": {
            "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
            "nli": [source_descriptor(path, PROJECT_ROOT) for path in args.nli],
            "nli_run": nli_metadata,
            "nli_secondary": [
                source_descriptor(path, PROJECT_ROOT) for path in (args.nli_secondary or [])
            ],
            "nli_secondary_run": secondary_metadata,
            "dependence": source_descriptor(args.dependence, PROJECT_ROOT),
        },
        "semantic_dedup": {
            "threshold": float(args.semantic_threshold),
            "prefix": args.semantic_prefix,
            "embeddings": (
                source_descriptor(args.semantic_embeddings, PROJECT_ROOT)
                if args.semantic_embeddings is not None
                else None
            ),
            "encoder": (
                {
                    "model": embedding_metadata["model"],
                    "model_revision": embedding_metadata["model_revision"],
                }
                if embedding_metadata is not None
                else None
            ),
        },
        "conflict_aware_exponents": {
            "conflict": CONFLICT_EXPONENT,
            "coverage": COVERAGE_EXPONENT,
            "uncertainty": UNCERTAINTY_EXPONENT,
            "cross_backbone": CROSS_BACKBONE_EXPONENT,
        },
        "scored_variants": list(variants),
        "variants": {},
        "paired_versus_reference": {},
        "reference": REFERENCE,
        "secondary_reference": args.secondary_reference,
    }
    all_predictions: list[dict[str, Any]] = []
    calibrated_test: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        dev = per_variant[variant]["dev"]
        test = per_variant[variant]["test"]
        calibrator = fit_logistic_calibrator(
            [r["evidence_logit"] for r in dev], [r["label"] for r in dev]
        )
        dev = [
            {**r, "probability_true": float(calibrator.predict([r["evidence_logit"]])[0])}
            for r in dev
        ]
        test = [
            {**r, "probability_true": float(calibrator.predict([r["evidence_logit"]])[0])}
            for r in test
        ]
        all_predictions.extend(dev)
        all_predictions.extend(test)
        calibrated_test[variant] = test
        labels = np.asarray([r["label"] for r in test])
        probs = np.asarray([r["probability_true"] for r in test])
        summary["variants"][variant] = {
            "calibrator": calibrator.to_dict(),
            "dev": summarize_episode_predictions(
                dev, n_resamples=args.bootstrap_resamples, seed=103
            ),
            "test": summarize_episode_predictions(
                test, n_resamples=args.bootstrap_resamples, seed=103
            ),
            "test_true_claim_accuracy": float(np.mean(probs[labels == 1] >= 0.5)),
            "test_mean_sources_kept": float(np.mean([r["sources_kept"] for r in test])),
            "test_mean_effective_evidence": float(
                np.mean([r["effective_evidence"] for r in test])
            ),
        }
    for variant in variants:
        if variant == REFERENCE:
            continue
        summary["paired_versus_reference"][variant] = _paired(
            calibrated_test[REFERENCE],
            calibrated_test[variant],
            n_resamples=args.bootstrap_resamples,
            seed=203,
        )
    if args.secondary_reference is not None:
        summary["paired_versus_secondary"] = {
            variant: _paired(
                calibrated_test[args.secondary_reference],
                calibrated_test[variant],
                n_resamples=args.bootstrap_resamples,
                seed=307,
            )
            for variant in variants
            if variant != args.secondary_reference
        }
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(all_predictions, predictions_path)
    summary["predictions"] = source_descriptor(predictions_path, PROJECT_ROOT)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                variant: {
                    "acc": round(summary["variants"][variant]["test"]["groups"]["overall"]["accuracy"], 4),
                    "ece": round(summary["variants"][variant]["test"]["groups"]["overall"]["ece"], 4),
                    "fa": round(summary["variants"][variant]["test"]["groups"]["overall"]["false_acceptance"], 4),
                }
                for variant in variants
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
