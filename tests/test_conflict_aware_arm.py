"""Unit tests for the transparent conflict-aware aggregation arm."""

from __future__ import annotations

import math

import numpy as np
import pytest

from run_episode_baselines import (
    CONFLICT_EXPONENT,
    COVERAGE_EXPONENT,
    UNCERTAINTY_EXPONENT,
    VARIANTS,
    conflict_aware_features,
    conflict_aware_logit,
    score_variant,
)


def _edge(support: float, neutral: float, refute: float) -> dict[str, float]:
    return {"support": support, "neutral": neutral, "refute": refute}


UNANIMOUS = [_edge(0.9, 0.05, 0.05)] * 4
SPLIT = [_edge(0.9, 0.05, 0.05)] * 2 + [_edge(0.05, 0.05, 0.9)] * 2
VAGUE = [_edge(0.05, 0.9, 0.05)] * 4


def test_variant_registry_contains_both_conflict_aware_arms() -> None:
    assert "conflict_aware_vote" in VARIANTS
    assert "conflict_aware_xbackbone" in VARIANTS


def test_conflict_and_agreement_satisfy_the_stated_identity() -> None:
    """``|N| = 1 - K`` exactly, which is what makes the discount transparent."""
    for pool in (UNANIMOUS, SPLIT, VAGUE, [_edge(0.7, 0.2, 0.1), _edge(0.1, 0.3, 0.6)]):
        features = conflict_aware_features(pool)
        assert abs(abs(features["agreement"]) - (1.0 - features["conflict"])) < 1e-12


def test_features_separate_unanimity_conflict_and_vagueness() -> None:
    unanimous = conflict_aware_features(UNANIMOUS)
    split = conflict_aware_features(SPLIT)
    vague = conflict_aware_features(VAGUE)
    assert unanimous["conflict"] == pytest.approx(0.0, abs=1e-9)
    assert split["conflict"] == pytest.approx(1.0, abs=1e-9)
    assert unanimous["agreement"] > 0.99
    assert split["agreement"] == pytest.approx(0.0, abs=1e-9)
    # A vague pool carries almost no coverage but nearly all uncertainty.
    assert vague["coverage"] < unanimous["coverage"]
    assert vague["uncertainty"] > unanimous["uncertainty"]
    # A pool with no informative mass carries no evidence in either direction.
    assert vague["uncertainty"] == pytest.approx(1.0)
    assert conflict_aware_logit(vague) == pytest.approx(0.0)
    for features in (unanimous, split, vague):
        assert 0.0 <= features["conflict"] <= 1.0
        assert -1.0 <= features["agreement"] <= 1.0
        assert 0.0 <= features["uncertainty"] <= 1.0


def test_logit_is_signed_zero_on_perfect_conflict_and_monotone_in_coverage() -> None:
    assert conflict_aware_logit(conflict_aware_features(SPLIT)) == pytest.approx(0.0, abs=1e-9)
    assert conflict_aware_logit(conflict_aware_features(UNANIMOUS)) > 0.0
    refuting = [_edge(0.05, 0.05, 0.9)] * 4
    assert conflict_aware_logit(conflict_aware_features(refuting)) < 0.0
    small = conflict_aware_logit(conflict_aware_features(UNANIMOUS[:2]))
    large = conflict_aware_logit(conflict_aware_features(UNANIMOUS))
    assert large > small > 0.0


def test_logit_matches_the_documented_closed_form() -> None:
    features = conflict_aware_features([_edge(0.7, 0.2, 0.1), _edge(0.1, 0.3, 0.6)])
    expected = (
        math.copysign(abs(features["agreement"]) ** CONFLICT_EXPONENT, features["agreement"])
        * features["coverage"] ** COVERAGE_EXPONENT
        * (1.0 - features["uncertainty"]) ** UNCERTAINTY_EXPONENT
    )
    assert conflict_aware_logit(features) == pytest.approx(expected)


def test_cross_backbone_feature_is_zero_without_and_positive_with_disagreement() -> None:
    assert conflict_aware_features(UNANIMOUS)["cross_backbone"] == 0.0
    assert conflict_aware_features(UNANIMOUS, UNANIMOUS)["cross_backbone"] == pytest.approx(0.0)
    disagreeing = conflict_aware_features(UNANIMOUS, [_edge(0.05, 0.05, 0.9)] * 4)
    assert disagreeing["cross_backbone"] > 1.0
    # The penalty can only shrink the magnitude of the logit, never flip it.
    plain = conflict_aware_logit(disagreeing)
    penalised = conflict_aware_logit(disagreeing, use_cross_backbone=True)
    assert 0.0 < penalised < plain


def test_secondary_probabilities_must_align() -> None:
    with pytest.raises(ValueError):
        conflict_aware_features(UNANIMOUS, UNANIMOUS[:2])


def test_score_variant_emits_the_shared_schema_and_no_probability_override() -> None:
    learned = np.eye(len(UNANIMOUS))
    record = score_variant("conflict_aware_vote", UNANIMOUS, learned, [], None)
    for key in ("stance_estimate", "effective_evidence", "evidence_logit", "raw_source_count"):
        assert key in record
    # The arm reports a raw logit only; calibration stays with the two-parameter
    # dev-fitted LogisticCalibrator applied by the runner.
    assert "probability_true" not in record
    assert record["sources_kept"] == len(UNANIMOUS)
    assert record["conflict"] == pytest.approx(0.0, abs=1e-9)


def test_cross_backbone_variant_requires_a_secondary_run() -> None:
    with pytest.raises(ValueError):
        score_variant("conflict_aware_xbackbone", UNANIMOUS, np.eye(len(UNANIMOUS)), [], None)
