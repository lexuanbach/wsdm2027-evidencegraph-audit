"""Stance-conditional effective evidence."""

from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.episode_inference import EDGE_VARIANCE_FLOOR, score_episode
from evidencegraph.stance_conditional import (
    ComparativeModel,
    both_stances_present,
    comparative_features,
    fit_comparative_model,
    log_loss,
    score_episode_by_stance,
)


def edge(stance: float, neutral: float = 0.0) -> dict[str, float]:
    rest = 1.0 - neutral
    return {
        "support": rest * (1.0 + stance) / 2.0,
        "neutral": neutral,
        "refute": rest * (1.0 - stance) / 2.0,
    }


def test_unanimous_support_matches_the_production_aggregate() -> None:
    """With one stance sign and no dependence, the split reduces to the base rule."""
    probabilities = [edge(0.9), edge(0.6), edge(0.4)]
    identity = np.eye(3)
    base = score_episode(probabilities, identity, use_dependence=False)
    split = score_episode_by_stance(probabilities, identity, use_dependence=False)
    assert split.support_stance == pytest.approx(base.stance_estimate)
    assert split.support_effective_evidence == pytest.approx(base.effective_evidence)
    assert split.support_evidence == pytest.approx(base.evidence_logit)
    assert split.refute_count == 0
    assert split.refute_effective_evidence == 0.0
    assert split.refute_evidence == 0.0


def test_independent_side_effective_evidence_is_the_precision_sum() -> None:
    probabilities = [edge(1.0), edge(1.0), edge(-1.0)]
    split = score_episode_by_stance(probabilities, np.eye(3), use_dependence=False)
    assert split.support_count == 2
    assert split.refute_count == 1
    assert split.support_effective_evidence == pytest.approx(2.0 / EDGE_VARIANCE_FLOOR)
    assert split.refute_effective_evidence == pytest.approx(1.0 / EDGE_VARIANCE_FLOOR)


def test_dependent_support_cluster_collapses_only_its_own_side() -> None:
    """Three exact clones on the support side count once; the refuter is untouched."""
    probabilities = [edge(1.0), edge(1.0), edge(1.0), edge(-1.0)]
    dependence = np.eye(4)
    for i in range(3):
        for j in range(3):
            dependence[i, j] = 1.0
    np.fill_diagonal(dependence, 1.0)
    split = score_episode_by_stance(probabilities, dependence, shrinkage=0.0)
    assert split.support_effective_evidence == pytest.approx(1.0 / EDGE_VARIANCE_FLOOR)
    assert split.refute_effective_evidence == pytest.approx(1.0 / EDGE_VARIANCE_FLOOR)
    assert split.difference == pytest.approx(0.0, abs=1e-9)


def test_threshold_drops_uninformative_edges() -> None:
    probabilities = [edge(0.9), edge(0.01), edge(-0.9)]
    kept = score_episode_by_stance(probabilities, np.eye(3), use_dependence=False)
    assert kept.support_count == 2 and kept.dropped_count == 0
    dropped = score_episode_by_stance(
        probabilities, np.eye(3), use_dependence=False, threshold=0.1
    )
    assert dropped.support_count == 1 and dropped.dropped_count == 1


def test_side_statistics_are_non_negative_and_symmetric_under_sign_flip() -> None:
    probabilities = [edge(0.8), edge(0.3), edge(-0.7)]
    mirrored = [edge(-0.8), edge(-0.3), edge(0.7)]
    a = score_episode_by_stance(probabilities, np.eye(3), use_dependence=False)
    b = score_episode_by_stance(mirrored, np.eye(3), use_dependence=False)
    assert a.support_evidence == pytest.approx(b.refute_evidence)
    assert a.refute_evidence == pytest.approx(b.support_evidence)
    assert a.difference == pytest.approx(-b.difference)


def test_dependence_matrix_must_align_and_be_symmetric() -> None:
    probabilities = [edge(0.5), edge(-0.5)]
    with pytest.raises(ValueError):
        score_episode_by_stance(probabilities, np.eye(3))
    asymmetric = np.asarray([[1.0, 0.9], [0.1, 1.0]])
    with pytest.raises(ValueError):
        score_episode_by_stance(probabilities, asymmetric)


@pytest.mark.parametrize("form", ["difference_2p", "asymmetric_3p", "generous_5p"])
def test_comparative_model_separates_a_linearly_separable_dev_split(form: str) -> None:
    records = []
    labels = []
    for index in range(40):
        positive = index % 2 == 0
        records.append(
            {
                "support_evidence": 3.0 if positive else 0.5,
                "refute_evidence": 0.5 if positive else 3.0,
                "support_effective_evidence": 9.0 if positive else 1.0,
                "refute_effective_evidence": 1.0 if positive else 9.0,
                "evidence_difference": 2.5 if positive else -2.5,
            }
        )
        labels.append(int(positive))
    design = np.asarray([comparative_features(r, form) for r in records])
    model = fit_comparative_model(design, labels, form=form)
    assert isinstance(model, ComparativeModel)
    predictions = model.predict(design)
    assert np.all((predictions >= 0.5) == np.asarray(labels, dtype=bool))
    assert log_loss(labels, predictions) < 0.3


def test_monotone_forms_keep_the_sign_constraints() -> None:
    """support raises the score, refutation lowers it."""
    records = []
    labels = []
    rng = np.random.default_rng(11)
    for _ in range(200):
        support = float(rng.uniform(0.0, 4.0))
        refute = float(rng.uniform(0.0, 4.0))
        records.append(
            {
                "support_evidence": support,
                "refute_evidence": refute,
                "evidence_difference": support - refute,
            }
        )
        labels.append(int(support - refute + rng.normal(0.0, 0.3) > 0.0))
    design = np.asarray([comparative_features(r, "asymmetric_3p") for r in records])
    model = fit_comparative_model(design, labels, form="asymmetric_3p")
    assert model.coefficients[0] >= 0.0
    assert model.coefficients[1] <= 0.0


def test_both_stances_present_respects_the_threshold() -> None:
    mixed = [edge(0.9), edge(-0.02)]
    assert both_stances_present(mixed)
    assert not both_stances_present(mixed, threshold=0.1)
    assert not both_stances_present([edge(0.9), edge(0.2)])


def test_unknown_comparative_form_is_rejected() -> None:
    with pytest.raises(ValueError):
        comparative_features({"evidence_difference": 1.0}, "nope")
    with pytest.raises(ValueError):
        fit_comparative_model(np.zeros((4, 1)), [0, 1, 0, 1], form="nope")
