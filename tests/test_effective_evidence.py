from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.effective_evidence import (
    covariance_from_dependence,
    effective_evidence,
    equicorrelation_covariance,
    gls_weights,
)


@pytest.mark.parametrize("n_sources", [1, 2, 5, 30])
def test_independent_sources_have_raw_count_effective_evidence(n_sources: int) -> None:
    assert effective_evidence(np.eye(n_sources)) == pytest.approx(n_sources)


@pytest.mark.parametrize("n_sources", [2, 5, 30])
def test_exact_duplicates_contribute_one_effective_source(n_sources: int) -> None:
    covariance = np.ones((n_sources, n_sources))
    assert effective_evidence(covariance) == pytest.approx(1.0)
    assert np.sum(gls_weights(covariance)) == pytest.approx(1.0)


@pytest.mark.parametrize("n_sources,rho", [(2, 0.2), (5, 0.5), (30, 0.8)])
def test_equicorrelation_matches_proposition_two(n_sources: int, rho: float) -> None:
    covariance = equicorrelation_covariance(n_sources, rho)
    expected = n_sources / (1.0 + (n_sources - 1) * rho)
    assert effective_evidence(covariance) == pytest.approx(expected)


def test_covariance_builder_is_symmetric_psd() -> None:
    inconsistent_pairwise_scores = np.asarray(
        [[1.0, 0.99, 0.01], [0.99, 1.0, 0.99], [0.01, 0.99, 1.0]]
    )
    covariance = covariance_from_dependence(inconsistent_pairwise_scores, shrinkage=0.05)
    assert np.allclose(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-10
