"""Dependence-aware covariance construction and evidence aggregation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _square(matrix: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.size == 0:
        raise ValueError("expected a non-empty square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("matrix contains non-finite values")
    return value


def nearest_psd(matrix: ArrayLike, eigenvalue_floor: float = 0.0) -> NDArray[np.float64]:
    """Project a symmetric matrix onto the positive-semidefinite cone."""
    value = _square(matrix)
    if eigenvalue_floor < 0:
        raise ValueError("eigenvalue_floor must be non-negative")
    symmetric = (value + value.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    # einsum avoids an Accelerate/BLAS warning observed for small rank-deficient
    # matrices on Apple Silicon while performing the same V diag(lam) V^T sum.
    projected = np.einsum("ik,k,jk->ij", eigenvectors, clipped, eigenvectors)
    return (projected + projected.T) / 2.0


def covariance_from_dependence(
    dependence: ArrayLike,
    variances: ArrayLike | None = None,
    shrinkage: float = 0.05,
    eigenvalue_floor: float = 1e-9,
    scorer_variance: float = 0.0,
) -> NDArray[np.float64]:
    """Convert pairwise shared-origin probabilities into a valid covariance.

    Dependence scores are treated as non-negative correlations. Shrinkage is
    applied toward the diagonal before a final PSD projection.

    ``scorer_variance`` is the reviewer's shared-scorer term ``sigma^2``: the
    returned matrix is ``Sigma_prov + sigma^2 J`` with ``J`` the all-ones
    matrix.  ``J`` is PSD, so the sum stays PSD.  Note that this is a rank-one
    update along ``1``; by Sherman-Morrison the GLS weights and the stance
    estimate are invariant to it and only ``w_eff`` changes (see
    ``shared_scorer_effective_evidence``).
    """
    dep = np.clip(_square(dependence), 0.0, 1.0)
    dep = (dep + dep.T) / 2.0
    np.fill_diagonal(dep, 1.0)
    n = dep.shape[0]
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if variances is None:
        variance = np.ones(n, dtype=np.float64)
    else:
        variance = np.asarray(variances, dtype=np.float64)
        if variance.shape != (n,) or np.any(variance <= 0) or not np.all(np.isfinite(variance)):
            raise ValueError("variances must be a finite positive vector")
    if not np.isfinite(scorer_variance) or scorer_variance < 0.0:
        raise ValueError("scorer_variance must be a finite non-negative scalar")
    correlation = (1.0 - shrinkage) * dep + shrinkage * np.eye(n)
    correlation = nearest_psd(correlation, eigenvalue_floor=eigenvalue_floor)
    scale = np.sqrt(variance)
    covariance = correlation * np.outer(scale, scale)
    if scorer_variance > 0.0:
        covariance = covariance + scorer_variance * np.ones((n, n), dtype=np.float64)
    return covariance


def shared_scorer_effective_evidence(base_effective: float, scorer_variance: float) -> float:
    """Sherman-Morrison prediction for ``w_eff`` under ``Sigma + sigma^2 J``.

    With ``w = 1^T Sigma^+ 1`` the updated value is ``w / (1 + sigma^2 w)``,
    which saturates at ``1 / sigma^2`` however large ``w`` becomes.
    """

    if base_effective < 0.0 or scorer_variance < 0.0:
        raise ValueError("effective evidence and scorer_variance must be non-negative")
    return float(base_effective / (1.0 + scorer_variance * base_effective))


def equicorrelation_covariance(
    n_sources: int, rho: float, variance: float = 1.0
) -> NDArray[np.float64]:
    if n_sources < 1:
        raise ValueError("n_sources must be positive")
    lower = -1.0 / (n_sources - 1) if n_sources > 1 else 0.0
    if not lower <= rho <= 1.0:
        raise ValueError(f"rho must be in [{lower}, 1]")
    if variance <= 0:
        raise ValueError("variance must be positive")
    return variance * ((1.0 - rho) * np.eye(n_sources) + rho * np.ones((n_sources, n_sources)))


RIDGE_MODES = ("covariance", "precision")


def regularized_precision(
    covariance: ArrayLike,
    rcond: float = 1e-10,
    ridge: float = 0.0,
    ridge_mode: str = "covariance",
    project: bool = True,
) -> NDArray[np.float64]:
    """Return the (optionally ridge-regularized) pseudoinverse of ``Sigma``.

    ``ridge_mode="covariance"`` returns ``(Sigma + lambda I)^+`` (Tikhonov on
    the covariance before inversion); ``ridge_mode="precision"`` returns
    ``Sigma^+ + lambda I``.  Both shrink the GLS weights toward uniform, and --
    unlike the shared-scorer term -- both move the stance estimate.

    ``ridge=0`` reproduces the unregularized pseudoinverse *bit for bit*: the
    ridge branches are skipped entirely, and with ``project=False`` no second
    ``nearest_psd`` eigendecomposition is applied.  Callers that have already
    projected must pass ``project=False``, or the frozen numbers move at the
    1e-14 level.
    """

    if ridge_mode not in RIDGE_MODES:
        raise ValueError(f"ridge_mode must be one of {RIDGE_MODES}")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative scalar")
    matrix = nearest_psd(covariance) if project else _square(covariance)
    n = matrix.shape[0]
    if ridge > 0.0 and ridge_mode == "covariance":
        matrix = matrix + ridge * np.eye(n, dtype=np.float64)
    precision = _psd_pseudoinverse(matrix, rcond=rcond)
    if ridge > 0.0 and ridge_mode == "precision":
        precision = precision + ridge * np.eye(n, dtype=np.float64)
    return precision


def effective_evidence(
    covariance: ArrayLike,
    rcond: float = 1e-10,
    ridge: float = 0.0,
    ridge_mode: str = "covariance",
) -> float:
    """Return 1^T Sigma^+ 1, using a pseudoinverse for exact copies."""
    covariance_array = nearest_psd(covariance)
    precision = regularized_precision(
        covariance_array,
        rcond=rcond,
        ridge=ridge,
        ridge_mode=ridge_mode,
        project=False,
    )
    ones = np.ones(covariance_array.shape[0], dtype=np.float64)
    value = float(np.einsum("i,ij,j->", ones, precision, ones))
    if value < -1e-8:
        raise ArithmeticError("effective evidence became negative")
    return max(0.0, value)


def gls_weights(
    covariance: ArrayLike,
    rcond: float = 1e-10,
    ridge: float = 0.0,
    ridge_mode: str = "covariance",
) -> NDArray[np.float64]:
    covariance_array = nearest_psd(covariance)
    precision = regularized_precision(
        covariance_array,
        rcond=rcond,
        ridge=ridge,
        ridge_mode=ridge_mode,
        project=False,
    )
    ones = np.ones(covariance_array.shape[0], dtype=np.float64)
    numerator = np.einsum("ij,j->i", precision, ones)
    denominator = float(np.einsum("i,i->", ones, numerator))
    if denominator <= 0:
        raise ArithmeticError("covariance has no estimable shared mean")
    return numerator / denominator


def aggregate_stances(
    stances: ArrayLike,
    covariance: ArrayLike,
    *,
    ridge: float = 0.0,
    ridge_mode: str = "covariance",
) -> tuple[float, float]:
    """Return the GLS stance mean and its effective-evidence precision."""
    values = np.asarray(stances, dtype=np.float64)
    covariance_array = _square(covariance)
    if values.shape != (covariance_array.shape[0],):
        raise ValueError("stances and covariance dimensions disagree")
    if np.any(np.abs(values) > 1.0) or not np.all(np.isfinite(values)):
        raise ValueError("stances must be finite values in [-1, 1]")
    weights = gls_weights(covariance_array, ridge=ridge, ridge_mode=ridge_mode)
    return (
        float(np.einsum("i,i->", weights, values)),
        effective_evidence(covariance_array, ridge=ridge, ridge_mode=ridge_mode),
    )


def _psd_pseudoinverse(matrix: NDArray[np.float64], rcond: float) -> NDArray[np.float64]:
    """Hermitian pseudoinverse reconstructed without backend matrix products."""
    if rcond < 0:
        raise ValueError("rcond must be non-negative")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    cutoff = rcond * max(1.0, float(np.max(eigenvalues)))
    inverse = np.zeros_like(eigenvalues)
    mask = eigenvalues > cutoff
    inverse[mask] = 1.0 / eigenvalues[mask]
    return np.einsum("ik,k,jk->ij", eigenvectors, inverse, eigenvectors)
