from __future__ import annotations

"""Euclidean projections onto reuse-capped exposure sets.

The capped-simplex water-filling routine computes Euclidean projections
onto reuse-capped exposure sets. The default minimum aggregate exposure
is zero, while an optional lower multiplier supports positive minimum
exposure requirements.
"""

import numpy as np


def allocation_is_feasible(
    allocation: np.ndarray,
    capital: float,
    kappa: float,
    minimum_multiplier: float = 0.0,
    tolerance: float = 1e-8,
) -> bool:
    """Check ``0 <= x_j <= capital`` and the aggregate exposure bounds."""
    x = np.asarray(allocation, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or capital < 0:
        return False
    if capital == 0:
        return bool(np.all(np.abs(x) <= tolerance))
    lower = minimum_multiplier * capital
    upper = min(float(kappa), float(x.size)) * capital
    return bool(
        np.all(x >= -tolerance)
        and np.all(x <= capital + tolerance)
        and lower - tolerance <= x.sum() <= upper + tolerance
    )


def project_allocation(
    allocation: np.ndarray,
    total_stake: float,
    kappa: float,
    minimum_multiplier: float = 0.0,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Euclidean projection onto a box intersected with aggregate bounds.

    The feasible set is

    ``{x: 0 <= x_j <= S, lower*S <= sum(x) <= kappa*S}``.

    The projection supports fractional reuse caps and defaults to
    ``lower=0``, corresponding to the feasible set analyzed in the paper.
    """
    values = np.asarray(allocation, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("allocation must be a finite nonempty vector")
    if not np.isfinite(total_stake) or total_stake < 0:
        raise ValueError("total_stake must be finite and nonnegative")
    if not np.isfinite(kappa) or not np.isfinite(minimum_multiplier):
        raise ValueError("reuse multipliers must be finite")
    if minimum_multiplier < 0 or kappa < minimum_multiplier:
        raise ValueError("require 0 <= minimum_multiplier <= kappa")
    if kappa > values.size + 1e-12:
        raise ValueError("kappa cannot exceed the number of coordinates")
    if total_stake == 0:
        return np.zeros_like(values)

    capped = np.clip(values, 0.0, total_stake)
    lower = minimum_multiplier * total_stake
    upper = min(float(kappa), float(values.size)) * total_stake
    total = float(capped.sum())
    if lower - tolerance <= total <= upper + tolerance:
        return capped

    target = lower if total < lower else upper
    # Water filling: x(lambda) = clip(values - lambda, 0, total_stake).
    lo = float(np.min(values - total_stake) - abs(target) - total_stake)
    hi = float(np.max(values) + abs(target) + total_stake)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        trial = np.clip(values - mid, 0.0, total_stake)
        if trial.sum() > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance:
            break
    result = np.clip(values - 0.5 * (lo + hi), 0.0, total_stake)

    # Remove tiny bisection drift without changing the projection materially.
    error = target - float(result.sum())
    if abs(error) > 10 * tolerance:
        free = np.flatnonzero((result > tolerance) & (result < total_stake - tolerance))
        if free.size:
            result[free] += error / free.size
            result = np.clip(result, 0.0, total_stake)
    return result


def project_reuse_cap(y: np.ndarray, W: float, kappa: float, tol: float = 1e-12) -> np.ndarray:
    """Compatibility wrapper for the paper model's zero-lower-bound set."""
    return project_allocation(y, W, kappa, minimum_multiplier=0.0, tolerance=tol)


def project_profile(
    y: np.ndarray,
    W: np.ndarray,
    kappa: np.ndarray,
    minimum_multiplier: np.ndarray | float = 0.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    W = np.asarray(W, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    lower = np.asarray(minimum_multiplier, dtype=float)
    if lower.ndim == 0:
        lower = np.full(y.shape[0], float(lower))
    return np.vstack(
        [
            project_allocation(y[i], W[i], kappa[i], minimum_multiplier=lower[i])
            for i in range(y.shape[0])
        ]
    )


def initial_k_allocation(
    stake: float,
    m: int,
    kappa: float,
    minimum_multiplier: float = 0.0,
    fill_fraction: float = 1.0,
) -> np.ndarray:
    """Uniform starting point projected onto the feasible exposure set."""
    if stake == 0:
        return np.zeros(m, dtype=float)
    raw_total = fill_fraction * kappa * stake
    return project_allocation(
        np.full(m, raw_total / m),
        stake,
        kappa,
        minimum_multiplier=minimum_multiplier,
    )


__all__ = [
    "allocation_is_feasible",
    "project_allocation",
    "project_reuse_cap",
    "project_profile",
    "initial_k_allocation",
]
