from __future__ import annotations

import numpy as np

from .environment import Environment
from .model import gross_claim_samples, individual_rewards, pseudo_gradient, social_welfare, total_reward, utility_vector
from .projection import project_profile
from .solvers import solve_best_response



def var_cvar(losses: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    losses = np.asarray(losses, dtype=float)

    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a nonempty 1D array")
    if not np.all(np.isfinite(losses)):
        raise ValueError("losses must contain only finite values")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")

    sorted_losses = np.sort(losses)
    n = sorted_losses.size
    tail_mass = (1.0 - level) * n

    full_count = int(np.floor(tail_mass))
    fractional = tail_mass - full_count

    tail_sum = (
        float(np.sum(sorted_losses[-full_count:]))
        if full_count > 0 else 0.0
    )

    if fractional > 1e-12:
        tail_sum += fractional * sorted_losses[-full_count - 1]

    cvar = tail_sum / tail_mass
    q = float(np.quantile(losses, level, method="higher"))

    return q, float(cvar)


def portfolio_statistics(x: np.ndarray, W: np.ndarray) -> dict[str, np.ndarray]:
    G = x.sum(axis=1)
    sq = np.sum(x**2, axis=1)
    eff = np.divide(G**2, sq, out=np.zeros_like(G), where=sq > 0)
    H = np.divide(sq, G**2, out=np.zeros_like(G), where=G > 0)
    breadth = np.sum(x > 1e-8, axis=1)
    return {"gross": G, "reuse_degree": G / W, "effective_breadth": eff, "hhi": H, "breadth": breadth}


def vi_residual(env: Environment, x: np.ndarray, V: np.ndarray | None = None, tau: float = 0.1) -> float:
    g = pseudo_gradient(env, x, V)
    return float(np.linalg.norm(x - project_profile(x + tau * g, env.W, env.kappa)))


def exploitability(env: Environment, x: np.ndarray, V: np.ndarray | None = None) -> float:
    current = utility_vector(env, x, V)
    gaps = []
    for i in range(env.n):
        y, info = solve_best_response(env, i, x, V=V, x0=x[i], tol=1e-11, maxiter=2000)
        trial = x.copy()
        trial[i] = y
        best = utility_vector(env, trial, V)[i]
        gaps.append(max(0.0, float(best - current[i])))
    return float(max(gaps, default=0.0))


def summarize_profile(
    env: Environment,
    x: np.ndarray,
    F: np.ndarray | None = None,
    V: np.ndarray | None = None,
    cvar_level: float = 0.99,
) -> dict[str, float]:
    C, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else V
    X = x.sum(axis=0)
    stats = portfolio_statistics(x, env.W)
    utilities = utility_vector(env, x, V_use)
    row: dict[str, float] = {
        "total_reward": total_reward(env, x),
        "mean_private_utility": float(np.mean(utilities)),
        "sum_private_utility": float(np.sum(utilities)),
        "social_welfare": social_welfare(env, x, V_use),
        "aggregate_expected_loss": float(d @ X),
        "aggregate_variance": float(X @ V_true @ X),
        "aggregate_std": float(np.sqrt(max(0.0, X @ V_true @ X))),
        "mean_gross_exposure": float(np.mean(stats["gross"])),
        "total_gross_exposure": float(np.sum(stats["gross"])),
        "mean_reuse_degree": float(np.mean(stats["reuse_degree"])),
        "mean_effective_breadth": float(np.mean(stats["effective_breadth"])),
        "mean_allocator_hhi": float(np.mean(stats["hhi"])),
        "position_hhi": float(np.sum((X / X.sum()) ** 2)) if X.sum() > 0 else 0.0,
        "vi_residual": vi_residual(env, x, V_use),
    }
    if F is not None:
        indiv, agg = gross_claim_samples(env, x, F)
        var95, cvar95 = var_cvar(agg, 0.95)
        var99, cvar99 = var_cvar(agg, 0.99)
        row.update({
            "aggregate_mc_mean": float(np.mean(agg)),
            "aggregate_mc_variance": float(np.var(agg)),
            "aggregate_var95": var95,
            "aggregate_cvar95": cvar95,
            "aggregate_var99": var99,
            "aggregate_cvar99": cvar99,
            "mean_individual_cvar99": float(np.mean([var_cvar(indiv[:, i], 0.99)[1] for i in range(env.n)])),
        })
    return row


def covariance_decomposition(x: np.ndarray, V: np.ndarray, clusters: tuple[int, ...]) -> tuple[float, float, float]:
    clusters_arr = np.asarray(clusters)
    within = 0.0
    across = 0.0
    total = float(x @ V @ x)
    for j in range(len(x)):
        for k in range(j + 1, len(x)):
            term = 2.0 * x[j] * x[k] * V[j, k]
            if clusters_arr[j] == clusters_arr[k]:
                within += term
            else:
                across += term
    diagonal = float(np.sum((x**2) * np.diag(V)))
    return diagonal + within, across, total


def allocation_turnover(x: np.ndarray, y: np.ndarray) -> float:
    denom = max(float(np.sum(x)), float(np.sum(y)), 1e-15)
    return 0.5 * float(np.sum(np.abs(x - y))) / denom
