from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize

from restaking_risk.generators import make_environment, make_homogeneous_equicorrelated
from restaking_risk.metrics import exploitability, vi_residual
from restaking_risk.model import gross_claim_samples, marginal_payoff_rates, payoff
from restaking_risk.projection import allocation_is_feasible, project_allocation, project_reuse_cap
from restaking_risk.solvers import cyclic_best_response, projected_pseudogradient


def test_projection_matches_independent_qp() -> None:
    rng = np.random.default_rng(4)
    for _ in range(20):
        y = rng.normal(size=12)
        W, kappa = 1.0, 3.5
        x = project_reuse_cap(y, W=W, kappa=kappa)
        assert allocation_is_feasible(x, W, kappa)
        assert np.linalg.norm(project_reuse_cap(x, W, kappa) - x) < 1e-10

        result = minimize(
            lambda z: 0.5 * np.sum((z - y) ** 2),
            np.clip(y, 0, W),
            method="SLSQP",
            bounds=[(0.0, W)] * len(y),
            constraints=[{"type": "ineq", "fun": lambda z: kappa * W - z.sum()}],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        assert result.success
        assert np.linalg.norm(x - result.x) < 2e-6


def test_minimum_exposure_bound_remains_available() -> None:
    y = np.array([-1.0, -0.5, 0.1, 0.2])
    x = project_allocation(y, total_stake=2.0, kappa=2.0, minimum_multiplier=1.0)
    assert allocation_is_feasible(x, 2.0, 2.0, minimum_multiplier=1.0)
    assert np.isclose(x.sum(), 2.0, atol=1e-9)


def test_gradient_matches_finite_differences() -> None:
    env = make_environment(n=3, m=5, kappa=2.0, regime="clustered", seed=12)
    _, _, V = env.risk_matrices()
    rng = np.random.default_rng(99)
    profile = np.vstack([project_reuse_cap(rng.uniform(0.1, 0.5, env.m), env.W[i], env.kappa[i]) for i in range(env.n)])
    i = 1
    analytic = marginal_payoff_rates(i, profile[i], profile, env, V=V)
    numeric = np.zeros(env.m)
    eps = 1e-6
    for j in range(env.m):
        plus = profile[i].copy(); plus[j] += eps
        minus = profile[i].copy(); minus[j] -= eps
        numeric[j] = (payoff(i, plus, profile, env, V=V, validate=False) - payoff(i, minus, profile, env, V=V, validate=False)) / (2 * eps)
    assert np.max(np.abs(analytic - numeric)) < 2e-6


def test_factor_moments_match_simulation() -> None:
    env = make_environment(n=1, m=8, kappa=1.0, regime="clustered", seed=8)
    C = env.binary_covariance()
    F = env.sample_slashes(250_000, seed=81)
    p_hat = F.mean(axis=0)
    C_hat = np.cov(F, rowvar=False, ddof=0)
    assert np.max(np.abs(p_hat - env.p)) < 0.002
    assert np.max(np.abs(C_hat - C)) < 0.0015


def test_loss_moments_match_analytic() -> None:
    env = make_environment(n=2, m=7, kappa=2.0, regime="global_clustered", seed=2)
    _, _, V = env.risk_matrices()
    x = np.vstack([
        project_reuse_cap(np.linspace(0.1, 0.9, env.m), env.W[0], env.kappa[0]),
        project_reuse_cap(np.linspace(0.8, 0.2, env.m), env.W[1], env.kappa[1]),
    ])
    F = env.sample_slashes(300_000, seed=23)
    individual, _ = gross_claim_samples(env, x, F)
    for i in range(env.n):
        analytic_mean = (env.alpha * env.p) @ x[i]
        analytic_var = x[i] @ V @ x[i]
        assert abs(individual[:, i].mean() - analytic_mean) < 0.002
        assert abs(individual[:, i].var() - analytic_var) < 0.002


def test_loss_variance_vanishes_with_failure_probability() -> None:
    base = make_environment(n=1, m=5, kappa=2.0, regime="clustered", seed=5)
    x = np.full(base.m, 0.3)
    variances = []
    for p in [0.08, 0.008, 0.0008]:
        env = base.with_updates(p=np.full(base.m, p))
        _, _, V = env.risk_matrices()
        variances.append(float(x @ V @ x))
    assert variances[2] < variances[1] < variances[0]
    assert variances[2] < 0.02 * variances[0]


def test_unique_shared_risk_formulas_and_independence_multiplier() -> None:
    m = 9
    env = make_homogeneous_equicorrelated(
        n=1, m=m, kappa=1, p=0.08, alpha=0.6, latent_loading_sq=0.35, seed=5
    )
    C, _, V = env.risk_matrices()
    rho = C[0, 1] / (env.p[0] * (1 - env.p[0]))
    sigma2 = env.alpha[0] ** 2 * env.p[0] * (1 - env.p[0])
    W = env.W[0]
    x_unique = np.full(m, W / m)
    x_shared = np.full(m, W)
    var_unique = x_unique @ V @ x_unique
    var_shared = x_shared @ V @ x_shared
    formula_unique = sigma2 * W**2 * (rho + (1 - rho) / m)
    formula_shared = sigma2 * W**2 * m * (1 + (m - 1) * rho)
    assert np.isclose(var_unique, formula_unique, rtol=1e-7, atol=1e-10)
    assert np.isclose(var_shared, formula_shared, rtol=1e-7, atol=1e-10)
    diag_var = x_unique @ np.diag(np.diag(V)) @ x_unique
    assert np.isclose(var_unique / diag_var, 1 + (m - 1) * rho, rtol=1e-7)


def test_symmetric_equilibrium_matches_scalar_characterization() -> None:
    n, m = 4, 6
    env = make_homogeneous_equicorrelated(
        n=n, m=m, kappa=2.5, p=0.05, alpha=0.55, latent_loading_sq=0.25,
        R=0.14, b=1.4, W=1.0, eta=1.6, seed=0,
    )
    C, d_vec, V = env.risk_matrices()
    sigma2 = V[0, 0]
    rho = V[0, 1] / sigma2
    d = d_vec[0]
    qbar = env.kappa[0] * env.W[0] / m

    def psi(q: float) -> float:
        return env.R[0] * (env.b[0] + (n - 1) * q) / (env.b[0] + n * q) ** 2 - d - env.eta[0] * sigma2 * (1 + (m - 1) * rho) * q

    if env.R[0] / env.b[0] <= d:
        qstar = 0.0
    elif psi(qbar) >= 0:
        qstar = qbar
    else:
        qstar = brentq(psi, 0, qbar)

    result = projected_pseudogradient(env, V=V, step_size=0.15, adaptive=True, tol=1e-9, maxiter=30_000)
    assert result.converged
    assert np.max(np.abs(result.x - qstar)) < 2e-4
    assert exploitability(env, result.x, V) < 1e-7


def test_cross_solver_agreement() -> None:
    env = make_environment(n=3, m=5, kappa=1.7, regime="clustered", seed=12)
    _, _, V = env.risk_matrices()
    br = cyclic_best_response(env, V=V, tol=1e-10, max_sweeps=500)
    pg = projected_pseudogradient(env, V=V, step_size=0.12, adaptive=True, tol=1e-9, maxiter=20_000)
    assert br.converged and pg.converged
    assert exploitability(env, br.x, V) < 1e-7
    assert exploitability(env, pg.x, V) < 1e-7
    assert vi_residual(env, br.x, V) < 2e-6
    assert vi_residual(env, pg.x, V) < 2e-6
    assert np.linalg.norm(br.x - pg.x) < 5e-4
