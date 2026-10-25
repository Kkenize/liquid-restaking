from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .environment import Environment
from .projection import project_reuse_cap


@dataclass
class RiskSolveResult:
    x: np.ndarray
    objective: float
    converged: bool
    iterations: int
    runtime: float
    message: str


def _reward_and_grad(env: Environment, x: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    reward = float(np.sum(env.R * x / (B + x)))
    grad = env.R * B / (B + x) ** 2
    return reward, grad


def solve_single_mean_variance(env: Environment, V: np.ndarray | None = None, eta: float | None = None) -> RiskSolveResult:
    if env.n != 1:
        raise ValueError("single-agent solver requires n=1")
    C, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else V
    eta_use = env.eta[0] if eta is None else eta
    B = env.b.copy()
    start = np.full(env.m, min(env.W[0], env.kappa[0] * env.W[0] / env.m))

    def fun(x: np.ndarray) -> float:
        reward, _ = _reward_and_grad(env, x, B)
        return -(reward - d @ x - 0.5 * eta_use * x @ V_use @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        _, rg = _reward_and_grad(env, x, B)
        return -(rg - d - eta_use * (V_use @ x))

    con = {"type": "ineq", "fun": lambda x: env.kappa[0] * env.W[0] - x.sum(), "jac": lambda x: -np.ones_like(x)}
    t0 = perf_counter()
    result = minimize(fun, start, jac=jac, method="SLSQP", bounds=[(0, env.W[0])] * env.m,
                      constraints=[con], options={"ftol": 1e-11, "maxiter": 3000, "disp": False})
    x = project_reuse_cap(result.x, env.W[0], env.kappa[0])
    return RiskSolveResult(x, -float(result.fun), bool(result.success), int(result.nit), perf_counter() - t0, str(result.message))


def solve_single_entropic(env: Environment, F: np.ndarray, theta: float) -> RiskSolveResult:
    if env.n != 1 or theta <= 0:
        raise ValueError("requires n=1 and theta>0")
    monetary = F * env.alpha[None, :]
    exact_mean_vector = env.alpha * env.p
    sample_mean_vector = monetary.mean(axis=0)
    centered_monetary = monetary - sample_mean_vector[None, :]
    B = env.b.copy()
    start = np.full(env.m, min(env.W[0], env.kappa[0] * env.W[0] / env.m))

    def risk_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        # Moment-matched Monte Carlo: preserve the exact first moment and estimate
        # only the higher cumulants from common random scenarios.
        centered_losses = centered_monetary @ x
        z = theta * centered_losses
        lse = logsumexp(z)
        log_mgf = lse - np.log(len(z))
        weights = np.exp(z - lse)
        risk = exact_mean_vector @ x + log_mgf / theta
        grad = exact_mean_vector + weights @ centered_monetary
        return float(risk), grad

    def fun(x: np.ndarray) -> float:
        reward, _ = _reward_and_grad(env, x, B)
        risk, _ = risk_grad(x)
        return -(reward - risk)

    def jac(x: np.ndarray) -> np.ndarray:
        _, rg = _reward_and_grad(env, x, B)
        _, grad_risk = risk_grad(x)
        return -(rg - grad_risk)

    con = {"type": "ineq", "fun": lambda x: env.kappa[0] * env.W[0] - x.sum(), "jac": lambda x: -np.ones_like(x)}
    t0 = perf_counter()
    result = minimize(fun, start, jac=jac, method="SLSQP", bounds=[(0, env.W[0])] * env.m,
                      constraints=[con], options={"ftol": 1e-10, "maxiter": 3000, "disp": False})
    x = project_reuse_cap(result.x, env.W[0], env.kappa[0])
    return RiskSolveResult(x, -float(result.fun), bool(result.success), int(result.nit), perf_counter() - t0, str(result.message))


def solve_single_cvar_premium(env: Environment, F: np.ndarray, zeta: float, level: float = 0.95) -> RiskSolveResult:
    if env.n != 1 or zeta < 0:
        raise ValueError("requires n=1 and zeta>=0")
    monetary = F * env.alpha[None, :]
    mean_event = monetary.mean(axis=0)
    B = env.b.copy()
    x_start = np.full(env.m, min(env.W[0], env.kappa[0] * env.W[0] / env.m))
    t_start = float(np.quantile(monetary @ x_start, level))
    z0 = np.concatenate([x_start, [t_start]])
    inv_tail = 1.0 / (1.0 - level)

    def risk_and_subgrad(z: np.ndarray) -> tuple[float, np.ndarray]:
        x, t = z[:-1], z[-1]
        losses = monetary @ x
        active = losses > t
        excess = np.maximum(losses - t, 0.0)
        cvar = t + inv_tail * float(np.mean(excess))
        mean_loss = float(mean_event @ x)
        risk = mean_loss + zeta * (cvar - mean_loss)
        tail_grad = inv_tail * np.mean(monetary * active[:, None], axis=0)
        grad_x = (1 - zeta) * mean_event + zeta * tail_grad
        grad_t = zeta * (1.0 - inv_tail * float(np.mean(active)))
        return risk, np.concatenate([grad_x, [grad_t]])

    def fun(z: np.ndarray) -> float:
        x = z[:-1]
        reward, _ = _reward_and_grad(env, x, B)
        risk, _ = risk_and_subgrad(z)
        return -(reward - risk)

    def jac(z: np.ndarray) -> np.ndarray:
        x = z[:-1]
        _, rg = _reward_and_grad(env, x, B)
        _, grad_risk = risk_and_subgrad(z)
        return -np.concatenate([rg, [0.0]]) + grad_risk

    con = {"type": "ineq", "fun": lambda z: env.kappa[0] * env.W[0] - z[:-1].sum()}
    bounds = [(0, env.W[0])] * env.m + [(0, env.kappa[0] * env.W[0])]
    t0 = perf_counter()
    result = minimize(fun, z0, jac=jac, method="SLSQP", bounds=bounds, constraints=[con],
                      options={"ftol": 1e-9, "maxiter": 5000, "disp": False})
    x = project_reuse_cap(result.x[:-1], env.W[0], env.kappa[0])
    return RiskSolveResult(x, -float(result.fun), bool(result.success), int(result.nit), perf_counter() - t0, str(result.message))
