from __future__ import annotations

"""Equilibrium and planner solvers.

The simultaneous projected pseudo-gradient routine computes each player's
gradient at the same pre-update profile and applies all projected moves
simultaneously. Convergence is monitored through a projected-gradient
mapping residual.

Fixed-step, adaptive, and square-root-decay schedules are supported.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize

from .environment import Environment
from .model import best_response_objective_and_grad, pseudo_gradient, social_welfare
from .projection import initial_k_allocation, project_profile, project_reuse_cap


@dataclass
class SolverResult:
    x: np.ndarray
    converged: bool
    iterations: int
    runtime: float
    message: str
    history: dict[str, list[float]]


def feasible_start(env: Environment, mode: str = "equal", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "zero":
        x = np.zeros((env.n, env.m))
    elif mode == "equal":
        x = np.vstack(
            [
                initial_k_allocation(
                    float(env.W[i]),
                    env.m,
                    float(env.kappa[i]),
                    minimum_multiplier=float(env.lower_multiplier[i]),
                    fill_fraction=1.0,
                )
                for i in range(env.n)
            ]
        )
    elif mode == "random":
        x = np.vstack(
            [
                project_reuse_cap(rng.uniform(0.0, env.W[i], env.m), env.W[i], env.kappa[i])
                for i in range(env.n)
            ]
        )
    else:
        raise ValueError(f"unknown start mode: {mode}")
    return project_profile(x, env.W, env.kappa, env.lower_multiplier)


def solve_best_response(
    env: Environment,
    i: int,
    x: np.ndarray,
    V: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    others = np.sum(x, axis=0) - x[i]
    start = x[i].copy() if x0 is None else np.asarray(x0, dtype=float)
    start = project_profile(start[None, :], env.W[i : i + 1], env.kappa[i : i + 1], env.lower_multiplier[i : i + 1])[0]

    def fun(y: np.ndarray) -> float:
        value, _ = best_response_objective_and_grad(env, i, y, others, V)
        return -value

    def jac(y: np.ndarray) -> np.ndarray:
        _, grad = best_response_objective_and_grad(env, i, y, others, V)
        return -grad

    constraints: list[dict] = [
        {
            "type": "ineq",
            "fun": lambda y: env.kappa[i] * env.W[i] - np.sum(y),
            "jac": lambda y: -np.ones_like(y),
        }
    ]
    if env.lower_multiplier[i] > 0:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y: np.sum(y) - env.lower_multiplier[i] * env.W[i],
                "jac": lambda y: np.ones_like(y),
            }
        )

    result = minimize(
        fun,
        start,
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, float(env.W[i]))] * env.m,
        constraints=constraints,
        options={"ftol": tol, "maxiter": maxiter, "disp": False},
    )
    y = project_profile(
        result.x[None, :],
        env.W[i : i + 1],
        env.kappa[i : i + 1],
        env.lower_multiplier[i : i + 1],
    )[0]
    return y, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": -float(result.fun),
    }


def cyclic_best_response(
    env: Environment,
    V: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    update_order: str | list[int] = "cyclic",
    damping: float = 1.0,
    tol: float = 1e-8,
    max_sweeps: int = 1000,
    seed: int = 0,
    callback: Callable[[np.ndarray, int], float] | None = None,
) -> SolverResult:
    start_time = perf_counter()
    rng = np.random.default_rng(seed)
    x = feasible_start(env, "equal", seed) if x0 is None else project_profile(x0, env.W, env.kappa, env.lower_multiplier)
    hist = {"step": [], "diagnostic": []}
    converged = False
    message = "maximum sweeps reached"
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        old = x.copy()
        if isinstance(update_order, list):
            order = list(update_order)
        elif update_order == "cyclic":
            order = list(range(env.n))
        elif update_order == "reverse":
            order = list(range(env.n - 1, -1, -1))
        elif update_order in {"random", "asynchronous"}:
            order = list(rng.permutation(env.n))
        else:
            raise ValueError(f"unknown update order {update_order}")
        for i in order:
            br, info = solve_best_response(env, i, x, V=V, x0=x[i])
            if not info["success"]:
                message = f"best response failed for player {i}: {info['message']}"
            x[i] = (1.0 - damping) * x[i] + damping * br
            x[i] = project_profile(
                x[i][None, :],
                env.W[i : i + 1],
                env.kappa[i : i + 1],
                env.lower_multiplier[i : i + 1],
            )[0]
        step = float(np.linalg.norm(x - old))
        hist["step"].append(step)
        hist["diagnostic"].append(float(callback(x, sweep)) if callback else np.nan)
        if step <= tol * max(1.0, float(np.linalg.norm(x))):
            converged = True
            message = "successive-sweep tolerance reached"
            break
    return SolverResult(x, converged, sweep, perf_counter() - start_time, message, hist)


def projected_pseudogradient(
    env: Environment,
    V: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    step_size: float = 0.05,
    tol: float = 1e-8,
    maxiter: int = 20_000,
    adaptive: bool = False,
    schedule: Literal["constant", "sqrt_decay"] = "constant",
    callback: Callable[[np.ndarray, int], float] | None = None,
    store_every: int = 1,
) -> SolverResult:
    """Simultaneous projected pseudo-gradient ascent.

    ``schedule='sqrt_decay'`` uses the ``step/sqrt(t)`` rule.
    ``schedule='constant'`` is the update covered by the paper's convergence
    theorem. Setting ``adaptive=True`` adds the practical residual line search
    used in the computational study.
    """
    if step_size <= 0 or tol < 0 or maxiter <= 0 or store_every <= 0:
        raise ValueError("invalid projected-gradient parameters")
    start_time = perf_counter()
    x = feasible_start(env, "equal") if x0 is None else project_profile(x0, env.W, env.kappa, env.lower_multiplier)
    hist: dict[str, list[float]] = {
        "step": [],
        "diagnostic": [],
        "step_size": [],
        "projected_residual": [],
    }
    converged = False
    message = "maximum iterations reached"
    base_tau = float(step_size)
    tau_adaptive = base_tau
    iteration = 0

    for iteration in range(1, maxiter + 1):
        tau = base_tau / np.sqrt(iteration) if schedule == "sqrt_decay" else tau_adaptive
        g = pseudo_gradient(env, x, V)
        candidate = project_profile(x + tau * g, env.W, env.kappa, env.lower_multiplier)

        if adaptive:
            residual_old = float(np.linalg.norm(x - candidate))
            trials = 0
            while trials < 20 and tau > 1e-8:
                g_new = pseudo_gradient(env, candidate, V)
                projected_new = project_profile(candidate + tau * g_new, env.W, env.kappa, env.lower_multiplier)
                residual_new = float(np.linalg.norm(candidate - projected_new))
                if residual_new <= 1.05 * max(residual_old, 1e-15):
                    break
                tau *= 0.5
                candidate = project_profile(x + tau * g, env.W, env.kappa, env.lower_multiplier)
                trials += 1
            tau_adaptive = tau
            if trials == 0 and iteration % 50 == 0:
                tau_adaptive = min(base_tau * 4.0, tau_adaptive * 1.15)

        delta = candidate - x
        step_norm = float(np.linalg.norm(delta))
        projected_residual = float(np.max(np.abs(delta)) / max(tau, 1e-15))
        x = candidate

        if iteration % store_every == 0 or iteration == 1:
            hist["step"].append(step_norm)
            hist["step_size"].append(float(tau))
            hist["projected_residual"].append(projected_residual)
            hist["diagnostic"].append(float(callback(x, iteration)) if callback else np.nan)

        if projected_residual <= tol:
            converged = True
            message = "projected-gradient mapping residual reached tolerance"
            break

    return SolverResult(x, converged, iteration, perf_counter() - start_time, message, hist)


# Explicit alias that highlights the provenance of the simultaneous algorithm.
find_gradient_equilibrium = projected_pseudogradient


def solve_planner(
    env: Environment,
    V: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 3000,
) -> SolverResult:
    start_time = perf_counter()
    start = feasible_start(env, "equal") if x0 is None else project_profile(x0, env.W, env.kappa, env.lower_multiplier)
    _, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)

    def unpack(z: np.ndarray) -> np.ndarray:
        return z.reshape(env.n, env.m)

    def fun(z: np.ndarray) -> float:
        return -social_welfare(env, unpack(z), V_use)

    def jac(z: np.ndarray) -> np.ndarray:
        x = unpack(z)
        X = x.sum(axis=0)
        reward_grad = env.R * env.b / (env.b + X) ** 2
        grad = np.empty_like(x)
        for i in range(env.n):
            grad[i] = reward_grad - d - env.eta[i] * (V_use @ x[i])
        return -grad.ravel()

    constraints: list[dict] = []
    for i in range(env.n):
        sl = slice(i * env.m, (i + 1) * env.m)
        upper_jac = np.zeros(env.n * env.m)
        upper_jac[sl] = -1.0
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda z, sl=sl, i=i: env.kappa[i] * env.W[i] - np.sum(z[sl]),
                "jac": lambda z, uj=upper_jac: uj,
            }
        )
        if env.lower_multiplier[i] > 0:
            lower_jac = -upper_jac
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, sl=sl, i=i: np.sum(z[sl]) - env.lower_multiplier[i] * env.W[i],
                    "jac": lambda z, lj=lower_jac: lj,
                }
            )

    result = minimize(
        fun,
        start.ravel(),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, float(env.W[i])) for i in range(env.n) for _ in range(env.m)],
        constraints=constraints,
        options={"ftol": tol, "maxiter": maxiter, "disp": False},
    )
    x = project_profile(unpack(result.x), env.W, env.kappa, env.lower_multiplier)
    return SolverResult(
        x=x,
        converged=bool(result.success),
        iterations=int(result.nit),
        runtime=perf_counter() - start_time,
        message=str(result.message),
        history={"step": [], "diagnostic": []},
    )


def natural_residual_solver(
    env: Environment,
    V: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    tau: float = 0.1,
    tol: float = 1e-12,
    maxiter: int = 5000,
) -> SolverResult:
    """Small-instance cross-check: minimize the squared natural VI residual."""
    start_time = perf_counter()
    start = feasible_start(env, "equal") if x0 is None else project_profile(x0, env.W, env.kappa, env.lower_multiplier)

    def unpack(z: np.ndarray) -> np.ndarray:
        return z.reshape(env.n, env.m)

    def residual(z: np.ndarray) -> np.ndarray:
        x = unpack(z)
        g = pseudo_gradient(env, x, V)
        projected = project_profile(x + tau * g, env.W, env.kappa, env.lower_multiplier)
        return (x - projected).ravel()

    def fun(z: np.ndarray) -> float:
        r = residual(z)
        return 0.5 * float(r @ r)

    constraints: list[dict] = []
    for i in range(env.n):
        sl = slice(i * env.m, (i + 1) * env.m)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda z, sl=sl, i=i: env.kappa[i] * env.W[i] - np.sum(z[sl]),
            }
        )
        if env.lower_multiplier[i] > 0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, sl=sl, i=i: np.sum(z[sl]) - env.lower_multiplier[i] * env.W[i],
                }
            )
    result = minimize(
        fun,
        start.ravel(),
        method="SLSQP",
        bounds=[(0.0, float(env.W[i])) for i in range(env.n) for _ in range(env.m)],
        constraints=constraints,
        options={"ftol": tol, "maxiter": maxiter, "disp": False},
    )
    x = project_profile(unpack(result.x), env.W, env.kappa, env.lower_multiplier)
    return SolverResult(
        x=x,
        converged=bool(result.success),
        iterations=int(result.nit),
        runtime=perf_counter() - start_time,
        message=str(result.message),
        history={"step": [], "diagnostic": []},
    )


__all__ = [
    "SolverResult",
    "feasible_start",
    "solve_best_response",
    "cyclic_best_response",
    "projected_pseudogradient",
    "find_gradient_equilibrium",
    "solve_planner",
    "natural_residual_solver",
]
