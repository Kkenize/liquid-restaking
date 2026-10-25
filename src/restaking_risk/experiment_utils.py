from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .environment import Environment
from .metrics import var_cvar
from .model import gross_claim_samples
from .projection import project_reuse_cap


def ensure_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    dirs = {
        "root": root,
        "processed": root / "data" / "processed",
        "tables": root / "results" / "tables",
        "logs": root / "results" / "logs",
        "figures": root / "figures",
        "configs": root / "configs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_records(records: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_json(payload: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def low_rank_plus_diagonal_covariance(samples: np.ndarray, rank: int) -> np.ndarray:
    S = np.cov(samples, rowvar=False, ddof=0)
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order][:rank], 0.0)
    vecs = vecs[:, order[:rank]]
    low = (vecs * vals) @ vecs.T
    residual = np.maximum(np.diag(S) - np.diag(low), 1e-10)
    C = low + np.diag(residual)
    return 0.5 * (C + C.T)


def optimize_scale_for_weights(
    env: Environment,
    weights: np.ndarray,
    V: np.ndarray,
    eta: float | None = None,
) -> np.ndarray:
    """Choose the utility-maximizing gross scale along a fixed portfolio direction."""
    if env.n != 1:
        raise ValueError("heuristic scale optimizer expects n=1")
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if weights.sum() <= 0:
        return np.zeros(env.m)
    weights /= weights.sum()
    eta_use = env.eta[0] if eta is None else eta
    _, d, _ = env.risk_matrices()
    cap_by_coord = np.min(np.divide(env.W[0], weights, out=np.full_like(weights, np.inf), where=weights > 0))
    gmax = min(env.kappa[0] * env.W[0], cap_by_coord)

    def utility(g: float) -> float:
        x = g * weights
        reward = np.sum(env.R * x / (env.b + x))
        return float(reward - d @ x - 0.5 * eta_use * x @ V @ x)

    result = minimize_scalar(lambda g: -utility(g), bounds=(0.0, gmax), method="bounded", options={"xatol": 1e-10})
    return project_reuse_cap(float(result.x) * weights, env.W[0], env.kappa[0])


def evaluate_single_allocation(
    env: Environment,
    x: np.ndarray,
    F_eval: np.ndarray,
    V_true: np.ndarray,
    eta: float | None = None,
) -> dict[str, float]:
    if env.n != 1:
        raise ValueError("single allocation evaluation expects n=1")
    eta_use = env.eta[0] if eta is None else eta
    _, d, _ = env.risk_matrices()
    reward = float(np.sum(env.R * x / (env.b + x)))
    mean = float(d @ x)
    var = float(x @ V_true @ x)
    losses = (F_eval * env.alpha[None, :]) @ x
    var95, cvar95 = var_cvar(losses, 0.95)
    var99, cvar99 = var_cvar(losses, 0.99)
    gross = float(x.sum())
    sq = float(x @ x)
    eff = gross**2 / sq if sq > 0 else 0.0
    return {
        "reward": reward,
        "expected_loss": mean,
        "variance": var,
        "std": float(np.sqrt(max(var, 0.0))),
        "var95": var95,
        "cvar95": cvar95,
        "var99": var99,
        "cvar99": cvar99,
        "mean_variance_utility": reward - mean - 0.5 * eta_use * var,
        "gross_exposure": gross,
        "effective_breadth": eff,
        "hhi": sq / gross**2 if gross > 0 else 0.0,
    }


def clone_with_players(base: Environment, n: int, total_capital: float, kappa: float, eta: float) -> Environment:
    return Environment(
        theta=np.eye(base.m),
        R=base.R.copy(),
        b=base.b.copy(),
        slashing_rate=base.alpha.copy(),
        mu=base.p.copy(),
        beta=base.beta.copy(),
        s=np.full(n, total_capital / n),
        risk_aversion=np.full(n, eta),
        K=np.full(n, kappa),
        minimum_exposure_multiplier=np.zeros(n),
        position_labels=base.position_labels,
        operator_cluster=base.operator_cluster,
        client_cluster=base.client_cluster,
        cloud_cluster=base.cloud_cluster,
        regime=base.regime,
        seed=base.seed,
    )
