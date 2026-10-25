from __future__ import annotations

"""Controlled synthetic environments for experiments A--F."""

import numpy as np

from .environment import Environment


def _factor_template(
    m: int,
    global_share: float,
    operator_share: float,
    client_share: float = 0.0,
    cloud_share: float = 0.0,
    n_operator: int = 4,
    n_client: int = 2,
    n_cloud: int = 4,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if global_share + operator_share + client_share + cloud_share > 0.95:
        raise ValueError("systematic variance shares must sum to at most 0.95")
    op = tuple(j % n_operator for j in range(m))
    client = tuple((j // n_operator) % n_client for j in range(m))
    cloud = tuple((2 * j + j // max(1, n_operator)) % n_cloud for j in range(m))
    n_factors = 1 + n_operator + n_client + n_cloud
    beta = np.zeros((m, n_factors))
    for j in range(m):
        beta[j, 0] = np.sqrt(global_share)
        beta[j, 1 + op[j]] = np.sqrt(operator_share)
        beta[j, 1 + n_operator + client[j]] = np.sqrt(client_share)
        beta[j, 1 + n_operator + n_client + cloud[j]] = np.sqrt(cloud_share)
    return beta, op, client, cloud


def make_environment(
    *,
    n: int = 4,
    m: int = 12,
    kappa: float = 1.0,
    regime: str = "clustered",
    seed: int = 0,
    heterogeneous: bool = True,
    W_scale: float = 1.0,
    reward_scale: float = 1.0,
    b_scale: float = 1.0,
    eta_scale: float = 1.0,
    global_share: float | None = None,
    minimum_exposure_multiplier: float = 0.0,
) -> Environment:
    rng = np.random.default_rng(seed)
    if regime == "near_independent":
        shares = (0.002, 0.008, 0.004, 0.004)
    elif regime == "clustered":
        shares = (0.03, 0.28, 0.10, 0.08)
    elif regime == "global_clustered":
        shares = (0.38, 0.10, 0.04, 0.03)
    elif regime == "topology":
        g = 0.20 if global_share is None else global_share
        remaining = 0.55 - g
        if remaining < 0:
            raise ValueError("global_share cannot exceed 0.55 in topology regime")
        shares = (g, 0.50 * remaining, 0.27 * remaining, 0.23 * remaining)
    elif regime == "ill_conditioned":
        shares = (0.58, 0.18, 0.08, 0.05)
    else:
        raise ValueError(f"unknown regime: {regime}")

    beta, op, client, cloud = _factor_template(
        m,
        global_share=shares[0],
        operator_share=shares[1],
        client_share=shares[2],
        cloud_share=shares[3],
        n_operator=min(4, m),
        n_client=min(3, m),
        n_cloud=min(4, m),
    )

    capital = W_scale * (0.85 + 0.30 * rng.random(n) if heterogeneous else np.ones(n))
    reward_pool = reward_scale * (
        0.105 + 0.055 * rng.random(m) if heterogeneous else np.full(m, 0.13)
    )
    background = b_scale * (
        1.15 + 0.85 * rng.random(m) if heterogeneous else np.full(m, 1.5)
    )
    failure_probability = (
        0.025 + 0.045 * rng.random(m) if heterogeneous else np.full(m, 0.045)
    )
    slash_fraction = (
        0.35 + 0.45 * rng.random(m) if heterogeneous else np.full(m, 0.55)
    )
    risk_aversion = eta_scale * (
        1.1 + 0.8 * rng.random(n) if heterogeneous else np.full(n, 1.5)
    )

    return Environment(
        theta=np.eye(m),
        R=reward_pool,
        b=background,
        slashing_rate=slash_fraction,
        mu=failure_probability,
        beta=beta,
        s=capital,
        risk_aversion=risk_aversion,
        K=np.full(n, float(kappa)),
        minimum_exposure_multiplier=np.full(n, float(minimum_exposure_multiplier)),
        position_labels=tuple(f"position_{j+1:02d}" for j in range(m)),
        operator_cluster=op,
        client_cluster=client,
        cloud_cluster=cloud,
        regime=regime,
        seed=seed,
    )


def make_homogeneous_equicorrelated(
    *,
    n: int,
    m: int,
    kappa: float,
    p: float,
    alpha: float,
    latent_loading_sq: float,
    R: float = 0.13,
    b: float = 1.5,
    W: float = 1.0,
    eta: float = 1.5,
    seed: int = 0,
) -> Environment:
    beta = np.full((m, 1), np.sqrt(latent_loading_sq))
    return Environment(
        theta=np.eye(m),
        R=np.full(m, R),
        b=np.full(m, b),
        slashing_rate=np.full(m, alpha),
        mu=np.full(m, p),
        beta=beta,
        s=np.full(n, W),
        risk_aversion=np.full(n, eta),
        K=np.full(n, kappa),
        minimum_exposure_multiplier=np.zeros(n),
        position_labels=tuple(f"position_{j+1:02d}" for j in range(m)),
        operator_cluster=tuple(0 for _ in range(m)),
        client_cluster=tuple(0 for _ in range(m)),
        cloud_cluster=tuple(0 for _ in range(m)),
        regime="homogeneous_equicorrelated",
        seed=seed,
    )


__all__ = ["make_environment", "make_homogeneous_equicorrelated"]
