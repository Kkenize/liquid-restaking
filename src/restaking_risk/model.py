from __future__ import annotations

"""Payoffs, pseudo-gradients, and gross slash claims.

The model computes operator-level payoffs and analytical marginal-payoff
gradients. The quadratic risk term uses the loss covariance

    V = diag(alpha) Cov(F) diag(alpha)

rather than a bare correlation matrix, and the objective follows the paper's
``eta/2`` convention.
"""

from typing import Any

import numpy as np

from .environment import Environment
from .projection import allocation_is_feasible


def validate_strategies(strategies: np.ndarray, game: Environment) -> np.ndarray:
    strategies = np.asarray(strategies, dtype=float)
    expected = (game.n, game.number_of_operators)
    if strategies.shape != expected:
        raise ValueError(f"strategies must have shape {expected}, received {strategies.shape}")
    if not np.all(np.isfinite(strategies)):
        raise ValueError("strategies contain a non-finite value")
    for i, row in enumerate(strategies):
        if not allocation_is_feasible(
            row,
            float(game.W[i]),
            float(game.kappa[i]),
            minimum_multiplier=float(game.lower_multiplier[i]),
        ):
            raise ValueError(f"allocator {i} violates its exposure constraints")
    return strategies


def calculate_av_stakes(strategies: np.ndarray, game: Environment, *, validate: bool = True) -> np.ndarray:
    a = validate_strategies(strategies, game) if validate else np.asarray(strategies, dtype=float)
    return a @ game.theta


def calculate_other_av_stake(
    restaker_index: int,
    strategies: np.ndarray,
    game: Environment,
    *,
    validate: bool = True,
) -> np.ndarray:
    if not 0 <= restaker_index < game.n:
        raise IndexError("restaker_index is outside the valid range")
    av_stakes = calculate_av_stakes(strategies, game, validate=validate)
    return av_stakes.sum(axis=0) - av_stakes[restaker_index]


def payoff_components(
    restaker_index: int,
    allocation: np.ndarray,
    strategies: np.ndarray,
    game: Environment,
    V: np.ndarray | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Return the reward, expected slashing loss, and risk-cost components."""
    if not 0 <= restaker_index < game.n:
        raise IndexError("restaker_index is outside the valid range")
    profile = validate_strategies(strategies, game) if validate else np.asarray(strategies, dtype=float)
    a = np.asarray(allocation, dtype=float)
    if a.shape != (game.number_of_operators,):
        raise ValueError("allocation must contain one value per operator")
    if validate and not allocation_is_feasible(
        a,
        float(game.W[restaker_index]),
        float(game.kappa[restaker_index]),
        minimum_multiplier=float(game.lower_multiplier[restaker_index]),
    ):
        raise ValueError("candidate allocation violates the exposure constraints")

    av_stake = a @ game.theta
    other_av_stake = calculate_other_av_stake(restaker_index, profile, game, validate=False)
    total_av_stake = av_stake + other_av_stake
    reward_share = av_stake / (game.b + total_av_stake)
    av_rewards = game.R * reward_share
    reward = float(av_rewards.sum())

    C, d, V_true = game.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)
    expected_by_position = d * av_stake
    expected_slashing_loss = float(expected_by_position.sum())
    loss_variance = float(av_stake @ V_use @ av_stake)
    risk_penalty = 0.5 * float(game.eta[restaker_index]) * loss_variance

    return {
        "restaker_index": restaker_index,
        "total_stake": float(game.W[restaker_index]),
        "risk_aversion": float(game.eta[restaker_index]),
        "strategy": a.copy(),
        "gross_exposure": float(av_stake.sum()),
        "av_stake": av_stake,
        "other_av_stake": other_av_stake,
        "total_av_stake": total_av_stake,
        "reward_share": reward_share,
        "av_rewards": av_rewards,
        "slashable_stake": av_stake * game.alpha,
        "expected_av_slashing_loss": expected_by_position,
        "reward": reward,
        "expected_slashing_loss": expected_slashing_loss,
        "loss_variance": loss_variance,
        "risk_penalty": risk_penalty,
        "payoff": reward - expected_slashing_loss - risk_penalty,
        "indicator_covariance": C,
    }


def payoff(
    restaker_index: int,
    allocation: np.ndarray,
    strategies: np.ndarray,
    game: Environment,
    V: np.ndarray | None = None,
    *,
    validate: bool = True,
) -> float:
    return float(payoff_components(restaker_index, allocation, strategies, game, V, validate=validate)["payoff"])


def marginal_payoff_rates(
    restaker_index: int,
    allocation: np.ndarray,
    strategies: np.ndarray,
    game: Environment,
    V: np.ndarray | None = None,
    *,
    validate: bool = False,
) -> np.ndarray:
    """Own-payoff gradient in the operator-allocation coordinates.

    The calculation maps the selected allocation through ``theta``,
    computes proportional-reward marginal returns, and maps the
    position-level derivatives back to operator space.
    """
    a = np.asarray(allocation, dtype=float)
    profile = validate_strategies(strategies, game) if validate else np.asarray(strategies, dtype=float)
    av_stake = a @ game.theta
    all_av_stake = profile @ game.theta
    other_av_stake = all_av_stake.sum(axis=0) - all_av_stake[restaker_index]
    total_av_stake = av_stake + other_av_stake

    marginal_position_reward = game.R * (game.b + other_av_stake) / (game.b + total_av_stake) ** 2
    operator_marginal_reward = game.theta @ marginal_position_reward

    _, d, V_true = game.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)
    operator_expected_loss = game.theta @ d
    operator_risk_matrix = game.theta @ V_use @ game.theta.T
    marginal_risk_penalty = float(game.eta[restaker_index]) * (operator_risk_matrix @ a)
    return operator_marginal_reward - operator_expected_loss - marginal_risk_penalty


def aggregate_exposure(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).sum(axis=0)


def individual_rewards(env: Environment, x: np.ndarray) -> np.ndarray:
    X = aggregate_exposure(x)
    return np.sum(env.R[None, :] * x / (env.b + X)[None, :], axis=1)


def total_reward(env: Environment, x: np.ndarray) -> float:
    X = aggregate_exposure(x)
    return float(np.sum(env.R * X / (env.b + X)))


def utility_vector(env: Environment, x: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
    _, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)
    rewards = individual_rewards(env, x)
    expected = x @ d
    risk = np.array([0.5 * env.eta[i] * x[i] @ V_use @ x[i] for i in range(env.n)])
    return rewards - expected - risk


def social_welfare(env: Environment, x: np.ndarray, V: np.ndarray | None = None) -> float:
    _, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)
    X = aggregate_exposure(x)
    expected = float(d @ X)
    private_risk = float(sum(0.5 * env.eta[i] * x[i] @ V_use @ x[i] for i in range(env.n)))
    return total_reward(env, x) - expected - private_risk


def pseudo_gradient(env: Environment, x: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
    """Compute the stacked own-payoff gradients using vectorized operations."""
    x = np.asarray(x, dtype=float)
    _, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)
    X = x.sum(axis=0)
    total = env.b + X
    others_plus_background = total[None, :] - x
    reward_grad = env.R[None, :] * others_plus_background / total[None, :] ** 2
    risk_grad = env.eta[:, None] * (x @ V_use.T)
    return reward_grad - d[None, :] - risk_grad


def best_response_objective_and_grad(
    env: Environment,
    i: int,
    y: np.ndarray,
    x_others_sum: np.ndarray,
    V: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    _, d, V_true = env.risk_matrices()
    V_use = V_true if V is None else np.asarray(V, dtype=float)
    background = env.b + np.asarray(x_others_sum, dtype=float)
    reward = float(np.sum(env.R * y / (background + y)))
    value = reward - float(d @ y) - 0.5 * float(env.eta[i]) * float(y @ V_use @ y)
    grad = env.R * background / (background + y) ** 2 - d - env.eta[i] * (V_use @ y)
    return value, grad


def gross_claim_samples(env: Environment, x: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    monetary_events = np.asarray(F, dtype=float) * env.alpha[None, :]
    individual = monetary_events @ np.asarray(x, dtype=float).T
    aggregate = individual.sum(axis=1)
    return individual, aggregate


__all__ = [
    "validate_strategies",
    "calculate_av_stakes",
    "calculate_other_av_stake",
    "payoff_components",
    "payoff",
    "marginal_payoff_rates",
    "aggregate_exposure",
    "individual_rewards",
    "total_reward",
    "utility_vector",
    "social_welfare",
    "pseudo_gradient",
    "best_response_objective_and_grad",
    "gross_claim_samples",
]
