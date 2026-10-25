from __future__ import annotations

"""Game parameters and feasible Bernoulli slash dependence.

This module defines the game's parameters (``theta``, ``R``, ``b``,
``slashing_rate``, ``mu``, ``s``, ``risk_aversion``, and ``K``)
and models correlated slashing through a Gaussian threshold model.
The resulting covariance matrix is attainable by joint Bernoulli
slash indicators.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import multivariate_normal

Array = np.ndarray


@dataclass(frozen=True)
class Game:
    """Parameters of the reuse-capped restaking allocation game.

    Convenience aliases ``W``, ``kappa``, ``p``, ``alpha``, and ``eta``
    expose the notation used in the paper and experiment scripts.

    ``theta`` maps operator allocations into slashable-position exposure.
    The paper experiments use the identity map so that the choice variable
    is direct exposure. The general mapping supports future extensions.
    """

    theta: Array
    R: Array
    b: Array
    slashing_rate: Array
    mu: Array
    beta: Array
    s: Array
    risk_aversion: Array
    K: float | Array
    minimum_exposure_multiplier: float | Array = 0.0
    position_labels: tuple[str, ...] = ()
    operator_cluster: tuple[int, ...] = ()
    client_cluster: tuple[int, ...] = ()
    cloud_cluster: tuple[int, ...] = ()
    regime: str = "custom"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "theta",
            "R",
            "b",
            "slashing_rate",
            "mu",
            "beta",
            "s",
            "risk_aversion",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))

        n = int(np.asarray(self.s).size)
        kappa = np.asarray(self.K, dtype=float)
        if kappa.ndim == 0:
            kappa = np.full(n, float(kappa))
        lower = np.asarray(self.minimum_exposure_multiplier, dtype=float)
        if lower.ndim == 0:
            lower = np.full(n, float(lower))
        object.__setattr__(self, "K", kappa)
        object.__setattr__(self, "minimum_exposure_multiplier", lower)
        self.validate()

    @property
    def number_of_restakers(self) -> int:
        return int(self.s.shape[0])

    @property
    def number_of_operators(self) -> int:
        return int(self.theta.shape[0])

    @property
    def number_of_avs(self) -> int:
        return int(self.theta.shape[1])

    # Paper-notation aliases used throughout the experiment suite.
    @property
    def n(self) -> int:
        return self.number_of_restakers

    @property
    def m(self) -> int:
        return self.number_of_avs

    @property
    def W(self) -> Array:
        return self.s

    @property
    def kappa(self) -> Array:
        return np.asarray(self.K, dtype=float)

    @property
    def p(self) -> Array:
        return self.mu

    @property
    def alpha(self) -> Array:
        return self.slashing_rate

    @property
    def eta(self) -> Array:
        return self.risk_aversion

    @property
    def lower_multiplier(self) -> Array:
        return np.asarray(self.minimum_exposure_multiplier, dtype=float)

    def validate(self) -> None:
        if self.theta.ndim != 2 or min(self.theta.shape) <= 0:
            raise ValueError("theta must be a nonempty two-dimensional matrix")
        n_ops, m = self.theta.shape
        n = self.number_of_restakers

        if self.R.shape != (m,) or self.b.shape != (m,):
            raise ValueError("R and b must each have one entry per slashable position")
        if self.slashing_rate.shape != (m,) or self.mu.shape != (m,):
            raise ValueError("slashing_rate and mu must each have one entry per position")
        if self.beta.ndim != 2 or self.beta.shape[0] != m:
            raise ValueError("beta must have shape (number_of_positions, number_of_factors)")
        if self.s.shape != (n,) or self.risk_aversion.shape != (n,):
            raise ValueError("s and risk_aversion must each have one entry per restaker")
        if self.kappa.shape != (n,) or self.lower_multiplier.shape != (n,):
            raise ValueError("K and minimum_exposure_multiplier must be scalar or length n")

        arrays = [
            self.theta,
            self.R,
            self.b,
            self.slashing_rate,
            self.mu,
            self.beta,
            self.s,
            self.risk_aversion,
            self.kappa,
            self.lower_multiplier,
        ]
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("all game parameters must be finite")
        if np.any(self.theta < 0):
            raise ValueError("theta must be nonnegative")
        if np.any(self.R <= 0) or np.any(self.b <= 0):
            raise ValueError("reward pools and background exposure must be positive")
        if np.any((self.slashing_rate <= 0) | (self.slashing_rate > 1)):
            raise ValueError("slashing rates must lie in (0,1]")
        if np.any((self.mu <= 0) | (self.mu >= 1)):
            raise ValueError("failure probabilities must lie in (0,1)")
        if np.any(self.s <= 0) or np.any(self.risk_aversion < 0):
            raise ValueError("capital must be positive and risk aversion nonnegative")
        if np.any(self.kappa < 0) or np.any(self.kappa > self.number_of_operators + 1e-12):
            raise ValueError("each reuse cap must lie in [0, number_of_operators]")
        if np.any(self.lower_multiplier < 0) or np.any(self.lower_multiplier > self.kappa + 1e-12):
            raise ValueError("minimum exposure multipliers must lie in [0,K]")
        if np.any(np.sum(self.beta**2, axis=1) > 1 + 1e-10):
            raise ValueError("each factor-loading row must have squared norm at most one")

        # The controlled paper experiments use direct position exposure.  We
        # retain a general nonnegative map, but require each row not to create
        # more exposure in a single position than the delegated unit itself.
        if np.any(self.theta > 1 + 1e-12):
            raise ValueError("entries of theta must not exceed one")
        if n_ops == 0:
            raise ValueError("at least one operator is required")

    @property
    def thresholds(self) -> Array:
        return ndtri(self.mu)

    @property
    def latent_covariance(self) -> Array:
        residual = 1.0 - np.sum(self.beta**2, axis=1)
        return self.beta @ self.beta.T + np.diag(residual)

    def binary_covariance(self) -> Array:
        """Exact covariance of the thresholded Bernoulli slash indicators."""
        cached = self.__dict__.get("_binary_covariance_cache")
        if cached is not None:
            return cached.copy()

        tau = self.thresholds
        latent = self.beta @ self.beta.T
        C = np.empty((self.m, self.m), dtype=float)
        C[np.diag_indices(self.m)] = self.mu * (1.0 - self.mu)
        for j in range(self.m):
            for k in range(j + 1, self.m):
                corr = float(np.clip(latent[j, k], -0.999999, 0.999999))
                joint = multivariate_normal.cdf(
                    np.array([tau[j], tau[k]]),
                    mean=np.zeros(2),
                    cov=np.array([[1.0, corr], [corr, 1.0]]),
                )
                C[j, k] = C[k, j] = float(joint - self.mu[j] * self.mu[k])

        C = 0.5 * (C + C.T)
        vals, vecs = np.linalg.eigh(C)
        if vals.min() < -1e-7:
            raise ValueError(f"binary covariance unexpectedly indefinite: {vals.min():.3e}")
        # Remove only numerical roundoff while preserving the Bernoulli diagonal.
        C = (vecs * np.maximum(vals, 0.0)) @ vecs.T
        np.fill_diagonal(C, self.mu * (1.0 - self.mu))
        C = 0.5 * (C + C.T)
        object.__setattr__(self, "_binary_covariance_cache", C.copy())
        return C

    @property
    def Sigma(self) -> Array:
        """Bernoulli slash-indicator correlation matrix (compatibility alias)."""
        C = self.binary_covariance()
        sd = np.sqrt(self.mu * (1.0 - self.mu))
        return C / np.outer(sd, sd)

    def risk_matrices(self) -> tuple[Array, Array, Array]:
        """Return indicator covariance C, expected-loss rates d, and loss covariance V."""
        cached = self.__dict__.get("_risk_matrices_cache")
        if cached is not None:
            return tuple(a.copy() for a in cached)
        C = self.binary_covariance()
        d = self.slashing_rate * self.mu
        V = self.slashing_rate[:, None] * C * self.slashing_rate[None, :]
        object.__setattr__(self, "_risk_matrices_cache", (C.copy(), d.copy(), V.copy()))
        return C, d, V

    def sample_slashes(self, n_samples: int, seed: int | None = None) -> Array:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        n_factors = self.beta.shape[1]
        factors = rng.standard_normal((n_samples, n_factors)) if n_factors else np.zeros((n_samples, 0))
        eps = rng.standard_normal((n_samples, self.m))
        residual = np.sqrt(np.maximum(0.0, 1.0 - np.sum(self.beta**2, axis=1)))
        latent = factors @ self.beta.T + eps * residual[None, :]
        return (latent <= self.thresholds[None, :]).astype(float)

    def with_updates(self, **updates: Any) -> "Game":
        aliases = {
            "W": "s",
            "kappa": "K",
            "p": "mu",
            "alpha": "slashing_rate",
            "eta": "risk_aversion",
        }
        normalized = {aliases.get(key, key): value for key, value in updates.items()}
        data = {
            "theta": self.theta.copy(),
            "R": self.R.copy(),
            "b": self.b.copy(),
            "slashing_rate": self.slashing_rate.copy(),
            "mu": self.mu.copy(),
            "beta": self.beta.copy(),
            "s": self.s.copy(),
            "risk_aversion": self.risk_aversion.copy(),
            "K": self.kappa.copy(),
            "minimum_exposure_multiplier": self.lower_multiplier.copy(),
            "position_labels": self.position_labels,
            "operator_cluster": self.operator_cluster,
            "client_cluster": self.client_cluster,
            "cloud_cluster": self.cloud_cluster,
            "regime": self.regime,
            "seed": self.seed,
        }
        data.update(normalized)
        return Game(**data)

    def to_json(self, path: str | Path) -> None:
        payload = asdict(self)
        for key in (
            "theta",
            "R",
            "b",
            "slashing_rate",
            "mu",
            "beta",
            "s",
            "risk_aversion",
            "K",
            "minimum_exposure_multiplier",
        ):
            payload[key] = np.asarray(payload[key]).tolist()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Game":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# Backward-compatible name used by the earlier experiment orchestration.
Environment = Game

__all__ = ["Game", "Environment"]
