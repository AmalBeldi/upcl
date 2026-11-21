"""
Monte Carlo approximation of the uncertainty-aware aggregation (Section 6.2).

Implements:
  - Definition 11: Monte Carlo approximation of Expected Utility.
  - Theorem 4: almost-sure convergence (verified empirically in tests/experiments).
  - Theorem 5: Hoeffding finite-sample error bound.
  - Corollary 5: minimum sample size N for a target (epsilon, delta) guarantee.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .context import ContextualVariable, ProbabilisticContextSpace, World


def sample_worlds(space: ProbabilisticContextSpace, n_samples: int, rng: np.random.Generator) -> Sequence[World]:
    """Draws N possible worlds i.i.d. according to P^t_u (Definition 11)."""
    idx = rng.choice(len(space.worlds), size=n_samples, p=space.probabilities)
    return [space.worlds[i] for i in idx]


def monte_carlo_expected_utility(
    scores_by_world: Sequence[float],
    space: ProbabilisticContextSpace,
    n_samples: int,
    rng: np.random.Generator = None,
) -> float:
    """S_hat_N(u,i) = (1/N) * sum_j r(u,i,omega_j), omega_j ~ P(omega).

    `scores_by_world` must align positionally with `space.worlds`.
    """
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(space.worlds), size=n_samples, p=space.probabilities)
    scores = np.asarray(scores_by_world, dtype=float)
    return float(np.mean(scores[idx]))


def monte_carlo_expected_utility_independent(
    variables: Sequence["ContextualVariable"],
    score_fn,
    n_samples: int,
    rng: np.random.Generator = None,
) -> float:
    """Monte Carlo estimate of Expected Utility under BID semantics that never
    enumerates the world space.

    Each of the K independent blocks is sampled from its own marginal, so one
    world costs O(K) and the whole estimate O(N*K) calls/operations,
    independent of |Omega| = prod_k |domain_k|. `score_fn(world)` receives a
    world in the same ((name, value), ...) form produced by `phi`.
    """
    if rng is None:
        rng = np.random.default_rng()
    names = [v.name for v in variables]
    columns = []
    for v in variables:
        values = list(v.distribution.keys())
        probs = np.asarray([v.distribution[x] for x in values], dtype=float)
        probs = probs / probs.sum()
        idx = rng.choice(len(values), size=n_samples, p=probs)
        columns.append([values[i] for i in idx])
    total = 0.0
    for j in range(n_samples):
        world = tuple((names[k], columns[k][j]) for k in range(len(names)))
        total += float(score_fn(world))
    return total / n_samples


def hoeffding_bound(n_samples: int, a: float, b: float, epsilon: float) -> float:
    """Theorem 5: Pr(|S_hat_N - S| > epsilon) <= 2 exp(-2 N epsilon^2 / (b-a)^2)."""
    if b <= a:
        raise ValueError("Require b > a for a nondegenerate score range.")
    raw = 2.0 * math.exp(-2.0 * n_samples * epsilon**2 / (b - a) ** 2)
    return float(min(raw, 1.0))


def required_sample_size(a: float, b: float, epsilon: float, delta: float) -> int:
    """Corollary 5: N >= (b-a)^2 / (2 epsilon^2) * ln(2/delta)."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    n = ((b - a) ** 2) / (2.0 * epsilon**2) * math.log(2.0 / delta)
    return int(math.ceil(n))
