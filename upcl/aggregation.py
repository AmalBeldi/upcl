"""
Uncertainty-aware aggregation operators Gamma_theta (Section 4.7).

Every operator maps {(r(u,i,omega), P(omega))}_{omega in Omega} to a single
real-valued score S_theta(u,i). This module implements exactly the five
policies discussed in the paper: Expected Utility, Maxmin, Hurwicz,
Conditional Value-at-Risk, and Distributionally Robust Optimization.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_arrays(scores: Sequence[float], probs: Sequence[float]):
    s = np.asarray(scores, dtype=float)
    p = np.asarray(probs, dtype=float)
    if s.shape != p.shape:
        raise ValueError("scores and probs must have the same length.")
    if not np.isclose(p.sum(), 1.0, atol=1e-6):
        raise ValueError(f"probs must sum to 1 (got {p.sum()}).")
    return s, p


def expected_utility(scores: Sequence[float], probs: Sequence[float]) -> float:
    """Gamma_EU (Section 4.7.0.0.1): risk-neutral expectation over worlds."""
    s, p = _as_arrays(scores, probs)
    return float(np.dot(p, s))


def maxmin(scores: Sequence[float], probs: Sequence[float] = None) -> float:
    """Gamma_MM (Section 4.7.0.0.2): worst-case (conservative) score.

    probs is accepted for a uniform call signature across operators but is
    not used, consistent with the paper's definition Gamma_MM = min_omega r.
    """
    s = np.asarray(scores, dtype=float)
    return float(np.min(s))


def hurwicz(scores: Sequence[float], probs: Sequence[float] = None, alpha: float = 0.5) -> float:
    """Gamma_H,alpha (Section 4.7.0.0.3): optimism-pessimism compromise.

    alpha in [0,1]; alpha -> 1 is fully optimistic, alpha -> 0 fully
    conservative (reduces to Maxmin when alpha = 0).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    s = np.asarray(scores, dtype=float)
    return float(alpha * np.max(s) + (1.0 - alpha) * np.min(s))


def _cvar_from_losses(losses: np.ndarray, probs: np.ndarray, beta: float) -> float:
    """Exact discrete-distribution CVaR_beta via the Rockafellar-Uryasev
    formula minimized over tau, evaluated at the (weighted) empirical
    quantile -- standard closed form for finite support distributions.
    """
    order = np.argsort(losses)
    losses_sorted = losses[order]
    probs_sorted = probs[order]
    cum = np.cumsum(probs_sorted)

    # VaR_beta: smallest loss such that P(L <= VaR) >= beta.
    idx = int(np.searchsorted(cum, beta))
    idx = min(idx, len(losses_sorted) - 1)
    var_beta = losses_sorted[idx]

    tail_mass = 1.0 - beta
    if tail_mass <= 1e-12:
        return float(var_beta)

    excess = np.maximum(losses_sorted - var_beta, 0.0)
    expected_excess = float(np.dot(probs_sorted, excess))
    return float(var_beta + expected_excess / tail_mass)


def cvar(
    scores: Sequence[float],
    probs: Sequence[float],
    beta: float = 0.9,
    is_loss: bool = False,
) -> float:
    """Gamma_CVaR,beta (Section 4.7.0.0.4).

    The paper aggregates *utility*, defined as Gamma_CVaR = -CVaR_beta(L).
    By default `scores` are utilities (r(u,i,omega)) and the loss is taken
    as L = -r; pass is_loss=True if you already provide losses L(u,i,omega).
    """
    s, p = _as_arrays(scores, probs)
    losses = s if is_loss else -s
    cvar_beta = _cvar_from_losses(losses, p, beta)
    return float(-cvar_beta)


def dro(
    scores: Sequence[float],
    probs: Sequence[float],
    radius: float = 0.1,
    ambiguity: str = "kl",
) -> float:
    """Gamma_DRO (Section 4.7.0.0.5): worst-case expectation over an
    ambiguity set U(P) of plausible contextual distributions.

    ambiguity="kl": U(P) = {Q : KL(Q || P) <= radius}. The robust value
        inf_{Q in U(P)} E_Q[r] is computed via the standard dual form
        (Ben-Tal et al. / Hu & Hong):
            inf_Q E_Q[r] = max_{lambda >= 0} [ -lambda * log E_P[exp(-r/lambda)] - lambda * radius ]
        solved by 1-D convex optimization over lambda > 0.
    ambiguity="box": U(P) is a componentwise simplex ball of given radius
        (L1 perturbation budget), solved by a simple linear program via
        greedy mass shifting onto the worst-case outcomes (closed form for
        the L1/simplex case).
    """
    s, p = _as_arrays(scores, probs)

    if ambiguity == "box":
        return _dro_l1_ball(s, p, radius)
    elif ambiguity == "kl":
        return _dro_kl_ball(s, p, radius)
    else:
        raise ValueError("ambiguity must be 'kl' or 'box'.")


def _dro_l1_ball(scores: np.ndarray, probs: np.ndarray, radius: float) -> float:
    """Worst-case expectation under an L1 (total variation-like) budget:
    move up to `radius` probability mass from the highest-score worlds to
    the lowest-score world. Closed-form greedy solution to the resulting
    linear program (a classical result for L1-ball DRO on a finite support).
    """
    radius = float(np.clip(radius, 0.0, 1.0))
    order = np.argsort(scores)  # ascending: worst outcome first
    p = probs.copy()
    budget = radius

    # Push mass toward the worst (lowest-score) outcome, capped by what can
    # be removed from the best outcomes without producing negative mass.
    worst_idx = order[0]
    for idx in order[::-1]:  # from best to worst score, excluding worst itself
        if idx == worst_idx or budget <= 1e-15:
            continue
        shift = min(p[idx], budget)
        p[idx] -= shift
        p[worst_idx] += shift
        budget -= shift

    return float(np.dot(p, scores))


def _dro_kl_ball(scores: np.ndarray, probs: np.ndarray, radius: float, n_grid: int = 200) -> float:
    """Robust value under a KL-divergence ambiguity ball, via the standard
    convex dual: inf_lambda>0 [-lambda * log E_P[exp(-r/lambda)] - lambda*radius].
    Solved by a 1-D bounded search (robust to the small dimensionality of
    the possible-world space typically involved).
    """
    from scipy.optimize import minimize_scalar

    if radius <= 1e-8:
        return float(np.dot(probs, scores))

    def dual_objective(lam: float) -> float:
        if lam <= 1e-9:
            return 1e18
        # log-sum-exp for numerical stability
        z = -scores / lam
        m = np.max(z)
        log_e = m + np.log(np.dot(probs, np.exp(z - m)))
        return -(-lam * log_e - lam * radius)  # negate: we minimize -objective

    score_range = float(np.max(scores) - np.min(scores)) + 1.0
    res = minimize_scalar(
        dual_objective, bounds=(1e-6, 1e4 * score_range), method="bounded"
    )
    return float(-res.fun)


AGGREGATION_OPERATORS = {
    "eu": expected_utility,
    "maxmin": maxmin,
    "hurwicz": hurwicz,
    "cvar": cvar,
    "dro": dro,
}
