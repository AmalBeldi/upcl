"""
Controlled contextual noise injection (Section 7.2).

Simulates the sensor / inference uncertainty motivated in Section 3
(positioning uncertainty, mood-inference uncertainty, ...) by perturbing a
categorical contextual variable's probability distribution with a
Dirichlet-noise process of adjustable intensity.

At intensity 0, the distribution is unchanged. As intensity -> 1, the
distribution is driven toward a uniform (maximally uncertain) distribution
over the domain, mimicking increasingly unreliable contextual sensing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .context import ContextualVariable


def inject_noise(
    variable: ContextualVariable,
    intensity: float,
    rng: np.random.Generator,
    mode: str = "dirichlet",
) -> ContextualVariable:
    """Returns a new ContextualVariable with a perturbed distribution.

    Parameters
    ----------
    intensity : float in [0, 1]
        0 = no noise (identical distribution); 1 = maximal noise.
    mode : "dirichlet" (concentration decreases with intensity, so sampled
        distributions spread further from the mean) or "uniform_mix"
        (linear interpolation toward the uniform distribution, useful for
        a simple, deterministic noise sweep in ablations).
    """
    if not (0.0 <= intensity <= 1.0):
        raise ValueError("intensity must lie in [0, 1].")

    values = list(variable.distribution.keys())
    probs = np.array([variable.distribution[v] for v in values], dtype=float)
    k = len(values)

    if mode == "uniform_mix":
        uniform = np.full(k, 1.0 / k)
        noisy = (1.0 - intensity) * probs + intensity * uniform
    elif mode == "dirichlet":
        if intensity <= 1e-9:
            noisy = probs
        else:
            # Lower concentration -> noisier draw around the base distribution.
            concentration = max(1e-3, (1.0 - intensity)) * 50.0 * probs + 1e-3
            noisy = rng.dirichlet(concentration)
    else:
        raise ValueError("mode must be 'dirichlet' or 'uniform_mix'.")

    noisy = noisy / noisy.sum()
    new_dist: Dict = {v: float(p) for v, p in zip(values, noisy)}
    return ContextualVariable(name=variable.name, distribution=new_dist)


def collapse_to_mode(variable: ContextualVariable) -> ContextualVariable:
    """Deterministic baseline used for comparison throughout Section 7:
    commits to the single most likely value (probability 1), discarding
    all residual uncertainty -- exactly the behaviour UPCL argues against
    in the motivating example (Section 3.1).
    """
    mode_value = max(variable.distribution.items(), key=lambda kv: kv[1])[0]
    return ContextualVariable(name=variable.name, distribution={mode_value: 1.0})
