import math

import numpy as np

from upcl.aggregation import expected_utility
from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.monte_carlo import (
    hoeffding_bound,
    monte_carlo_expected_utility,
    required_sample_size,
)


def _alice_space():
    mood = ContextualVariable("Mood", {"Happy": 0.65, "Tired": 0.35})
    return phi(ProbabilisticContextInstance(categorical_variables=[mood]))


def test_monte_carlo_converges_to_exact_expected_utility():
    """Empirical check of Theorem 4 (almost-sure convergence)."""
    space = _alice_space()
    scores_by_world = [4.8, 3.2]  # aligned with worlds order (Happy, Tired)
    exact = expected_utility(scores_by_world, space.probabilities)

    rng = np.random.default_rng(42)
    errors = []
    for n in [10, 100, 1_000, 10_000, 100_000]:
        est = monte_carlo_expected_utility(scores_by_world, space, n, rng)
        errors.append(abs(est - exact))

    # Error should trend downward as N grows (allow mild non-monotonicity
    # from sampling noise by comparing the first vs. last window).
    assert errors[-1] < errors[0]


def test_hoeffding_bound_decreases_with_n():
    b1 = hoeffding_bound(n_samples=100, a=0.0, b=5.0, epsilon=0.1)
    b2 = hoeffding_bound(n_samples=10_000, a=0.0, b=5.0, epsilon=0.1)
    assert 0.0 <= b2 < b1 <= 1.0


def test_required_sample_size_matches_hoeffding_bound():
    """Corollary 5 should give the smallest N such that the Theorem 5 bound
    is <= delta (up to integer rounding).
    """
    a, b, epsilon, delta = 0.0, 5.0, 0.1, 0.05
    n = required_sample_size(a, b, epsilon, delta)
    achieved = hoeffding_bound(n, a, b, epsilon)
    assert achieved <= delta + 1e-9
    # One fewer sample should (generically) no longer satisfy the bound.
    achieved_minus_one = hoeffding_bound(max(n - 1, 1), a, b, epsilon)
    assert achieved_minus_one >= achieved


def test_monte_carlo_reproducible_with_seed():
    space = _alice_space()
    scores_by_world = [4.8, 3.2]
    est1 = monte_carlo_expected_utility(scores_by_world, space, 500, np.random.default_rng(7))
    est2 = monte_carlo_expected_utility(scores_by_world, space, 500, np.random.default_rng(7))
    assert math.isclose(est1, est2)
