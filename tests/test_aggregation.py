import math

from upcl.aggregation import cvar, dro, expected_utility, hurwicz, maxmin


def test_expected_utility_matches_alice_example():
    """Reproduces the paper's Section 5.1 running example exactly:
    S(A) = 0.65*4.8 + 0.35*3.2 = 4.24 ; S(B) = 0.65*3.9 + 0.35*4.5 = 4.11.
    """
    s_a = expected_utility([4.8, 3.2], [0.65, 0.35])
    s_b = expected_utility([3.9, 4.5], [0.65, 0.35])
    assert math.isclose(s_a, 4.24, abs_tol=1e-9)
    assert math.isclose(s_b, 4.11, abs_tol=1e-9)
    assert s_a > s_b  # i* = A, as stated in the paper


def test_maxmin_is_worst_case():
    assert maxmin([4.8, 3.2]) == 3.2


def test_hurwicz_bounds():
    scores = [4.8, 3.2]
    assert math.isclose(hurwicz(scores, alpha=1.0), max(scores))
    assert math.isclose(hurwicz(scores, alpha=0.0), min(scores))
    mid = hurwicz(scores, alpha=0.5)
    assert min(scores) < mid < max(scores)


def test_hurwicz_reduces_to_maxmin_at_alpha_zero():
    scores = [4.8, 3.2, 1.0]
    assert math.isclose(hurwicz(scores, alpha=0.0), maxmin(scores))


def test_cvar_matches_expectation_at_beta_zero():
    """As beta -> 0, CVaR_beta(L) -> E[L], so Gamma_CVaR -> -E[L] = E[r]."""
    scores = [4.8, 3.2, 2.0]
    probs = [0.5, 0.3, 0.2]
    val = cvar(scores, probs, beta=1e-9)
    eu = expected_utility(scores, probs)
    assert math.isclose(val, eu, abs_tol=1e-6)


def test_cvar_is_at_most_expected_utility():
    """CVaR focuses on the worst tail, so it must not exceed the mean utility."""
    scores = [4.8, 3.2, 2.0, 1.0]
    probs = [0.4, 0.3, 0.2, 0.1]
    assert cvar(scores, probs, beta=0.7) <= expected_utility(scores, probs) + 1e-9


def test_dro_kl_ball_worst_case_bounded_by_min_and_mean():
    scores = [4.8, 3.2, 2.0]
    probs = [0.5, 0.3, 0.2]
    val = dro(scores, probs, radius=0.5, ambiguity="kl")
    assert min(scores) - 1e-6 <= val <= expected_utility(scores, probs) + 1e-6


def test_dro_kl_ball_shrinks_toward_expectation_as_radius_to_zero():
    scores = [4.8, 3.2, 2.0]
    probs = [0.5, 0.3, 0.2]
    val = dro(scores, probs, radius=1e-9, ambiguity="kl")
    assert math.isclose(val, expected_utility(scores, probs), abs_tol=1e-3)


def test_dro_box_worst_case_bounded():
    scores = [4.8, 3.2, 2.0]
    probs = [0.5, 0.3, 0.2]
    val = dro(scores, probs, radius=0.3, ambiguity="box")
    assert min(scores) - 1e-9 <= val <= expected_utility(scores, probs) + 1e-9
