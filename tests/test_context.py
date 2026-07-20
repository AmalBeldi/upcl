import math

from upcl.context import (
    BooleanFact,
    ContextualVariable,
    ProbabilisticContextInstance,
    degenerate_context,
    phi,
)


def test_bid_normalization_single_variable():
    """Proposition 4: BID worlds sum to 1 for a single categorical block."""
    mood = ContextualVariable("mood", {"happy": 0.65, "tired": 0.35})
    space = phi(ProbabilisticContextInstance(categorical_variables=[mood]))
    assert len(space.worlds) == 2
    assert math.isclose(sum(space.probabilities), 1.0, abs_tol=1e-9)


def test_bid_normalization_multiple_variables():
    """Proposition 4 with two independent blocks: |Omega| = m1 * m2."""
    mood = ContextualVariable("mood", {"happy": 0.6, "tired": 0.4})
    location = ContextualVariable("location", {"home": 0.7, "work": 0.2, "outdoor": 0.1})
    space = phi(ProbabilisticContextInstance(categorical_variables=[mood, location]))
    assert len(space.worlds) == 6
    assert math.isclose(sum(space.probabilities), 1.0, abs_tol=1e-9)


def test_tid_normalization():
    """Proposition 3: TID worlds (2^N subsets) sum to 1."""
    facts = [BooleanFact("isWeekend", 0.3), BooleanFact("isRaining", 0.4)]
    space = phi(ProbabilisticContextInstance(boolean_facts=facts))
    assert len(space.worlds) == 4  # 2^2 subsets
    assert math.isclose(sum(space.probabilities), 1.0, abs_tol=1e-9)


def test_hybrid_tid_bid_normalization():
    """Section 4.5.0.0.3: hybrid worlds also sum to 1 under independence."""
    mood = ContextualVariable("mood", {"happy": 0.65, "tired": 0.35})
    facts = [BooleanFact("isWeekend", 0.3)]
    instance = ProbabilisticContextInstance(categorical_variables=[mood], boolean_facts=facts)
    space = phi(instance)
    assert len(space.worlds) == 2 * 2
    assert math.isclose(sum(space.probabilities), 1.0, abs_tol=1e-9)


def test_invalid_distribution_raises():
    raised = False
    try:
        ContextualVariable("mood", {"happy": 0.5, "tired": 0.6})  # sums to 1.1
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for a non-normalized distribution."


def test_degenerate_context_single_world():
    """Setup used by Theorem 2 (Reduction to Deterministic Context)."""
    space = degenerate_context({"mood": "happy"})
    assert len(space.worlds) == 1
    assert space.probabilities == [1.0]


def test_alice_running_example_worlds():
    """Reproduces the exact worlds/probabilities of Section 4.4's running example."""
    mood = ContextualVariable("Mood", {"Happy": 0.65, "Tired": 0.35})
    space = phi(ProbabilisticContextInstance(categorical_variables=[mood]))
    as_dict = space.as_dict()
    assert math.isclose(as_dict[(("Mood", "Happy"),)], 0.65, abs_tol=1e-9)
    assert math.isclose(as_dict[(("Mood", "Tired"),)], 0.35, abs_tol=1e-9)
