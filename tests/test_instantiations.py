import math

from upcl.context import (
    ContextualVariable,
    ProbabilisticContextInstance,
    degenerate_context,
    phi,
)
from upcl.instantiations import upcl_model, upcl_post, upcl_pre
from upcl.recommenders.toy import HashedContextualRecommender, LookupWorldwiseRecommender


def _alice_space():
    mood = ContextualVariable("Mood", {"Happy": 0.65, "Tired": 0.35})
    return phi(ProbabilisticContextInstance(categorical_variables=[mood]))


def test_upcl_pre_reproduces_alice_example():
    space = _alice_space()
    w_happy, w_tired = space.worlds  # order follows insertion order of the dict

    table = {
        ("Alice", "A", w_happy): 4.8,
        ("Alice", "A", w_tired): 3.2,
        ("Alice", "B", w_happy): 3.9,
        ("Alice", "B", w_tired): 4.5,
    }
    recommender = LookupWorldwiseRecommender(table)

    best, scores = upcl_pre("Alice", ["A", "B"], space, recommender, policy="eu")
    assert best == "A"
    assert math.isclose(scores["A"], 4.24, abs_tol=1e-9)
    assert math.isclose(scores["B"], 4.11, abs_tol=1e-9)


def test_upcl_post_reproduces_alice_example():
    space = _alice_space()
    w_happy, w_tired = space.worlds

    def base_recommender(user, item):
        return {"A": 4.3, "B": 4.1}[item]

    def adjustment(r0, user, item, world):
        table = {
            ("A", w_happy): 4.8,
            ("A", w_tired): 3.2,
            ("B", w_happy): 3.9,
            ("B", w_tired): 4.5,
        }
        return table[(item, world)]

    best, scores = upcl_post("Alice", ["A", "B"], space, base_recommender, adjustment, policy="eu")
    assert best == "A"
    assert math.isclose(scores["A"], 4.24, abs_tol=1e-9)
    assert math.isclose(scores["B"], 4.11, abs_tol=1e-9)


def test_upcl_model_runs_end_to_end():
    space = _alice_space()
    hashed = HashedContextualRecommender(seed=1)

    def encoder(worlds, probs):
        # A minimal permutation-invariant encoder: probability-weighted mix
        # of one-hot-like world identifiers, standing in for a learned E_phi.
        return sum(p * hash(w) % 1000 for w, p in zip(worlds, probs))

    def recommender(user, item, z):
        return hashed.base(user, item) + (hash((item, round(z, 3))) % 100) / 1000.0

    best, scores = upcl_model("Alice", ["A", "B", "C"], space, encoder, recommender)
    assert best in {"A", "B", "C"}
    assert len(scores) == 3


def test_theorem2_reduction_to_deterministic():
    """Theorem 2: with a degenerate distribution (P(omega*) = 1), UPCL-Pre
    must reduce exactly to deterministic context-aware recommendation,
    i.e. S(u,i) = f(u,i,omega*) and i* = argmax_i f(u,i,omega*).
    """
    space = degenerate_context({"Mood": "Happy"})
    (w_star,) = space.worlds

    table = {("Alice", "A", w_star): 4.8, ("Alice", "B", w_star): 3.9}
    recommender = LookupWorldwiseRecommender(table)

    best, scores = upcl_pre("Alice", ["A", "B"], space, recommender, policy="eu")
    assert best == "A"
    assert math.isclose(scores["A"], 4.8, abs_tol=1e-9)
    assert math.isclose(scores["B"], 3.9, abs_tol=1e-9)
