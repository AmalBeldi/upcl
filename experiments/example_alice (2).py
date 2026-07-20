#!/usr/bin/env python3
"""
Reproduces the paper's running example (Sections 3.1, 4.4, 5.1, 5.3)
end to end, exactly, using UPCL-Pre and UPCL-Post, and prints the same
numbers reported in the paper.

Run:
    python experiments/example_alice.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.instantiations import upcl_post, upcl_pre
from upcl.recommenders.toy import LookupWorldwiseRecommender


def main() -> None:
    mood = ContextualVariable("Mood", {"Happy": 0.65, "Tired": 0.35})
    space = phi(ProbabilisticContextInstance(categorical_variables=[mood]))
    w_happy, w_tired = space.worlds

    print("Probabilistic Context Space (Section 4.4):")
    for w, p in zip(space.worlds, space.probabilities):
        print(f"  world={w!r:35} P={p:.2f}")

    # --- UPCL-Pre (Section 5.1) ---
    table = {
        ("Alice", "A", w_happy): 4.8,
        ("Alice", "A", w_tired): 3.2,
        ("Alice", "B", w_happy): 3.9,
        ("Alice", "B", w_tired): 4.5,
    }
    pre_recommender = LookupWorldwiseRecommender(table)
    best_pre, scores_pre = upcl_pre("Alice", ["A", "B"], space, pre_recommender, policy="eu")

    print("\nUPCL-Pre (Expected Utility aggregation):")
    for item, score in scores_pre.items():
        print(f"  S(u,{item}) = {score:.4f}")
    print(f"  => i* = {best_pre}  (paper reports i* = A, S(A)=4.24, S(B)=4.11)")

    # --- UPCL-Post (Section 5.3) ---
    def base_recommender(user, item):
        return {"A": 4.3, "B": 4.1}[item]

    def adjustment(r0, user, item, world):
        return table[("Alice", item, world)]

    best_post, scores_post = upcl_post(
        "Alice", ["A", "B"], space, base_recommender, adjustment, policy="eu"
    )

    print("\nUPCL-Post (Expected Utility aggregation):")
    print(f"  base scores: r0(A)={base_recommender('Alice','A')}, r0(B)={base_recommender('Alice','B')}")
    for item, score in scores_post.items():
        print(f"  S(u,{item}) = {score:.4f}")
    print(f"  => i* = {best_post}  (paper reports i* = A)")

    # --- Alternative risk policies on the same worlds, for illustration ---
    from upcl.aggregation import cvar, hurwicz, maxmin

    print("\nAlternative aggregation policies on UPCL-Pre scores (illustrative):")
    for item in ["A", "B"]:
        world_scores = [table[("Alice", item, w)] for w in space.worlds]
        mm = maxmin(world_scores)
        hz = hurwicz(world_scores, alpha=0.3)
        cv = cvar(world_scores, space.probabilities, beta=0.7)
        print(f"  item {item}: Maxmin={mm:.2f}  Hurwicz(a=0.3)={hz:.2f}  CVaR(b=0.7)={cv:.2f}")


if __name__ == "__main__":
    main()
