#!/usr/bin/env python3
"""
Scalability analysis (Section 7.4): recommendation quality and wall-clock
time as a function of the number of sampled possible worlds N, empirically
verifying the finite-sample bound of Corollary 5.

Run:
    python experiments/scalability_analysis.py --k-vars 6 --domain-size 4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.aggregation import expected_utility
from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.monte_carlo import hoeffding_bound, monte_carlo_expected_utility, required_sample_size
from upcl.recommenders.toy import HashedContextualRecommender


def build_large_context(k_vars: int, domain_size: int, rng: np.random.Generator):
    variables = []
    for k in range(k_vars):
        probs = rng.dirichlet(np.ones(domain_size))
        dist = {f"v{k}_{j}": float(probs[j]) for j in range(domain_size)}
        variables.append(ContextualVariable(f"C{k}", dist))
    return ProbabilisticContextInstance(categorical_variables=variables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-vars", type=int, default=6, help="Number of contextual variables K.")
    parser.add_argument("--domain-size", type=int, default=4, help="Domain size per variable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    instance = build_large_context(args.k_vars, args.domain_size, rng)

    t0 = time.perf_counter()
    space = phi(instance)
    t_exact_build = time.perf_counter() - t0

    exact_size = len(space.worlds)
    print(f"|Omega| = {args.domain_size}^{args.k_vars} = {exact_size} possible worlds "
          f"(built in {t_exact_build*1000:.1f} ms)")

    recommender = HashedContextualRecommender(seed=args.seed)
    user, item = "u0", "i0"

    t0 = time.perf_counter()
    world_scores = [recommender.worldwise(user, item, w) for w in space.worlds]
    exact_value = expected_utility(world_scores, space.probabilities)
    t_exact_eval = time.perf_counter() - t0
    print(f"Exact Expected Utility: S(u,i) = {exact_value:.4f}  "
          f"(full enumeration in {t_exact_eval*1000:.1f} ms)")

    a, b = 0.0, recommender.scale
    n_required = required_sample_size(a, b, args.epsilon, args.delta)
    print(
        f"\nCorollary 5: to guarantee P(|S_hat_N - S| > {args.epsilon}) <= {args.delta}, "
        f"need N >= {n_required} samples."
    )

    print(f"\n{'N':>10} | {'S_hat_N':>10} | {'|error|':>10} | {'Hoeffding bound':>16} | {'time (ms)':>10}")
    print("-" * 70)
    sample_sizes = sorted(set([10, 100, 1_000, n_required, max(n_required * 2, 10_000)]))
    for n in sample_sizes:
        rng_mc = np.random.default_rng(args.seed + n)
        t0 = time.perf_counter()
        est = monte_carlo_expected_utility(world_scores, space, n, rng_mc)
        t_mc = time.perf_counter() - t0
        err = abs(est - exact_value)
        bound = hoeffding_bound(n, a, b, args.epsilon)
        print(f"{n:>10} | {est:>10.4f} | {err:>10.4f} | {bound:>16.4g} | {t_mc*1000:>10.2f}")

    print(
        "\nAs N grows, exact enumeration cost is O(|Omega|) = O(domain_size^K) "
        "(exponential in K), while Monte Carlo cost is O(N), demonstrating "
        "the scalability argument of Section 6.2 / 7.4."
    )


if __name__ == "__main__":
    main()
