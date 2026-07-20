#!/usr/bin/env python3
"""
Synthetic robustness benchmark (Sections 7.2-7.3).

Since no real dataset ships with this anonymous repository, this script
demonstrates the *entire* experimental protocol structure -- controlled
contextual noise injection, deterministic vs. uncertainty-aware
comparison, ranking-quality evaluation, and paired significance testing
-- on a fully synthetic population of users/items with a reproducible
hashed recommender (upcl.recommenders.toy.HashedContextualRecommender).

This is a structural stand-in: swap `HashedContextualRecommender` for a
trained recommender and `build_synthetic_context` for a real per-user
`ProbabilisticContextInstance` (e.g. from upcl.datasets.loaders) to turn
this into the real-dataset experiment described in Section 7.1.

Run:
    python experiments/synthetic_benchmark.py --n-users 50 --n-items 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.instantiations import upcl_pre
from upcl.noise import collapse_to_mode, inject_noise
from upcl.recommenders.toy import HashedContextualRecommender
from upcl.stats import paired_ttest, wilcoxon_signed_rank


def build_synthetic_context(user: str, rng: np.random.Generator, noise_intensity: float):
    """A per-user two-variable (mood, location) probabilistic context,
    perturbed by `noise_intensity` in [0, 1] to simulate degrading sensor /
    inference quality (Section 7.2).
    """
    base_mood = rng.dirichlet([5, 3, 2])  # skewed base distribution
    mood = ContextualVariable(
        "mood", {"happy": float(base_mood[0]), "neutral": float(base_mood[1]), "tired": float(base_mood[2])}
    )
    base_loc = rng.dirichlet([6, 2, 2])
    location = ContextualVariable(
        "location", {"home": float(base_loc[0]), "work": float(base_loc[1]), "outdoor": float(base_loc[2])}
    )

    mood = inject_noise(mood, noise_intensity, rng, mode="uniform_mix")
    location = inject_noise(location, noise_intensity, rng, mode="uniform_mix")

    return ProbabilisticContextInstance(categorical_variables=[mood, location])


def ndcg_at_k(ranked_items, relevance: dict, k: int = 5) -> float:
    """Simple NDCG@k using the hashed recommender's ground-truth relevance
    (here: the exact-expectation UPCL-Pre score under the *unperturbed*
    context, used as a proxy for "true" relevance) as gain.
    """
    gains = [relevance.get(i, 0.0) for i in ranked_items[:k]]
    dcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))
    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(ideal_gains))
    return float(dcg / idcg) if idcg > 0 else 0.0


def run_condition(
    users, items, recommender, noise_intensity, rng, use_upcl: bool
):
    """Returns per-user NDCG@5 either for UPCL-Pre (full distribution) or
    for the deterministic baseline (mode-collapsed context, i.e. classical
    contextual pre-filtering as critiqued in Section 3.1).
    """
    scores_per_user = []
    for user in users:
        instance = build_synthetic_context(user, rng, noise_intensity)

        # "Ground truth" relevance uses the *unperturbed* (noise_intensity=0)
        # expected utility, standing in for the true, uncorrupted context.
        clean_instance = build_synthetic_context(user, rng, 0.0)
        clean_space = phi(clean_instance)
        relevance = {}
        for item in items:
            world_scores = [recommender.worldwise(user, item, w) for w in clean_space.worlds]
            relevance[item] = float(np.dot(world_scores, clean_space.probabilities))

        if use_upcl:
            space = phi(instance)
        else:
            collapsed = ProbabilisticContextInstance(
                categorical_variables=[collapse_to_mode(v) for v in instance.categorical_variables]
            )
            space = phi(collapsed)

        _, scores = upcl_pre(user, items, space, recommender.worldwise, policy="eu")
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked_items = [i for i, _ in ranked]
        scores_per_user.append(ndcg_at_k(ranked_items, relevance, k=min(5, len(items))))

    return scores_per_user


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=30)
    parser.add_argument("--n-items", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    users = [f"u{i}" for i in range(args.n_users)]
    items = [f"i{i}" for i in range(args.n_items)]
    recommender = HashedContextualRecommender(seed=args.seed)

    print(f"{'noise':>6} | {'UPCL-Pre NDCG@5':>16} | {'Deterministic NDCG@5':>20} | significance")
    print("-" * 75)

    for noise in args.noise_levels:
        upcl_scores = run_condition(users, items, recommender, noise, rng, use_upcl=True)
        det_scores = run_condition(users, items, recommender, noise, rng, use_upcl=False)

        t_res = paired_ttest(upcl_scores, det_scores)
        w_res = wilcoxon_signed_rank(upcl_scores, det_scores)

        print(
            f"{noise:>6.2f} | {np.mean(upcl_scores):>16.4f} | {np.mean(det_scores):>20.4f} | "
            f"t p={t_res.p_value:.3g}, wilcoxon p={w_res.p_value:.3g}"
        )

    print(
        "\nNote: this is a fully synthetic, seed-reproducible stand-in for the "
        "real-dataset robustness protocol of Section 7.2. Swap in a real "
        "recommender and upcl.datasets.loaders.load_csv_dataset(...) to run "
        "this against LDOS-CoMoDa / Frappe / other CARS benchmarks."
    )


if __name__ == "__main__":
    main()
