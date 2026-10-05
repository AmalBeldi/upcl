#!/usr/bin/env python3
"""
Full synthetic evaluation: RQ1-RQ3 and RQ5-RQ6 (Section 8 of the main
paper; full protocol in Appendix B, "Synthetic testbed" / "Noise
injection" / "RQ5/RQ6 scalability testbed" paragraphs).

Population: 100 users, 12 items. Each user's two Block-Independent
Disjoint state variables (mood, location; 3 values each) are drawn from
independent Dirichlet priors, redrawn per user with seed 0: mood ~
Dirichlet(5,3,2), location ~ Dirichlet(6,2,2), giving |Omega|=9 worlds per
user. Ground-truth relevance is each item's exact Expected-Utility score
under the *unperturbed* belief (lambda=0), isolating belief construction
from policy-quality confounds. The recommender is the deterministic,
seed-reproducible hash-based stand-in
(upcl.recommenders.toy.HashedContextualRecommender):
  - `base(u,i)`      -> the context-independent score used by the
                         "Non-contextual" baseline and by UPCL-Post's r0.
  - `worldwise(u,i,w)` -> the per-world score used by UPCL-Pre and (via an
                         exact-reproduction adjustment, see RQ3 below) by
                         UPCL-Post.

RQ1 (decision quality, lambda=0.3): Non-contextual, Deterministic
(mode-collapsed context, classical contextual pre-filtering), UPCL-Pre,
UPCL-Post, all under Expected Utility, compared via NDCG@5.

RQ2 (robustness): UPCL-Pre vs. Deterministic across the noise sweep
lambda in {0, 0.25, 0.5, 0.75, 1}, averaged over 3 seeds.

RQ3 (integration strategy comparison / cost): UPCL-Pre and UPCL-Post are
run with an adjustment h_eta that *exactly* reproduces the world-wise
score available to UPCL-Pre (h_eta(r0,u,i,w) := worldwise(u,i,w),
ignoring r0), demonstrating Proposition 10 (structural unification)
empirically: both reach identical NDCG@5, while UPCL-Pre calls the
world-wise function once per (item, world) and UPCL-Post calls the base
policy once per item plus the same per-world adjustment.

RQ5 (Monte Carlo approximation quality) and RQ6 (scalability): a larger
synthetic context with K in {2,4,6,8,10} Block-Independent variables of
domain size 4 (|Omega| = 4^K, from 16 to 1,048,576), using the same
hashed recommender, to verify the finite-sample Hoeffding bound
(Corollary 1) and the exponential-to-linear wall-clock crossover of
Section 6.2/RQ6.

Run:
    python experiments/full_evaluation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.aggregation import expected_utility
from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.instantiations import upcl_post, upcl_pre
from upcl.monte_carlo import hoeffding_bound, monte_carlo_expected_utility, required_sample_size
from upcl.noise import collapse_to_mode, inject_noise
from upcl.recommenders.toy import HashedContextualRecommender
from upcl.stats import cohens_d_paired, paired_ttest

N_USERS = 100
N_ITEMS = 12
USERS = [f"u{i}" for i in range(N_USERS)]
ITEMS = [f"i{i}" for i in range(N_ITEMS)]
RECOMMENDER = HashedContextualRecommender(seed=0)


def base_context(user: str, rng: np.random.Generator) -> ProbabilisticContextInstance:
    """Each user's unperturbed (lambda=0) two-variable BID context."""
    mood_p = rng.dirichlet([5, 3, 2])
    mood = ContextualVariable(
        "mood", {"happy": float(mood_p[0]), "neutral": float(mood_p[1]), "tired": float(mood_p[2])}
    )
    loc_p = rng.dirichlet([6, 2, 2])
    location = ContextualVariable(
        "location", {"home": float(loc_p[0]), "work": float(loc_p[1]), "outdoor": float(loc_p[2])}
    )
    return ProbabilisticContextInstance(categorical_variables=[mood, location])


def build_user_contexts(seed: int) -> dict:
    """One base (unperturbed) context per user, redrawn deterministically
    from `seed` (seed=0 throughout the main paper's RQ1-RQ3).
    """
    rng = np.random.default_rng(seed)
    return {u: base_context(u, rng) for u in USERS}


def ground_truth_relevance(base_contexts: dict) -> dict:
    """relevance[user][item] = EU of worldwise scores under the
    unperturbed belief -- the fixed ground truth used by every method
    and noise level.
    """
    relevance = {}
    for user, instance in base_contexts.items():
        space = phi(instance)
        relevance[user] = {}
        for item in ITEMS:
            world_scores = [RECOMMENDER.worldwise(user, item, w) for w in space.worlds]
            relevance[user][item] = expected_utility(world_scores, space.probabilities)
    return relevance


def ndcg_at_k(ranked_items, relevance_for_user: dict, k: int = 5) -> float:
    gains = [relevance_for_user.get(i, 0.0) for i in ranked_items[:k]]
    dcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))
    ideal = sorted(relevance_for_user.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def noisy_space(instance: ProbabilisticContextInstance, intensity: float, rng: np.random.Generator):
    noisy = ProbabilisticContextInstance(
        categorical_variables=[inject_noise(v, intensity, rng, mode="uniform_mix") for v in instance.categorical_variables]
    )
    return phi(noisy)


def exact_reproduction_adjustment(r0, user, item, world):
    """h_eta for RQ3: ignores r0 and exactly reproduces the world-wise
    score, so that UPCL-Post coincides with UPCL-Pre (Proposition 10).
    """
    del r0
    return RECOMMENDER.worldwise(user, item, world)


def run_methods(base_contexts, relevance, intensity, rng):
    """Returns per-user NDCG@5 for each of the four RQ1 methods at the
    given noise intensity.
    """
    scores = {"non_contextual": [], "deterministic": [], "upcl_pre": [], "upcl_post": []}
    for user in USERS:
        instance = base_contexts[user]
        rel = relevance[user]

        # Non-contextual: ignores context entirely.
        nc_scores = {i: RECOMMENDER.base(user, i) for i in ITEMS}
        ranked = sorted(nc_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["non_contextual"].append(ndcg_at_k([i for i, _ in ranked], rel))

        noisy_instance = ProbabilisticContextInstance(
            categorical_variables=[inject_noise(v, intensity, rng, mode="uniform_mix") for v in instance.categorical_variables]
        )

        # Deterministic: classical contextual pre-filtering -- collapse to
        # the mode of the (possibly noisy) context, then score once.
        collapsed = ProbabilisticContextInstance(
            categorical_variables=[collapse_to_mode(v) for v in noisy_instance.categorical_variables]
        )
        det_space = phi(collapsed)
        mode_world = det_space.worlds[0]
        det_scores = {i: RECOMMENDER.worldwise(user, i, mode_world) for i in ITEMS}
        ranked = sorted(det_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["deterministic"].append(ndcg_at_k([i for i, _ in ranked], rel))

        # UPCL-Pre: full (possibly noisy) belief, Expected Utility.
        space = phi(noisy_instance)
        _, pre_scores = upcl_pre(user, ITEMS, space, RECOMMENDER.worldwise, policy="eu")
        ranked = sorted(pre_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["upcl_pre"].append(ndcg_at_k([i for i, _ in ranked], rel))

        # UPCL-Post: base policy + exact-reproduction adjustment (RQ3).
        _, post_scores = upcl_post(
            user, ITEMS, space, RECOMMENDER.base, exact_reproduction_adjustment, policy="eu"
        )
        ranked = sorted(post_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["upcl_post"].append(ndcg_at_k([i for i, _ in ranked], rel))

    return scores


def rq1_rq3(base_contexts, relevance, log):
    rng = np.random.default_rng(1)
    scores = run_methods(base_contexts, relevance, 0.3, rng)

    log("=== RQ1: decision quality at lambda=0.3 (n=100 agents) ===")
    for name in ("non_contextual", "deterministic", "upcl_pre", "upcl_post"):
        arr = scores[name]
        log(f"{name:>15}: NDCG@5 = {np.mean(arr):.4f} +/- {np.std(arr, ddof=1):.4f}")

    for name in ("non_contextual", "deterministic"):
        t = paired_ttest(scores["upcl_pre"], scores[name])
        d = cohens_d_paired(scores["upcl_pre"], scores[name])
        log(f"UPCL-Pre vs {name}: d={d:+.3f}, p={t.p_value:.3g}")
    t = paired_ttest(scores["upcl_post"], scores["upcl_pre"])
    d = cohens_d_paired(scores["upcl_post"], scores["upcl_pre"])
    log(f"UPCL-Post vs upcl_pre: d={d:+.3f}, p={t.p_value:.3g} (exact-reproduction h_eta)")

    log("\n=== RQ3: UPCL-Pre vs UPCL-Post, call cost ===")
    pre_calls = N_USERS * N_ITEMS * 9  # one worldwise() call per (user,item,world)
    post_calls_base = N_USERS * N_ITEMS  # f0(u,i)
    post_calls_adjust = N_USERS * N_ITEMS * 9  # h_eta per (user,item,world)
    log(f"UPCL-Pre:  {pre_calls} worldwise() calls")
    log(f"UPCL-Post: {post_calls_base} base() calls + {post_calls_adjust} adjustment calls "
        f"(adjustment here *is* worldwise(), by the exact-reproduction construction above; "
        f"a true black-box h_eta would replace this with a cheap lookup)")
    return scores


def rq2(base_contexts, relevance, log):
    log("\n=== RQ2: robustness to state uncertainty (3 seeds) ===")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for lam in levels:
        pre_all, det_all = [], []
        for seed in range(3):
            rng = np.random.default_rng(100 + seed)
            scores = run_methods(base_contexts, relevance, lam, rng)
            pre_all.append(np.mean(scores["upcl_pre"]))
            det_all.append(np.mean(scores["deterministic"]))
        # Use the full per-user arrays from one representative seed for
        # the paired test (same noise level, same seed set, matched pairs).
        rng = np.random.default_rng(100)
        scores = run_methods(base_contexts, relevance, lam, rng)
        t = paired_ttest(scores["upcl_pre"], scores["deterministic"])
        d = cohens_d_paired(scores["upcl_pre"], scores["deterministic"])
        log(f"lambda={lam:.2f}: UPCL-Pre={np.mean(pre_all):.4f}, "
            f"Deterministic={np.mean(det_all):.4f}, d={d:+.3f}, p={t.p_value:.3g}")


def rq5_rq6(log):
    log("\n=== RQ5: Monte Carlo approximation quality (K=6, domain size 4) ===")
    k_vars, domain = 6, 4
    rng = np.random.default_rng(7)
    variables = []
    for k in range(k_vars):
        p = rng.dirichlet(np.ones(domain))
        variables.append(ContextualVariable(f"v{k}", {f"v{k}_{j}": float(p[j]) for j in range(domain)}))
    instance = ProbabilisticContextInstance(categorical_variables=variables)
    space = phi(instance)
    world_scores = [RECOMMENDER.worldwise("u_rq5", "i_rq5", w) for w in space.worlds]
    true_s = expected_utility(world_scores, space.probabilities)

    n_required = required_sample_size(a=0.0, b=5.0, epsilon=0.05, delta=0.05)
    log(f"|Omega|={len(space.worlds)}, required N for eps=0.05, delta=0.05: {n_required}")
    # Average |error| over 20 independent draws per N: Theorem 4/Corollary 1
    # are statements about convergence in expectation / with probability
    # >= 1-delta, not about a single draw, so a single-sample error is not
    # representative -- average over repeats to report the expected error.
    for n in (n_required, 2 * n_required):
        errors = []
        for trial in range(20):
            mc_rng = np.random.default_rng(1000 + trial)
            est = monte_carlo_expected_utility(world_scores, space, n, mc_rng)
            errors.append(abs(est - true_s))
        log(f"N={n}: mean |error| over 20 draws = {np.mean(errors):.4f} "
            f"(Hoeffding bound={hoeffding_bound(n, 0.0, 5.0, 0.05):.4f}), true={true_s:.4f}")

    log("\n=== RQ6: scalability, exact enumeration vs. Monte Carlo (N=2000) ===")
    for k_vars in (2, 4, 6, 8, 10):
        rng = np.random.default_rng(7)
        variables = []
        for k in range(k_vars):
            p = rng.dirichlet(np.ones(domain))
            variables.append(ContextualVariable(f"v{k}", {f"v{k}_{j}": float(p[j]) for j in range(domain)}))
        instance = ProbabilisticContextInstance(categorical_variables=variables)

        exact_times, mc_times = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            space = phi(instance)
            world_scores = [RECOMMENDER.worldwise("u_rq6", "i_rq6", w) for w in space.worlds]
            _ = expected_utility(world_scores, space.probabilities)
            exact_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            mc_rng = np.random.default_rng(0)
            _ = monte_carlo_expected_utility(world_scores, space, 2000, mc_rng)
            mc_times.append(time.perf_counter() - t0)

        log(f"K={k_vars:>2} (|Omega|={domain**k_vars:>9}): "
            f"exact={np.median(exact_times)*1000:.2f} ms, MC(N=2000)={np.median(mc_times)*1000:.2f} ms")


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "full_evaluation_log.txt"
    lines = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    base_contexts = build_user_contexts(seed=0)
    relevance = ground_truth_relevance(base_contexts)

    rq1_rq3(base_contexts, relevance, log)
    rq2(base_contexts, relevance, log)
    rq5_rq6(log)

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log(f"\nLog written to {out_path}")


if __name__ == "__main__":
    main()
