#!/usr/bin/env python3
"""
UPCL-Model on the structured synthetic testbed (Section 8, "UPCL-Model
(Learned Integration)" paragraph; full protocol in Appendix B,
"UPCL-Model testbed" paragraph; results reported in Table B.1.3).

The hash-based recommender used for RQ1-RQ3/RQ5-RQ6 (full_evaluation.py)
is unstructured by construction and carries no signal a trained f_phi
could generalize from, so UPCL-Model is instead evaluated on a second,
fully synthetic generator with a *learnable* low-rank structure:

  StructuredContextualRecommender: fixed per-user p_u in R^5, per-item
  q_i in R^5, and per-value context bias vectors b_mood, b_loc in R^5
  (all ~ N(0,1), seed 0), with
    r(u,i,omega) = 5 / (1 + exp(-(p_u + 0.6*b_mood[m] + 0.6*b_loc[l]) . q_i / sqrt(5)))
  for omega=(m,l).

UPCL-Model's encoder E_phi(Omega,P) returns the probability-weighted
mean of one-hot(mood) (+) one-hot(location) (a 6-dim vector z); its
recommender f_phi(u,i,z) = mu + b_u[u] + b_i[i] + p_u.q_i + w_i[i].z is a
small contextual matrix factorization (5 latent factors) fit by SGD (60
epochs, lr=0.02, L2=0.05) on 5 realized (single-world, degenerate-z)
interactions per (user,item) pair at lambda=0.3, drawn from the same
100-user/12-item population later used for evaluation (no held-out
users: all methods, including the oracle-based UPCL-Pre/Post below, see
the same population, so the only test-time randomness for every method
is the noise-perturbed context realization).

Non-contextual / Deterministic / UPCL-Pre / UPCL-Post are all evaluated
directly against the *true* structured r(u,i,omega) (oracle-based, no
fitting), exactly as in full_evaluation.py, so that only UPCL-Model's
row reflects what learning the context jointly with the recommender
actually costs or buys.

Run:
    python experiments/upcl_model_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.aggregation import expected_utility
from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.instantiations import upcl_model, upcl_post, upcl_pre
from upcl.noise import collapse_to_mode, inject_noise
from upcl.stats import cohens_d_paired, paired_ttest

N_USERS, N_ITEMS, N_FACTORS = 100, 12, 5
USERS = [f"u{i}" for i in range(N_USERS)]
ITEMS = [f"i{i}" for i in range(N_ITEMS)]
MOODS = ["happy", "neutral", "tired"]
LOCS = ["home", "work", "outdoor"]


class StructuredContextualRecommender:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {u: rng.normal(size=N_FACTORS) for u in USERS}
        self.q = {i: rng.normal(size=N_FACTORS) for i in ITEMS}
        self.b_mood = {m: rng.normal(size=N_FACTORS) for m in MOODS}
        self.b_loc = {l: rng.normal(size=N_FACTORS) for l in LOCS}

    def worldwise(self, user, item, world) -> float:
        wd = dict(world)
        m, l = wd["mood"], wd["location"]
        v = self.p[user] + 0.6 * self.b_mood[m] + 0.6 * self.b_loc[l]
        logit = float(np.dot(v, self.q[item]) / np.sqrt(N_FACTORS))
        return 5.0 / (1.0 + np.exp(-logit))


def base_context(user: str, rng: np.random.Generator) -> ProbabilisticContextInstance:
    mood_p = rng.dirichlet([5, 3, 2])
    mood = ContextualVariable("mood", {v: float(p) for v, p in zip(MOODS, mood_p)})
    loc_p = rng.dirichlet([6, 2, 2])
    location = ContextualVariable("location", {v: float(p) for v, p in zip(LOCS, loc_p)})
    return ProbabilisticContextInstance(categorical_variables=[mood, location])


def encode(worlds, probs) -> np.ndarray:
    """z = probability-weighted mean of one-hot(mood) (+) one-hot(location)."""
    z = np.zeros(len(MOODS) + len(LOCS))
    for w, p in zip(worlds, probs):
        wd = dict(w)
        z[MOODS.index(wd["mood"])] += p
        z[len(MOODS) + LOCS.index(wd["location"])] += p
    return z


def ndcg_at_k(ranked_items, relevance_for_user, k: int = 5) -> float:
    gains = [relevance_for_user.get(i, 0.0) for i in ranked_items[:k]]
    dcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))
    ideal = sorted(relevance_for_user.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def fit_f_phi(recommender, base_contexts, lam, seed=0, epochs=60, lr=0.02, l2=0.05, n_realized=5):
    """SGD fit of f_phi(u,i,z) = mu + b_u + b_i + p_u.q_i + w_i.z on
    `n_realized` single-world (degenerate-z) interactions per (user,item)
    pair drawn at noise intensity `lam`.
    """
    rng = np.random.default_rng(seed)
    z_dim = len(MOODS) + len(LOCS)
    mu = 0.0
    b_u = {u: 0.0 for u in USERS}
    b_i = {i: 0.0 for i in ITEMS}
    p_u = {u: rng.normal(scale=0.1, size=N_FACTORS) for u in USERS}
    q_i = {i: rng.normal(scale=0.1, size=N_FACTORS) for i in ITEMS}
    w_i = {i: rng.normal(scale=0.1, size=z_dim) for i in ITEMS}

    samples = []  # (user, item, y, z_degenerate)
    for user in USERS:
        instance = base_contexts[user]
        noisy = ProbabilisticContextInstance(
            categorical_variables=[inject_noise(v, lam, rng, mode="uniform_mix") for v in instance.categorical_variables]
        )
        space = phi(noisy)
        for item in ITEMS:
            for _ in range(n_realized):
                idx = rng.choice(len(space.worlds), p=space.probabilities)
                world = space.worlds[idx]
                y = recommender.worldwise(user, item, world)
                z_degenerate = encode([world], [1.0])  # single realized world -> one-hot z
                samples.append((user, item, y, z_degenerate))

    for _ in range(epochs):
        rng.shuffle(samples)
        for user, item, y, z in samples:
            pred = mu + b_u[user] + b_i[item] + float(np.dot(p_u[user], q_i[item])) + float(np.dot(w_i[item], z))
            err = pred - y
            mu -= lr * err
            b_u[user] -= lr * (err + l2 * b_u[user])
            b_i[item] -= lr * (err + l2 * b_i[item])
            grad_p = err * q_i[item] + l2 * p_u[user]
            grad_q = err * p_u[user] + l2 * q_i[item]
            p_u[user] -= lr * grad_p
            q_i[item] -= lr * grad_q
            w_i[item] -= lr * (err * z + l2 * w_i[item])

    def f_phi(user, item, z):
        return mu + b_u[user] + b_i[item] + float(np.dot(p_u[user], q_i[item])) + float(np.dot(w_i[item], z))

    return f_phi


def run_all_methods(recommender, base_contexts, relevance, f_phi, lam, rng):
    scores = {"non_ctx": [], "determ": [], "upcl_pre": [], "upcl_post": [], "upcl_model": []}
    for user in USERS:
        instance = base_contexts[user]
        rel = relevance[user]

        # Non-contextual: fixed canonical world (first category of each
        # variable), identical for every user and noise level.
        canonical_world = tuple((v.name, next(iter(v.distribution))) for v in instance.categorical_variables)
        nc_scores = {i: recommender.worldwise(user, i, canonical_world) for i in ITEMS}
        ranked = sorted(nc_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["non_ctx"].append(ndcg_at_k([i for i, _ in ranked], rel))

        noisy = ProbabilisticContextInstance(
            categorical_variables=[inject_noise(v, lam, rng, mode="uniform_mix") for v in instance.categorical_variables]
        )
        collapsed = ProbabilisticContextInstance(
            categorical_variables=[collapse_to_mode(v) for v in noisy.categorical_variables]
        )
        det_space = phi(collapsed)
        mode_world = det_space.worlds[0]
        det_scores = {i: recommender.worldwise(user, i, mode_world) for i in ITEMS}
        ranked = sorted(det_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["determ"].append(ndcg_at_k([i for i, _ in ranked], rel))

        space = phi(noisy)
        _, pre_scores = upcl_pre(user, ITEMS, space, recommender.worldwise, policy="eu")
        ranked = sorted(pre_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["upcl_pre"].append(ndcg_at_k([i for i, _ in ranked], rel))

        def exact_adjustment(r0, u, i, w, _rec=recommender):
            del r0
            return _rec.worldwise(u, i, w)

        def base_recommender(u, i, _rec=recommender, _w=mode_world):
            return _rec.worldwise(u, i, _w)

        _, post_scores = upcl_post(user, ITEMS, space, base_recommender, exact_adjustment, policy="eu")
        ranked = sorted(post_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["upcl_post"].append(ndcg_at_k([i for i, _ in ranked], rel))

        _, model_scores = upcl_model(user, ITEMS, space, encode, f_phi)
        ranked = sorted(model_scores.items(), key=lambda kv: kv[1], reverse=True)
        scores["upcl_model"].append(ndcg_at_k([i for i, _ in ranked], rel))

    return scores


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "upcl_model_evaluation_log.txt"
    lines = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    recommender = StructuredContextualRecommender(seed=0)
    rng0 = np.random.default_rng(0)
    base_contexts = {u: base_context(u, rng0) for u in USERS}

    relevance = {}
    for user, instance in base_contexts.items():
        space = phi(instance)
        relevance[user] = {}
        for item in ITEMS:
            ws = [recommender.worldwise(user, item, w) for w in space.worlds]
            relevance[user][item] = expected_utility(ws, space.probabilities)

    log("Fitting f_phi (SGD, 60 epochs) on realized interactions at lambda=0.3 ...")
    f_phi = fit_f_phi(recommender, base_contexts, lam=0.3, seed=0)

    log("\n=== UPCL-Model: full noise sweep, NDCG@5 (structured testbed, 100 agents) ===")
    levels = [0.0, 0.25, 0.3, 0.5, 0.75, 1.0]
    sweep = {}
    for lam in levels:
        rng = np.random.default_rng(200)
        scores = run_all_methods(recommender, base_contexts, relevance, f_phi, lam, rng)
        sweep[lam] = scores
        log(f"lambda={lam:.2f}: non_ctx={np.mean(scores['non_ctx']):.4f}, "
            f"determ={np.mean(scores['determ']):.4f}, upcl_pre={np.mean(scores['upcl_pre']):.4f}, "
            f"upcl_post={np.mean(scores['upcl_post']):.4f}, upcl_model={np.mean(scores['upcl_model']):.4f}")

    log("\n=== At lambda=0.3: UPCL-Model vs. baselines ===")
    s = sweep[0.3]
    for name in ("non_ctx", "determ", "upcl_pre"):
        t = paired_ttest(s["upcl_model"], s[name])
        d = cohens_d_paired(s["upcl_model"], s[name])
        log(f"UPCL-Model vs {name:>10}: d={d:+.3f}, p={t.p_value:.3g}")

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log(f"\nLog written to {out_path}")


if __name__ == "__main__":
    main()
