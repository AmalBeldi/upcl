#!/usr/bin/env python3
"""
Multi-seed UPCL-Post vs. Deterministic evaluation on a prepared real-world
CARS dataset -- the missing counterpart to Appendix B's
"real_dataset_multiseed.py", now built directly on top of this
repository's actual upcl/ library (context.phi, noise.inject_noise,
noise.collapse_to_mode, instantiations.upcl_post, decision.psi_ranked,
stats.paired_ttest) rather than a standalone reimplementation, so the
formal claims (Phi, Gamma_theta, Psi) and the empirical pipeline are
provably the same code path.

Protocol (matches Appendix B's "DePaulMovie real-dataset setup" /
"Statistical rigor" paragraphs):
  - f0: matrix-factorization base recommender (n_factors, epochs, lr, l2),
    trained once per seed on that seed's 80% training split.
  - h_eta: per-(item, world) mean residual of (rating - f0 prediction) on
    the training split, shrunk toward 0 by n/(n+shrink_k).
  - For each test interaction, the TRUE observed context is a point mass;
    noise.inject_noise(..., mode="uniform_mix") mixes it toward uniform at
    intensity lambda to simulate contextual state uncertainty, exactly as
    described for the noise sweep (Section 7.2 / Appendix B).
  - UPCL-Post integrates over the resulting belief via Gamma_EU
    (instantiations.upcl_post, policy="eu"); the deterministic baseline
    collapses the same noisy belief back to its mode
    (noise.collapse_to_mode) before scoring -- both consume the identical
    f0/h_eta, so the comparison isolates the effect of belief-aware
    aggregation.
  - Evaluation metric: MRR (mean reciprocal rank of the true item among
    all candidates). 20 independent 80/20 seeds; paired Cohen's d and
    Holm-Bonferroni across noise levels vs. lambda=0.

USAGE
-----
    python experiments/real_dataset_multiseed.py \
        --data data/ldos_comoda_prepared.csv \
        --context-cols mood \
        --seeds 20 \
        --out outputs/ldos_comoda_multiseed_log.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.context import ContextualVariable, ProbabilisticContextInstance, phi
from upcl.instantiations import upcl_post
from upcl.noise import collapse_to_mode, inject_noise
from upcl.stats import paired_ttest


# ---------------------------------------------------------------------
# f0 (matrix factorization) and h_eta (shrunk per-(item, world) residual)
# -- experiment-specific models plugged into upcl_post's base_recommender
# / adjustment hooks, exactly as instantiations.py's docstring intends.
# ---------------------------------------------------------------------

def train_mf(train_df, n_users, n_items, n_factors=10, epochs=30,
             lr=0.01, l2=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mu = train_df["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    P = 0.1 * rng.standard_normal((n_users, n_factors))
    Q = 0.1 * rng.standard_normal((n_items, n_factors))

    rows = train_df[["u_idx", "i_idx", "rating"]].to_numpy()
    for _ in range(epochs):
        rng.shuffle(rows)
        for u, i, r in rows:
            u, i = int(u), int(i)
            pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
            err = r - pred
            bu[u] += lr * (err - l2 * bu[u])
            bi[i] += lr * (err - l2 * bi[i])
            p_u = P[u].copy()
            P[u] += lr * (err * Q[i] - l2 * P[u])
            Q[i] += lr * (err * p_u - l2 * Q[i])

    def predict(u, i):
        return float(mu + bu[int(u)] + bi[int(i)] + P[int(u)] @ Q[int(i)])

    return predict


def build_residual_table(train_df, context_cols, predict_f0, shrink_k=5,
                          granularity="item"):
    """Per-(key, world) mean residual, shrunk toward 0 by n/(n+shrink_k),
    where `key` is either the item_id (DePaulMovie protocol, dense enough
    catalog) or the user_id (used for sparse-catalog datasets like
    LDOS-CoMoDa, where per-user density is far higher than per-item
    density -- see chat discussion). `world` is the same tuple-of-
    (name, value) key upcl.context.World uses.
    """
    key_col = "item_id" if granularity == "item" else "u_idx"
    resid = train_df.apply(
        lambda row: row["rating"] - predict_f0(row["u_idx"], row["i_idx"]), axis=1
    )
    tmp = train_df[[key_col] + context_cols].copy()
    tmp["resid"] = resid.values
    tmp["world_key"] = list(zip(*[tmp[c] for c in context_cols]))

    grp = tmp.groupby([key_col, "world_key"])["resid"]
    means = grp.mean()
    counts = grp.count()
    shrunk = means * (counts / (counts + shrink_k))
    return shrunk.to_dict()  # {(key, (v1, v2, ...)): shrunk_residual}


def make_adjustment(residual_table, context_cols, granularity="item"):
    def adjustment(r0, user, item, world):
        # `world` is ((col_1, val_1), (col_2, val_2), ...) in context_cols order
        # (upcl.context.phi preserves the ContextualVariable insertion order).
        world_key = tuple(v for _, v in world)
        key = item if granularity == "item" else user
        return r0 + residual_table.get((key, world_key), 0.0)

    return adjustment


def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def holm_bonferroni(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    sig = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            sig[idx] = True
        else:
            break
    return sig


def run(data_path, context_cols, n_seeds=20,
        lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), noise_mode="uniform_mix",
        policy="eu", granularity="item", n_factors=10, shrink_k=5):
    df = pd.read_csv(data_path)

    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    u_index = {u: k for k, u in enumerate(users)}
    i_index = {i: k for k, i in enumerate(items)}
    df["u_idx"] = df["user_id"].map(u_index)
    df["i_idx"] = df["item_id"].map(i_index)
    n_users, n_items = len(users), len(items)

    # Full domain of each context variable, needed so inject_noise can mix
    # a point mass toward a *real* uniform distribution over all observed
    # values, not just the single value present in one row.
    domains = {c: sorted(df[c].unique().tolist()) for c in context_cols}

    per_lambda = {lam: {"upcl_post": [], "deterministic": []} for lam in lambdas}

    for seed in range(n_seeds):
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split = int(0.8 * len(shuffled))
        train_df, test_df = shuffled.iloc[:split], shuffled.iloc[split:]

        predict_f0_idx = train_mf(train_df, n_users, n_items, n_factors=n_factors, seed=seed)
        residual_table = build_residual_table(
            train_df, context_cols, predict_f0_idx, shrink_k=shrink_k, granularity=granularity
        )

        def base_recommender(user, item, _predict=predict_f0_idx, _i_index=i_index):
            return _predict(user, _i_index[item])

        adjustment = make_adjustment(residual_table, context_cols, granularity=granularity)

        rng = np.random.default_rng(1000 + seed)

        for lam in lambdas:
            upcl_ranks, det_ranks = [], []
            for row in test_df.itertuples():
                true_assignment = {c: getattr(row, c) for c in context_cols}
                variables = [
                    ContextualVariable(
                        name=c,
                        distribution={v: (1.0 if v == true_assignment[c] else 0.0)
                                      for v in domains[c]},
                    )
                    for c in context_cols
                ]
                noisy_variables = [
                    inject_noise(v, lam, rng, mode=noise_mode) for v in variables
                ]

                noisy_instance = ProbabilisticContextInstance(categorical_variables=noisy_variables)
                noisy_space = phi(noisy_instance)

                det_variables = [collapse_to_mode(v) for v in noisy_variables]
                det_instance = ProbabilisticContextInstance(categorical_variables=det_variables)
                det_space = phi(det_instance)

                u = row.u_idx
                _, upcl_scores = upcl_post(
                    u, items, noisy_space, base_recommender, adjustment, policy=policy
                )
                _, det_scores = upcl_post(
                    u, items, det_space, base_recommender, adjustment, policy=policy
                )

                target = row.item_id
                ranked_upcl = [it for it, _ in sorted(upcl_scores.items(), key=lambda kv: kv[1], reverse=True)]
                ranked_det = [it for it, _ in sorted(det_scores.items(), key=lambda kv: kv[1], reverse=True)]
                upcl_ranks.append(1.0 / (ranked_upcl.index(target) + 1))
                det_ranks.append(1.0 / (ranked_det.index(target) + 1))

            per_lambda[lam]["upcl_post"].append(float(np.mean(upcl_ranks)))
            per_lambda[lam]["deterministic"].append(float(np.mean(det_ranks)))

        print(f"seed {seed} done (train={len(train_df)}, test={len(test_df)}, "
              f"items={n_items}, users={n_users})")

    results = {}
    pvals = []
    lam_order = list(lambdas)
    for lam in lam_order:
        post = per_lambda[lam]["upcl_post"]
        det = per_lambda[lam]["deterministic"]
        p = 1.0 if lam == 0.0 else paired_ttest(post, det).p_value
        pvals.append(p)
        results[lam] = {
            "upcl_post_mean": float(np.mean(post)),
            "upcl_post_std": float(np.std(post, ddof=1)),
            "deterministic_mean": float(np.mean(det)),
            "deterministic_std": float(np.std(det, ddof=1)),
            "cohens_d": cohens_d_paired(post, det),
            "p_value": float(p),
        }

    for lam, sig in zip(lam_order, holm_bonferroni(pvals)):
        results[lam]["holm_significant"] = bool(sig)

    meta = {"n_users": n_users, "n_items": n_items, "n_rows": len(df),
            "context_cols": context_cols, "policy": policy,
            "granularity": granularity, "n_factors": n_factors, "shrink_k": shrink_k}
    return results, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--context-cols", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--policy", default="eu")
    parser.add_argument("--granularity", choices=["item", "user"], default="item")
    parser.add_argument("--n-factors", type=int, default=10)
    parser.add_argument("--shrink-k", type=float, default=5)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out", default="outputs/real_dataset_multiseed_log.json")
    args = parser.parse_args()

    results, meta = run(args.data, args.context_cols, n_seeds=args.seeds,
                         lambdas=tuple(args.lambdas),
                         policy=args.policy, granularity=args.granularity,
                         n_factors=args.n_factors, shrink_k=args.shrink_k)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"\nWrote {args.out}")
    print(meta)
    for lam, r in results.items():
        print(
            f"lambda={lam}: UPCL-Post={r['upcl_post_mean']:.4f}+-{r['upcl_post_std']:.4f}, "
            f"Det={r['deterministic_mean']:.4f}+-{r['deterministic_std']:.4f}, "
            f"d={r['cohens_d']:+.2f}, p={r['p_value']:.4g}, Holm-sig={r['holm_significant']}"
        )


if __name__ == "__main__":
    main()
