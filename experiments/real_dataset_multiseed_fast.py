#!/usr/bin/env python3
"""
Fast, vectorized re-expression of real_dataset_multiseed.py, for datasets
with a large item catalog (LDOS-CoMoDa: 1232 items) where the generic
per-item-per-world Python loop through upcl.instantiations.upcl_post is
too slow to iterate on quickly (~40+ minutes/run).

This is NOT a different method -- it is a closed-form simplification of
the exact same computation (upcl.noise.inject_noise(mode="uniform_mix"),
upcl.aggregation.expected_utility, upcl.noise.collapse_to_mode), valid
because with a SINGLE context variable of domain size k, uniform_mix
gives a closed form:
    p(true value)  = (1 - lambda) + lambda / k
    p(other value) = lambda / k          for each of the k-1 others
and collapse_to_mode always returns the true value for lambda < 1 (since
(1-lambda) + lambda/k > lambda/k for lambda<1), and returns the smallest
domain value (ties broken by iteration/sort order, matching
noise.collapse_to_mode's max()-based tie-break) at lambda == 1 exactly.

This closed form was validated against the real (slow) pipeline on
DePaulMovie: same qualitative behavior (deterministic baseline flat until
lambda=1, then a small jump) -- see chat. Use this only for single-block
(one context column) sweeps; for multi-variable context, fall back to
the generic real_dataset_multiseed.py (slower but exact for the general
case without needing the closed form).

USAGE
-----
    python experiments/real_dataset_multiseed_fast.py \
        --data data/ldos_comoda_full.csv --context-col mood \
        --granularity user --seeds 20 \
        --out outputs/ldos_comoda_user_granularity_log.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


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

    f0_matrix = mu + bu[:, None] + bi[None, :] + P @ Q.T  # (n_users, n_items)
    return f0_matrix


def build_residual_matrix(train_df, key_col, n_keys, n_worlds, resid_values, shrink_k=5):
    tmp = pd.DataFrame({
        "key": train_df[key_col].to_numpy(),
        "world": train_df["w_idx"].to_numpy(),
        "resid": resid_values,
    })
    grp = tmp.groupby(["key", "world"])["resid"]
    means, counts = grp.mean(), grp.count()
    shrunk = means * (counts / (counts + shrink_k))
    mat = np.zeros((n_keys, n_worlds))
    for (k, w), v in shrunk.items():
        mat[int(k), int(w)] = v
    return mat


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


def run(data_path, context_col, granularity="item", n_seeds=20,
        lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), eval_positive_only=False,
        det_mode="collapse"):
    """det_mode:
      "collapse" (default) -- deterministic baseline uses
        noise.collapse_to_mode: argmax of the noisy belief, which always
        recovers the true value except at lambda=1 exactly (ties).
      "sample" -- alternate hypothesis being tested: deterministic baseline
        commits to a single value drawn AT RANDOM from the noisy belief
        (true value w.p. (1-lambda)+lambda/k, each other value w.p.
        lambda/k), i.e. it can genuinely be wrong with probability
        proportional to lambda, unlike "collapse".
    """
    df = pd.read_csv(data_path)

    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    worlds = sorted(df[context_col].unique())
    u_index = {u: k for k, u in enumerate(users)}
    i_index = {i: k for k, i in enumerate(items)}
    w_index = {w: k for k, w in enumerate(worlds)}
    df["u_idx"] = df["user_id"].map(u_index)
    df["i_idx"] = df["item_id"].map(i_index)
    df["w_idx"] = df[context_col].map(w_index)
    n_users, n_items, n_worlds = len(users), len(items), len(worlds)
    n_keys = n_items if granularity == "item" else n_users
    key_col = "i_idx" if granularity == "item" else "u_idx"

    per_lambda = {lam: {"upcl_post": [], "deterministic": []} for lam in lambdas}

    for seed in range(n_seeds):
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split = int(0.8 * len(shuffled))
        train_df, test_df = shuffled.iloc[:split], shuffled.iloc[split:]
        if eval_positive_only:
            # Implicit-feedback datasets (e.g. Frappe): train f0/h_eta on
            # both positive and negative (rating=0) rows, but only rank
            # true POSITIVE interactions for MRR -- ranking a synthetic
            # negative sample "correctly" is not a meaningful target.
            test_df = test_df[test_df["rating"] > 0].reset_index(drop=True)

        f0_matrix = train_mf(train_df, n_users, n_items, seed=seed)  # (users, items)
        resid_values = (
            train_df["rating"].to_numpy()
            - f0_matrix[train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy()]
        )
        residual_matrix = build_residual_matrix(
            train_df, key_col, n_keys, n_worlds, resid_values
        )  # (n_keys, n_worlds)
        residual_sum = residual_matrix.sum(axis=1)  # sum over all worlds, per key

        test_u = test_df["u_idx"].to_numpy()
        test_i = test_df["i_idx"].to_numpy()
        test_w = test_df["w_idx"].to_numpy()
        det_rng = np.random.default_rng(2000 + seed)

        for lam in lambdas:
            upcl_ranks, det_ranks = [], []
            for row_idx in range(len(test_df)):
                u, i_true, t = test_u[row_idx], test_i[row_idx], test_w[row_idx]
                f0_row = f0_matrix[u]  # (n_items,)

                if det_mode == "sample":
                    if det_rng.random() < lam:
                        # drew a "wrong" value uniformly among the k-1 others
                        others = [w for w in range(n_worlds) if w != t]
                        w_det = det_rng.choice(others) if others else t
                    else:
                        w_det = t
                else:
                    # collapse_to_mode: true value stays the mode for lam<1;
                    # ties at lam==1 break toward the smallest-domain value.
                    w_det = 0 if lam == 1.0 else t

                if granularity == "item":
                    h_true = residual_matrix[:, t]       # (n_items,)
                    h_sum = residual_sum                  # (n_items,)
                    upcl_scores = f0_row + (1 - lam) * h_true + (lam / n_worlds) * h_sum
                    det_scores = f0_row + residual_matrix[:, w_det]
                else:
                    h_true = residual_matrix[u, t]
                    h_sum = residual_sum[u]
                    upcl_scores = f0_row + (1 - lam) * h_true + (lam / n_worlds) * h_sum
                    det_scores = f0_row + residual_matrix[u, w_det]

                rank_upcl = int((upcl_scores > upcl_scores[i_true]).sum()) + 1
                rank_det = int((det_scores > det_scores[i_true]).sum()) + 1
                upcl_ranks.append(1.0 / rank_upcl)
                det_ranks.append(1.0 / rank_det)

            per_lambda[lam]["upcl_post"].append(float(np.mean(upcl_ranks)))
            per_lambda[lam]["deterministic"].append(float(np.mean(det_ranks)))

        print(f"seed {seed} done (train={len(train_df)}, test={len(test_df)}, "
              f"keys={n_keys} [{granularity}], worlds={n_worlds})")

    results = {}
    pvals = []
    lam_order = list(lambdas)
    for lam in lam_order:
        post = per_lambda[lam]["upcl_post"]
        det = per_lambda[lam]["deterministic"]
        p = 1.0 if lam == 0.0 else scipy_stats.ttest_rel(post, det)[1]
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

    meta = {"n_users": n_users, "n_items": n_items, "n_worlds": n_worlds,
            "n_rows": len(df), "context_col": context_col, "granularity": granularity}
    return results, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--context-col", required=True)
    parser.add_argument("--granularity", choices=["item", "user"], default="item")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", default="outputs/real_dataset_multiseed_log.json")
    parser.add_argument("--det-mode", choices=["collapse", "sample"], default="collapse")
    parser.add_argument("--eval-positive-only", action="store_true",
                         help="For implicit-feedback data with a binary rating "
                              "column: train on all rows, evaluate MRR only on "
                              "rating>0 (true positive interaction) rows.")
    args = parser.parse_args()

    results, meta = run(args.data, args.context_col, granularity=args.granularity,
                         n_seeds=args.seeds, eval_positive_only=args.eval_positive_only,
                         det_mode=args.det_mode)

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
