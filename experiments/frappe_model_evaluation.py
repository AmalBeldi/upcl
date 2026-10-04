#!/usr/bin/env python3
"""
UPCL-Model (Section 5.2) evaluated on the real Frappe-x1 benchmark --
the real-dataset counterpart to the synthetic UPCL-Model evaluation
(Section 8, "UPCL-Model (Learned Integration)" paragraph / Table 6),
closing the limitation noted there and in the Discussion/Limitations
paragraph: UPCL-Model was previously validated only on the synthetic
structured testbed, never on real interaction data.

Design (mirrors the main paper's synthetic UPCL-Model protocol exactly,
Section 8): E_phi maps a belief over the single binary context variable
`isweekend` to the probability-weighted mean of its one-hot encoding,
i.e. z = (P(weekday), P(weekend)); f_phi is a small contextual matrix
factorization,
    f_phi(u, i, z) = mu + b_u[u] + b_i[i] + P[u].Q[i] + z . C[i]
with a learned per-item, per-context-value bias C[i, c] fit JOINTLY
with the base MF term by SGD on the OBSERVED (single-world, z =
one-hot(true value)) training interactions -- unlike UPCL-Post's h_eta,
which is a post-hoc shrunk residual average computed in a separate
second stage after f0 is already fixed. This is the same conceptual
difference the main paper draws between the two instantiations
(Section 5.2 vs 5.3): uncertainty enters the recommender itself at
training time via C, rather than being aggregated by Gamma_theta after
independently-trained per-world adjustments.

At serving time, under noise injected on the SAME uniform_mix schedule
as every other noise-sweep experiment in this paper (Section 7.2),
z_eval is the raw (uncollapsed) noisy belief -- not the mode -- so
UPCL-Model consumes graded uncertainty exactly as Eq. (unified
architecture) and Section 5.2 specify.

Three systems are compared, exactly mirroring the synthetic UPCL-Model
paragraph's own comparison set:
  - non-contextual: f0(u,i) alone, no context term at all (C == 0)
  - deterministic:  z collapsed to its mode before being fed to f_phi
                     (noise.collapse_to_mode's closed-form equivalent)
  - UPCL-Model:      f_phi(u,i,z) with the full noisy belief

Because `isweekend` has a domain of exactly 2 values, the per-test-row
MRR computation uses the same closed-form uniform_mix vectorization as
real_dataset_multiseed_fast.py (validated there against the exact,
upcl.instantiations-based pipeline on DePaulMovie), for the same
runtime reason: Frappe's 4,082-item catalog makes a pure per-world
Python loop through upcl.instantiations.upcl_model impractically slow
for a 10+-seed multi-lambda sweep.

USAGE
-----
    python experiments/frappe_model_evaluation.py \
        --data data/frappe_prepared.csv --context-col isweekend \
        --seeds 10 --out outputs/frappe_model_evaluation_log.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def train_contextual_mf(train_df, n_users, n_items, n_worlds, n_factors=10,
                         epochs=30, lr=0.01, l2=0.05, seed=0):
    """Jointly fits mu, b_u, b_i, P, Q (the non-contextual f0 term) and
    C[i, c] (the per-item, per-context-value bias f_phi adds on top),
    from the OBSERVED single-world context of each training row --
    "fit by SGD on realized (single-world) historical interactions"
    (Section 8, UPCL-Model paragraph), here on real rather than
    synthetic interactions.
    """
    rng = np.random.default_rng(seed)
    mu = train_df["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    P = 0.1 * rng.standard_normal((n_users, n_factors))
    Q = 0.1 * rng.standard_normal((n_items, n_factors))
    C = np.zeros((n_items, n_worlds))  # context bias, jointly trained

    rows = train_df[["u_idx", "i_idx", "w_idx", "rating"]].to_numpy()
    for _ in range(epochs):
        rng.shuffle(rows)
        for u, i, c, r in rows:
            u, i, c = int(u), int(i), int(c)
            pred = mu + bu[u] + bi[i] + P[u] @ Q[i] + C[i, c]
            err = r - pred
            bu[u] += lr * (err - l2 * bu[u])
            bi[i] += lr * (err - l2 * bi[i])
            p_u = P[u].copy()
            P[u] += lr * (err * Q[i] - l2 * P[u])
            Q[i] += lr * (err * p_u - l2 * Q[i])
            C[i, c] += lr * (err - l2 * C[i, c])

    f0_matrix = mu + bu[:, None] + bi[None, :] + P @ Q.T  # (n_users, n_items)
    return f0_matrix, C


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


def run(data_path, context_col, n_seeds=10,
        lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), n_factors=10):
    df = pd.read_csv(data_path)

    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    worlds = sorted(df[context_col].unique())
    n_worlds = len(worlds)
    u_index = {u: k for k, u in enumerate(users)}
    i_index = {i: k for k, i in enumerate(items)}
    w_index = {w: k for k, w in enumerate(worlds)}
    df["u_idx"] = df["user_id"].map(u_index)
    df["i_idx"] = df["item_id"].map(i_index)
    df["w_idx"] = df[context_col].map(w_index)
    n_users, n_items = len(users), len(items)

    per_lambda = {lam: {"model": [], "deterministic": [], "noncontextual": []}
                  for lam in lambdas}

    for seed in range(n_seeds):
        t0 = time.time()
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split = int(0.8 * len(shuffled))
        train_df = shuffled.iloc[:split]
        # Implicit-feedback: train on positives+negatives, evaluate MRR
        # only on held-out positive interactions (same convention as
        # real_dataset_multiseed_fast.py's --eval-positive-only).
        test_df = shuffled.iloc[split:]
        test_df = test_df[test_df["rating"] > 0].reset_index(drop=True)

        f0_matrix, C = train_contextual_mf(
            train_df, n_users, n_items, n_worlds, n_factors=n_factors, seed=seed
        )
        C_sum = C.sum(axis=1)  # (n_items,) -- sum over all context values

        test_u = test_df["u_idx"].to_numpy()
        test_i = test_df["i_idx"].to_numpy()
        test_w = test_df["w_idx"].to_numpy()

        for lam in lambdas:
            model_ranks, det_ranks, nc_ranks = [], [], []
            for row_idx in range(len(test_df)):
                u, i_true, t = test_u[row_idx], test_i[row_idx], test_w[row_idx]
                f0_row = f0_matrix[u]  # (n_items,)

                # closed-form uniform_mix belief: p(true)=(1-lam)+lam/k,
                # p(other)=lam/k each (validated in real_dataset_multiseed_fast.py)
                model_scores = f0_row + (1 - lam) * C[:, t] + (lam / n_worlds) * C_sum

                # deterministic: z collapsed to the mode (true value, except
                # lam==1 exactly, where ties break to the smallest index)
                w_det = 0 if lam == 1.0 else t
                det_scores = f0_row + C[:, w_det]

                nc_scores = f0_row  # non-contextual: no C term at all

                rank_model = int((model_scores > model_scores[i_true]).sum()) + 1
                rank_det = int((det_scores > det_scores[i_true]).sum()) + 1
                rank_nc = int((nc_scores > nc_scores[i_true]).sum()) + 1
                model_ranks.append(1.0 / rank_model)
                det_ranks.append(1.0 / rank_det)
                nc_ranks.append(1.0 / rank_nc)

            per_lambda[lam]["model"].append(float(np.mean(model_ranks)))
            per_lambda[lam]["deterministic"].append(float(np.mean(det_ranks)))
            per_lambda[lam]["noncontextual"].append(float(np.mean(nc_ranks)))

        print(f"seed {seed} done (train={len(train_df)}, test={len(test_df)}, "
              f"n_items={n_items}, n_worlds={n_worlds}) [{time.time()-t0:.1f}s]",
              flush=True)

    results = {}
    lam_order = list(lambdas)
    pvals_det, pvals_nc = [], []
    for lam in lam_order:
        model = per_lambda[lam]["model"]
        det = per_lambda[lam]["deterministic"]
        nc = per_lambda[lam]["noncontextual"]
        p_det = 1.0 if lam == 0.0 else scipy_stats.ttest_rel(model, det)[1]
        p_nc = scipy_stats.ttest_rel(model, nc)[1]
        pvals_det.append(p_det)
        pvals_nc.append(p_nc)
        results[lam] = {
            "model_mean": float(np.mean(model)), "model_std": float(np.std(model, ddof=1)),
            "deterministic_mean": float(np.mean(det)), "deterministic_std": float(np.std(det, ddof=1)),
            "noncontextual_mean": float(np.mean(nc)), "noncontextual_std": float(np.std(nc, ddof=1)),
            "cohens_d_vs_det": cohens_d_paired(model, det),
            "cohens_d_vs_nc": cohens_d_paired(model, nc),
            "p_vs_det": float(p_det),
            "p_vs_nc": float(p_nc),
        }
    for lam, sig in zip(lam_order, holm_bonferroni(pvals_det)):
        results[lam]["holm_significant_vs_det"] = bool(sig)
    for lam, sig in zip(lam_order, holm_bonferroni(pvals_nc)):
        results[lam]["holm_significant_vs_nc"] = bool(sig)

    meta = {"n_users": n_users, "n_items": n_items, "n_worlds": n_worlds,
            "n_rows": len(df), "context_col": context_col, "n_seeds": n_seeds,
            "n_factors": n_factors}
    return results, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--context-col", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-factors", type=int, default=10)
    parser.add_argument("--out", default="outputs/frappe_model_evaluation_log.json")
    args = parser.parse_args()

    results, meta = run(args.data, args.context_col, n_seeds=args.seeds,
                         n_factors=args.n_factors)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"\nWrote {args.out}")
    print(meta)
    for lam, r in results.items():
        print(
            f"lambda={lam}: Model={r['model_mean']:.4f}+-{r['model_std']:.4f}, "
            f"Det={r['deterministic_mean']:.4f}+-{r['deterministic_std']:.4f}, "
            f"NC={r['noncontextual_mean']:.4f}+-{r['noncontextual_std']:.4f}, "
            f"d(vs det)={r['cohens_d_vs_det']:+.2f} p={r['p_vs_det']:.4g} Holm-sig={r['holm_significant_vs_det']}, "
            f"d(vs nc)={r['cohens_d_vs_nc']:+.2f} p={r['p_vs_nc']:.4g} Holm-sig={r['holm_significant_vs_nc']}"
        )


if __name__ == "__main__":
    main()
