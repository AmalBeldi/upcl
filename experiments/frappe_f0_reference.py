#!/usr/bin/env python3
"""
Non-contextual reference for Frappe (Appendix B.1.7): MRR of the base score f0
alone (no context term), under exactly the protocol of
real_dataset_multiseed_fast.py (same seeds, 80/20 split, same MF, positive-only
evaluation). Also recomputes UPCL-Post at lambda=0 as a consistency check.

    python experiments/frappe_f0_reference.py --data data/frappe_prepared.csv \
        --context-col isweekend --seeds 20 --out outputs/frappe_f0_reference_log.json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from real_dataset_multiseed_fast import train_mf, build_residual_matrix, cohens_d_paired
from scipy import stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--context-col", required=True)
    ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    users, items, worlds = sorted(df.user_id.unique()), sorted(df.item_id.unique()), sorted(df[a.context_col].unique())
    df["u_idx"] = df.user_id.map({u: k for k, u in enumerate(users)})
    df["i_idx"] = df.item_id.map({i: k for k, i in enumerate(items)})
    df["w_idx"] = df[a.context_col].map({w: k for k, w in enumerate(worlds)})
    nu, ni, nw = len(users), len(items), len(worlds)
    f0m, postm = [], []
    for seed in range(a.seeds):
        sh = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sp = int(0.8 * len(sh)); tr, te = sh.iloc[:sp], sh.iloc[sp:]
        te = te[te.rating > 0].reset_index(drop=True)
        F = train_mf(tr, nu, ni, seed=seed)
        res = tr.rating.to_numpy() - F[tr.u_idx.to_numpy(), tr.i_idx.to_numpy()]
        H = build_residual_matrix(tr, "i_idx", ni, nw, res)
        u, i, w = te.u_idx.to_numpy(), te.i_idx.to_numpy(), te.w_idx.to_numpy()
        r0, r1 = [], []
        for k in range(len(te)):
            s0 = F[u[k]]; s1 = s0 + H[:, w[k]]
            r0.append(1.0 / (int((s0 > s0[i[k]]).sum()) + 1))
            r1.append(1.0 / (int((s1 > s1[i[k]]).sum()) + 1))
        f0m.append(float(np.mean(r0))); postm.append(float(np.mean(r1)))
        print(f"seed {seed}: f0={f0m[-1]:.4f} post(lam=0)={postm[-1]:.4f}", flush=True)
    t = stats.ttest_rel(postm, f0m)
    out = {"f0_mean": float(np.mean(f0m)), "f0_std": float(np.std(f0m, ddof=1)),
           "post_lam0_mean": float(np.mean(postm)), "post_lam0_std": float(np.std(postm, ddof=1)),
           "d_post_vs_f0": cohens_d_paired(postm, f0m), "p": float(t[1]),
           "f0_per_seed": f0m, "post_per_seed": postm}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2)); print(out["f0_mean"], out["f0_std"], out["post_lam0_mean"], out["d_post_vs_f0"], out["p"])
if __name__ == "__main__":
    main()
