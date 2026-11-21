#!/usr/bin/env python3
"""
Non-contextual f0 reference for datasets run with real_dataset_multiseed.py
(DePaulMovie, LDOS-CoMoDa): MRR of the base MF score f0 alone vs. UPCL-Post at
lambda=0 (true context, i.e. f0 + shrunk residual of the true world), under the
same seeds/splits/MF/residual as real_dataset_multiseed.py.

    python experiments/f0_reference_multictx.py --data data/depaulmovie_tidy_clean.csv \
        --context-cols Time Location Companion --seeds 20 \
        --out outputs/depaulmovie_f0_reference_log.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_dataset_multiseed import train_mf, build_residual_table, cohens_d_paired

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--context-cols", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--granularity", default="item")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    users, items = sorted(df.user_id.unique()), sorted(df.item_id.unique())
    df["u_idx"] = df.user_id.map({u: k for k, u in enumerate(users)})
    df["i_idx"] = df.item_id.map({i: k for k, i in enumerate(items)})
    nu, ni = len(users), len(items); cols = a.context_cols
    f0m, postm = [], []
    for seed in range(a.seeds):
        sh = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sp = int(0.8 * len(sh)); tr, te = sh.iloc[:sp], sh.iloc[sp:]
        pred = train_mf(tr, nu, ni, seed=seed)
        tab = build_residual_table(tr, cols, pred, shrink_k=5, granularity=a.granularity)
        F = np.array([[pred(u, i) for i in range(ni)] for u in range(nu)])
        keyed = {}
        r0, r1 = [], []
        for row in te.itertuples():
            wk = tuple(getattr(row, c) for c in cols)
            s0 = F[row.u_idx]
            h = np.array([tab.get((items[i] if a.granularity == "item" else row.u_idx, wk), 0.0) for i in range(ni)]) \
                if a.granularity == "item" else np.full(ni, tab.get((row.u_idx, wk), 0.0))
            s1 = s0 + h; t = row.i_idx
            # rank by sorted order with stable tie-breaking by item order (as sorted(..., reverse=True) on dict)
            r0.append(1.0 / (int((s0 > s0[t]).sum()) + 1)); r1.append(1.0 / (int((s1 > s1[t]).sum()) + 1))
        f0m.append(float(np.mean(r0))); postm.append(float(np.mean(r1)))
        print(f"seed {seed}: f0={f0m[-1]:.4f} post(lam=0)={postm[-1]:.4f}", flush=True)
    t = stats.ttest_rel(postm, f0m)
    out = {"f0_mean": float(np.mean(f0m)), "f0_std": float(np.std(f0m, ddof=1)),
           "post_lam0_mean": float(np.mean(postm)), "post_lam0_std": float(np.std(postm, ddof=1)),
           "d_post_vs_f0": cohens_d_paired(postm, f0m), "p": float(t[1]),
           "f0_per_seed": f0m, "post_per_seed": postm}
    Path(a.out).write_text(json.dumps(out, indent=2)); print(out["f0_mean"], out["f0_std"], out["post_lam0_mean"], out["d_post_vs_f0"], out["p"])
if __name__ == "__main__":
    main()
