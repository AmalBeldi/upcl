"""Sensitivity of the DePaulMovie real-dataset result to the MF base-recommender's
latent-factor count (n_factors) and the residual-shrinkage constant (shrink_k).

Motivation (see appendix.tex, Section B.1): DePaulMovie's per-(item,world) cell
density is low (~3.8 rows/cell on average), so the estimated residual table
h_eta that UPCL-Post adjusts the MF base score with is itself noisy. This
ablation asks whether the (non-significant) DePaulMovie result in Table 6 is
an artifact of one particular (n_factors=10, shrink_k=5) choice, or whether
the same qualitative picture (small, non-significant effect) holds across a
grid of MF-capacity / shrinkage settings.

Protocol: same exact pipeline as real_dataset_multiseed.py (DePaulMovie,
3 context variables Time+Location+Companion, UPCL-Post vs. deterministic
collapse, paired over seeds), but run in-process over a grid of
(n_factors, shrink_k) at a single representative noise level lambda=0.3,
with a reduced seed count (10 instead of 20) to keep total runtime to a
few minutes while still supporting a paired t-test / Cohen's d per cell.
"""
import json
import time
from pathlib import Path

from real_dataset_multiseed import run

DATA = "../data/depaulmovie_tidy_clean.csv"
CONTEXT_COLS = ["Time", "Location", "Companion"]
N_SEEDS = 10
LAMBDA = 0.3

N_FACTORS_GRID = [5, 10, 20]
SHRINK_K_GRID = [1, 5, 20]

OUT = "outputs/mf_shrink_k_ablation.json"

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for nf in N_FACTORS_GRID:
        for sk in SHRINK_K_GRID:
            t1 = time.time()
            results, meta = run(
                DATA, CONTEXT_COLS, n_seeds=N_SEEDS, lambdas=(LAMBDA,),
                policy="eu", granularity="item", n_factors=nf, shrink_k=sk,
            )
            r = results[LAMBDA]
            row = {
                "n_factors": nf, "shrink_k": sk,
                "upcl_post_mean": r["upcl_post_mean"], "upcl_post_std": r["upcl_post_std"],
                "deterministic_mean": r["deterministic_mean"], "deterministic_std": r["deterministic_std"],
                "cohens_d": r["cohens_d"], "p_value": r["p_value"],
                "holm_significant": r["holm_significant"],
            }
            rows.append(row)
            dt = time.time() - t1
            print(f"n_factors={nf:3d} shrink_k={sk:5.1f}  "
                  f"d={r['cohens_d']:+.3f} p={r['p_value']:.4g} sig={r['holm_significant']}  "
                  f"({dt:.1f}s)", flush=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "data": DATA, "context_cols": CONTEXT_COLS, "n_seeds": N_SEEDS,
            "lambda": LAMBDA, "n_factors_grid": N_FACTORS_GRID,
            "shrink_k_grid": SHRINK_K_GRID, "rows": rows,
        }, f, indent=2)

    print(f"\nTotal time: {time.time()-t0:.1f}s. Wrote {OUT}")
