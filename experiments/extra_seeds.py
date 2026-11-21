#!/usr/bin/env python3
"""
Additional-seed robustness check for RQ2 and RQ4 (Appendix B.1.12).
The headline numbers of the main paper (RQ2: 3 noise seeds; RQ4: 5 seeds) are
unchanged; this script re-runs both protocols with 20 seeds.

RQ2: 20 population seeds (belief draws; uniform-mix noise itself is
     deterministic, so the paper's 3 noise seeds were identical replicates),
     100 users each. Per population: UPCL-Pre vs deterministic NDCG@5 (paired
     over users, Cohen's d). Across populations: mean/min/max d and a paired
     t-test on the per-population mean NDCG (n=20), Holm across noise levels.
RQ4: seeds 0..19 (the first five are the paper's), 500 trials each, same
     operators. Per-seed mean and 5th percentile; paired tests vs Bayesian-LCB
     across seeds (n=20), Holm within each metric.

    python experiments/extra_seeds.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_evaluation import build_user_contexts, ground_truth_relevance, run_methods
from upcl.stats import cohens_d_paired, holm_bonferroni, paired_ttest
import rq4_full_sweep as rq4

N_SEEDS = 20
OUT = Path(__file__).resolve().parents[1] / "outputs" / "extra_seeds_log.txt"
lines = []
def log(s=""):
    print(s, flush=True); lines.append(s)

def main():
    log(f"=== RQ2 with {N_SEEDS} population seeds (UPCL-Pre vs deterministic, NDCG@5, 100 users each) ===")
    log("Note: uniform-mix noise is deterministic, so the paper's 3 'noise seeds' were identical replicates;")
    log("the real source of randomness is the draw of the user beliefs (build_user_contexts seed).")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    pops = []
    for s_ in range(N_SEEDS):
        b = build_user_contexts(seed=s_); pops.append((b, ground_truth_relevance(b)))
    pv, rows = [], []
    for lam in levels:
        pre_m, det_m, ds = [], [], []
        for b, rel in pops:
            sc = run_methods(b, rel, lam, np.random.default_rng(100))
            pre_m.append(float(np.mean(sc["upcl_pre"]))); det_m.append(float(np.mean(sc["deterministic"])))
            ds.append(cohens_d_paired(sc["upcl_pre"], sc["deterministic"]))
        t = paired_ttest(pre_m, det_m)
        pv.append(t.p_value)
        rows.append((lam, np.mean(pre_m), np.std(pre_m, ddof=1), np.mean(det_m), np.std(det_m, ddof=1),
                     np.mean(ds), min(ds), max(ds), cohens_d_paired(pre_m, det_m), t.p_value))
    sig = holm_bonferroni(pv)
    log(f"{'lambda':>6} | UPCL-Pre mean+-sd | Determ. mean+-sd | per-population user-level d (mean [min,max]) | population-level d, p (Holm)")
    for r, s2 in zip(rows, sig):
        log(f"{r[0]:>6.2f} | {r[1]:.4f}+-{r[2]:.4f} | {r[3]:.4f}+-{r[4]:.4f} | {r[5]:+.3f} [{r[6]:+.3f},{r[7]:+.3f}] | d={r[8]:+.2f}, p={r[9]:.3g} ({'yes' if s2 else 'no'})")

    log(f"\n=== RQ4 with {N_SEEDS} seeds, 500 trials each (mean +- sd across seeds) ===")
    rq4.N_SEEDS = N_SEEDS
    per = {p: {"mean": [], "p5": []} for p in rq4.POLICIES}
    for s in range(N_SEEDS):
        real = rq4.run_seed(s)
        for p in rq4.POLICIES:
            per[p]["mean"].append(float(np.mean(real[p]))); per[p]["p5"].append(float(np.percentile(real[p], 5)))
        print(f"  RQ4 seed {s} done", flush=True)
    log(f"{'policy':>12} | mean realized | 5th pct")
    for p in rq4.POLICIES:
        log(f"{p:>12} | {np.mean(per[p]['mean']):.3f}+-{np.std(per[p]['mean'], ddof=1):.3f} | {np.mean(per[p]['p5']):.3f}+-{np.std(per[p]['p5'], ddof=1):.3f}")
    focus = ["EU", "Hurwicz_0.5", "CVaR", "DRO", "MM"]
    for metric, lab in (("mean", "mean"), ("p5", "5th percentile")):
        pv, rws = [], []
        for p in focus:
            t = paired_ttest(per[p][metric], per["LCB"][metric]); pv.append(t.p_value)
            rws.append((p, cohens_d_paired(per[p][metric], per["LCB"][metric]), t.p_value))
        for (p, d, pp), sg in zip(rws, holm_bonferroni(pv)):
            log(f"  [{lab}] {p} vs LCB: d={d:+.3f}, p={pp:.3g}, Holm={'yes' if sg else 'no'}")
    OUT.write_text("\n".join(lines) + "\n"); log(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
