#!/usr/bin/env python3
"""
Statistical rigor table for RQ1/RQ2 (Appendix B.1.2, Table "rigor"):
paired Cohen's d and Holm-Bonferroni outcome for every comparison
summarized in Section 8's RQ1 and RQ2 paragraphs.

Reuses the exact same population, contexts, recommender, and per-user
score arrays as `full_evaluation.py` (same functions, imported directly)
so this table is guaranteed consistent with Table 1 (RQ1) and the RQ2
text rather than risking drift from a second, independently-written
implementation.

Run:
    python experiments/rigor_and_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_evaluation import build_user_contexts, ground_truth_relevance, run_methods
from upcl.stats import cohens_d_paired, holm_bonferroni, paired_ttest


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "rigor_and_baseline_log.txt"
    lines = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    base_contexts = build_user_contexts(seed=0)
    relevance = ground_truth_relevance(base_contexts)

    log("=== RQ1 family (m=4, lambda=0.3): UPCL-Pre/UPCL-Post vs. baselines ===")
    rng = np.random.default_rng(1)
    scores = run_methods(base_contexts, relevance, 0.3, rng)
    rq1_pairs = [
        ("UPCL-Pre vs. non-contextual", scores["upcl_pre"], scores["non_contextual"]),
        ("UPCL-Pre vs. deterministic", scores["upcl_pre"], scores["deterministic"]),
        ("UPCL-Post vs. non-contextual", scores["upcl_post"], scores["non_contextual"]),
        ("UPCL-Post vs. deterministic", scores["upcl_post"], scores["deterministic"]),
    ]
    rq1_p = [paired_ttest(a, b).p_value for _, a, b in rq1_pairs]
    rq1_sig = holm_bonferroni(rq1_p)
    for (name, a, b), p, sig in zip(rq1_pairs, rq1_p, rq1_sig):
        d = cohens_d_paired(a, b)
        log(f"{name:>32}: d={d:+.3f}, p={p:.3g}, Holm-sig={sig}")

    log("\n=== RQ2 family (m=5 noise levels): UPCL-Pre vs. deterministic ===")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    rq2_d, rq2_p = [], []
    for lam in levels:
        rng = np.random.default_rng(100)
        scores = run_methods(base_contexts, relevance, lam, rng)
        t = paired_ttest(scores["upcl_pre"], scores["deterministic"])
        d = cohens_d_paired(scores["upcl_pre"], scores["deterministic"])
        rq2_d.append(d)
        rq2_p.append(t.p_value)
    rq2_sig = holm_bonferroni(rq2_p)
    for lam, d, p, sig in zip(levels, rq2_d, rq2_p, rq2_sig):
        log(f"lambda={lam:.2f}: d={d:+.3f}, p={p:.3g}, Holm-sig={sig}")

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log(f"\nLog written to {out_path}")


if __name__ == "__main__":
    main()
