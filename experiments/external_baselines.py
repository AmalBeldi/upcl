#!/usr/bin/env python3
"""
Comparison with uncertainty-aware decision baselines on the synthetic
testbed (Appendix B, "Comparison with uncertainty-aware baselines").

Same population, contexts and recommender as `full_evaluation.py`
(100 users, 12 items, |Omega|=9, seed 0). The decision is one-shot: each
method observes a *noisy* belief (uniform-mix noise of intensity lambda)
and recommends one item. The true state is then drawn from the
*unperturbed* belief and the method is scored by the realized utility
worldwise(u, item, w*) of the item it recommended. Realized utility, not
a ranking metric built from Expected Utility, is used here so that the
comparison does not favour Expected-Utility-based methods by construction.

Methods (all see the same noisy belief):
  non_contextual     argmax of the context-free base score
  deterministic      mode-collapsed context (classical pre-filtering)
  thompson           one world sampled from the belief (Thompson-style)
  credible_set       uniform mean over the smallest world set covering
                     90% of the belief mass (conformal-style set, no
                     probabilities inside the set)
  bayes_lcb          mean - 1 * std of world-wise scores under the belief
  upcl_eu / upcl_cvar / upcl_maxmin / upcl_hurwicz / upcl_dro
                     UPCL-Pre with the corresponding aggregation operator

Two per-user statistics over M true-state draws: mean realized utility
and 5th percentile of realized utility. Paired t-tests over the 100
users, Cohen's d (paired), Holm-Bonferroni within each noise level.

Run:
    python experiments/external_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_evaluation import ITEMS, RECOMMENDER, USERS, build_user_contexts
from upcl.aggregation import cvar, dro, hurwicz, maxmin
from upcl.context import ProbabilisticContextInstance, phi
from upcl.noise import inject_noise
from upcl.stats import cohens_d_paired, holm_bonferroni, paired_ttest

M_DRAWS = 200
LEVELS = [0.0, 0.3, 0.5, 1.0]
BASELINES = ["non_contextual", "deterministic", "thompson", "credible_set", "bayes_lcb"]
UPCL = ["upcl_eu", "upcl_cvar", "upcl_maxmin", "upcl_hurwicz", "upcl_dro"]


def score_matrix(user, worlds):
    return np.array([[RECOMMENDER.worldwise(user, i, w) for w in worlds] for i in ITEMS])


def credible_set_mask(probs, mass=0.9):
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    k = int(np.searchsorted(cum, mass - 1e-12) + 1)
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[:k]] = True
    return mask


def choices_fixed(S, probs, base_scores):
    """Item index chosen by each deterministic-given-belief method."""
    mean = S @ probs
    std = np.sqrt(np.maximum((S ** 2) @ probs - mean ** 2, 0.0))
    mask = credible_set_mask(probs)
    out = {
        "non_contextual": int(np.argmax(base_scores)),
        "deterministic": int(np.argmax(S[:, int(np.argmax(probs))])),
        "credible_set": int(np.argmax(S[:, mask].mean(axis=1))),
        "bayes_lcb": int(np.argmax(mean - std)),
        "upcl_eu": int(np.argmax(mean)),
    }
    out["upcl_cvar"] = int(np.argmax([cvar(S[i], probs, beta=0.7) for i in range(len(ITEMS))]))
    out["upcl_maxmin"] = int(np.argmax([maxmin(S[i]) for i in range(len(ITEMS))]))
    out["upcl_hurwicz"] = int(np.argmax([hurwicz(S[i], probs, alpha=0.5) for i in range(len(ITEMS))]))
    out["upcl_dro"] = int(np.argmax([dro(S[i], probs, radius=0.5, ambiguity="kl") for i in range(len(ITEMS))]))
    return out


def run_level(base_contexts, lam, seed):
    rng = np.random.default_rng(seed)
    methods = BASELINES + UPCL
    mean_u = {m: [] for m in methods}
    p5_u = {m: [] for m in methods}
    for user in USERS:
        inst = base_contexts[user]
        true_space = phi(inst)
        noisy = ProbabilisticContextInstance(
            categorical_variables=[
                inject_noise(v, lam, rng, mode="uniform_mix") for v in inst.categorical_variables
            ]
        )
        noisy_space = phi(noisy)
        assert true_space.worlds == noisy_space.worlds
        S = score_matrix(user, noisy_space.worlds)
        p_true = np.array(true_space.probabilities)
        p_noisy = np.array(noisy_space.probabilities)
        base_scores = [RECOMMENDER.base(user, i) for i in ITEMS]
        fixed = choices_fixed(S, p_noisy, base_scores)

        w_star = rng.choice(len(p_true), size=M_DRAWS, p=p_true)
        w_ts = rng.choice(len(p_noisy), size=M_DRAWS, p=p_noisy)
        for m in methods:
            if m == "thompson":
                chosen = np.argmax(S[:, w_ts], axis=0)  # one choice per draw
                realized = S[chosen, w_star]
            else:
                realized = S[fixed[m], w_star]
            mean_u[m].append(float(np.mean(realized)))
            p5_u[m].append(float(np.percentile(realized, 5)))
    return mean_u, p5_u


def compare(stat, focus, others, log, label):
    pv, rows = [], []
    for f in focus:
        for o in others:
            t = paired_ttest(stat[f], stat[o])
            pv.append(t.p_value)
            rows.append((f, o, cohens_d_paired(stat[f], stat[o]), t.p_value))
    sig = holm_bonferroni(pv)
    for (f, o, d, p), s in zip(rows, sig):
        log(f"  [{label}] {f} vs {o}: d={d:+.3f}, p={p:.3g}, Holm={'yes' if s else 'no'}")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "outputs" / "external_baselines_log.txt"
    out.parent.mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    base_contexts = build_user_contexts(0)
    log(f"External-baseline comparison: {len(USERS)} users, {len(ITEMS)} items, "
        f"M={M_DRAWS} true-state draws per user; metric = realized utility of the "
        f"recommended item.\n")
    for lam in LEVELS:
        mean_u, p5_u = run_level(base_contexts, lam, seed=2000 + int(lam * 100))
        log(f"=== lambda={lam:.2f} ===")
        log(f"{'method':>15} | mean realized utility | 5th pct realized utility")
        for m in BASELINES + UPCL:
            log(f"{m:>15} | {np.mean(mean_u[m]):.3f} +/- {np.std(mean_u[m], ddof=1):.3f}"
                f"       | {np.mean(p5_u[m]):.3f} +/- {np.std(p5_u[m], ddof=1):.3f}")
        compare(mean_u, ["upcl_eu"], BASELINES, log, "mean")
        compare(p5_u, ["upcl_cvar", "upcl_maxmin"], BASELINES, log, "p5")
        log()
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
