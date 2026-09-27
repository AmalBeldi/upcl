"""
Additional ablation: sensitivity of CVaR (beta) and DRO (KL radius rho)
to their own hyperparameters, on the exact same RQ4 adversarial testbed
and protocol as rq4_full_sweep.py (5 seeds x 500 trials, same safe/risky
reward distributions, same per-trial Dirichlet world probabilities).

Motivation: Table 3 (Appendix B) shows the risk-return trade-off across
Hurwicz's alpha. This ablation shows the analogous trade-off is smooth
and monotonic for CVaR's confidence level beta and DRO's ambiguity
radius rho, supporting the claim that Gamma_theta's genericity
(Proposition on Unified Aggregation Interface) extends to *any* choice
of theta, not just the specific settings (beta=0.7, rho=0.3) reported
in the main paper.
"""
import numpy as np
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq4_full_sweep import (
    N_SEEDS, N_TRIALS, N_WORLDS, SAFE_MU, SAFE_SIGMA, RISKY_MU, RISKY_SIGMA,
    cvar_discrete, dro_kl_worst_case,
)

CVAR_BETAS = [0.5, 0.6, 0.7, 0.8, 0.9]
DRO_RHOS = [0.1, 0.2, 0.3, 0.5, 1.0]


def run_seed_hparam(seed, cvar_betas, dro_rhos):
    rng = np.random.default_rng(seed)
    realized_cvar = {b: np.zeros(N_TRIALS) for b in cvar_betas}
    realized_dro = {r: np.zeros(N_TRIALS) for r in dro_rhos}
    for t in range(N_TRIALS):
        probs = rng.dirichlet(np.ones(N_WORLDS))
        safe_r = rng.normal(SAFE_MU, SAFE_SIGMA, size=N_WORLDS)
        risky_r = rng.normal(RISKY_MU, RISKY_SIGMA, size=N_WORLDS)
        true_world = rng.choice(N_WORLDS, p=probs)

        for b in cvar_betas:
            s_safe = -cvar_discrete(safe_r, probs, b)
            s_risky = -cvar_discrete(risky_r, probs, b)
            chosen = safe_r if s_safe >= s_risky else risky_r
            realized_cvar[b][t] = chosen[true_world]

        for rho in dro_rhos:
            s_safe = dro_kl_worst_case(safe_r, probs, rho)
            s_risky = dro_kl_worst_case(risky_r, probs, rho)
            chosen = safe_r if s_safe >= s_risky else risky_r
            realized_dro[rho][t] = chosen[true_world]
    return realized_cvar, realized_dro


def main():
    cvar_agg = {b: {"mean": [], "p5": []} for b in CVAR_BETAS}
    dro_agg = {r: {"mean": [], "p5": []} for r in DRO_RHOS}

    for seed in range(N_SEEDS):
        rc, rd = run_seed_hparam(seed, CVAR_BETAS, DRO_RHOS)
        for b in CVAR_BETAS:
            cvar_agg[b]["mean"].append(float(np.mean(rc[b])))
            cvar_agg[b]["p5"].append(float(np.percentile(rc[b], 5)))
        for r in DRO_RHOS:
            dro_agg[r]["mean"].append(float(np.mean(rd[r])))
            dro_agg[r]["p5"].append(float(np.percentile(rd[r], 5)))

    print("CVaR beta sensitivity (mean realized +- std, 5th pct +- std over 5 seeds):")
    cvar_summary = {}
    for b in CVAR_BETAS:
        m = np.array(cvar_agg[b]["mean"]); p5 = np.array(cvar_agg[b]["p5"])
        cvar_summary[b] = dict(mean=float(m.mean()), mean_std=float(m.std(ddof=1)),
                                p5=float(p5.mean()), p5_std=float(p5.std(ddof=1)))
        print(f"  beta={b}: mean={m.mean():.3f}+-{m.std(ddof=1):.3f}, "
              f"5th pct={p5.mean():.3f}+-{p5.std(ddof=1):.3f}")

    print("\nDRO rho sensitivity (mean realized +- std, 5th pct +- std over 5 seeds):")
    dro_summary = {}
    for r in DRO_RHOS:
        m = np.array(dro_agg[r]["mean"]); p5 = np.array(dro_agg[r]["p5"])
        dro_summary[r] = dict(mean=float(m.mean()), mean_std=float(m.std(ddof=1)),
                               p5=float(p5.mean()), p5_std=float(p5.std(ddof=1)))
        print(f"  rho={r}: mean={m.mean():.3f}+-{m.std(ddof=1):.3f}, "
              f"5th pct={p5.mean():.3f}+-{p5.std(ddof=1):.3f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/rq4_hyperparam_sensitivity.json", "w") as f:
        json.dump({"cvar": cvar_summary, "dro": dro_summary}, f, indent=2)


if __name__ == "__main__":
    main()
