"""
RQ4 -- Risk-Sensitive Aggregation, full multi-seed protocol.

Protocol (matches Appendix B "RQ4 adversarial testbed" description):
  - 5 seeds (0..4), 500 trials per seed.
  - Each trial: 4 possible worlds, world probabilities drawn from a
    symmetric Dirichlet(1,1,1,1) prior (fresh draw per trial).
  - Two candidate items:
      "safe":  per-world reward ~ N(3.0, 0.3^2), drawn i.i.d. per world
      "risky": per-world reward ~ N(3.3, 1.5^2), drawn i.i.d. per world
  - A policy (aggregation operator Gamma_theta) scores each item from
    its 4 per-world rewards + the world probabilities; the item with
    the higher aggregated score is chosen.
  - The *realized* utility is the reward of the chosen item under the
    world that is actually drawn for that trial, sampled from the same
    per-trial world-probability distribution (this is what makes
    "realized utility" meaningfully differ across policies: a policy
    that is fooled by a high mean/max can still lose when the adverse
    world materializes).

Aggregation operators evaluated: Expected Utility (EU), Maxmin (MM),
Hurwicz (alpha in {0.1,...,0.9}), CVaR (beta=0.7), DRO (KL ball,
radius 0.3), and the Bayesian-LCB point-estimate baseline (z=1).

Output: per-seed and 5-seed mean+-std of (a) mean realized utility,
(b) 5th-percentile (worst-case) realized utility, (c) std of realized
utility, for every policy above. Feeds both Table 2 (main paper,
headline row) and Table 3 (Appendix, full Hurwicz sweep) from the SAME
run, guaranteeing the two tables are numerically consistent by
construction.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import json
import os

N_SEEDS = 5
N_TRIALS = 500
N_WORLDS = 4
SAFE_MU, SAFE_SIGMA = 3.0, 0.3
RISKY_MU, RISKY_SIGMA = 3.3, 1.5
HURWICZ_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
CVAR_BETA = 0.7
DRO_RHO = 0.3


def cvar_discrete(rewards, probs, beta):
    """Rockafellar-Uryasev CVaR of the LOSS L = -reward, discrete probs."""
    losses = -rewards

    def ru_objective(tau):
        return tau + (1.0 / (1.0 - beta)) * np.sum(probs * np.maximum(losses - tau, 0.0))

    lo, hi = losses.min() - 1.0, losses.max() + 1.0
    res = minimize_scalar(ru_objective, bounds=(lo, hi), method="bounded")
    return res.fun  # CVaR_beta(L)


def dro_kl_worst_case(rewards, probs, rho):
    """inf_{Q in KL-ball(rho) around P} E_Q[reward], via convex dual:
    sup_{beta>0} [ -beta*log(sum_i p_i*exp(-r_i/beta)) - beta*rho ]."""

    def dual(beta):
        # numerically stable log-sum-exp
        z = -rewards / beta
        m = z.max()
        lse = m + np.log(np.sum(probs * np.exp(z - m)))
        return -(-beta * lse - beta * rho)  # minimize negative of the objective

    res = minimize_scalar(dual, bounds=(1e-3, 50.0), method="bounded")
    return -res.fun


def scores_for_item(rewards, probs):
    """Return a dict of aggregation-policy scores for one item's 4
    per-world rewards and the shared world-probability vector."""
    out = {}
    out["EU"] = float(np.sum(probs * rewards))
    out["MM"] = float(rewards.min())
    for a in HURWICZ_ALPHAS:
        out[f"Hurwicz_{a}"] = float(a * rewards.max() + (1 - a) * rewards.min())
    out["CVaR"] = float(-cvar_discrete(rewards, probs, CVAR_BETA))  # utility form
    out["DRO"] = float(dro_kl_worst_case(rewards, probs, DRO_RHO))
    mean_r = np.sum(probs * rewards)
    var_r = np.sum(probs * (rewards - mean_r) ** 2)
    out["LCB"] = float(mean_r - 1.0 * np.sqrt(var_r))
    return out


POLICIES = ["EU", "MM"] + [f"Hurwicz_{a}" for a in HURWICZ_ALPHAS] + ["CVaR", "DRO", "LCB"]


def run_seed(seed):
    rng = np.random.default_rng(seed)
    realized = {p: np.zeros(N_TRIALS) for p in POLICIES}
    for t in range(N_TRIALS):
        probs = rng.dirichlet(np.ones(N_WORLDS))
        safe_r = rng.normal(SAFE_MU, SAFE_SIGMA, size=N_WORLDS)
        risky_r = rng.normal(RISKY_MU, RISKY_SIGMA, size=N_WORLDS)
        true_world = rng.choice(N_WORLDS, p=probs)

        safe_scores = scores_for_item(safe_r, probs)
        risky_scores = scores_for_item(risky_r, probs)

        for p in POLICIES:
            chosen_r = safe_r if safe_scores[p] >= risky_scores[p] else risky_r
            realized[p][t] = chosen_r[true_world]
    return realized


def main():
    per_seed = {p: {"mean": [], "p5": [], "std": []} for p in POLICIES}
    all_trials = {p: [] for p in POLICIES}  # for paired stats across seeds pooled

    for seed in range(N_SEEDS):
        realized = run_seed(seed)
        for p in POLICIES:
            r = realized[p]
            per_seed[p]["mean"].append(float(np.mean(r)))
            per_seed[p]["p5"].append(float(np.percentile(r, 5)))
            per_seed[p]["std"].append(float(np.std(r)))
            all_trials[p].append(r)

    summary = {}
    for p in POLICIES:
        summary[p] = {
            "mean_of_means": float(np.mean(per_seed[p]["mean"])),
            "std_of_means": float(np.std(per_seed[p]["mean"], ddof=1)),
            "mean_of_p5": float(np.mean(per_seed[p]["p5"])),
            "std_of_p5": float(np.std(per_seed[p]["p5"], ddof=1)),
            "mean_of_std": float(np.mean(per_seed[p]["std"])),
        }

    print("=" * 78)
    print(f"{'Policy':<12}{'Mean realized':>18}{'5th pct (worst)':>20}{'Std':>10}")
    print("-" * 78)
    for p in POLICIES:
        s = summary[p]
        print(f"{p:<12}{s['mean_of_means']:>10.3f}+-{s['std_of_means']:<5.3f}"
              f"{s['mean_of_p5']:>12.3f}+-{s['std_of_p5']:<5.3f}"
              f"{s['mean_of_std']:>10.3f}")
    print("=" * 78)

    # Paired comparisons vs LCB baseline (pool trials across seeds, paired by
    # (seed, trial) index) for EU, Hurwicz(0.5), CVaR, DRO -- Cohen's d + p.
    def paired_stats(a, b):
        a = np.concatenate(a)
        b = np.concatenate(b)
        diff = a - b
        d = float(np.mean(diff) / np.std(diff, ddof=1))
        t, p = stats.ttest_rel(a, b)
        return d, float(p)

    print("\nPaired comparisons vs. Bayesian-LCB (pooled across seeds, n=2500):")
    for p in ["EU", "Hurwicz_0.5", "CVaR", "DRO"]:
        d, pval = paired_stats(all_trials[p], all_trials["LCB"])
        print(f"  {p:<12} vs LCB: d={d:+.3f}, p={pval:.3g}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/rq4_full_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
