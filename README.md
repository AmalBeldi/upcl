# UPCL — Unified Probabilistic Context Layer

Reproducibility / supplementary-material repository for the paper
**"A Unified Probabilistic Context Layer for Risk-Sensitive
Decision-Making Under State Uncertainty"** (anonymous submission).

> This repository is prepared for **double-blind review**. It contains no
> author names, no institutional affiliations, and no identifying commit
> history. It is intended to be browsed via an anonymized mirror such as
> [anonymous.4open.science](https://anonymous.4open.science).

## What this is

A reference implementation of the framework described in the paper:
the probabilistic context representation (TID / BID / hybrid semantics),
the transformation operator Φ, the generic aggregation operator Γ_θ
(Expected Utility, Maxmin, Hurwicz, CVaR, DRO), the decision operator Ψ,
the three integration strategies (UPCL-Pre, UPCL-Model, UPCL-Post), and
the Monte Carlo approximation with its finite-sample (Hoeffding) bound.

It exists to make every formal claim in the paper **checkable** — each
theorem/proposition has a corresponding unit test, and each worked example
in the text is reproduced exactly by a runnable script.

## Repository structure

```
upcl/                       Core library (importable, no dataset required)
  context.py                 Definitions 1-7, TID/BID/hybrid semantics, Phi
  aggregation.py              Gamma_theta: EU, Maxmin, Hurwicz, CVaR, DRO
  decision.py                 Psi: argmax decision operator
  instantiations.py           UPCL-Pre, UPCL-Model, UPCL-Post (Section 5)
  monte_carlo.py               Monte Carlo estimator + Hoeffding bound
  noise.py                     Controlled contextual noise injection (7.2)
  stats.py                     Paired significance testing (7.3)
  recommenders/toy.py           Reproducible synthetic recommenders
  datasets/loaders.py            Loaders/schema for real CARS benchmarks

experiments/
  example_alice.py                  Reproduces the paper's running example exactly
  synthetic_benchmark.py            Noise-robustness benchmark + significance tests
  scalability_analysis.py           Monte Carlo convergence, Corollary 5 check
  rq4_full_sweep.py                 RQ4 risk-sensitive aggregation, multi-seed
  rq4_hyperparam_sensitivity.py     CVaR/DRO hyperparameter sensitivity ablation
  prepare_real_dataset.py           Tidy-CSV preparation for any real CARS benchmark
  real_dataset_multiseed.py         UPCL-Post vs. deterministic, exact per-world pipeline
                                     (DePaulMovie, LDOS-CoMoDa)
  real_dataset_multiseed_fast.py    Same protocol, closed-form vectorization for
                                     single-context-variable benchmarks at scale (Frappe)
  frappe_model_evaluation.py        UPCL-Model (jointly-trained contextual bias) on Frappe
  mf_shrink_ablation.py             Sensitivity of the DePaulMovie result to MF capacity
                                     and residual shrinkage
  real_dataset_pipeline.py          Earlier, dataset-agnostic scaffold (generic preset
                                     runner, placeholder f0/h) -- superseded by the
                                     scripts above for every real-dataset number reported
                                     in the paper, kept here as a simpler starting point

outputs/                      Frozen result logs (JSON/txt) backing every reported
                               real-dataset and RQ4 number, so nothing needs to be
                               re-run to check a number in the paper

tests/                        One test module per core component
run_tests.py                  Dependency-free fallback runner (see below)
```

## Mapping from paper to code

| Paper element | Code |
|---|---|
| Def. 1-5 (contextual variable → probabilistic context space) | `upcl/context.py` |
| Def. 6-7, transformation Φ | `upcl.context.phi` |
| TID semantics (4.5) | `upcl.context._tid_worlds` |
| BID semantics (4.5) | `upcl.context._bid_worlds` |
| Prop. 3 / Prop. 4 (TID/BID normalization) | `tests/test_context.py` |
| Thm. 1 / Thm. 3 (probabilistic consistency) | `tests/test_context.py` (sum-to-one checks) |
| Def. 8-10, Γ_θ, Ψ | `upcl/aggregation.py`, `upcl/decision.py` |
| Section 5.1-5.3 (UPCL-Pre/Model/Post) | `upcl/instantiations.py` |
| Running example (3.1, 4.4, 5.1, 5.3) | `experiments/example_alice.py` |
| Thm. 2 (reduction to deterministic CARS) | `tests/test_instantiations.py::test_theorem2_reduction_to_deterministic` |
| Def. 11, Thm. 4 (Monte Carlo, convergence) | `upcl/monte_carlo.py`, `tests/test_monte_carlo.py` |
| Thm. 5, Cor. 5 (Hoeffding bound, sample size) | `upcl/monte_carlo.py`, `experiments/scalability_analysis.py` |
| Section 7.2 (noise injection) | `upcl/noise.py` |
| Section 7.3 (significance testing) | `upcl/stats.py` |
| Section 7.1 (dataset loading) | `upcl/datasets/loaders.py` |
| RQ4 -- risk-sensitive aggregation | `experiments/rq4_full_sweep.py`, `experiments/rq4_hyperparam_sensitivity.py` |
| Real-dataset results -- DePaulMovie, LDOS-CoMoDa | `experiments/real_dataset_multiseed.py` |
| Real-dataset results -- Frappe (UPCL-Post) | `experiments/real_dataset_multiseed_fast.py` |
| Real-dataset results -- Frappe (UPCL-Model) | `experiments/frappe_model_evaluation.py` |
| MF capacity / shrinkage sensitivity ablation | `experiments/mf_shrink_ablation.py` |

## Installation

```bash
pip install -r requirements.txt
```

No dataset is required to run the core library, the unit tests, or
`example_alice.py` / `synthetic_benchmark.py` / `scalability_analysis.py`.

## Running

```bash
# Unit tests (use pytest if available; a dependency-free fallback is included)
pytest tests/ -v
# or, if pytest is unavailable in your environment:
python run_tests.py

# Reproduce the paper's worked example exactly
python experiments/example_alice.py

# Synthetic robustness benchmark under controlled contextual noise
python experiments/synthetic_benchmark.py --n-users 50 --n-items 20

# Monte Carlo scalability analysis (Section 7.4 / Corollary 5)
python experiments/scalability_analysis.py --k-vars 6 --domain-size 4
```

## Real-dataset results (Section 7.4 / 8 / Appendix B.1)

This repository does not redistribute any dataset (`data/*.csv` is
git-ignored). The frozen logs in `outputs/` let every reported number be
checked without re-running anything; the commands below reproduce them
from scratch given the raw benchmark.

1. Obtain the raw benchmark from its official source: DePaulMovie
   (Zheng, Burke & Mobasher, 2014), Frappe-x1 (Baltrunas et al., 2015),
   or LDOS-CoMoDa (Odić et al., 2013 -- requires requesting access from
   its maintainers at the University of Ljubljana).
2. Convert it to the tidy `(user_id, item_id, rating, <context columns>)`
   schema with `experiments/prepare_real_dataset.py` (see its docstring
   for the exact flags per benchmark).
3. Run the matching protocol, from inside `experiments/`:
   ```bash
   # UPCL-Post vs. deterministic, DePaulMovie (3 context variables)
   python real_dataset_multiseed.py --data ../data/depaulmovie_tidy_clean.csv \
       --context-cols Time Location Companion --seeds 20 \
       --out ../outputs/depaulmovie_multiseed_log.json

   # UPCL-Post vs. deterministic, LDOS-CoMoDa (1 context variable, small/sparse)
   python real_dataset_multiseed.py --data ../data/ldos_comoda_full.csv \
       --context-cols mood --seeds 10 \
       --out ../outputs/ldos_comoda_multiseed_log.json

   # UPCL-Post vs. deterministic, Frappe (1 context variable, large catalog --
   # closed-form vectorization, see the script's docstring for why)
   python real_dataset_multiseed_fast.py --data ../data/frappe_prepared.csv \
       --context-col isweekend --eval-positive-only --seeds 20 \
       --out ../outputs/frappe_multiseed_log.json

   # UPCL-Model (jointly-trained contextual bias) vs. non-contextual/
   # deterministic baselines, Frappe
   python frappe_model_evaluation.py --data ../data/frappe_prepared.csv \
       --context-col isweekend --seeds 10 \
       --out ../outputs/frappe_model_evaluation_log.json

   # MF-capacity / shrinkage sensitivity ablation, DePaulMovie
   python mf_shrink_ablation.py
   ```

`experiments/real_dataset_pipeline.py` is an earlier, dataset-agnostic
scaffold (generic `--preset` runner with a placeholder popularity-based
`f0` and identity adjustment) kept for a simpler starting point; it is
not what produced any number reported in the paper.

## Status

The synthetic core (representation, aggregation, the three
instantiations, Monte Carlo guarantees) is complete and fully tested.
The real-dataset evaluation is complete for three benchmarks of varying
size and context density (DePaulMovie, Frappe, LDOS-CoMoDa) and for both
learned integration strategies (UPCL-Post, UPCL-Model); results are
reported in the paper exactly as obtained, including a mixed/negative
one (UPCL-Model's jointly-trained contextual bias overfits on Frappe
relative to a non-contextual baseline -- see the paper's Discussion and
Appendix B.1.9 for the full discussion).

## License

Released under the MIT License (see `LICENSE`), with no named copyright
holder for the duration of double-blind review.
