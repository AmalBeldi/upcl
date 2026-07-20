# UPCL — Unified Probabilistic Context Layer

Reproducibility / supplementary-material repository for the paper
**"Towards a Unified Probabilistic Framework for Uncertainty-Aware Context
Modeling in Recommender Systems"** (anonymous submission).

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
  example_alice.py             Reproduces the paper's running example exactly
  synthetic_benchmark.py       Noise-robustness benchmark + significance tests
  scalability_analysis.py      Monte Carlo convergence, Corollary 5 check
  real_dataset_pipeline.py     Extension point for a real CARS dataset

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

## Extending to a real dataset (Section 7.1)

This repository does not redistribute any dataset. To run the full
protocol on a real context-aware recommendation benchmark (e.g.
LDOS-CoMoDa, Frappe):

1. Download the benchmark from its official source.
2. Convert it to the tidy CSV schema documented in
   `upcl/datasets/loaders.py` (one row per interaction; context columns
   may be a plain categorical value or a JSON-encoded probability
   distribution if you have externally estimated contextual uncertainty).
3. Place the CSV under `data/` (git-ignored).
4. Run:
   ```bash
   python experiments/real_dataset_pipeline.py \
       --preset ldos_comoda --data-path data/ldos_comoda.csv
   ```
5. Replace the placeholder popularity-based `f0` and identity adjustment
   `h` in `real_dataset_pipeline.py` with a trained recommender and a
   learned contextual adjustment function for a non-trivial evaluation.

## Status

The synthetic core (representation, aggregation, the three
instantiations, Monte Carlo guarantees) is complete and fully tested. The
real-dataset pipeline is a working scaffold: it runs end-to-end against
any correctly-formatted CSV, but the base recommender and post-processing
adjustment are intentionally minimal placeholders pending the full
empirical evaluation described in Section 7.

## License

Released under the MIT License (see `LICENSE`), with no named copyright
holder for the duration of double-blind review.
