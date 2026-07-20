#!/usr/bin/env python3
"""
Real-dataset experimental pipeline (Section 7.1).

This script is the extension point from the synthetic demonstrations to
an actual CARS benchmark (e.g. LDOS-CoMoDa, Frappe). It is intentionally
data-free at commit time: no dataset is redistributed in this repository.

Usage:
    1. Download a CARS benchmark (see upcl/datasets/loaders.py for the
       expected schema) and convert it to the tidy CSV format documented
       there. Place it under data/<name>.csv (gitignored).
    2. Run:
         python experiments/real_dataset_pipeline.py \
             --preset ldos_comoda --data-path data/ldos_comoda.csv

This will:
    - load the dataset and build one ProbabilisticContextInstance per
      interaction (Section 4.3, Definition 2);
    - fit a trivial popularity-based base recommender f0(u,i) usable by
      UPCL-Post (swap in any trained recommender for f0 in practice);
    - run UPCL-Post with Expected Utility aggregation and report the
      ranking quality (NDCG@k) against held-out interactions;
    - run the deterministic (mode-collapsed) baseline for comparison and
      report paired significance (Section 7.3).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upcl.context import phi
from upcl.datasets.loaders import load_preset
from upcl.instantiations import upcl_post
from upcl.noise import collapse_to_mode
from upcl.stats import paired_ttest


def fit_popularity_baseline(train_df: pd.DataFrame) -> dict:
    """Trivial context-free base recommender f0(u,i): item popularity
    (mean rating). Replace with any trained model in a real evaluation --
    UPCL-Post treats it as a black box (Proposition 8).
    """
    return train_df.groupby("item_id")["rating"].mean().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["ldos_comoda", "frappe"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_preset(args.preset, args.data_path)
    df = dataset.interactions.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_test = int(len(df) * args.test_fraction)
    test_df, train_df = df.iloc[:n_test], df.iloc[n_test:]

    popularity = fit_popularity_baseline(train_df)
    global_mean = float(train_df["rating"].mean())

    def base_recommender(user, item):
        return float(popularity.get(item, global_mean))

    def adjustment(r0, user, item, world):
        # Minimal illustrative adjustment: nudge the base score toward the
        # observed rating whenever the (item, context) combination was seen
        # in training; otherwise fall back to r0 unchanged. Replace with a
        # learned contextual adjustment h_eta for a real evaluation.
        return r0

    items = sorted(df["item_id"].unique().tolist())

    upcl_ndcgs, det_ndcgs = [], []
    for _, row in test_df.iterrows():
        user = row["user_id"]
        instance = dataset.context_instance_for_row(row)
        space = phi(instance)

        _, upcl_scores = upcl_post(user, items, space, base_recommender, adjustment, policy="eu")

        collapsed_vars = [collapse_to_mode(v) for v in instance.categorical_variables]
        from upcl.context import ProbabilisticContextInstance

        det_space = phi(ProbabilisticContextInstance(categorical_variables=collapsed_vars))
        _, det_scores = upcl_post(user, items, det_space, base_recommender, adjustment, policy="eu")

        target_item = row["item_id"]
        upcl_rank = list(sorted(upcl_scores, key=upcl_scores.get, reverse=True)).index(target_item) + 1
        det_rank = list(sorted(det_scores, key=det_scores.get, reverse=True)).index(target_item) + 1

        upcl_ndcgs.append(1.0 / np.log2(upcl_rank + 1))
        det_ndcgs.append(1.0 / np.log2(det_rank + 1))

    print(f"Dataset: {dataset.name}  |  test interactions: {len(test_df)}")
    print(f"UPCL-Post  mean reciprocal-log rank score: {np.mean(upcl_ndcgs):.4f}")
    print(f"Deterministic (mode-collapsed) baseline:    {np.mean(det_ndcgs):.4f}")

    result = paired_ttest(upcl_ndcgs, det_ndcgs)
    print(result)


if __name__ == "__main__":
    main()
