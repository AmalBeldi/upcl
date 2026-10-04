#!/usr/bin/env python3
"""
Prepare a tidy real-dataset CSV for real_dataset_multiseed.py.

This is the missing counterpart to Appendix B's "prepare_depaulmovie_dataset.py":
it takes a raw benchmark export, drops rows with unrecorded context, and
writes a clean (user_id, item_id, rating, <context cols as small integer
codes>) CSV that upcl.datasets.loaders.load_csv_dataset can read directly.

Works for any CARS benchmark, not just one dataset -- pass --context-cols
to select which columns are the contextual (BID) blocks. For DePaulMovie,
that would be e.g. --context-cols Time Location Companion; for LDOS-CoMoDa,
e.g. --context-cols mood weather social (or a subset, see --k-core below).

USAGE
-----
    python experiments/prepare_real_dataset.py \
        --raw data/ldos_comoda_raw.csv \
        --user-col userID --item-col itemID --rating-col rating \
        --context-cols mood \
        --missing-code -1 \
        --k-core 3 \
        --out data/ldos_comoda_prepared.csv
"""

import argparse

import pandas as pd


def k_core_filter(df: pd.DataFrame, k: int, id_cols, max_iters: int = 10) -> pd.DataFrame:
    sub = df.copy()
    for _ in range(max_iters):
        before = len(sub)
        for col in id_cols:
            counts = sub[col].value_counts()
            sub = sub[sub[col].isin(counts[counts >= k].index)]
        if len(sub) == before:
            break
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--context-cols", nargs="+", required=True)
    parser.add_argument("--missing-code", default="-1",
                         help="Value marking 'no recorded context' (string or int).")
    parser.add_argument("--k-core", type=int, default=1,
                         help="Iteratively drop items/users with fewer than k "
                              "interactions before dropping missing-context rows. "
                              "1 = no filtering.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.raw)
    n_raw = len(df)

    try:
        missing_code = int(args.missing_code)
    except ValueError:
        missing_code = args.missing_code

    if args.k_core > 1:
        df = k_core_filter(df, args.k_core, [args.item_col, args.user_col])
        print(f"{args.k_core}-core filtering: {len(df)}/{n_raw} rows kept "
              f"({len(df)/n_raw:.1%}), {df[args.item_col].nunique()} items, "
              f"{df[args.user_col].nunique()} users.")

    n_before_context = len(df)
    for c in args.context_cols:
        df = df[df[c].astype(str).str.strip() != str(missing_code)]
    n_dropped = n_before_context - len(df)
    print(f"Dropped {n_dropped} rows "
          f"({n_dropped/n_before_context if n_before_context else 0:.1%}) "
          f"with no recorded context in {args.context_cols}.")

    out = df[[args.user_col, args.item_col, args.rating_col] + args.context_cols].copy()
    out.columns = ["user_id", "item_id", "rating"] + args.context_cols

    for c in args.context_cols:
        codes, uniques = pd.factorize(out[c], sort=True)
        out[c] = codes
        print(f"  Context block '{c}': {len(uniques)} distinct values.")

    worlds = out[args.context_cols].drop_duplicates()
    density = len(out) / (out["item_id"].nunique() * len(worlds))
    print(f"Final: {len(out)} rows, {out['item_id'].nunique()} items, "
          f"{out['user_id'].nunique()} users, {len(worlds)} distinct worlds, "
          f"avg {density:.2f} rows per (item, world) cell.")

    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
