"""
Budget-controlled BALANCED training set.
All 6 dialects equally represented within the same total budget as EXP2/EXP3,
to isolate the effect of distribution from the effect of dataset size.
"""
import os
import argparse
import numpy as np
import pandas as pd

import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS_WITH_SAE as DIALECTS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="./data/train.csv")
    p.add_argument("--total_budget", type=int, default=None,
                   help="total samples after subsampling. "
                        "Default = number of SAE samples in train_csv "
                        "(matches EXP2 SAE-only size)")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train = pd.read_csv(args.train_csv)
    if "dialect" not in train.columns:
        raise ValueError("train_csv has no `dialect` column.")

    sae_available = (train["dialect"] == "SAE").sum()
    total_budget  = args.total_budget or sae_available
    # Uniform over the 12 (dialect x label) groups, matching the 12-group
    # GroupDRO objective. Previously we balanced only the 6 dialects and left
    # the label split within each dialect to chance. Every group pool is large
    # here (>= 1994), so a full 12-group balance is feasible.
    n_per_group = total_budget // 12

    # capacity check
    for g in range(12):
        g_avail = (train["group"] == g).sum()
        if n_per_group > g_avail:
            raise ValueError(
                f"Need {n_per_group} samples for group {g}, only {g_avail} available."
            )

    rng = np.random.default_rng(args.seed)
    parts = []
    for g in range(12):
        g_rows = train[train["group"] == g]
        idx = rng.choice(len(g_rows), size=n_per_group, replace=False)
        parts.append(g_rows.iloc[idx])
    bal = pd.concat(parts, ignore_index=True)

    if args.output is None:
        args.output = f"./data/train_balanced_s{args.seed}.csv"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bal.to_csv(args.output, index=False)

    print(f"\n=== Balanced training set (budget-controlled) ===")
    print(f"  Total budget   : {total_budget:,}")
    print(f"  Per group (12) : {n_per_group:,}")
    print(f"  Actual total   : {len(bal):,}")
    print(f"  Saved to       : {args.output}")
    print(f"\nPer-dialect counts:")
    print(bal["dialect"].value_counts().reindex(DIALECTS).to_string())
    print(f"\nPer-group counts (g = 6*label + dialect_idx):")
    for g in range(12):
        n = (bal["group"] == g).sum()
        lab = "nsfw" if g >= 6 else "safe"
        dia = DIALECTS[g % 6]
        print(f"  g={g:2d}  ({lab:4s}, {dia:7s})  n={n:6d}")


if __name__ == "__main__":
    main()