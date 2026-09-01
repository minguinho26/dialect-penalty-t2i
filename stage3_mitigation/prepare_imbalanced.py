"""
Experiment 3: SAE-dominant imbalanced training set with FIXED total budget.
Subsamples training data to a fixed total size, with a tunable SAE/non-SAE ratio,
to enable controlled comparison with Experiment 2 (SAE-only) and across imbalance levels.

Usage:
    python prepare_imbalanced.py --sae_frac 0.95
    python prepare_imbalanced.py --sae_frac 0.99 --total_budget 3692
    python prepare_imbalanced.py --sae_frac 0.95 --output ./data/train_imb_95.csv
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
    p.add_argument("--train_csv", default="./data/train.csv",
                   help="long-format train CSV produced by prepare_data.py")
    p.add_argument("--sae_frac", type=float, required=True,
                   help="proportion of SAE within total budget (e.g. 0.95)")
    p.add_argument("--total_budget", type=int, default=None,
                   help="total samples after subsampling. "
                        "Default = number of SAE samples in train_csv "
                        "(matches Experiment 2 SAE-only size)")
    p.add_argument("--output", default=None,
                   help="output CSV path. Default: ./data/train_imb_{frac}.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not (0.0 < args.sae_frac < 1.0):
        raise ValueError(f"--sae_frac must be in (0, 1), got {args.sae_frac}")

    # ---------- load ----------
    train = pd.read_csv(args.train_csv)
    if "dialect" not in train.columns:
        raise ValueError("train_csv has no `dialect` column. "
                         "Re-run prepare_data.py to regenerate.")

    # ---------- determine budget ----------
    sae_available = (train["dialect"] == "SAE").sum()
    total_budget  = args.total_budget or sae_available

    n_sae        = int(round(total_budget * args.sae_frac))
    n_nonsae     = total_budget - n_sae
    # Stratify the non-SAE budget over the 10 (dialect x label) groups, not just
    # the 5 dialects. GroupDRO optimizes over the 12 (dialect x label) cells, so
    # balancing only by dialect (label-blind) left the label split within each
    # dialect to chance and could leave whole (dialect, label) groups empty at
    # extreme SAE fractions. We therefore draw an equal count from each non-SAE
    # group. SAE stays pooled across labels: its pool is not 50/50 (2160 benign
    # vs 1994 toxic) and n_sae exceeds the toxic-SAE pool at high fractions, so a
    # 50/50 SAE split is infeasible; both SAE groups stay well populated anyway.
    n_nonsae_per_group = n_nonsae // (2 * (len(DIALECTS) - 1))   # 10 non-SAE groups

    # capacity check
    if n_sae > sae_available:
        raise ValueError(
            f"Requested {n_sae} SAE samples but only {sae_available} available. "
            f"Lower --total_budget or --sae_frac."
        )
    nonsae_groups = [g for g in range(12) if g % len(DIALECTS) != 0]
    for g in nonsae_groups:
        g_available = (train["group"] == g).sum()
        if n_nonsae_per_group > g_available:
            raise ValueError(
                f"Requested {n_nonsae_per_group} samples for group {g} "
                f"but only {g_available} available."
            )

    # ---------- subsample ----------
    rng = np.random.default_rng(args.seed)
    parts = []
    # SAE: pooled across labels (majority reference; both label groups large)
    sae_rows = train[train["dialect"] == "SAE"]
    sae_idx  = rng.choice(len(sae_rows), size=n_sae, replace=False)
    parts.append(sae_rows.iloc[sae_idx])
    # non-SAE: equal draw from each (dialect, label) group -> no empty groups
    for g in nonsae_groups:
        g_rows = train[train["group"] == g]
        idx    = rng.choice(len(g_rows), size=n_nonsae_per_group, replace=False)
        parts.append(g_rows.iloc[idx])
    imb = pd.concat(parts, ignore_index=True)

    # ---------- save ----------
    if args.output is None:
        # e.g. 0.95 -> "095",  0.99 -> "099",  0.995 -> "0995"
        tag = f"{args.sae_frac:.3f}".replace(".", "")
        args.output = f"./data/train_imb_{tag}.csv"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    imb.to_csv(args.output, index=False)

    # ---------- report ----------
    print(f"\n=== Imbalanced training set ===")
    print(f"  Total budget   : {total_budget:,}")
    print(f"  SAE fraction   : {args.sae_frac:.4f}")
    print(f"  SAE samples    : {n_sae:,}")
    print(f"  per non-SAE grp: {n_nonsae_per_group:,}  (× 10 groups = {n_nonsae_per_group*10:,})")
    print(f"  Actual total   : {len(imb):,}")
    print(f"  Saved to       : {args.output}")

    print(f"\nPer-dialect counts:")
    print(imb["dialect"].value_counts().reindex(DIALECTS).to_string())

    print(f"\nPer-group counts (g = 6*label + dialect_idx):")
    for g in range(12):
        n = (imb["group"] == g).sum()
        lab = "nsfw" if g >= 6 else "safe"
        dia = DIALECTS[g % 6]
        print(f"  g={g:2d}  ({lab:4s}, {dia:7s})  n={n:6d}")


if __name__ == "__main__":
    main()