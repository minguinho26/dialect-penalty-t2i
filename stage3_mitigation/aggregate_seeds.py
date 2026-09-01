"""
Aggregate seed runs.
Usage:
    python aggregate_seeds.py --exp exp2   # EXP2: SAE-only ERM
    python aggregate_seeds.py --exp exp3   # EXP3: imbalanced ERM vs DRO at 0.99
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

SEEDS    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS_WITH_SAE as DIALECTS

KEY_METRICS = (
    ["test_accuracy", "test_worst_group_acc",
     "test_TPR_spread", "test_FPR_spread", "test_gap_SAE_nonSAE"]
    + [f"test_TPR_{d}" for d in DIALECTS]
    + [f"test_FPR_{d}" for d in DIALECTS]
)

EXPERIMENTS = {
    "exp2": {
        "label":   "EXP2 SAE-only ERM",
        "methods": {
            "sae_only_erm": "./results/seeds_exp2/sae_only_erm_s{seed}/test_metrics.json",
        },
    },
    "exp3": {
        "label":   "EXP3 (budget-matched)",
        "methods": {
            "erm_balanced":   "./results/seeds_exp3/erm_balanced_s{seed}/test_metrics.json",
            "erm_imbalanced": "./results/seeds_exp3/erm_0990_s{seed}/test_metrics.json",
            "dro_imbalanced": "./results/seeds_exp3/dro_0990_s{seed}/test_metrics.json",
        },
    },
}


def load(path_template, seed):
    path = Path(path_template.format(seed=seed))
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def aggregate(exp_key):
    exp = EXPERIMENTS[exp_key]
    print(f"\n=== {exp['label']}  |  seeds={SEEDS} ===\n")

    data = {m: [load(p, s) for s in SEEDS]
            for m, p in exp["methods"].items()}

    rows = []
    for k in KEY_METRICS:
        row = {"metric": k}
        for m in exp["methods"]:
            vals = [r[k] for r in data[m] if r is not None and k in r]
            if vals:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"]  = float(np.std(vals))
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"]  = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("metric")
    out_csv = f"./results/aggregate_{exp_key}.csv"
    df.to_csv(out_csv)

    # pretty print
    methods = list(exp["methods"].keys())
    header  = f"{'metric':<25s}"
    for m in methods:
        header += f" | {m.upper():<22s}"
    print(header)
    print("-" * len(header))

    for k in KEY_METRICS:
        line = f"{k:<25s}"
        for m in methods:
            mean = df.loc[k, f"{m}_mean"]
            std  = df.loc[k, f"{m}_std"]
            line += f" | {mean:.4f} ± {std:.4f}    "
        print(line)

    print(f"\n--- per-seed worst_group_acc ---")
    for m in methods:
        vals = [r["test_worst_group_acc"] for r in data[m] if r]
        print(f"  {m.upper():15s}:  {[f'{v:.4f}' for v in vals]}   "
              f"(mean={np.mean(vals):.4f}, std={np.std(vals):.4f})")

    print(f"\n[saved] {out_csv}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()))
    args = p.parse_args()
    aggregate(args.exp)