"""
Aggregate EXP1, EXP2, EXP3 results into a single CSV.

EXP1: NSFW-T zero-shot (single deterministic run)
EXP2: SAE-only ERM (3 seeds aggregated)
EXP3: Imbalanced ERM and DRO at SAE_frac=0.99 (3 seeds aggregated)

Outputs:
  ./results/all_experiments.csv           — long format, one row per (metric, exp, stat)
  ./results/all_experiments_wide.csv      — wide format, one row per metric, columns per (exp, stat)
  ./results/all_experiments_pretty.csv    — paper-ready, "mean ± std" string per cell
"""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS_WITH_SAE as DIALECTS
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

KEY_METRICS = (
    ["test_accuracy", "test_worst_group_acc", "test_mean_group_acc",
     "test_TPR_spread", "test_FPR_spread", "test_gap_SAE_nonSAE",
     "test_acc_SAE", "test_acc_nonSAE"]
    + [f"test_TPR_{d}" for d in DIALECTS]
    + [f"test_FPR_{d}" for d in DIALECTS]
    + [f"test_acc_{d}" for d in DIALECTS]
    + [f"test_dTPR_{d}" for d in DIALECTS if d != "SAE"]
    + [f"test_dFPR_{d}" for d in DIALECTS if d != "SAE"]
    + ["test_max_dFPR", "test_min_dTPR", "test_mean_abs_dFPR", "test_mean_abs_dTPR"]
)


def ensure_delta_metrics(run):
    """
    Derive dTPR/dFPR from raw TPR/FPR if they are missing in test_metrics.json.
    This allows generating ΔTPR/ΔFPR tables for older results without retraining.
    """
    if run is None or "test_TPR_SAE" not in run:
        return run
    if "test_dTPR_AAVE" in run:
        return run
    dt, df = [], []
    for d in DIALECTS:
        if d == "SAE":
            continue
        if f"test_TPR_{d}" in run:
            run[f"test_dTPR_{d}"] = run[f"test_TPR_{d}"] - run["test_TPR_SAE"]; dt.append(run[f"test_dTPR_{d}"])
        if f"test_FPR_{d}" in run:
            run[f"test_dFPR_{d}"] = run[f"test_FPR_{d}"] - run["test_FPR_SAE"]; df.append(run[f"test_dFPR_{d}"])
    if dt:
        run["test_min_dTPR"] = float(min(dt)); run["test_mean_abs_dTPR"] = float(np.mean(np.abs(dt)))
    if df:
        run["test_max_dFPR"] = float(max(df)); run["test_mean_abs_dFPR"] = float(np.mean(np.abs(df)))
    return run


def load_single_run(path):
    """Load EXP1 (single deterministic run, no seeds)."""
    p = Path(path)
    if not p.exists():
        print(f"[skip] {path} not found")
        return None
    with open(p) as f:
        return json.load(f)


def load_seed_runs(path_template, seeds=SEEDS):
    """Load multi-seed runs and return list of dicts."""
    runs = []
    for s in seeds:
        p = Path(path_template.format(seed=s))
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def stats_from_runs(runs, metric):
    """Compute mean and std over seed runs for one metric."""
    vals = [r[metric] for r in runs if r is not None and metric in r]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1_path",
                    default="./results/exp1_nsfwt_zeroshot/test_metrics.json")
    ap.add_argument("--exp2_template",
                    default="./results/seeds_exp2/sae_only_erm_s{seed}/test_metrics.json")
    ap.add_argument("--exp3_erm_template",
                    default="./results/sweep_ratio/erm_0990_s{seed}/test_metrics.json")
    ap.add_argument("--exp3_dro_template",
                    default="./results/sweep_ratio/dro_0990_s{seed}/test_metrics.json")
    ap.add_argument("--exp3_erm_balanced_template",
                default="./results/seeds_exp3/erm_balanced_s{seed}/test_metrics.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                    help="List of seeds to aggregate. Default is 0-9.")
    ap.add_argument("--output_dir", default="./results")
    args = ap.parse_args()

    if args.seeds:
        # load_seed_runs captures SEEDS reference at definition time,
        # so we must modify it in-place rather than reassigning.
        SEEDS[:] = args.seeds

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp1              = ensure_delta_metrics(load_single_run(args.exp1_path))
    exp2_runs         = [ensure_delta_metrics(r) for r in load_seed_runs(args.exp2_template)]
    exp3_bal_erm_runs = [ensure_delta_metrics(r) for r in load_seed_runs(args.exp3_erm_balanced_template)]   # NEW
    exp3_erm_runs     = [ensure_delta_metrics(r) for r in load_seed_runs(args.exp3_erm_template)]
    exp3_dro_runs     = [ensure_delta_metrics(r) for r in load_seed_runs(args.exp3_dro_template)]

    columns = [
        ("EXP1_NSFW-T_zeroshot",     "single", exp1,              None),
        ("EXP2_SAE-only_ERM",        "seeds",  exp2_runs,         None),
        ("EXP3_ERM_balanced",        "seeds",  exp3_bal_erm_runs, None),  # NEW
        ("EXP3_ERM_imbalanced",      "seeds",  exp3_erm_runs,     None),
        ("EXP3_DRO_imbalanced",      "seeds",  exp3_dro_runs,     None),
    ]

    # =====================================================
    # 1. WIDE format: one row per metric
    # =====================================================
    wide_rows = []
    for metric in KEY_METRICS:
        row = {"metric": metric}
        for col_name, kind, runs, _ in columns:
            if kind == "single":
                row[f"{col_name}_value"] = (
                    runs.get(metric) if runs is not None else None
                )
                row[f"{col_name}_std"] = None
            else:  # seeds
                mean, std = stats_from_runs(runs, metric)
                row[f"{col_name}_mean"] = mean
                row[f"{col_name}_std"]  = std
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows).set_index("metric")
    wide_df.to_csv(out_dir / "all_experiments_wide.csv")
    print(f"[saved] {out_dir / 'all_experiments_wide.csv'}")

    # =====================================================
    # 2. LONG format: one row per (metric, experiment)
    # =====================================================
    long_rows = []
    for metric in KEY_METRICS:
        for col_name, kind, runs, _ in columns:
            row = {"metric": metric, "experiment": col_name}
            if kind == "single":
                row["mean"] = runs.get(metric) if runs is not None else None
                row["std"]  = None
                row["n_seeds"] = 1
            else:
                mean, std = stats_from_runs(runs, metric)
                row["mean"]    = mean
                row["std"]     = std
                row["n_seeds"] = sum(1 for r in runs if r is not None)
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "all_experiments_long.csv", index=False)
    print(f"[saved] {out_dir / 'all_experiments_long.csv'}")

    # =====================================================
    # 3. PRETTY format: "mean ± std" strings
    # =====================================================
    def fmt(mean, std):
        if mean is None:
            return ""
        if std is None or std == 0:    # single run or no variance
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    pretty_rows = []
    for metric in KEY_METRICS:
        row = {"metric": metric}
        for col_name, kind, runs, _ in columns:
            if kind == "single":
                v = runs.get(metric) if runs is not None else None
                row[col_name] = fmt(v, None)
            else:
                mean, std = stats_from_runs(runs, metric)
                row[col_name] = fmt(mean, std)
        pretty_rows.append(row)

    pretty_df = pd.DataFrame(pretty_rows).set_index("metric")
    pretty_df.to_csv(out_dir / "all_experiments_pretty.csv")
    print(f"[saved] {out_dir / 'all_experiments_pretty.csv'}")

    # =====================================================
    # 4. Console preview
    # =====================================================
    print("\n" + "=" * 100)
    print("CONSOLIDATED RESULTS")
    print("=" * 100)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 30,
                           "display.width", 200):
        print(pretty_df.to_string())


if __name__ == "__main__":
    main()