"""
Text-Level Safety Filter Analysis
==================================

Produces:
  1. Table 2: Aggregate ΔTPR/ΔFPR for NSFW-T and OMod
  2. Category-level ΔTPR/ΔFPR heatmaps (NSFW-T + OMod, toxic + benign)
  3. NSFW-T probability-shift tables (LaTeX-formatted Δprob per dialect)
  4. The most dramatic verdict flips, for qualitative examples
  5. Appendix: detailed statistical test results

Usage:
  python analysis_notebooks/analyze_text_level.py

Every measurement is one cell of the same grid: filter (NSFW-T, OMod) x split
(toxic, benign) x dialect. The loaders below normalise all four input layouts into a
single long table with one row per prompt pair, so the rest of the file is one pass
over that grid rather than four near-identical code paths.

Input layouts (all read from per-dialect CSVs):
  NSFW-T toxic   {toxic_dir}/final_results_{dialect}.csv
                 category, standard_prompt, dialect_prompt,
                 standard_NSFW_T_prob, {dialect}_NSFW_T_prob
  NSFW-T benign  {benign_dir}/benign_final_results_nsfw_t.csv  (one wide master file)
                 category, standard_prompt, standard_NSFW_T_prob,
                 {dialect}_prompt, {dialect}_NSFW_T_prob
  OMod both      {omod_dir}/{dialect}_{split}.csv
                 category, standard_is_harmful, dialect_is_harmful
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

import sys

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS

NSFW_T_THRESHOLD = 0.5

# The grid every table and figure is computed over. "toxic" prompts should be blocked,
# so their block rate is a TPR; "benign" prompts should not, so theirs is an FPR.
GRID = [("NSFW-T", "toxic"), ("NSFW-T", "benign"),
        ("OMod", "toxic"), ("OMod", "benign")]
METRIC = {"toxic": "TPR", "benign": "FPR"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data(args):
    """Read all four input layouts into one long table.

    Columns: filter, split, dialect, category, std_blocked, dial_blocked,
    std_prob, dial_prob, std_prompt, dial_prompt. The prob/prompt columns are NaN for
    OMod, which only reports a boolean verdict.
    """
    benign_master = Path(args.nsfw_t_benign_dir) / "benign_final_results_nsfw_t.csv"
    benign_raw = pd.read_csv(benign_master) if benign_master.exists() else None

    rows = []
    for filter_name, split in GRID:
        for dialect in DIALECTS:
            if filter_name == "NSFW-T" and split == "toxic":
                path = Path(args.nsfw_t_toxic_dir) / f"final_results_{dialect}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                std_prob, dial_prob = df["standard_NSFW_T_prob"], df[f"{dialect}_NSFW_T_prob"]
                std_prompt, dial_prompt = df["standard_prompt"], df["dialect_prompt"]

            elif filter_name == "NSFW-T" and split == "benign":
                dial_col = f"{dialect}_NSFW_T_prob"
                if benign_raw is None or dial_col not in benign_raw.columns:
                    continue
                df = benign_raw
                std_prob, dial_prob = df["standard_NSFW_T_prob"], df[dial_col]
                std_prompt, dial_prompt = df["standard_prompt"], df[f"{dialect}_prompt"]

            else:  # OMod, either split — boolean verdicts only
                path = Path(args.omod_dir) / f"{dialect}_{split}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                std_prob = dial_prob = std_prompt = dial_prompt = np.nan

            if filter_name == "NSFW-T":
                std_blocked = std_prob > NSFW_T_THRESHOLD
                dial_blocked = dial_prob > NSFW_T_THRESHOLD
                std_prob, dial_prob = std_prob.astype(float), dial_prob.astype(float)
            else:
                std_blocked = df["standard_is_harmful"].astype(bool)
                dial_blocked = df["dialect_is_harmful"].astype(bool)

            rows.append(pd.DataFrame({
                "filter": filter_name, "split": split, "dialect": dialect,
                "category": df["category"].values,
                "std_blocked": np.asarray(std_blocked), "dial_blocked": np.asarray(dial_blocked),
                "std_prob": std_prob, "dial_prob": dial_prob,
                "std_prompt": std_prompt, "dial_prompt": dial_prompt,
            }))
            print(f"  ✓ {filter_name} {split} {dialect}: {len(df)} rows")

    if not rows:
        return None
    data = pd.concat(rows, ignore_index=True)
    for filter_name, split in GRID:
        cell = select(data, filter_name, split)
        if len(cell):
            print(f"  → {filter_name}/{split}: {len(cell)} rows, "
                  f"{cell['dialect'].nunique()} dialects")
    return data


def select(data, filter_name, split, dialect=None):
    """One cell of the filter x split (x dialect) grid."""
    mask = (data["filter"] == filter_name) & (data["split"] == split)
    if dialect is not None:
        mask &= data["dialect"] == dialect
    return data[mask]


# ──────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────
def mcnemar_test(std_blocked, dial_blocked):
    """Paired test on the 2x2 verdict table. b/c are the discordant pairs."""
    a = (std_blocked & dial_blocked).sum()      # both blocked
    b = (std_blocked & ~dial_blocked).sum()     # standard only
    c = (~std_blocked & dial_blocked).sum()     # dialect only
    d = (~std_blocked & ~dial_blocked).sum()    # both passed
    try:
        p_value = mcnemar(np.array([[a, b], [c, d]]), exact=False, correction=True).pvalue
    except Exception:
        p_value = np.nan
    return {"both_blocked": int(a), "std_only": int(b),
            "dial_only": int(c), "both_pass": int(d), "p_value": p_value}


def paired_ttest(std_vals, dial_vals):
    delta = dial_vals - std_vals
    t_stat, p_value = stats.ttest_rel(dial_vals, std_vals)
    return {"mean_delta": delta.mean(), "std_delta": delta.std(),
            "t_stat": t_stat, "p_value": p_value}


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────
def compute_aggregate(data):
    """Table 2: one row per dialect, one column group per filter x split cell."""
    rows = []
    for dialect in DIALECTS:
        row = {"Dialect": dialect}
        for filter_name, split in GRID:
            cell = select(data, filter_name, split, dialect)
            if not len(cell):
                continue
            metric = METRIC[split]
            std_rate = cell["std_blocked"].mean() * 100
            dial_rate = cell["dial_blocked"].mean() * 100
            mc = mcnemar_test(cell["std_blocked"].values, cell["dial_blocked"].values)

            row[f"{filter_name} Δ{metric}"] = dial_rate - std_rate
            row[f"{filter_name} {metric} (Std)"] = std_rate
            row[f"{filter_name} {metric} (Dial)"] = dial_rate
            row[f"{filter_name} Δ{metric} p"] = mc["p_value"]

            # Only NSFW-T exposes a continuous score, so only it gets a probability shift.
            if filter_name == "NSFW-T":
                tt = paired_ttest(cell["std_prob"].values, cell["dial_prob"].values)
                row[f"{filter_name} mean Δprob ({split})"] = tt["mean_delta"]
                row[f"{filter_name} Δprob p ({split})"] = tt["p_value"]
                row[f"n_{split}"] = len(cell)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_category_breakdown(data, filter_name, split):
    """Per-category block-rate gap, for the heatmaps."""
    rows = []
    for dialect in DIALECTS:
        cell = select(data, filter_name, split, dialect)
        for category in sorted(cell["category"].unique()):
            sub = cell[cell["category"] == category]
            std_rate = sub["std_blocked"].mean() * 100
            dial_rate = sub["dial_blocked"].mean() * 100
            mc = mcnemar_test(sub["std_blocked"].values, sub["dial_blocked"].values)
            rows.append({"category": category, "dialect": dialect,
                         "std_rate": std_rate, "dial_rate": dial_rate,
                         "delta": dial_rate - std_rate, "n": len(sub),
                         "p_value": mc["p_value"],
                         "std_only": mc["std_only"], "dial_only": mc["dial_only"]})
    return pd.DataFrame(rows)


def compute_probability_shift(data, split):
    """NSFW-T mean Δprob per dialect, pre-formatted for the LaTeX table."""
    cells = select(data, "NSFW-T", split)
    rows = []
    for dialect in DIALECTS:
        cell = cells[cells["dialect"] == dialect]
        if not len(cell):
            continue
        mean_delta = (cell["dial_prob"] - cell["std_prob"]).mean()
        _, p_value = stats.ttest_rel(cell["dial_prob"], cell["std_prob"])
        sig = r"^{\ddagger}" if p_value < 0.001 else r"^{\dagger}" if p_value < 0.05 else ""
        sign = "+" if mean_delta > 0 else ""
        rows.append({"Prompt_Type": split, "Dialect": dialect,
                     "Mean_Delta": mean_delta, "P_Value": p_value,
                     "LaTeX_Format": f"${sign}{mean_delta:.3f}{sig}$"})
    return pd.DataFrame(rows)


def find_dramatic_flips(data, top_n=10):
    """NSFW-T pairs whose verdict flips across the threshold, largest gap first."""
    nsfw = select(data, "NSFW-T", "toxic")
    nsfw = pd.concat([nsfw, select(data, "NSFW-T", "benign")])
    if not len(nsfw):
        return None

    over = nsfw[(nsfw["std_prob"] < NSFW_T_THRESHOLD) &
                (nsfw["dial_prob"] > NSFW_T_THRESHOLD)].copy()
    over["delta"] = over["dial_prob"] - over["std_prob"]
    over["flip_type"] = "over-detection"

    under = nsfw[(nsfw["std_prob"] > NSFW_T_THRESHOLD) &
                 (nsfw["dial_prob"] < NSFW_T_THRESHOLD)].copy()
    under["delta"] = under["std_prob"] - under["dial_prob"]
    under["flip_type"] = "under-detection"

    flips = pd.concat([over.nlargest(top_n, "delta"), under.nlargest(top_n, "delta")])
    return flips[["dialect", "category", "flip_type", "std_prob", "dial_prob",
                  "delta", "std_prompt", "dial_prompt"]]


# ──────────────────────────────────────────────
# Output: figures, LaTeX, console
# ──────────────────────────────────────────────
def category_pivot(breakdown, sort_ascending):
    """Category x dialect matrix of deltas, ordered by mean absolute gap."""
    pivot = breakdown.pivot_table(index="category", columns="dialect",
                                  values="delta", aggfunc="mean")
    pivot = pivot[[d for d in DIALECTS if d in pivot.columns]]
    order = pivot.abs().mean(axis=1).sort_values(ascending=sort_ascending).index
    return pivot.loc[order]


def plot_heatmap(breakdown, output_path, filter_name, metric_label, title_suffix=""):
    pivot = category_pivot(breakdown, sort_ascending=True)
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1)

    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt="+.1f", center=0, cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": metric_label, "shrink": 0.8}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{filter_name} {metric_label} by category and dialect{title_suffix}")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


def write_aggregate_latex(agg_df, output_path):
    """Table 2, as two stacked panels: ΔTPR on toxic, then ΔFPR on benign."""
    dialects = [d for d in DIALECTS if d in agg_df["Dialect"].values]

    def cell(dialect, column):
        row = agg_df[agg_df["Dialect"] == dialect].iloc[0]
        value, p_value = row.get(column), row.get(f"{column} p")
        if pd.isna(value):
            return "--"
        text = f"{value:+.1f}"
        return f"\\textbf{{{text}}}" if p_value is not None and p_value < 0.05 else text

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Aggregate $\Delta$TPR (toxic) and $\Delta$FPR (benign) "
        r"relative to Standard AmE. Bold values indicate statistical "
        r"significance ($p < 0.05$).}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{" + "l" + "r" * len(dialects) + "}",
        r"\toprule",
    ]
    panels = [("toxic", r"(a) Toxic Prompts --- $\Delta$TPR (\%p)"),
              ("benign", r"(b) Benign Prompts --- $\Delta$FPR (\%p)")]
    for i, (split, heading) in enumerate(panels):
        if i:
            lines.append(r"\midrule")
        lines += [
            r"\multicolumn{" + str(len(dialects) + 1) + r"}{c}{\textit{" + heading + r"}} \\",
            r"\midrule",
            "Filter & " + " & ".join(dialects) + r" \\",
            r"\midrule",
        ]
        for filter_name in ("NSFW-T", "OMod"):
            column = f"{filter_name} Δ{METRIC[split]}"
            if column in agg_df.columns:
                lines.append(" & ".join([filter_name] + [cell(d, column) for d in dialects])
                             + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    Path(output_path).write_text(latex)
    print(f"  ✓ Saved LaTeX table: {output_path}")
    return latex


def print_aggregate(agg_df):
    print(f"\n{'='*70}\nTABLE 2: Aggregate Results\n{'='*70}")
    columns = ["Dialect"] + [c for c in agg_df.columns
                             if c.split()[-1] in ("ΔTPR", "ΔFPR", "p")
                             and "prob" not in c and c != "Dialect"]
    print(agg_df[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_category_breakdown(breakdown, filter_name, split):
    print(f"\n{'='*70}\nCATEGORY BREAKDOWN: {filter_name} ({split})\n{'='*70}")
    print(category_pivot(breakdown, sort_ascending=False)
          .to_string(float_format=lambda x: f"{x:+.1f}"))


def print_dramatic_flips(flips):
    if flips is None:
        return
    print(f"\n{'='*70}\nDRAMATIC VERDICT FLIPS (for qualitative examples)\n{'='*70}")
    for flip_type in ("over-detection", "under-detection"):
        subset = flips[flips["flip_type"] == flip_type]
        if not len(subset):
            continue
        print(f"\n--- {flip_type.upper()} (Std→Dial) ---")
        for _, row in subset.head(5).iterrows():
            print(f"  [{row['dialect']}|{row['category']}] "
                  f"Std={row['std_prob']:.3f} → Dial={row['dial_prob']:.3f} "
                  f"(Δ={row['delta']:.3f})")
            if pd.notna(row.get("std_prompt")):
                print(f"    Std: {str(row['std_prompt'])[:90]}...")
                print(f"    Dial: {str(row['dial_prompt'])[:90]}...")


def print_detailed_stats(agg_df):
    print(f"\n{'='*70}\nAPPENDIX: Detailed Statistical Results\n{'='*70}")
    for _, row in agg_df.iterrows():
        print(f"\n--- {row['Dialect']} ---")
        for key in sorted(k for k in row.index if k != "Dialect"):
            value = row[key]
            if pd.isna(value):
                continue
            # p-value columns are named "... p" or "... p (split)". Matching a bare
            # "p" would also catch ΔFPR and Δprob, which are effect sizes, not p-values.
            if key.endswith(" p") or " p (" in key:
                print(f"  {key}: {'<0.001' if value < 0.001 else f'{value:.3f}'}")
            elif isinstance(value, float):
                print(f"  {key}: {value:+.3f}" if abs(value) < 100 else f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Text-Level Safety Filter Analysis")
    parser.add_argument("--nsfw_t_toxic_dir", default="./text_level_toxic_results")
    parser.add_argument("--nsfw_t_benign_dir", default="./text_level_benign_results")
    parser.add_argument("--omod_dir", default="./openai_moderation_results")
    parser.add_argument("--output_dir", default="./text_level_results/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n📦 Loading data...")
    data = load_data(args)
    if data is None:
        print("\n⚠ No data loaded. Check directory paths.")
        return

    print("\n📊 Computing aggregate results...")
    agg_df = compute_aggregate(data)
    print_aggregate(agg_df)
    agg_df.to_csv(out / "aggregate_results.csv", index=False)
    write_aggregate_latex(agg_df, out / "table_aggregate.tex")

    print("\n📊 Computing category-level breakdowns and heatmaps...")
    for filter_name, split in GRID:
        if not len(select(data, filter_name, split)):
            continue
        breakdown = compute_category_breakdown(data, filter_name, split)
        label = f"{'nsfw_t' if filter_name == 'NSFW-T' else 'omod'}_{split}"
        print_category_breakdown(breakdown, filter_name, split)
        breakdown.to_csv(out / f"category_{label}.csv", index=False)
        plot_heatmap(breakdown, out / f"fig_heatmap_{label}.pdf", filter_name,
                     f"Δ{METRIC[split]} ({split})", f" on {split} prompts")

    print("\n📊 Computing NSFW-T probability shifts...")
    for split in ("toxic", "benign"):
        shift = compute_probability_shift(data, split)
        if len(shift):
            print(f"\n[{split.upper()}] mean Δprob per dialect")
            print(shift[["Dialect", "LaTeX_Format"]].to_string(index=False))
            shift.to_csv(out / f"nsfw_t_prob_shift_{split}.csv", index=False)

    print("\n📊 Finding dramatic flips...")
    flips = find_dramatic_flips(data)
    print_dramatic_flips(flips)
    if flips is not None:
        flips.to_csv(out / "dramatic_flips.csv", index=False)

    print_detailed_stats(agg_df)
    print(f"\n✅ All results saved to {out}/")


if __name__ == "__main__":
    main()
