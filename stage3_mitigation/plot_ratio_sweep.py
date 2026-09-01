"""
Plot worst-group accuracy and FPR spread vs SAE training fraction.
ERM should degrade with imbalance; DRO should remain robust.

X-axis points:
  - 0.90 ~ 0.99: imbalanced sweep (ERM and DRO)
  - 1.00:        EXP2 SAE-only ERM
Reference (horizontal line):
  - EXP3 Balanced ERM (budget-matched, all dialects equally represented)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# ----- config -----
FRACS  = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
SEEDS  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SWEEP_DIR  = Path("./results/sweep_ratio")
EXP2_DIR   = Path("./results/seeds_exp2")          # SAE-only ERM (x=1.00)
EXP3_BAL_DIR = Path("./results/seeds_exp3")        # Balanced ERM (horizontal ref)

EXP2_NAME = "sae_only_erm"
BAL_NAME  = "erm_balanced"

mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
})


def _load_seed_runs(template: str, seeds=SEEDS):
    out = []
    for s in seeds:
        p = Path(template.format(seed=s))
        if p.exists():
            with open(p) as f:
                out.append(json.load(f))
    return out


def collect(method: str, metric: str):
    """Return (xs, means, mins, maxs) for the sweep range PLUS the SAE-only point.

    The SAE-only point only exists for ERM. For DRO we omit x=1.0.
    Bands are plotted using actual min/max across seeds (not std).
    """
    xs, means, mins, maxs = [], [], [], []

    # --- sweep range (0.90 ... 0.99) ---
    for frac in FRACS:
        tag = f"{int(round(frac * 1000)):04d}"
        runs = _load_seed_runs(
            str(SWEEP_DIR / f"{method}_{tag}_s{{seed}}" / "test_metrics.json")
        )
        vals = [r[metric] for r in runs if metric in r]
        if vals:
            xs.append(frac)
            means.append(np.mean(vals))
            mins .append(np.min(vals))
            maxs .append(np.max(vals))

    # --- SAE-only at x = 1.0 (ERM only) ---
    if method == "erm":
        runs = _load_seed_runs(
            str(EXP2_DIR / f"{EXP2_NAME}_s{{seed}}" / "test_metrics.json")
        )
        vals = [r[metric] for r in runs if metric in r]
        if vals:
            xs.append(1.00)
            means.append(np.mean(vals))
            mins .append(np.min(vals))
            maxs .append(np.max(vals))

    return np.array(xs), np.array(means), np.array(mins), np.array(maxs)


def collect_balanced(metric: str):
    """Return (mean, min, max) over seeds for the balanced ERM reference."""
    runs = _load_seed_runs(
        str(EXP3_BAL_DIR / f"{BAL_NAME}_s{{seed}}" / "test_metrics.json")
    )
    vals = [r[metric] for r in runs if metric in r]
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def _draw_curves(ax, metric_key, show_individual_seeds=True):
    # ----- Balanced ERM (horizontal reference) -----
    bal_mean, bal_min, bal_max = collect_balanced(metric_key)
    if bal_mean is not None:
        label = f"Balanced ERM (Worst-group accuracy: {100*bal_mean:.1f}%)"
        ax.axhline(y=bal_mean, color="#2ca02c", linestyle="--",
                   lw=1.5, alpha=0.75, label=label)
        if bal_max - bal_min > 1e-6:
            ax.axhspan(bal_min, bal_max, color="#2ca02c", alpha=0.08)

    # ----- Sweep curves with min-max band + individual seeds -----
    for method, color, marker in [("erm", "#1f77b4", "o"),
                                   ("dro", "#d62728", "s")]:
        xs, m, lo, hi = collect(method, metric_key)
        if len(xs) == 0:
            continue
        label = "ERM" if method == "erm" else "Group DRO"

        # min-max band (background)
        ax.fill_between(xs, np.clip(lo, 0, 1), np.clip(hi, 0, 1),
                        color=color, alpha=0.12)

        # individual seed points (small, semi-transparent)
        if show_individual_seeds:
            for x, frac in zip(xs, xs):
                if frac < 0.999:    # sweep range
                    tag = f"{int(round(frac * 1000)):04d}"
                    template = str(SWEEP_DIR / f"{method}_{tag}_s{{seed}}" / "test_metrics.json")
                else:               # SAE-only at x=1.0
                    template = str(EXP2_DIR / f"{EXP2_NAME}_s{{seed}}" / "test_metrics.json")
                runs = _load_seed_runs(template)
                vals = [r[metric_key] for r in runs if metric_key in r]
                # jitter x slightly to avoid overlap
                jitter = np.random.RandomState(0).uniform(-0.003, 0.003, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals,
                           color=color, alpha=0.35, s=18,
                           edgecolors="none", zorder=2)

        # mean curve (foreground, thick)
        ax.plot(xs, m, marker=marker, color=color, lw=1.5, ms=4,
                label=label, zorder=3)

    # SAE-only vertical guide
    ax.axvline(x=1.00, color="gray", linestyle=":", alpha=0.5, lw=1.0)
    ax.text(1.002, 0.02, "SAE-only (fraction=100%)", color="gray", fontsize=8,
            rotation=90, ha="left", va="bottom",
            transform=ax.get_xaxis_transform())


def plot_metric(metric_key, ylabel, ylim, savename):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    _draw_curves(ax, metric_key, show_individual_seeds = False)

    ax.set_xlabel("SAE fraction in training data")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.945, 1.005)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(SWEEP_DIR / f"{savename}.pdf", bbox_inches="tight")
    fig.savefig(SWEEP_DIR / f"{savename}.png", dpi=200, bbox_inches="tight")
    print(f"[saved] {SWEEP_DIR / savename}.pdf")
    plt.close(fig)

def write_table():
    rows = []

    def _add_rows(runs, frac, method):
        for metric in ["test_worst_group_acc", "test_FPR_spread", "test_accuracy"]:
            vals = [r[metric] for r in runs if metric in r]
            if vals:
                rows.append({
                    "sae_frac": frac, 
                    "method": method, 
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "min":  float(np.min(vals)), 
                    "max":  float(np.max(vals)), 
                    "n_seeds": len(vals),
                })

    # 1. sweep range (0.90 ~ 0.99)
    for frac in FRACS:
        tag = f"{int(round(frac * 1000)):04d}"
        for method in ["erm", "dro"]:
            runs = _load_seed_runs(
                str(SWEEP_DIR / f"{method}_{tag}_s{{seed}}" / "test_metrics.json")
            )
            _add_rows(runs, frac, method)

    # 2. SAE-only at x=1.00 (ERM only)
    runs = _load_seed_runs(
        str(EXP2_DIR / f"{EXP2_NAME}_s{{seed}}" / "test_metrics.json")
    )
    _add_rows(runs, 1.00, "erm")

    # 3. Balanced reference (no x position, encoded as method='erm_balanced')
    runs = _load_seed_runs(
        str(EXP3_BAL_DIR / f"{BAL_NAME}_s{{seed}}" / "test_metrics.json")
    )
    _add_rows(runs, "balanced", "erm_balanced")

    # Save as CSV
    out_path = SWEEP_DIR / "sweep_table.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    plot_metric("test_worst_group_acc", "Worst-group accuracy",
                (0.4, 1.005), "sweep_worst_group")
    
    plot_metric("test_mean_group_acc", "Mean accuracy",
                (0.4, 1.005), "sweep_mean")
    
    write_table()