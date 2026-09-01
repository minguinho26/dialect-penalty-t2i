"""make_table8_markdown.py — Print Table 8 (dialect penalty mitigation) as markdown.

Data sources:
  - Each frac(0.95~0.995) × {erm,dro}:  results/sweep_ratio/{m}_{tag}_s{seed}/test_metrics.json
  - 1.0 (SAE-only, ERM):              results/seeds_exp2/sae_only_erm_s{seed}/test_metrics.json
  - Balanced (ERM):                   results/seeds_exp3/erm_balanced_s{seed}/test_metrics.json
"""
import os
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

COLS = [
    ("test_accuracy",       "Mean Acc. (%)",  True, 2),
    ("test_worst_group_acc","Worst-Group (%)",True, 2),
    # Escape pipe character since it is parsed as a column delimiter in markdown tables.
    ("test_mean_abs_dTPR",  "\\|ΔTPR\\| (%)", True, 2),
    ("test_mean_abs_dFPR",  "\\|ΔFPR\\| (%)", True, 2),
]

DEFAULT_FRACS = [0.975, 0.98, 0.985, 0.99, 0.995]
ALL_FRACS = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]


def tag(frac):
    return f"{int(round(frac * 1000)):04d}"


def load_seeds(path_tmpl, seeds=None):
    out = []
    for s in (seeds if seeds is not None else SEEDS):
        p = path_tmpl.format(seed=s)
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def agg(runs, key):
    vals = [r[key] for r in runs if key in r]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def fmt(mean, std, scale, nd):
    if mean is None:
        return "—"
    if scale:
        mean, std = mean * 100, std * 100
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def row_cells(runs, ratio_label, algo_label):
    cells = [ratio_label, algo_label]
    for key, _, scale, nd in COLS:
        m, s = agg(runs, key)
        cells.append(fmt(m, s, scale, nd))
    # Show actual number of seeds per row to clarify mixed 5/10 seed counts.
    cells.append(str(len(runs)))
    return cells


def build_rows(fracs, seeds=None, ablation=False, sweep_subdir="sweep_ratio",
               saeonly_subdir="seeds_exp2", balanced_subdir="seeds_exp3"):
    rows = []  # list of cell-lists
    for frac in fracs:
        ratio = f"{frac * 100:.1f}%"
        erm = load_seeds(os.path.join(RES, sweep_subdir, f"erm_{tag(frac)}_s{{seed}}", "test_metrics.json"), seeds)
        dro = load_seeds(os.path.join(RES, sweep_subdir, f"dro_{tag(frac)}_s{{seed}}", "test_metrics.json"), seeds)
        ermgb = load_seeds(os.path.join(RES, sweep_subdir, f"ermgb_{tag(frac)}_s{{seed}}", "test_metrics.json"), seeds) \
            if ablation else []
        # Disambiguate the sampler only when the ablation cell is present:
        # then "ERM" is the random-sampling baseline, "ERM (GB)" adds the
        # sampler alone, and "G. DRO" adds the loss on top of the sampler.
        erm_label = "ERM (rand.)" if ermgb else "ERM"
        if erm:
            rows.append(row_cells(erm, ratio, erm_label))
        if ermgb:
            rows.append(row_cells(ermgb, "", "ERM (GB)"))
        if dro:
            rows.append(row_cells(dro, "", "G. DRO"))
    # DRO is not used for SAE-only because minority groups have zero samples.
    sae = load_seeds(os.path.join(RES, saeonly_subdir, "sae_only_erm_s{seed}", "test_metrics.json"), seeds)
    if sae:
        rows.append(row_cells(sae, "100.0%", "ERM"))
    bal = load_seeds(os.path.join(RES, balanced_subdir, "erm_balanced_s{seed}", "test_metrics.json"), seeds)
    if bal:
        rows.append(row_cells(bal, "Balanced", "ERM"))
    return rows


def to_markdown(rows):
    headers = ["SAE Ratio", "Algorithm"] + [h for _, h, _, _ in COLS] + ["Seeds"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Print all fracs (0.95~0.995)")
    ap.add_argument("--ablation", action="store_true",
                    help="Add ERM+group-balanced row (results/<sweep-subdir>/ermgb_*). "
                         "Ablation to isolate sampling effect from loss effect.")
    ap.add_argument("--sweep-subdir", default="sweep_ratio",
                    help="Sweep results folder under results/ (use sweep_ratio_v2 for re-runs)")
    ap.add_argument("--saeonly-subdir", default="seeds_exp2",
                    help="SAE-only (100%%) anchor folder (v2 is sweep_ratio_v2)")
    ap.add_argument("--balanced-subdir", default="seeds_exp3",
                    help="Balanced anchor folder (v2 is sweep_ratio_v2)")
    ap.add_argument("--data-prefix", default="train_imb",
                    help="(Unused, for logging) data file prefix used in sweep")
    ap.add_argument("--fracs", nargs="*", type=float, default=None,
                    help="Specific fracs only (e.g., --fracs 0.99 0.995)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="List of seeds to aggregate. Missing seed files are skipped, and actual count is shown in Seeds column.")
    ap.add_argument("--out", default=os.path.join(RES, "table8_penalty.md"))
    args = ap.parse_args()

    fracs = args.fracs if args.fracs else (ALL_FRACS if args.all else DEFAULT_FRACS)
    rows = build_rows(fracs, seeds=args.seeds, ablation=args.ablation,
                      sweep_subdir=args.sweep_subdir,
                      saeonly_subdir=args.saeonly_subdir,
                      balanced_subdir=args.balanced_subdir)
    md = to_markdown(rows)

    n_seeds = sorted({int(c[-1]) for c in rows})
    if not n_seeds:
        print("[warn] no runs found for the requested fracs/subdirs — nothing to aggregate.")
        seeds_note = "0 seeds"
    elif len(n_seeds) == 1:
        seeds_note = f"{n_seeds[0]} seeds"
    else:
        seeds_note = (f"{min(n_seeds)}–{max(n_seeds)} seeds (per-row count in the Seeds column; "
                      "extreme-imbalance rows use additional seeds)")
    caption = (
        "\n**Table 8 (revised): Mitigating the dialect penalty under data imbalance.** "
        "Mean/worst-group accuracy plus dialect-penalty metrics (|ΔTPR|, |ΔFPR|; "
        "SAE-relative). The penalty (FPR-side) only emerges under extreme SAE imbalance "
        "(≥99%); GroupDRO degrades more gracefully than ERM there. ΔTPR≈0 throughout "
        f"(toxic prompts are topically salient). Mean ± std over {seeds_note}."
    )
    out = md + "\n" + caption + "\n"
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
