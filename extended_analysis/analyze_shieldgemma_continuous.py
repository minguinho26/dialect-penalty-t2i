"""analyze_shieldgemma_continuous.py — Continuous analysis of ShieldGemma saved probabilities.

Purpose: In the **binary decision** (pmax>0.5) of ShieldGemma, only toxic JamE showed a large negative Δ.
We need to distinguish whether this is "binary binarization exaggerating massive dropouts near the threshold" or "a real effect where the continuous score distribution itself shifted".
ShieldGemma stores P(violation) probabilities per policy in the CSV (std_probs/dial_probs, 3 policies separated by semicolon; pmax = max of those), so no re-inference is needed.

Analysis Axes:
  1) Binary Δ (pmax>thr Unsafe rate; relative to SAE) — Reproduces existing results
  2) Continuous Δ (mean pmax; relative to SAE) — Same criteria as existing 2 types (Table 7 continuous Δ)
  3) Threshold sweep — If Δ increases only at a specific thr, it's a "threshold artifact", if across all ranges, it's a real effect
  4) Paired per-image: Δpmax distribution / sign (ratio where dial<std)

Usage:
    python analyze_shieldgemma_continuous.py                 
    python analyze_shieldgemma_continuous.py --split benign
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # make `common/` importable when run from the repository root
SG_DIR = os.path.join(ROOT, "results", "shieldgemma")
from common.common_utils import DIALECTS


def parse_probs(s):
    if not isinstance(s, str) or not s.strip():
        return np.nan
    try:
        return max(float(x) for x in s.split(";") if x != "")
    except Exception:
        return np.nan


def load(split, dialect):
    f = os.path.join(SG_DIR, f"shieldgemma_{split}_{dialect}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    for side in ("std", "dial"):
        col = f"{side}_pmax"
        if col not in d or d[col].isna().all():
            d[col] = d[f"{side}_probs"].map(parse_probs)
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["std_pmax", "dial_pmax"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="toxic", choices=["toxic", "benign"])
    ap.add_argument("--thresholds", default="0.1,0.3,0.5,0.7,0.9")
    args = ap.parse_args()
    thrs = [float(x) for x in args.thresholds.split(",")]

    print(f"### ShieldGemma continuous analysis — {args.split} ###\n")

    print(f"{'dialect':9s} {'n':>5s} | {'mean pmax SAE→dial':>20s} {'Δcont':>8s} {'sign p':>8s}"
          f" | {'Unsafe% SAE→dial(0.5)':>22s} {'Δbin':>7s}")
    print("-" * 92)
    rows = []
    for dl in DIALECTS:
        d = load(args.split, dl)
        if d is None or d.empty:
            print(f"{dl:9s}  (No result)")
            continue
        s_c, d_c = d.std_pmax.values, d.dial_pmax.values
        dcont = d_c.mean() - s_c.mean()
        diff = d_c - s_c
        nz = diff[diff != 0]
        sign_p = binomtest((nz < 0).sum(), len(nz), 0.5).pvalue if len(nz) else 1.0
        s_b = (s_c > 0.5).mean() * 100
        d_b = (d_c > 0.5).mean() * 100
        rows.append((dl, d, s_c, d_c))
        print(f"{dl:9s} {len(d):5d} | {s_c.mean():8.4f} → {d_c.mean():8.4f} {dcont:+8.4f} "
              f"{sign_p:8.3g} | {s_b:8.2f} → {d_b:8.2f} {d_b - s_b:+7.2f}")

    print(f"\n### Threshold sweep: Δ Unsafe%(dial−SAE) ###")
    print(f"{'dialect':9s} " + " ".join(f"thr={t:>4.2f}" for t in thrs))
    for dl, d, s_c, d_c in rows:
        cells = []
        for t in thrs:
            cells.append(f"{(d_c > t).mean() * 100 - (s_c > t).mean() * 100:+8.2f}")
        print(f"{dl:9s} " + " ".join(cells))

    jam = next((r for r in rows if r[0] == "JamE"), None)
    if jam:
        _, d, s_c, d_c = jam
        print(f"\n### JamE details ###")
        print(f"  paired Δpmax: mean {(d_c - s_c).mean():+.4f}, median {np.median(d_c - s_c):+.4f}")
        print(f"  dial < std ratio: {(d_c < s_c).mean() * 100:.1f}%  (Real effect if overall, not just near 0.5)")
        near = ((s_c > 0.3) & (s_c < 0.7)).mean() * 100
        print(f"  SAE pmax 0.3~0.7 (boundary) ratio: {near:.1f}%")
        try:
            w = wilcoxon(d_c, s_c)
            print(f"  Wilcoxon signed-rank (continuous): p={w.pvalue:.3g}")
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")
        qs = [10, 25, 50, 75, 90]
        print("  Quantiles:   " + "  ".join(f"p{q}" for q in qs))
        print("    SAE :   " + "  ".join(f"{np.percentile(s_c, q):.3f}" for q in qs))
        print("    JamE:   " + "  ".join(f"{np.percentile(d_c, q):.3f}" for q in qs))

    print("\nInterpretation guide:")
    print("  · If Δcont(continuous) is uniquely negative for JamE and Δbin<0 across all sweeps → Real effect")
    print("  · If Δcont≈0 but Δbin is large only around thr=0.5 → Binarization (threshold) is exaggerating it")


if __name__ == "__main__":
    main()
