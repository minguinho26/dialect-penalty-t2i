"""analyze_evaluators.py — Third-party image evaluator cross-validation table.

Compiles outputs from score_unguarded_vlm.py to produce:
  - Unsafe rate (SAE vs dialect) per split×dialect×evaluator, Δ(pp), paired McNemar p

Input locations (auto-discovered):
  results/shieldgemma/shieldgemma_{split}_{dialect}.csv

Usage:
    python analyze_evaluators.py
    python analyze_evaluators.py --only-complete
"""

import os
import sys
import glob
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # make `common/` importable when run from the repository root
from common.common_utils import DIALECTS as DIALECT_ORDER, mcnemar_exact

RESULTS = os.path.join(ROOT, "results")
OUT_CSV = os.path.join(RESULTS, "evaluator_summary.csv")

EXPECTED = {"toxic": 2216, "benign": 2400}


def collect(only_complete=False):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*_*_*.csv"))):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        if len(parts) < 3 or parts[1] not in EXPECTED:
            continue
        tag, split, dialect = parts[0], parts[1], "_".join(parts[2:])
        if os.path.basename(os.path.dirname(path)) != tag:
            continue
        df = pd.read_csv(path)
        v = df[df.std_rating.isin(["Safe", "Unsafe"]) & df.dial_rating.isin(["Safe", "Unsafe"])]
        if v.empty:
            continue
        complete = len(df) >= EXPECTED[split]
        if only_complete and not complete:
            continue
        s = (v.std_rating == "Unsafe")
        d = (v.dial_rating == "Unsafe")
        b = int((s & ~d).sum())
        c = int((~s & d).sum())
        rows.append({
            "evaluator": tag, "split": split, "dialect": dialect,
            "n": len(v), "complete": complete,
            "sae_unsafe_pct": round(100 * s.mean(), 2),
            "dial_unsafe_pct": round(100 * d.mean(), 2),
            "delta_pp": round(100 * (d.mean() - s.mean()), 2),
            "disc_sae_only": b, "disc_dial_only": c,
            "mcnemar_p": round(mcnemar_exact(b, c), 4),
            "invalid_pairs": len(df) - len(v),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-complete", action="store_true")
    args = ap.parse_args()

    out = collect(args.only_complete)
    if out.empty:
        print("No results — run scoring scripts first")
        return
    out["dialect"] = pd.Categorical(out["dialect"], DIALECT_ORDER, ordered=True)
    out = out.sort_values(["split", "dialect", "evaluator"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    for split in ["toxic", "benign"]:
        sub = out[out.split == split]
        if sub.empty:
            continue
        print(f"\n=== {split} : Δ(pp) = dialect − SAE, * = p<0.05 ===")
        piv = sub.pivot_table(index="dialect", columns="evaluator",
                              values="delta_pp", observed=True)
        sig = sub.pivot_table(index="dialect", columns="evaluator",
                              values="mcnemar_p", observed=True)
        for d in piv.index:
            cells = []
            for e in piv.columns:
                v, p = piv.loc[d, e], sig.loc[d, e]
                cells.append("     —   " if pd.isna(v)
                             else f"{v:+7.2f}{'*' if p < 0.05 else ' '}")
            print(f"  {str(d):9s} " + " ".join(cells))
        print(f"  {'':9s} " + " ".join(f"{e:>8s}" for e in piv.columns))

    inc = out[~out.complete]
    if not inc.empty:
        print("\n⚠ Incomplete (in-progress) folders — interpret with caution:")
        for _, r in inc.iterrows():
            print(f"   {r.evaluator}/{r.split}/{r.dialect}: n={r['n']}")
    bad = out[out.invalid_pairs > 0]
    if not bad.empty:
        print("\n⚠ Parsing failures exist:")
        for _, r in bad.iterrows():
            print(f"   {r.evaluator}/{r.split}/{r.dialect}: {r.invalid_pairs} rows")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
