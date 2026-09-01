"""analyze_backtranslate.py - Backtranslation aggregation: summarizes the three scoring methods (OMod / LLM judge / content) into tables.

Core question: "Did the translation alter the toxicity level?"
  -> If the original SAE and the back-translated SAE match in (a) toxicity intensity, and (b) meaning, then 'preservation' is confirmed = refutation established.

Report:
  [OMod]  If the signed Δ(bt-std) per category is ≈ 0 and |Δ| is small, intensity is preserved.
          (If the signed Δ leans negative, it means 'translation lowered toxicity')
  [judge] After independent pointwise S1-S5 scoring (SAE/bt), if mean|ΔS|≈0 and consistency≈1, the ordinal toxicity level is preserved.
          (Faisal et al. 2025 protocol; replaced with T2I descriptive rubric).
  [content] High yes (=same meaning) ratio indicates meaning preservation.

Usage:
    python analyze_backtranslate.py
Outputs: results/backtranslation_audit/bt_summary_{omod,judge,content}.csv + console report
"""
import os
import numpy as np
import pandas as pd

from backtranslate_common import atomic_to_csv, OUT_DIR

OMOD = os.path.join(OUT_DIR, "omod_scores.csv")
JUDGE = os.path.join(OUT_DIR, "llmjudge_scores.csv")
CONTENT = os.path.join(OUT_DIR, "content_scores.csv")


def _cats(df):
    return sorted(c[4:] for c in df.columns if c.startswith("std_") and f"bt_{c[4:]}" in df.columns
                  and c[4:] != "flagged")


def analyze_omod():
    if not os.path.exists(OMOD):
        print("[OMod] No results - skipping"); return
    df = pd.read_csv(OMOD)
    df = df[df["std_flagged"].notna()]
    cats = _cats(df)
    print("\n" + "=" * 66 + "\n[OMod] Toxicity intensity preservation (original SAE vs back-translated SAE)\n" + "=" * 66)
    print(f"  Pairs: {len(df)}   Categories: {len(cats)}")

    df["std_max"] = df[[f"std_{c}" for c in cats]].max(axis=1)
    df["bt_max"] = df[[f"bt_{c}" for c in cats]].max(axis=1)

    rows = []
    for c in cats + ["max"]:
        s, b = df[f"std_{c}"], df[f"bt_{c}"]
        d = b - s
        corr = np.corrcoef(s, b)[0, 1] if s.std() > 0 and b.std() > 0 else np.nan
        rows.append({"metric": c, "mean_std": s.mean(), "mean_bt": b.mean(),
                     "signed_delta(bt-std)": d.mean(), "mean_abs_delta": d.abs().mean(),
                     "pearson_r": corr})
    summ = pd.DataFrame(rows)
    fa = (df["std_flagged"].astype(bool) == df["bt_flagged"].astype(bool)).mean()
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Flag match rate (std==bt): {fa:.3f}   "
          f"std_flag ratio={df['std_flagged'].astype(bool).mean():.3f}  "
          f"bt_flag ratio={df['bt_flagged'].astype(bool).mean():.3f}")
    print("\n  [max-score signed Δ(bt-std) per split]")
    for sp, g in df.groupby("split"):
        print(f"    {sp:7s} n={len(g):4d}  signed Δ={ (g['bt_max']-g['std_max']).mean():+.4f}  "
              f"|Δ|={ (g['bt_max']-g['std_max']).abs().mean():.4f}  flag_match={ (g['std_flagged'].astype(bool)==g['bt_flagged'].astype(bool)).mean():.3f}")
    atomic_to_csv(summ, os.path.join(OUT_DIR, "bt_summary_omod.csv"), index=False)


def analyze_judge():
    if not os.path.exists(JUDGE):
        print("\n[judge] No results - skipping"); return
    df = pd.read_csv(JUDGE)
    df = df[df["score_std"].notna() & df["score_bt"].notna()].copy()
    print("\n" + "=" * 66 +
          "\n[LLM judge] Pointwise S1-S5 toxicity preservation (Faisal et al. 2025 protocol, independent detector)\n"
          + "=" * 66)
    print(f"  Pairs: {len(df)}")

    def stat(g):
        d = g["delta"]
        n = len(g)
        return pd.Series({
            "n": n,
            "mean_S_std": g["score_std"].mean(),
            "mean_S_bt": g["score_bt"].mean(),
            "signed_ΔS": d.mean(),
            "mean_|ΔS|": d.abs().mean(),
            "exact%(ΔS=0)": 100 * (d == 0).mean(),
            "within1%(|ΔS|<=1)": 100 * (d.abs() <= 1).mean(),
            "consistency": g["consistency"].mean(),
        })

    print("\n  [Overall]"); print("   ", {k: round(v, 3) for k, v in stat(df).to_dict().items()})
    print("\n  [Per split]")
    print(df.groupby("split").apply(stat).to_string(float_format=lambda x: f"{x:.3f}"))
    out = df.groupby(["split", "dialect"]).apply(stat).reset_index()
    atomic_to_csv(out, os.path.join(OUT_DIR, "bt_summary_judge.csv"), index=False)
    print("\n  Interpretation: mean|ΔS|≈0, signed_ΔS≈0, consistency≈1 -> back-translation preserves toxicity level.")


def analyze_content():
    if not os.path.exists(CONTENT):
        print("\n[content] No results - skipping"); return
    df = pd.read_csv(CONTENT)
    df = df[df["same_AB"].notna()]
    print("\n" + "=" * 66 + "\n[content-evaluator] Meaning preservation (original SAE vs back-translated SAE)\n" + "=" * 66)
    print(f"  Pairs: {len(df)}")

    def stat(g):
        n = len(g)
        yes = (g["agree"] == "yes").sum()
        no = (g["agree"] == "no").sum()
        inc = (g["agree"] == "inconsistent").sum()
        return pd.Series({"n": n, "same%(yes)": 100 * yes / n,
                          "diff%(no)": 100 * no / n, "inconsistent%": 100 * inc / n})

    print("\n  [Overall]"); print("   ", stat(df).to_dict())
    print("\n  [Per split]"); print(df.groupby("split").apply(stat).to_string(float_format=lambda x: f"{x:.1f}"))
    print("\n  [Per dialect]"); print(df.groupby("dialect").apply(stat).to_string(float_format=lambda x: f"{x:.1f}"))
    out = df.groupby(["split", "dialect"]).apply(stat).reset_index()
    atomic_to_csv(out, os.path.join(OUT_DIR, "bt_summary_content.csv"), index=False)
    print("\n  Interpretation: High same%(yes) indicates meaning preservation.")


if __name__ == "__main__":
    analyze_omod()
    analyze_judge()
    analyze_content()
    print("\n[done] Saved results/backtranslation_audit/bt_summary_*.csv")
