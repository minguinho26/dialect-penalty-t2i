"""
Convert wide-format dialect CSVs (one row = 6 dialect paraphrases) into a
long-format training set with (label x dialect) group labels.

Split policy: stratified 90/10 at the ROW level on the `category` column,
which preserves both the per-category and per-label distributions in train
and val while preventing paraphrase leakage between splits.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Sets the base directory to the directory where this file is located, regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
def _p(rel):
    return os.path.join(_HERE, rel)

import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS_WITH_SAE as DIALECTS

PROMPT_COLS = [f"{d}_prompt" for d in DIALECTS]
N_DIALECTS  = len(DIALECTS)            # 6
N_GROUPS    = 2 * N_DIALECTS           # 12  (label x dialect)

VAL_RATIO = 0.10
SEED      = 42

os.makedirs(_p("data"), exist_ok=True)

# ---------- 1. load ----------
toxic_df  = pd.read_csv(_p("original_dataset/toxic_prompts_sae_and_all_dialects.csv"))
benign_df = pd.read_csv(_p("original_dataset/benign_prompts_sae_and_all_dialects.csv"))

for name, df in [("toxic", toxic_df), ("benign", benign_df)]:
    if "category" not in df.columns:
        raise ValueError(
            f"{name} CSV has no `category` column. "
            f"Per-category stratified split requires it."
        )

toxic_df["label"]  = 1
benign_df["label"] = 0

keep = ["category", "label"] + PROMPT_COLS
wide = pd.concat([toxic_df[keep], benign_df[keep]], ignore_index=True)

# Drop rows with any missing dialect prompt to keep paraphrase sets intact.
wide = wide.dropna(subset=PROMPT_COLS).reset_index(drop=True)
wide["sample_id"] = np.arange(len(wide))

# ---------- 2. per-category stratified ROW-level split ----------
# Guard against categories too small for stratified split (need >=2 samples,
# and val_ratio*count >= 1). Smallest in the table is 58 -> safe.
cat_counts = wide["category"].value_counts()
too_small  = cat_counts[cat_counts < int(np.ceil(1.0 / VAL_RATIO))]
if len(too_small) > 0:
    print(f"[warn] categories with <{int(np.ceil(1.0 / VAL_RATIO))} samples "
          f"may yield zero val examples for that category:\n{too_small}")

train_wide, val_wide = train_test_split(
    wide,
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=wide["category"],   # <-- key change
)

# ---------- 3. wide -> long ----------
def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, r in df_wide.iterrows():
        for d_idx, d in enumerate(DIALECTS):
            records.append({
                "text":        str(r[f"{d}_prompt"]).strip(),
                "label":       int(r["label"]),
                "dialect":     d,
                "dialect_idx": d_idx,
                "group":       N_DIALECTS * int(r["label"]) + d_idx,  # 0..11
                "sample_id":   int(r["sample_id"]),
                "category":    r["category"],
            })
    out = pd.DataFrame.from_records(records)
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out

train_long = to_long(train_wide)
val_long   = to_long(val_wide)

# ---------- 4. save ----------
# Keep `category` and `dialect` for downstream per-cell evaluation,
# while train_group_dro.py only requires (text, label, group).
train_long.to_csv(_p("data/train.csv"), index=False)
val_long  .to_csv(_p("data/val.csv"),   index=False)

# ---------- 5. sanity checks ----------
def report(name, df_wide, df_long):
    print(f"\n=== {name} ===")
    print(f"rows (wide)   : {len(df_wide):,}")
    print(f"samples (long): {len(df_long):,}   "
          f"(= rows x {N_DIALECTS} dialects, minus dropped empties)")
    print("per-category counts (wide rows):")
    print(df_wide["category"].value_counts().sort_index().to_string())

report("TRAIN", train_wide, train_long)
report("VAL",   val_wide,   val_long)

print("\n=== leakage audit ===")

# (1) base-prompt disjointness: train/test must share no sample_id
tr_ids = set(train_long["sample_id"])
va_ids = set(val_long["sample_id"])
overlap_ids = tr_ids & va_ids
print(f"shared sample_id (train ∩ val) : {len(overlap_ids)}  "
      f"(must be 0)")

# (2) exact-text disjointness across splits (catches identical strings
#     that could arise from different base prompts)
tr_txt = set(train_long["text"])
va_txt = set(val_long["text"])
exact_dups = tr_txt & va_txt
print(f"exact-duplicate texts across splits : {len(exact_dups)}")

# (3) near-duplicate check: max 5-gram Jaccard between any cross-split
#     SAE-prompt pair (SAE only, to keep it tractable & interpretable)
def char_ngrams(s, n=5):
    s = " ".join(str(s).split())
    return {s[i:i+n] for i in range(max(len(s) - n + 1, 1))}

tr_sae = train_wide["SAE_prompt"].astype(str).tolist()
va_sae = val_wide["SAE_prompt"].astype(str).tolist()
tr_grams = [char_ngrams(s) for s in tr_sae]

max_j = 0.0
arg = None
for j, vg in enumerate((char_ngrams(s) for s in va_sae)):
    for i, tg in enumerate(tr_grams):
        inter = len(vg & tg)
        if inter == 0:
            continue
        jac = inter / len(vg | tg)
        if jac > max_j:
            max_j, arg = jac, (i, j)
print(f"max cross-split SAE 5-gram Jaccard  : {max_j:.4f}")
if arg is not None:
    print(f"  train#{arg[0]}: {tr_sae[arg[0]][:80]!r}")
    print(f"  val#{arg[1]}  : {va_sae[arg[1]][:80]!r}")

# Per-category split ratio check
print("\n=== split ratio per category (val / total) ===")
ratio = (val_wide["category"].value_counts()
         / wide["category"].value_counts()).sort_index()
print(ratio.round(3).to_string())

# Group-level breakdown for Group DRO
print("\n=== group definition  (g = 6*label + dialect_idx) ===")
for g in range(N_GROUPS):
    lab = "nsfw" if g // N_DIALECTS == 1 else "safe"
    dia = DIALECTS[g % N_DIALECTS]
    n_tr = (train_long["group"] == g).sum()
    n_va = (val_long  ["group"] == g).sum()
    print(f"  g={g:2d}  ({lab:4s}, {dia:7s})   train={n_tr:6d}  val={n_va:5d}")