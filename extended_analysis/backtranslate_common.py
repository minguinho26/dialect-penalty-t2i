"""backtranslate_common.py — Common loader/utils for Backtranslation scoring scripts.

Reads back-translation results (results/backtranslation/bt_{split}_{dialect}.csv)
and organizes them into pairs of (Original SAE = standard_prompt) vs (Back-translated SAE = back_translated).
Shared across three scoring scripts (OMod / Gemini judge / content-evaluator).

Core design:
  - Comparison is always SAE↔SAE (both Standard English) → ensures the detector's own dialect bias does not intervene.
  - pair_id is based on the dialect_prompt hash, not index → stable for resuming
    (keys won't shift even if the back-translation file is re-saved after filling in blanks).
"""
import os
import sys
import hashlib
import tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # make `common/` importable when run from the repository root
from common.common_utils import DIALECTS  # noqa: E402

BT_DIR = os.path.join(ROOT, "results", "backtranslation")
OUT_DIR = os.path.join(ROOT, "results", "backtranslation_audit")
SPLITS = ["toxic", "benign"]

# Sentinel used by exp_backtranslate.py for confirmed hard blocks (PROHIBITED) (requires sync on both sides).
# Since it's not actual back-translated text, it is always excluded from scoring targets.
BLOCKED_SENTINEL = "__BLOCKED__"


def atomic_to_csv(df, out_path, **to_csv_kwargs):
    """Atomic CSV save: temporary file in the same directory → flush+fsync → os.replace.

    ⚠ If to_csv writes 'directly' to the target path, it truncates the file first, so in case of disk full (ENOSPC),
    I/O error, or kill during writing, **the existing data is completely destroyed**.
    Actual incident: 2026-07-24 01:00 UTC, disk at 100%, to_csv in back-translation script failed with ENOSPC after
    truncate, resulting in bt_toxic_AAVE.csv becoming 0 bytes and losing 2,212 successful entries.

    Scoring scripts rewrite the entire cumulative results per row as 'interruption prep', thus sharing the same risk
    (thousands of scored entries could vanish with one write failure). Wrapping with this function preserves the original on failure.

    ※ exp_backtranslate.py has an identical implementation (_atomic_to_csv) — that one is kept self-contained to minimize
      import dependencies for its daily cron path. Fix both if one is modified.
    """
    d = os.path.dirname(out_path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_bt_", suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            df.to_csv(f, **to_csv_kwargs)
            f.flush()
            os.fsync(f.fileno())      # Ensures it hits the actual disk before replacement
        os.replace(tmp, out_path)     # Atomic: no intermediate state exists
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pair_id(split: str, dialect: str, dialect_prompt: str) -> str:
    h = hashlib.md5(f"{split}|{dialect}|{dialect_prompt}".encode("utf-8")).hexdigest()[:10]
    return f"{split}|{dialect}|{h}"


def load_bt_pairs(bt_dir=BT_DIR, splits=None, dialects=None, drop_empty=True):
    splits = splits or SPLITS
    dialects = dialects or DIALECTS
    frames = []
    for split in splits:
        for d in dialects:
            p = os.path.join(bt_dir, f"bt_{split}_{d}.csv")
            if not os.path.exists(p):
                print(f"  [warn] Missing: {p}")
                continue
            df = pd.read_csv(p)
            df["split"] = split
            df["dialect"] = d
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Cannot find bt_*.csv in {bt_dir}")

    out = pd.concat(frames, ignore_index=True)
    for c in ("standard_prompt", "back_translated", "dialect_prompt"):
        out[c] = out[c].astype(str).str.strip()
    out = out[out["back_translated"] != BLOCKED_SENTINEL].reset_index(drop=True)
    if drop_empty:
        bt = out["back_translated"]
        out = out[(bt.str.len() > 0) & (bt.str.lower() != "nan")].reset_index(drop=True)

    out["pair_id"] = [
        _pair_id(s, d, dp)
        for s, d, dp in zip(out["split"], out["dialect"], out["dialect_prompt"])
    ]
    cols = ["pair_id", "split", "dialect", "category",
            "standard_prompt", "dialect_prompt", "back_translated"]
    return out[[c for c in cols if c in out.columns]]


def load_done(out_path: str, key_col="pair_id", need_cols=None):
    """resume: Returns a set of 'valid scored' pair_ids from existing results and existing rows.

    If need_cols is provided, only rows where those columns are all non-null are considered done (blanks are retried).
    """
    if not os.path.exists(out_path):
        return set(), []
    prev = pd.read_csv(out_path)
    if key_col not in prev.columns:
        return set(), []
    if need_cols:
        mask = prev[need_cols].notna().all(axis=1)
        for c in need_cols:
            if prev[c].dtype == object:
                mask &= prev[c].astype(str).str.strip().ne("")
        done = set(prev.loc[mask, key_col])
    else:
        done = set(prev[key_col])
    return done, prev.to_dict("records")


def first_upper(text: str) -> str:
    """Same preprocessing as the paper's OMod script: capitalizes first letter after strip."""
    t = str(text).strip()
    return t[:1].upper() + t[1:] if t else t
