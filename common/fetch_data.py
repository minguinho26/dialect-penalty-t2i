#!/usr/bin/env python3
"""fetch_data.py - Downloads prompt datasets from HuggingFace and saves them
in the local layout expected by the experiment scripts.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS

REPO_ID = "Minguinho-zeze/dialect-penalty-t2i"

RENAME = {"sae": "standard_prompt", **{d.lower(): f"{d}_prompt" for d in DIALECTS}}

# The column order expected by the scripts
COLS_TOXIC = ["category", "standard_prompt"] + [f"{d}_prompt" for d in DIALECTS]
COLS_BENIGN = ["category", "raw_caption", "standard_prompt"] + [f"{d}_prompt" for d in DIALECTS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="Save location (default: repo root)")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--skip-crosslingual", action="store_true")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("`datasets` is required:  pip install datasets", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)

    dst = out / "prompts_dataset"
    dst.mkdir(parents=True, exist_ok=True)
    for split, cols in [("toxic", COLS_TOXIC), ("benign", COLS_BENIGN)]:
        try:
            df = load_dataset(args.repo_id, "prompts", split=split).to_pandas()
        except Exception as e:
            print(f"\n'{split}' download failed: {type(e).__name__}: {e}", file=sys.stderr)
            print("\nThis dataset is gated. Please check the following:", file=sys.stderr)
            print(f"  1. Gain access approval at https://huggingface.co/datasets/{args.repo_id}", file=sys.stderr)
            print("  2. Log in using `hf auth login`", file=sys.stderr)
            return 1
        df = df.rename(columns=RENAME)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Missing expected columns: {missing}", file=sys.stderr)
            return 1
        p = dst / f"{split}_prompts.csv"
        df[cols].to_csv(p, index=False)
        print(f"  {p}  ({len(df):,} rows)")

    if not args.skip_crosslingual:
        try:
            cl = load_dataset(args.repo_id, "crosslingual_prompts", split="full").to_pandas()
        except Exception as e:
            print(f"  (Skipping crosslingual: {type(e).__name__})")
            return 0
        for lang, sub in cl.groupby("language"):
            d = out / "foreign_languages" / f"{lang}_translation_results"
            d.mkdir(parents=True, exist_ok=True)
            for label, s in sub.groupby("label"):
                p = d / f"{lang}_translated_{label}_prompts.csv"
                s.drop(columns=["language", "label"]).to_csv(p, index=False)
                print(f"  {p}  ({len(s):,} rows)")

    print("\nDone. You can now execute the stage1/stage2 scripts from the repo root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
