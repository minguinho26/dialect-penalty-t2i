"""score_bt_omod.py — Backtranslation Scoring (1/3): OpenAI Moderation (OMod).

For each original SAE (standard_prompt) vs back-translated SAE (back_translated),
we call omni-moderation-latest and store the **flag (boolean) as well as continuous scores per category**.
(difference test: Since both texts are SAE, the absolute error of OMod cancels out → Δ is reliable)

Uses the same model and preprocessing (first_upper) as the paper (ablation_openai_moderation_api.py) to match distributions.

Usage:
    export OPENAI_API_KEY=...
    python score_bt_omod.py --limit 10            
    python score_bt_omod.py                       

Artifacts: results/backtranslation_audit/omod_scores.csv
  columns: pair_id, split, dialect, category, standard_prompt, back_translated,
           std_flagged, bt_flagged, std_<cat>..., bt_<cat>...   (<cat>=OMod category_scores key)
"""
import os
import time
import argparse
import pandas as pd
from tqdm import tqdm

import sys
# Make `common/` importable when run from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtranslate_common import load_bt_pairs, load_done, first_upper, atomic_to_csv, OUT_DIR
from common.env_keys import get_openai_client


def moderate(client, text, cache):
    """Moderates 1 text. Returns (flagged, {category: score}). Returns (None, {}) on failure."""
    key = text
    if key in cache:
        return cache[key]
    try:
        resp = client.moderations.create(model="omni-moderation-latest", input=text)
        r = resp.results[0]
        raw = r.category_scores.model_dump() if hasattr(r.category_scores, "model_dump") \
            else dict(r.category_scores)
        # OMod SDK returns compound categories redundantly with both field name ("harassment_threatening")
        # and alias ("harassment/threatening"). We normalize '/' and '-' to '_' to dedupe into 13 canonical keys
        # (automatically deduplicated by the dict).
        scores = {k.replace("/", "_").replace("-", "_"): v for k, v in raw.items()}
        out = (bool(r.flagged), scores)
    except Exception as e:
        print(f"  ⚠ moderation error: {e}")
        out = (None, {})
    cache[key] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Top N pairs for quick check")
    ap.add_argument("--dialects", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--sleep", type=float, default=0.3, help="Wait between calls (sec); for RPM safety")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "omod_scores.csv"))
    args = ap.parse_args()

    from openai import OpenAI
    client = get_openai_client()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pairs = load_bt_pairs(splits=args.splits, dialects=args.dialects)
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"[OMod] Target {len(pairs)} pairs")

    done, rows = load_done(args.out, need_cols=["std_flagged", "bt_flagged"])
    if done:
        print(f"[OMod] resume: Skipping {len(done)} already scored pairs")
    cache = {}

    for _, r in tqdm(pairs.iterrows(), total=len(pairs), desc="[OMod]"):
        if r["pair_id"] in done:
            continue
        std_txt = first_upper(r["standard_prompt"])
        bt_txt = first_upper(r["back_translated"])
        std_flag, std_sc = moderate(client, std_txt, cache)
        bt_flag, bt_sc = moderate(client, bt_txt, cache)
        time.sleep(args.sleep)

        row = {
            "pair_id": r["pair_id"], "split": r["split"], "dialect": r["dialect"],
            "category": r.get("category", ""),
            "standard_prompt": r["standard_prompt"], "back_translated": r["back_translated"],
            "std_flagged": std_flag, "bt_flagged": bt_flag,
        }
        for k, v in std_sc.items():
            row[f"std_{k}"] = v
        for k, v in bt_sc.items():
            row[f"bt_{k}"] = v
        rows.append(row)
        atomic_to_csv(pd.DataFrame(rows), args.out, index=False)  # Real-time save (atomic for interruptions)

    ok = sum(1 for x in rows if x.get("std_flagged") is not None)
    print(f"[OMod] Done: {ok}/{len(rows)} → {args.out}")


if __name__ == "__main__":
    main()
