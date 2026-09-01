"""exp_direction_typo_vs_dialect.py — Dialect directionality analysis.

Motivation:
  "typo and dialect rewrite might be qualitatively different perturbations even if
   the magnitude of CLIP embedding displacement is the same (the representation-direction shift might differ)."

We already controlled the 'magnitude' of displacement using binary search to match sim(SAE, Typo) ≈ sim(SAE, Dialect) (Table 3, 9).
This script quantifies the remaining axis, i.e., 'direction', and shows that the directional difference is significant using two baselines.

For each prompt, extract the CLIP text embedding (L2-normalized unit vector) and define the displacement from SAE:
  d_dial[d]      = e_dialect_d - e_SAE
  d_typo[d,s]    = e_typo_{d,s} - e_SAE

Measure three directional alignments (cosine):
  (1) dial_typo  : cos(d_dial[d], d_typo[d,s])           ← Core metric
  (2) typo_typo  : cos(d_typo[d,si], d_typo[d,sj])       ← Baseline: directional consistency between typos (low if random)
  (3) dial_dial  : cos(d_dial[di], d_dial[dj])           ← Baseline: directional consistency between dialects (high if structural)

Interpretation:
  - dial_dial is high + typo_typo is low + dial_typo is low
    → dialect moves in a 'consistent specific direction', typo moves in a 'random direction'.
      Thus, even with matched magnitude, dialect penalty cannot be reduced to generic typo.

Additionally, it re-verifies the sim values to check the apparent mismatch in magnitude for benign prompts:
  - Internal embedding sim (e_SAE·e_dial, e_SAE·e_typo)
  - Saved sim in data (from the paper pipeline during binary search; {d}_target_sim_s*, {d}_typo_sim_s*)
  Comparing both pipeline's sim side-by-side checks if matching holds at the sim level.

Pure direction calculation logic (numpy) is separated into displacement_stats / pair_direction_cos → easy to unit test.

Usage:
    python exp_direction_typo_vs_dialect.py                 
    python exp_direction_typo_vs_dialect.py --limit 20      
    python exp_direction_typo_vs_dialect.py --split toxic   
"""

import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # make `common/` importable when run from the repository root

from common.common_utils import DIALECTS, SEEDS

TYPO_FILES = {
    "benign": os.path.join(ROOT, "text_level_typo_results/benign_prompts_with_multiseed_typos.csv"),
    "toxic": os.path.join(ROOT, "text_level_typo_results/toxic_prompts_with_multiseed_typos.csv"),
}
OUT_DIR = os.path.join(ROOT, "results")


# ─────────────────────────────────────────────────────────────
# Pure direction calculation logic (numpy only → back-end independent unit testing possible)
# ─────────────────────────────────────────────────────────────
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def displacement_stats(e_sae: np.ndarray, e_dial: np.ndarray, e_typo: np.ndarray) -> dict:
    """Returns direction alignment, magnitude, and sim of dialect/typo displacement relative to SAE.

    e_* are assumed to be L2-normalized CLIP text embeddings (unit vectors).
    sim_* is the cosine similarity with e_SAE (dot product since they are unit vectors).
    """
    d_dial = e_dial - e_sae
    d_typo = e_typo - e_sae
    return {
        "dir_cos": _cosine(d_dial, d_typo),         # Core: alignment of the two movement directions
        "mag_dial": float(np.linalg.norm(d_dial)),
        "mag_typo": float(np.linalg.norm(d_typo)),
        "sim_dial": float(np.dot(e_sae, e_dial)),   # Internal sim
        "sim_typo": float(np.dot(e_sae, e_typo)),
    }


def pair_direction_cos(e_sae: np.ndarray, e_a: np.ndarray, e_b: np.ndarray) -> float:
    """Displacement direction cosine between two targets (a, b) relative to SAE. Used for baselines (typo-typo, dial-dial)."""
    return _cosine(e_a - e_sae, e_b - e_sae)


# ─────────────────────────────────────────────────────────────
# CLIP text embedding (Robust version)
# ─────────────────────────────────────────────────────────────
def clip_text_unit(evaluator, text: str) -> np.ndarray:
    """L2-normalized CLIP text embedding (numpy), **without truncation**.

    In order for the "same magnitude, different direction" premise to hold, the direction must be measured
    in the **same space** as the metric used by the paper to match typo magnitude. That metric is
    `SafetyEvaluator.text_text_similarity` → `_get_text_features`, which is
    `CLIPModel.get_text_features` (with text_projection applied, joint 768-d) + 75 token chunk
    mean-pool. Here we **exactly replicate** that logic using only transformers CLIPModel (no clip/open_clip/lpips),
    including chunking to avoid truncation.
    """
    import torch

    model = evaluator.clip_model
    proc = evaluator.clip_processor
    device = evaluator.device
    tok = proc.tokenizer

    # NOTE: There's a bug in this transformers version where CLIPModel.get_text_features() returns a ModelOutput
    #       instead of a tensor. So we directly call the internals of get_text_features (text_model → pooler_output →
    #       text_projection) to get the same projected feature.
    def _proj_feat(input_ids, attention_mask):
        out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return model.text_projection(out.pooler_output)          # (1, 768), before normalization

    tokens = tok(text, truncation=False, add_special_tokens=False).input_ids
    chunk_size = tok.model_max_length - 2        # 77 - (BOS, EOS) = 75
    bos, eos = tok.bos_token_id, tok.eos_token_id

    with torch.no_grad():
        if len(tokens) <= chunk_size:
            inputs = proc(text=[text], return_tensors="pt", truncation=True).to(device)
            feats = _proj_feat(inputs["input_ids"], inputs.get("attention_mask"))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        else:
            chunks = []
            for i in range(0, len(tokens), chunk_size):
                ids = [bos] + tokens[i:i + chunk_size] + [eos]
                input_ids = torch.tensor([ids]).to(device)
                attn = torch.ones(1, len(ids), dtype=torch.long).to(device)
                chunks.append(_proj_feat(input_ids, attn))       # Per-chunk (before norm) feature
            mean_feats = torch.cat(chunks, dim=0).mean(dim=0, keepdim=True)
            feats = mean_feats / mean_feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).detach().cpu().float().numpy().astype(np.float64)


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ─────────────────────────────────────────────────────────────
# Row-level processing: Generates three types of records
# ─────────────────────────────────────────────────────────────
def process_row(row, embed, split: str):
    """Generates dial_typo / typo_typo / dial_dial records for one base prompt."""
    e_sae = embed(row.get("standard_prompt"))
    if e_sae is None:
        return [], [], []

    e_dial = {d: embed(row.get(f"{d}_prompt")) for d in DIALECTS}
    e_typo = {(d, s): embed(row.get(f"{d}_typo_s{s}")) for d in DIALECTS for s in SEEDS}

    rec_dt, rec_tt, rec_dd = [], [], []

    for d in DIALECTS:
        if e_dial[d] is None:
            continue
        for s in SEEDS:
            if e_typo[(d, s)] is None:
                continue
            st = displacement_stats(e_sae, e_dial[d], e_typo[(d, s)])
            st.update({
                "split": split, "dialect": d, "seed": s,
                # Include saved sim (from paper pipeline) for benign matching re-verification
                "sim_dial_saved": _to_float(row.get(f"{d}_target_sim_s{s}")),
                "sim_typo_saved": _to_float(row.get(f"{d}_typo_sim_s{s}")),
            })
            rec_dt.append(st)

    for d in DIALECTS:
        for si, sj in itertools.combinations(SEEDS, 2):
            a, b = e_typo[(d, si)], e_typo[(d, sj)]
            if a is None or b is None:
                continue
            rec_tt.append({
                "split": split, "dialect": d, "seed_i": si, "seed_j": sj,
                "cos": pair_direction_cos(e_sae, a, b),
            })

    for di, dj in itertools.combinations(DIALECTS, 2):
        a, b = e_dial[di], e_dial[dj]
        if a is None or b is None:
            continue
        rec_dd.append({
            "split": split, "dialect_i": di, "dialect_j": dj,
            "cos": pair_direction_cos(e_sae, a, b),
        })

    return rec_dt, rec_tt, rec_dd


def run(split: str, evaluator, limit: int | None = None):
    df = pd.read_csv(TYPO_FILES[split])
    if limit is not None:
        df = df.head(limit)

    emb_cache: dict[str, np.ndarray] = {}

    def embed(text):
        if not isinstance(text, str) or text.strip() == "" or text.lower() == "nan":
            return None
        if text not in emb_cache:
            emb_cache[text] = clip_text_unit(evaluator, text)
        return emb_cache[text]

    dt_all, tt_all, dd_all = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{split}]"):
        dt, tt, dd = process_row(row, embed, split)
        dt_all.extend(dt); tt_all.extend(tt); dd_all.extend(dd)

    return pd.DataFrame(dt_all), pd.DataFrame(tt_all), pd.DataFrame(dd_all)


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────
def summarize_main(dt: pd.DataFrame) -> pd.DataFrame:
    return (
        dt.groupby(["split", "dialect"])
        .agg(
            dial_typo_cos_mean=("dir_cos", "mean"),
            dial_typo_cos_std=("dir_cos", "std"),
            mag_dial_mean=("mag_dial", "mean"),
            mag_typo_mean=("mag_typo", "mean"),
            sim_dial_mine=("sim_dial", "mean"),
            sim_typo_mine=("sim_typo", "mean"),
            sim_dial_saved=("sim_dial_saved", "mean"),
            sim_typo_saved=("sim_typo_saved", "mean"),
            n=("dir_cos", "count"),
        )
        .reset_index()
    )


def summarize_baseline(df: pd.DataFrame, name: str, by=("split",)) -> pd.DataFrame:
    by = list(by)
    return (
        df.groupby(by)
        .agg(**{f"{name}_cos_mean": ("cos", "mean"),
                f"{name}_cos_std": ("cos", "std"),
                "n": ("cos", "count")})
        .reset_index()
    )


def build_evaluator():
    """Evaluator for CLIP text embedding.

    clip_text_unit only references evaluator.clip_model/.clip_processor/.device.
    To use the same space as the paper's `_get_text_features`, we need `CLIPModel.get_text_features`,
    so we directly load transformers `CLIPModel` (openai/clip-vit-large-patch14, cached).
    We bypass loading the full SafetyEvaluator because it imports clip/open_clip/lpips at top-level
    which breaks in this environment.
    """
    from common.common_utils import get_device
    from transformers import CLIPModel, CLIPProcessor
    import types

    device = get_device()
    MID = "openai/clip-vit-large-patch14"
    ev = types.SimpleNamespace()
    ev.device = device
    ev.clip_model = CLIPModel.from_pretrained(MID).to(device).eval()
    ev.clip_processor = CLIPProcessor.from_pretrained(MID)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["benign", "toxic", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="Top N for quick verification")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    evaluator = build_evaluator()

    splits = ["benign", "toxic"] if args.split == "both" else [args.split]
    dts, tts, dds = [], [], []
    for s in splits:
        dt, tt, dd = run(s, evaluator, args.limit)
        dts.append(dt); tts.append(tt); dds.append(dd)
    dt = pd.concat(dts, ignore_index=True)
    tt = pd.concat(tts, ignore_index=True)
    dd = pd.concat(dds, ignore_index=True)

    dt.to_csv(os.path.join(args.out_dir, "direction_typo_vs_dialect_per_pair.csv"), index=False)
    main_sum = summarize_main(dt)
    tt_sum = summarize_baseline(tt, "typo_typo", by=["split", "dialect"])
    dd_sum = summarize_baseline(dd, "dial_dial")
    main_sum.to_csv(os.path.join(args.out_dir, "direction_summary_main.csv"), index=False)
    tt_sum.to_csv(os.path.join(args.out_dir, "direction_baseline_typo_typo.csv"), index=False)
    dd_sum.to_csv(os.path.join(args.out_dir, "direction_baseline_dialect_dialect.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== [MAIN] dialect-typo displacement direction (SAE anchor) ===")
    print("Lower dial_typo_cos means 'same magnitude, different direction'.")
    print(main_sum.round(4).to_string(index=False))

    print("\n=== [BASELINE] typo-typo (per dialect; same target, seed pairs) — for same-magnitude comparison with dial_typo ===")
    print(tt_sum.round(4).to_string(index=False))

    print("\n=== [BASELINE] dialect-dialect (same prompt, dialect pairs) — high if structural ===")
    print(dd_sum.round(4).to_string(index=False))

    print("\n=== [SANITY] sim matching re-verification (dialect vs typo; mine=internal embedding, saved=data saved value) ===")
    cols = ["split", "dialect", "sim_dial_mine", "sim_typo_mine", "sim_dial_saved", "sim_typo_saved"]
    print(main_sum[cols].round(4).to_string(index=False))

    print(f"\n[saved] {args.out_dir}/direction_summary_main.csv (+ typo_typo, dialect_dialect baselines, per_pair)")


if __name__ == "__main__":
    main()
