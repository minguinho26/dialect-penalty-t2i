"""
compute_within_category_baseline.py

CLIP similarity baseline definition:
Same-category baseline experiment to address the ambiguity in the CLIP text similarity table (Table 2, sec 4.1.1) regarding "what similarity is being measured against what."

WARNING: Metric alignment:
  Values in Table 2 are 'text-text' CLIP cosine (clip_text_cosine_sim).
  However, the existing baseline file clip_i_analyze_study/*_same_category_baseline.csv uses 'image-image' CLIP cosine (based on n_pairs, _get_image_features), which is a different metric. This script recalculates the same-category baseline using the 'exact same text metric' as Table 2 to ensure an apples-to-apples comparison.

Measurement targets (2 types):
  (1) SAE <-> Dialect : Mean pairwise cosine similarity between the original SAE prompt and the dialect translation (= values in the paper's table).
  (2) Within-category SAE baseline :
        Mean pairwise cosine similarity between two 'different' SAE prompts belonging to the same category.
        -> A baseline demonstrating "this level of expressive/semantic difference naturally exists even within the same category." If (1) is greater than (2), it means the dialect translation preserves meaning better than "two different sentences in the same category."

The embedding pipeline uses mean-pool + L2 normalize on the Compel-based CLIP text embedding, identical to the generation process (same method as safety_models.compute_clip_text_similarity).

Usage:
    python extended_analysis/compute_within_category_baseline.py
"""

import os
import copy

import numpy as np
import pandas as pd
import torch

from diffusers import StableDiffusionPipeline
from compel import Compel

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # make `common/` importable when run from the repository root
from common.common_utils import DIALECTS

device = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TYPES = ["toxic", "benign"]
ROOT = os.path.join(PROJECT_ROOT, "exp_image_level_promptguard")
OUT_DIR = os.path.join(PROJECT_ROOT,
                       "extended_analysis/results_within_category_baseline_text")
os.makedirs(OUT_DIR, exist_ok=True)

diffusion_model_name = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(
    diffusion_model_name,
    torch_dtype=torch.float32,
).to(device)
compel = Compel(
    tokenizer=copy.deepcopy(pipe.tokenizer),
    text_encoder=copy.deepcopy(pipe.text_encoder),
    truncate_long_prompts=False,
)
del pipe

_vec_cache: dict[str, np.ndarray] = {}


@torch.no_grad()
def get_vec(prompt: str) -> np.ndarray:
    """
    Creates a 768-d vector by mean-pooling + L2 normalizing a single prompt, identical to compute_clip_text_similarity. (numpy, normalized)
    """
    if prompt in _vec_cache:
        return _vec_cache[prompt]
    emb = compel(prompt)                 # (1, seq_len, 768)
    vec = emb.mean(dim=1)                # (1, 768)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    vec_np = vec.squeeze(0).float().cpu().numpy()
    _vec_cache[prompt] = vec_np
    return vec_np


def within_category_baseline(prompts_by_cat: dict[str, list[str]]):
    """
    Takes category -> [SAE prompt, ...], computes the mean pairwise cosine similarity between two different prompts per category.

    Returns: (per_category_df, macro_mean, macro_std)
        macro_mean = average of mean_sims per category (identical method to image baseline code)
    """
    rows = []
    for cat, prompts in prompts_by_cat.items():
        uniq = sorted(set(prompts))
        if len(uniq) < 2:
            continue
        mat = np.stack([get_vec(p) for p in uniq])   # (N, 768), already normalized
        sim = mat @ mat.T                             # (N, N)
        iu = np.triu_indices(len(uniq), k=1)          # Only different prompts
        pair = sim[iu]
        rows.append({
            "category": cat,
            "n_prompts": len(uniq),
            "n_pairs": len(pair),
            "mean_sim": float(pair.mean()),
            "std_sim": float(pair.std()),
        })
    per_cat = pd.DataFrame(rows)
    macro_mean = float(per_cat["mean_sim"].mean())
    macro_std = float(per_cat["mean_sim"].std())
    return per_cat, macro_mean, macro_std


for prompt_type in PROMPT_TYPES:
    print("\n" + "=" * 78)
    print(f" PROMPT TYPE = {prompt_type}")
    print("=" * 78)

    summary_rows = []
    pooled_by_cat: dict[str, list[str]] = {}

    for dialect in DIALECTS:
        csv_path = (
            f"{ROOT}/image_level_{prompt_type}_results_{dialect}/"
            f"image_level_{prompt_type}_analyze.csv"
        )
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)

        sae_dial_mean = float(df["clip_text_cosine_sim"].mean())
        sae_dial_std = float(df["clip_text_cosine_sim"].std())

        by_cat = {
            cat: g["std_prompt"].tolist()
            for cat, g in df.groupby("category")
        }
        for cat, ps in by_cat.items():
            pooled_by_cat.setdefault(cat, []).extend(ps)

        per_cat, base_mean, base_std = within_category_baseline(by_cat)

        # We no longer save per_cat to csv for each dialect because they are all identical
        # to the pooled SAE baseline (since the dataset is parallel).

        summary_rows.append({
            "Dialect": dialect,
            "N": len(df),
            "SAE<->Dial mean": round(sae_dial_mean, 4),
            "SAE<->Dial std": round(sae_dial_std, 4),
            "WithinCat baseline mean": round(base_mean, 4),
            "WithinCat baseline std": round(base_std, 4),
            "Gap (SAE-Dial - baseline)": round(sae_dial_mean - base_mean, 4),
        })

        print(
            f"  {dialect:8s} | N={len(df):5d} | "
            f"SAE<->Dial = {sae_dial_mean:.4f} | "
            f"WithinCat baseline = {base_mean:.4f} | "
            f"gap = {sae_dial_mean - base_mean:+.4f}"
        )

    if pooled_by_cat:
        per_cat_pool, pool_mean, pool_std = within_category_baseline(pooled_by_cat)
        per_cat_pool.insert(0, "dialect", "SAE_baseline")
        per_cat_pool.to_csv(
            f"{OUT_DIR}/{prompt_type}_SAE_baseline_per_cat.csv",
            index=False,
        )
        print(f"\n  [SAE Baseline across all dialects] "
              f"WithinCat baseline mean = {pool_mean:.4f} (std {pool_std:.4f})")
        print("  per-category:")
        print(per_cat_pool[["category", "n_prompts", "n_pairs",
                            "mean_sim", "std_sim"]].to_string(index=False))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT_DIR}/{prompt_type}_summary.csv", index=False)
    print(f"\n  [SUMMARY: {prompt_type}]")
    print(summary_df.to_string(index=False))

print(f"\nDone. Results saved in {OUT_DIR}/")
