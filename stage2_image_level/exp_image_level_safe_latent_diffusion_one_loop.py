"""
End-to-End Safe Latent Diffusion (SLD) Bias
=====================================================================

4 Conditions (per prompt pair):
  (1) Standard + No Guard       <- SLD with sld_guidance_scale=0
  (2) Standard + SLD guarded    <- SLD with strong config
  (3) Dialect  + No Guard       <- SLD with sld_guidance_scale=0
  (4) Dialect  + SLD guarded    <- SLD with strong config

Usage:
  python exp_image_level_safe_latent_diffusion_one_loop.py --dialect all --prompt-type toxic
  python exp_image_level_safe_latent_diffusion_one_loop.py --dialect all --prompt-type benign
"""

import argparse
import copy
import os
import types
from pathlib import Path

import pandas as pd
import torch
import lpips
from PIL import Image
from tqdm import tqdm

import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.preflight import require

# measure_unsafe_diffusion() loads ./multihead_checkpoints/{head}.pt per image, so fail here
# with download instructions rather than deep inside the generation loop.
require("multihead")

from common.safety_evaluator import *
from common.common_utils import get_device, DIALECTS

from common.env_keys import hf_login
# Reads token only from environment variable (HF_TOKEN). Proceeds with anonymous access if not set.
hf_login()

ALL_DIALECTS = DIALECTS
INFERENCE_STEPS = 50
BASE_SEED = 42

SLD_CONFIG = {
    "guidance_scale": 10,
    "sld_guidance_scale": 2000,
    "sld_warmup_steps": 7,
    "sld_threshold": 0.025,
    "sld_momentum_scale": 0.5,
    "sld_mom_beta": 0.7,
}

def _encode_prompt_with_compel(
    self,
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt,
    enable_safety_guidance,
):
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    if isinstance(prompt, list):
        prompt_embeds = torch.cat([self._compel(p) for p in prompt], dim=0)
    else:
        prompt_embeds = self._compel(prompt)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got "
                f"{type(negative_prompt)} != {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, "
                f"but `prompt`: {prompt} has batch size {batch_size}."
            )
        else:
            uncond_tokens = negative_prompt

        if len(uncond_tokens) == 1:
            negative_prompt_embeds = self._compel(uncond_tokens[0])
        else:
            negative_prompt_embeds = torch.cat(
                [self._compel(t) for t in uncond_tokens], dim=0
            )

        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        if enable_safety_guidance:
            safety_embeddings = self._compel(self._safety_text_concept)

            seq_len = safety_embeddings.shape[1]
            safety_embeddings = safety_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            safety_embeddings = safety_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # Pads the 3 tensors to the maximum sequence length to prevent shape mismatch.
            padded_embeds = self._compel.pad_conditioning_tensors_to_same_length(
                [negative_prompt_embeds, prompt_embeds, safety_embeddings]
            )
            negative_prompt_embeds, prompt_embeds, safety_embeddings = padded_embeds

            prompt_embeds = torch.cat([
                negative_prompt_embeds, prompt_embeds, safety_embeddings
            ])
        else:
            # Pads the 2 tensors to the maximum sequence length.
            padded_embeds = self._compel.pad_conditioning_tensors_to_same_length(
                [negative_prompt_embeds, prompt_embeds]
            )
            negative_prompt_embeds, prompt_embeds = padded_embeds

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds


class SLDGenerator:
    def __init__(self, device):
        from diffusers import DiffusionPipeline
        from compel import Compel

        self.device = device
        print(f"📦 Loading SLD pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "AIML-TUDA/stable-diffusion-safe",
            cache_dir="./checkpoints", 
        ).to(device)

        compel = Compel(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            truncate_long_prompts = False)
        self.pipe._compel = compel

        self.pipe._encode_prompt = types.MethodType(
            _encode_prompt_with_compel, self.pipe
        )

        self._compel_eval = Compel(
            tokenizer=copy.deepcopy(self.pipe.tokenizer),
            text_encoder=copy.deepcopy(self.pipe.text_encoder),
            truncate_long_prompts = False
        )

        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        self.pipe.safety_checker = dummy_checker

        print(f"  ✓ SLD safety concept: {self.pipe.safety_concept[:80]}...")
        print(f"  ✓ SLD config: {SLD_CONFIG}")
        print(f"  ✓ Compel monkey-patch complete")
        print(f"  ✓ Built-in Safety Checker disabled (Dummy)")

    def generate(self, prompt: str, seed: int, use_guard: bool) -> Image.Image:
        generator = torch.Generator(self.device).manual_seed(seed)

        if use_guard:
            result = self.pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=INFERENCE_STEPS,
                **SLD_CONFIG,
            )
        else:
            result = self.pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=INFERENCE_STEPS,
                sld_guidance_scale=0,
            )

        return result.images[0]

    def compute_clip_text_similarity(self, text_a: str, text_b: str) -> float:
        with torch.no_grad():
            emb_a = self._compel_eval(text_a)
            emb_b = self._compel_eval(text_b)

            vec_a = emb_a.mean(dim=1)
            vec_b = emb_b.mean(dim=1)

            vec_a = vec_a / vec_a.norm(dim=-1, keepdim=True)
            vec_b = vec_b / vec_b.norm(dim=-1, keepdim=True)

            cosine_sim = (vec_a @ vec_b.T).item()

        return cosine_sim


def evaluate_guardrail(base_img_pil, guard_img_pil):
    """
    base_img_pil: original image generated without guardrail (I_base)
    guard_img_pil: image generated with guardrail applied (I_guard)
    """
    
    base_scores = evaluator.evaluate(base_img_pil)
    guard_scores = evaluator.evaluate(guard_img_pil)
    
    clip_i_score = evaluator.image_image_similarity(base_img_pil, guard_img_pil)
   
    base_tensor = transform_lpips(base_img_pil).unsqueeze(0).to(device)
    guard_tensor = transform_lpips(guard_img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lpips_score = lpips_metric(base_tensor, guard_tensor).item()
        
    return {
        "base": {
            "nsfw_i": base_scores["nsfw_i_score"],
            "q16": base_scores["q16_score"]
        },
        "guard": {
            "nsfw_i": guard_scores["nsfw_i_score"],
            "q16": guard_scores["q16_score"]
        },
        "clip_i": clip_i_score,                  # Closer to 1.0 indicates better content preservation
        "lpips": lpips_score                     # Closer to 0 indicates less change in the image (less guardrail intervention)
    }


def load_all_data(csv_path: Path, dialects: list) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cols_to_keep = ["category", "standard_prompt"]
    
    for d in dialects:
        dial_col = f"{d}_prompt"
        if dial_col in df.columns:
            cols_to_keep.append(dial_col)
        else:
            print(f"⚠️ Warning: Column '{dial_col}' not found in dataset. Skipping.")
            
    out = df[cols_to_keep].copy()
    out = out.dropna(subset=["standard_prompt"]).reset_index(drop=True)
    print(f"  ✓ Multi-dialect data loaded: {len(out)} prompts, {out['category'].nunique()} categories")
    return out


def load_checkpoint(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    done = set(zip(df["category"], df["prompt_id"], df["dialect_type"]))
    print(f"  ✓ Checkpoint: {len(done)} experiment settings processed")
    return done


def append_result(csv_path: Path, row: dict):
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def run_experiment(dialects: list, prompt_type: str, pilot: int = 0, resume: bool = False):
    global device
    device = get_device()

    print(f"\n{'='*60}")
    print(f"SLD {prompt_type.capitalize()} Multi-Dialect Experiment")
    print(f"Dialects: {dialects}")
    print(f"Device: {device} | Pilot: {pilot if pilot > 0 else 'OFF (full)'}")
    print(f"{'='*60}")

    DF_CSV_PATH = Path(f"./prompts_dataset/{prompt_type}_prompts.csv")
    data = load_all_data(DF_CSV_PATH, dialects)

    generator = SLDGenerator(device=device)

    if pilot > 0:
        data = data.groupby("category").head(pilot).reset_index(drop=True)
        print(f"  ✓ Pilot mode: {pilot} per category -> Total {len(data)}")

    total_processed = 0
    categories = data["category"].unique()
    
    base_out_dir = Path("exp_results")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        cat_data = data[data["category"] == cat]
        print(f"\n▶ {cat} ({len(cat_data)} prompts)")

        for local_idx, (_, item) in enumerate(tqdm(cat_data.iterrows(), total=len(cat_data), desc=cat)):
            
            seed = BASE_SEED + local_idx
            std_prompt = str(item["standard_prompt"]).strip()
            std_prompt = std_prompt[0].upper() + std_prompt[1:]
            
            # Tracks whether the Standard image has been generated to prevent duplicate generation.
            std_images_generated = False
            std_noguard_image, std_guard_image = None, None
            std_results, std_noguard_clip_score, std_guard_clip_score = None, None, None
            std_noguard_unsafe, std_guard_unsafe = None, None

            for dialect in dialects:
                dial_col = f"{dialect}_prompt"
                if dial_col not in item or pd.isna(item[dial_col]):
                    continue
                
                dial_out_dir = base_out_dir / f"image_level_{prompt_type}_results_{dialect}"
                img_dir = dial_out_dir / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                
                csv_path = dial_out_dir / f"image_level_{prompt_type}_analyze.csv"
                
                done_set = load_checkpoint(csv_path) if resume else set()
                if not resume and csv_path.exists():
                    csv_path.unlink()

                if (cat, local_idx, dialect) in done_set:
                    continue
                
                std_noguard_img_path = img_dir / f"{cat}_{local_idx}_std_noguard.png"
                std_guard_img_path = img_dir / f"{cat}_{local_idx}_std_guarded.png"
                
                if not std_images_generated:
                    std_noguard_image = generator.generate(std_prompt, seed=seed, use_guard=False)
                    std_guard_image = generator.generate(std_prompt, seed=seed, use_guard=True)
                    
                    std_results = evaluate_guardrail(std_noguard_image, std_guard_image)
                    std_noguard_clip_score = evaluator.text_image_similarity(std_noguard_image, std_prompt)
                    std_guard_clip_score = evaluator.text_image_similarity(std_guard_image, std_prompt)
                    
                    std_noguard_unsafe = measure_unsafe_diffusion(std_noguard_image)
                    std_guard_unsafe = measure_unsafe_diffusion(std_guard_image)
                    std_images_generated = True
                
                # Copies the Standard image to each dialect folder for easier comparison later.
                std_noguard_image.save(std_noguard_img_path)
                std_guard_image.save(std_guard_img_path)

                dial_prompt = str(item[dial_col]).strip()
                dial_prompt = dial_prompt[0].upper() + dial_prompt[1:]
                
                dial_noguard_img_path = img_dir / f"{cat}_{local_idx}_dial_noguard.png"
                dial_guard_img_path = img_dir / f"{cat}_{local_idx}_dial_guarded.png"
                
                dial_noguard_image = generator.generate(dial_prompt, seed=seed, use_guard=False)
                dial_noguard_image.save(dial_noguard_img_path)
                
                dial_guard_image = generator.generate(dial_prompt, seed=seed, use_guard=True)
                dial_guard_image.save(dial_guard_img_path)
                
                dial_results = evaluate_guardrail(dial_noguard_image, dial_guard_image)
                
                sim_t_sae_i_dial_noguard = evaluator.text_image_similarity(dial_noguard_image, std_prompt)
                sim_t_sae_i_dial_guarded = evaluator.text_image_similarity(dial_guard_image, std_prompt)
                
                dial_noguard_clip_score_biased = evaluator.text_image_similarity(dial_noguard_image, dial_prompt)
                dial_guard_clip_score_biased = evaluator.text_image_similarity(dial_guard_image, dial_prompt)
                
                clip_text_sim = evaluator.text_text_similarity(std_prompt, dial_prompt)
                clip_i_score_between_std_and_dial = evaluator.image_image_similarity(std_noguard_image, dial_noguard_image)
                
                dial_noguard_unsafe = measure_unsafe_diffusion(dial_noguard_image)
                dial_guard_unsafe = measure_unsafe_diffusion(dial_guard_image)

                rel_std_noguard = f"images/{std_noguard_img_path.name}"
                rel_std_guarded = f"images/{std_guard_img_path.name}"
                rel_dial_noguard = f"images/{dial_noguard_img_path.name}"
                rel_dial_guarded = f"images/{dial_guard_img_path.name}"

                row = {
                    "prompt_id": local_idx,
                    "category": cat,
                    "dialect_type": dialect,
                    "std_prompt": std_prompt,
                    "dial_prompt": dial_prompt,

                    "std_noguard_img": rel_std_noguard,
                    "std_guarded_img": rel_std_guarded,
                    "dial_noguard_img": rel_dial_noguard,
                    "dial_guarded_img": rel_dial_guarded,

                    "std_base_nsfw_i": std_results["base"]["nsfw_i"],
                    "std_base_q16": std_results["base"]["q16"],
                    "std_guard_nsfw_i": std_results["guard"]["nsfw_i"],
                    "std_guard_q16": std_results["guard"]["q16"],
                    "std_clip_i_between_base_guard": std_results["clip_i"],
                    "std_lpips_between_base_guard": std_results["lpips"],

                    "dial_base_nsfw_i": dial_results["base"]["nsfw_i"],
                    "dial_base_q16": dial_results["base"]["q16"],
                    "dial_guard_nsfw_i": dial_results["guard"]["nsfw_i"],
                    "dial_guard_q16": dial_results["guard"]["q16"],
                    "dial_clip_i_between_base_guard": dial_results["clip_i"],
                    "dial_lpips_between_base_guard": dial_results["lpips"],
                    
                    "clip_text_cosine_sim": clip_text_sim,
                    
                    "std_noguard_clip_score": std_noguard_clip_score,
                    "std_guard_clip_score": std_guard_clip_score,
                    
                    "sim_t_sae_i_dial_noguard": sim_t_sae_i_dial_noguard,
                    "sim_t_sae_i_dial_guarded": sim_t_sae_i_dial_guarded,
                    
                    "dial_noguard_clip_score_biased": dial_noguard_clip_score_biased,
                    "dial_guard_clip_score_biased": dial_guard_clip_score_biased,

                    "clip_i_score_between_std_and_dial": clip_i_score_between_std_and_dial
                }
                
                row.update({f"std_noguard_unsafe_{k}": v for k, v in std_noguard_unsafe.items()})
                row.update({f"std_guard_unsafe_{k}": v for k, v in std_guard_unsafe.items()})
                row.update({f"dial_noguard_unsafe_{k}": v for k, v in dial_noguard_unsafe.items()})
                row.update({f"dial_guard_unsafe_{k}": v for k, v in dial_guard_unsafe.items()})

                append_result(csv_path, row)
                total_processed += 1

    print(f"\n✅ Completed: {total_processed} dialect pairs processed")
    print(f"📁 Results are saved under '{base_out_dir}' separated by dialect!")
    return base_out_dir

def main():
    parser = argparse.ArgumentParser(description="SLD: Over-Censorship Evaluation")
    parser.add_argument("--dialect", type=str, required=True, help="AAVE / ChcE / CollSgE / IndE / JamE / all")
    parser.add_argument("--pilot", type=int, default=0, help="N per category (0 = all)")
    parser.add_argument("--prompt-type", type=str, default='benign', help="benign or toxic")
    args = parser.parse_args()

    dialects = ALL_DIALECTS if args.dialect == "all" else [args.dialect]
    
    run_experiment(dialects, args.prompt_type, pilot=args.pilot, resume=True)


if __name__ == "__main__":
    main()