"""
Phase 2: End-to-End PromptGuard Bias
=====================================================

4 Conditions (per prompt):
  (1) Standard + No Guard       <- SD baseline
  (2) Standard + PromptGuard    <- Standard guarded
  (3) Dialect  + No Guard       <- Dialect baseline
  (4) Dialect  + PromptGuard    <- Dialect guarded

3 Metrics (per image):
  (a) NSFW-I score   - false positive indicator
  (b) Q16 score      - false positive indicator
  (c) CLIP-sim       - Context preservation (measures over-steering)

Usage:
  python exp_image_level_promptguard.py --dialect ChcE --pilot 5

  python exp_image_level_promptguard.py --dialect AAVE

  python exp_image_level_promptguard.py --dialect all

  python exp_image_level_promptguard.py --dialect ChcE --resume
"""

import argparse
from pathlib import Path
import lpips

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.safety_models import SafetyEvaluator
from common.common_utils import get_device, DIALECTS

ALL_DIALECTS = DIALECTS
SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
INFERENCE_STEPS = 50
BASE_SEED = 42

PROMPTGUARD_TOKENS = [
    "<pg_disturbing>", "<pg_political>", "<pg_self_harm>",
    "<pg_sexual>", "<pg_violent>",
]
PROMPTGUARD_WEIGHTS = [
    "./Promptguard_weights/disturbing.safetensors",
    "./Promptguard_weights/political.safetensors",
    "./Promptguard_weights/self_harm.safetensors",
    "./Promptguard_weights/sextual.safetensors",
    "./Promptguard_weights/violent.safetensors",
]
GUARD_SUFFIX = " " + " ".join(PROMPTGUARD_TOKENS)

device = "cuda" if torch.cuda.is_available() else "cpu"

evaluator = SafetyEvaluator(device=device)
lpips_metric = lpips.LPIPS(net='vgg').to(device)

transform_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class ImageGenerator:
    def __init__(self, device):
        from diffusers import StableDiffusionPipeline
        from compel import Compel

        self.device = device
        print(f"📦 Loading SD v1.4...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID, torch_dtype=torch.float16,
        ).to(device)

        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        self.pipe.safety_checker = dummy_checker
        print("  ✓ Safety Checker disabled")

        try:
            self.pipe.load_textual_inversion(
                pretrained_model_name_or_path=PROMPTGUARD_WEIGHTS,
                token=PROMPTGUARD_TOKENS,
            )
            print("  ✓ PromptGuard embedding loaded")
        except Exception as e:
            print(f"  ⚠ Failed to load PromptGuard: {e}")
            raise

        self.compel = Compel(tokenizer=self.pipe.tokenizer, 
                             text_encoder=self.pipe.text_encoder,
                             truncate_long_prompts = False)

    def generate(self, prompt: str, seed: int, use_guard: bool) -> Image.Image:
        final_prompt = prompt + GUARD_SUFFIX if use_guard else prompt
        
        prompt_embeds = self.compel(final_prompt)
        negative_prompt_embeds = self.compel("")  # SD uses an empty negative prompt by default
        
        # Pads embeddings to the same length to prevent shape mismatch errors during CFG computation.
        [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        # Must explicitly provide both positive and negative embeddings
        return self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            num_inference_steps=INFERENCE_STEPS,
        ).images[0]


def load_data(dialect: str, csv_path: Path) -> pd.DataFrame:
    """
    Returns (standard_prompt, dialect_prompt) pairs from CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    dial_col = f"{dialect}_prompt"
    assert dial_col in df.columns, f"Column '{dial_col}' not found. Available: {list(df.columns)}"

    out = df[["category", "standard_prompt", dial_col]].copy()
    out = out.rename(columns={dial_col: "dialect_prompt"})
    out = out.dropna(subset=["standard_prompt", "dialect_prompt"]).reset_index(drop=True)

    print(f"  ✓ Data loaded: {len(out)} prompts, "
          f"{out['category'].nunique()} categories, dialect={dialect}")
    return out


def load_checkpoint(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    done = set(zip(df["category"], df["prompt_id"]))
    print(f"  ✓ Checkpoint: {len(done)} prompts processed")
    return done


def append_result(csv_path: Path, row: dict):
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


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

# (label, prompt_source, use_guard)
CONDITIONS = [
    ("std_noguard",  "std",  False),
    ("std_guarded",  "std",  True),
    ("dial_noguard", "dial", False),
    ("dial_guarded", "dial", True),
]


def run_experiment(dialect: str, prompt_type: str, pilot: int = 0, resume: bool = False):

    device = get_device()

    print(f"\n{'='*60}")
    print(f"Promptguard {prompt_type.capitalize()} Experiment: {dialect}")
    print(f"Device: {device} | Pilot: {pilot if pilot > 0 else 'OFF (full)'}")
    print(f"{'='*60}")

    out_dir = Path(f"./exp_image_level_promptguard/image_level_{prompt_type}_results_{dialect}")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"image_level_{prompt_type}_analyze.csv"

    done_set = load_checkpoint(csv_path) if resume else set()
    if not resume and csv_path.exists():
        csv_path.unlink()

    DF_CSV_PATH = Path(f"./prompts_dataset/{prompt_type}_prompts.csv")
    data = load_data(dialect, DF_CSV_PATH)

    generator = ImageGenerator(device=device)
    
    if pilot > 0:
        data = data.groupby("category").head(pilot).reset_index(drop=True)
        print(f"  ✓ Pilot mode: {pilot} per category -> Total {len(data)}")

    total_processed = 0
    categories = data["category"].unique()

    for cat in categories:
        cat_data = data[data["category"] == cat]
        print(f"\n▶ {cat} ({len(cat_data)} prompts)")

        for local_idx, (_, item) in enumerate(tqdm(cat_data.iterrows(),
                                                    total=len(cat_data), desc=cat)):
            if (cat, local_idx) in done_set:
                continue

            seed = BASE_SEED + local_idx
            std_prompt = item["standard_prompt"]
            dial_prompt = item["dialect_prompt"]

            std_prompt = std_prompt.strip()
            std_prompt = str(std_prompt)[0].upper() + str(std_prompt)[1:]

            dial_prompt = dial_prompt.strip()
            dial_prompt = str(dial_prompt)[0].upper() + str(dial_prompt)[1:]

            save_path_dict = {}

            cond_images = {}

            for cond_label, prompt_src, use_guard in CONDITIONS:
                prompt_text = std_prompt if prompt_src == "std" else dial_prompt

                img = generator.generate(prompt_text, seed=seed, use_guard=use_guard)

                img_path = img_dir / f"{cat}_{local_idx}_{cond_label}.png"
                img.save(img_path)

                cond_images[cond_label] = img
                save_path_dict[f"{cond_label}_img"] = str(img_path).replace('exp_image_level_promptguard/', '')

            std_noguard_image = cond_images["std_noguard"]
            std_guard_image = cond_images["std_guarded"]

            dial_noguard_image = cond_images["dial_noguard"]
            dial_guard_image = cond_images["dial_guarded"]

            std_results = evaluate_guardrail(std_noguard_image, std_guard_image)
            dial_results = evaluate_guardrail(dial_noguard_image, dial_guard_image)
           
            std_noguard_clip_score = evaluator.text_image_similarity(std_noguard_image, std_prompt)
            std_guard_clip_score = evaluator.text_image_similarity(std_guard_image, std_prompt)

            # Evaluation with dialect prompt as anchor (for appendix/defense)
            dial_noguard_clip_score = evaluator.text_image_similarity(dial_noguard_image, dial_prompt)
            dial_guard_clip_score = evaluator.text_image_similarity(dial_guard_image, dial_prompt)
            
            # Main evaluation metric uses standard prompt as the anchor for consistency
            sim_t_sae_i_dial_noguard = evaluator.text_image_similarity(dial_noguard_image, std_prompt)
            sim_t_sae_i_dial_guarded = evaluator.text_image_similarity(dial_guard_image, std_prompt)

            clip_text_sim = evaluator.text_text_similarity(std_prompt, dial_prompt)
            clip_i_score_between_std_and_dial = evaluator.image_image_similarity(std_noguard_image, dial_noguard_image)

            row = {
                "prompt_id": local_idx,
                "category": cat,
                "std_prompt": std_prompt,
                "dial_prompt": dial_prompt,

                "std_noguard_img": save_path_dict['std_noguard_img'],
                "std_guarded_img": save_path_dict['std_guarded_img'],
                "dial_noguard_img": save_path_dict['dial_noguard_img'],
                "dial_guarded_img": save_path_dict['dial_guarded_img'],

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
                "dial_noguard_clip_score": dial_noguard_clip_score,
                "dial_guard_clip_score": dial_guard_clip_score,

                "clip_i_score_between_std_and_dial": clip_i_score_between_std_and_dial,
                
                "sim_t_sae_i_dial_noguard": sim_t_sae_i_dial_noguard,
                "sim_t_sae_i_dial_guarded": sim_t_sae_i_dial_guarded
            }

            append_result(csv_path, row)
            total_processed += 1

    print(f"\n✅ Completed: {total_processed} prompts")
    print(f"📁 Result: {csv_path}")
    return csv_path

def main():
    parser = argparse.ArgumentParser(
        description="PromptGuard Evaluation"
    )
    parser.add_argument("--dialect", type=str, required=True,
                        help="AAVE / ChcE / CollSgE / IndE / JamE / all")
    parser.add_argument("--pilot", type=int, default=0,
                        help="N per category (0 = all)")
    parser.add_argument("--prompt-type", type=str, default='benign',
                        help="benign or toxic")
    args = parser.parse_args()

    dialects = ALL_DIALECTS if args.dialect == "all" else [args.dialect]

    for dialect in dialects:
        assert dialect in ALL_DIALECTS, f"Unknown dialect: {dialect}"
        run_experiment(dialect, args.prompt_type, pilot=args.pilot, resume=True)


if __name__ == "__main__":
    main()