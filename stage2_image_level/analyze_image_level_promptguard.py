import pandas as pd

from PIL import Image
import torch
import lpips
from torchvision import transforms
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.safety_evaluator import SafetyEvaluator

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

evaluator = SafetyEvaluator(device=device)
lpips_metric = lpips.LPIPS(net='vgg').to(device)

transform_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


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


for prompt_type in ['toxic']: # 'toxic', 'benign'
    for dialect in ["AAVE", "ChcE", "CollSgE", "IndE", "JamE"]: # , "AAVE", "ChcE", "CollSgE", "IndE", "JamE"
        
        csv_path = f'./exp_image_level_promptguard/image_level_{prompt_type}_results_{dialect}/image_level_scores_{dialect}.csv'
        df = pd.read_csv(csv_path)

        results = []

        for idx, row in tqdm(df.iterrows(), desc = f"Evaluating {prompt_type} prompts for {dialect}"):

            std_noguard_image = Image.open(f"./exp_image_level_promptguard/{row['std_noguard_img']}")
            std_guard_image = Image.open(f"./exp_image_level_promptguard/{row['std_guarded_img']}")

            dial_noguard_image = Image.open(f"./exp_image_level_promptguard/{row['dial_noguard_img']}")
            dial_guard_image = Image.open(f"./exp_image_level_promptguard/{row['dial_guarded_img']}")

            std_results = evaluate_guardrail(std_noguard_image, std_guard_image)
            dial_results = evaluate_guardrail(dial_noguard_image, dial_guard_image)

            std_noguard_clip_score = evaluator.text_image_similarity(std_noguard_image, row['std_prompt'])
            std_guard_clip_score = evaluator.text_image_similarity(std_guard_image, row['std_prompt'])
            
            dial_noguard_clip_score = evaluator.text_image_similarity(dial_noguard_image, row['dial_prompt'])
            dial_guard_clip_score = evaluator.text_image_similarity(dial_guard_image, row['dial_prompt'])
            
            clip_text_sim = evaluator.text_text_similarity(row['std_prompt'], row['dial_prompt'])

            clip_i_score_between_std_and_dial = evaluator.image_image_similarity(std_noguard_image, dial_noguard_image)

            new_row = {
                "category": row['category'],
                "std_prompt": row['std_prompt'],
                "dial_prompt": row['dial_prompt'],

                "std_noguard_img": row['std_noguard_img'],
                "std_guarded_img": row['std_guarded_img'],
                "dial_noguard_img": row['dial_noguard_img'],
                "dial_guarded_img": row['dial_guarded_img'],

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

                "clip_i_score_between_std_and_dial": clip_i_score_between_std_and_dial
            }

            results.append(new_row)

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./exp_image_level_promptguard/image_level_{prompt_type}_results_{dialect}/image_level_{prompt_type}_analyze.csv', index=False)       
