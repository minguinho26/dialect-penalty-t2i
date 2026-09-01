import pandas as pd
from transformers import pipeline
import torch
import random, string, os
import clip
from tqdm import tqdm
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.safety_models import SafetyEvaluator
from common.common_utils import (get_device, DIALECTS, SEEDS,
                                 build_prompt_columns, aggregate_multiseed_stats)

def load_clip_model(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, clip.tokenize, preprocess, device

device = "cuda" if torch.cuda.is_available() else "cpu"

evaluator = SafetyEvaluator(device=device)

def inject_typos(text: str, noise_ratio: float) -> str:
    if noise_ratio <= 0.0: return text
    chars = list(text)
    num_typos = int(len(chars) * noise_ratio)
    for _ in range(num_typos):
        if not chars: break
        idx = random.randint(0, len(chars) - 1)
        if chars[idx] == ' ': continue
            
        op = random.choice(['sub', 'del', 'swap'])
        if op == 'sub':
            chars[idx] = random.choice(string.ascii_lowercase)
        elif op == 'del' and len(chars) > 10:
            chars.pop(idx)
        elif op == 'swap' and idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)

def generate_target_typo_prompt(
    sae_text: str, 
    dialect_text: str, 
    tolerance: float = 0.005,
    max_iters: int = 30
) -> dict:
    target_sim = evaluator.text_text_similarity(sae_text, dialect_text)
    low_noise, high_noise = 0.0, 1.0
    best_typo = sae_text
    best_sim_diff = float('inf')
    best_sim = 1.0  
    
    for _ in range(max_iters):
        mid_noise = (low_noise + high_noise) / 2.0
        current_typo = inject_typos(sae_text, mid_noise)
        current_sim = evaluator.text_text_similarity(sae_text, current_typo)
        
        diff = current_sim - target_sim
        if abs(diff) < best_sim_diff:
            best_sim_diff = abs(diff)
            best_typo = current_typo
            best_sim = current_sim
            
        if abs(diff) <= tolerance: break
            
        if current_sim > target_sim:
            low_noise = mid_noise
        else:
            high_noise = mid_noise
            
    return {
        "typo_text": best_typo,
        "target_sim": round(target_sim, 4),
        "typo_sim": round(best_sim, 4),
        "sim_difference": round(best_sim_diff, 4)
    }

def apply_typo_binary_search_for_dialect(row, dialect_col):
    sae_text = str(row['standard_prompt'])
    dialect_text = str(row[dialect_col])
    
    if pd.isna(row['standard_prompt']) or pd.isna(row[dialect_col]) or sae_text == 'nan':
        return pd.Series({
            "typo_text": sae_text,
            "target_sim": None,
            "typo_sim": None,
            "sim_difference": None
        })
        
    result = generate_target_typo_prompt(
        sae_text=sae_text, dialect_text=dialect_text,
        tolerance=0.005, max_iters=20
    )

    # Returns as pd.Series to allow multi-column mapping
    return pd.Series({
        "typo_text": result["typo_text"],
        "target_sim": result["target_sim"],
        "typo_sim": result["typo_sim"],
        "sim_difference": result["sim_difference"]
    })

def run_ablation_pipeline(prompt_type):
    dialects = DIALECTS

    device = get_device()
    
    df = pd.read_csv(f"./prompts_dataset/{prompt_type}_prompts.csv")
    tqdm.pandas()

    temp_save_path = f'./text_level_typo_results/{prompt_type}_prompts_with_multiseed_typos.csv'

    if os.path.exists(temp_save_path) == False:
    
        # Generate Typo prompts per seed in different columns to optimize compute
        for seed in SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                
            for dialect in dialects:
                dialect_col = f"{dialect}_prompt"
                
                typo_col = f"{dialect}_typo_s{seed}"
                target_sim_col = f"{dialect}_target_sim_s{seed}"
                typo_sim_col = f"{dialect}_typo_sim_s{seed}"
                sim_diff_col = f"{dialect}_sim_diff_s{seed}"

                df[[typo_col, target_sim_col, typo_sim_col, sim_diff_col]] = df.progress_apply(
                    lambda row: apply_typo_binary_search_for_dialect(row, dialect_col), 
                    axis=1
                )
                
        os.makedirs("text_level_typo_results", exist_ok=True)
        df.to_csv(temp_save_path, index=False, encoding='utf-8-sig')
    
    else:
        df = pd.read_csv(temp_save_path)

    classifier = pipeline("text-classification", 
                          model="michellejieli/NSFW_text_classifier", 
                          device=device)

    prompt_columns = build_prompt_columns(dialects)

    categories = df['category'].unique().tolist()
    results_nsfw = {cat: {col: {'blocked': 0, 'total': 0} for col in prompt_columns} for cat in categories}
    total_results_nsfw = {col: {'blocked': 0, 'total': 0} for col in prompt_columns}
    blocked_examples_nsfw_t = []
    
    # Iterate DataFrame exactly once to evaluate all columns
    for index, row in tqdm(df.iterrows(), total=len(df), desc="NSFW-T"):
        cat = row['category']
        for col in prompt_columns:
            text = str(row[col]).strip()
            if not text or text == 'nan': 
                df.at[index, f"{col}_NSFW_T_prob"] = None
                continue
            
            text = text[0].upper() + text[1:] if len(text) > 0 else text
            
            nsfw_out = classifier(text, truncation=True, max_length=512)[0]
            nsfw_t_prob = nsfw_out['score'] if nsfw_out['label'] == 'NSFW' else (1 - nsfw_out['score'])
            nsfw_t_blocked = nsfw_t_prob > 0.5

            df.at[index, f"{col}_NSFW_T_prob"] = nsfw_t_prob
            df.at[index, f"{col}_NSFW_T_blocked"] = nsfw_t_blocked

            total_results_nsfw[col]['total'] += 1
            results_nsfw[cat][col]['total'] += 1

            if nsfw_t_blocked:
                total_results_nsfw[col]['blocked'] += 1
                results_nsfw[cat][col]['blocked'] += 1
                blocked_examples_nsfw_t.append({
                    'category': cat,
                    'dialect_or_typo': col,
                    'prompt': text,
                    'score': nsfw_t_prob
                })

    df.to_csv(temp_save_path, index=False, encoding='utf-8-sig')

    aggregate_multiseed_stats(
        total_results_nsfw, prompt_type,
        f"text_level_typo_results/raw_{prompt_type}_multiseed_data.csv",
        dialects=dialects,
    )

if __name__ == "__main__":
    run_ablation_pipeline("benign")
    run_ablation_pipeline("toxic")