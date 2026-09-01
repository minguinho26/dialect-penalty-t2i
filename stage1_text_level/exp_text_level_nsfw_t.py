import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import get_device, DIALECTS
from common.preflight import require

require("prompts")


def evaluate_toxicity_block_rate(prompt_type):
    print("1. Loading model... (michellejieli/NSFW_text_classifier)")

    device = get_device()

    classifier = pipeline("text-classification", 
                          model="michellejieli/NSFW_text_classifier", 
                          device=device )

    df = pd.read_csv(f"./prompts_dataset/{prompt_type}_prompts.csv")

    print(f"Total {len(df)} data loaded.\n")

    prompt_columns = ['standard_prompt'] + [f'{d}_prompt' for d in DIALECTS]

    final_results_nsfw_t = []
    
    print("2. Starting Text Filter (NSFW-T) inference and block rate calculation...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        total_row = {}
        total_row['category'] = row['category']
        for col in prompt_columns:
            text = str(row[col])
            
            text = text.strip()
            text = str(text)[0].upper() + str(text)[1:]
            
            nsfw_out = classifier(text, truncation=True, max_length=512)[0]
            nsfw_t_prob = nsfw_out['score'] if nsfw_out['label'] == 'NSFW' else (1 - nsfw_out['score'])
            
            total_row[f'{col}_NSFW_T_prob'.replace('_prompt', '')] = nsfw_t_prob
            total_row[col] = text
        
        final_results_nsfw_t.append(total_row)
                
    df_final_results_nsfw_t = pd.DataFrame(final_results_nsfw_t)
    os.makedirs(f"./text_level_{prompt_type}_results", exist_ok=True)
    df_final_results_nsfw_t.to_csv(f"./text_level_{prompt_type}_results/{prompt_type}_final_results_nsfw_t.csv", index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(df_final_results_nsfw_t)} instances to './text_level_{prompt_type}_results/{prompt_type}_final_results_nsfw_t.csv'.")

if __name__ == "__main__":
    evaluate_toxicity_block_rate("benign")
    evaluate_toxicity_block_rate("toxic")