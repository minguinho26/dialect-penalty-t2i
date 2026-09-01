import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm

import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS, SEEDS
from common.env_keys import get_openai_client

client = get_openai_client()


os.makedirs('openai_moderation_results', exist_ok=True)
for DATA_TYPE in ['toxic', 'benign']:
    for dialect in DIALECTS:
        for seed in SEEDS:
            print(f"\n======================================")
            print(f"Processing: {dialect} / Seed: {seed}")
            print(f"======================================")
            
            input_file = f"text_level_typo_results/{DATA_TYPE}_results_{dialect}_seed_{seed}.csv"
            
            if not os.path.exists(input_file):
                print(f"File {input_file} not found. Skipping!")
                continue
                
            df = pd.read_csv(input_file)
            results = []
            
            for index, row in tqdm(df.iterrows(), total=len(df)):
                result = {
                    'category': row['category'],
                    'standard_prompt': row['standard_prompt'],
                    'typo_prompt': row['typo_prompt']
                }
                
                
                text = str(row['typo_prompt']).strip()
                    
                text = text[0].upper() + text[1:]
            
                try:
                    response = client.moderations.create(
                        model="omni-moderation-latest",
                        input=text,
                    )
                    
                    time.sleep(1) 
                    
                    result[f'typo_is_harmful'] = response.results[0].flagged
                    
                except Exception as e:
                    print(f"\n  Warning: API Error (typo_prompt at index {index}): {e}")
                    # Sets to None on error to prevent missing column creation
                    result[f'typo_is_harmful'] = None 
                
                results.append(result)
                
            moderate_df = pd.DataFrame(results)
            os.makedirs('./openai_moderation_typo_results', exist_ok=True)
            output_file = f"./openai_moderation_typo_results/{dialect}_{DATA_TYPE}_results_with_typo_seed_{seed}.csv"
            moderate_df.to_csv(output_file, index=False)
            
            print(f"Saved successfully: {output_file}")