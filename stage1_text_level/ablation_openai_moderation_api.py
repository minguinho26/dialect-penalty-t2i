import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.env_keys import get_openai_client  # keys are read from environment variables only

client = get_openai_client()
        
# benign
total_df = pd.read_csv('text_level_benign_results/benign_final_results_nsfw_t.csv')

for dialect in ['AAVE']: # 'ChcE', 'CollSgE', 'AAVE', 'IndE', 'JamE'
    results = []
    
    df = total_df[['category', 'standard_prompt', f'{dialect}_prompt']]
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        result = {}
        
        standard_text = row['standard_prompt'].strip()
        standard_text = str(standard_text)[0].upper() + str(standard_text)[1:]

        standard_response = client.moderations.create(
            model="omni-moderation-latest",
            input=standard_text,
            )
        
        dialect_text = row[f'{dialect}_prompt'].strip()
        dialect_text = str(dialect_text)[0].upper() + str(dialect_text)[1:]

        dialect_response = client.moderations.create(
            model="omni-moderation-latest",
            input=dialect_text,
            )
        
        time.sleep(1)

        result['category'] = row['category']
        result['standard_prompt'] = standard_text
        result['standard_is_harmful'] = standard_response.results[0].flagged
        result['dialect_prompt'] = dialect_text
        result['dialect_is_harmful'] = dialect_response.results[0].flagged
    
        results.append(result)

    results_df = pd.DataFrame(results)
    os.makedirs('openai_moderation_results', exist_ok=True)
    
    results_df.to_csv(f"openai_moderation_results/{dialect}_benign.csv", index=False)        