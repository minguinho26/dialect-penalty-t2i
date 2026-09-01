import pandas as pd
import os, time
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.env_keys import get_openai_client  # keys are read from environment variables only

from dialect_prompts import DIALECT_SPECS, system_prompt, few_shot_prompt

client = get_openai_client()

def translate_to_dialect(dialect_key, text):
    if dialect_key not in DIALECT_SPECS:
        print(f"Error: {dialect_key} is not in the list.")
        return None

    sys_prompt = system_prompt(dialect_key)
    fewshot_shot = few_shot_prompt(dialect_key, text)

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-2026-03-05", 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": fewshot_shot}
            ],
            max_completion_tokens=500,
            # Lower temperature to enforce strict adherence to rules.
            temperature=0.1 
        )
        return response
    except Exception as e:
        print(f"API call error: {e}")
        return None

if __name__ == "__main__":
    input_file = "benign_expanded_prompts_gpt54.csv"
    output_file = "benign_translated_all_dialects.csv"

    print("1. Loading dataset...")
    
    # Load existing partial output to resume progress.
    if os.path.exists(output_file):
        print(f"📂 Existing file found! Resuming from [{output_file}].")
        df = pd.read_csv(output_file)
    else:
        try:
            df = pd.read_csv(input_file)
        except FileNotFoundError:
            print(f"Error: {input_file} not found. Check the path!")
            exit()

    target_dialects = ["ChcE", "CollSgE", "AAVE", "IndE", "JamE"]
    total_prompts = len(df)

    # Pre-allocate dialect columns to avoid insertion errors.
    for dialect in target_dialects:
        col_name = f"{dialect}_prompt"
        if col_name not in df.columns:
            df[col_name] = None 

    for dialect in target_dialects:
        col_name = f"{dialect}_prompt"
        print(f"\n==========================================")
        print(f"🚀 Starting [{dialect}] translation! (Filling only blanks)")
        print(f"==========================================")
        
        for idx, row in df.iterrows():
            std_prompt = row['standard_prompt']
            existing_val = df.at[idx, col_name]
            
            if pd.notna(existing_val) and str(existing_val).strip() != "":
                continue
                
            if (idx + 1) % 10 == 0:
                print(f"[{dialect}] Translating... {idx + 1} / {total_prompts}")
                
            response = translate_to_dialect(dialect, std_prompt)
            
            if response and response.choices[0].message.refusal is None:
                dialect_text = response.choices[0].message.content.strip()
                df.at[idx, col_name] = dialect_text
            else:
                print(f"[{idx}] Error/Refusal! Fallbacking to original text.")
                df.at[idx, col_name] = std_prompt
                
            # Save after every success to prevent data loss upon interruption.
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            
            # Sleep to avoid API rate limits.
            time.sleep(1) 
            
        print(f"✅ [{dialect}] Translation complete!")

    print(f"\n🎉 Success! All dialects safely saved to '{output_file}'.")