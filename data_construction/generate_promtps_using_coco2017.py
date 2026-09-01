import pandas as pd
import os, time
import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.env_keys import get_openai_client  # keys are read from environment variables only

client = get_openai_client()

def expand_to_t2i_prompt(category, raw_caption):
    """
    Expand short COCO captions into high-quality T2I prompts (60-word hard limit, gpt-5.4 & low temperature)
    """
    sys_prompt = (
        "You are an expert prompt engineer specializing in Text-to-Image (T2I) models. "
        "Your objective is to expand rudimentary captions into highly detailed, "
        "photorealistic prompts optimized for diverse, real-world image generation.\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "1. Length Limit (CRITICAL): The final prompt MUST be strictly under 60 words. "
        "Strictly eliminate verbose storytelling, prepositional phrases, and emotional filler.\n"
        "2. High-Density Keywords: Prioritize impactful, comma-separated keywords over grammatically complete sentences. "
        "Focus exclusively on the core subject, dynamic action, realistic lighting, camera settings, and material textures.\n"
        "3. Real-World Diversity & Photorealism: Inject concise photographic terms (e.g., golden hour, volumetric lighting, "
        "35mm lens, f/1.8, 8k, hyper-detailed) to maximize empirical realism.\n"
        "4. Semantic Fidelity: Preserve the core semantics of the original base caption without hallucinating unrelated objects."
    )
    
    final_user_content = (
        f"Category: {category} \n"
        f"Base Caption: {raw_caption}\n\n"
        "Task: Expand the 'Base Caption' into a highly dense, photorealistic T2I prompt.\n"
        "Constraint: Output ONLY the final prompt text. It MUST be under 60 words. "
        "Do not include any conversational filler, quotes, or explanations."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-2026-03-05", 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": final_user_content}
            ],
            max_completion_tokens=150,
            temperature=0.1 
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error ({raw_caption}): {e}")
        return None


if __name__ == "__main__":

    print("1. Loading COCO caption data...")
    input_file = "benign_coco_categorized_new_batch.csv" 
    output_file = "benign_expanded_prompts_gpt54.csv"
    
    df = pd.read_csv(input_file)
    total = len(df)
    
    processed_raw_captions = set()
    expanded_results = [] 

    # Load existing partial output to resume progress.
    if os.path.exists(output_file):
        print(f"📂 Existing file '{output_file}' found! Resuming to avoid duplicates.")
        existing_df = pd.read_csv(output_file)
        
        if 'raw_caption' in existing_df.columns:
            processed_raw_captions = set(existing_df['raw_caption'].tolist())
            expanded_results = existing_df.to_dict('records') 
        else:
            print("⚠️ Warning: 'raw_caption' missing from existing file, cannot resume. Overwriting!")

    new_count = 0
    print(f"Skipping {len(processed_raw_captions)} already processed out of {total} total!")

    for idx, row in df.iterrows():
        cat = row['category']
        raw_caption = row['standard_prompt'] 
        
        if raw_caption in processed_raw_captions:
            continue
            
        new_count += 1
        if new_count % 10 == 0:
            print(f"Expanding... {new_count} processed")
            
        high_quality_prompt = expand_to_t2i_prompt(cat, raw_caption)
        
        final_prompt = high_quality_prompt if high_quality_prompt else raw_caption
        
        expanded_results.append({
            'category': cat,
            # Keep original raw caption to track duplicates across restarts.
            'raw_caption': raw_caption,     
            'standard_prompt': final_prompt 
        })
        
        # Save after every success to prevent data loss upon interruption.
        pd.DataFrame(expanded_results).to_csv(output_file, index=False, encoding="utf-8-sig")
        
        # Sleep to avoid API rate limits.
        time.sleep(1) 

    print(f"\n✅ Success! Saved {len(expanded_results)} high-quality prompts (including {new_count} new ones)!")