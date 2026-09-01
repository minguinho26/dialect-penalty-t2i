"""
Evaluator Weights Preparation
====================
NSFW-I: LAION CLIP-based NSFW Detector (autokeras, ViT-L/14, 768-d)
Q16:    Schramowski et al. (manual setup)
"""

import pickle
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

import urllib.request

WEIGHTS_DIR = Path("./evaluator_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


def prepare_nsfw_model():
    model_dir = WEIGHTS_DIR / "clip_autokeras_binary_nsfw"

    if model_dir.exists() and (model_dir / "saved_model.pb").exists():
        print(f"NSFW-I model already exists: {model_dir}")
        return True

    print("Downloading LAION NSFW detector (ViT-L/14)...")
    url = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"

    zip_path = WEIGHTS_DIR / "clip_autokeras_binary_nsfw.zip"

    try:
        urlretrieve(url, str(zip_path))
        print(f"  Download complete: {zip_path}")
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(WEIGHTS_DIR))
        print(f"  Extraction complete: {model_dir}")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False

    zip_path.unlink(missing_ok=True)

    try:
        import autokeras as ak
        from tensorflow.keras.models import load_model

        print("  Verifying model...")
        model = load_model(str(model_dir), custom_objects=ak.CUSTOM_OBJECTS, compile=False)
        dummy = np.random.rand(10, 768).astype("float32")
        preds = model.predict(dummy, batch_size=10)
        print(f"  Verification passed")
        return True
    except ImportError:
        print("  autokeras/tensorflow not installed - skipping verification")
        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def prepare_q16_head():
    probe_path = Path("./prompts.p")
    out_path = WEIGHTS_DIR / "q16_prompts.pt"

    if out_path.exists():
        print(f"Q16 head already exists: {out_path}")
        return True

    # Download directly from GitHub if the original pickle file is missing
    if not probe_path.exists():
        print("prompts.p file not found locally. Initiating download from Q16 official repo...")
        url = "https://raw.githubusercontent.com/ml-research/Q16/main/data/ViT-L-14/prompts.p"
        try:
            urllib.request.urlretrieve(url, probe_path)
            print("prompts.p download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("  -> Analysis possible with NSFW-I + CLIP-sim without Q16")
            return False

    # Convert the numpy pickle file to PyTorch state_dict
    if probe_path.exists():
        import torch
        import torch.nn as nn
        
        try:
            prompts = torch.HalfTensor(pickle.load(open(probe_path, 'rb')))
            torch.save(prompts, out_path)
            
            print(f"Q16 head conversion complete: {out_path}")

            os.remove(probe_path)

            return True
            
        except Exception as e:
            print(f"Error during weight conversion: {e}")
            return False

    return False


def verify():
    print(f"\n{'='*60}")
    print("Weight Status")
    print(f"{'='*60}")
    nsfw_dir = WEIGHTS_DIR / "clip_autokeras_binary_nsfw"
    if nsfw_dir.exists() and (nsfw_dir / "saved_model.pb").exists():
        print(f"  NSFW-I: {nsfw_dir}")
    else:
        print(f"  NSFW-I: Not ready")

    q16_path = WEIGHTS_DIR / "q16_prompts.pt"
    if q16_path.exists():
        print(f"  Q16: {q16_path}")
    else:
        print(f"  Q16: Not ready (optional)")


if __name__ == "__main__":
    prepare_nsfw_model()
    prepare_q16_head()
    verify()