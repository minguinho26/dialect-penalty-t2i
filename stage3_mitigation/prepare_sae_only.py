import os
"""
Experiment 2: SAE-only training data (mimics michellejieli's SAE-only Reddit training).
"""
import pandas as pd

# Resolve path relative to this file's directory instead of CWD to avoid path issues when running from different locations.
_HERE = os.path.dirname(os.path.abspath(__file__))
def _p(rel):
    return os.path.join(_HERE, rel)

train = pd.read_csv(_p("data/train.csv"))
sae_train = train[train["dialect"] == "SAE"].reset_index(drop=True)
sae_train.to_csv(_p("data/train_sae_only.csv"), index=False)

print(f"SAE-only train samples: {len(sae_train):,}")
print(sae_train["group"].value_counts().sort_index())