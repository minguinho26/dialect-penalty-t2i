import os
import pandas as pd
from transformers import AutoTokenizer

# Resolve path relative to this file's directory instead of CWD to avoid path issues when running from different locations.
_HERE = os.path.dirname(os.path.abspath(__file__))
def _p(rel):
    return os.path.join(_HERE, rel)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

train = pd.read_csv(_p("data/train.csv"))
test  = pd.read_csv(_p("data/val.csv"))

def length_stats(df, name):
    lens = [len(tokenizer.encode(t, truncation=False)) for t in df["text"].astype(str)]
    s = pd.Series(lens)
    print(f"\n=== {name} (n={len(df):,}) ===")
    print(f"  mean   : {s.mean():.1f}")
    print(f"  median : {s.median():.1f}")
    print(f"  95%ile : {s.quantile(0.95):.1f}")
    print(f"  99%ile : {s.quantile(0.99):.1f}")
    print(f"  max    : {s.max()}")
    print(f"  > 128  : {(s > 128).sum():,} ({(s > 128).mean()*100:.1f}%)")
    print(f"  > 256  : {(s > 256).sum():,} ({(s > 256).mean()*100:.1f}%)")
    print(f"  > 512  : {(s > 512).sum():,} ({(s > 512).mean()*100:.1f}%)")

length_stats(train, "train")
length_stats(test,  "test")