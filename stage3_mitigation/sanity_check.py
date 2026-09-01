# Verify data leakage because the balanced dataset yields unusually perfect results.
import pandas as pd

print("=" * 60)
print("Check 1: train.csv (long) vs test (val.csv) leakage")
print("=" * 60)
train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/val.csv")

tr_ids = set(train["sample_id"].unique())
te_ids = set(test["sample_id"].unique())
print(f"  Train sample_ids: {len(tr_ids):,}")
print(f"  Test  sample_ids: {len(te_ids):,}")
print(f"  Overlap         : {len(tr_ids & te_ids)}   (MUST be 0)")

tr_txt = set(train["text"].astype(str).str.strip().str.lower())
te_txt = set(test ["text"].astype(str).str.strip().str.lower())
print(f"  Exact text overlap: {len(tr_txt & te_txt)}   (should be ~0)")

print("\n" + "=" * 60)
print("Check 2: balanced training sets vs test")
print("=" * 60)
for seed in [0, 1, 2]:
    bal = pd.read_csv(f"./data/train_bal_s{seed}.csv")
    bal_ids = set(bal["sample_id"].unique())
    bal_txt = set(bal["text"].astype(str).str.strip().str.lower())
    print(f"  seed={seed}:  bal_size={len(bal):>5}, "
          f"sample_id overlap={len(bal_ids & te_ids)}, "
          f"text overlap={len(bal_txt & te_txt)}")

print("\n" + "=" * 60)
print("Check 3: test set composition")
print("=" * 60)
print(f"  Total test samples: {len(test):,}")
print(f"  Per dialect:")
print(test["dialect"].value_counts().to_string())
print(f"  Per group:")
print(test["group"].value_counts().sort_index().to_string())