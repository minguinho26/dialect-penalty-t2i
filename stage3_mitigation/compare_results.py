import json, os
from pathlib import Path

EXPERIMENTS = {
    "Exp 1: NSFW-T (zero-shot)": "./results/exp1_nsfwt_zeroshot/test_metrics.json",
    "Exp 2: SAE-only ERM":       "./results/exp2_sae_only_erm/test_metrics.json",
    "Exp 3: ERM (imbalanced)":   "./results/exp3_erm/test_metrics.json",
    "Exp 3: UW (imbalanced)":    "./results/exp3_uw/test_metrics.json",
    "Exp 3: DRO (imbalanced)":   "./results/exp3_dro/test_metrics.json",
}

import sys, pathlib

# Make `common/` importable when run from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.common_utils import DIALECTS_WITH_SAE as DIALECTS
KEY_METRICS = (
    ["test_accuracy", "test_worst_group_acc",
     "test_TPR_spread", "test_FPR_spread", "test_gap_SAE_nonSAE"]
    + [f"test_TPR_{d}" for d in DIALECTS]
    + [f"test_FPR_{d}" for d in DIALECTS]
)

import matplotlib.pyplot as plt

def plot_sweep(sweep_dir="./results/sweep"):
    fracs, erm_wga, dro_wga = [], [], []
    for tag in ["0950", "0990", "0995"]:
        frac = float(tag) / (10 ** (len(tag)-1))   # "099" -> 0.99
        for method, store in [("erm", erm_wga), ("dro", dro_wga)]:
            path = f"{sweep_dir}/{method}_{tag}/test_metrics.json"
            if os.path.exists(path):
                with open(path) as f:
                    m = json.load(f)
                store.append((frac, m["test_worst_group_acc"]))
    
    plt.plot(*zip(*erm_wga), 'o-', label='ERM')
    plt.plot(*zip(*dro_wga), 's-', label='Group DRO')
    plt.xlabel("SAE fraction in training data")
    plt.ylabel("Worst-group accuracy on test set")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{sweep_dir}/imbalance_sweep.pdf")

# rows = []
# for name, path in EXPERIMENTS.items():
#     if not Path(path).exists():
#         print(f"[skip] {name}: {path} not found")
#         continue
#     with open(path) as f:
#         m = json.load(f)
#     rows.append({"experiment": name, **{k: m.get(k, None) for k in KEY_METRICS}})

# df = pd.DataFrame(rows).set_index("experiment").T

# # Pretty print
# pd.set_option("display.float_format", lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
# print(df.to_string())
# df.to_csv("./results/comparison_table.csv")
plot_sweep()