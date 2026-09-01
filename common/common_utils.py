"""common_utils.py

torch is lazy-imported inside functions so that DIALECTS/SEEDS constants can be imported lightly even in environments without torch (e.g., OpenAI API-only scripts).
"""

DIALECTS = ["AAVE", "ChcE", "CollSgE", "IndE", "JamE"]

# Stage 3 treats Standard American English as a sixth group, since the mitigation
# models are trained and evaluated per group rather than against an SAE baseline.
DIALECTS_WITH_SAE = ["SAE"] + DIALECTS

SEEDS = [0, 1, 2, 3, 4]


def get_device():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Detected CUDA. Using: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Detected MPS. Mac GPU let's go~")
    else:
        device = torch.device("cpu")
        print("No GPU found. Sigh... using CPU.")
    return device


def mcnemar_exact(std_only: int, dial_only: int) -> float:
    """Two-sided exact binomial McNemar p over the discordant pairs.

    `std_only` and `dial_only` are the counts flagged by exactly one side. With no
    discordant pairs there is nothing to test, so the result is 1.0.
    """
    from scipy.stats import binomtest

    n = std_only + dial_only
    return 1.0 if n == 0 else binomtest(min(std_only, dial_only), n, 0.5).pvalue


def build_prompt_columns(dialects=None, seeds=None):
    dialects = DIALECTS if dialects is None else dialects
    seeds = SEEDS if seeds is None else seeds
    cols = ["standard_prompt"] + [f"{d}_prompt" for d in dialects]
    for seed in seeds:
        cols += [f"{d}_typo_s{seed}" for d in dialects]
    return cols


def aggregate_multiseed_stats(total_results, prompt_type, out_csv_path,
                              dialects=None, seeds=None):
    """
    Args:
        total_results: dict with format {col: {'blocked': int, 'total': int}}
        prompt_type: "benign" (FPR) or "toxic" (TPR)
        out_csv_path: path to save the aggregated CSV
    """
    import pandas as pd

    dialects = DIALECTS if dialects is None else dialects
    seeds = SEEDS if seeds is None else seeds
    metric = "FPR" if prompt_type == "benign" else "TPR"

    print("\n" + "=" * 60)
    if prompt_type == "benign":
        print("📈 [Final Stats] Multi-Seed based FPR and Bias Gap aggregation")
    else:
        print("📈 [Final Stats] Multi-Seed based TPR aggregation")
    print("=" * 60)

    def _rate(col):
        return (total_results[col]["blocked"] / max(1, total_results[col]["total"])) * 100

    std_rate = _rate("standard_prompt")
    all_seeds_stats = []
    for seed in seeds:
        for dialect in dialects:
            d_rate = _rate(f"{dialect}_prompt")
            t_rate = _rate(f"{dialect}_typo_s{seed}")
            all_seeds_stats.append({
                "Seed": seed,
                "Dialect": dialect,
                f"Std_{metric}(%)": std_rate,
                f"Dialect_{metric}(%)": d_rate,
                f"Typo_{metric}(%)": t_rate,
                "Bias_Gap(Dialect-Typo)": d_rate - t_rate,
            })

    stats_df = pd.DataFrame(all_seeds_stats)
    stats_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return stats_df


def apply_trusted_torch_load():
    """
    Prevents duplicate wrapping if called multiple times.
    """
    import torch

    if getattr(torch.load, "_is_trusted_patch", False):
        return

    _original_torch_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    _trusted_load._is_trusted_patch = True
    torch.load = _trusted_load
