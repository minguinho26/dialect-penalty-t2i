"""
preflight.py — Checks if the data and weights required by the script are present at the entry point.
If we proceed without them, FileNotFoundError will be raised inside torch.load or pd.read_csv,
but the location alone won't tell us what asset to get and from where. This script stops execution early
and prints the paths and instructions on how to acquire them.

Usage:
    from common.preflight import require

    require("latent_guard")
    require("prompts", "multihead")
"""
from __future__ import annotations

import pathlib
import sys

__all__ = ["require", "check", "ASSETS"]

ASSETS: dict[str, tuple[list[str], str, str]] = {
    "prompts": (
        ["prompts_dataset/toxic_prompts.csv", "prompts_dataset/benign_prompts.csv"],
        "Paired dialect prompt dataset",
        "python common/fetch_data.py (HuggingFace, gated - requires `hf auth login`)",
    ),
    "evaluator_weights": (
        ["evaluator_weights/q16_prompts.pt",
         "evaluator_weights/clip_autokeras_binary_nsfw"],
        "Q16 and LAION CLIP NSFW image evaluator (Third-party)",
        "python common/prepare_evaluator_weights.py",
    ),
    "latent_guard": (
        ["latent_guard_checkpoint/model_parameters.pth",
         "latent_guard_checkpoint/CoPro_v1.0.json"],
        "Latent Guard checkpoint and CoPro concept cache (Third-party, cannot be redistributed)",
        "Download from the original repository and place in ./latent_guard_checkpoint/: "
        "https://github.com/rt219/LatentGuard",
    ),
    "multihead": (
        ["multihead_checkpoints"],
        "Multi-head image safety classifier heads (Unsafe Diffusion, Qu et al. — "
        "Third-party, cannot be redistributed)",
        "Download the five .pt files and place them in ./multihead_checkpoints/: "
        "https://github.com/YitingQu/unsafe-diffusion/tree/main/checkpoints/multi-headed",
    ),
}


def check(name: str, root: str | pathlib.Path = ".") -> list[str]:
    if name not in ASSETS:
        raise KeyError(f"Unknown asset: {name} (Available: {', '.join(ASSETS)})")
    paths, _, _ = ASSETS[name]
    root = pathlib.Path(root)
    return [p for p in paths if not (root / p).exists()]


def require(*names: str, root: str | pathlib.Path = ".") -> None:
    problems = [(n, miss) for n in names if (miss := check(n, root))]
    if not problems:
        return
    print("\nRequired assets are missing.\n", file=sys.stderr)
    for name, missing in problems:
        _, desc, how = ASSETS[name]
        print(f"  [{name}] {desc}", file=sys.stderr)
        for m in missing:
            print(f"      Missing: {m}", file=sys.stderr)
        print(f"      Solution: {how}\n", file=sys.stderr)
    print("The script must be executed from the repository root (paths are relative to the root).",
          file=sys.stderr)
    raise SystemExit(1)
