#!/usr/bin/env python3
"""check_env.py — Checks the status of torch family packages and major dependencies before execution.

Container images (RunPod, Colab, NGC) distribute torch, torchvision, and torchaudio built with the same ABI.
If only torch is replaced, the other two will remain referencing the old ABI and fail to load:

    RuntimeError: operator torchvision::nms does not exist
    OSError: libtorchaudio.so: undefined symbol: _ZN3c104cuda29...

Since transformers references both packages at import time, importing Trainer will also fail.
is_torchvision_available() / is_torchaudio_available() only checks for installation, not loadability,
so broken packages will pass this guard and raise an exception later.

    python common/check_env.py
"""
from __future__ import annotations

import importlib
import sys

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


def _probe(name: str):
    try:
        m = importlib.import_module(name)
        return OK, getattr(m, "__version__", "?")
    except ImportError:
        return None, "Not installed"
    except Exception as e:                      # e.g., failed to load .so
        return BAD, f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"


def main() -> int:
    print(f"python      {sys.version.split()[0]}")

    status, ver = _probe("torch")
    if status is None:
        print(f"[{BAD}] torch not installed. Use the image's torch or install manually.")
        return 1
    if status == BAD:
        print(f"[{BAD}] torch: {ver}")
        return 1
    import torch
    print(f"[{OK}] torch       {ver}")
    print(f"       CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    problems = []
    for comp in ("torchvision", "torchaudio"):
        st, info = _probe(comp)
        if st is None:
            print(f"[{OK}] {comp:11s} Not installed (No problem - transformers will skip it)")
        elif st == BAD:
            print(f"[{BAD}] {comp:11s} {info}")
            problems.append(comp)
        else:
            print(f"[{OK}] {comp:11s} {info}")

    for pkg in ("transformers", "accelerate", "numpy", "pandas", "sklearn"):
        st, info = _probe(pkg)
        label = {None: WARN, BAD: BAD}.get(st, OK)
        print(f"[{label}] {pkg:11s} {info}")
        if st == BAD:
            problems.append(pkg)

    # Even if individual packages are intact, importing Trainer might fail, so we check it separately.
    try:
        from transformers import Trainer, TrainingArguments  # noqa: F401
        print(f"[{OK}] transformers.Trainer import successful")
    except Exception as e:
        print(f"[{BAD}] transformers.Trainer import failed: "
              f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")
        problems.append("transformers.Trainer")

    if not problems:
        print("\nEnvironment normal. You can start training.")
        return 0

    print(f"\nBroken packages: {', '.join(dict.fromkeys(problems))}")
    print("\nIf you change torch to a different version than the image default, the co-built torchvision and torchaudio will break. Resolve it using one of the following:\n")
    print("  A. Remove unused companion packages (stage 3 needs neither, simplest)")
    print("       pip uninstall -y torchvision torchaudio\n")
    print("  B. Install all three at once with the same build")
    print(f"       pip install torch=={ver.split('+')[0]} torchvision torchaudio \\")
    print("           --index-url https://download.pytorch.org/whl/cu128")
    return 1


if __name__ == "__main__":
    sys.exit(main())
