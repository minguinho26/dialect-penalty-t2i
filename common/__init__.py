"""Shared utilities for the dialect-penalty-t2i experiments.

Scripts are run from the repository root (`python stage1_text_level/foo.py`), which puts
the *script's* directory on sys.path rather than the repository root. Every script that
needs this package therefore starts with the same three lines:

    # Make `common/` importable when run from the repository root.
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

and then imports normally: `from common.common_utils import DIALECTS`.
"""
