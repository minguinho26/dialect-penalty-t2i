"""env_keys.py - Central loader for reading API keys/tokens exclusively from environment variables.

Keys are read from environment variables. The repository root's .env file (gitignored) is also reflected in the same path.

Recognized variables:
    OPENAI_API_KEY      OpenAI (moderation API, prompt generation/dialect translation)
    ANTHROPIC_API_KEY   Anthropic (LLM judge)
    GEMINI_API_KEY      Google Gemini (back-translation)
    HF_TOKEN            HuggingFace Hub (gated model download)
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "load_env_file",
    "get_key",
    "require_key",
    "mask",
    "get_openai_client",
    "get_anthropic_client",
    "get_gemini_client",
    "hf_login",
    "report",
    "KNOWN_KEYS",
]

KNOWN_KEYS = {
    "OPENAI_API_KEY": ("OpenAI moderation / prompt generation", "https://platform.openai.com/api-keys"),
    "ANTHROPIC_API_KEY": ("Anthropic LLM judge", "https://console.anthropic.com/settings/keys"),
    "GEMINI_API_KEY": ("Gemini back-translation", "https://aistudio.google.com/apikey"),
    "HF_TOKEN": ("HuggingFace Hub (gated models)", "https://huggingface.co/settings/tokens"),
}

_ENV_FILE_LOADED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE format. Custom implementation to work without python-dotenv."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Remove inline comments only if the value is not wrapped in quotes
        if val and val[0] in ("'", '"'):
            quote = val[0]
            end = val.find(quote, 1)
            val = val[1:end] if end > 0 else val[1:]
        elif " #" in val:
            val = val.split(" #", 1)[0].strip()
        if key:
            out[key] = val
    return out


def load_env_file(path: str | os.PathLike | None = None, override: bool = False) -> int:
    global _ENV_FILE_LOADED
    env_path = Path(path) if path is not None else _repo_root() / ".env"
    if not env_path.is_file():
        _ENV_FILE_LOADED = True
        return 0
    applied = 0
    for key, val in _parse_env_file(env_path).items():
        if override or not os.environ.get(key):
            os.environ[key] = val
            applied += 1
    _ENV_FILE_LOADED = True
    return applied


def mask(value: str | None) -> str:
    """Mask key so it is safe to log."""
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}...{value[-2:]} (len={len(value)})"


def get_key(name: str, default: str | None = None) -> str | None:
    if not _ENV_FILE_LOADED:
        load_env_file()
    val = os.environ.get(name)
    if val is not None:
        val = val.strip()
    return val or default


def require_key(name: str) -> str:
    """Read the key, but throw an exception with instructions if it does not exist.
    os.environ.get() returns None when the key does not exist, which causes errors later at the time of SDK call. We stop it early here.
    """
    val = get_key(name)
    if val:
        return val
    purpose, issuer = KNOWN_KEYS.get(name, ("", ""))
    lines = [f"Environment variable {name} is not set."]
    if purpose:
        lines.append(f"  Purpose: {purpose}")
    if issuer:
        lines.append(f"  Issuer: {issuer}")
    lines += [
        "",
        "Please set it using one of the following methods:",
        f"  export {name}=...                     # Apply to the current shell only",
        f"  echo '{name}=...' >> {_repo_root() / '.env'}   # Use .env (gitignored)",
    ]
    raise RuntimeError("\n".join(lines))


def get_openai_client(**kwargs):
    """Authenticated OpenAI client. Import is deferred to call time as an optional dependency."""
    from openai import OpenAI
    return OpenAI(api_key=require_key("OPENAI_API_KEY"), **kwargs)


def get_anthropic_client(**kwargs):
    import anthropic
    return anthropic.Anthropic(api_key=require_key("ANTHROPIC_API_KEY"), **kwargs)


def get_gemini_client(**kwargs):
    from google import genai
    return genai.Client(api_key=require_key("GEMINI_API_KEY"), **kwargs)


def hf_login(required: bool = False) -> bool:
    """Log in to huggingface_hub if HF_TOKEN exists.
    Tokens are not needed for public models, so required=False is the default.
    Calling login(None) without a token falls into an interactive prompt, so we strictly block it here.
    """
    token = require_key("HF_TOKEN") if required else get_key("HF_TOKEN")
    if not token:
        print("[env_keys] HF_TOKEN not set -> Proceeding with anonymous access "
              "(Set HF_TOKEN if gated models are needed).")
        return False
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print(f"[env_keys] HuggingFace login successful: {mask(token)}")
    return True


def report() -> None:
    n = load_env_file()
    root = _repo_root()
    print(f"repo root : {root}")
    print(f".env      : {'Loaded (%d applied)' % n if n else 'None or no items to apply'}")
    print()
    width = max(len(k) for k in KNOWN_KEYS)
    for name, (purpose, _) in KNOWN_KEYS.items():
        val = get_key(name)
        status = "OK " if val else "MISSING"
        print(f"  [{status}] {name:<{width}}  {mask(val):<24}  {purpose}")
    print()
    missing = [k for k in KNOWN_KEYS if not get_key(k)]
    if missing:
        print("Unset keys:", ", ".join(missing))
        print("Only scripts using the specific key will be affected. Not all of them are required.")
    else:
        print("All keys are set.")


if __name__ == "__main__":
    report()
