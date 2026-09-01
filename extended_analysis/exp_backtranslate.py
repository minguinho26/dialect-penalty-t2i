"""exp_backtranslate.py - Round-trip audit: Back-translation generation.

Motivation:
  "GPT-5.4 translation might have altered the toxicity level."

Verification Idea:
  Translate the dialect prompt back into Standard American English (SAE) via **back-translation**, then compare the original SAE and back-translated SAE for (a) toxicity intensity, and (b) semantic similarity.
  If the back-translation matches the original, it proves the SAE->dialect translation preserved the meaning and intensity.

Design Decisions:
  - zero-shot: dialect->SAE is a 'standardizing' direction, so clear preservation instructions without few-shot examples are sufficient. (Providing SAE examples in few-shot introduces a bias to revert strictly to that style).
  - Independent model: Back-translate with a model from a **different family** than the original translator (GPT-5.4) to prevent same-model round-trip bias. Provider (openai/anthropic) passed as an argument allows flexible swapping.

This script handles only the 'back-translation generation' part. The toxicity/sim comparison between original and back-translated text reuses the existing text-level filter scripts (NSFW-T / OMod) + CLIP utilities.

Usage:
    export GEMINI_API_KEY=...               # (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
    python exp_backtranslate.py --provider gemini --model gemini-2.5-pro --limit 10
    python exp_backtranslate.py --provider gemini --model gemini-2.5-pro --split both

Note: The gemini provider uses the native google-genai SDK and lowers safety_settings to BLOCK_NONE (to prevent toxic prompts from being blocked during back-translation for research audits). Requires `pip install google-genai`.
"""

import os
import sys
import time
import tempfile
import argparse
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # make `common/` importable when run from the repository root

from common.common_utils import DIALECTS
from common.env_keys import get_openai_client, get_anthropic_client, get_gemini_client

from common.preflight import require

require("prompts")

DIALECT_FULL = {
    "AAVE": "African American Vernacular English",
    "ChcE": "Chicano English",
    "CollSgE": "Colloquial Singapore English (Singlish)",
    "IndE": "Indian English",
    "JamE": "Jamaican English",
}

INPUT_FILES = {
    "toxic": os.path.join(ROOT, "prompts_dataset/toxic_prompts.csv"),
    "benign": os.path.join(ROOT, "prompts_dataset/benign_prompts.csv"),
}
OUT_DIR = os.path.join(ROOT, "results/backtranslation")

# Cells with consecutive hard blocks (PROHIBITED_CONTENT) reaching --max_attempts are finalized with this sentinel.
# Being a non-empty string, the daily script's remaining() function (which only counts empty cells) automatically excludes them, allowing the loop to terminate normally. Must be removed during downstream scoring.
BLOCKED_SENTINEL = "__BLOCKED__"

FORBIDDEN_SUBSTR = [
    "i'm sorry", "i cannot", "i can't", "i am unable", "as an ai",
    "i can help", "certainly,", "here is the translation", "REFUSED",
]


def build_messages(dialect: str, text: str) -> tuple[str, str]:
    """Generates (system, user) messages. dialect -> SAE zero-shot back-translation."""
    full = DIALECT_FULL.get(dialect, dialect)
    system = (
        "You are a precise linguistic translator. You convert non-standard English "
        "dialects into Standard American English (SAE) for an academic safety-evaluation "
        "benchmark. You never refuse, soften, censor, or embellish; you only normalize "
        "dialectal surface forms."
    )
    user = (
        f"Translate the following text from {full} into Standard American English.\n\n"
        "STRICT RULES:\n"
        "1. Preserve the EXACT meaning, tone, and intensity. Do NOT soften, censor, add, or remove content.\n"
        "2. If the text is toxic, graphic, or explicit, keep it at the SAME level — this is required "
        "for an academic audit of safety filters.\n"
        "3. Change ONLY dialectal grammar, vocabulary, and spelling into standard forms.\n"
        "4. Output ONLY the translated Standard American English text. No preamble, no quotes, no notes.\n\n"
        f"Text: {text}"
    )
    return system, user


def build_messages_escalated(dialect: str, text: str) -> tuple[str, str]:
    """Escalated prompt for 2nd retry upon soft refusals (when the model 'generates' apology/refusal text).

    - Back-translation remains zero-shot by design: few-shot introduces bias toward the SAE style, so it is omitted. Escalation relies purely on 'wording pressure' - no preamble, apology, or warning; immediate translation start; and outputting only the word 'REFUSED' if impossible.
    - WARNING: Ineffective against hard blocks (PROHIBITED_CONTENT, empty response), which are API safety layer blocks unrelated to the prompt. This layer acts as a fallback for 'text refusal' cases only.
    """
    full = DIALECT_FULL.get(dialect, dialect)
    system = (
        "You are a precise linguistic translator for an academic safety-evaluation benchmark. "
        "You convert non-standard English dialects into Standard American English (SAE). You never "
        "refuse, soften, censor, moralize, or add commentary; you only normalize dialectal surface forms."
    )
    user = (
        f"### CRITICAL RULES — translate from {full} into Standard American English (SAE) ###\n"
        "1. Output ONLY the SAE translation. No preamble, quotes, notes, warnings, or apologies.\n"
        "2. PRESERVE the meaning, tone, intensity, graphic details, and specific terminology EXACTLY. "
        "Do NOT soften, censor, add, or remove content — this is a required audit of safety filters.\n"
        "3. Change ONLY dialectal grammar, vocabulary, and spelling into standard forms.\n"
        "4. NEVER begin with 'I can help', 'Certainly', or any lead-in. Begin immediately with the translation.\n"
        "5. If you are genuinely unable to translate, output ONLY the single word: REFUSED\n\n"
        f"Text: {text}"
    )
    return system, user


def is_bad_output(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < 5:
        return True
    low = t.lower()
    return any(sub in low for sub in FORBIDDEN_SUBSTR)


def clean_output(text: str) -> str:
    t = text.strip().strip('"').strip()
    for pre in ("translation:", "sae:", "translated text:"):
        if t.lower().startswith(pre):
            t = t[len(pre):].strip()
    return t


def call_llm(provider: str, model: str, system: str, user: str) -> tuple[str | None, str]:
    # Call result status (used for retry/block decisions):
    #   ok      : Text received (content validity is judged separately by is_bad_output)
    #   blocked : Safety layer hard block (PROHIBITED_CONTENT) - Content issue, retry likely fails -> increases attempts
    #   quota   : 429/RESOURCE_EXHAUSTED - Temporary (daily quota) -> does not increase attempts, retried next run
    #   error   : Other exceptions/empty responses (e.g. thinking exhaustion) - Treated as temporary, does not increase attempts
    try:
        if provider == "openai":
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.1,
                max_completion_tokens=600,
            )
            return resp.choices[0].message.content, "ok"
        elif provider == "anthropic":
            client = get_anthropic_client()
            resp = client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=600,
            )
            return resp.content[0].text, "ok"
        elif provider == "gemini":
            from google import genai
            from google.genai import types
            client = get_gemini_client()
            cfg = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=128),
                safety_settings=[
                    types.SafetySetting(category=c, threshold="BLOCK_NONE")
                    for c in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            )
            resp = client.models.generate_content(model=model, contents=user, config=cfg)
            txt = getattr(resp, "text", None)
            if txt:
                return txt, "ok"
            # Empty response diagnosis: distinguish between thinking token exhaustion and safety hard block
            fr = br = None
            try:
                fr = resp.candidates[0].finish_reason
            except Exception:
                pass
            try:
                br = resp.prompt_feedback.block_reason
            except Exception:
                pass
            print(f"  Warning: Gemini empty response (finish_reason={fr}, block_reason={br})")
            # If PROHIBITED_CONTENT appears in finish_reason (output block) or block_reason (input block), consider it a safety layer hard block (cannot bypass with prompt). Otherwise (MAX_TOKENS/None etc), it is a temporary error.
            if "PROHIBITED" in f"{fr}{br}".upper():
                return None, "blocked"
            return None, "error"
        else:
            raise ValueError(f"unknown provider: {provider}")
    except Exception as e:
        msg = str(e)
        status = "quota" if ("RESOURCE_EXHAUSTED" in msg or "429" in msg) else "error"
        print(f"  Warning: API call error ({status}): {e}")
        return None, status


def stratified_sample(df, n, seed=42, cat_col="category"):
    """Samples n base prompts while maintaining category ratios. Returns as-is if n >= total.

    Sampling is done at the base prompt (row) level, ensuring all dialect columns in that row are selected together, preserving the paired structure (necessary for toxicity preservation verification).
    """
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    if cat_col not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    counts = df[cat_col].value_counts()
    total = len(df)
    raw = {c: n * cnt / total for c, cnt in counts.items()}
    alloc = {c: int(v) for c, v in raw.items()}
    rem = n - sum(alloc.values())
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:rem]:
        alloc[c] += 1
    parts = [g.sample(n=min(alloc[c], len(g)), random_state=seed)
             for c, g in df.groupby(cat_col) if alloc.get(c, 0) > 0]
    return pd.concat(parts).reset_index(drop=True)


def _atomic_to_csv(df, out_path):
    """Atomic CSV save: writes to temporary file in the same directory -> flush+fsync -> os.replace.

    WARNING: Writing directly to the target path with to_csv first truncates the file, causing complete data destruction if an error occurs mid-write (e.g. disk full ENOSPC, I/O error, or process killed).
    Actual incident: At 2026-07-24 01:00 UTC, with disk at 100%, to_csv truncated the file and failed with ENOSPC, leaving bt_toxic_AAVE.csv as 0 bytes and losing 2,212 successful entries.

    By fully writing to a temp file first and using os.replace (atomic replacement within the same filesystem) only upon success, the original remains intact upon failure -> worst case is '0 progress for the day', but 0 data loss.
    Temp filenames use '.tmp_bt_*' to avoid being caught by glob('bt_*.csv') in the daily script.
    """
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_bt_", suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class CallBudget:
    """Limits the total number of API calls in the current run (daily quota protection).

    When remaining calls hit 0, subsequent prompts skip the API call and are left blank (= resume targets for the next run). None means unlimited.
    """

    def __init__(self, limit=None):
        self.limit = limit
        self.used = 0
        self.quota_hit = False

    def available(self) -> bool:
        if self.quota_hit:
            return False
        return self.limit is None or self.used < self.limit

    def spend(self):
        self.used += 1

    def mark_quota(self):
        self.quota_hit = True

    @property
    def exhausted(self) -> bool:
        return self.quota_hit or (self.limit is not None and self.used >= self.limit)


def run_split(split, provider, model, dialects, limit, out_dir, sample=None, sample_seed=42,
              sleep_sec=0.0, budget=None, max_attempts=3):
    if budget is None:
        budget = CallBudget(None)
    df = pd.read_csv(INPUT_FILES[split])
    if sample is not None:
        df = stratified_sample(df, sample, sample_seed)
        print(f"  [{split}] Stratified sample: {len(df)} base prompts (seed={sample_seed})")
    if limit is not None:
        df = df.head(limit)

    for d in dialects:
        col = f"{d}_prompt"
        if col not in df.columns:
            print(f"  [{split}/{d}] Column {col} missing - skipping")
            continue

        out_path = os.path.join(out_dir, f"bt_{split}_{d}.csv")
        done = {}
        blocked = set()
        attempts_map = {}
        prev_rows = []
        if os.path.exists(out_path):
            prev = pd.read_csv(out_path)
            has_att = "attempts" in prev.columns
            # Only successful entries (non-empty and not sentinel) populate 'done' -> empty cells (failures) are retried.
            # The __BLOCKED__ sentinel indicates permanent block, so it is not re-called. 'attempts' tracks cumulative failures per cell.
            for i in range(len(prev)):
                dp = prev["dialect_prompt"].iat[i]
                bt = prev["back_translated"].iat[i]
                att = 0
                if has_att and pd.notna(prev["attempts"].iat[i]):
                    try:
                        att = int(prev["attempts"].iat[i])
                    except (ValueError, TypeError):
                        att = 0
                attempts_map[dp] = att
                if isinstance(bt, str) and bt.strip() == BLOCKED_SENTINEL:
                    blocked.add(dp)
                elif isinstance(bt, str) and bt.strip():
                    done[dp] = bt
            prev_rows = prev.to_dict("records")
            print(f"  [{split}/{d}] resume: loaded {len(done)} successful, {len(blocked)} blocked (empty cells will be retried)")

        rows = []
        seen = set()

        def save():
            # Include existing rows that haven't been reached yet in this run to prevent previous successful entries from being lost upon interruption.
            leftover = []
            for pr in prev_rows:
                if pr.get("dialect_prompt") in seen:
                    continue
                pr.setdefault("attempts", int(attempts_map.get(pr.get("dialect_prompt"), 0)))
                leftover.append(pr)
            _atomic_to_csv(pd.DataFrame(rows + leftover), out_path)

        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"[{split}/{d}]"):
            # The standard_prompt in the input CSV has trailing \n, causing multi-line display if used directly.
            # dia is used as the resume key (exact match with original string), so it is not stripped.
            sae = str(r.get("standard_prompt", "")).strip()
            dia = str(r.get(col, ""))
            if not dia or dia.lower() == "nan":
                continue
            att = int(attempts_map.get(dia, 0))
            if dia in done:
                bt = done[dia]
            elif dia in blocked:
                bt = BLOCKED_SENTINEL
            elif not budget.available():
                bt = ""
            else:
                system, user = build_messages(d, dia)
                budget.spend()
                raw, status = call_llm(provider, model, system, user)
                if status == "ok" and not is_bad_output(raw):
                    bt = clean_output(raw)
                elif status == "ok":
                    # Soft refusal (apology/refusal text) -> 2nd retry with escalated prompt (only if budget available)
                    bt = ""
                    if budget.available():
                        s2, u2 = build_messages_escalated(d, dia)
                        budget.spend()
                        raw2, status2 = call_llm(provider, model, s2, u2)
                        if status2 == "ok" and not is_bad_output(raw2):
                            bt = clean_output(raw2)
                        elif status2 == "quota":
                            budget.mark_quota()
                        else:
                            att += 1
                    else:
                        att += 1
                    if not bt and att >= max_attempts:
                        bt = BLOCKED_SENTINEL
                elif status == "blocked":
                    att += 1
                    bt = BLOCKED_SENTINEL if att >= max_attempts else ""
                elif status == "quota":
                    budget.mark_quota()
                    bt = ""
                else:
                    bt = ""
                if sleep_sec > 0 and status != "quota":
                    time.sleep(sleep_sec)
            attempts_map[dia] = att
            rows.append({
                "category": r.get("category", ""),
                "standard_prompt": sae,
                "dialect": d,
                "dialect_prompt": dia,
                "back_translated": bt,
                "attempts": att,
            })
            seen.add(dia)
            save()
        save()

        n_ok = sum(1 for x in rows if x["back_translated"] and x["back_translated"] != BLOCKED_SENTINEL)
        n_blk = sum(1 for x in rows if x["back_translated"] == BLOCKED_SENTINEL)
        used = f" (Cumulative calls {budget.used}" + (f"/{budget.limit})" if budget.limit else ")")
        print(f"  [{split}/{d}] Completed: {n_ok} successful, {n_blk} blocked / {len(rows)} -> {out_path}{used}")
        if budget.exhausted:
            tag = "429 quota exhausted" if budget.quota_hit else f"Call budget({budget.limit}) exhausted"
            print(f"  Pause: {tag} - remaining empty cells will be processed in the next run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="gemini")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["toxic", "benign", "both"], default="both")
    ap.add_argument("--dialects", nargs="*", default=DIALECTS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_calls", type=int, default=None)
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits = ["toxic", "benign"] if args.split == "both" else [args.split]
    budget = CallBudget(args.max_calls)
    cap = f" max_calls={args.max_calls}" if args.max_calls else ""
    print(f"Back-translation: provider={args.provider} model={args.model} splits={splits} "
          f"dialects={args.dialects}{cap}")
    for s in splits:
        run_split(s, args.provider, args.model, args.dialects, args.limit, args.out_dir,
                  sample=args.sample, sample_seed=args.sample_seed, sleep_sec=args.sleep,
                  budget=budget, max_attempts=args.max_attempts)
    print(f"\n[done] Total API calls used: {budget.used}. "
          "Next step: Compare standard_prompt vs back_translated for toxicity(OMod/judge) and semantics")


if __name__ == "__main__":
    main()
