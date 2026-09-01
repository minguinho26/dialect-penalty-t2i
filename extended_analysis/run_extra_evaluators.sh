#!/usr/bin/env bash
# run_extra_evaluators.sh
#
# Scores the unguarded images with ShieldGemma 2, a CLIP-independent second opinion on top
# of the paper's own NSFW-I and multi-head evaluators (both of which share one CLIP encoder).
#
# Generated images are not released, so point --zip-dir at your own regenerated set: one zip
# per split/dialect, named <split>_<dialect>_noguard.zip. score_unguarded_vlm.py reads the
# PNGs straight out of each zip, so they never need extracting.
#
#   bash extended_analysis/run_extra_evaluators.sh shieldgemma
#   bash extended_analysis/run_extra_evaluators.sh shieldgemma toxic --zip-dir /path/to/zips
#
# Scoring resumes from the per-dialect CSV, so re-running skips finished pairs.

set -u
cd "$(dirname "$0")/.."

usage() {
  echo "Usage: run_extra_evaluators.sh shieldgemma [toxic|benign] [--zip-dir DIR]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
EVAL="$1"
case "$EVAL" in
  shieldgemma) ;;
  *) echo "Unknown evaluator: $EVAL" >&2; usage ;;
esac
shift

ONLY_SPLIT=""
ZIP_DIR="results/$EVAL/zips"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip-dir) ZIP_DIR="${2:?--zip-dir needs a path}"; shift 2 ;;
    toxic|benign) ONLY_SPLIT="$1"; shift ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

PY="python"
DIALECTS=(AAVE ChcE CollSgE IndE JamE)

echo "Evaluator: $EVAL ${ONLY_SPLIT:+(split=$ONLY_SPLIT)}  zips: $ZIP_DIR"
MISSING=()
FAILED=()

for split in toxic benign; do
  [[ -n "$ONLY_SPLIT" && "$split" != "$ONLY_SPLIT" ]] && continue
  for dialect in "${DIALECTS[@]}"; do
    zip_path="$ZIP_DIR/${split}_${dialect}_noguard.zip"
    echo ""; echo "---- [$EVAL/${split}_${dialect}] ----"

    if [[ ! -f "$zip_path" ]]; then
      echo "  No zip at $zip_path - skipping"
      MISSING+=("${split}_${dialect}")
      continue
    fi

    if "$PY" extended_analysis/score_unguarded_vlm.py \
        --evaluator "$EVAL" --split "$split" --dialect "$dialect" --zip "$zip_path"; then
      echo "  Done"
    else
      echo "  Scoring failed - rerun to resume from the partial CSV"
      FAILED+=("${split}_${dialect}")
    fi
  done
done

echo ""; echo "Done"
[[ ${#MISSING[@]} -gt 0 ]] && printf 'Missing zips (%d):\n' "${#MISSING[@]}" && printf '  - %s\n' "${MISSING[@]}"
[[ ${#FAILED[@]} -gt 0 ]] && printf 'Failures (%d):\n' "${#FAILED[@]}" && printf '  - %s\n' "${FAILED[@]}"
[[ ${#MISSING[@]} -eq 0 && ${#FAILED[@]} -eq 0 ]] && echo "All complete."
exit 0
