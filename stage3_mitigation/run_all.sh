#!/bin/bash
# =====================================================================
# Stage 3
#
# Requires GPU for DistilBERT fine-tuning.
# data/ is gitignored, so it gets regenerated in step [1].
# =====================================================================
set -e
cd "$(dirname "$0")"

echo "==== [1/6] prepare base data ===="
python prepare_data.py

echo "==== [2/6] prepare SAE-only data ===="
python prepare_sae_only.py

echo "==== [3/6] EXP1: NSFW-T zero-shot baseline ===="
python eval_only_exp1.py --output_dir ./results/exp1_nsfwt_zeroshot

echo "==== [4/6] EXP2 (SAE-only ERM) + EXP3 balanced  [seed 0-9] ===="
bash run_exp2_exp3_balanced.sh

echo "==== [5/6] EXP3 ratio sweep: ERM / DRO  [frac 0.95-0.995 x seed 0-9] ===="
bash run_exp3_various_ratio_sweep.sh

echo "==== [6/6] gather results ===="
python gather_all_results.py

echo ""
echo "==== DONE ===="
