#!/bin/bash
set -e

mkdir -p ./results/seeds_exp2
mkdir -p ./results/seeds_exp3

FRAC=0.99
TAG=0990

# EXP2 uses deterministic SAE-only data (already produced by prepare_sae_only.py)
EXP2_DATA=./data/train_sae_only.csv

for SEED in 0 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "=========================================="
    echo "  SEED=${SEED}"
    echo "=========================================="

    # ----------------------------------------
    # EXP2: SAE-only ERM (NSFW-T setup mimic)
    # ----------------------------------------
    echo ""
    echo "--- EXP2: SAE-only ERM ---"
    OUT=./results/seeds_exp2/sae_only_erm_s${SEED}
    python train_group_dro.py \
        --train_csv $EXP2_DATA \
        --no_group_balance \
        --metric_for_best_model accuracy \
        --seed $SEED \
        --output_dir $OUT
    find $OUT -name "checkpoint-*" -type d -exec rm -rf {} +

    # ----------------------------------------
    # EXP3 BALANCED: budget-matched balanced  (NEW)
    # ----------------------------------------
    echo ""
    echo "--- EXP3 prep: balanced data (budget-matched) ---"
    BAL_DATA=./data/train_bal_s${SEED}.csv
    python prepare_balanced.py \
        --seed $SEED \
        --output $BAL_DATA

    echo ""
    echo "--- EXP3 BALANCED: ERM ---"
    OUT=./results/seeds_exp3/erm_balanced_s${SEED}
    python train_group_dro.py \
        --train_csv $BAL_DATA \
        --no_group_balance \
        --seed $SEED \
        --output_dir $OUT
    find $OUT -name "checkpoint-*" -type d -exec rm -rf {} +
done