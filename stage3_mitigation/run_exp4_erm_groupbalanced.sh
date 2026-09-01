#!/bin/bash
# =====================================================================
# run_exp4_erm_groupbalanced.sh
#
# Ablation requested by the corresponding author: disentangle the loss
# from the sampler. GroupDRO uses BOTH group-balanced sampling AND the
# robust (worst-group) loss, while the ERM baseline uses random sampling
# AND the mean loss, so the two effects are confounded. This script fills
# the missing 2x2 cell: ERM (mean loss) + group-balanced sampling.
#
#                       random sampling      group-balanced sampling
#   ERM (mean loss)     existing "ERM"       THIS SCRIPT  -> ermgb_*
#   GroupDRO loss       (not run)            existing "DRO"
#
#   sampling effect : ERM(random)  -> ERM(GB)    [loss held fixed = mean]
#   loss effect     : ERM(GB)      -> GroupDRO   [sampler held fixed = GB]
#
# No change to train_group_dro.py is needed: the loss (--robust) and the
# sampler (--no_group_balance) are independent flags. Omitting BOTH gives
# mean loss + group-balanced sampling.
#
# We REUSE the exact train subsamples used by the erm_*/dro_* runs
# (data/train_imb_<tag>_s<seed>.csv), so every ermgb_<tag>_s<seed> is
# paired per-seed with its erm_/dro_ counterparts.
# =====================================================================
set -e

mkdir -p ./results/sweep_ratio

# Ratios where the paper reports both ERM and G. DRO (Table 8).
FRACS=(0.975 0.98 0.985 0.99 0.995)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

for FRAC in "${FRACS[@]}"; do
    TAG=$(printf "%04d" "$(awk "BEGIN{printf \"%d\", $FRAC*1000}")")

    for SEED in "${SEEDS[@]}"; do
        DATA=./data/train_imb_${TAG}_s${SEED}.csv

        # Reuse the subsample already used by erm_/dro_ (paired per seed).
        # Regenerate only if it is somehow missing.
        if [ ! -f "$DATA" ]; then
            echo "[prepare] $DATA missing, regenerating (sae_frac=$FRAC seed=$SEED)"
            python prepare_imbalanced.py \
                --sae_frac "$FRAC" --seed "$SEED" \
                --output "$DATA"
        fi

        OUT=./results/sweep_ratio/ermgb_${TAG}_s${SEED}
        echo ""
        echo "=========================================="
        echo "  ERM + group-balanced sampling  |  SAE_frac=${FRAC}  |  SEED=${SEED}"
        echo "=========================================="

        if [ ! -f "$OUT/test_metrics.json" ]; then
            # ERM (no --robust)  +  group-balanced sampling (no --no_group_balance)
            python train_group_dro.py \
                --train_csv "$DATA" \
                --seed "$SEED" \
                --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else
            echo "  [skip] ERM+GB already done at $OUT"
        fi
    done
done

echo ""
echo "=========================================="
echo "  ERM+GB sweep done."
echo "  Aggregate with:"
echo "    python make_table8_markdown.py --ablation --seeds 0 1 2 3 4 5 6 7 8 9"
echo "=========================================="
