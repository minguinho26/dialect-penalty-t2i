#!/bin/bash
set -e

mkdir -p ./results/sweep_ratio

# Sweep grid: SAE_frac values
FRACS=(0.95 0.955 0.96 0.965 0.97 0.975 0.98 0.985 0.99 0.995)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

for FRAC in "${FRACS[@]}"; do
    # tag: 0.50 -> "0500",  0.97 -> "0970",  0.99 -> "0990"
    # (was: `... | bc`; bc is not installed here -> TAG collapsed to 0000 for
    #  every frac, colliding output paths so only the first frac ever ran.
    #  awk is always present and needs no external calc tool.)
    TAG=$(printf "%04d" "$(awk "BEGIN{printf \"%d\", $FRAC*1000}")")
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "=========================================="
        echo "  SAE_frac=${FRAC}  |  SEED=${SEED}"
        echo "=========================================="
        
        DATA=./data/train_imb_${TAG}_s${SEED}.csv
        
        # 1) prepare
        python prepare_imbalanced.py \
            --sae_frac $FRAC --seed $SEED \
            --output $DATA
        
        # 2) ERM
        OUT=./results/sweep_ratio/erm_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py \
                --train_csv $DATA \
                --no_group_balance \
                --seed $SEED \
                --output_dir $OUT
            find $OUT -name "checkpoint-*" -type d -exec rm -rf {} +
        else
            echo "  [skip] ERM already done at $OUT"
        fi
        
        # 3) DRO
        OUT=./results/sweep_ratio/dro_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py \
                --train_csv $DATA \
                --robust --robust_step_size 0.01 --gamma 0.1 \
                --seed $SEED \
                --output_dir $OUT
            find $OUT -name "checkpoint-*" -type d -exec rm -rf {} +
        else
            echo "  [skip] DRO already done at $OUT"
        fi
    done
done

echo ""
echo "=========================================="
echo "  Sweep done. Plotting..."
echo "=========================================="
python plot_ratio_sweep.py