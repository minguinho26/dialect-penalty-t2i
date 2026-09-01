#!/bin/bash
# =====================================================================
# run_exp7_innersplit_ablation.sh
#
# Full Table-8 sweep in the INNER-SPLIT setting (train on the 90% inner
# split, matching the original submission's training), on the 12-group
# stratified data. This is the setup reported in the paper because it
# matches how the submitted results were produced.
#
#   sweep  : erm / ermgb / dro  x  {97.5..99.5}  x 10 seeds
#   anchors: SAE-only ERM (100%) + Balanced ERM  x 10 seeds
#   all with --inner_split, into results/sweep_ratio_v2_is/
#
# Already-done runs are skipped, so re-running only fills the gaps.
# =====================================================================
set -e

mkdir -p ./results/sweep_ratio_v2_is

FRACS=(0.975 0.98 0.985 0.99 0.995)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# ---------- imbalance sweep ----------
for FRAC in "${FRACS[@]}"; do
    TAG=$(printf "%04d" "$(awk "BEGIN{printf \"%d\", $FRAC*1000}")")

    for SEED in "${SEEDS[@]}"; do
        DATA=./data/train_imb12_${TAG}_s${SEED}.csv
        if [ ! -f "$DATA" ]; then
            python prepare_imbalanced.py --sae_frac "$FRAC" --seed "$SEED" \
                --output "$DATA" > /dev/null
        fi

        echo ""
        echo "######  inner-split  SAE_frac=${FRAC}  SEED=${SEED}  ######"

        OUT=./results/sweep_ratio_v2_is/erm_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py --train_csv "$DATA" --inner_split --no_group_balance \
                --seed "$SEED" --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else echo "  [skip] $OUT"; fi

        OUT=./results/sweep_ratio_v2_is/ermgb_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py --train_csv "$DATA" --inner_split \
                --seed "$SEED" --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else echo "  [skip] $OUT"; fi

        OUT=./results/sweep_ratio_v2_is/dro_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py --train_csv "$DATA" --inner_split \
                --robust --robust_step_size 0.01 --gamma 0.1 \
                --seed "$SEED" --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else echo "  [skip] $OUT"; fi
    done
done

# ---------- anchors (SAE-only 100%, Balanced), also inner-split ----------
python prepare_sae_only.py > /dev/null
SAE_DATA=./data/train_sae_only.csv

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "######  inner-split ANCHORS  SEED=${SEED}  ######"

    OUT=./results/sweep_ratio_v2_is/sae_only_erm_s${SEED}
    if [ ! -f "$OUT/test_metrics.json" ]; then
        python train_group_dro.py --train_csv "$SAE_DATA" --inner_split --no_group_balance \
            --metric_for_best_model accuracy --seed "$SEED" --output_dir "$OUT"
        find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
    else echo "  [skip] $OUT"; fi

    BAL_DATA=./data/train_bal12_s${SEED}.csv
    python prepare_balanced.py --seed "$SEED" --output "$BAL_DATA" > /dev/null
    OUT=./results/sweep_ratio_v2_is/erm_balanced_s${SEED}
    if [ ! -f "$OUT/test_metrics.json" ]; then
        python train_group_dro.py --train_csv "$BAL_DATA" --inner_split --no_group_balance \
            --seed "$SEED" --output_dir "$OUT"
        find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
    else echo "  [skip] $OUT"; fi
done

echo ""
echo "######################################################"
echo "#  Inner-split full sweep done. Aggregate with:"
echo "#    python make_table8_markdown.py --ablation \\"
echo "#      --sweep-subdir sweep_ratio_v2_is \\"
echo "#      --saeonly-subdir sweep_ratio_v2_is --balanced-subdir sweep_ratio_v2_is \\"
echo "#      --seeds 0 1 2 3 4 5 6 7 8 9 --out results/table8_v2_is_full.md"
echo "######################################################"
