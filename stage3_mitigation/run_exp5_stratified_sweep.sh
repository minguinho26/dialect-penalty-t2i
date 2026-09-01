#!/bin/bash
# =====================================================================
# run_exp5_stratified_sweep.sh
#
# Full re-run of the imbalance sweep after fixing prepare_imbalanced.py
# to stratify the non-SAE budget over the 12 (dialect x label) groups
# instead of the 6 dialects. The old label-blind subsampling left whole
# (dialect, label) groups empty at extreme SAE fractions, which is
# inconsistent with the 12-group GroupDRO objective. This re-run removes
# that artifact.
#
# Fresh, isolated outputs (old results are preserved for comparison):
#   data:    data/train_imb12_<tag>_s<seed>.csv
#   results: results/sweep_ratio_v2/{erm,ermgb,dro}_<tag>_s<seed>/
#
# Per (frac, seed) all three configs share ONE data file, so erm / ermgb /
# dro remain paired per seed:
#   erm   = mean loss + random sampling        (--no_group_balance)
#   ermgb = mean loss + group-balanced sampling (default)
#   dro   = worst-group loss + group-balanced sampling (--robust)
# =====================================================================
set -e

mkdir -p ./results/sweep_ratio_v2

FRACS=(0.975 0.98 0.985 0.99 0.995)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

for FRAC in "${FRACS[@]}"; do
    TAG=$(printf "%04d" "$(awk "BEGIN{printf \"%d\", $FRAC*1000}")")

    for SEED in "${SEEDS[@]}"; do
        DATA=./data/train_imb12_${TAG}_s${SEED}.csv

        echo ""
        echo "######################################################"
        echo "#  SAE_frac=${FRAC}  SEED=${SEED}  (12-group stratified)"
        echo "######################################################"

        # 1) prepare (12-group stratified)
        python prepare_imbalanced.py \
            --sae_frac "$FRAC" --seed "$SEED" \
            --output "$DATA" > /dev/null

        # 2) ERM (random sampling)
        OUT=./results/sweep_ratio_v2/erm_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py --train_csv "$DATA" --no_group_balance \
                --seed "$SEED" --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else echo "  [skip] $OUT"; fi

        # 3) ERM + group-balanced sampling
        OUT=./results/sweep_ratio_v2/ermgb_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py --train_csv "$DATA" \
                --seed "$SEED" --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else echo "  [skip] $OUT"; fi

        # 4) GroupDRO (worst-group loss + group-balanced sampling)
        OUT=./results/sweep_ratio_v2/dro_${TAG}_s${SEED}
        if [ ! -f "$OUT/test_metrics.json" ]; then
            python train_group_dro.py --train_csv "$DATA" \
                --robust --robust_step_size 0.01 --gamma 0.1 \
                --seed "$SEED" --output_dir "$OUT"
            find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
        else echo "  [skip] $OUT"; fi
    done
done

echo ""
echo "######################################################"
echo "#  Stratified sweep done. Aggregate with:"
echo "#    python make_table8_markdown.py --ablation \\"
echo "#      --sweep-subdir sweep_ratio_v2 --data-prefix train_imb12 \\"
echo "#      --seeds 0 1 2 3 4 5 6 7 8 9 --out results/table8_v2.md"
echo "######################################################"
