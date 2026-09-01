#!/bin/bash
# =====================================================================
# run_exp6_anchors_v2.sh
#
# Re-run the two anchor conditions for the 12-group (dialect x label)
# world, into the same fresh results dir as the stratified sweep
# (results/sweep_ratio_v2), with 10 seeds:
#
#   SAE-only ERM (100% SAE)  -> sweep_ratio_v2/sae_only_erm_s<seed>
#   Balanced ERM (1/12 each) -> sweep_ratio_v2/erm_balanced_s<seed>
#
# SAE-only data is deterministic (all SAE rows) and unaffected by the
# label-stratification fix; only the balanced set changes (now uniform
# over the 12 groups via the fixed prepare_balanced.py).
#
# Run run_exp5_stratified_sweep.sh for the ERM/ERM-GB/GroupDRO sweep.
# =====================================================================
set -e

mkdir -p ./results/sweep_ratio_v2

SEEDS=(0 1 2 3 4 5 6 7 8 9)

# SAE-only source data (deterministic; regenerate once)
python prepare_sae_only.py > /dev/null
SAE_DATA=./data/train_sae_only.csv

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "######################################################"
    echo "#  ANCHORS  SEED=${SEED}"
    echo "######################################################"

    # --- SAE-only ERM (100% SAE) ---
    OUT=./results/sweep_ratio_v2/sae_only_erm_s${SEED}
    if [ ! -f "$OUT/test_metrics.json" ]; then
        python train_group_dro.py --train_csv "$SAE_DATA" --no_group_balance \
            --metric_for_best_model accuracy \
            --seed "$SEED" --output_dir "$OUT"
        find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
    else echo "  [skip] $OUT"; fi

    # --- Balanced ERM (uniform over 12 groups) ---
    BAL_DATA=./data/train_bal12_s${SEED}.csv
    python prepare_balanced.py --seed "$SEED" --output "$BAL_DATA" > /dev/null

    OUT=./results/sweep_ratio_v2/erm_balanced_s${SEED}
    if [ ! -f "$OUT/test_metrics.json" ]; then
        python train_group_dro.py --train_csv "$BAL_DATA" --no_group_balance \
            --seed "$SEED" --output_dir "$OUT"
        find "$OUT" -name "checkpoint-*" -type d -exec rm -rf {} +
    else echo "  [skip] $OUT"; fi
done

echo ""
echo "######################################################"
echo "#  Anchors done. Aggregate the full v2 table with:"
echo "#    python make_table8_markdown.py --ablation \\"
echo "#      --sweep-subdir sweep_ratio_v2 \\"
echo "#      --saeonly-subdir sweep_ratio_v2 --balanced-subdir sweep_ratio_v2 \\"
echo "#      --seeds 0 1 2 3 4 5 6 7 8 9 --out results/table8_v2.md"
echo "######################################################"
