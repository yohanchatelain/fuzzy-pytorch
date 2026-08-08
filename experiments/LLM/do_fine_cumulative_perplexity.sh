#!/bin/bash
set -e
source "$(dirname "$0")/sweep_common.sh"

# Arguments
CONTEXT_LENGTH="${1:-256}"
MODE="${2:-sr}"
MAX_JOBS="${3:-$MAX_JOBS}"

# As in the blockwise sweep: a propagation curve, not a headline number.
MAX_TOKENS="${MAX_TOKENS:-$CONTEXT_LENGTH}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode] [max_jobs]"
    exit 1
fi

# Restricted to the 3072-length reduction, as in do_fine_blockwise_perplexity.sh.
TARGET_LAYERS=("mlp_c_proj")
PRECISIONS=(4 6)
BLOCK_CONFIGS=("0" "0-1" "0-2" "0-3" "0-4" "0-5")

SEEDS=$(seeds_for_mode "$MODE")
LOG_DIR="${RESULTS_ROOT}/${CONTEXT_LENGTH}/cumulative_${MODE}"
mkdir -p "$LOG_DIR"

echo "Cumulative sweep"
echo "  context/window: ${CONTEXT_LENGTH}   tokens scored: ${MAX_TOKENS}"
echo "  layers:         ${TARGET_LAYERS[*]}"
echo "  mode:           ${MODE}   seeds: ${SEEDS}"
echo "  runtime:        ${CONTAINER_RUNTIME}   max jobs: ${MAX_JOBS}"
echo "  logs:           ${LOG_DIR}"

ensure_dataset_cache

for block_conf in "${BLOCK_CONFIGS[@]}"; do
    for layer in "${TARGET_LAYERS[@]}"; do
        for prec in "${PRECISIONS[@]}"; do
            for seed in $SEEDS; do
                safe_conf="${block_conf//-/_}"
                log_file="$LOG_DIR/blocks_${safe_conf}_layer_${layer}_prec_${prec}_seed_${seed}.log"
                [ -f "$log_file" ] && { echo "skip (exists): $log_file"; continue; }
                throttle
                echo "Starting blocks=$block_conf $layer prec=$prec seed=$seed"
                run_container "$log_file" "$(backend_opts "$MODE" "$seed")" \
                    test_fine_perplexity.py --layer "$layer" --precision "$prec" \
                    --block_idx "$block_conf" --context_length "$CONTEXT_LENGTH" \
                    --max_tokens "$MAX_TOKENS" &
            done
        done
    done
done

wait
echo "All cumulative runs completed for mode=${MODE}. Logs in ${LOG_DIR}"
