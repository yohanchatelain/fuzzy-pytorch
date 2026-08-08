#!/bin/bash
set -e
source "$(dirname "$0")/sweep_common.sh"

# Arguments
CONTEXT_LENGTH="${1:-256}"
MODE="${2:-sr}"
MAX_JOBS="${3:-$MAX_JOBS}"

# This level maps positional sensitivity rather than producing a headline
# number, so it stays at a single window; per-component and per-sublayer are
# the levels that carry the paper's quantitative claims.
MAX_TOKENS="${MAX_TOKENS:-$CONTEXT_LENGTH}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode] [max_jobs]"
    exit 1
fi

# Restricted to the 3072-length reduction. It is where the analysis predicts the
# largest SR advantage, and crossing all four sublayers with six blocks would
# spend most of the compute budget on a qualitative figure.
TARGET_LAYERS=("mlp_c_proj")
PRECISIONS=(4 6)
BLOCKS=(0 1 2 3 4 5)

SEEDS=$(seeds_for_mode "$MODE")
LOG_DIR="${RESULTS_ROOT}/${CONTEXT_LENGTH}/blockwise_${MODE}"
mkdir -p "$LOG_DIR"

echo "Blockwise sweep"
echo "  context/window: ${CONTEXT_LENGTH}   tokens scored: ${MAX_TOKENS}"
echo "  layers:         ${TARGET_LAYERS[*]}"
echo "  mode:           ${MODE}   seeds: ${SEEDS}"
echo "  runtime:        ${CONTAINER_RUNTIME}   max jobs: ${MAX_JOBS}"
echo "  logs:           ${LOG_DIR}"

ensure_dataset_cache

for block in "${BLOCKS[@]}"; do
    for layer in "${TARGET_LAYERS[@]}"; do
        for prec in "${PRECISIONS[@]}"; do
            for seed in $SEEDS; do
                log_file="$LOG_DIR/block_${block}_layer_${layer}_prec_${prec}_seed_${seed}.log"
                [ -f "$log_file" ] && { echo "skip (exists): $log_file"; continue; }
                throttle
                echo "Starting block=$block $layer prec=$prec seed=$seed"
                run_container "$log_file" "$(backend_opts "$MODE" "$seed")" \
                    test_fine_perplexity.py --layer "$layer" --precision "$prec" \
                    --block_idx "$block" --context_length "$CONTEXT_LENGTH" \
                    --max_tokens "$MAX_TOKENS" &
            done
        done
    done
done

wait
echo "All blockwise runs completed for mode=${MODE}. Logs in ${LOG_DIR}"
