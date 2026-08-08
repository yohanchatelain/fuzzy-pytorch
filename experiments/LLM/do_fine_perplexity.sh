#!/bin/bash
set -e
source "$(dirname "$0")/sweep_common.sh"

# Arguments
CONTEXT_LENGTH="${1:-256}"
MODE="${2:-sr}"
BLOCK_IDX="${3:-all}"
MAX_JOBS="${4:-$MAX_JOBS}"

# Score several windows rather than one: at MAX_TOKENS = CONTEXT_LENGTH the
# harness scores a single window, which is high-variance and comparable only
# within a sweep level.
MAX_TOKENS="${MAX_TOKENS:-1024}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode] [block_idx] [max_jobs]"
    exit 1
fi

LOG_DIR="${RESULTS_ROOT}/${CONTEXT_LENGTH}/fine_${MODE}_block_${BLOCK_IDX}"
mkdir -p "$LOG_DIR"

TARGET_LAYERS=("attn_c_attn" "attn_c_proj" "mlp_c_fc" "mlp_c_proj")

# Predicted stagnation onset is t ~ log2(n): 9.6 for the 768-length reductions
# and 11.6 for mlp_c_proj at 3072. Sweeping only {4,6,8} sits entirely inside
# the stagnation regime and cannot observe the transition, so 10 and 12 are
# included to bracket both onsets.
PRECISIONS=(4 6 8 10 12)

SEEDS=$(seeds_for_mode "$MODE")

echo "Fine-grained per-sublayer sweep"
echo "  context/window: ${CONTEXT_LENGTH}   tokens scored: ${MAX_TOKENS}"
echo "  mode:           ${MODE}   seeds: ${SEEDS}"
echo "  blocks:         ${BLOCK_IDX}"
echo "  runtime:        ${CONTAINER_RUNTIME}   max jobs: ${MAX_JOBS}"
echo "  logs:           ${LOG_DIR}"

run_baseline() {
    local log_file="$LOG_DIR/baseline_prec_24.log"
    if [ ! -f "$log_file" ]; then
        echo "Running baseline 24-bit configuration..."
        run_container "$log_file" "$(backend_opts "$MODE" 1)" \
            test_fine_perplexity.py --layer attn_c_attn --precision 24 \
            --block_idx all --context_length "$CONTEXT_LENGTH" --max_tokens "$MAX_TOKENS"
    else
        echo "Baseline log already exists at $log_file"
    fi
}

ensure_dataset_cache
run_baseline

for layer in "${TARGET_LAYERS[@]}"; do
    for prec in "${PRECISIONS[@]}"; do
        for seed in $SEEDS; do
            log_file="$LOG_DIR/layer_${layer}_prec_${prec}_seed_${seed}.log"
            [ -f "$log_file" ] && { echo "skip (exists): $log_file"; continue; }
            throttle
            echo "Starting $layer prec=$prec seed=$seed"
            run_container "$log_file" "$(backend_opts "$MODE" "$seed")" \
                test_fine_perplexity.py --layer "$layer" --precision "$prec" \
                --block_idx "$BLOCK_IDX" --context_length "$CONTEXT_LENGTH" \
                --max_tokens "$MAX_TOKENS" &
        done
    done
done

wait
echo "All fine-grained runs completed for mode=${MODE}, block=${BLOCK_IDX}. Logs in ${LOG_DIR}"
