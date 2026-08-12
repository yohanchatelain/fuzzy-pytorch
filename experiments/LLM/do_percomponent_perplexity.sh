#!/bin/bash
set -e
source "$(dirname "$0")/sweep_common.sh"

# Arguments
CONTEXT_LENGTH="${1:-384}"
MODE="${2:-sr}"
MAX_JOBS="${3:-$MAX_JOBS}"

# See do_fine_perplexity.sh: score several windows, not one.
MAX_TOKENS="${MAX_TOKENS:-1024}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode] [max_jobs]"
    exit 1
fi

LOG_DIR="${RESULTS_ROOT}/${CONTEXT_LENGTH}/ppl_percomponents_${MODE}"
mkdir -p "$LOG_DIR"

TARGET_GROUPS=("attention" "mlp" "lm_head")
# Unit spaced, as in do_fine_perplexity.sh: a two-bit grid cannot separate a
# stagnation threshold from a steep trend.
PRECISIONS=($(seq "${T_MIN:-4}" "${T_MAX:-14}"))

SEEDS=$(seeds_for_mode "$MODE")

echo "Per-component sweep"
echo "  context/window: ${CONTEXT_LENGTH}   tokens scored: ${MAX_TOKENS}"
echo "  mode:           ${MODE}   seeds: ${SEEDS}"
echo "  runtime:        ${CONTAINER_RUNTIME}   max jobs: ${MAX_JOBS}"
echo "  logs:           ${LOG_DIR}"

ensure_dataset_cache

for group in "${TARGET_GROUPS[@]}"; do
    for prec in "${PRECISIONS[@]}"; do
        for seed in $SEEDS; do
            log_file="$LOG_DIR/group_${group}_prec_${prec}_seed_${seed}.log"
            [ -f "$log_file" ] && { echo "skip (exists): $log_file"; continue; }
            throttle
            echo "Starting $group prec=$prec seed=$seed"
            run_container "$log_file" "$(backend_opts "$MODE" "$seed")" \
                test_percomponent_perplexity.py --group "$group" --precision "$prec" \
                --context_length "$CONTEXT_LENGTH" --max_tokens "$MAX_TOKENS" &
        done
    done
done

wait
echo "All ${MODE} runs completed. Logs in ${LOG_DIR}"
