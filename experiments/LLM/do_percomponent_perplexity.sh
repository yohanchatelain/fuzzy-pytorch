#!/bin/bash
set -e

# Arguments
CONTEXT_LENGTH="${1:-384}"
MODE="${2:-sr}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode]"
    exit 1
fi

# Organize log directory
if [ "$MODE" = "sr" ]; then
    LOG_DIR="perplexity_logs/${CONTEXT_LENGTH}/ppl_percomponents_sr"
else
    LOG_DIR="perplexity_logs/${CONTEXT_LENGTH}/ppl_percomponents_rn"
fi

mkdir -p "$LOG_DIR"

TARGET_GROUPS=("attention" "mlp" "lm_head")
PRECISIONS=(4 6 8 10)
MAX_JOBS=6

echo "Starting per-component perplexity evaluation (Context Length: ${CONTEXT_LENGTH}, Mode: ${MODE}, Max Jobs: ${MAX_JOBS})..."
echo "Logs will be written to ${LOG_DIR}"

VFC_BACKEND="libinterflop_prism.so"

running=0

run_config() {
    local group=$1
    local prec=$2
    local log_file="$LOG_DIR/group_${group}_prec_${prec}.log"
    
    echo "Starting $group at precision $prec..."
    
    local backend_opts="libinterflop_prism.so"
    if [ "$MODE" = "rn" ]; then
        backend_opts="libinterflop_prism.so --mode=rn"
    fi

    local opts=(
        --rm
        -e PYTHONPATH="/experiments/LLM/omp_ext"
        -e OMP_NUM_THREADS=1
        -e MKL_NUM_THREADS=1
        -e VFC_BACKENDS="$backend_opts"
    )
    
    podman run "${opts[@]}" \
        localhost/big-data-lab-team/fuzzy-llm-experiments:latest \
        python3 -u test_percomponent_perplexity.py \
            --group "$group" \
            --precision "$prec" \
            --context_length "$CONTEXT_LENGTH" > "$log_file" 2>&1
    
    echo "Completed $group at precision $prec"
}

for group in "${TARGET_GROUPS[@]}"; do
    for prec in "${PRECISIONS[@]}"; do
        
        run_config "$group" "$prec" &
        running=$((running + 1))
        
        # If we reached max concurrent jobs, wait for at least one to finish
        if [ "$running" -ge "$MAX_JOBS" ]; then
            wait -n || true
            running=$((running - 1))
        fi
    done
done

# Wait for all remaining background jobs to finish
wait
echo "All ${MODE} runs completed. Logs are in ${LOG_DIR}"
