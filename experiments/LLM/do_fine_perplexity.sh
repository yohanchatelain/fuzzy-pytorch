#!/bin/bash
set -e

# Arguments
CONTEXT_LENGTH="${1:-256}"
MODE="${2:-sr}"
BLOCK_IDX="${3:-all}"
MAX_JOBS="${4:-6}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode] [block_idx] [max_jobs]"
    echo "Example: $0 256 sr all 10"
    exit 1
fi

LOG_DIR="perplexity_logs/${CONTEXT_LENGTH}/fine_${MODE}_block_${BLOCK_IDX}"
mkdir -p "$LOG_DIR"

TARGET_LAYERS=("attn_c_attn" "attn_c_proj" "mlp_c_fc" "mlp_c_proj")
PRECISIONS=(4 6 8)

echo "Starting fine-grained per-component perplexity evaluation..."
echo "Context Length: ${CONTEXT_LENGTH}"
echo "Rounding Mode:  ${MODE}"
echo "Block Selection: ${BLOCK_IDX}"
echo "Max Jobs:       ${MAX_JOBS}"
echo "Logs will be written to ${LOG_DIR}"

VFC_BACKEND="libinterflop_prism.so"
running=0

run_config() {
    local layer=$1
    local prec=$2
    local log_file="$LOG_DIR/layer_${layer}_prec_${prec}.log"
    
    echo "Starting layer $layer at precision $prec..."
    
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
        python3 -u test_fine_perplexity.py \
            --layer "$layer" \
            --precision "$prec" \
            --block_idx "$BLOCK_IDX" \
            --context_length "$CONTEXT_LENGTH" > "$log_file" 2>&1
    
    echo "Completed layer $layer at precision $prec"
}

# 1. Run baseline 24-bit once if it hasn't been run or to have a clean baseline log
run_baseline() {
    local log_file="$LOG_DIR/baseline_prec_24.log"
    if [ ! -f "$log_file" ]; then
        echo "Running baseline 24-bit configuration..."
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
            python3 -u test_fine_perplexity.py \
                --layer "attn_c_attn" \
                --precision 24 \
                --block_idx "all" \
                --context_length "$CONTEXT_LENGTH" > "$log_file" 2>&1
        echo "Completed baseline 24-bit run"
    else
        echo "Baseline log already exists at $log_file"
    fi
}

run_baseline

# 2. Sweep target layers and precisions
for layer in "${TARGET_LAYERS[@]}"; do
    for prec in "${PRECISIONS[@]}"; do
        
        run_config "$layer" "$prec" &
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
echo "All fine-grained runs completed for mode=${MODE}, block=${BLOCK_IDX}. Logs are in ${LOG_DIR}"
