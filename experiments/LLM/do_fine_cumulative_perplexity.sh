#!/bin/bash
set -e

# Arguments
CONTEXT_LENGTH="${1:-256}"
MODE="${2:-sr}"
MAX_JOBS="${3:-6}"

if [ "$MODE" != "sr" ] && [ "$MODE" != "rn" ]; then
    echo "Error: Mode must be 'sr' or 'rn'."
    echo "Usage: $0 [context_length] [mode] [max_jobs]"
    echo "Example: $0 256 sr 10"
    exit 1
fi

TARGET_LAYERS=("attn_c_attn" "attn_c_proj" "mlp_c_fc" "mlp_c_proj")
PRECISIONS=(4 6)
BLOCK_CONFIGS=("0" "0-1" "0-2" "0-3" "0-4" "0-5")

echo "Starting cumulative fine-grained perplexity evaluation (Bottom-Up)..."
echo "Context Length: ${CONTEXT_LENGTH}"
echo "Rounding Mode:  ${MODE}"
echo "Max Jobs:       ${MAX_JOBS}"

running=0

run_config() {
    local block_conf=$1
    local layer=$2
    local prec=$3
    local log_dir="perplexity_logs/${CONTEXT_LENGTH}/fine_${MODE}_cumul_${block_conf}"
    mkdir -p "$log_dir"
    local log_file="$log_dir/layer_${layer}_prec_${prec}.log"
    
    echo "Starting Cumulative Blocks ${block_conf}, Layer ${layer} at precision ${prec}..."
    
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
            --block_idx "$block_conf" \
            --context_length "$CONTEXT_LENGTH" > "$log_file" 2>&1
    
    echo "Completed Cumulative Blocks ${block_conf}, Layer ${layer} at precision ${prec}"
}

# Sweep block configs, layers, and precisions
for block_conf in "${BLOCK_CONFIGS[@]}"; do
    for layer in "${TARGET_LAYERS[@]}"; do
        for prec in "${PRECISIONS[@]}"; do
            
            run_config "$block_conf" "$layer" "$prec" &
            running=$((running + 1))
            
            # If we reached max concurrent jobs, wait for at least one to finish
            if [ "$running" -ge "$MAX_JOBS" ]; then
                wait -n || true
                running=$((running - 1))
            fi
        done
    done
done

# Wait for all remaining background jobs to finish
wait
echo "All cumulative fine-grained runs completed for mode=${MODE}."
