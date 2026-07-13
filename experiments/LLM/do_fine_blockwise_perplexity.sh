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
BLOCKS=(0 1 2 3 4 5)

echo "Starting blockwise fine-grained perplexity evaluation..."
echo "Context Length: ${CONTEXT_LENGTH}"
echo "Rounding Mode:  ${MODE}"
echo "Max Jobs:       ${MAX_JOBS}"

VFC_BACKEND="libinterflop_prism.so"
running=0

run_config() {
    local block=$1
    local layer=$2
    local prec=$3
    local log_dir="perplexity_logs/${CONTEXT_LENGTH}/fine_${MODE}_block_${block}"
    mkdir -p "$log_dir"
    local log_file="$log_dir/layer_${layer}_prec_${prec}.log"
    
    echo "Starting Block $block, Layer $layer at precision $prec..."
    
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
            --block_idx "$block" \
            --context_length "$CONTEXT_LENGTH" > "$log_file" 2>&1
    
    echo "Completed Block $block, Layer $layer at precision $prec"
}

# Sweep blocks, layers, and precisions
for block in "${BLOCKS[@]}"; do
    for layer in "${TARGET_LAYERS[@]}"; do
        for prec in "${PRECISIONS[@]}"; do
            
            run_config "$block" "$layer" "$prec" &
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
echo "All blockwise fine-grained runs completed for mode=${MODE}."
