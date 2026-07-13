#!/bin/bash
set -e

echo "=== Starting Parallel Mixed Rounding / Mixed Precision Evaluations ==="

CONFIGS=("proposed_recipe")
MAX_JOBS=4
running=0

for config in "${CONFIGS[@]}"; do
    echo "Launching config: $config in background..."
    podman run --rm \
        -e PYTHONPATH="/experiments/LLM/omp_ext" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e VFC_BACKENDS="libinterflop_prism.so" \
        localhost/big-data-lab-team/fuzzy-llm-experiments:latest \
        python3 -u test_ideal_mixed.py --config "$config" > "mixed_eval_${config}.log" 2>&1 &
    
    running=$((running + 1))
    
    if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n || true
        running=$((running - 1))
    fi
done

# Wait for all background runs to complete
wait

echo "=== All evaluations completed! ==="
for config in "${CONFIGS[@]}"; do
    grep "Result ->" "mixed_eval_${config}.log" || true
done
