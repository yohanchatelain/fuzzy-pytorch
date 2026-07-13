#!/bin/bash
set -e

# Create local Hugging Face cache directory if it doesn't exist
mkdir -p hf_cache

# Making it generic for easy addition of other precisions later
PRECISIONS=(2 4 6 8 10 12 24)
OUTPUT_CSV="perplexity_results.csv"

# Clear/Initialize the output file
echo "precision,perplexity" > "$OUTPUT_CSV"

# Temporary directory for logs
LOG_DIR="perplexity_logs/global"
mkdir -p "$LOG_DIR"

echo "Starting parallel runs for precisions: ${PRECISIONS[*]}..."

pids=()
job_precs=()

for prec in "${PRECISIONS[@]}"; do
    if [ "$prec" -eq 24 ]; then
        VFC_BACKEND="libinterflop_prism.so"
    else
        VFC_BACKEND="libinterflop_prism.so --precision-binary32=$prec --precision-binary64=$prec"
    fi

    # Run in background and redirect output to a log file
    podman run --rm \
        -e PYTHONPATH="/experiments/LLM/omp_ext" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e VFC_BACKENDS="$VFC_BACKEND" \
        localhost/big-data-lab-team/fuzzy-llm-experiments:latest \
        python3 -u test_perplexity.py > "$LOG_DIR/prec_$prec.log" 2>&1 &
    
    pids+=($!)
    job_precs+=($prec)
done

# Wait for all background jobs to finish
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    prec=${job_precs[$i]}
    wait "$pid"
    echo "Completed precision $prec bits"
done

echo ""
echo "All runs completed. Extracting perplexity to $OUTPUT_CSV..."

# Collate the outputs into CSV
for prec in "${PRECISIONS[@]}"; do
    log_file="$LOG_DIR/prec_$prec.log"
    
    # Extract perplexity value
    if grep -q "Final Perplexity:" "$log_file"; then
        ppl_val=$(grep "Final Perplexity:" "$log_file" | awk '{print $3}')
        echo "$prec,$ppl_val" >> "$OUTPUT_CSV"
    else
        echo "$prec,ERROR" >> "$OUTPUT_CSV"
        echo "Error: Perplexity evaluation failed or timed out for $prec bits. Log output:"
        cat "$log_file"
    fi
done

echo "Done! Generations written to $OUTPUT_CSV"
cat "$OUTPUT_CSV"
