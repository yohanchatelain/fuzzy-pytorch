#!/bin/bash
set -e

# Create local Hugging Face cache directory if it doesn't exist
mkdir -p hf_cache

PRECISIONS=(2 4 6 8 10 12 24)
OUTPUT_FILE="llm_generations.txt"

# Clear/Initialize the output file
echo "=== LLM Generations under Stochastic Rounding (PRISM) ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "=========================================================" >> "$OUTPUT_FILE"

# Temporary directory for logs
LOG_DIR="run_logs"
mkdir -p "$LOG_DIR"

echo "Starting parallel runs for precisions: ${PRECISIONS[*]}..."
echo "This will take about 4 minutes..."

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
        -e PYTHONPATH="/workspace" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e VFC_BACKENDS="$VFC_BACKEND" \
        localhost/big-data-lab-team/fuzzy-llm-experiments:latest \
        python3 -u test_llm.py > "$LOG_DIR/prec_$prec.log" 2>&1 &

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
echo "All runs completed. Collating outputs into $OUTPUT_FILE..."

# Collate the outputs
for prec in "${PRECISIONS[@]}"; do
    log_file="$LOG_DIR/prec_$prec.log"
    echo "" >> "$OUTPUT_FILE"
    echo "--------------------------------------------------------" >> "$OUTPUT_FILE"
    echo "Precision: $prec bits" >> "$OUTPUT_FILE"
    echo "--------------------------------------------------------" >> "$OUTPUT_FILE"

    # Extract the generated text between "--- GENERATED TEXT ---" and "----------------------"
    if grep -q -e "--- GENERATED TEXT ---" "$log_file"; then
        sed -n '/--- GENERATED TEXT ---/,/----------------------/p' "$log_file" | sed '1d;$d' >> "$OUTPUT_FILE"
    else
        echo "Error: Generation failed or timed out. Log output:" >> "$OUTPUT_FILE"
        cat "$log_file" >> "$OUTPUT_FILE"
    fi
done

# Clean up log directory (commented out for debugging)
# rm -rf "$LOG_DIR"

echo "Done! Generations written to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
