#!/bin/bash
set -e

mkdir -p hf_cache
LOG_DIR="mixed_generations_log"
mkdir -p "$LOG_DIR"

OUTPUT_FILE="mixed_generations_output.txt"
echo "=== LLM Mixed Generation (MLP 6-bit, others 10-bit) ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "=========================================================" >> "$OUTPUT_FILE"

echo "Starting Mixed Precision Generation..."
echo "Logs will be written to $LOG_DIR"

# Common environment variables
VFC_BACKEND="libinterflop_prism.so --precision-binary32=10 --precision-binary64=10"

# 1. Run SR Version
echo "Starting SR version in background..."
log_file_sr="$LOG_DIR/mixed_sr.log"

podman run --rm \
    -e PYTHONPATH="/workspace" \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e VFC_BACKENDS="$VFC_BACKEND" \
    localhost/big-data-lab-team/fuzzy-llm-experiments:latest \
    python3 -u test_mixed_generation.py > "$log_file_sr" 2>&1 &
pid_sr=$!

# 2. Run RN Version
echo "Starting RN version in background..."
log_file_rn="$LOG_DIR/mixed_rn.log"

podman run --rm \
    -e PYTHONPATH="/workspace" \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e VFC_BACKENDS="libinterflop_prism.so --precision-binary32=10 --precision-binary64=10 --mode=rn" \
    localhost/big-data-lab-team/fuzzy-llm-experiments:latest \
    python3 -u test_mixed_generation.py > "$log_file_rn" 2>&1 &
pid_rn=$!

echo "Waiting for both evaluations to complete in parallel..."
wait "$pid_sr"
echo "SR version completed."

wait "$pid_rn"
echo "RN version completed."

# Collect Results
echo "" >> "$OUTPUT_FILE"
echo "--- SR Results ---" >> "$OUTPUT_FILE"
cat "$log_file_sr" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "--- RN Results ---" >> "$OUTPUT_FILE"
cat "$log_file_rn" >> "$OUTPUT_FILE"

echo "Done! Outputs written to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
