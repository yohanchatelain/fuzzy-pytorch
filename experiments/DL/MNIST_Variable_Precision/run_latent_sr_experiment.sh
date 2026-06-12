#!/bin/bash
#
# MNIST Latent Space Variability Experiment with PRISM Variable-Precision SR
#
# This script:
#   1. Trains a baseline MNIST model under IEEE mode (deterministic)
#   2. Runs inference N times under SR at various precision levels
#   3. Saves layer activations for analysis
#
# Prerequisites:
#   - Container image: big-data-lab-team/fuzzy-pytorch:sr
#   - Run from: experiments/DL/MNIST_Variable_Precision/
#
# Usage:
#   ./run_latent_sr_experiment.sh
#

set -e
set -o pipefail

# === Configuration ===
DOCKER_IMAGE="localhost/big-data-lab-team/fuzzy-pytorch:sr"
N_SAMPLES=3
WORK_DIR="$(pwd)/mnist_sr_experiment"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Precision levels to sweep (mantissa bits for binary32)
# 24 = full IEEE float32, lower = more SR noise
PRECISIONS=(1 2 4 8 12 16 24)

# Thread control (reproducibility)
DOCKER_ENV="-e OMP_NUM_THREADS=1 -e NUMPEXPR_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1"

echo "=================================================="
echo "MNIST Latent Space Variability with PRISM SR"
echo "=================================================="
echo "Docker image: ${DOCKER_IMAGE}"
echo "Samples per precision: ${N_SAMPLES}"
echo "Precisions: ${PRECISIONS[*]}"
echo "Output directory: ${WORK_DIR}"
echo "=================================================="

# === Create output directories ===
mkdir -p "${WORK_DIR}/model"
mkdir -p "${WORK_DIR}/data"
for prec in "${PRECISIONS[@]}"; do
    mkdir -p "${WORK_DIR}/results/precision_${prec}"
done

# === Step 1: Train baseline model on host ===
MODEL_PATH="${WORK_DIR}/model/mnist_cnn.pt"
if [ -f "${MODEL_PATH}" ]; then
    echo ""
    echo "[Step 1] Model already exists at ${MODEL_PATH}, skipping training."
else
    echo ""
    echo "[Step 1] Training MNIST model under IEEE mode (deterministic)..."
    echo "         This may take a few minutes."
    python3 "${SCRIPTS_DIR}/mnist_train.py" \
        --save-path "${MODEL_PATH}" \
        --data-dir "${WORK_DIR}/data" \
        --epochs 5
    echo "[Step 1] Training complete. Model saved to ${MODEL_PATH}"
fi

# === Step 2: Run inference under SR at each precision level ===
echo ""
echo "[Step 2] Running inference experiments..."

for prec in "${PRECISIONS[@]}"; do
    echo ""
    echo "--- Precision: ${prec} bits (binary32) ---"

    if [ "${prec}" -eq 24 ]; then
        VFC_BACKEND="libinterflop_prism.so"
    else
        VFC_BACKEND="libinterflop_prism.so --precision-binary32=${prec} --precision-binary64=${prec}"
    fi

    echo "  Running ${N_SAMPLES} inference sweeps in same process..."
    podman run --rm \
        ${DOCKER_ENV} \
        -e LD_PRELOAD="/usr/local/lib/libhwy.so" \
        -e VFC_BACKENDS="${VFC_BACKEND}" \
        -v "${SCRIPTS_DIR}:/scripts:ro" \
        -v "${WORK_DIR}:/mnist" \
        ${DOCKER_IMAGE} \
        python3 /scripts/mnist_latent_experiment.py \
            --model-path /mnist/model/mnist_cnn.pt \
            --output-dir /mnist/results/precision_${prec} \
            --data-dir /mnist/data \
            --num-runs ${N_SAMPLES}
done

echo ""
echo "=================================================="
echo "[Done] All experiments complete!"
echo ""
echo "Results saved to: ${WORK_DIR}/results/"
echo "=================================================="
