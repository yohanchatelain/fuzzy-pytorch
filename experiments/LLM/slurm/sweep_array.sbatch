#!/bin/bash
#SBATCH --job-name=llm-sweep
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#
# One array task per sweep configuration. Submit with the config list produced
# by gen_configs.sh:
#
#   ./gen_configs.sh > ~/slurm-sweeps/configs.txt
#   sbatch --array=1-$(wc -l < ~/slurm-sweeps/configs.txt)%384 \
#          sweep_array.sbatch ~/slurm-sweeps/configs.txt
#
# One CPU per task on purpose. The threading control measured 850s at one
# thread against 788s at eight for the same configuration, so a second core
# buys about 7% while a second task buys a whole configuration.
#
# Each task writes the log the sweep scripts would have written, so the
# resulting tree is indistinguishable from a local run and parse_sweep_logs.py
# reads it unchanged. Slurm's own stdout goes to /dev/null: the payload writes
# the file itself, and an empty log would otherwise be mistaken for a finished
# configuration.
set -u

CONFIG_FILE="${1:?usage: sbatch sweep_array.sbatch <config file>}"
SIF="${SIF:-$HOME/sif/fuzzy-llm.sif}"
WORKSPACE="${WORKSPACE:-$HOME/fuzzy-llm/experiments/LLM}"

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")
[ -z "$line" ] && { echo "no configuration at line $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

log_file=${line%%|*}
rest=${line#*|}
backend=${rest%%|*}
args=${rest#*|}

# Resumability: another task, or an earlier submission, may have finished this
# configuration already.
if [ -f "$log_file" ] && grep -q "Result ->" "$log_file" 2>/dev/null; then
    exit 0
fi

mkdir -p "$(dirname "$log_file")"

# --containall keeps the host environment out; the binds supply the harness and
# the dataset cache, both on shared home. --pwd sets the working directory
# inside the container, where eval_utils looks for the WikiText slice.
apptainer exec \
    --containall \
    --bind "${WORKSPACE}:/workspace" \
    --pwd /workspace \
    --env PYTHONPATH=/workspace \
    --env VFC_BACKENDS_LOGGER=False \
    --env VFC_BACKENDS="$backend" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env HF_HOME=/hf_cache \
    "$SIF" \
    python3 -u $args > "$log_file" 2>&1

status=$?

# A task that died without producing a result must not leave a log behind: the
# sweep scripts and gen_configs.sh both treat an existing log as work done.
if ! grep -q "Result ->" "$log_file" 2>/dev/null; then
    rm -f "$log_file"
    exit "${status:-1}"
fi
