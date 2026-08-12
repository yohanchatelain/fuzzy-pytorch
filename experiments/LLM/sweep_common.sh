#!/bin/bash
# Shared settings for the perplexity sweeps.
#
# Override from the environment rather than editing:
#   CONTAINER_RUNTIME   podman (default) or docker; slashbin has only docker
#   RESULTS_ROOT        where logs go; on a cluster point this at shared storage
#   MAX_JOBS            concurrent containers; each pins itself to one thread,
#                       so this is bounded by cores and by ~1 GB of RAM apiece
#   IMAGE               container image to run

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
RESULTS_ROOT="${RESULTS_ROOT:-perplexity_logs}"
MAX_JOBS="${MAX_JOBS:-6}"
IMAGE="${IMAGE:-localhost/big-data-lab-team/fuzzy-llm-experiments:latest}"

# SR is stochastic, so an SR arm needs replication to carry an error bar. RN
# through PRISM fixes the rounding threshold at 1/2 and is bit-reproducible
# across repetitions, so replicating it would only burn compute.
SR_SEEDS="${SR_SEEDS:-1 2 3 4 5}"
RN_SEEDS="${RN_SEEDS:-1}"

seeds_for_mode() {
    if [ "$1" = "rn" ]; then echo "$RN_SEEDS"; else echo "$SR_SEEDS"; fi
}

# VFC_BACKENDS string for a given mode and seed.
backend_opts() {
    local mode=$1 seed=$2
    local opts="libinterflop_prism.so --seed=${seed}"
    [ "$mode" = "rn" ] && opts="${opts} --mode=rn"
    echo "$opts"
}

# The scripts are mounted from the host rather than taken from the image, so
# editing them does not mean rebuilding a 17 GB image. fuzzy_torch, which sets
# PRISM's precision and rounding mode, is installed in the image.
SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MOUNT_OPTS="${MOUNT_OPTS:-}"   # e.g. ":z" on SELinux hosts

# Threads per run. PRISM reconciles a precision change with every running
# thread, so a run no longer has to be pinned to one thread for the per-module
# scoping to take effect; the pin is now only a way to bound how much of the
# machine each concurrent container claims. Leave empty to let the container
# decide. Set to 1 to reproduce the single-threaded control.
THREADS_PER_RUN="${THREADS_PER_RUN:-}"

thread_env() {
    [ -z "$THREADS_PER_RUN" ] && return 0
    printf '%s' "-e OMP_NUM_THREADS=${THREADS_PER_RUN} -e MKL_NUM_THREADS=${THREADS_PER_RUN}"
}

# Run one configuration. Args: <log file> <backend opts> <python args...>
run_container() {
    local log_file=$1; shift
    local backend=$1; shift
    # shellcheck disable=SC2046  # word splitting is what thread_env is for
    "$CONTAINER_RUNTIME" run --rm \
        -v "${SRC_DIR}:/workspace${MOUNT_OPTS}" \
        -w /workspace \
        -e PYTHONPATH="/workspace" \
        $(thread_env) \
        -e VFC_BACKENDS="$backend" \
        "$IMAGE" \
        python3 -u "$@" > "$log_file" 2>&1
}

# eval_utils caches the WikiText slice in the working directory. Fetch it once
# up front so that a few hundred concurrent containers do not each hit the
# network for the same file.
ensure_dataset_cache() {
    local cache="${SRC_DIR}/wikitext-2-test.txt"
    [ -s "$cache" ] && return 0
    echo "==> Priming WikiText-2 cache at ${cache}"
    "$CONTAINER_RUNTIME" run --rm \
        -v "${SRC_DIR}:/workspace${MOUNT_OPTS}" -w /workspace \
        -e PYTHONPATH="/workspace" \
        "$IMAGE" python3 -c "
from transformers import AutoTokenizer
import eval_utils
eval_utils.load_wikitext_slice(AutoTokenizer.from_pretrained('distilgpt2'), fraction=100)
" >/dev/null 2>&1 || { echo "WARNING: could not prime dataset cache; runs will each download it"; return 0; }
    [ -s "$cache" ] && echo "==> Cache primed ($(wc -c < "$cache") bytes)"
}

# Block until fewer than MAX_JOBS children are running.
throttle() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n 2>/dev/null || true
    done
}
