#!/bin/bash
set -e

# Usage helper
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <remote-connection> <script.sh> [args...]"
    echo "Example: $0 otterfall do_mixed_generation.sh"
    exit 1
fi

REMOTE_CONN="$1"
SCRIPT_FILE="$2"
# The image build context is the repository root: the Containerfile installs
# fuzzy_torch from python/ as well as the scripts from experiments/LLM.
REMOTE_ROOT="~/fuzzy-llm"
REMOTE_PATH="${REMOTE_ROOT}/experiments/LLM"

# Ensure we are in the correct directory (directory of the script)
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

# 1. Sync the workspace to the remote host using tar over ssh
echo "==> Syncing workspace to remote host '$REMOTE_CONN' using tar over ssh..."
ssh "$REMOTE_CONN" "mkdir -p ${REMOTE_PATH}"
tar -C "$REPO_ROOT" -cf - \
    --exclude='hf_cache' \
    --exclude='perplexity_logs' \
    --exclude='run_logs' \
    --exclude='mixed_generations_log' \
    --exclude='.git' \
    --exclude='build' \
    --exclude='*.so' \
    --exclude='*.png' \
    --exclude='*.csv' \
    --exclude='*.log' \
    --exclude='run_remote_tmux.sh' \
    python experiments/LLM | ssh "$REMOTE_CONN" "tar -xf - -C ${REMOTE_ROOT}/"

# 2. Enable systemd user lingering so processes persist after SSH disconnection
echo "==> Ensuring user lingering is enabled on remote server..."
ssh "$REMOTE_CONN" "loginctl enable-linger \$(whoami) 2>/dev/null || true"

# 3. Build the container image on the remote host to capture any synced script
#    changes. The base must already carry a PRISM with the configuration epoch;
#    see RUNBOOK.md for the one-off refresh that produces fuzzy-pytorch:sr-epoch.
echo "==> Building container image on remote host '$REMOTE_CONN'..."
# slashbin has docker and no podman, and sweep_common.sh already takes the
# runtime from the environment, so this must too rather than hardcoding one.
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
ssh "$REMOTE_CONN" "cd ${REMOTE_ROOT} && ${CONTAINER_RUNTIME} build -t localhost/big-data-lab-team/fuzzy-llm-experiments:latest -f experiments/LLM/Containerfile ."

# 4. Detect tmux binary on remote host
echo "==> Detecting tmux binary on remote host '$REMOTE_CONN'..."
TMUX_CMD=$(ssh "$REMOTE_CONN" '
    if command -v tmux &>/dev/null; then
        echo "tmux"
    elif [ -x "$HOME/bin/tmux" ]; then
        echo "$HOME/bin/tmux"
    elif [ -x "$HOME/.local/bin/tmux" ]; then
        echo "$HOME/.local/bin/tmux"
    else
        echo "NOT_FOUND"
    fi
')

if [ "$TMUX_CMD" = "NOT_FOUND" ]; then
    echo "Error: tmux was not found on the remote host '$REMOTE_CONN'."
    echo "Please ensure tmux is installed or available in ~/bin/tmux."
    exit 1
fi
echo "==> Using remote tmux: $TMUX_CMD"

# 5. Kill any existing tmux session with the same name to prevent overlaps
ssh "$REMOTE_CONN" "$TMUX_CMD kill-session -t llm-experiments 2>/dev/null || true"

# Shift arguments to get target script's arguments
shift 2
SCRIPT_ARGS="$@"

# 6. Start the script in a detached tmux session on the remote host
echo "==> Starting '$SCRIPT_FILE' inside remote tmux session 'llm-experiments'..."
ssh "$REMOTE_CONN" "$TMUX_CMD new-session -d -s llm-experiments 'cd ${REMOTE_PATH} && ./$SCRIPT_FILE $SCRIPT_ARGS'"

echo ""
echo "=========================================================================="
echo "SUCCESS: Experiment started in a remote tmux session on '$REMOTE_CONN'."
echo "Your laptop can now hibernate or lose connection safely."
echo "=========================================================================="
echo "To monitor or attach to the run, execute:"
echo "  ssh -t $REMOTE_CONN '$TMUX_CMD attach -t llm-experiments'"
echo "=========================================================================="
