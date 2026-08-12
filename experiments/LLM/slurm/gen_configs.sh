#!/bin/bash
# Emit one line per sweep configuration that still needs running, in the form
#
#   <log file>|<VFC_BACKENDS string>|<python arguments>
#
# A configuration whose log already holds a Result line is skipped, so this is
# resumable and can be regenerated at any point to see what is left. Logs that
# exist but are incomplete -- a job that was killed mid-run -- are removed
# rather than skipped, because the sweep scripts treat any existing log as done
# and would otherwise leave a silent hole.
#
# The array job in sweep_array.sbatch consumes this file by line number.
set -u

RESULTS_ROOT="${RESULTS_ROOT:-$HOME/perplexity_logs_epoch}"
CONTEXT="${CONTEXT:-256}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
T_MIN="${T_MIN:-4}"
T_MAX="${T_MAX:-14}"
SR_SEEDS="${SR_SEEDS:-1 2 3 4 5}"
RN_SEEDS="${RN_SEEDS:-1}"

PRECISIONS=$(seq "$T_MIN" "$T_MAX")
SUBLAYERS="attn_c_attn attn_c_proj mlp_c_fc mlp_c_proj"
GROUPS="attention mlp lm_head"
CUMULATIVE_BLOCKS="0 0-1 0-2 0-3 0-4 0-5"
CUMULATIVE_PRECISIONS="4 6"

backend_opts() {   # mode seed
    local opts="libinterflop_prism.so --seed=$2"
    [ "$1" = "rn" ] && opts="$opts --mode=rn"
    echo "$opts"
}

seeds_for() { [ "$1" = "rn" ] && echo "$RN_SEEDS" || echo "$SR_SEEDS"; }

emit() {           # log_file backend python_args...
    local log=$1; shift
    local backend=$1; shift
    if [ -f "$log" ]; then
        if grep -q "Result ->" "$log" 2>/dev/null; then return 0; fi
        # An incomplete log is either a dead run, which must be removed so the
        # configuration is redone, or a live one. Deleting a live one sends the
        # writer's output to an unlinked inode and loses it silently, so
        # anything touched in the last 30 minutes is left alone and skipped.
        if [ -n "$(find "$log" -mmin -30 2>/dev/null)" ]; then
            echo "skipping $log: incomplete but written recently, assuming a live run" >&2
            return 0
        fi
        rm -f "$log"
    fi
    mkdir -p "$(dirname "$log")"
    echo "${log}|${backend}|$*"
}

for mode in sr rn; do
    # Per-sublayer, all blocks. Carries the stagnation threshold.
    d="$RESULTS_ROOT/$CONTEXT/fine_${mode}_block_all"
    emit "$d/baseline_prec_24.log" "$(backend_opts "$mode" 1)" \
        test_fine_perplexity.py --layer attn_c_attn --precision 24 --block_idx all \
        --context_length "$CONTEXT" --max_tokens "$MAX_TOKENS"
    for layer in $SUBLAYERS; do
        for t in $PRECISIONS; do
            for seed in $(seeds_for "$mode"); do
                emit "$d/layer_${layer}_prec_${t}_seed_${seed}.log" \
                    "$(backend_opts "$mode" "$seed")" \
                    test_fine_perplexity.py --layer "$layer" --precision "$t" \
                    --block_idx all --context_length "$CONTEXT" --max_tokens "$MAX_TOKENS"
            done
        done
    done

    # Per-component.
    d="$RESULTS_ROOT/$CONTEXT/ppl_percomponents_${mode}"
    for group in $GROUPS; do
        for t in $PRECISIONS; do
            for seed in $(seeds_for "$mode"); do
                emit "$d/group_${group}_prec_${t}_seed_${seed}.log" \
                    "$(backend_opts "$mode" "$seed")" \
                    test_percomponent_perplexity.py --group "$group" --precision "$t" \
                    --context_length "$CONTEXT" --max_tokens "$MAX_TOKENS"
            done
        done
    done

    # Cumulative prefixes, mlp_c_proj only.
    d="$RESULTS_ROOT/$CONTEXT/cumulative_${mode}"
    for blocks in $CUMULATIVE_BLOCKS; do
        for t in $CUMULATIVE_PRECISIONS; do
            for seed in $(seeds_for "$mode"); do
                emit "$d/blocks_${blocks//-/_}_layer_mlp_c_proj_prec_${t}_seed_${seed}.log" \
                    "$(backend_opts "$mode" "$seed")" \
                    test_fine_perplexity.py --layer mlp_c_proj --precision "$t" \
                    --block_idx "$blocks" --context_length "$CONTEXT" --max_tokens "$MAX_TOKENS"
            done
        done
    done
done
