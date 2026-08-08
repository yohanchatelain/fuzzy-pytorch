"""Test LLM perplexity with DistilGPT-2 on WikiText-2, applying per-component lower precision via Verificarlo."""

import time
import torch
import argparse
import eval_utils

def main():
    parser = argparse.ArgumentParser(description="Per-component perplexity evaluation.")
    parser.add_argument("--group", type=str, required=True, choices=["attention", "mlp", "lm_head"])
    parser.add_argument("--precision", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256, help="Sliding window length (and stride) for evaluation")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Total tokens to score. Defaults to context_length, which scores a "
             "single window and gives a high-variance perplexity comparable only "
             "within a sweep level. Set larger (e.g. 1024) to average several windows.",
    )
    args = parser.parse_args()

    model, tokenizer, encodings = eval_utils.load_model_and_dataset("distilgpt2", fraction=100)

    max_length = args.context_length
    stride = args.context_length
    max_tokens = args.max_tokens if args.max_tokens is not None else args.context_length
    seq_len = min(encodings.input_ids.size(1), max_tokens)

    n_windows = max(1, -(-seq_len // stride))
    print(f"\nTotal tokens to evaluate: {seq_len} in {n_windows} window(s) of {max_length}")

    # Define coarse groups
    groups = {
        "attention": [block.attn for block in model.transformer.h],
        "mlp": [block.mlp for block in model.transformer.h],
        "lm_head": [model.lm_head]
    }

    group_name = args.group
    precision = args.precision
    default_precision = 24
    modules = groups[group_name]

    print(f"\n--- Evaluating Group: {group_name} | Precision: {precision} ---")

    handles = []
    for mod in modules:
        pre_handle = mod.register_forward_pre_hook(eval_utils.make_pre_hook(precision))
        post_handle = mod.register_forward_hook(eval_utils.make_post_hook(default_precision))
        handles.extend([pre_handle, post_handle])

    start_time = time.time()
    ppl = eval_utils.evaluate_perplexity(model, encodings, seq_len, stride, max_length, verbose=False)
    elapsed = time.time() - start_time

    # Clean up hooks
    for h in handles:
        h.remove()

    print(f"Result -> Group: {group_name} | Precision: {precision:2d} | Perplexity: {ppl:8.2f} | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
