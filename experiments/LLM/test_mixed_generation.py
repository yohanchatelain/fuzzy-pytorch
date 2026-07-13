"""Test LLM generation with mixed precision.
MLP at 6 bits, other components at 10 bits.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import eval_utils

def main():
    # Lock threads to prevent standard non-determinism
    torch.set_num_threads(1)
    torch.manual_seed(42)

    # We want everything else at 10 bits, so we set global precision to 10 bits.
    if eval_utils.HAS_INTERFLOP:
        eval_utils.set_precision(24)

    model_id = "distilgpt2"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Setup hooks for MLP -> 6 bits
    mlp_precision = 6
    default_precision = 24
    modules = [block.mlp for block in model.transformer.h]

    handles = []
    for mod in modules:
        pre_handle = mod.register_forward_pre_hook(eval_utils.make_pre_hook(mlp_precision))
        post_handle = mod.register_forward_hook(eval_utils.make_post_hook(default_precision))
        handles.extend([pre_handle, post_handle])

    prompts = [
        "The days of the week are Monday, Tuesday, Wednesday, Thursday,",
        "Alphabet: A, B, C, D, E, F,",
        "The months of the year are January, February, March, April,"
    ]

    print(f"\nStarting evaluation (MLP at {mlp_precision} bits, others at {default_precision} bits)...")

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")

        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.time() - start_time

        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)

        print(f"\n--- PROMPT {i+1} ---")
        print(prompt)
        print("Top 5 next token predictions:")
        for prob, idx in zip(top_probs, top_indices):
            token_str = tokenizer.decode([idx])
            print(f"  {token_str!r}: {prob.item():.4f}")
        print(f"(Forward pass took {elapsed:.2f} seconds)")
        print("----------------------")

    # Clean up hooks
    for h in handles:
        h.remove()

    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
