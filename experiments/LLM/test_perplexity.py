"""Test LLM perplexity with DistilGPT-2 on WikiText-2.

This script evaluates the perplexity of the DistilGPT-2 model on a small subset
of the WikiText-2 dataset. It processes the text in chunks and computes the mean
negative log-likelihood, which is then exponentiated to get perplexity.
"""

import math
import time
import torch
import eval_utils

def main():
    model, tokenizer, encodings = eval_utils.load_model_and_dataset("distilgpt2", fraction=100)

    max_length = 128
    stride = 128
    max_tokens = 1024
    seq_len = min(encodings.input_ids.size(1), max_tokens)

    print(f"\nTotal tokens to evaluate: {seq_len}")
    print("Starting perplexity evaluation (Parallel chunks)...")

    start_time = time.time()
    ppl = eval_utils.evaluate_perplexity(model, encodings, seq_len, stride, max_length, verbose=True)
    elapsed = time.time() - start_time

    print("\n--- RESULTS ---")
    print(f"Final Perplexity: {ppl:.2f}")
    print(f"Time taken:       {elapsed:.2f} seconds")
    print(f"Forward passes:   {math.ceil(seq_len / max_length)}")

if __name__ == "__main__":
    main()
