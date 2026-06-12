"""Test LLM perplexity with DistilGPT-2 on WikiText-2.

This script evaluates the perplexity of the DistilGPT-2 model on a small subset
of the WikiText-2 dataset. It processes the text in chunks and computes the mean
negative log-likelihood, which is then exponentiated to get perplexity.
"""

import math
import time
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    torch.set_num_threads(1)

    model_id = "distilgpt2"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Load exactly 1% of the WikiText-2 test set (~2,800 tokens)
    print("Downloading dataset slice...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    lines = text.split("\n")
    
    # Get 1% of the lines
    slice_lines = lines[:max(1, len(lines) // 100)]
    encodings = tokenizer("\n\n".join(slice_lines), return_tensors="pt")

    max_length = 128
    stride = 128
    max_tokens = 1024
    seq_len = min(encodings.input_ids.size(1), max_tokens)

    print(f"\nTotal tokens to evaluate: {seq_len}")
    print("Starting perplexity evaluation (Parallel chunks)...")

    nlls = []
    start_time = time.time()

    # Evaluate in chunks of 128 tokens
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        
        print(f"Evaluating chunk from {begin_loc} to {end_loc}...", flush=True)
        with torch.no_grad():
            # A single forward pass computes the loss for all tokens in the chunk
            print("Running forward pass...", flush=True)
            outputs = model(input_ids, labels=target_ids)
            print("Forward pass complete.", flush=True)
            
            # The loss is already the mean negative log-likelihood for the chunk
            log_likelihood = outputs.loss * (trg_len - 1)
            nlls.append(log_likelihood)

    total_predicted_tokens = seq_len - len(nlls)
    ppl = torch.exp(torch.stack(nlls).sum() / total_predicted_tokens)
    elapsed = time.time() - start_time

    print("\n--- RESULTS ---")
    print(f"Final Perplexity: {ppl.item():.2f}")
    print(f"Time taken:       {elapsed:.2f} seconds")
    print(f"Forward passes:   {math.ceil(seq_len / max_length)}")

if __name__ == "__main__":
    main()
