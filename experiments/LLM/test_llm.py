"""Test LLM generation with DistilGPT-2.

This script loads the DistilGPT-2 model and generates a fixed number of tokens
from a prompt. It locks the threads and seeds the random number generator to
isolate numerical noise (like Stochastic Rounding) from sampling noise.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Lock threads to prevent standard non-determinism
    torch.set_num_threads(1)
    torch.manual_seed(42)

    model_id = "distilgpt2"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\nStarting generation (this may take a moment on CPU...)")
    start_time = time.time()

    # Generate 20 tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # Greedy decoding to isolate math noise from sampling noise
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start_time

    print("\n--- GENERATED TEXT ---")
    print(text)
    print("----------------------")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
