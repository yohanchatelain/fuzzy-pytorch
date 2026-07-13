import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)

    model_id = "distilgpt2"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    prompts = [
        "The days of the week are Monday, Tuesday, Wednesday, Thursday,",
        "Alphabet: A, B, C, D, E, F,",
        "The months of the year are January, February, March, April,"
    ]

    print("\nStarting baseline evaluation (Standard Precision)...")

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

if __name__ == "__main__":
    main()
