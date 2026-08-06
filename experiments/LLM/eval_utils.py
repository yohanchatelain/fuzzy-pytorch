"""Utility functions for LLM evaluations (dataset loading, perplexity computation, and precision hooks)."""

import math
import os
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WIKITEXT_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
LOCAL_CACHE_FILE = "wikitext-2-test.txt"

# Interflop/OpenMP extensions loading for dynamic precision hooks.
#
# Without omp_ext the per-module hooks below cannot change anything, and the run
# silently degrades to full precision while still producing a plausible-looking
# perplexity. That failure is indistinguishable from a real result in the logs,
# so it is fatal rather than a warning.
try:
    import omp_ext
except ImportError as exc:
    raise ImportError(
        "omp_ext is required: without it the per-module precision hooks are "
        "no-ops and every sweep silently runs at full precision. Build it with "
        "`cd omp_ext && python3 setup.py build_ext --inplace` and put it on "
        "PYTHONPATH."
    ) from exc

print("Successfully loaded omp_ext for PRISM TLS broadcasting")

def set_precision(precision, is_binary64=False):
    omp_ext.set_precision(precision)
    applied = omp_ext.get_precision()
    if applied != precision:
        raise RuntimeError(
            f"PRISM precision did not take effect: asked for {precision}, "
            f"thread reports {applied}"
        )

def set_rounding_mode(mode):
    omp_ext.set_rounding_mode(mode)
    applied = omp_ext.get_rounding_mode()
    if applied != mode:
        raise RuntimeError(
            f"PRISM rounding mode did not take effect: asked for {mode}, "
            f"thread reports {applied}"
        )

def make_pre_hook(target_precision, target_mode=None):
    """Creates a pre-hook to drop precision and optionally change rounding mode (0:SR, 1:RN) before the module runs."""
    def hook(module, input):
        set_precision(target_precision, is_binary64=False)
        if target_mode is not None:
            set_rounding_mode(target_mode)
    return hook

def make_post_hook(default_precision, default_mode=None):
    """Creates a post-hook to restore precision and optionally rounding mode (0:SR, 1:RN) after the module runs."""
    def hook(module, input, output):
        set_precision(default_precision, is_binary64=False)
        if default_mode is not None:
            set_rounding_mode(default_mode)
    return hook

def load_wikitext_slice(tokenizer, fraction=100):
    """Loads a slice (default 1%) of the WikiText-2 dataset from a local cache or downloads it."""
    # Check local cache first
    if os.path.exists(LOCAL_CACHE_FILE):
        print(f"Loading dataset slice from local cache '{LOCAL_CACHE_FILE}'...")
        with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Downloading dataset slice from remote URL...")
        try:
            text = urllib.request.urlopen(WIKITEXT_URL).read().decode("utf-8")
            # Write to local cache for future runs
            with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e

    lines = text.split("\n")
    # Take the specified fraction of the lines (default 1%)
    slice_lines = lines[:max(1, len(lines) // fraction)]
    encodings = tokenizer("\n\n".join(slice_lines), return_tensors="pt")
    return encodings

def evaluate_perplexity(model, encodings, seq_len, stride, max_length, verbose=False):
    """Evaluates perplexity of the model over the given encodings in chunks."""
    nlls = []

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()

        if verbose:
            print(f"Evaluating chunk from {begin_loc} to {end_loc}...", flush=True)

        with torch.no_grad():
            if verbose:
                print("Running forward pass...", flush=True)
            outputs = model(input_ids, labels=target_ids)
            if verbose:
                print("Forward pass complete.", flush=True)

            log_likelihood = outputs.loss * (trg_len - 1)
            nlls.append(log_likelihood)

    total_predicted_tokens = seq_len - len(nlls)
    ppl = torch.exp(torch.stack(nlls).sum() / total_predicted_tokens)
    return ppl.item()

def load_model_and_dataset(model_id="distilgpt2", fraction=100, num_threads=1):
    """Loads the tokenizer, causal LM model, and WikiText-2 dataset slice."""
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    encodings = load_wikitext_slice(tokenizer, fraction=fraction)
    return model, tokenizer, encodings
