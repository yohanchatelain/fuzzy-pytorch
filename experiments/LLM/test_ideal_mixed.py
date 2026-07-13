import argparse
import time
import torch
import eval_utils

SR_MODE = 0
RN_MODE = 1

def uniform_block_rules(attn_fc_prec, attn_fc_mode, proj_prec, proj_mode):
    return lambda block_idx: {
        "attn_c_attn": (attn_fc_prec, attn_fc_mode),
        "attn_c_proj": (proj_prec, proj_mode),
        "mlp_c_fc": (attn_fc_prec, attn_fc_mode),
        "mlp_c_proj": (proj_prec, proj_mode),
    }

def proposed_recipe_rules(block_idx):
    if block_idx == 0:
        return {
            "attn_c_attn": (8, RN_MODE),
            "attn_c_proj": (8, RN_MODE),
            "mlp_c_fc": (8, RN_MODE),
            "mlp_c_proj": (8, RN_MODE),
        }
    else:
        return {
            "attn_c_attn": (7, RN_MODE),
            "attn_c_proj": (4, SR_MODE),
            "mlp_c_fc": (6, RN_MODE),
            "mlp_c_proj": (6, SR_MODE),
        }

CONFIG_RECIPES = {
    "mixed": {
        "desc": "6-bit SR on _proj, 6-bit RN on _attn & _fc, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(6, RN_MODE, 6, SR_MODE)
    },
    "pure_rn": {
        "desc": "6-bit RN on all layers, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(6, RN_MODE, 6, RN_MODE)
    },
    "pure_sr": {
        "desc": "6-bit SR on all layers, 10-bit SR on lm_head",
        "lm_head": (10, SR_MODE),
        "block_rules": uniform_block_rules(6, SR_MODE, 6, SR_MODE)
    },
    "pure_rn_hybrid": {
        "desc": "8-bit RN on _proj, 6-bit RN on _attn & _fc, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(6, RN_MODE, 8, RN_MODE)
    },
    "pure_rn_7": {
        "desc": "7-bit RN on all layers, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(7, RN_MODE, 7, RN_MODE)
    },
    "pure_sr_7": {
        "desc": "7-bit SR on all layers, 10-bit SR on lm_head",
        "lm_head": (10, SR_MODE),
        "block_rules": uniform_block_rules(7, SR_MODE, 7, SR_MODE)
    },
    "mixed_6_8": {
        "desc": "6-bit SR on _proj, 8-bit RN on _attn & _fc, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(8, RN_MODE, 6, SR_MODE)
    },
    "pure_rn_hybrid_8_7": {
        "desc": "8-bit RN on _proj, 7-bit RN on _attn & _fc, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(7, RN_MODE, 8, RN_MODE)
    },
    "mixed_7_7": {
        "desc": "7-bit SR on _proj, 7-bit RN on _attn & _fc, 10-bit RN on lm_head",
        "lm_head": (10, RN_MODE),
        "block_rules": uniform_block_rules(7, RN_MODE, 7, SR_MODE)
    },
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate mixed precision configurations.")
    parser.add_argument(
        "--config",
        type=str,
        default="mixed",
        choices=list(CONFIG_RECIPES.keys()),
        help="Select rounding mode configuration mix."
    )
    args = parser.parse_args()

    model, tokenizer, encodings = eval_utils.load_model_and_dataset("distilgpt2", fraction=100)
    context_length = 256
    seq_len = min(encodings.input_ids.size(1), context_length)

    recipe = CONFIG_RECIPES[args.config]
    print(f"\nEvaluating configuration: {args.config}")
    print(f"  Description: {recipe['desc']}")
    print(f"  All other operations: 24-bit (Baseline)")

    default_precision = 24
    default_mode = SR_MODE

    handles = []

    # 1. Instrument transformer blocks
    for idx, block in enumerate(model.transformer.h):
        layer_specs = recipe["block_rules"](idx)
        submodules = [
            ("attn_c_attn", block.attn.c_attn),
            ("attn_c_proj", block.attn.c_proj),
            ("mlp_c_fc", block.mlp.c_fc),
            ("mlp_c_proj", block.mlp.c_proj),
        ]
        for name, mod in submodules:
            prec, mode = layer_specs[name]
            handles.append(mod.register_forward_pre_hook(eval_utils.make_pre_hook(prec, mode)))
            handles.append(mod.register_forward_hook(eval_utils.make_post_hook(default_precision, default_mode)))

    # 2. Instrument lm_head
    lm_prec, lm_mode = recipe["lm_head"]
    handles.append(model.lm_head.register_forward_pre_hook(eval_utils.make_pre_hook(lm_prec, lm_mode)))
    handles.append(model.lm_head.register_forward_hook(eval_utils.make_post_hook(default_precision, default_mode)))

    print(f"\nInstrumented {len(handles) // 2} layers.")

    start_time = time.time()
    ppl = eval_utils.evaluate_perplexity(model, encodings, seq_len, stride=context_length, max_length=context_length, verbose=False)
    elapsed = time.time() - start_time

    # Clean up hooks
    for h in handles:
        h.remove()

    print(f"\nResult -> Configuration: {args.config} | Perplexity: {ppl:.2f} | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
