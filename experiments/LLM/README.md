# LLM Variable-Precision & Rounding Mode Experiment

This directory contains experimental suites measuring how reduced precision
with SR and RN affects Large Language Models (LLMs), using `distilgpt2` on
WikiText-2.

The experiments investigate precision scaling at several granularity levels:

1. **Monolithic / Global**: Uniform precision reduction across all transformer layers.

2. **Coarse Per-Component**: Isolating structural components (`attention`, `mlp`, `lm_head`).

3. **Fine-Grained Per-Layer & Blockwise**: Sensitivity analysis of individual sublayers (`attn_c_attn`, `attn_c_proj`, `mlp_c_fc`, `mlp_c_proj`) across specific transformer block indices (`0` through `5`).

4. **Mixed Precision / Rounding Recipes**: Evaluating custom heterogeneous bitwidth and rounding mode assignments per layer and block.

### 1. Fine-Grained Sublayer Perplexity Evaluation
```bash
./do_fine_perplexity.sh
python3 plot_fine_perplexity.py
```

### 2. Blockwise Sensitivity & Cumulative Error Propagation
```bash
./do_fine_blockwise_perplexity.sh
python3 plot_blockwise_perplexity.py

./do_fine_cumulative_perplexity.sh
python3 plot_cumulative_perplexity.py
```

### 3. Mixed Precision & Rounding Recipe Evaluation
```bash
./do_mixed_precision_eval.sh
```

---

## Active Files & Scripts Inventory

### Python Evaluation & Utility Modules
| File | Description |
|---|---|
| `eval_utils.py` | Core shared utility module. Provides dataset loading (`load_wikitext_slice`), perplexity evaluation (`evaluate_perplexity`), and Verificarlo dynamic hooks (`make_pre_hook`, `make_post_hook`). |
| `test_baseline.py` | Sanity check running standard 32-bit/24-bit inference on fixed prompts to inspect logits and execution latency. |
| `test_llm.py` | Evaluates greedy text generation across uniform global precision levels (`(2 4 6 8 10 12 24)` bits). |
| `test_perplexity.py` | Evaluates global monolithic perplexity on WikiText-2 across context lengths and precision levels. |
| `test_percomponent_perplexity.py` | Evaluates coarse group-level perplexity (`attention`, `mlp`, `lm_head`) while holding non-target groups at 24-bit FP32. |
| `test_fine_perplexity.py` | Evaluates fine-grained per-layer (`attn_c_attn`, `attn_c_proj`, `mlp_c_fc`, `mlp_c_proj`) and blockwise (`all`, `0-5`, `0-2`, `0,1`) perplexity. |
| `test_ideal_mixed.py` | Advanced mixed-precision evaluation script testing heterogeneous rounding configurations. |
| `test_mixed_generation.py` | Evaluates text generation when `mlp` layers are set to 6-bit while other layers remain at higher precision. |
| `omp_ext/` | C++/Python extension module (`setup.py`) for thread-local storage (`TLS`) broadcasting of target precision and rounding mode flags. |


### Shell Orchestrators
| File | Description |
|---|---|
| `do_baseline.sh` | Wrapper running `test_baseline.py`. |
| `do_generation.sh` | Runs `test_llm.py` across sweeping precision levels (`2 4 6 8 10 12 24` bits) concurrently. |
| `do_global_perplexity.sh` | Runs `test_perplexity.py` across sweeping precision levels. |
| `do_percomponent_perplexity.sh` | Runs `test_percomponent_perplexity.py` sweeping `attention`, `mlp`, `lm_head` across context lengths (`128, 256, 384, 512`). |
| `do_fine_perplexity.sh` | Runs `test_fine_perplexity.py` across isolated sublayers across all blocks (`all`). |
| `do_fine_blockwise_perplexity.sh` | Runs `test_fine_perplexity.py` sweeping sublayers independently across single block indices (`0` through `5`). |
| `do_fine_cumulative_perplexity.sh` | Runs `test_fine_perplexity.py` sweeping sublayers cumulatively from block `0` up to `0-5`. |
| `do_mixed_precision_eval.sh` | Runs `test_ideal_mixed.py` across targeted mixed precision and rounding mode configurations in parallel. |
| `do_mixed_generation.sh` | Runs `test_mixed_generation.py`. |
| `run_remote_tmux.sh` | Deploys and executes evaluation sweeps inside a `tmux` session on remote cluster nodes. |

### Plotting & Reporting Scripts
| File | Description |
|---|---|
| `plot_fine_perplexity.py` | Plots 2x2 subplots (`fine_perplexity_comparison.png`) comparing SR vs. RN per sublayer. |
| `plot_blockwise_perplexity.py` | Plots blockwise sensitivity across blocks 0-5 (`blockwise_perplexity_comparison.png`). |
| `plot_cumulative_perplexity.py` | Plots cumulative error propagation curves (`cumulative_perplexity_comparison.png`). |
| `table_perplexity_sr_vs_rn.py` | Utility script parsing coarse evaluation logs to generate clean Markdown tables comparing SR vs. RN across context lengths. |
