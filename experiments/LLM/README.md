# LLM Variable-Precision Stochastic Rounding Experiment

This experiment measures how **reduced floating-point precision** under **PRISM Stochastic Rounding (SR)** affects Large Language Models (LLMs). We use `distilgpt2` to test text generation divergence and perplexity degradation.

By sweeping mantissa precision (`(2 4 6 8 10 12 24)` bits), we can see the tipping point where language capabilities break down under numerical noise.

## Files

| File | Description |
|------|-------------|
| `test_llm.py` | Generates text given a prompt using `distilgpt2`. Greedy decoding is used to isolate numerical noise. |
| `test_perplexity.py` | Evaluates model perplexity on a subset of WikiText-2. |
| `do_generation.sh` | Orchestrator script that runs `test_llm.py` across different precisions. |
| `do_perplexity.sh` | Orchestrator script that runs `test_perplexity.py` across different precisions. |
| `plot_ppl.py` | Generates a line plot of perplexity versus precision. |

## Prerequisites

- Container image: `big-data-lab-team/fuzzy-pytorch:sr` (built from `../../containers/`)
- Python 3 with `transformers`, `torch`, `pandas`, `matplotlib` (for plotting)

## Reproduction

### 1. Test Text Generation

```bash
cd experiments/LLM/
./do_generation.sh
```
This runs the text generation script across various precisions concurrently and produces `llm_generations.txt`.

### 2. Test Perplexity

```bash
./do_perplexity.sh
```
This computes perplexity on WikiText-2 chunks for each precision level, logging results to `perplexity_results.csv`.

### 3. Generate Perplexity Plot

```bash
python3 plot_ppl.py --input perplexity_results.csv --output perplexity_plot.png
```

## Outputs

Generated artifacts (which are git-ignored) include:
- `hf_cache/`: Local cache for HuggingFace models.
- `ppl_logs/` & `run_logs/`: Raw execution logs.
- `llm_generations.txt`: Collated text generation results.
- `perplexity_results.csv`: Aggregated perplexity metrics.
- `perplexity_plot.png`: Plot of the perplexity values.
