# MNIST Variable-Precision Stochastic Rounding Experiment

This experiment measures how **reduced floating-point precision** under **PRISM Stochastic Rounding (SR)** affects the latent representations of a simple MNIST CNN.

By sweeping mantissa precision from 1 to 24 bits (full IEEE binary32) and running inference multiple times under SR, we capture per-layer activation variability and visualize it as log₂(std) heatmaps.

## Files

| File | Description |
|------|-------------|
| `common.py` | Shared `Net` model architecture and `load_runs()` I/O utility |
| `mnist_train.py` | Train the CNN baseline under IEEE mode (deterministic) |
| `mnist_latent_experiment.py` | Run inference and capture layer activations (designed for SR containers) |
| `run_latent_sr_experiment.sh` | Orchestrator: trains model, sweeps precisions via Podman/PRISM |
| `plot_embeddings_sr.py` | Generate per-layer log₂(std) heatmaps for each precision level |
| `plot_stdev_vs_precision.py` | Plot mean activation std vs. precision (line plot across layers) |

## Prerequisites

- Container image: `big-data-lab-team/fuzzy-pytorch:sr` (built from `../../containers/`)
- Python 3 with PyTorch, torchvision, matplotlib, seaborn, numpy

## Reproduction

### 1. Run the experiment

```bash
cd experiments/DL/MNIST_Variable_Precision/
./run_latent_sr_experiment.sh
```

This will:
1. Train a baseline MNIST model (IEEE mode, 5 epochs)
2. Run inference 30× under SR at precisions: 1, 2, 4, 8, 12, 16, 24 bits
3. Save activations to `mnist_sr_experiment/results/precision_<N>/`

### 2. Generate plots

```bash
# Heatmaps of log2(std) per layer and precision
python3 plot_embeddings_sr.py \
    --results-dir ./mnist_sr_experiment/results

# Std vs. precision line plot
python3 plot_stdev_vs_precision.py \
    --results-dir ./mnist_sr_experiment/results
```

## Outputs

All generated data lives under `mnist_sr_experiment/` (git-ignored):

```
mnist_sr_experiment/
├── data/              # Downloaded MNIST dataset
├── model/             # Trained model weights (mnist_cnn.pt)
└── results/
    ├── precision_1/   # activations_run_*.pkl
    ├── precision_2/
    ├── ...
    ├── precision_24/
    └── figures/       # Generated plots (.png, .pdf)
```
