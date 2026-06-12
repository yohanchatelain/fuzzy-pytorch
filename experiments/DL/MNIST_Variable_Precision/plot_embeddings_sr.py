"""Visualize MNIST embedding variability under Stochastic Rounding.

Shows the std-dev as a power of 2 (log2(std)) heatmap figures,
showing how SR affects the latent representations at each layer.

For conv/pool layers: shows 2D heatmap of mean log2(std) across
    channels (spatial map of numerical stability).
For fc layers: shows 1D heatmap of log2(std) per neuron.

Usage:
    python3 plot_embeddings_sr.py \\
        --results-dir ./mnist_sr_experiment/results \\
        --precisions 24 4 \\
        --sample-idx 0 \\
        --output-dir ./mnist_sr_experiment/results/figures
"""

import argparse
import os
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import load_runs


def find_digit_sample(runs, digit=7):
    """Find the index of the first sample classified as `digit` in run 1."""
    targets = runs[0]["target"]
    indices = (targets == digit).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        print(f"Warning: digit {digit} not found in batch, using sample 0")
        return 0
    return indices[0].item()


def compute_log2_std(runs, layer, sample_idx):
    """Compute std-dev as a power of 2 for a specific sample across runs.

    Returns the log2(std) array for sample `sample_idx`.
    For conv layers: shape = (C, H, W)
    For fc layers: shape = (D,)
    """
    # Stack the layer activations across runs for the given sample
    tensors = []
    for run in runs:
        if layer in run:
            tensors.append(run[layer][sample_idx])

    if not tensors:
        return None

    # Stack: (N_runs, ...) where ... is the layer shape
    stacked = torch.stack(tensors).float()

    # Compute std across runs
    std = torch.std(stacked, dim=0)
    # Replace 0 std with a small value to avoid log2(0)
    std = torch.clamp(std, min=1e-10)
    std_log2 = torch.log2(std)

    return std_log2


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MNIST embedding variability under SR"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to results directory",
    )
    parser.add_argument(
        "--precisions", type=int, nargs="+", default=None,
        help="Precisions to plot (default: auto-detect)",
    )
    parser.add_argument(
        "--digit", type=int, default=7,
        help="Which digit to analyze (default: 7)",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=None,
        help="Direct sample index (overrides --digit)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save figures",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Discover precision levels
    if args.precisions:
        precisions = sorted(args.precisions, reverse=True)
    else:
        prec_dirs = sorted(glob.glob(os.path.join(results_dir, "precision_*")))
        precisions = sorted(
            [int(os.path.basename(d).split("_")[1]) for d in prec_dirs],
            reverse=True,
        )

    print(f"Precisions: {precisions}")

    layers = ["conv1", "conv2", "fc1", "fc2"]
    layer_labels = {
        "conv1": "Conv1 (32×26×26)",
        "conv2": "Conv2 (64×24×24)",
        "fc1": "FC1 (128-dim)",
        "fc2": "FC2 (10-dim logits)",
    }

    # ================================================================
    # Figure: Per-precision, per-layer significant digits heatmaps
    # One figure per precision level, with subplots for each layer
    # ================================================================
    for prec in precisions:
        prec_dir = os.path.join(results_dir, f"precision_{prec}")
        runs = load_runs(prec_dir)
        n_runs = len(runs)
        print(f"\n--- Precision {prec} bits ({n_runs} runs) ---")

        # Find the sample index for the requested digit
        if args.sample_idx is not None:
            sample_idx = args.sample_idx
            digit_label = f"sample {sample_idx}"
        else:
            sample_idx = find_digit_sample(runs, args.digit)
            actual_label = runs[0]["target"][sample_idx].item()
            digit_label = f"digit {actual_label} (idx={sample_idx})"

        print(f"  Analyzing {digit_label}")

        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5))
        if len(layers) == 1:
            axes = [axes]

        for i, layer in enumerate(layers):
            ax = axes[i]
            sig = compute_log2_std(runs, layer, sample_idx)

            if sig is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(layer_labels.get(layer, layer))
                continue

            sig_np = sig.numpy() if isinstance(sig, torch.Tensor) else sig

            if "fc" in layer:
                # FC layers: 1D heatmap (expand to 2D for visualization)
                data = np.expand_dims(sig_np, axis=0)
                sns.heatmap(
                    data, ax=ax, cmap="RdYlGn_r", vmin=-15, vmax=5, center=0,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={"label": "log2(std)"},
                )
                ax.set_title(layer_labels.get(layer, layer), fontsize=13)
            else:
                # Conv layers: 2D spatial heatmap
                # sig_np shape: (C, H, W) — average across channels
                data = np.mean(sig_np, axis=0, dtype=np.float32)
                sns.heatmap(
                    data, ax=ax, cmap="RdYlGn_r", vmin=-15, vmax=5, center=0,
                    xticklabels=False, yticklabels=False, cbar=False,
                )
                ax.set_title(layer_labels.get(layer, layer), fontsize=13)

            mean_sig = float(np.nanmean(sig_np))
            ax.set_xlabel(f"mean={mean_sig:.2f} log2(std)", fontsize=10)
            print(f"  {layer}: mean log2(std) = {mean_sig:.2f}")

        plt.suptitle(
            f"Log2(std) of Latent Representations — "
            f"PRISM SR, precision={prec} bits\n"
            f"{digit_label}, {n_runs} stochastic runs",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()

        fname = f"embeddings_sr_prec{prec}"
        fig.savefig(
            os.path.join(output_dir, f"{fname}.png"),
            dpi=150, bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(output_dir, f"{fname}.pdf"),
            bbox_inches="tight",
        )
        print(f"  Saved: {fname}.png/pdf")
        plt.close()

    # ================================================================
    # Combined figure: all precisions side by side for each layer
    # ================================================================
    print("\n--- Generating combined comparison figure ---")

    # Load all runs once
    all_runs = {}
    for prec in precisions:
        prec_dir = os.path.join(results_dir, f"precision_{prec}")
        all_runs[prec] = load_runs(prec_dir)

    # Use same sample across all precisions
    if args.sample_idx is not None:
        sample_idx = args.sample_idx
    else:
        sample_idx = find_digit_sample(all_runs[precisions[0]], args.digit)

    actual_label = all_runs[precisions[0]][0]["target"][sample_idx].item()

    fig, axes = plt.subplots(
        len(layers), len(precisions),
        figsize=(5 * len(precisions), 4 * len(layers)),
    )

    if len(precisions) == 1:
        axes = axes.reshape(-1, 1)
    if len(layers) == 1:
        axes = axes.reshape(1, -1)

    for row, layer in enumerate(layers):
        for col, prec in enumerate(precisions):
            ax = axes[row, col]
            sig = compute_log2_std(all_runs[prec], layer, sample_idx)

            if sig is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            sig_np = sig.numpy() if isinstance(sig, torch.Tensor) else sig

            if "fc" in layer:
                data = np.expand_dims(sig_np, axis=0)
                sns.heatmap(
                    data, ax=ax, cmap="RdYlGn_r", vmin=-15, vmax=5, center=0,
                    xticklabels=False, yticklabels=False,
                    cbar=(col == len(precisions) - 1),
                    cbar_kws={"label": "log2(std)"}
                    if col == len(precisions) - 1 else {},
                )
            else:
                data = np.mean(sig_np, axis=0, dtype=np.float32)
                sns.heatmap(
                    data, ax=ax, cmap="RdYlGn_r", vmin=-15, vmax=5, center=0,
                    xticklabels=False, yticklabels=False,
                    cbar=(col == len(precisions) - 1),
                    cbar_kws={"label": "log2(std)"}
                    if col == len(precisions) - 1 else {},
                )

            mean_sig = float(np.nanmean(sig_np))

            if row == 0:
                ax.set_title(f"Precision = {prec} bits", fontsize=13)
            if col == 0:
                ax.set_ylabel(layer_labels.get(layer, layer), fontsize=12)

            ax.set_xlabel(f"μ={mean_sig:.2f}", fontsize=9)

    plt.suptitle(
        f"Log2(std) across Layers and Precisions — "
        f"digit {actual_label}\n"
        f"PRISM Stochastic Rounding",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()

    fname = "embeddings_sr_comparison"
    fig.savefig(
        os.path.join(output_dir, f"{fname}.png"),
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(output_dir, f"{fname}.pdf"),
        bbox_inches="tight",
    )
    print(f"Saved: {fname}.png/pdf")
    plt.close()

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
