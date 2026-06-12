"""Plot mean activation standard deviation vs. precision level.

Shows how reducing mantissa precision increases numerical variability
in each layer's activations under PRISM Stochastic Rounding.

Usage:
    python3 plot_stdev_vs_precision.py \\
        --results-dir ./mnist_sr_experiment/results \\
        --output-dir ./mnist_sr_experiment/results/figures
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_runs


def get_mean_std_for_layer(runs, layer):
    """Compute standard deviation across runs, then mean over all other dims."""
    tensors = []
    for run in runs:
        if layer in run:
            tensors.append(run[layer])
    if not tensors:
        return None
    stacked = torch.stack(tensors).float()  # (N_runs, Batch, ...)
    std = torch.std(stacked, dim=0)  # (Batch, ...)
    # Mean over batch and spatial/channel dimensions
    return torch.mean(std).item()


def main():
    parser = argparse.ArgumentParser(
        description="Plot activation std vs. precision level"
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="./mnist_sr_experiment/results",
        help="Path to results directory (default: ./mnist_sr_experiment/results)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save figures (default: <results-dir>/figures)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    prec_dirs = sorted(glob.glob(os.path.join(results_dir, "precision_*")))
    precisions = sorted(
        [int(os.path.basename(d).split("_")[1]) for d in prec_dirs]
    )

    layers = ["conv1", "conv2", "fc1", "fc2"]

    data = {layer: [] for layer in layers}
    valid_precisions = []
    accuracies = []

    print(f"Discovered precisions: {precisions}")
    for prec in precisions:
        prec_dir = os.path.join(results_dir, f"precision_{prec}")
        runs = load_runs(prec_dir)
        if len(runs) < 2:
            print(f"Skipping precision {prec}, not enough runs ({len(runs)})")
            continue

        valid_precisions.append(prec)
        print(f"Processing precision {prec}...")
        for layer in layers:
            mean_std = get_mean_std_for_layer(runs, layer)
            data[layer].append(mean_std)

        accs = []
        for run in runs:
            pred = run["pred"]
            target = run["target"]
            correct = pred.eq(target.view_as(pred)).sum().item()
            accs.append(correct / target.size(0))
        accuracies.append(np.mean(accs) * 100.0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D"]

    for layer, marker in zip(layers, markers):
        ax1.plot(
            valid_precisions, data[layer],
            marker=marker, linestyle="-", linewidth=2, label=layer,
        )

    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("Precision (bits)", fontsize=14)
    ax1.set_ylabel("Mean Standard Deviation (Log Scale)", fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_xticks(valid_precisions)
    
    y_ticks = [2**-24, 2**-16, 2**-12, 2**-8, 2**-4, 2**-2, 2**-1, 2**0, 2**1, 2**2]
    y_ticklabels = ["$2^{-24}$", "$2^{-16}$", "$2^{-12}$", "$2^{-8}$", "$2^{-4}$", "$2^{-2}$", "$2^{-1}$", "$2^{0}$", "$2^{1}$", "$2^{2}$"]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticklabels)

    ax2 = ax1.twinx()
    ax2.plot(
        valid_precisions, accuracies,
        marker="*", linestyle="--", linewidth=2, color="black", label="Accuracy"
    )
    ax2.set_ylabel("Accuracy (%)", fontsize=14, color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0, 105)

    plt.title("Activation Standard Deviation and Accuracy vs Precision (MNIST)", fontsize=16)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", fontsize=12)

    fname = "stdev_vs_precision"
    plt.savefig(
        os.path.join(output_dir, f"{fname}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, f"{fname}.pdf"),
        bbox_inches="tight",
    )
    print(f"Plots saved to {output_dir}/{fname}.[png|pdf]")


if __name__ == "__main__":
    main()
