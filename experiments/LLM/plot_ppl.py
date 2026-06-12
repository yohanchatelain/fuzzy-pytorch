"""Plot perplexity vs precision level.

Generates a plot from the perplexity CSV results, showing the degradation
of perplexity as precision decreases under SR.

Usage:
    python3 plot_ppl.py --input perplexity_results.csv --output perplexity_plot.png
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Perplexity vs Precision")
    parser.add_argument(
        "--input", type=str, default="perplexity_results.csv",
        help="Input CSV file with perplexity results",
    )
    parser.add_argument(
        "--output", type=str, default="perplexity_plot.png",
        help="Output plot file path",
    )
    parser.add_argument(
        "--baseline", type=float, default=83.39,
        help="Baseline perplexity for reference line",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Sort just in case
    df = df.sort_values(by="precision")

    # Data
    precs = df["precision"].values
    ppls = df["perplexity"].values

    plt.figure(figsize=(10, 6))

    # Plot the perplexity curve
    plt.plot(
        precs, ppls, marker="o", linestyle="-", linewidth=2,
        markersize=8, label="Quantized Perplexity"
    )

    # Plot the baseline
    plt.axhline(
        y=args.baseline, color="r", linestyle="--", linewidth=2,
        label=f"Baseline ({args.baseline})"
    )

    plt.yscale("log")
    plt.xlabel("Precision (bits)", fontsize=14)
    plt.ylabel("Perplexity (Log Scale)", fontsize=14)
    plt.title("Perplexity vs Precision", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # Adjust axes
    plt.xticks(precs)

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
