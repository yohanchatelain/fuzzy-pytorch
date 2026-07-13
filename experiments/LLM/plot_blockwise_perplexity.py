import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse and plot blockwise perplexity results.")
    parser.add_argument(
        "--context_length", type=int, default=256,
        help="Context length to process"
    )
    parser.add_argument(
        "--output", type=str, default="blockwise_perplexity_comparison.png",
        help="Output plot file path"
    )
    args = parser.parse_args()

    data = []

    # Regex pattern:
    # Result -> Layer: attn_c_attn | Block: 0 | Precision: 4 | Perplexity:    138.55
    pattern = re.compile(
        r"Result -> Layer:\s+(?P<layer>\w+)\s+\|\s+Block:\s+(?P<block>\w+)\s+\|\s+Precision:\s+(?P<precision>\d+)\s+\|\s+Perplexity:\s+(?P<perplexity>[\d\.]+)"
    )

    # Search for files under perplexity_logs/{context_length}/fine_{sr/rn}_block_{0-5}/
    for mode in ["sr", "rn"]:
        for block_idx in range(6):
            log_dir = f"perplexity_logs/{args.context_length}/fine_{mode}_block_{block_idx}"
            if not os.path.exists(log_dir):
                continue
            log_files = glob.glob(os.path.join(log_dir, "*.log"))
            for log_file in log_files:
                with open(log_file, "r") as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            layer = match.group("layer")
                            block = int(match.group("block"))
                            precision = int(match.group("precision"))
                            perplexity = float(match.group("perplexity"))
                            data.append({
                                "Mode": mode.upper(),
                                "Block": block,
                                "Layer": layer,
                                "Precision": precision,
                                "Perplexity": perplexity
                            })
                            break

    if not data:
        print("No valid blockwise results found. Run do_fine_blockwise_perplexity.sh first.")
        return

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["Mode", "Block", "Layer", "Precision"])
    df = df.sort_values(by=["Layer", "Mode", "Precision", "Block"])

    # Print summary tables for 4-bit and 6-bit
    print(f"\n### Blockwise Perplexity Results (Context Length: {args.context_length})")
    for prec in [4, 6]:
        print(f"\n#### Precision: {prec}-bit")
        prec_df = df[df["Precision"] == prec]
        for layer in prec_df["Layer"].unique():
            layer_df = prec_df[prec_df["Layer"] == layer]
            # Pivot to show Blocks (0-5) as index, and Mode (SR, RN) as columns
            pivot_df = layer_df.pivot(index="Block", columns="Mode", values="Perplexity")
            print(f"\nLayer: `{layer}`")
            print(pivot_df.to_markdown())

    # Plotting: 2x2 grid of subplots grouped by layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    layers = ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]
    # Configurations to plot: (Mode, Precision, Style/Color)
    configs = [
        ("SR", 4, "o-", "#1f77b4"),  # SR 4-bit: Solid blue
        ("SR", 6, "o--", "#aec7e8"), # SR 6-bit: Dashed light blue
        ("RN", 4, "s-", "#ff7f0e"),  # RN 4-bit: Solid orange
        ("RN", 6, "s--", "#ffbb78"), # RN 6-bit: Dashed light orange
    ]

    for i, layer in enumerate(layers):
        ax = axes[i]
        layer_df = df[df["Layer"] == layer]
        
        for mode, prec, linestyle, color in configs:
            sub_df = layer_df[(layer_df["Mode"] == mode) & (layer_df["Precision"] == prec)].sort_values("Block")
            if not sub_df.empty:
                ax.plot(
                    sub_df["Block"],
                    sub_df["Perplexity"],
                    linestyle[1:], # Line style without marker
                    marker=linestyle[0],
                    color=color,
                    label=f"{mode} {prec}b",
                    linewidth=2
                )
        
        ax.set_title(f"Layer: {layer}", fontsize=14)
        ax.set_xlabel("Block Index (Depth)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(range(6))
        if i % 2 == 0:
            ax.set_ylabel("Perplexity (Log Scale)", fontsize=11)
        ax.legend(title="Config", fontsize=10, loc="upper right")

    plt.suptitle(
        f"GPT2 Blockwise Sensitivity: SR vs RN (Context: {args.context_length})", 
        fontsize=16, 
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\nPlot saved to {args.output}")

if __name__ == "__main__":
    main()
