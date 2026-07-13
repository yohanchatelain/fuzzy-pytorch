import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse and plot cumulative fine-grained perplexity results.")
    parser.add_argument(
        "--context_length", type=int, default=256,
        help="Context length to process"
    )
    parser.add_argument(
        "--output", type=str, default="cumulative_perplexity_comparison.png",
        help="Output plot file path"
    )
    args = parser.parse_args()

    data = []

    # Regex pattern:
    # Result -> Layer: attn_c_attn | Block: 0-2 | Precision: 4 | Perplexity:    138.55
    pattern = re.compile(
        r"Result -> Layer:\s+(?P<layer>\w+)\s+\|\s+Block:\s+(?P<block>[\w,-]+)\s+\|\s+Precision:\s+(?P<precision>\d+)\s+\|\s+Perplexity:\s+(?P<perplexity>[\d\.]+)"
    )

    block_configs = ["0", "0-1", "0-2", "0-3", "0-4", "0-5"]
    depth_mapping = {
        "0": 1,
        "0-1": 2,
        "0-2": 3,
        "0-3": 4,
        "0-4": 5,
        "0-5": 6
    }

    # Search for files under perplexity_logs/{context_length}/fine_{sr/rn}_cumul_{conf}/
    for mode in ["sr", "rn"]:
        for conf in block_configs:
            log_dir = f"perplexity_logs/{args.context_length}/fine_{mode}_cumul_{conf}"
            if not os.path.exists(log_dir):
                continue
            log_files = glob.glob(os.path.join(log_dir, "*.log"))
            for log_file in log_files:
                with open(log_file, "r") as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            layer = match.group("layer")
                            block_str = match.group("block")
                            precision = int(match.group("precision"))
                            perplexity = float(match.group("perplexity"))
                            depth = depth_mapping.get(block_str, len(block_str.split("-")))
                            data.append({
                                "Mode": mode.upper(),
                                "Config": block_str,
                                "Depth": depth,
                                "Layer": layer,
                                "Precision": precision,
                                "Perplexity": perplexity
                            })
                            break

    if not data:
        print("No valid cumulative results found. Run do_fine_cumulative_perplexity.sh first.")
        return

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["Mode", "Config", "Layer", "Precision"])
    df = df.sort_values(by=["Layer", "Mode", "Precision", "Depth"])

    # Print summary tables for 4-bit and 6-bit
    print(f"\n### Cumulative Perplexity Results (Context Length: {args.context_length})")
    for prec in [4, 6]:
        print(f"\n#### Precision: {prec}-bit")
        prec_df = df[df["Precision"] == prec]
        for layer in prec_df["Layer"].unique():
            layer_df = prec_df[prec_df["Layer"] == layer]
            # Pivot to show Config as index, and Mode (SR, RN) as columns
            pivot_df = layer_df.pivot(index="Config", columns="Mode", values="Perplexity")
            # Reorder rows according to block_configs
            valid_confs = [c for c in block_configs if c in pivot_df.index]
            pivot_df = pivot_df.reindex(valid_confs)
            print(f"\nLayer: `{layer}`")
            print(pivot_df.to_markdown())

    # Plotting: 2x2 grid of subplots grouped by layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    layers = ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]
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
            sub_df = layer_df[(layer_df["Mode"] == mode) & (layer_df["Precision"] == prec)].sort_values("Depth")
            if not sub_df.empty:
                ax.plot(
                    sub_df["Depth"],
                    sub_df["Perplexity"],
                    linestyle[1:], # Line style without marker
                    marker=linestyle[0],
                    color=color,
                    label=f"{mode} {prec}b",
                    linewidth=2
                )
        
        ax.set_title(f"Layer: {layer}", fontsize=14)
        ax.set_xlabel("Cumulative Quantized Blocks (Bottom-Up Depth)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(["0", "0-1", "0-2", "0-3", "0-4", "0-5"])
        if i % 2 == 0:
            ax.set_ylabel("Perplexity (Log Scale)", fontsize=11)
        ax.legend(title="Config", fontsize=10, loc="upper left")

    plt.suptitle(
        f"GPT2 Cumulative Sensitivity: SR vs RN (Context: {args.context_length})", 
        fontsize=16, 
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\nPlot saved to {args.output}")

if __name__ == "__main__":
    main()
