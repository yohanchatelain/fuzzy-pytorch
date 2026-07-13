import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse and plot fine-grained perplexity results.")
    parser.add_argument(
        "--context_length", type=int, default=256,
        help="Context length to process"
    )
    parser.add_argument(
        "--block_idx", type=str, default="all",
        help="Block index to process ('all' or specific number)"
    )
    parser.add_argument(
        "--output", type=str, default="fine_perplexity_comparison.png",
        help="Output plot file path"
    )
    args = parser.parse_args()

    # Search for logs in the respective SR and RN directories
    sr_dir = f"perplexity_logs/{args.context_length}/fine_sr_block_{args.block_idx}"
    rn_dir = f"perplexity_logs/{args.context_length}/fine_rn_block_{args.block_idx}"

    data = []

    # Regex pattern:
    # Result -> Layer: attn_c_attn | Block: all | Precision: 24 | Perplexity:    43.32 | Time: 900.22s
    pattern = re.compile(
        r"Result -> Layer:\s+(?P<layer>\w+)\s+\|\s+Block:\s+(?P<block>\w+)\s+\|\s+Precision:\s+(?P<precision>\d+)\s+\|\s+Perplexity:\s+(?P<perplexity>[\d\.]+)"
    )

    def parse_dir(log_dir, mode):
        if not os.path.exists(log_dir):
            print(f"Directory not found: {log_dir}")
            return
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        for log_file in log_files:
            with open(log_file, "r") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        layer = match.group("layer")
                        block = match.group("block")
                        precision = int(match.group("precision"))
                        perplexity = float(match.group("perplexity"))
                        # Baseline 24-bit is shared across all layers
                        if precision == 24:
                            # Add baseline for all target layers
                            for target in ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]:
                                data.append({
                                    "Mode": mode,
                                    "Layer": target,
                                    "Block": block,
                                    "Precision": precision,
                                    "Perplexity": perplexity
                                })
                        else:
                            data.append({
                                "Mode": mode,
                                "Layer": layer,
                                "Block": block,
                                "Precision": precision,
                                "Perplexity": perplexity
                            })
                        break

    parse_dir(sr_dir, "SR")
    parse_dir(rn_dir, "RN")

    if not data:
        print("No valid results found. Run do_fine_perplexity.sh first.")
        return

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["Mode", "Layer", "Block", "Precision"])
    df = df.sort_values(by=["Mode", "Layer", "Precision"])

    # Print markdown tables
    print(f"\n### Fine-Grained Perplexity Results (Context Length: {args.context_length}, Block: {args.block_idx})")
    for mode in df["Mode"].unique():
        print(f"\n#### Rounding Mode: {mode}")
        mode_df = df[df["Mode"] == mode]
        pivot_df = mode_df.pivot(index="Precision", columns="Layer", values="Perplexity")
        # Sort index descending to show higher precision at top
        pivot_df = pivot_df.sort_index(ascending=False)
        print(pivot_df.to_markdown())

    # Plot results: 2x2 subplots grouped by layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    
    layers = ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]
    modes = ["SR", "RN"]
    mode_colors = {
        "SR": "#1f77b4",  # Blue
        "RN": "#ff7f0e"   # Orange
    }

    for i, layer in enumerate(layers):
        ax = axes[i]
        
        for mode in modes:
            layer_mode_df = df[(df["Layer"] == layer) & (df["Mode"] == mode)].sort_values("Precision")
            if not layer_mode_df.empty:
                ax.plot(
                    layer_mode_df["Precision"], 
                    layer_mode_df["Perplexity"], 
                    marker='o', 
                    color=mode_colors[mode], 
                    label=mode, 
                    linewidth=2
                )
        
        ax.set_title(f"Layer: {layer}", fontsize=14)
        ax.set_xlabel("Precision (bits)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks([4, 6, 8, 24])
        if i % 2 == 0:
            ax.set_ylabel("Perplexity (Log Scale)", fontsize=11)
        ax.legend(title="Rounding Mode", fontsize=10, loc="upper right")

    plt.suptitle(
        f"GPT2 Per-layer Perplexity: SR vs RN (Context: {args.context_length}, Block: {args.block_idx})", 
        fontsize=16, 
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\nPlot saved to {args.output}")

if __name__ == "__main__":
    main()
