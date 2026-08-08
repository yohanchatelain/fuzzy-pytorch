#!/usr/bin/env python3
"""Emit the paper's four sweep figures as PDFs.

    python3 plot_paper_figures.py [LOG_ROOT] [OUT_DIR]

SR and RN are distinguished by colour *and* by linestyle and marker, so the
figures survive greyscale printing and colour-vision deficiency without relying
on hue. The two hues are the first two categorical slots of a palette validated
for CVD separation; do not substitute arbitrary colours.

Perplexity spans nine orders of magnitude across the sweeps, so every axis is
logarithmic. That is a real property of the data -- reducing the language-model
head to four bits is catastrophic -- and not a presentational choice.
"""

import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import parse_sweep_logs

# Categorical slots 1 and 2 of the validated palette: CVD separation dE 24.7
# (protan), 33.6 at normal vision against a light surface.
SR_COLOR = "#2a78d6"
RN_COLOR = "#eb6834"

# Secondary encoding, so identity is never carried by hue alone.
STYLE = {
    "sr": dict(color=SR_COLOR, marker="o", linestyle="-", label="SR"),
    "rn": dict(color=RN_COLOR, marker="s", linestyle="--", label="RN (untied)"),
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

SUBLAYERS = ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]
REDUCTION_LEN = {
    "attn_c_attn": 768,
    "attn_c_proj": 768,
    "mlp_c_fc": 768,
    "mlp_c_proj": 3072,
}


def style_axes(ax):
    """Recessive grid and axes; text in ink tokens, never a series colour."""
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    })


def series(summary, key_fn, xs, mode):
    """Extract (x, median, lo, hi) for one mode, skipping missing points."""
    out = []
    for x in xs:
        cell = summary.get((key_fn(x), mode))
        if cell:
            out.append((x, cell[0], cell[1], cell[2]))
    return out


def draw(ax, pts, mode, band=True):
    if not pts:
        return
    x = [p[0] for p in pts]
    ax.plot(x, [p[1] for p in pts], **STYLE[mode],
            markerfacecolor="white", markeredgewidth=1.2, zorder=3)
    # Spread across seeds. RN is bit-reproducible so its band has zero width;
    # drawing it anyway would imply a measurement that was not made.
    if band and any(p[3] > p[2] for p in pts):
        ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts],
                        color=STYLE[mode]["color"], alpha=0.18, linewidth=0,
                        zorder=2)


# ---------------------------------------------------------------- figure 1
def fig_global(records, out):
    s = parse_sweep_logs.summarize(records, "global", keys=("precision",))
    xs = sorted({k[0][0] for k in s})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for mode in ("sr", "rn"):
        draw(ax, series(s, lambda t: (t,), xs, mode), mode)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel("Virtual precision $t$ (significand bits)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Uniform precision everywhere", color=INK)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.savefig(out / "global_sweep.pdf")
    plt.close(fig)
    print(f"  wrote {out / 'global_sweep.pdf'}")


# ---------------------------------------------------------------- figure 2
def fig_sublayer(records, out):
    s = parse_sweep_logs.summarize(records, "sublayer")
    # The t=24 run is the in-level reference, not a swept point.
    ref = next((v[0] for k, v in s.items() if k[0][1] == 24), None)
    xs = sorted({k[0][1] for k in s if k[0][1] != 24})

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, SUBLAYERS):
        for mode in ("sr", "rn"):
            draw(ax, series(s, lambda t, L=layer: (L, t), xs, mode), mode)
        if ref:
            ax.axhline(ref, color=MUTED, linewidth=0.7, linestyle=":", zorder=1)
        ax.set_title(f"{layer}  ($n={REDUCTION_LEN[layer]}$)", color=INK)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        style_axes(ax)

    for ax in axes[1]:
        ax.set_xlabel("Virtual precision $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Perplexity")
    axes[0, 0].legend(frameon=False, loc="upper right")
    if ref:
        # Below the axes rather than inside them: at this scale every panel has
        # data near the reference line, so an in-panel annotation collides.
        fig.text(0.5, -0.04,
                 f"Dotted line: $t{{=}}24$ reference perplexity ({ref:.2f}). "
                 "Bands show min-max across five SR seeds.",
                 ha="center", color=MUTED, fontsize=6.5)
    fig.savefig(out / "sublayer_sweep.pdf")
    plt.close(fig)
    print(f"  wrote {out / 'sublayer_sweep.pdf'}")


# ------------------------------------------------------------ figures 3 & 4
def fig_positional(records, out, level, filename, xlabel, title, order_key):
    s = parse_sweep_logs.summarize(records, level, keys=("block", "precision"))
    blocks = sorted({k[0][0] for k in s}, key=order_key)
    precisions = sorted({k[0][1] for k in s})

    # Each precision gets its own y-scale. At t=4 the perplexities run into the
    # thousands while at t=6 they sit between 57 and 69; a shared axis flattens
    # the t=6 panel and hides the SR/RN crossover, which is the finding that
    # panel exists to show. These are small multiples, not one chart with two
    # scales.
    fig, axes = plt.subplots(1, len(precisions),
                            figsize=(6.5, 2.4), sharey=False)
    axes = [axes] if len(precisions) == 1 else list(axes)
    idx = {b: i for i, b in enumerate(blocks)}

    for ax, prec in zip(axes, precisions):
        for mode in ("sr", "rn"):
            pts = [(idx[b], *s[((b, prec), mode)])
                   for b in blocks if ((b, prec), mode) in s]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        **STYLE[mode], markerfacecolor="white",
                        markeredgewidth=1.2, zorder=3)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks, fontsize=7)
        # Log only where the range earns it; a narrow range on a log axis reads
        # as a straight line and hides the shape.
        lo, hi = ax.get_ylim()
        ax.set_yscale("log" if hi / max(lo, 1e-9) > 5 else "linear")
        ax.set_xlabel(xlabel)
        ax.set_title(f"$t={prec}$", color=INK)
        ax.set_ylabel("Perplexity")
        style_axes(ax)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title, color=INK, fontsize=8, y=1.04)
    fig.savefig(out / filename)
    plt.close(fig)
    print(f"  wrote {out / filename}")


def main():
    setup()
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "perplexity_logs")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    out.mkdir(parents=True, exist_ok=True)

    records = parse_sweep_logs.load(root)
    print(f"parsed {len(records)} configurations from {root}")

    fig_global(records, out)
    fig_sublayer(records, out)
    fig_positional(records, out, "blockwise", "blockwise_sensitivity.pdf",
                   "Reduced block",
                   "mlp_c_proj reduced in one block at a time",
                   order_key=lambda b: int(b))
    fig_positional(records, out, "cumulative", "cumulative_propagation.pdf",
                   "Blocks reduced",
                   "mlp_c_proj reduced cumulatively in blocks 0..k",
                   order_key=lambda b: (len(b), b))


if __name__ == "__main__":
    main()
