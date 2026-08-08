"""Parse perplexity sweep logs into tidy records.

The sweeps write one log per configuration. Mode and seed live in the path
(``fine_sr_block_all/layer_mlp_c_proj_prec_6_seed_3.log``) while the measured
values live in the final ``Result ->`` line, so both have to be read.

A log without a ``Result ->`` line is an incomplete run and is skipped rather
than counted, which matters because an interrupted sweep leaves such files
behind.
"""

import re
from collections import defaultdict
from pathlib import Path
from statistics import median

RESULT = re.compile(r"^Result ->")
PRECISION = re.compile(r"Precision:\s*(\d+)")
PERPLEXITY = re.compile(r"Perplexity:\s*([\d.eE+-]+)")
TIME = re.compile(r"Time:\s*([\d.]+)s")
GROUP = re.compile(r"Group:\s*(\S+)")
LAYER = re.compile(r"Layer:\s*(\S+)")
BLOCK = re.compile(r"Block:\s*(\S+)")
SEED = re.compile(r"_seed_(\d+)\.log$")


def _mode_from_path(path: Path) -> str:
    """SR and RN runs are separated by directory, not by anything in the line."""
    parts = "/".join(path.parts)
    if "_sr" in parts or "global_sr" in parts:
        return "sr"
    if "_rn" in parts or "global_rn" in parts:
        return "rn"
    if "ieee_reference" in parts:
        return "ieee"
    return "unknown"


def _level(result: str, block: str | None, relpath: str) -> str:
    """Classify by directory first.

    The block string alone cannot separate the blockwise from the cumulative
    sweep: the cumulative sweep's first point is a single block ("0"), spelled
    identically to a blockwise point. Only the directory distinguishes them.
    """
    if "Global" in result or "global_" in relpath:
        return "global"
    if GROUP.search(result):
        return "component"
    if "cumulative" in relpath:
        return "cumulative"
    if "blockwise" in relpath:
        return "blockwise"
    if block == "all":
        return "sublayer"
    return "blockwise"


def load(root) -> list[dict]:
    """Return one record per completed configuration under ``root``."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*.log")):
        text = path.read_text(errors="replace")
        line = next((l for l in text.splitlines() if RESULT.match(l)), None)
        if line is None:
            continue  # incomplete run

        relpath = "/".join(path.relative_to(root).parts)
        block_m = BLOCK.search(line)
        block = block_m.group(1) if block_m else None
        target_m = GROUP.search(line) or LAYER.search(line)
        seed_m = SEED.search(path.name)

        records.append(
            dict(
                level=_level(line, block, relpath),
                mode=_mode_from_path(path.relative_to(root)),
                target=target_m.group(1) if target_m else "global",
                block=block or "all",
                precision=int(PRECISION.search(line).group(1)),
                perplexity=float(PERPLEXITY.search(line).group(1)),
                seconds=float(TIME.search(line).group(1)),
                seed=int(seed_m.group(1)) if seed_m else 1,
                path=str(path),
            )
        )
    return records


def summarize(records, level, keys=("target", "precision")):
    """Collapse seeds into median plus min/max, keyed by ``keys`` and mode.

    Returns ``{(key_tuple, mode): (median, low, high, n)}``. SR arms have several
    seeds; RN arms have one because PRISM's RN is bit-reproducible, so low and
    high collapse onto the median there.
    """
    buckets = defaultdict(list)
    for r in records:
        if r["level"] != level:
            continue
        buckets[(tuple(r[k] for k in keys), r["mode"])].append(r["perplexity"])
    return {
        k: (median(v), min(v), max(v), len(v)) for k, v in buckets.items()
    }


if __name__ == "__main__":
    import sys

    recs = load(sys.argv[1] if len(sys.argv) > 1 else "perplexity_logs")
    by_level = defaultdict(int)
    for r in recs:
        by_level[(r["level"], r["mode"])] += 1
    print(f"{len(recs)} completed configurations")
    for k in sorted(by_level):
        print(f"  {k[0]:<11} {k[1]:<5} {by_level[k]:>4}")
