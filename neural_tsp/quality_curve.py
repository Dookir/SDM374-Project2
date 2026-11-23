"""Plot finished public experiments at their measured wall times."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = defaultdict(list)
    shared = None
    for path in args.summary:
        entry = json.loads(path.read_text())
        if not entry.get("complete") or not entry.get("instances"):
            continue
        dataset = (entry["dataset"], entry["instances"])
        if shared is None:
            shared = dataset
        elif shared != dataset:
            parser.error("Plot only experiments using the same dataset and sample count")
        runs[entry["mode"]].append((entry["total_seconds"] / entry["instances"], entry["avg_aug_gap"]))
    if not runs:
        parser.error("No complete benchmark result")
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for mode, points in sorted(runs.items()):
        seconds, gaps = zip(*sorted(points))
        ax.plot(seconds, gaps, marker="o", label=mode)
    ax.set(xlabel="Measured wall time per instance (s)", ylabel="Gap (%)", xscale="log")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
