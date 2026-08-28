#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

# The cluster home/cache directories are read-only in some jobs.  Keep plotting
# caches in /tmp so post-processing does not emit font/cache warnings or fail.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dfsattn-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dfsattn-xdg-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate DFSAttn sub-block retention histograms and draw per-layer/head distributions."
    )
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def weighted_quantile(counts, quantile):
    total = counts.sum()
    if total == 0:
        return float("nan")
    return (np.searchsorted(np.cumsum(counts), quantile * total, side="left") + 1) / len(counts)


def main():
    args = parse_args()
    paths = sorted(args.input_root.rglob("subblock_retention_hist.csv"))
    if not paths:
        raise SystemExit(f"No subblock_retention_hist.csv files found under {args.input_root}")

    hist = defaultdict(lambda: None)
    sample_names = set()
    for path in paths:
        sample_names.add(path.parent.name)
        with path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                key = (
                    int(row["layer_idx"]),
                    int(row["head_idx"]),
                    float(row["attention_mass"]),
                )
                total = int(row["total_subblocks"])
                if hist[key] is None:
                    hist[key] = np.zeros(total, dtype=np.int64)
                hist[key][int(row["retained_subblocks"]) - 1] += int(row["count"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({key[0] for key in hist})
    heads = sorted({key[1] for key in hist})
    masses = sorted({key[2] for key in hist})
    summary_path = args.output_dir / "subblock_retention_summary.csv"
    with summary_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["layer_idx", "head_idx", "attention_mass", "observations", "mean", "p50", "p90"]
        )
        for layer in layers:
            for head in heads:
                for mass in masses:
                    counts = hist.get((layer, head, mass))
                    if counts is None or counts.sum() == 0:
                        continue
                    ratios = np.arange(1, len(counts) + 1) / len(counts)
                    writer.writerow(
                        [
                            layer,
                            head,
                            mass,
                            int(counts.sum()),
                            float(np.average(ratios, weights=counts)),
                            weighted_quantile(counts, 0.5),
                            weighted_quantile(counts, 0.9),
                        ]
                    )

    for mass in masses:
        mass_tag = f"{mass:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        mean_matrix = np.full((len(layers), len(heads)), np.nan)
        for layer_pos, layer in enumerate(layers):
            available_heads = [head for head in heads if hist.get((layer, head, mass)) is not None]
            if not available_heads:
                continue
            columns = min(6, len(available_heads))
            rows = math.ceil(len(available_heads) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.5 * rows), squeeze=False)
            for axis in axes.flat:
                axis.set_visible(False)
            for plot_idx, head in enumerate(available_heads):
                axis = axes.flat[plot_idx]
                axis.set_visible(True)
                counts = hist[(layer, head, mass)]
                ratios = np.arange(1, len(counts) + 1) / len(counts)
                probabilities = counts / counts.sum()
                axis.bar(ratios, probabilities, width=0.8 / len(counts), color="#377eb8")
                mean = float(np.average(ratios, weights=counts))
                mean_matrix[layer_pos, heads.index(head)] = mean
                axis.axvline(mean, color="#e41a1c", linewidth=1.2, label=f"mean={mean:.2f}")
                axis.set_title(f"Head {head}", fontsize=9)
                axis.set_xlim(0, 1.01)
                axis.set_xlabel("Retention ratio", fontsize=8)
                axis.set_ylabel("Probability", fontsize=8)
                axis.tick_params(labelsize=7)
                axis.legend(fontsize=7, frameon=False)
            fig.suptitle(
                f"Layer {layer}: sub-block retention for {mass:.0%} attention mass\n"
                f"aggregated over {len(sample_names)} sample(s) and all Top-k refresh steps"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(args.output_dir / f"layer_{layer:02d}_mass_{mass_tag}.png", dpi=160)
            plt.close(fig)

        fig, axis = plt.subplots(figsize=(max(8, len(heads) * 0.45), max(6, len(layers) * 0.28)))
        image = axis.imshow(mean_matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(heads)), heads)
        axis.set_yticks(range(len(layers)), layers)
        axis.set_xlabel("Head")
        axis.set_ylabel("Layer")
        axis.set_title(f"Mean sub-block retention ratio for {mass:.0%} attention mass")
        fig.colorbar(image, ax=axis, label="Mean retention ratio")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"layer_head_mean_mass_{mass_tag}.png", dpi=180)
        plt.close(fig)

    print(f"Aggregated {len(paths)} sample histogram(s) into {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
