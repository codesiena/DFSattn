#!/usr/bin/env python3
"""Compare same-step DFSAttn and FlashInfer64 attention debug dumps."""

import argparse
import json

import torch


def tensor_metrics(left: torch.Tensor, right: torch.Tensor, chunk_size: int = 1 << 20):
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    left = left.reshape(-1)
    right = right.reshape(-1)
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq_diff = 0.0
    sum_sq_right = 0.0
    count = left.numel()
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        a = left[start:end].float()
        b = right[start:end].float()
        diff = a - b
        max_abs = max(max_abs, float(diff.abs().max().item()))
        sum_abs += float(diff.abs().sum().item())
        sum_sq_diff += float(torch.dot(diff, diff).item())
        sum_sq_right += float(torch.dot(b, b).item())
    return {
        "max_abs": max_abs,
        "mean_abs": sum_abs / max(count, 1),
        "relative_l2": (sum_sq_diff / max(sum_sq_right, 1e-40)) ** 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dfs_dump", help=".pt dump from the native DFSAttn run")
    parser.add_argument("ours_dump", help=".pt dump from the FlashInfer64 run")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    dfs = torch.load(args.dfs_dump, map_location="cpu", weights_only=True)
    ours = torch.load(args.ours_dump, map_location="cpu", weights_only=True)
    report = {
        "dfs_metadata": dfs["metadata"],
        "ours_metadata": ours["metadata"],
        "same_input_checks": {
            name: tensor_metrics(dfs[name], ours[name])
            for name in ("query", "key", "value")
        },
        "dense_reference_cross_run": tensor_metrics(
            dfs["dense_output"], ours["dense_output"]
        ),
        "dfs_vs_same_input_dense": tensor_metrics(
            dfs["sparse_output"], dfs["dense_output"]
        ),
        "ours_vs_same_input_dense": tensor_metrics(
            ours["sparse_output"], ours["dense_output"]
        ),
        "dfs_vs_ours_sparse_output": tensor_metrics(
            dfs["sparse_output"], ours["sparse_output"]
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
