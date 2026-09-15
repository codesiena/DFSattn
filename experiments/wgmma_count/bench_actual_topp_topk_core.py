#!/usr/bin/env python3
"""Profile the real topp_topk Core mask selected from a HyVideo QKV dump."""

import argparse
import json
import os
from pathlib import Path

import torch


DEFAULT_DUMP = (
    "/cnic/work/liutt/mywork/attention_time/res/"
    "wgmma_exact_topp_topk_20260915/snapshot/prompt_0/"
    "step012_layer000_flashinfer64_topp_topk.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", choices=("128x96", "64x64"), required=True)
    parser.add_argument("--input-dump", default=DEFAULT_DUMP)
    parser.add_argument("--top-p", type=float, default=0.16)
    parser.add_argument("--order", default="hilbert3d")
    parser.add_argument("--num-frames", type=int, default=129)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--summary-json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile == "64x64":
        os.environ["FLASHINFER_FA3_FORCE_64X64"] = "1"
        q_macro, k_macro = 64, 64
    else:
        os.environ.pop("FLASHINFER_FA3_FORCE_64X64", None)
        q_macro, k_macro = 128, 96

    from dfsattn import flashinfer64_attention as fi64
    from analyze_macro_topk_error import _make_video_perm

    fi64.Q_MACRO = q_macro
    fi64.K_MACRO = k_macro
    fi64.Q_MICROS_PER_MACRO = q_macro // fi64.MICRO
    fi64.K_MICROS_PER_MACRO = k_macro // fi64.MICRO

    payload = torch.load(args.input_dump, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    device = torch.device("cuda:0")
    q = payload["query"].to(device)
    k = payload["key"].to(device)
    v = payload["value"].to(device)
    heads, full_sequence, dim = q.shape[1:]
    video_len = int(metadata["video_len"])
    valid_sequence = int(metadata["valid_sequence"])
    video_perm = _make_video_perm(
        args.order, args.num_frames, args.height, args.width, video_len, device
    )
    q, k, v, _ = fi64._permute_qkv(q, k, v, video_perm, video_len)
    q = q[:, :valid_sequence].contiguous()
    k = k[:, :valid_sequence].contiguous()
    v = v[:, :valid_sequence].contiguous()

    micro_scores = fi64.compute_micro_tile_scores(q, k)
    macro_scores = fi64.aggregate_macro_scores(micro_scores)
    core_mask = fi64._select_hyvideo_core_tiles(
        macro_scores,
        route_mode="topp_topk",
        tile_top_p=args.top_p,
        tile_top_ratio=1.0,
        sequence=valid_sequence,
        video_len=video_len,
    )
    occupancy = torch.zeros_like(core_mask, dtype=torch.uint8)
    execution_mask, empty_qblocks = fi64._make_core_execution_mask(core_mask, occupancy)

    q_blocks, k_blocks = core_mask.shape[1:]
    q_sizes = torch.full((q_blocks,), q_macro, dtype=torch.int64, device=device)
    k_sizes = torch.full((k_blocks,), k_macro, dtype=torch.int64, device=device)
    q_sizes[-1] = valid_sequence - (q_blocks - 1) * q_macro
    k_sizes[-1] = valid_sequence - (k_blocks - 1) * k_macro
    logical_qk_pairs = int(
        (core_mask * q_sizes[None, :, None] * k_sizes[None, None, :]).sum().item()
    )
    dense_qk_pairs = heads * valid_sequence * valid_sequence

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    plan = fi64._DirectMacroCSRPlan(workspace, q)
    plan.plan(q, execution_mask, ("actual_topp_topk", args.tile, args.top_p))
    del micro_scores, macro_scores

    for _ in range(args.warmup):
        output = plan.run(q, k, v, return_lse=False)
    torch.cuda.synchronize()

    iterations = 1 if args.profile else args.iterations
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = plan.run(q, k, v, return_lse=False)
    end.record()
    torch.cuda.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    selected_per_row = core_mask.sum(-1).float()
    result = {
        "source_dump": str(Path(args.input_dump).resolve()),
        "source_step": metadata.get("step_idx"),
        "source_layer": metadata.get("layer_idx"),
        "source_backend": metadata.get("backend"),
        "route_mode": "topp_topk",
        "top_p": args.top_p,
        "tile": args.tile,
        "heads": heads,
        "full_sequence": full_sequence,
        "valid_sequence": valid_sequence,
        "video_len": video_len,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "selected_core_blocks": int(core_mask.sum().item()),
        "execution_anchor_blocks": int((execution_mask & ~core_mask).sum().item()),
        "empty_qblocks": int(empty_qblocks.sum().item()),
        "selected_blocks_per_row_mean": selected_per_row.mean().item(),
        "selected_blocks_per_row_min": selected_per_row.min().item(),
        "selected_blocks_per_row_max": selected_per_row.max().item(),
        "logical_qk_pairs": logical_qk_pairs,
        "dense_qk_pairs": dense_qk_pairs,
        "core_density": logical_qk_pairs / dense_qk_pairs,
        "iterations": iterations,
        "mean_run_ms": start.elapsed_time(end) / iterations,
        "output_sum_fp32": output.float().sum().item(),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("ACTUAL_TOPP_TOPK_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
