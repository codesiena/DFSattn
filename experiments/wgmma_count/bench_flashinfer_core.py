#!/usr/bin/env python3
"""Profile one FlashInfer FA3 Core launch with a controlled CTA tile."""

import argparse
import json
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", choices=("128x96", "64x64"), required=True)
    parser.add_argument("--sequence", type=int, default=3072)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile == "64x64":
        os.environ["FLASHINFER_FA3_FORCE_64X64"] = "1"
        q_macro, k_macro = 64, 64
    else:
        os.environ.pop("FLASHINFER_FA3_FORCE_64X64", None)
        q_macro, k_macro = 128, 96

    if args.sequence % q_macro or args.sequence % k_macro:
        raise ValueError("sequence must be divisible by both tile dimensions")

    # Import after selecting the JIT variant.
    from dfsattn import flashinfer64_attention as fi64

    fi64.Q_MACRO = q_macro
    fi64.K_MACRO = k_macro

    torch.manual_seed(20260914)
    device = torch.device("cuda:0")
    shape = (args.heads, args.sequence, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    core_mask = torch.ones(
        (args.heads, args.sequence // q_macro, args.sequence // k_macro),
        dtype=torch.bool,
        device=device,
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    plan = fi64._DirectMacroCSRPlan(workspace, q)
    plan.plan(q, core_mask, (args.tile, args.sequence, args.heads))

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

    if args.save_output:
        torch.save(output.cpu(), args.save_output)

    result = {
        "tile": args.tile,
        "sequence": args.sequence,
        "heads": args.heads,
        "q_blocks": args.sequence // q_macro,
        "k_blocks": args.sequence // k_macro,
        "logical_qk_pairs": args.heads * args.sequence * args.sequence,
        "iterations": iterations,
        "mean_run_ms": start.elapsed_time(end) / iterations,
        "output_sum_fp32": output.float().sum().item(),
        "output_abs_sum_fp32": output.float().abs().sum().item(),
    }
    print("WGMMA_BENCH_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
