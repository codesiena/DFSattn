#!/usr/bin/env python3
"""Kernel-only Go/No-Go benchmark for the Hybrid-DFSAttn execution backend.

Examples:
  python idea2/benchmark_hybrid_attention.py --device cuda --seq-len 4096 --heads 16 --head-dim 128
  python idea2/benchmark_hybrid_attention.py --device cuda --mask path/to/block_mask.npy

The input mask must be the *16x16 logical DFS mask* with shape [H, Q16, K16].
It reports planning separately and includes it in ``hybrid total``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
import sys
from typing import Callable, List, Tuple

import numpy as np
import torch

# Permit ``python idea2/benchmark_hybrid_attention.py`` from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfsattn.hybrid_block_attention import (
    FINE_BLOCK,
    fine_sparse_attention,
    hybrid_sparse_attention,
    hybrid_sparse_attention_from_plan,
    partition_block_mask,
    plan_statistics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = REPO_ROOT / "examples" / "vbench_11_prompts.txt"
DEFAULT_OUTPUT_DIR = Path("/cnic/work/liutt/mywork/attention_time/res/dual")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn: Callable[[], object], device: torch.device, warmup: int, repeats: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples: List[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1_000)
    return tuple(float(x) for x in np.percentile(samples, [25, 50, 75]))


def synthetic_mask(heads: int, sequence: int, density: float, device: torch.device) -> torch.Tensor:
    blocks = (sequence + FINE_BLOCK - 1) // FINE_BLOCK
    mask = torch.rand((heads, blocks, blocks), device=device) < density
    # A valid attention state requires at least one key block for every q block.
    empty = ~mask.any(dim=-1)
    if empty.any():
        h, q = empty.nonzero(as_tuple=True)
        mask[h, q, q % blocks] = True
    return mask


def load_mask(path: str, device: torch.device) -> torch.Tensor:
    mask = np.load(path, allow_pickle=False)
    if mask.ndim == 2:
        mask = mask[None]
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def run_vbench(args: argparse.Namespace) -> None:
    """Run the real 11-prompt video experiment through the inference driver."""

    backend = args.backend
    if backend == "wan":
        driver = REPO_ROOT / "wan21_t2v_inference.py"
        prompt_source = "T2V_Wan_VBench"
        model_id = args.model_id or os.environ.get("WAN_MODEL_ID")
    else:
        driver = REPO_ROOT / "hyvideo_t2v_inference.py"
        prompt_source = "T2V_Hyv_VBench"
        model_id = args.model_id or os.environ.get("HYVIDEO_MODEL_ID")
    if not model_id:
        raise SystemExit(
            f"Set {'WAN_MODEL_ID' if backend == 'wan' else 'HYVIDEO_MODEL_ID'} "
            "or pass --model-id before running VBench."
        )

    prompt_file = Path(args.prompt_file).expanduser().resolve()
    if not prompt_file.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_file}")
    prompts = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    start = max(0, args.start_idx)
    end = min(len(prompts) - 1, args.end_idx if args.end_idx >= 0 else len(prompts) - 1)
    if start > end:
        raise SystemExit(f"Invalid prompt range [{start}, {end}] for {len(prompts)} prompts.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    for prompt_idx in range(start, end + 1):
        output_file = output_dir / f"{prompt_idx}.mp4"
        timing_file = output_dir / f"{prompt_idx}.timing.csv"
        if args.skip_existing and output_file.is_file() and output_file.stat().st_size > 0:
            print(f"skip prompt {prompt_idx}: {output_file}")
            continue
        command = [
            "python", str(driver),
            "--model_id", str(model_id),
            "--prompt", str(prompt_file),
            "--prompt_source", prompt_source,
            "--prompt_idx", str(prompt_idx),
            "--output_file", str(output_file),
            "--mode", "dfs",
            "--tile_size", "16",
            "--block_size", "16",
            "--sparse_execution", "hybrid",
            "--hybrid_threshold", str(args.hybrid_threshold),
            "--record_timing", "true",
            "--timing_csv", str(timing_file),
            "--seed", str(args.seed),
            "--num_inference_steps", str(args.num_inference_steps),
            "--block_mask_dir", str(mask_dir),
            "--block_mask_save_bool", "true",
        ]
        if args.height is not None:
            command.extend(["--height", str(args.height)])
        if args.width is not None:
            command.extend(["--width", str(args.width)])
        if args.num_frames is not None:
            command.extend(["--num_frames", str(args.num_frames)])
        print(f"run prompt {prompt_idx}/{len(prompts) - 1}: {prompts[prompt_idx]}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-vbench", action="store_true", help="Run real VBench prompts through the Hybrid video inference driver")
    parser.add_argument("--backend", choices=("wan", "hyvideo"), default="wan", help="Video model used by --run-vbench")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="Prompt list used by --run-vbench")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory used by --run-vbench")
    parser.add_argument("--model-id", default=None, help="Model path; otherwise WAN_MODEL_ID/HYVIDEO_MODEL_ID")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--hybrid-threshold", type=int, default=8, help="Keep the current Core promotion threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--mask", help="Saved [heads, q16, k16] bool NumPy mask")
    args = parser.parse_args()

    if args.run_vbench:
        run_vbench(args)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this process.")
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        # CPU BF16 kernels vary heavily by host; FP32 is the portable smoke test.
        dtype = torch.float32

    torch.manual_seed(0)
    mask = load_mask(args.mask, device) if args.mask else synthetic_mask(args.heads, args.seq_len, args.density, device)
    sequence = args.seq_len if args.mask is None else mask.shape[1] * FINE_BLOCK
    q = torch.randn((1, mask.shape[0], sequence, args.head_dim), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    baseline = fine_sparse_attention(q, k, v, mask)
    base_p25, base_median, base_p75 = measure(lambda: fine_sparse_attention(q, k, v, mask), device, args.warmup, args.repeats)
    print(f"device={device}, dtype={dtype}, qkv=[1,{mask.shape[0]},{sequence},{args.head_dim}]")
    print("method,threshold,core64,residual16,logical_qk,actual_qk,planning_ms,execute_merge_ms,total_ms,p25_ms,p75_ms,speedup,max_abs_error")
    print(f"fine16_reference,-,0,{int(mask.sum().item())},{int(mask.sum().item()) * FINE_BLOCK**2},{int(mask.sum().item()) * FINE_BLOCK**2},0.000,{base_median:.3f},{base_median:.3f},{base_p25:.3f},{base_p75:.3f},1.000,0.000e+00")

    for threshold in args.thresholds:
        plan_p25, plan_median, plan_p75 = measure(lambda: partition_block_mask(mask, threshold=threshold), device, args.warmup, args.repeats)
        plan = partition_block_mask(mask, threshold=threshold)
        stats = plan_statistics(plan)
        def execute() -> torch.Tensor:
            return hybrid_sparse_attention_from_plan(q, k, v, plan)
        exec_p25, exec_median, exec_p75 = measure(execute, device, args.warmup, args.repeats)
        def end_to_end() -> torch.Tensor:
            return hybrid_sparse_attention(q, k, v, mask, threshold=threshold)
        full_p25, full_median, full_p75 = measure(end_to_end, device, args.warmup, args.repeats)
        candidate = execute()
        error = (candidate.float() - baseline.float()).abs().max().item()
        print(
            f"hybrid,{threshold},{stats['core_tiles_64']},{stats['residual_tiles_16']},"
            f"{stats['logical_qk']},{stats['actual_qk']},{plan_median:.3f},{exec_median:.3f},"
            f"{full_median:.3f},{full_p25:.3f},{full_p75:.3f},{base_median / full_median:.3f},{error:.3e}"
        )


if __name__ == "__main__":
    main()
