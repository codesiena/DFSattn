"""
Single Dense-Step Scanning Experiment for DFSAttn.

Scans all denoising steps (every step_interval) and replaces a single step
with full dense attention in an otherwise all-sparse generation process.
Generates baselines (all-dense, all-sparse) and all single-dense variants.

Usage:
    python experiments/scan_single_dense.py \
        --model_id $WAN_MODEL_ID \
        --prompt_file examples/vbench_33_prompts.txt \
        --prompt_source T2V_Wan_VBench \
        --start_idx 0 --end_idx 2 \
        --step_interval 10 \
        --output_dir results/scan_single_dense
"""

import argparse
import os
import sys
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from dfsattn.utils.seed import seed_everything
from dfsattn.utils.logger import logger
from dfsattn.utils.order import morton3d_perm, block3d_perm, hwf, hilbert3d_perm, hilbert2d_perm
from dfsattn.attn_processor import get_attn_processors, set_attn_processor
from dfsattn.replace_wan import Wan_DFSAttn_Processor2_0
from dfsattn.attention_wan import DFS_Attention, _dfs_attention_instances
from dataloader import load_prompt_or_image


def parse_args():
    parser = argparse.ArgumentParser(description="Single Dense-Step Scanning Experiment")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or local checkpoint path")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")

    parser.add_argument("--prompt_file", type=str, default="examples/vbench_33_prompts.txt")
    parser.add_argument("--prompt_source", type=str, default="T2V_Wan_VBench")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/scan_single_dense")
    parser.add_argument("--step_interval", type=int, default=10, help="Interval between tested dense steps (every N steps)")

    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--order", type=str, default="hilbert3d", choices=["org", "morton", "blk", "hwf", "hilbert3d", "hilbert2d"])
    parser.add_argument("--skip_layers", type=list[int], default=[0])
    parser.add_argument("--skip_steps", type=int, default=0, help="Number of initial dense steps (0 = all-sparse baseline)")
    parser.add_argument("--cache_interval", type=int, default=12)
    parser.add_argument("--sparsity_dcrt", type=float, default=0.1)
    parser.add_argument("--cache_flag", type=bool, default=True)

    return parser.parse_args()


def clear_dfs_state():
    DFS_Attention.clear_cache()
    _dfs_attention_instances.clear()


def build_attn_processors(pipe, args, mode, skip_steps, dense_step_idx):
    latent_f = args.num_frames // 4 + 1
    latent_h = args.height // 16
    latent_w = args.width // 16
    video_len = latent_f * latent_h * latent_w

    if args.order == "org":
        video_perm = None
    elif args.order == "morton":
        video_perm = morton3d_perm(latent_f, latent_h, latent_w)
    elif args.order == "blk":
        video_perm = block3d_perm(latent_f, latent_h, latent_w, a=4, b=4, c=4)
    elif args.order == "hwf":
        video_perm = hwf(latent_f, latent_h, latent_w)
    elif args.order == "hilbert3d":
        video_perm = hilbert3d_perm(latent_f, latent_h, latent_w)
    elif args.order == "hilbert2d":
        video_perm = hilbert2d_perm(latent_f, latent_h, latent_w)

    attn_processors = {}
    processors_id = 0
    for k, v in get_attn_processors(pipe.transformer).items():
        if "attn1" in k:
            attn_processors[k] = Wan_DFSAttn_Processor2_0(
                mode,
                args.sparsity,
                args.tile_size,
                args.block_size,
                video_perm,
                processors_id,
                args.skip_layers,
                skip_steps,
                args.cache_interval,
                args.sparsity_dcrt,
                args.cache_flag,
                False,
                dense_step_idx,
            )
        elif "attn2" in k:
            attn_processors[k] = Wan_DFSAttn_Processor2_0(
                mode,
                args.sparsity,
                args.tile_size,
                args.block_size,
                video_perm,
                processors_id,
                args.skip_layers,
                skip_steps,
                args.cache_interval,
                args.sparsity_dcrt,
                args.cache_flag,
                True,
                dense_step_idx,
            )
            processors_id += 1

    set_attn_processor(pipe.transformer, attn_processors)


def run_generation(pipe, args, prompt, config_name, output_file):
    seed_everything(args.seed)
    clear_dfs_state()

    logger.info(f"[{config_name}] Generating: {prompt.strip()[:60]}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    t_start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=6.0,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]
    t_elapsed = time.time() - t_start

    export_to_video(output, output_file, fps=24)
    logger.info(f"[{config_name}] Done in {t_elapsed:.1f}s -> {output_file}")
    return t_elapsed


def main():
    args = parse_args()

    logger.info("Loading model...")
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model_id, vae=vae, torch_dtype=torch.bfloat16)
    flow_shift = 5.0 if args.height >= 720 else 3.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to("cuda")
    logger.info("Model loaded.")

    for prompt_idx in range(args.start_idx, args.end_idx + 1):
        prompt, _ = load_prompt_or_image(args.prompt_source, prompt_idx, args.prompt_file, None)
        prompt_short = f"prompt_{prompt_idx}"
        base_dir = os.path.join(args.output_dir, prompt_short)

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing [{prompt_short}]: {prompt.strip()[:80]}")
        logger.info(f"{'='*60}")

        # --- all-dense baseline ---
        build_attn_processors(pipe, args, mode="flash", skip_steps=0, dense_step_idx=-1)
        run_generation(pipe, args, prompt, "all_dense", os.path.join(base_dir, "all_dense.mp4"))

        # --- all-sparse baseline ---
        build_attn_processors(pipe, args, mode="dfs", skip_steps=0, dense_step_idx=-1)
        run_generation(pipe, args, prompt, "all_sparse", os.path.join(base_dir, "all_sparse.mp4"))

        # --- single-dense-step scan ---
        dense_steps = list(range(0, args.num_inference_steps, args.step_interval))
        for t in dense_steps:
            build_attn_processors(pipe, args, mode="dfs", skip_steps=0, dense_step_idx=t)
            run_generation(pipe, args, prompt, f"dense_t{t}", os.path.join(base_dir, f"dense_t{t}.mp4"))

    logger.info("\nAll experiments completed.")


if __name__ == "__main__":
    main()
