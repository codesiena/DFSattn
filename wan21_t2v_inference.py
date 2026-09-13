import argparse
import os
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import time
from dfsattn.utils.seed import seed_everything
from dfsattn.utils.logger import logger
from dfsattn.utils.order import morton3d_perm, block3d_perm, hwf, hilbert3d_perm, hilbert2d_perm
from dataloader import load_prompt_or_image, prompt_folder_name
from dfsattn.attn_processor import get_attn_processors, set_attn_processor
from dfsattn.replace_wan import Wan_DFSAttn_Processor2_0
from dfsattn.attention_wan import DFS_Attention

def str2bool(value):
    return value.lower() in ("true", "1", "yes", "y")

def parse_mask_heads(value):
    if value.strip().lower() == "all":
        return None
    try:
        heads = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "block-mask heads must be comma-separated integers or 'all'"
        ) from exc
    if not heads or any(head < 0 for head in heads):
        raise argparse.ArgumentTypeError("block-mask head indices must be non-negative")
    return heads

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=os.getenv("WAN_MODEL_ID"), help="Model ID or local checkpoint path to use for generation")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Wan_VBench", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")

    parser.add_argument("--mode", type=str, default="dfs", choices=["dfs", "flash", "torch", "vanilla"])
    parser.add_argument("--sparsity", type=float, default=0.3, help="The sparsity of sparse attention pattern.")
    parser.add_argument("--tile_size", type=int, default=16, help="The tile size of pooling in dfs attention.")
    parser.add_argument("--block_size", type=int, default=128, help="The block size in dfs attention.")
    parser.add_argument("--sparse_execution", choices=["native", "hybrid", "flex64", "flashinfer64"], default="native", help="DFS execution backend. flashinfer64 is the independent 64x64 tile Top-p + residual token Top-k backend.")
    parser.add_argument("--hybrid_threshold", type=int, default=8, help="Promote a 4x4 group of active 16x16 blocks to a 64x64 Core tile at this occupancy (1-16).")
    parser.add_argument("--flashinfer64_top_p", type=float, default=0.25, help="Per-head/per-q64-block cumulative probability mass for the independent FlashInfer64 selector.")
    parser.add_argument("--flashinfer64_token_top_k", type=int, default=None, help="Deprecated; FlashInfer64 Top-k is ratio-based. Use --flashinfer64_token_top_ratio.")
    parser.add_argument("--flashinfer64_token_top_ratio", type=float, default=0.10, help="Residual token fraction in unselected 64x64 tiles for flashinfer64.")
    parser.add_argument("--order", type=str, default="hilbert3d", choices=["org", "morton", "blk", "hwf", "hilbert3d", "hilbert2d"])
    parser.add_argument("--skip_layers", type=list[int], default=[0], help="Layer indices to skip in dfs attention")
    parser.add_argument("--skip_steps", type=int, default=12, help="Number of steps to skip in dfs attention")
    parser.add_argument("--cache_interval", type=int, default=12, help="Diffusion-step interval between sparse mask refreshes")
    parser.add_argument("--sparsity_dcrt", type=float, default=0.1, help="Sparsity decrement applied every cache interval")
    parser.add_argument("--cache_flag", type=bool, default=True, help="Cache the sparse mask in dfs attention")
    parser.add_argument("--dense_step_idx", type=int, default=-1, help="Force a specific denoising step to use full dense attention (-1 = never)")
    parser.add_argument("--dense_interval", type=int, default=0, help="Force a full dense attention step every N denoising steps after warmup (0 = never)")
    parser.add_argument("--rest_steps", type=int, default=0, help="Sparse steps after warmup before the second dense warmup (0 = skip phase)")
    parser.add_argument("--skip_steps2", type=int, default=0, help="Second dense warmup steps after rest_steps (0 = skip phase)")
    parser.add_argument("--record_density", type=str2bool, default=False, help="Record the actual density of sparse attention masks per step")
    parser.add_argument("--density_csv", type=str, default=None, help="CSV path for per-step execution-tile metrics")
    parser.add_argument("--record_timing", type=str2bool, default=True, help="Record video-level CUDA-event totals for Top-k and attention execution")
    parser.add_argument("--timing_csv", type=str, default=None, help="CSV path for video-level timing totals (default: timing.csv beside output_file)")
    parser.add_argument("--block_mask_dir", type=str, default=None, help="Root directory for top-k masks; a prompt-number/text subdirectory is created automatically")
    parser.add_argument("--block_mask_heads", type=parse_mask_heads, default=(0,), help="Comma-separated heads to render as heatmaps, or 'all'")
    parser.add_argument("--block_mask_layer_interval", type=int, default=15, help="Export every Nth layer (default: layers 0, 15, 30, ...)")
    parser.add_argument("--block_mask_save_bool", type=str2bool, default=False, help="Also save the raw bool .npy mask (default: false)")

    args = parser.parse_args()
    if args.model_id is None:
        parser.error("Please pass --model_id or set WAN_MODEL_ID.")
    if args.sparse_execution == "hybrid" and args.block_size != 16:
        parser.error("--sparse_execution hybrid requires --block_size 16 to preserve the logical DFS mask.")
    if args.sparse_execution == "native" and args.block_size != 128:
        parser.error("--sparse_execution native uses the upstream 128x128 block-sparse kernel; use --sparse_execution hybrid with --block_size 16.")
    if args.sparse_execution == "flex64" and args.block_size != 64:
        parser.error("--sparse_execution flex64 requires --block_size 64.")
    if not 0.0 < args.flashinfer64_top_p <= 1.0:
        parser.error("--flashinfer64_top_p must be in (0, 1].")
    if args.flashinfer64_token_top_k is not None:
        parser.error("--flashinfer64_token_top_k is no longer supported; use --flashinfer64_token_top_ratio.")
    if not 0.0 < args.flashinfer64_token_top_ratio <= 1.0:
        parser.error("--flashinfer64_token_top_ratio must be in (0, 1].")
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model_id, vae=vae, torch_dtype=torch.bfloat16)
    flow_shift = float(os.getenv("FLOW_SHIFT", "3.0" if args.height <= 480 else "5.0"))  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to("cuda")

    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)
    args.prompt = args.prompt.strip()
    if args.block_mask_dir is not None:
        prompt_mask_dir = os.path.join(
            args.block_mask_dir,
            prompt_folder_name(args.prompt_idx, args.prompt),
        )
        DFS_Attention.configure_mask_export(
            prompt_mask_dir,
            args.block_mask_heads,
            args.block_mask_layer_interval,
            args.block_mask_save_bool,
        )
        logger.info("Top-k masks for prompt {} will be saved to {}", args.prompt_idx, prompt_mask_dir)
    else:
        DFS_Attention.configure_mask_export(
            None,
            args.block_mask_heads,
            args.block_mask_layer_interval,
            args.block_mask_save_bool,
        )
    DFS_Attention.set_record_timing(args.record_timing)

    latent_f, latent_h, latent_w= args.num_frames // 4 + 1, args.height // 16, args.width // 16
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
    for k,v in get_attn_processors(pipe.transformer).items():
        if "attn1" in k:
            attn_processors[k] = Wan_DFSAttn_Processor2_0(
                args.mode,
                args.sparsity,
                args.tile_size,
                args.block_size,
                video_perm,
                processors_id,
                args.skip_layers,
                args.skip_steps,
                args.cache_interval,
                args.sparsity_dcrt,
                args.cache_flag,
                False,
                args.dense_step_idx,
                args.dense_interval,
                args.rest_steps,
                args.skip_steps2,
                args.record_density,
                args.sparse_execution,
                args.hybrid_threshold,
                args.flashinfer64_top_p,
                args.flashinfer64_token_top_k,
                args.flashinfer64_token_top_ratio,
            )
        elif "attn2" in k:
            attn_processors[k] = Wan_DFSAttn_Processor2_0(
                args.mode,
                args.sparsity,
                args.tile_size,
                args.block_size,
                video_perm,
                processors_id,
                args.skip_layers,
                args.skip_steps,
                args.cache_interval,
                args.sparsity_dcrt,
                args.cache_flag,
                True,
                args.dense_step_idx,
                args.dense_interval,
                args.rest_steps,
                args.skip_steps2,
                args.record_density,
                args.sparse_execution,
                args.hybrid_threshold,
                args.flashinfer64_top_p,
                args.flashinfer64_token_top_k,
                args.flashinfer64_token_top_ratio,
            )
            processors_id += 1
            
    set_attn_processor(pipe.transformer, attn_processors)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        e2e_gpu_start = torch.cuda.Event(enable_timing=True)
        e2e_gpu_end = torch.cuda.Event(enable_timing=True)
        e2e_gpu_start.record()
    else:
        e2e_gpu_start = e2e_gpu_end = None
    total_start_time = time.perf_counter()

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=6.0,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]

    if e2e_gpu_end is not None:
        e2e_gpu_end.record()
        torch.cuda.synchronize()
        e2e_gpu_ms = e2e_gpu_start.elapsed_time(e2e_gpu_end)
    else:
        e2e_gpu_ms = None
    total_generation_time = time.perf_counter() - total_start_time
    
    logger.info(f"End-to-end generation wall time: {total_generation_time:.3f} s")
    if e2e_gpu_ms is not None:
        logger.info(f"End-to-end generation GPU time: {e2e_gpu_ms:.3f} ms")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.record_timing:
        timing_path = args.timing_csv or os.path.join(output_dir or ".", "timing.csv")
        extra_totals = {"e2e_generation_wall": total_generation_time * 1_000.0}
        if e2e_gpu_ms is not None:
            extra_totals["e2e_generation_gpu"] = e2e_gpu_ms
        summary = DFS_Attention.dump_timing_records(timing_path, extra_totals=extra_totals)
        logger.info("Timing CSV saved to {}: {}", timing_path, summary)

    export_to_video(output, args.output_file, fps=24)

    
