import argparse
import os
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
import time
from dfsattn.utils.seed import seed_everything
from dfsattn.utils.logger import logger
from dfsattn.utils.order import block3d_perm, hwf, fwh, hilbert3d_perm, hilbert2d_perm
from dataloader import load_prompt_or_image, prompt_folder_name
from dfsattn.replace_hyvideo import AttentionDebugComplete, HunyuanVideo_DFSAttn_Processor2_0
from dfsattn.attention_hyvideo import DFS_Attention

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

def parse_attention_masses(value):
    try:
        masses = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("attention masses must be comma-separated floats") from exc
    if not masses or any(not 0.0 < mass <= 1.0 for mass in masses):
        raise argparse.ArgumentTypeError("attention masses must be in (0, 1]")
    return masses

def parse_layer_indices(value):
    try:
        layers = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layer indices must be comma-separated integers") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("layer indices must be non-negative")
    return layers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=os.getenv("HYVIDEO_MODEL_ID"), help="Model ID or local checkpoint path to use for generation")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--negative_prompt", type=str, default="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion", help="Negative text prompt to avoid certain features")
    parser.add_argument("--prompt", type=str, default="As night falls, on Michigan Avenue in Chicago, the towering buildings are decorated with colorful neon lights on their exterior walls, forming a brilliant ocean of light that contrasts sharply with the dark night sky.", help="Text prompt for video generation")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Hyv_VBench", "T2V_Hyv_Web", "T2V_Xingyang_Motion", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")

    parser.add_argument("--mode", type=str, default="dfs", choices=["dfs", "flash", "torch", "vanilla"])
    parser.add_argument("--sparsity", type=float, default=0.3, help="The sparsity of sparse attention pattern.")
    parser.add_argument("--tile_size", type=int, default=16, help="The tile size of pooling in dfs attention.")
    parser.add_argument("--block_size", type=int, default=128, help="The block size in dfs attention.")
    parser.add_argument("--block_top_p", type=float, default=None, help="Use block Top-p with this hierarchical-score probability mass (0 < p <= 1); omit to retain DFS block Top-k.")
    parser.add_argument("--token_top_k", type=int, default=0, help="Exact oracle residual token Top-k from the unselected block region (0 = block selector only).")
    parser.add_argument("--residual_candidate_blocks", type=int, default=4, help="Number of highest-scoring unselected blocks searched by residual token Top-k (default: 4).")
    parser.add_argument("--selector_mode", choices=["topk", "kp"], default="topk", help="Coarse selector: upstream Top-k, or Top-k followed by local sub-block Top-p")
    parser.add_argument("--fine_top_p", type=float, default=0.9, help="Sub-block cumulative mass used by --selector_mode kp (default: 0.9)")
    parser.add_argument("--sparse_execution", choices=["native", "hybrid", "flashinfer64"], default="native", help="DFS execution backend; flashinfer64 now uses Q128xK96 Core plus Q16xK16 Residual.")
    parser.add_argument("--hybrid_threshold", type=int, default=8, help="Promote a 4x4 group of active 16x16 blocks to a 64x64 Core tile at this occupancy (1-16).")
    parser.add_argument("--flashinfer64_top_p", type=float, default=0.25, help="Per-head/per-Q128 macro cumulative mass sent to FlashInfer.")
    parser.add_argument("--flashinfer64_token_top_k", type=int, default=None, help="Deprecated; FlashInfer64 Top-k is ratio-based. Use --flashinfer64_token_top_ratio.")
    parser.add_argument("--flashinfer64_token_top_ratio", type=float, default=0.10, help="Deprecated compatibility field; the new Residual uses micro Top-p.")
    parser.add_argument("--flashinfer64_route_mode", choices=["topp_topk", "topk_topp"], default="topk_topp", help="Choose macro Top-p or ratio Top-k for the Q128xK96 FlashInfer Core; topk_topp is the paper path.")
    parser.add_argument("--flashinfer64_tile_top_ratio", type=float, default=0.25, help="Fraction of K96 macro tiles selected for each head/Q128 in topk_topp mode.")
    parser.add_argument("--flashinfer64_token_top_p", type=float, default=0.9, help="Target total Q16/K16 proxy mass covered by Core plus Residual; rejected mass is not renormalized.")
    parser.add_argument("--flashinfer64_promotion_threshold", type=int, default=24, help="Promote a Q128xK96 tile when at least this many of its 48 Q16xK16 microtiles are selected.")
    parser.add_argument("--flashinfer64_route_cache", type=str2bool, default=False, help="Reuse compact routes across cache intervals. False is the bounded-memory bring-up mode; expanded FlashInfer plans are never cached.")
    parser.add_argument("--order", type=str, default="hilbert3d", choices=["org", "hilbert2d", "blk", "hwf", "fwh","hilbert3d"])
    parser.add_argument("--skip_layers", type=list[int], default=[], help="Layer indices to skip in dfs attention")
    parser.add_argument("--skip_steps", type=int, default=12, help="Number of steps to skip in dfs attention")
    parser.add_argument("--cache_interval", type=int, default=12, help="Diffusion-step interval between sparse mask refreshes")
    parser.add_argument("--sparsity_dcrt", type=float, default=0.1, help="Sparsity decrement applied every cache interval")
    parser.add_argument("--cache_flag", type=bool, default=True, help="Cache the sparse mask in dfs attention")
    parser.add_argument("--dense_interval", type=int, default=0, help="Force a full dense attention step every N denoising steps after warmup (0 = never)")
    parser.add_argument("--rest_steps", type=int, default=0, help="Sparse steps after warmup before the second dense warmup (0 = skip phase)")
    parser.add_argument("--skip_steps2", type=int, default=0, help="Second dense warmup steps after rest_steps (0 = skip phase)")
    parser.add_argument("--save_dense_warmup", type=str2bool, default=False, help="Save a decoded video immediately after the initial dense warmup")
    parser.add_argument("--dense_warmup_output", type=str, default=None, help="Output path for the post-dense-warmup video")
    parser.add_argument("--record_density", type=str2bool, default=False, help="Record the actual density of sparse attention masks per step")
    parser.add_argument("--record_timing", type=str2bool, default=True, help="Record video-level CUDA-event totals for Top-k and attention execution")
    parser.add_argument("--timing_csv", type=str, default=None, help="CSV path for video-level timing totals (default: timing.csv beside output_file)")
    parser.add_argument("--attention_debug_dir", type=str, default=None, help="Dump exact Q/K/V, sparse output, and same-input dense output at the first sparse step.")
    parser.add_argument("--attention_debug_step", type=int, default=-1, help="Diffusion step to dump (-1 means --skip_steps).")
    parser.add_argument("--attention_debug_layers", type=parse_layer_indices, default=(0,), help="Comma-separated layers to dump; layer 0 is the first meaningful divergence point.")
    parser.add_argument("--attention_debug_stop", type=str2bool, default=True, help="Stop cleanly after dumping the highest requested layer.")
    parser.add_argument("--block_mask_dir", type=str, default=None, help="Root directory for top-k masks; a prompt-number/text subdirectory is created automatically")
    parser.add_argument("--block_mask_heads", type=parse_mask_heads, default=(0,), help="Comma-separated heads to render as heatmaps, or 'all'")
    parser.add_argument("--block_mask_layer_interval", type=int, default=15, help="Export every Nth layer (default: layers 0, 15, 30, ...)")
    parser.add_argument("--block_mask_save_bool", type=str2bool, default=False, help="Also save the raw bool .npy mask (default: false)")
    parser.add_argument("--subblock_profile_dir", type=str, default=None, help="Profile fine-tile retention inside Top-k-selected coarse blocks and write histograms here")
    parser.add_argument("--subblock_profile_masses", type=parse_attention_masses, default=(0.9,), help="Comma-separated target attention masses (default: 0.9)")
   
    args = parser.parse_args()
    if args.model_id is None:
        parser.error("Please pass --model_id or set HYVIDEO_MODEL_ID.")
    if args.block_top_p is not None and not 0.0 < args.block_top_p <= 1.0:
        parser.error("--block_top_p must be in (0, 1].")
    if args.token_top_k < 0:
        parser.error("--token_top_k must be non-negative.")
    if args.residual_candidate_blocks < 0:
        parser.error("--residual_candidate_blocks must be non-negative.")
    if args.token_top_k > 0 and args.residual_candidate_blocks == 0:
        parser.error("--residual_candidate_blocks must be greater than 0 when --token_top_k is enabled.")
    if not 0.0 < args.fine_top_p <= 1.0:
        parser.error("--fine_top_p must be in (0, 1].")
    if not 0.0 < args.flashinfer64_top_p <= 1.0:
        parser.error("--flashinfer64_top_p must be in (0, 1].")
    if args.flashinfer64_token_top_k is not None:
        parser.error("--flashinfer64_token_top_k is no longer supported; use --flashinfer64_token_top_ratio.")
    if not 0.0 < args.flashinfer64_token_top_ratio <= 1.0:
        parser.error("--flashinfer64_token_top_ratio must be in (0, 1].")
    if not 0.0 < args.flashinfer64_tile_top_ratio <= 1.0:
        parser.error("--flashinfer64_tile_top_ratio must be in (0, 1].")
    if not 0.0 <= args.flashinfer64_token_top_p <= 1.0:
        parser.error("--flashinfer64_token_top_p must be in [0, 1].")
    if not 1 <= args.flashinfer64_promotion_threshold <= 48:
        parser.error("--flashinfer64_promotion_threshold must be in [1, 48].")
    if args.selector_mode == "kp" and args.block_top_p is not None:
        parser.error("--selector_mode kp uses coarse Top-k and cannot be combined with --block_top_p.")
    if args.selector_mode == "kp" and args.token_top_k > 0:
        parser.error("--selector_mode kp cannot currently be combined with residual --token_top_k.")
    if args.selector_mode == "kp" and (
        args.block_size != 128 or args.tile_size != 16 or args.sparse_execution != "hybrid"
    ):
        parser.error("--selector_mode kp requires --block_size 128 --tile_size 16 --sparse_execution hybrid.")
    if args.selector_mode != "kp" and args.sparse_execution == "hybrid" and args.block_size != 16:
        parser.error("--sparse_execution hybrid requires --block_size 16 to preserve the logical DFS mask.")
    if args.selector_mode != "kp" and args.sparse_execution == "native" and args.block_size != 128:
        parser.error("--sparse_execution native uses the upstream 128x128 block-sparse kernel; use --sparse_execution hybrid with --block_size 16.")
    if args.sparse_execution == "flashinfer64" and args.selector_mode != "topk":
        parser.error("--sparse_execution flashinfer64 is independent and requires --selector_mode topk.")
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    pipe = HunyuanVideoPipeline.from_pretrained(args.model_id, transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
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
    if args.subblock_profile_dir is not None:
        prompt_profile_dir = os.path.join(
            args.subblock_profile_dir,
            prompt_folder_name(args.prompt_idx, args.prompt),
        )
        DFS_Attention.configure_subblock_retention_profile(
            prompt_profile_dir, args.subblock_profile_masses
        )
        logger.info(
            "Sub-block retention profile for prompt {} will be saved to {}",
            args.prompt_idx,
            prompt_profile_dir,
        )
    else:
        DFS_Attention.configure_subblock_retention_profile(None)
    DFS_Attention.set_record_timing(args.record_timing)

    latent_f, latent_h, latent_w= args.num_frames // 4 + 1, args.height // 16, args.width // 16
    video_len = latent_f * latent_h * latent_w
    if args.order == "org":
        video_perm = None
    elif args.order == "hilbert2d":
        video_perm = hilbert2d_perm(latent_f, latent_h, latent_w)
    elif args.order == "blk":
        video_perm = block3d_perm(latent_f, latent_h, latent_w, a=4, b=4, c=4)
    elif args.order == "hwf":
        video_perm = hwf(latent_f, latent_h, latent_w)
    elif args.order == "fwh":
        video_perm = fwh(latent_f, latent_h, latent_w)
    elif args.order == "hilbert3d":
        video_perm = hilbert3d_perm(latent_f, latent_h, latent_w)

    attn_processors = {}
    processors_id = 0
    for k,v in transformer.attn_processors.items():
        if "token_refiner" in k:
            attn_processors[k] = v
        else:
            attn_processors[k] = HunyuanVideo_DFSAttn_Processor2_0(
                args.mode,
                args.sparsity,
                args.tile_size,
                args.block_size,
                video_len,
                video_perm,
                processors_id,
                args.skip_layers,
                args.skip_steps,
                args.cache_interval,
                args.sparsity_dcrt,
                args.cache_flag,
                args.dense_interval,
                args.rest_steps,
                args.skip_steps2,
                args.record_density,
                args.sparse_execution,
                args.hybrid_threshold,
                args.block_top_p,
                args.token_top_k,
                args.residual_candidate_blocks,
                args.selector_mode,
                args.fine_top_p,
                args.flashinfer64_top_p,
                args.flashinfer64_token_top_k,
                args.flashinfer64_token_top_ratio,
                args.flashinfer64_route_mode,
                args.flashinfer64_tile_top_ratio,
                args.flashinfer64_token_top_p,
                args.flashinfer64_promotion_threshold,
                args.flashinfer64_route_cache,
                args.attention_debug_dir,
                args.attention_debug_step,
                args.attention_debug_layers,
                args.attention_debug_stop,
            )
            processors_id += 1   
    transformer.set_attn_processor(attn_processors)

    # Wall time is the end-to-end generation latency.  CUDA events provide the
    # corresponding GPU stream time; both boundaries are synchronized so no
    # previous/following asynchronous work leaks into the measurement.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        e2e_gpu_start = torch.cuda.Event(enable_timing=True)
        e2e_gpu_end = torch.cuda.Event(enable_timing=True)
        e2e_gpu_start.record()
    else:
        e2e_gpu_start = e2e_gpu_end = None
    total_start_time = time.perf_counter()

    dense_warmup_saved = [False]

    def save_dense_warmup_callback(pipe, step_idx, timestep, callback_kwargs):
        if dense_warmup_saved[0] or not args.save_dense_warmup or args.skip_steps <= 0:
            return callback_kwargs
        if step_idx != args.skip_steps - 1:
            return callback_kwargs

        warmup_output = args.dense_warmup_output or os.path.splitext(args.output_file)[0] + "_dense_warmup.mp4"
        os.makedirs(os.path.dirname(warmup_output) or ".", exist_ok=True)
        warmup_latents = callback_kwargs["latents"].to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        warmup_video = pipe.vae.decode(warmup_latents, return_dict=False)[0]
        warmup_frames = pipe.video_processor.postprocess_video(warmup_video, output_type="np")[0]
        export_to_video(warmup_frames, warmup_output, fps=24)
        dense_warmup_saved[0] = True
        logger.info("Dense warmup video saved to {}", warmup_output)
        return callback_kwargs

    callback_on_step_end = save_dense_warmup_callback if args.save_dense_warmup else None

    try:
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=6.0,
            num_inference_steps=args.num_inference_steps,
            callback_on_step_end=callback_on_step_end,
        ).frames[0]
    except AttentionDebugComplete as exc:
        logger.info("{}; stopping before the remaining denoising steps", exc)
        raise SystemExit(0)

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

    if args.record_density:
        output_dir = os.path.dirname(args.output_file) or "."
        DFS_Attention.dump_density_records(os.path.join(output_dir, "density_records.csv"))
        DFS_Attention.dump_density_summary(
            os.path.join(output_dir, "density_summary.csv"),
            prompt_idx=args.prompt_idx,
            prompt=args.prompt,
        )
        DFS_Attention.dump_sparsity_records(os.path.join(output_dir, "sparsity_records.csv"))

    export_to_video(output, args.output_file, fps=24)

    subblock_profile_path = DFS_Attention.dump_subblock_retention_profile()
    if subblock_profile_path is not None:
        logger.info("Sub-block retention histogram saved to {}", subblock_profile_path)

    
