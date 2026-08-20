"""
Analyze single dense-step scanning experiment results.

For each prompt, computes PSNR / SSIM / LPIPS of each variant against all_dense,
and produces per-prompt + averaged summary tables.
"""

import os
import argparse
import numpy as np
import cv2
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)  # (T, H, W, 3) uint8


def compute_psnr(ref, tgt):
    return psnr(ref.astype(np.float64), tgt.astype(np.float64), data_range=255)


def compute_ssim(ref, tgt):
    scores = []
    for r, t in zip(ref, tgt):
        scores.append(ssim(r, t, channel_axis=2, data_range=255))
    return np.mean(scores)


def compute_lpips(lpips_model, ref, tgt, device):
    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float().div(255).to(device)  # (T, 3, H, W)
    tgt_t = torch.from_numpy(tgt).permute(0, 3, 1, 2).float().div(255).to(device)
    with torch.no_grad():
        scores = lpips_model(ref_t, tgt_t)
    return scores.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/scan_single_dense")
    parser.add_argument("--prompt_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--step_interval", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device)

    dense_steps = list(range(0, args.num_inference_steps, args.step_interval))
    variants = ["all_sparse"] + [f"dense_t{t}" for t in dense_steps]

    all_results = {}

    for pid in args.prompt_indices:
        prompt_dir = os.path.join(args.results_dir, f"prompt_{pid}")
        ref_path = os.path.join(prompt_dir, "all_dense.mp4")
        ref_frames = read_video_frames(ref_path)
        print(f"\nPrompt {pid}: reference {ref_frames.shape}")

        results = {}
        for variant in variants:
            tgt_path = os.path.join(prompt_dir, f"{variant}.mp4")
            tgt_frames = read_video_frames(tgt_path)

            v_psnr = compute_psnr(ref_frames, tgt_frames)
            v_ssim = compute_ssim(ref_frames, tgt_frames)
            v_lpips = compute_lpips(lpips_model, ref_frames, tgt_frames, device)

            results[variant] = {"PSNR": v_psnr, "SSIM": v_ssim, "LPIPS": v_lpips}

        all_results[pid] = results

    # --- Print per-prompt tables ---
    header = f"{'Variant':<16} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}"
    sep = "-" * 44

    for pid in sorted(all_results.keys()):
        print(f"\n{'='*44}")
        print(f"  Prompt {pid}")
        print(f"{'='*44}")
        print(header)
        print(sep)
        for variant in variants:
            r = all_results[pid][variant]
            print(f"{variant:<16} {r['PSNR']:>8.2f} {r['SSIM']:>8.4f} {r['LPIPS']:>8.4f}")

    # --- Averaged summary ---
    print(f"\n{'='*60}")
    print("  AVERAGED ACROSS PROMPTS")
    print(f"{'='*60}")
    print(f"{'Variant':<16} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 44)

    avg_results = {}
    for variant in variants:
        avg_psnr = np.mean([all_results[pid][variant]["PSNR"] for pid in all_results])
        avg_ssim = np.mean([all_results[pid][variant]["SSIM"] for pid in all_results])
        avg_lpips = np.mean([all_results[pid][variant]["LPIPS"] for pid in all_results])
        avg_results[variant] = {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}
        print(f"{variant:<16} {avg_psnr:>8.2f} {avg_ssim:>8.4f} {avg_lpips:>8.4f}")

    # --- Recovery gain relative to all_sparse ---
    print(f"\n{'='*60}")
    print("  RECOVERY GAIN vs ALL_SPARSE (positive = better)")
    print(f"{'='*60}")
    print(f"{'Variant':<16} {'ΔPSNR':>8} {'ΔSSIM':>8} {'ΔLPIPS':>8}")
    print("-" * 44)

    base = avg_results["all_sparse"]
    for variant in variants:
        if variant == "all_sparse":
            continue
        d_psnr = avg_results[variant]["PSNR"] - base["PSNR"]
        d_ssim = avg_results[variant]["SSIM"] - base["SSIM"]
        d_lpips = base["LPIPS"] - avg_results[variant]["LPIPS"]  # lower LPIPS = better
        print(f"{variant:<16} {d_psnr:>+8.2f} {d_ssim:>+8.4f} {d_lpips:>+8.4f}")

    print()


if __name__ == "__main__":
    main()
