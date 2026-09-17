#!/usr/bin/env python3
"""Evaluate the sampled-LSE suite against matching dense VBench videos."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


METRIC_COLUMNS = ("num_frames", "psnr_mean", "ssim_mean", "lpips_mean")
DENSITY_COLUMN = "mean_final_density_sparse_steps_avg_over_layers"


def prompt_references(prompt_file: Path, dense_dir: Path) -> dict[int, Path]:
    references = {}
    for prompt_idx, line in enumerate(prompt_file.read_text(encoding="utf-8").splitlines()):
        category, category_idx, _ = line.split("|", 2)
        references[prompt_idx] = dense_dir / f"vbench_{category}_{category_idx}.mp4"
    return references


def read_density(config_dir: Path) -> dict[int, float]:
    with (config_dir / "density_summary.csv").open(newline="", encoding="utf-8") as handle:
        return {
            int(row["prompt_idx"]): float(row[DENSITY_COLUMN])
            for row in csv.DictReader(handle)
        }


def read_time(config_dir: Path, prompt_idx: int) -> tuple[float, float]:
    timing_path = config_dir / f"{prompt_idx}_timing.csv"
    with timing_path.open(newline="", encoding="utf-8") as handle:
        phases = {row["phase"]: row for row in csv.DictReader(handle)}
    return (
        float(phases["e2e_generation_wall"]["total_ms"]) / 1000.0,
        float(phases["e2e_generation_gpu"]["total_ms"]) / 1000.0,
    )


def evaluate_one(
    python: str, metric_script: Path, reference: Path, video: Path
) -> dict[str, float]:
    result = subprocess.run(
        [
            python,
            str(metric_script),
            "--reference",
            str(reference),
            "--video",
            str(video),
            "--device",
            "cuda",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    parsed = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in METRIC_COLUMNS:
            parsed[key] = float(value.strip())
    missing = set(METRIC_COLUMNS) - parsed.keys()
    if missing:
        raise RuntimeError(f"Missing {sorted(missing)} in metric output for {video}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path(
            "/cnic/work/liutt/mywork/attention_time/res/"
            "hymor_sampled_lse_suite_20260917"
        ),
    )
    args = parser.parse_args()

    metric_script = Path(
        "/cnic/work/liutt/mywork/attention/Sparse-VideoGen/metric/videometric.py"
    )
    prompt_root = Path(
        "/cnic/work/liutt/mywork/attention/Sparse-VideoGen/data/vbench_data"
    )
    dense_root = Path("/cnic/work/liutt/mywork/attention/ttresult/vbench/t2v")
    dataset_info = {
        33: (
            prompt_root / "vbench_33_prompts.txt",
            dense_root / "dense/Step_50-Res_480p",
        ),
        66: (
            prompt_root / "vbench_66_prompts.txt",
            dense_root / "densep66/Step_50-Res_480p",
        ),
    }
    references = {
        dataset: prompt_references(prompt_file, dense_dir)
        for dataset, (prompt_file, dense_dir) in dataset_info.items()
    }

    config_dirs = sorted(path.parent for path in args.suite_root.rglob("density_summary.csv"))
    jobs = []
    metadata = {}
    metric_results = {}
    for config_dir in config_dirs:
        dataset = 33 if "vbench_33" in config_dir.parts else 66
        densities = read_density(config_dir)
        cached = {}
        detail_path = config_dir / "video_metrics.csv"
        if detail_path.is_file():
            with detail_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    prompt_idx = int(row["prompt_idx"])
                    if Path(row["reference"]) == references[dataset][prompt_idx]:
                        cached[str(config_dir / f"{prompt_idx}.mp4")] = {
                            key: float(row[key]) for key in METRIC_COLUMNS
                        }
        for video in sorted(config_dir.glob("*.mp4"), key=lambda path: int(path.stem)):
            prompt_idx = int(video.stem)
            reference = references[dataset][prompt_idx]
            if not reference.is_file():
                raise FileNotFoundError(reference)
            wall_s, gpu_s = read_time(config_dir, prompt_idx)
            key = str(video)
            metadata[key] = {
                "config_dir": config_dir,
                "dataset": dataset,
                "prompt_idx": prompt_idx,
                "video": video,
                "reference": reference,
                "density": densities[prompt_idx],
                "e2e_wall_s": wall_s,
                "e2e_gpu_s": gpu_s,
            }
            if key in cached:
                metric_results[key] = cached[key]
            else:
                jobs.append((key, reference, video))

    print(
        f"Evaluating {len(jobs)} videos and reusing {len(metric_results)} cached "
        f"results across {len(config_dirs)} configurations"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, sys.executable, metric_script, reference, video): key
            for key, reference, video in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            metric_results[key] = future.result()
            print(f"[{completed}/{len(jobs)}] {metadata[key]['video']}", flush=True)

    summary_rows = []
    detail_fields = (
        "dataset", "prompt_idx", "video", "reference", "density",
        "e2e_wall_s", "e2e_gpu_s", *METRIC_COLUMNS,
    )
    for config_dir in config_dirs:
        rows = []
        for key, item in metadata.items():
            if item["config_dir"] != config_dir:
                continue
            row = {field: item[field] for field in detail_fields if field in item}
            row.update(metric_results[key])
            rows.append(row)
        rows.sort(key=lambda row: int(row["prompt_idx"]))
        detail_path = config_dir / "video_metrics.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=detail_fields)
            writer.writeheader()
            writer.writerows(rows)

        count = len(rows)
        relative = config_dir.relative_to(args.suite_root)
        summary_rows.append({
            "configuration": str(relative),
            "dataset": rows[0]["dataset"],
            "num_videos": count,
            "mean_density": sum(float(row["density"]) for row in rows) / count,
            "mean_e2e_wall_s": sum(float(row["e2e_wall_s"]) for row in rows) / count,
            "mean_e2e_gpu_s": sum(float(row["e2e_gpu_s"]) for row in rows) / count,
            "mean_psnr": sum(float(row["psnr_mean"]) for row in rows) / count,
            "mean_ssim": sum(float(row["ssim_mean"]) for row in rows) / count,
            "mean_lpips": sum(float(row["lpips_mean"]) for row in rows) / count,
        })

    summary_path = args.suite_root / "video_metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
