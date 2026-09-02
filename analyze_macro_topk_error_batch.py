"""Run macro Top-k error analysis for one debug dump per prompt.

The input root may contain ``prompt_*/.../*.pt`` files produced by
``hyvideo_t2v_720p_dfs.sh`` with ``ATTENTION_DEBUG_DIR`` enabled.  Only dumps
whose metadata says ``flashinfer64_core_only=true`` are accepted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--macro-top-ratio", type=float, default=0.2)
    p.add_argument("--order", default="hilbert3d")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=129)
    p.add_argument("--q-macro-count", type=int, default=8)
    p.add_argument("--candidate-batch", type=int, default=32)
    p.add_argument("--heads", type=str, default=None, help="comma-separated original head ids")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.input_root.rglob("*.pt"))
    selected = []
    for path in files:
        try:
            payload = torch.load(path, map_location="meta", weights_only=False)
            metadata = payload.get("metadata", {})
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        if metadata.get("flashinfer64_core_only") is not True:
            print(f"[skip] {path}: not a flashinfer64 Core-only dump")
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            f"no flashinfer64_core_only dump found under {args.input_root}"
        )

    summaries = []
    analyzer = Path(__file__).with_name("analyze_macro_topk_error.py")
    for path in selected:
        prompt_name = path.relative_to(args.input_root).parts[0] if len(path.relative_to(args.input_root).parts) > 1 else path.stem
        output = args.output_dir / prompt_name
        command = [
            sys.executable, str(analyzer),
            "--input-dump", str(path), "--base-dump", str(path),
            "--output-dir", str(output),
            "--macro-top-ratio", str(args.macro_top_ratio),
            "--order", args.order,
            "--height", str(args.height), "--width", str(args.width),
            "--num-frames", str(args.num_frames),
            "--q-macro-count", str(args.q_macro_count),
            "--candidate-batch", str(args.candidate_batch),
        ]
        if args.heads is not None:
            command.extend(["--heads", args.heads])
        subprocess.run(command, check=True)
        summaries.append(json.loads((output / "summary.json").read_text(encoding="utf-8")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "batch_summary.json").write_text(
        json.dumps({"num_videos": len(summaries), "videos": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"num_videos": len(summaries), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
