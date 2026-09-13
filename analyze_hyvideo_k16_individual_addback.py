#!/usr/bin/env python3
"""Record exact independent K16 add-backs for Hunyuan Q128 x K96 Core.

The input is one or more Core-only ``attention_debug`` dumps.  For every
sampled (head, Q16), the analyzer recomputes Dense and Core attention from the
saved Q/K/V, then independently restores each K16 belonging to every omitted
K96 macro.  Outputs intentionally follow the curated Wan K16-DeltaE schema,
except that a K96 macro contains six rather than eight K16 tiles.

Independent DeltaE values are not additive because every add-back changes the
softmax denominator.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from analyze_macro_topk_error import (
    K_MACRO,
    MICRO,
    Q_MACRO,
    _as_hsd,
    _ceil_div,
    _load_payload,
    _make_video_perm,
    fixed_macro_topk,
    permute_qkv,
)
from dfsattn.flashinfer64_attention import (
    K_MICROS_PER_MACRO,
    Q_MICROS_PER_MACRO,
    aggregate_macro_scores,
    compute_micro_tile_scores,
)


EPS = 1.0e-20

EQ_FIELDS = [
    "step", "layer", "head", "q16", "q_block", "q16_offset",
    "token_start", "token_end", "eq",
]

DETAIL_FIELDS = [
    "step", "layer", "head", "sample_type", "q16", "q_block",
    "q16_offset", "kv_macro", "k16_local", "k16_global",
    "k_token_start", "k_token_end", "eq_scan", "eq_core_direct",
    "eq_full_macro", "delta_e_macro", "eq_k16", "delta_e_k16",
    "recovery_ratio_vs_macro", "dense_tile_mass", "proxy_tile_score",
    "delta_rank_within_macro", "dense_mass_rank_within_macro",
    "proxy_rank_within_macro",
]

MACRO_SUMMARY_FIELDS = [
    "step", "layer", "head", "sample_type", "q16", "q_block",
    "q16_offset", "kv_macro", "eq_scan", "eq_core_direct",
    "eq_full_macro", "delta_e_macro", "k16_count", "positive_k16_count",
    "best_delta_k16_local", "best_delta_k16_global", "best_delta_e_k16",
    "best_delta_eq_k16", "best_delta_recovery_ratio",
    "dense_top_k16_local", "dense_top_k16_global", "dense_top_delta_e_k16",
    "dense_top_eq_k16", "dense_top_recovery_ratio",
    "proxy_top_k16_local", "proxy_top_k16_global", "proxy_top_delta_e_k16",
    "proxy_top_eq_k16", "proxy_top_recovery_ratio",
]


def _open_csv_gz(path: Path, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6)
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def _rank_desc(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device)
    return ranks


def _relative_error(output: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, output.ndim))
    return torch.linalg.vector_norm(output - dense, dim=dims) / (
        torch.linalg.vector_norm(dense, dim=dims) + EPS
    )


def _q16_errors(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    heads, sequence, dim = sparse.shape
    padded = _ceil_div(sequence, MICRO) * MICRO
    if padded != sequence:
        sparse = F.pad(sparse, (0, 0, 0, padded - sequence))
        dense = F.pad(dense, (0, 0, 0, padded - sequence))
    sparse = sparse.float().view(heads, -1, MICRO, dim)
    dense = dense.float().view(heads, -1, MICRO, dim)
    return torch.linalg.vector_norm(sparse - dense, dim=(-2, -1)) / (
        torch.linalg.vector_norm(dense, dim=(-2, -1)) + 1.0e-8
    )


def _group_statistics(
    weights: torch.Tensor, values: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return group denominator [Q,G] and numerator [G,Q,D]."""
    queries, sequence = weights.shape
    dim = values.shape[-1]
    padded = _ceil_div(sequence, group_size) * group_size
    if padded != sequence:
        weights = F.pad(weights, (0, padded - sequence))
        values = F.pad(values, (0, 0, 0, padded - sequence))
    groups = padded // group_size
    grouped_w = weights.view(queries, groups, group_size)
    grouped_v = values.view(groups, group_size, dim)
    z = grouped_w.sum(-1)
    n = torch.einsum("qgi,gid->gqd", grouped_w, grouped_v)
    return z, n


def _errors_after_groups(
    core_z: torch.Tensor,
    core_n: torch.Tensor,
    group_z: torch.Tensor,
    group_n: torch.Tensor,
    dense_output: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    errors = []
    for start in range(0, group_n.shape[0], batch_size):
        stop = min(start + batch_size, group_n.shape[0])
        denominator = core_z[None, :, None] + group_z[:, start:stop].transpose(0, 1)[..., None]
        output = (core_n[None] + group_n[start:stop]) / denominator.clamp_min(EPS)
        errors.append(_relative_error(output, dense_output[None]))
    return torch.cat(errors)


@torch.no_grad()
def compute_k16_addbacks(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    v_head: torch.Tensor,
    core_row: torch.Tensor,
    q16_id: int,
    candidate_batch: int = 128,
) -> dict[str, torch.Tensor | float | int]:
    """Compute exact Core, K96 and independent K16 errors for one Q16."""
    sequence, dim = q_head.shape
    q_start = q16_id * MICRO
    q_end = min(q_start + MICRO, sequence)
    if q_start >= sequence:
        raise ValueError(f"q16 {q16_id} is outside sequence {sequence}")
    if core_row.numel() != _ceil_div(sequence, K_MACRO):
        raise ValueError("Core row and K96 geometry disagree")
    if not bool(core_row.any()):
        raise ValueError("Core row is empty")

    queries = q_head[q_start:q_end].float()
    keys = k_head.float()
    values = v_head.float()
    logits = queries @ keys.transpose(0, 1) * (dim ** -0.5)
    weights = torch.exp(logits - logits.max(-1, keepdim=True).values)

    dense_z = weights.sum(-1)
    dense_n = weights @ values
    dense_output = dense_n / dense_z[:, None].clamp_min(EPS)

    macro_z, macro_n = _group_statistics(weights, values, K_MACRO)
    core_z = macro_z[:, core_row].sum(-1)
    core_n = macro_n[core_row].sum(0)
    core_output = core_n / core_z[:, None].clamp_min(EPS)
    core_error = float(_relative_error(core_output[None], dense_output[None])[0])
    macro_errors = _errors_after_groups(
        core_z, core_n, macro_z, macro_n, dense_output, candidate_batch
    )

    tile_z, tile_n = _group_statistics(weights, values, MICRO)
    tile_errors = _errors_after_groups(
        core_z, core_n, tile_z, tile_n, dense_output, candidate_batch
    )
    tile_delta = core_error - tile_errors
    dense_tile_mass = (tile_z / dense_z[:, None].clamp_min(EPS)).mean(0)

    key_means = []
    for start in range(0, sequence, MICRO):
        key_means.append(keys[start:min(start + MICRO, sequence)].mean(0))
    key_means = torch.stack(key_means)
    proxy_scores = torch.softmax(
        queries.mean(0) @ key_means.transpose(0, 1) * (dim ** -0.5), dim=-1
    )
    return {
        "q_block": q_start // Q_MACRO,
        "q16_offset": q16_id % Q_MICROS_PER_MACRO,
        "eq_core_direct": core_error,
        "macro_errors": macro_errors,
        "tile_errors": tile_errors,
        "tile_delta": tile_delta,
        "dense_tile_mass": dense_tile_mass,
        "proxy_tile_score": proxy_scores,
    }


def _choose_samples(
    errors: torch.Tensor,
    analyzable: torch.Tensor,
    top_error: int,
    controls: int,
    seed: int,
) -> list[tuple[int, int, str]]:
    heads, q16_count = errors.shape
    valid_flat = analyzable.flatten().nonzero(as_tuple=False).flatten()
    if not valid_flat.numel():
        return []
    valid_errors = errors.flatten()[valid_flat]
    labels: dict[tuple[int, int], set[str]] = {}

    count = min(max(0, top_error), valid_flat.numel())
    if count:
        chosen = valid_flat[torch.topk(valid_errors, count).indices]
        for index in chosen.tolist():
            labels.setdefault((index // q16_count, index % q16_count), set()).add("high_error")

    count = min(max(0, controls), valid_flat.numel())
    if count:
        order = torch.argsort(valid_errors)
        pool = valid_flat[order[:max(count, valid_flat.numel() // 2)]].cpu().numpy()
        rng = np.random.default_rng(seed)
        for index in rng.choice(pool, size=count, replace=False).tolist():
            labels.setdefault((index // q16_count, index % q16_count), set()).add("low_error_control")

    return [
        (head, q16, "+".join(sorted(kinds)))
        for (head, q16), kinds in sorted(labels.items())
    ]


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    if not array.size:
        return {}
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _write_progress(path: Path, **payload: Any) -> None:
    payload["updated_at_unix"] = time.time()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _snapshot_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("step*_layer*_flashinfer64_topk_topp.pt"))


def run(args: argparse.Namespace) -> None:
    resolved_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(resolved_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    snapshots = _snapshot_files(args.input_root)
    if not snapshots:
        raise FileNotFoundError(f"No Core-only topk_topp dumps under {args.input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "analysis_progress.json"
    detail_handle, detail_writer = _open_csv_gz(
        args.output_dir / "k16_deltaE_all.csv.gz", DETAIL_FIELDS
    )
    macro_handle, macro_writer = _open_csv_gz(
        args.output_dir / "k96_k16_summary.csv.gz", MACRO_SUMMARY_FIELDS
    )
    eq_handle, eq_writer = _open_csv_gz(args.output_dir / "eq_all.csv.gz", EQ_FIELDS)

    started = time.time()
    detail_rows = macro_rows = positive_k16 = sample_rows = 0
    best_ratios: list[float] = []
    dense_ratios: list[float] = []
    proxy_ratios: list[float] = []
    dense_matches = proxy_matches = 0
    top_heap: list[tuple[float, int, dict[str, Any]]] = []
    serial = 0
    snapshot_summaries = []
    try:
        for snapshot_index, path in enumerate(snapshots, 1):
            payload = _load_payload(path)
            metadata = payload.get("metadata", {})
            if metadata.get("flashinfer64_core_only") is not True:
                raise ValueError(f"{path} is not a Core-only dump")
            if metadata.get("flashinfer64_route_mode") != "topk_topp":
                raise ValueError(f"{path} is not a topk_topp dump")
            ratio = float(metadata.get("flashinfer64_tile_top_ratio", args.macro_top_ratio))
            if abs(ratio - args.macro_top_ratio) > 1e-7:
                raise ValueError(f"{path}: ratio {ratio} != requested {args.macro_top_ratio}")

            q_orig = _as_hsd(payload["query"], "query").to(device)
            k_orig = _as_hsd(payload["key"], "key").to(device)
            v_orig = _as_hsd(payload["value"], "value").to(device)
            sparse_orig = _as_hsd(payload["sparse_output"], "sparse_output").to(device)
            dense_orig = _as_hsd(payload["dense_output"], "dense_output").to(device)
            heads, full_sequence, _ = q_orig.shape
            video_len = int(metadata.get("video_len", full_sequence))
            valid_sequence = int(metadata.get("valid_sequence", full_sequence))
            perm = _make_video_perm(
                args.order, args.num_frames, args.height, args.width, video_len, device
            )
            q, k, v, permutation, _ = permute_qkv(
                q_orig, k_orig, v_orig, perm, video_len
            )
            q, k, v = q[:, :valid_sequence], k[:, :valid_sequence], v[:, :valid_sequence]
            sparse = sparse_orig[:, permutation][:, :valid_sequence]
            dense = dense_orig[:, permutation][:, :valid_sequence]
            errors = _q16_errors(sparse, dense).cpu()

            micro_scores = compute_micro_tile_scores(q, k)
            macro_scores = aggregate_macro_scores(micro_scores)
            core = fixed_macro_topk(macro_scores, ratio, valid_sequence, video_len)
            del micro_scores, macro_scores

            q16_count = errors.shape[1]
            q_parent = torch.arange(q16_count) // Q_MICROS_PER_MACRO
            analyzable_rows = core.detach().cpu()[:, q_parent].logical_not().any(-1)
            samples = _choose_samples(
                errors, analyzable_rows, args.top_error, args.controls,
                args.random_seed + int(metadata.get("step_idx", 0)) * 100
                + int(metadata.get("layer_idx", 0)),
            )
            step = int(metadata.get("step_idx", -1))
            layer = int(metadata.get("layer_idx", -1))
            for head in range(heads):
                for q16 in range(q16_count):
                    start = q16 * MICRO
                    eq_writer.writerow({
                        "step": step, "layer": layer, "head": head,
                        "q16": q16, "q_block": q16 // Q_MICROS_PER_MACRO,
                        "q16_offset": q16 % Q_MICROS_PER_MACRO,
                        "token_start": start,
                        "token_end": min(start + MICRO, valid_sequence),
                        "eq": float(errors[head, q16]),
                    })

            for sample_index, (head, q16, sample_type) in enumerate(samples, 1):
                q_macro = q16 // Q_MICROS_PER_MACRO
                result = compute_k16_addbacks(
                    q[head], k[head], v[head], core[head, q_macro], q16,
                    candidate_batch=args.candidate_batch,
                )
                eq_scan = float(errors[head, q16])
                eq_core = float(result["eq_core_direct"])
                omitted = (~core[head, q_macro]).nonzero(as_tuple=False).flatten().tolist()
                for kv_macro in omitted:
                    first = kv_macro * K_MICROS_PER_MACRO
                    stop = min(first + K_MICROS_PER_MACRO, result["tile_errors"].numel())
                    local_errors = result["tile_errors"][first:stop]
                    local_delta = result["tile_delta"][first:stop]
                    local_dense = result["dense_tile_mass"][first:stop]
                    local_proxy = result["proxy_tile_score"][first:stop]
                    macro_error = float(result["macro_errors"][kv_macro])
                    macro_delta = eq_core - macro_error
                    recovery = (
                        local_delta / macro_delta if abs(macro_delta) > 1.0e-12
                        else torch.full_like(local_delta, float("nan"))
                    )
                    best_local = int(torch.argmax(local_delta))
                    dense_local = int(torch.argmax(local_dense))
                    proxy_local = int(torch.argmax(local_proxy))
                    dense_matches += dense_local == best_local
                    proxy_matches += proxy_local == best_local
                    if macro_delta > 1.0e-12:
                        best_ratios.append(float(recovery[best_local]))
                        dense_ratios.append(float(recovery[dense_local]))
                        proxy_ratios.append(float(recovery[proxy_local]))
                    base = {
                        "step": step, "layer": layer, "head": head,
                        "sample_type": sample_type, "q16": q16,
                        "q_block": q_macro,
                        "q16_offset": q16 % Q_MICROS_PER_MACRO,
                        "kv_macro": kv_macro, "eq_scan": eq_scan,
                        "eq_core_direct": eq_core,
                        "eq_full_macro": macro_error,
                        "delta_e_macro": macro_delta,
                    }
                    delta_ranks = _rank_desc(local_delta)
                    dense_ranks = _rank_desc(local_dense)
                    proxy_ranks = _rank_desc(local_proxy)
                    for local in range(stop - first):
                        global_k16 = first + local
                        delta_value = float(local_delta[local])
                        positive_k16 += delta_value > 0
                        row = {
                            **base, "k16_local": local,
                            "k16_global": global_k16,
                            "k_token_start": global_k16 * MICRO,
                            "k_token_end": min((global_k16 + 1) * MICRO, valid_sequence),
                            "eq_k16": float(local_errors[local]),
                            "delta_e_k16": delta_value,
                            "recovery_ratio_vs_macro": float(recovery[local]),
                            "dense_tile_mass": float(local_dense[local]),
                            "proxy_tile_score": float(local_proxy[local]),
                            "delta_rank_within_macro": int(delta_ranks[local]),
                            "dense_mass_rank_within_macro": int(dense_ranks[local]),
                            "proxy_rank_within_macro": int(proxy_ranks[local]),
                        }
                        detail_writer.writerow(row)
                        detail_rows += 1
                        serial += 1
                        item = (delta_value, serial, row)
                        if len(top_heap) < args.top_cases:
                            heapq.heappush(top_heap, item)
                        elif delta_value > top_heap[0][0]:
                            heapq.heapreplace(top_heap, item)
                    macro_writer.writerow({
                        **base, "k16_count": stop - first,
                        "positive_k16_count": int((local_delta > 0).sum()),
                        "best_delta_k16_local": best_local,
                        "best_delta_k16_global": first + best_local,
                        "best_delta_e_k16": float(local_delta[best_local]),
                        "best_delta_eq_k16": float(local_errors[best_local]),
                        "best_delta_recovery_ratio": float(recovery[best_local]),
                        "dense_top_k16_local": dense_local,
                        "dense_top_k16_global": first + dense_local,
                        "dense_top_delta_e_k16": float(local_delta[dense_local]),
                        "dense_top_eq_k16": float(local_errors[dense_local]),
                        "dense_top_recovery_ratio": float(recovery[dense_local]),
                        "proxy_top_k16_local": proxy_local,
                        "proxy_top_k16_global": first + proxy_local,
                        "proxy_top_delta_e_k16": float(local_delta[proxy_local]),
                        "proxy_top_eq_k16": float(local_errors[proxy_local]),
                        "proxy_top_recovery_ratio": float(recovery[proxy_local]),
                    })
                    macro_rows += 1
                sample_rows += 1
                if sample_index % 8 == 0 or sample_index == len(samples):
                    _write_progress(
                        progress_path, status="running", snapshot=path.name,
                        snapshot_index=snapshot_index, snapshots_total=len(snapshots),
                        sample_completed=sample_index, sample_total=len(samples),
                        detail_rows=detail_rows, elapsed_seconds=time.time() - started,
                    )
            snapshot_summaries.append({
                "path": str(path), "step": step, "layer": layer,
                "heads": heads, "q16_per_head": q16_count,
                "samples": len(samples), "core_density": float(core.float().mean()),
                "eq_mean": float(errors.mean()), "eq_max": float(errors.max()),
            })
            del q, k, v, sparse, dense, core, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        detail_handle.close(); macro_handle.close(); eq_handle.close()

    summary = {
        "schema_version": 1,
        "model": "HunyuanVideo",
        "macro_geometry": "Q128xK96",
        "micro_geometry": "Q16xK16",
        "k16_rows_per_full_macro": K_MICROS_PER_MACRO,
        "macro_top_ratio": args.macro_top_ratio,
        "snapshot_count": len(snapshots), "sample_q16_count": sample_rows,
        "macro_output_rows": macro_rows, "k16_detail_rows": detail_rows,
        "positive_k16_count": positive_k16,
        "positive_k16_fraction": positive_k16 / detail_rows if detail_rows else 0.0,
        "best_delta_recovery_ratio": _quantiles(best_ratios),
        "dense_mass_top_recovery_ratio": _quantiles(dense_ratios),
        "proxy_top_recovery_ratio": _quantiles(proxy_ratios),
        "dense_mass_top_matches_best_delta_fraction": dense_matches / macro_rows if macro_rows else 0.0,
        "proxy_top_matches_best_delta_fraction": proxy_matches / macro_rows if macro_rows else 0.0,
        "snapshots": snapshot_summaries,
        "top_cases": [item[2] for item in sorted(top_heap, reverse=True)],
        "elapsed_seconds": time.time() - started,
        "detail_csv": str(args.output_dir / "k16_deltaE_all.csv.gz"),
        "macro_summary_csv": str(args.output_dir / "k96_k16_summary.csv.gz"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    _write_progress(
        progress_path, status="complete", stage="complete", percent=100.0,
        detail_rows=detail_rows, elapsed_seconds=summary["elapsed_seconds"],
    )
    print(json.dumps({
        "status": "complete", "snapshot_count": len(snapshots),
        "sample_q16_count": sample_rows, "macro_output_rows": macro_rows,
        "k16_detail_rows": detail_rows, "elapsed_seconds": summary["elapsed_seconds"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--macro-top-ratio", type=float, default=0.20)
    parser.add_argument("--order", default="hilbert3d")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=129)
    parser.add_argument("--top-error", type=int, default=64)
    parser.add_argument("--controls", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--candidate-batch", type=int, default=128)
    parser.add_argument("--top-cases", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
