"""Diagnose errors caused by coarse Q128 x K96 macro Top-k selection.

The script is intentionally offline.  It consumes an attention debug dump from
``hyvideo_t2v_inference.py`` and reproduces the selector used by
``FlashInfer64Attention``.  The recommended workflow is:

1. generate a debug dump with ``FLASHINFER64_CORE_ONLY=True``,
   ``FLASHINFER64_ROUTE_MODE=topk_topp``,
   ``FLASHINFER64_TILE_TOP_RATIO=0.2`` and
   ``FLASHINFER64_TOKEN_TOP_P=0``;
2. pass that dump as ``--base-dump`` and analyze the highest-error Q16/Q128
   regions;
3. inspect ``macro_addback.csv`` and
   ``macro_addback_entropy_max.png``.

The add-back computation is exact for the selected query macro: it recomputes
the softmax over Core + one omitted macro, including the changed denominator.
Only the query macro containing the candidate can change, which makes the
calculation substantially smaller than rerunning the whole attention matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dfsattn.flashinfer64_attention import (  # noqa: E402
    K_MACRO,
    MICRO,
    Q_MACRO,
    Q_MICROS_PER_MACRO,
    K_MICROS_PER_MACRO,
    _ceil_div,
    _select_hyvideo_core_tiles,
    aggregate_macro_scores,
    compute_micro_tile_scores,
)


EPS = 1.0e-20


def _load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"query", "key", "value", "dense_output"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{path} is missing tensors: {sorted(missing)}")
    return payload


def _as_hsd(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, heads, sequence, dim], got {tuple(tensor.shape)}")
    return tensor[0].contiguous()


def _make_video_perm(order: str, num_frames: int, height: int, width: int, video_len: int,
                     device: torch.device) -> torch.Tensor:
    if order == "org":
        perm = torch.arange(video_len, dtype=torch.long)
    else:
        from dfsattn.utils.order import block3d_perm, fwh, hilbert2d_perm, hilbert3d_perm, hwf

        latent_f = num_frames // 4 + 1
        latent_h, latent_w = height // 16, width // 16
        expected = latent_f * latent_h * latent_w
        if expected != video_len:
            raise ValueError(
                "geometry does not reproduce video_len: "
                f"{latent_f}*{latent_h}*{latent_w}={expected}, expected {video_len}"
            )
        if order == "hilbert2d":
            perm = hilbert2d_perm(latent_f, latent_h, latent_w)
        elif order == "blk":
            perm = block3d_perm(latent_f, latent_h, latent_w, a=4, b=4, c=4)
        elif order == "hwf":
            perm = hwf(latent_f, latent_h, latent_w)
        elif order == "fwh":
            perm = fwh(latent_f, latent_h, latent_w)
        elif order == "hilbert3d":
            perm = hilbert3d_perm(latent_f, latent_h, latent_w)
        else:
            raise ValueError(f"unknown order {order!r}")
        perm = perm.to(dtype=torch.long)
    if video_len < 0 or video_len > perm.numel():
        raise ValueError(f"invalid video_len {video_len}")
    return perm.to(device)


def permute_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                video_perm: torch.Tensor, video_len: Optional[int]) -> tuple[torch.Tensor, ...]:
    sequence = q.shape[1]
    if video_len is None or video_len == sequence:
        perm = video_perm
    else:
        perm = torch.cat((video_perm, torch.arange(video_len, sequence, device=q.device)))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(sequence, device=q.device)
    return q[:, perm].contiguous(), k[:, perm].contiguous(), v[:, perm].contiguous(), perm, inverse


def macro_structure(micro_scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return M, A, H, C for every [head, Q128, K96] macro tile.

    The 48-way normalization follows the experiment definition, including
    zero-padded boundary microtiles.  ``valid_microtiles`` is emitted so that
    boundary effects can be filtered in downstream analyses if desired.
    """
    heads, q_micro, k_micro = micro_scores.shape
    q_pad = _ceil_div(q_micro, Q_MICROS_PER_MACRO) * Q_MICROS_PER_MACRO
    k_pad = _ceil_div(k_micro, K_MICROS_PER_MACRO) * K_MICROS_PER_MACRO
    padded = F.pad(micro_scores, (0, k_pad - k_micro, 0, q_pad - q_micro))
    grouped = padded.view(
        heads, q_pad // Q_MICROS_PER_MACRO, Q_MICROS_PER_MACRO,
        k_pad // K_MICROS_PER_MACRO, K_MICROS_PER_MACRO,
    )
    mass = grouped.sum(dim=(2, 4))
    peak = grouped.amax(dim=(2, 4))
    safe_mass = mass.clamp_min(EPS)
    probs = grouped / safe_mass[:, :, None, :, None]
    entropy = -(probs * probs.clamp_min(EPS).log()).sum(dim=(2, 4)) / math.log(48)
    concentration = peak / safe_mass
    valid_q = (torch.arange(q_pad, device=micro_scores.device) < q_micro).view(
        1, q_pad // Q_MICROS_PER_MACRO, Q_MICROS_PER_MACRO, 1, 1
    )
    valid_k = (torch.arange(k_pad, device=micro_scores.device) < k_micro).view(
        1, 1, 1, k_pad // K_MICROS_PER_MACRO, K_MICROS_PER_MACRO
    )
    valid = (valid_q & valid_k).sum(dim=(2, 4)).expand(heads, -1, -1)
    return {
        "M_g": mass,
        "A_g": peak,
        "H_g": entropy,
        "C_g": concentration,
        "valid_microtiles": valid,
    }


def fixed_macro_topk(macro_scores: torch.Tensor, ratio: float,
                     sequence: int, video_len: Optional[int]) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("macro top-k ratio must be in (0, 1]")
    # top_p=1 is unused in topk_topp mode.  Keeping the production selector
    # here prevents this analysis from silently drifting from the runtime.
    return _select_hyvideo_core_tiles(
        macro_scores,
        route_mode="topk_topp",
        tile_top_p=1.0,
        tile_top_ratio=ratio,
        sequence=sequence,
        video_len=video_len,
    )


def _key_indices(k_macro: int, sequence: int, device: torch.device) -> torch.Tensor:
    start = k_macro * K_MACRO
    end = min(sequence, start + K_MACRO)
    return torch.arange(start, end, device=device, dtype=torch.long)


@torch.no_grad()
def attention_for_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       core_mask: torch.Tensor, q_macro_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact Core output/LSE for selected Q128 rows on CPU or CUDA."""
    heads, sequence, dim = q.shape
    output = torch.zeros((heads, sequence, dim), dtype=torch.float32, device=q.device)
    lse = torch.full((heads, sequence), -float("inf"), dtype=torch.float32, device=q.device)
    scale = dim ** -0.5
    for q_macro in q_macro_ids:
        q_start, q_end = q_macro * Q_MACRO, min(sequence, (q_macro + 1) * Q_MACRO)
        for head in range(heads):
            k_macros = core_mask[head, q_macro].nonzero(as_tuple=False).flatten().tolist()
            if not k_macros:
                continue
            key_ids = torch.cat([_key_indices(km, sequence, q.device) for km in k_macros])
            q_block = q[head, q_start:q_end].float()
            k_block = k[head, key_ids].float()
            v_block = v[head, key_ids].float()
            logits = q_block @ k_block.transpose(0, 1) * scale
            row_lse = torch.logsumexp(logits, dim=-1)
            weights = torch.exp(logits - row_lse[:, None])
            output[head, q_start:q_end] = weights @ v_block
            lse[head, q_start:q_end] = row_lse
    return output, lse


def _csv_writer(path: Path, fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


@torch.no_grad()
def addback_rows(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 dense: torch.Tensor, base_output: torch.Tensor, base_lse: torch.Tensor,
                 core_mask: torch.Tensor, structure: dict[str, torch.Tensor],
                 q_macro_ids: Sequence[int], analysis_q_mask: torch.Tensor,
                 candidate_batch: int = 16, head_ids: Optional[Sequence[int]] = None) -> list[dict]:
    """Measure exact one-macro add-back recovery for selected Q macros."""
    heads, sequence, dim = q.shape
    scale = dim ** -0.5
    dense_norm_global = torch.linalg.vector_norm(dense[analysis_q_mask]).item()
    base_diff_global = base_output[analysis_q_mask] - dense[analysis_q_mask]
    base_sq_global = float((base_diff_global * base_diff_global).sum().item())
    rows: list[dict] = []
    head_ids = list(range(heads)) if head_ids is None else list(head_ids)
    if len(head_ids) != heads:
        raise ValueError("head_ids must have one entry per analyzed head")
    for q_macro in q_macro_ids:
        q_start, q_end = q_macro * Q_MACRO, min(sequence, (q_macro + 1) * Q_MACRO)
        dense_block = dense[:, q_start:q_end]
        for head in range(heads):
            omitted = (~core_mask[head, q_macro]).nonzero(as_tuple=False).flatten().tolist()
            base_block = base_output[head, q_start:q_end]
            base_lse_block = base_lse[head, q_start:q_end]
            base_err = base_block - dense_block[head]
            base_sq_block = float((base_err * base_err).sum().item())
            dense_sq_block = float((dense_block[head] * dense_block[head]).sum().item())
            for start in range(0, len(omitted), candidate_batch):
                candidate_ids = omitted[start:start + candidate_batch]
                # Grouping by K length keeps the common full 96-token case
                # batched, while handling the final partial macro exactly.
                by_len: dict[int, list[int]] = {}
                for km in candidate_ids:
                    by_len.setdefault(int(_key_indices(km, sequence, q.device).numel()), []).append(km)
                for key_len, ids in by_len.items():
                    key_batches = torch.stack([_key_indices(km, sequence, q.device) for km in ids])
                    k_batch = k[head, key_batches].float()
                    v_batch = v[head, key_batches].float()
                    q_block = q[head, q_start:q_end].float()
                    logits = torch.einsum("qd,ckd->cqk", q_block, k_batch) * scale
                    candidate_lse = torch.logsumexp(logits, dim=-1)
                    weights = torch.exp(logits - candidate_lse[..., None])
                    candidate_out = torch.einsum("cqk,ckd->cqd", weights, v_batch)
                    new_lse = torch.logaddexp(base_lse_block[None], candidate_lse)
                    core_weight = torch.exp(base_lse_block[None] - new_lse)
                    candidate_weight = torch.exp(candidate_lse - new_lse)
                    new_out = (
                        core_weight[..., None] * base_block[None]
                        + candidate_weight[..., None] * candidate_out
                    )
                    new_err = new_out - dense_block[head][None]
                    new_sq_block = (new_err * new_err).sum(dim=(1, 2))
                    for idx, km in enumerate(ids):
                        new_sq = float(new_sq_block[idx].item())
                        new_global_sq = base_sq_global - base_sq_block + new_sq
                        old_l2 = math.sqrt(max(base_sq_global, 0.0))
                        new_l2 = math.sqrt(max(new_global_sq, 0.0))
                        old_block_l2 = math.sqrt(max(base_sq_block, 0.0))
                        new_block_l2 = math.sqrt(max(new_sq, 0.0))
                        rows.append({
                            "head": head_ids[head],
                            "q_macro": q_macro,
                            "k_macro": km,
                            "macro_score": float(structure["macro_score"][head, q_macro, km]),
                            "M_g": float(structure["M_g"][head, q_macro, km]),
                            "A_g": float(structure["A_g"][head, q_macro, km]),
                            "H_g": float(structure["H_g"][head, q_macro, km]),
                            "C_g": float(structure["C_g"][head, q_macro, km]),
                            "valid_microtiles": int(structure["valid_microtiles"][head, q_macro, km]),
                            "base_error_l2_qmacro": old_block_l2,
                            "addback_error_l2_qmacro": new_block_l2,
                            "delta_E_qmacro": old_block_l2 - new_block_l2,
                            "delta_E_relative_qmacro": (old_block_l2 - new_block_l2) / (math.sqrt(dense_sq_block) + EPS),
                            "delta_E_relative_global": (old_l2 - new_l2) / (dense_norm_global + EPS),
                        })
    return rows


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for pos in order[i:j]:
            ranks[pos] = rank
        i = j
    return ranks


def _corr(xs: Sequence[float], ys: Sequence[float], spearman: bool = False) -> float:
    if len(xs) < 2:
        return float("nan")
    x = list(map(float, xs)); y = list(map(float, ys))
    if spearman:
        x, y = _rank(x), _rank(y)
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    xn = math.sqrt(sum((v - xm) ** 2 for v in x)); yn = math.sqrt(sum((v - ym) ** 2 for v in y))
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / (xn * yn + EPS)


def write_scatter(path: Path, rows: Sequence[dict]) -> None:
    import matplotlib.pyplot as plt

    x = [r["H_g"] for r in rows]
    y = [r["A_g"] for r in rows]
    c = [r["delta_E_relative_qmacro"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    plot = ax.scatter(x, y, c=c, s=8, alpha=0.55, cmap="magma", linewidths=0)
    fig.colorbar(plot, ax=ax, label=r"add-back recovery $\Delta E_g$ (relative, Q macro)")
    ax.set_xlabel(r"Normalized entropy $H_g$")
    ax.set_ylabel(r"Maximum local importance $A_g$")
    ax.set_title("Macro Top-k error diagnosis: one-macro add-back")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_head_scatter(path: Path, rows: Sequence[dict]) -> None:
    """Plot the same relation separately for each analyzed attention head."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    head_ids = sorted({r["head"] for r in rows})
    if not head_ids:
        return
    values = [r["delta_E_relative_qmacro"] for r in rows]
    norm = Normalize(vmin=min(values), vmax=max(values))
    fig, axes = plt.subplots(
        1, len(head_ids), figsize=(5 * len(head_ids), 4.5),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]
    last_plot = None
    for ax, head in zip(axes, head_ids):
        head_rows = [r for r in rows if r["head"] == head]
        last_plot = ax.scatter(
            [r["H_g"] for r in head_rows],
            [r["A_g"] for r in head_rows],
            c=[r["delta_E_relative_qmacro"] for r in head_rows],
            norm=norm, cmap="magma", s=8, alpha=0.55, linewidths=0,
        )
        ax.set_title(f"head {head}")
        ax.grid(alpha=0.2)
        ax.set_xlabel(r"$H_g$")
    axes[0].set_ylabel(r"$A_g$")
    if last_plot is not None:
        fig.colorbar(last_plot, ax=axes.tolist(), label=r"$\Delta E_g$ (relative, Q macro)")
    fig.suptitle("Macro Top-k error diagnosis by attention head")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dump", type=Path, required=True, help="attention debug .pt dump containing Q/K/V and dense_output")
    parser.add_argument("--base-dump", type=Path, default=None, help="core-only dump used as the fixed macro Top-k sparse baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--macro-top-ratio", type=float, default=0.2)
    parser.add_argument("--order", choices=["org", "hilbert2d", "blk", "hwf", "fwh", "hilbert3d"], default="hilbert3d")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=129)
    parser.add_argument("--video-len", type=int, default=None, help="override metadata video_len")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--q-macro-count", type=int, default=8, help="number of highest-error Q128 macros to add-back; 0 means all")
    parser.add_argument("--q-macro-ids", type=str, default=None, help="explicit comma-separated Q128 macro ids; overrides --q-macro-count")
    parser.add_argument("--candidate-batch", type=int, default=16)
    parser.add_argument("--heads", type=str, default=None, help="comma-separated original head ids to analyze; default: all heads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_batch < 1:
        raise ValueError("--candidate-batch must be positive")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    payload = _load_payload(args.input_dump)
    q_all = _as_hsd(payload["query"], "query").to(device)
    k_all = _as_hsd(payload["key"], "key").to(device)
    v_all = _as_hsd(payload["value"], "value").to(device)
    dense_all = _as_hsd(payload["dense_output"], "dense_output").to(device).float()
    all_heads, sequence, dim = q_all.shape
    head_ids = (
        list(range(all_heads))
        if args.heads is None
        else [int(x.strip()) for x in args.heads.split(",") if x.strip()]
    )
    if not head_ids or len(set(head_ids)) != len(head_ids) or any(h < 0 or h >= all_heads for h in head_ids):
        raise ValueError(f"--heads must be unique ids in [0, {all_heads})")
    q_orig, k_orig, v_orig = q_all[head_ids], k_all[head_ids], v_all[head_ids]
    dense_orig = dense_all[head_ids]
    heads = len(head_ids)
    metadata = payload.get("metadata", {})
    video_len = args.video_len if args.video_len is not None else metadata.get("video_len", sequence)
    video_len = int(video_len)
    valid_sequence = int(metadata.get("valid_sequence", sequence))
    if not 0 < valid_sequence <= sequence:
        raise ValueError(f"invalid metadata valid_sequence={valid_sequence} for sequence={sequence}")
    video_perm = _make_video_perm(args.order, args.num_frames, args.height, args.width, video_len, device)
    q_full, k_full, v_full, perm, inverse = permute_qkv(q_orig, k_orig, v_orig, video_perm, video_len)
    # The runtime treats the suffix after valid_sequence as an independent
    # dense segment (padding/text tail).  Macro routing only sees the valid
    # prefix, so never let padding create extra Q128/K96 blocks here.
    q, k, v = q_full[:, :valid_sequence], k_full[:, :valid_sequence], v_full[:, :valid_sequence]
    dense_full = dense_orig[:, perm]
    dense = dense_full[:, :valid_sequence]

    print(f"input={args.input_dump} shape=[{heads},{sequence},{dim}] device={device}")
    print(f"macro selector: Q{Q_MACRO}xK{K_MACRO}, top-k ratio={args.macro_top_ratio}, top-p=0, order={args.order}")
    micro_scores = compute_micro_tile_scores(q, k)
    macro_scores = aggregate_macro_scores(micro_scores)
    core_mask = fixed_macro_topk(macro_scores, args.macro_top_ratio, valid_sequence, video_len)
    struct = macro_structure(micro_scores)
    struct["macro_score"] = macro_scores

    if args.base_dump is not None:
        base_payload = _load_payload(args.base_dump)
        base_metadata = base_payload.get("metadata", {})
        if "flashinfer64_core_only" in base_metadata:
            if not base_metadata["flashinfer64_core_only"]:
                raise ValueError(
                    "--base-dump is not a Core-only dump; rerun with "
                    "FLASHINFER64_CORE_ONLY=True and TOKEN_TOP_P=0"
                )
            if abs(float(base_metadata.get("flashinfer64_tile_top_ratio", args.macro_top_ratio)) - args.macro_top_ratio) > 1e-6:
                raise ValueError("--base-dump tile_top_ratio does not match --macro-top-ratio")
        else:
            print("warning: base dump has no routing metadata; verify it is Core-only Top-k=0.2")
        if "sparse_output" not in base_payload:
            raise KeyError("--base-dump must contain sparse_output")
        base_all = _as_hsd(base_payload["sparse_output"], "sparse_output").to(device).float()
        if tuple(base_all.shape) != tuple(dense_all.shape):
            raise ValueError("base sparse_output shape does not match input dump")
        base_orig = base_all[head_ids]
        base = base_orig[:, perm][:, :valid_sequence]
        base_source = str(args.base_dump)
    else:
        base_source = "recomputed Core attention"
        base = None

    error_sq_qmacro = torch.zeros((heads, _ceil_div(valid_sequence, Q_MACRO)), device=device)
    if base is not None:
        diff = base - dense
        for qm in range(error_sq_qmacro.shape[1]):
            s, e = qm * Q_MACRO, min(valid_sequence, (qm + 1) * Q_MACRO)
            error_sq_qmacro[:, qm] = (diff[:, s:e] ** 2).sum(dim=(1, 2))
    else:
        # Compute all Core output in one pass when no baseline dump is given.
        all_qm = list(range(error_sq_qmacro.shape[1]))
        base, base_lse = attention_for_core(q, k, v, core_mask, all_qm)
        diff = base - dense
        for qm in all_qm:
            s, e = qm * Q_MACRO, min(valid_sequence, (qm + 1) * Q_MACRO)
            error_sq_qmacro[:, qm] = (diff[:, s:e] ** 2).sum(dim=(1, 2))

    # Query-level errors are reported in original token order, while Q16/Q128
    # grouping follows the reordered route used by the runtime.
    eq = torch.linalg.vector_norm(base - dense, dim=-1) / (torch.linalg.vector_norm(dense, dim=-1) + EPS)
    q16_count = _ceil_div(valid_sequence, MICRO)
    q16_rows = []
    for head in range(heads):
        for qb in range(q16_count):
            s, e = qb * MICRO, min(valid_sequence, (qb + 1) * MICRO)
            vals = eq[head, s:e]
            q16_rows.append({
                "head": head_ids[head], "q16": qb, "q_macro": qb // Q_MICROS_PER_MACRO,
                "mean_eq": float(vals.mean()), "max_eq": float(vals.max()),
                "p95_eq": float(torch.quantile(vals, 0.95)),
                "mean_abs_error": float((base[head, s:e] - dense[head, s:e]).abs().mean()),
            })

    qerr_handle, qerr_writer = _csv_writer(args.output_dir / "query_errors.csv",
        ["head", "query_index_original", "query_index_reordered", "q16_reordered", "q_macro", "eq"])
    try:
        for head in range(heads):
            for reordered_idx in range(valid_sequence):
                qerr_writer.writerow({
                    "head": head_ids[head],
                    "query_index_original": int(perm[reordered_idx]),
                    "query_index_reordered": reordered_idx,
                    "q16_reordered": reordered_idx // MICRO,
                    "q_macro": reordered_idx // Q_MACRO,
                    "eq": float(eq[head, reordered_idx]),
                })
    finally:
        qerr_handle.close()
    q16_handle, q16_writer = _csv_writer(args.output_dir / "q16_error_summary.csv",
        ["head", "q16", "q_macro", "mean_eq", "max_eq", "p95_eq", "mean_abs_error"])
    try:
        q16_writer.writerows(q16_rows)
    finally:
        q16_handle.close()

    q_macro_count = error_sq_qmacro.shape[1]
    if args.q_macro_ids:
        q_macro_ids = [int(x.strip()) for x in args.q_macro_ids.split(",") if x.strip()]
    elif args.q_macro_count == 0:
        q_macro_ids = list(range(q_macro_count))
    else:
        score = error_sq_qmacro.mean(dim=0)
        q_macro_ids = torch.topk(score, min(args.q_macro_count, q_macro_count), sorted=False).indices.sort().values.tolist()
    if any(x < 0 or x >= q_macro_count for x in q_macro_ids):
        raise ValueError(f"Q macro ids must be in [0, {q_macro_count})")
    q_macro_ids = sorted(set(q_macro_ids))
    # Recompute only the LSE for selected Q macros.  This is needed for the
    # exact changed-denominator add-back even when base output came from a dump.
    _, base_lse = attention_for_core(q, k, v, core_mask, q_macro_ids)

    analysis_q_mask = torch.zeros((heads, valid_sequence), dtype=torch.bool, device=device)
    for qm in q_macro_ids:
        s, e = qm * Q_MACRO, min(valid_sequence, (qm + 1) * Q_MACRO)
        analysis_q_mask[:, s:e] = True
    rows = addback_rows(
        q, k, v, dense, base, base_lse, core_mask, struct, q_macro_ids,
        analysis_q_mask, candidate_batch=args.candidate_batch, head_ids=head_ids,
    )
    addback_fields = [
        "head", "q_macro", "k_macro", "macro_score", "M_g", "A_g", "H_g", "C_g",
        "valid_microtiles", "base_error_l2_qmacro", "addback_error_l2_qmacro",
        "delta_E_qmacro", "delta_E_relative_qmacro", "delta_E_relative_global",
    ]
    add_handle, add_writer = _csv_writer(args.output_dir / "macro_addback.csv", addback_fields)
    try:
        add_writer.writerows(rows)
    finally:
        add_handle.close()
    rows.sort(key=lambda r: r["delta_E_relative_qmacro"], reverse=True)
    top_handle, top_writer = _csv_writer(args.output_dir / "top_addback_macros.csv", addback_fields)
    try:
        top_writer.writerows(rows[: min(1000, len(rows))])
    finally:
        top_handle.close()
    head_handle, head_writer = _csv_writer(
        args.output_dir / "head_summary.csv",
        ["head", "addback_rows", "positive_addback_rows", "positive_recovery_sum",
         "spearman_H_vs_delta", "spearman_A_vs_delta", "spearman_C_vs_delta",
         "spearman_M_vs_delta"],
    )
    try:
        for head_id in sorted({r["head"] for r in rows}):
            head_rows = [r for r in rows if r["head"] == head_id]
            head_writer.writerow({
                "head": head_id,
                "addback_rows": len(head_rows),
                "positive_addback_rows": sum(r["delta_E_relative_qmacro"] > 0 for r in head_rows),
                "positive_recovery_sum": sum(max(0.0, r["delta_E_relative_qmacro"]) for r in head_rows),
                "spearman_H_vs_delta": _corr([r["H_g"] for r in head_rows], [r["delta_E_relative_qmacro"] for r in head_rows], spearman=True),
                "spearman_A_vs_delta": _corr([r["A_g"] for r in head_rows], [r["delta_E_relative_qmacro"] for r in head_rows], spearman=True),
                "spearman_C_vs_delta": _corr([r["C_g"] for r in head_rows], [r["delta_E_relative_qmacro"] for r in head_rows], spearman=True),
                "spearman_M_vs_delta": _corr([r["M_g"] for r in head_rows], [r["delta_E_relative_qmacro"] for r in head_rows], spearman=True),
            })
    finally:
        head_handle.close()
    write_scatter(args.output_dir / "macro_addback_entropy_max.png", rows)
    write_head_scatter(args.output_dir / "macro_addback_entropy_max_by_head.png", rows)

    positive = [r for r in rows if r["delta_E_relative_qmacro"] > 0]
    hs = [r["H_g"] for r in rows]; aas = [r["A_g"] for r in rows]
    ds = [r["delta_E_relative_qmacro"] for r in rows]
    h25 = sorted(hs)[max(0, int(0.25 * len(hs)) - 1)] if hs else float("nan")
    a75 = sorted(aas)[max(0, int(0.75 * len(aas)) - 1)] if aas else float("nan")
    quadrant = [r for r in positive if r["H_g"] <= h25 and r["A_g"] >= a75]
    total_positive = sum(r["delta_E_relative_qmacro"] for r in positive)
    summary = {
        "input_dump": str(args.input_dump),
        "base_dump": base_source,
        "shape": [heads, sequence, dim],
        "head_ids": head_ids,
        "valid_sequence": valid_sequence,
        "video_len": video_len,
        "order": args.order,
        "macro_geometry": "Q128xK96 / micro Q16xK16 / 8x6=48 microtiles",
        "macro_top_ratio": args.macro_top_ratio,
        "macro_top_p": 0.0,
        "core_selected_mean": float(core_mask.float().mean()),
        "core_selected_per_head_mean": float(core_mask.float().mean(dim=(1, 2)).mean()),
        "q_macro_ids": q_macro_ids,
        "addback_rows": len(rows),
        "positive_addback_rows": len(positive),
        "spearman_H_vs_delta": _corr(hs, ds, spearman=True),
        "spearman_A_vs_delta": _corr(aas, ds, spearman=True),
        "spearman_C_vs_delta": _corr([r["C_g"] for r in rows], ds, spearman=True),
        "spearman_M_vs_delta": _corr([r["M_g"] for r in rows], ds, spearman=True),
        "low_entropy_high_peak_thresholds": {"H_25": h25, "A_75": a75},
        "low_entropy_high_peak_positive_recovery_share": (
            sum(r["delta_E_relative_qmacro"] for r in quadrant) / (total_positive + EPS)
        ),
        "low_entropy_high_peak_positive_count": len(quadrant),
        "recommendation": (
            "supports residual prioritization of low-H/high-A omitted macros"
            if quadrant and sum(r["delta_E_relative_qmacro"] for r in quadrant) > 0.5 * (total_positive + EPS)
            else "does not by itself support a low-H/high-A residual rule; inspect correlations and top_addback_macros.csv"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
