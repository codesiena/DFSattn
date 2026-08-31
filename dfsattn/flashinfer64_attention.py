"""Hardware-aligned hierarchical sparse attention for HunyuanVideo.

The historical module/backend name is retained for CLI compatibility.  The
implementation is now Q128xK96 FlashInfer Core plus Q16xK16 grouped Triton
Residual, with macro Top-k/Top-p, global complement micro Top-p, occupancy promotion,
exact LSE merge, optional compact-route reuse, and one model-wide ephemeral
FlashInfer plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import flashinfer
    from flashinfer.sparse import (
        VariableBlockSparseAttentionWrapper as FlashInferVariableBlockSparseAttention,
    )
except Exception:  # pragma: no cover
    flashinfer = None
    FlashInferVariableBlockSparseAttention = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


Q_MACRO = 128
K_MACRO = 96
MICRO = 16
Q_MICROS_PER_MACRO = Q_MACRO // MICRO
K_MICROS_PER_MACRO = K_MACRO // MICRO
BLOCK_SIZE = Q_MACRO  # historical compatibility only
RESIDUAL_BUCKET_CAPS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# SVG's bounded-memory lifecycle: layers execute sequentially, so one wrapper
# and one expanded plan are enough for the whole model.  Every plan overwrites
# the previous layer's plan instead of being retained by 60 backend instances.
_shared_core_wrapper: Any = None
_shared_core_workspace: Optional[torch.Tensor] = None
_shared_core_signature = None


@dataclass
class FlashInfer64Route:
    core_mask: torch.Tensor          # [H, ceil(S/128), ceil(S/96)]
    residual_mask: Optional[torch.Tensor]  # CPU reference only; never retained on CUDA
    residual_indices: torch.Tensor   # CSR column indices [nnz]
    residual_indptr: torch.Tensor    # CSR row pointers [H*ceil(S/16)+1]
    residual_buckets: Tuple[Tuple[int, torch.Tensor], ...]  # (capacity, CSR row ids)
    sequence: int
    video_len: int
    route_mode: str
    top_p: float
    tile_top_ratio: float
    token_top_p: float
    promotion_threshold: int
    promoted_tiles: int
    core_interactions: int
    residual_interactions: int
    residual_micro_tiles: int
    residual_count_mean: float
    residual_count_p50: float
    residual_count_p95: float
    residual_count_max: float
    residual_count_nonempty_ratio: float


def _timing_start(recorder: Any, device: torch.device) -> Any:
    return None if recorder is None else recorder.start(device)


def _timing_stop(recorder, start, phase, step, layer, device) -> None:
    if recorder is not None:
        recorder.stop(start, phase=phase, step_idx=step, layer_idx=layer, device=device)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _make_permutation(sequence, video_perm, video_len, device):
    if video_perm is None:
        perm = torch.arange(sequence, device=device, dtype=torch.long)
    else:
        vp = video_perm.to(device=device, dtype=torch.long)
        if video_len is None or video_len == sequence:
            perm = vp
        else:
            if vp.numel() != video_len:
                raise ValueError(f"video_perm has {vp.numel()} entries, expected {video_len}")
            perm = torch.cat((vp, torch.arange(video_len, sequence, device=device)))
    if perm.numel() != sequence:
        raise ValueError(f"permutation has {perm.numel()} entries, expected {sequence}")
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(sequence, device=device)
    return perm, inverse


def _permute_qkv(q, k, v, video_perm, video_len):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError("q, k, v must match [1, heads, sequence, head_dim]")
    perm, inverse = _make_permutation(q.shape[2], video_perm, video_len, q.device)
    return q[0, :, perm].contiguous(), k[0, :, perm].contiguous(), v[0, :, perm].contiguous(), inverse


def _block_means(x: torch.Tensor, block: int) -> torch.Tensor:
    heads, sequence, dim = x.shape
    padded = _ceil_div(sequence, block) * block
    if padded != sequence:
        x = F.pad(x, (0, 0, 0, padded - sequence))
    grouped = x.view(heads, padded // block, block, dim)
    valid = (torch.arange(padded, device=x.device) < sequence).view(1, -1, block, 1)
    return (grouped * valid).sum(2) / valid.sum(2).clamp_min(1).to(x.dtype)


def compute_micro_tile_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Normalized proxy mass at Q16/K16 selection granularity."""
    qm, km = _block_means(q, MICRO), _block_means(k, MICRO)
    logits = torch.bmm(qm, km.transpose(1, 2)).float() * (q.shape[-1] ** -0.5)
    return torch.softmax(logits, dim=-1)


def aggregate_macro_scores(micro_scores: torch.Tensor) -> torch.Tensor:
    """Aggregate Q16/K16 mass into hardware-native Q128/K96 macro mass."""
    heads, q_micro, k_micro = micro_scores.shape
    padded_q = _ceil_div(q_micro, Q_MICROS_PER_MACRO) * Q_MICROS_PER_MACRO
    padded_k = _ceil_div(k_micro, K_MICROS_PER_MACRO) * K_MICROS_PER_MACRO
    scores = F.pad(micro_scores, (0, padded_k - k_micro, 0, padded_q - q_micro))
    scores = scores.view(
        heads, padded_q // Q_MICROS_PER_MACRO, Q_MICROS_PER_MACRO,
        padded_k // K_MICROS_PER_MACRO, K_MICROS_PER_MACRO,
    )
    macro = scores.sum(-1).mean(2)
    return macro / macro.sum(-1, keepdim=True).clamp_min(torch.finfo(macro.dtype).tiny)


def compute_64_tile_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compatibility name; returns Q128/K96 macro scores."""
    return aggregate_macro_scores(compute_micro_tile_scores(q, k))


def select_64_tiles_from_scores(tile_scores, *, top_p: float):
    """Compatibility name; selects hardware macro tiles by cumulative mass."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p >= 1.0:
        return torch.ones_like(tile_scores, dtype=torch.bool)
    values, indices = torch.sort(tile_scores, dim=-1, descending=True, stable=True)
    keep = (torch.cumsum(values, -1) - values) < top_p
    keep[..., 0] = True
    return torch.zeros_like(tile_scores, dtype=torch.bool).scatter_(-1, indices, keep)


def select_64_tiles_topk_from_scores(tile_scores, *, top_ratio: float):
    if not 0.0 < top_ratio <= 1.0:
        raise ValueError(f"tile top_ratio must be in (0, 1], got {top_ratio}")
    count = max(1, math.ceil(tile_scores.shape[-1] * top_ratio))
    indices = torch.topk(tile_scores, count, dim=-1, sorted=False).indices
    return torch.zeros_like(tile_scores, dtype=torch.bool).scatter_(-1, indices, True)


def _select_hyvideo_core_tiles(
    tile_scores, *, route_mode, tile_top_p, tile_top_ratio, sequence, video_len,
):
    """Macro selector preserving dense text KV and dense text-query blocks."""
    if route_mode not in ("topp_topk", "topk_topp"):
        raise ValueError(f"unknown route_mode {route_mode!r}")
    if video_len is None or video_len >= sequence:
        return (
            select_64_tiles_from_scores(tile_scores, top_p=tile_top_p)
            if route_mode == "topp_topk"
            else select_64_tiles_topk_from_scores(tile_scores, top_ratio=tile_top_ratio)
        )
    if video_len < 0:
        raise ValueError("video_len must be non-negative")

    _, _, k_blocks = tile_scores.shape
    q_dense_start, k_dense_start = video_len // Q_MACRO, video_len // K_MACRO
    core = torch.zeros_like(tile_scores, dtype=torch.bool)
    if q_dense_start > 0 and k_dense_start > 0:
        video_scores = tile_scores[:, :q_dense_start, :k_dense_start]
        if route_mode == "topp_topk":
            video_scores = video_scores / video_scores.sum(-1, keepdim=True).clamp_min(
                torch.finfo(video_scores.dtype).tiny
            )
            core[:, :q_dense_start, :k_dense_start] = select_64_tiles_from_scores(
                video_scores, top_p=tile_top_p
            )
        else:
            forced_text = k_blocks - k_dense_start
            total_keep = max(1, math.ceil(k_blocks * tile_top_ratio))
            video_keep = min(k_dense_start, max(0, total_keep - forced_text))
            if video_keep:
                idx = torch.topk(video_scores, video_keep, dim=-1, sorted=False).indices
                core[:, :q_dense_start, :k_dense_start].scatter_(-1, idx, True)
    core[..., k_dense_start:] = True
    core[:, q_dense_start:, :] = True
    return core


def select_64_tiles(q, k, *, top_p):
    return select_64_tiles_from_scores(compute_64_tile_scores(q, k), top_p=top_p)


def _select_residual_to_total_mass(micro_scores, core_mask, *, total_top_p):
    """Select rejected microtiles until Core+Residual reaches total Top-p.

    ``micro_scores`` is already normalized over all K16 tiles for each
    (head,Q16).  The complement is deliberately *not* renormalized: Core mass
    is subtracted from the requested total mass, and raw rejected mass fills
    only the remainder.
    """
    if not 0.0 <= total_top_p <= 1.0:
        raise ValueError("total_top_p must be in [0, 1]")
    _, q_micro, k_micro = micro_scores.shape
    q_parent = torch.arange(q_micro, device=micro_scores.device) // Q_MICROS_PER_MACRO
    k_parent = torch.arange(k_micro, device=micro_scores.device) // K_MICROS_PER_MACRO
    covered_by_core = core_mask[:, q_parent[:, None], k_parent[None, :]]
    core_mass = micro_scores.masked_fill(~covered_by_core, 0.0).sum(-1)
    required_mass = (total_top_p - core_mass).clamp_min(0.0)
    rejected_mass = micro_scores.masked_fill(covered_by_core, 0.0)
    values, indices = torch.sort(rejected_mass, dim=-1, descending=True, stable=True)
    keep = (
        ((torch.cumsum(values, -1) - values) < required_mass[..., None])
        & (required_mass[..., None] > 0)
        & (values > 0)
    )
    return torch.zeros_like(covered_by_core).scatter_(-1, indices, keep)


def _promote_residual_microtiles(residual, core_mask, *, promotion_threshold):
    """Promote dense 8x6 residual groups and keep both supports disjoint."""
    max_occupancy = Q_MICROS_PER_MACRO * K_MICROS_PER_MACRO
    if not 1 <= promotion_threshold <= max_occupancy:
        raise ValueError(f"promotion_threshold must be in [1, {max_occupancy}]")
    heads, q_micro, k_micro = residual.shape
    q_macro, k_macro = core_mask.shape[1:]
    q_parent = torch.arange(q_micro, device=residual.device) // Q_MICROS_PER_MACRO
    k_parent = torch.arange(k_micro, device=residual.device) // K_MICROS_PER_MACRO
    padded_q, padded_k = q_macro * Q_MICROS_PER_MACRO, k_macro * K_MICROS_PER_MACRO
    grouped = F.pad(residual, (0, padded_k - k_micro, 0, padded_q - q_micro)).view(
        heads, q_macro, Q_MICROS_PER_MACRO, k_macro, K_MICROS_PER_MACRO
    )
    promote = (grouped.sum((2, 4)) >= promotion_threshold) & ~core_mask
    promoted_tiles = int(promote.sum().item())
    core_mask = core_mask | promote
    residual &= ~promote[:, q_parent[:, None], k_parent[None, :]]
    return core_mask, residual, promoted_tiles


def _build_residual_csr(residual, *, collect_stats: bool = True):
    """Pack Q16/K16 support into CSR and group non-empty rows by length.

    The old layout padded every row to the global maximum selected-K16 count.
    CSR stores each selected K16 exactly once, while length buckets bound the
    remaining per-kernel padding to a small local range.
    """
    heads, q_micro, k_micro = residual.shape
    flat = residual.reshape(heads * q_micro, k_micro)
    counts = flat.sum(-1, dtype=torch.int32)
    indptr = torch.empty(
        counts.numel() + 1, device=residual.device, dtype=torch.int32
    )
    indptr[0] = 0
    torch.cumsum(counts, dim=0, out=indptr[1:])
    key_ids = torch.arange(k_micro, device=residual.device, dtype=torch.int32)
    indices = key_ids.expand(flat.shape[0], -1).masked_select(flat).contiguous()

    caps = list(RESIDUAL_BUCKET_CAPS)
    while caps[-1] < k_micro:
        caps.append(caps[-1] * 2)
    all_rows = torch.arange(counts.numel(), device=residual.device, dtype=torch.int32)
    buckets = []
    lower = 0
    for cap in caps:
        row_ids = all_rows.masked_select((counts > lower) & (counts <= cap))
        if row_ids.numel():
            buckets.append((cap, row_ids.contiguous()))
        lower = cap

    if collect_stats and counts.numel():
        count_float = counts.float()
        p50, p95 = torch.quantile(
            count_float, torch.tensor((0.50, 0.95), device=residual.device)
        ).tolist()
        count_mean = float(count_float.mean().item())
        count_max = float(counts.max().item())
        nonempty_ratio = float((counts > 0).float().mean().item())
    else:
        count_mean = p50 = p95 = count_max = nonempty_ratio = float("nan")
    stats = (count_mean, float(p50), float(p95), count_max, nonempty_ratio)
    return indices, indptr, tuple(buckets), stats


def _count_route_interactions(core_mask, residual_mask, sequence):
    """Count valid token interactions once, before the CUDA bool mask dies."""
    q_sizes = torch.full(
        (core_mask.shape[1],), Q_MACRO, device=core_mask.device, dtype=torch.int64
    )
    k_sizes = torch.full(
        (core_mask.shape[2],), K_MACRO, device=core_mask.device, dtype=torch.int64
    )
    q_sizes[-1] = sequence - (q_sizes.numel() - 1) * Q_MACRO
    k_sizes[-1] = sequence - (k_sizes.numel() - 1) * K_MACRO
    core_n = int(
        (core_mask * q_sizes[None, :, None] * k_sizes[None, None]).sum().item()
    )
    q16 = torch.full(
        (residual_mask.shape[1],), MICRO, device=core_mask.device, dtype=torch.int64
    )
    k16 = torch.full(
        (residual_mask.shape[2],), MICRO, device=core_mask.device, dtype=torch.int64
    )
    q16[-1] = sequence - (q16.numel() - 1) * MICRO
    k16[-1] = sequence - (k16.numel() - 1) * MICRO
    residual_n = int(
        (residual_mask * q16[None, :, None] * k16[None, None]).sum().item()
    )
    return core_n, residual_n, int(residual_mask.sum().item())


def _build_residual_topk(*args, **kwargs):
    raise RuntimeError(
        "legacy per-query token residual was replaced by cached Q16/K16 routing; "
        "inspect FlashInfer64Attention.route.residual_indices"
    )


def _ensure_flashinfer_vector_workspace(wrapper: Any) -> None:
    if getattr(wrapper, "_backend", None) != "fa3":
        return
    indices = getattr(wrapper, "_paged_kv_indices_buf", None)
    indptr = getattr(wrapper, "_paged_kv_indptr_buf", None)
    vector_indices = getattr(wrapper, "_vector_sparse_indices_buffer", None)
    vector_indptr = getattr(wrapper, "_vector_sparse_indptr_buffer", None)
    if any(x is None for x in (indices, indptr, vector_indices, vector_indptr)):
        raise RuntimeError("installed FlashInfer FA3 wrapper lacks vector sparse workspaces")
    if vector_indices.numel() <= indices.numel():
        vector_indices = torch.empty(indices.numel() + 1, dtype=torch.int32, device=indices.device)
    if vector_indptr.numel() <= indptr.numel():
        vector_indptr = torch.empty(indptr.numel() + 1, dtype=torch.int32, device=indptr.device)
    wrapper.reset_workspace_buffer(
        float_workspace_buffer=wrapper._float_workspace_buffer,
        int_workspace_buffer=wrapper._int_workspace_buffer,
        vector_sparse_indices_buffer=vector_indices,
        vector_sparse_indptr_buffer=vector_indptr,
    )


def _prepare_flashinfer_vector_workspace(
    wrapper: Any,
    core_mask: torch.Tensor,
    col_sizes: torch.Tensor,
) -> None:
    """Right-size vector buffers *before* plan performs its capacity check.

    FlashInfer's constructor reserves 128M int32 indices (512MB) but only
    32768 indptr entries.  Long-video plans can exceed the latter and sparse
    plans usually need far less than the former.  Computing exact capacities
    from the compact macro route both fixes the original ValueError and avoids
    retaining the 512MB default allocation in every cached layer wrapper.
    """
    required_indices = int((core_mask.to(torch.int64) * col_sizes[:, None]).sum().item()) + 1
    required_indptr = core_mask.shape[0] * core_mask.shape[1] + 2
    vector_indices = torch.empty(
        required_indices, dtype=torch.int32, device=core_mask.device
    )
    vector_indptr = torch.empty(
        required_indptr, dtype=torch.int32, device=core_mask.device
    )
    wrapper.reset_workspace_buffer(
        float_workspace_buffer=wrapper._float_workspace_buffer,
        int_workspace_buffer=wrapper._int_workspace_buffer,
        vector_sparse_indices_buffer=vector_indices,
        vector_sparse_indptr_buffer=vector_indptr,
    )


if triton is not None:
    @triton.jit
    def _residual_micro_attention_kernel(
        q_ptr, k_ptr, v_ptr, index_ptr, indptr_ptr, row_ids_ptr, out_ptr, lse_ptr,
        sequence, q_micro_blocks, k_micro_blocks,
        stride_qh, stride_qs, stride_kh, stride_ks, stride_vh, stride_vs,
        stride_oh, stride_os,
        head_dim: tl.constexpr, block_d: tl.constexpr,
        bucket_capacity: tl.constexpr, micro_size: tl.constexpr,
        scale: tl.constexpr,
    ):
        """One grouped MMA program per non-empty CSR row in one length bucket."""
        pid = tl.program_id(0)
        csr_row = tl.load(row_ids_ptr + pid)
        head, qb = csr_row // q_micro_blocks, csr_row % q_micro_blocks
        row_start = tl.load(indptr_ptr + csr_row)
        row_end = tl.load(indptr_ptr + csr_row + 1)
        row_count = row_end - row_start
        rows = tl.arange(0, micro_size)
        cols = tl.arange(0, micro_size)
        dims = tl.arange(0, block_d)
        q_pos = qb * micro_size + rows
        q_valid = q_pos < sequence
        q = tl.load(
            q_ptr + head * stride_qh + q_pos[:, None] * stride_qs + dims[None, :],
            mask=q_valid[:, None] & (dims[None, :] < head_dim), other=0.0,
        )
        m = tl.full((micro_size,), float("-inf"), tl.float32)
        l = tl.zeros((micro_size,), tl.float32)
        acc = tl.zeros((micro_size, block_d), tl.float32)
        for slot in range(0, bucket_capacity):
            slot_valid = slot < row_count
            kb = tl.load(index_ptr + row_start + slot, mask=slot_valid, other=k_micro_blocks)
            k_pos = kb * micro_size + cols
            k_valid = slot_valid & (kb < k_micro_blocks) & (k_pos < sequence)
            kt = tl.load(
                k_ptr + head * stride_kh + k_pos[None, :] * stride_ks + dims[:, None],
                mask=k_valid[None, :] & (dims[:, None] < head_dim), other=0.0,
            )
            score = tl.dot(q, kt) * scale
            pair_valid = q_valid[:, None] & k_valid[None, :]
            score = tl.where(pair_valid, score, float("-inf"))
            tile_m = tl.max(score, axis=1)
            new_m = tl.maximum(m, tile_m)
            alpha = tl.where(m != float("-inf"), tl.exp(m - new_m), 0.0)
            weights = tl.where(pair_valid, tl.exp(score - new_m[:, None]), 0.0)
            l = l * alpha + tl.sum(weights, axis=1)
            acc = acc * alpha[:, None]
            vv = tl.load(
                v_ptr + head * stride_vh + k_pos[:, None] * stride_vs + dims[None, :],
                mask=k_valid[:, None] & (dims[None, :] < head_dim), other=0.0,
            )
            acc += tl.dot(weights.to(vv.dtype), vv)
            m = new_m
        output = acc / tl.maximum(l[:, None], 1.0e-20)
        tl.store(
            out_ptr + head * stride_oh + q_pos[:, None] * stride_os + dims[None, :], output,
            mask=q_valid[:, None] & (dims[None, :] < head_dim),
        )
        row_lse = tl.where(l > 0, m + tl.log(l), float("-inf"))
        tl.store(lse_ptr + head * sequence + q_pos, row_lse, mask=q_valid)


def _run_residual_micro(q, k, v, route, recorder=None, step=-1, layer=-1):
    heads, sequence, dim = q.shape
    indices = route.residual_indices
    if indices.numel() == 0:
        return torch.zeros_like(q), torch.full(
            (heads, sequence), float("-inf"), device=q.device, dtype=torch.float32
        )
    if triton is None or not q.is_cuda:
        mask = route.residual_mask.repeat_interleave(MICRO, 1).repeat_interleave(MICRO, 2)
        mask = mask[:, :sequence, :sequence]
        scores = torch.bmm(q, k.transpose(1, 2)).float() * (dim ** -0.5)
        scores.masked_fill_(~mask, float("-inf"))
        lse = torch.logsumexp(scores, -1)
        valid = mask.any(-1, keepdim=True)
        weights = torch.softmax(torch.where(valid, scores, torch.zeros_like(scores)), -1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return torch.bmm(weights.to(v.dtype), v), lse
    q_micro_blocks = _ceil_div(sequence, MICRO)
    output = torch.zeros_like(q)
    lse = torch.full((heads, sequence), float("-inf"), device=q.device, dtype=torch.float32)
    for bucket_capacity, row_ids in route.residual_buckets:
        bucket_start = _timing_start(recorder, q.device)
        _residual_micro_attention_kernel[(row_ids.numel(),)](
            q, k, v, indices, route.residual_indptr, row_ids, output, lse,
            sequence, q_micro_blocks, _ceil_div(sequence, MICRO),
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), output.stride(0), output.stride(1),
            head_dim=dim, block_d=triton.next_power_of_2(dim),
            bucket_capacity=bucket_capacity, micro_size=MICRO,
            scale=dim ** -0.5, num_warps=4,
        )
        _timing_stop(
            recorder, bucket_start,
            f"flashinfer_residual_bucket_le{bucket_capacity}",
            step, layer, q.device,
        )
    return output, lse


def _merge_lse_states(core_output, core_lse, residual_output, residual_lse):
    m = torch.maximum(core_lse, residual_lse)
    wc = torch.where(torch.isfinite(core_lse), torch.exp(core_lse - m), torch.zeros_like(m))
    wr = torch.where(torch.isfinite(residual_lse), torch.exp(residual_lse - m), torch.zeros_like(m))
    z = (wc + wr).clamp_min(torch.finfo(wc.dtype).tiny)
    return (wc[..., None] * core_output.float() + wr[..., None] * residual_output.float()) / z[..., None]


class FlashInfer64Attention:
    """Per-layer compact route; expanded FlashInfer plan is model-wide."""

    def __init__(self):
        self.route: Optional[FlashInfer64Route] = None
        self.last_stats = None

    def _plan_core(self, q, core_mask):
        global _shared_core_wrapper, _shared_core_workspace, _shared_core_signature
        if not q.is_cuda:
            return None
        if flashinfer is None or FlashInferVariableBlockSparseAttention is None:
            raise ImportError("hierarchical backend requires FlashInfer")
        heads, sequence, dim = q.shape
        signature = (q.device.index, q.dtype, heads, sequence, dim)
        if _shared_core_wrapper is None or _shared_core_signature != signature:
            workspace_mb = int(os.environ.get("FLASHINFER64_WORKSPACE_MB", "128"))
            if workspace_mb < 16:
                raise ValueError("FLASHINFER64_WORKSPACE_MB must be at least 16")
            _shared_core_workspace = torch.empty(
                workspace_mb * 1024 * 1024, device=q.device, dtype=torch.uint8
            )
            _shared_core_wrapper = FlashInferVariableBlockSparseAttention(
                _shared_core_workspace, backend="auto"
            )
            _shared_core_signature = signature
        q_blocks, k_blocks = core_mask.shape[1:]
        row_sizes = torch.full((heads, q_blocks), Q_MACRO, device=q.device, dtype=torch.int32)
        col_sizes = torch.full((heads, k_blocks), K_MACRO, device=q.device, dtype=torch.int32)
        row_sizes[:, -1] = sequence - (q_blocks - 1) * Q_MACRO
        col_sizes[:, -1] = sequence - (k_blocks - 1) * K_MACRO
        _prepare_flashinfer_vector_workspace(_shared_core_wrapper, core_mask, col_sizes)
        _shared_core_wrapper.plan(
            core_mask, row_sizes, col_sizes, heads, heads, dim, causal=False,
            pos_encoding_mode="NONE", sm_scale=dim ** -0.5,
            q_data_type=q.dtype, kv_data_type=q.dtype,
        )
        _ensure_flashinfer_vector_workspace(_shared_core_wrapper)
        return _shared_core_wrapper

    def _run_core(self, q, k, v, route, wrapper, recorder, step, layer):
        start = _timing_start(recorder, q.device)
        if not q.is_cuda:
            mask = route.core_mask.repeat_interleave(Q_MACRO, 1).repeat_interleave(K_MACRO, 2)
            mask = mask[:, :q.shape[1], :k.shape[1]]
            scores = torch.bmm(q, k.transpose(1, 2)).float() * (q.shape[-1] ** -0.5)
            scores.masked_fill_(~mask, float("-inf"))
            lse = torch.logsumexp(scores, -1)
            output = torch.bmm(torch.softmax(scores, -1).to(v.dtype), v)
        else:
            if wrapper is None:
                raise RuntimeError("shared FlashInfer plan is missing")
            output, lse = wrapper.run(q, k, v, return_lse=True)
            lse = lse * math.log(2.0)
        _timing_stop(recorder, start, "flashinfer_core_run", step, layer, q.device)
        return output, lse

    @torch.no_grad()
    def __call__(
        self, q, k, v, *, video_perm, video_len,
        route_mode="topk_topp", tile_top_p: float, tile_top_ratio: float = 0.25,
        token_top_k: Optional[int], token_top_ratio: float, token_top_p: float = 0.9,
        promotion_threshold: int = 24,
        reuse_route: bool = False,
        valid_sequence: Optional[int] = None, refresh_route: bool,
        record_density: bool = False, timing_recorder: Any = None,
        step_idx: int = -1, layer_idx: int = -1,
    ):
        del token_top_k, token_top_ratio  # retained only for old launch scripts
        start = _timing_start(timing_recorder, q.device)
        qh, kh, vh, inverse = _permute_qkv(q, k, v, video_perm, video_len)
        _timing_stop(timing_recorder, start, "flashinfer_qkv_permute", step_idx, layer_idx, q.device)
        full_sequence = qh.shape[1]
        valid_sequence = full_sequence if valid_sequence is None else valid_sequence
        if not 0 < valid_sequence <= full_sequence:
            raise ValueError("valid_sequence is outside the input sequence")
        padding_q, padding_k, padding_v = qh[:, valid_sequence:], kh[:, valid_sequence:], vh[:, valid_sequence:]
        qh, kh, vh = qh[:, :valid_sequence].contiguous(), kh[:, :valid_sequence].contiguous(), vh[:, :valid_sequence].contiguous()
        sequence = valid_sequence

        def restore(main):
            if valid_sequence < full_sequence:
                pad = F.scaled_dot_product_attention(padding_q[None], padding_k[None], padding_v[None], dropout_p=0.0)[0]
                main = torch.cat((main, pad), dim=1)
            return main.to(q.dtype)[None, :, inverse]

        route_valid = (
            reuse_route and self.route is not None and self.route.sequence == sequence
            and self.route.video_len == (-1 if video_len is None else video_len)
            and self.route.route_mode == route_mode and self.route.top_p == tile_top_p
            and self.route.tile_top_ratio == tile_top_ratio and self.route.token_top_p == token_top_p
            and self.route.promotion_threshold == promotion_threshold
            and self.route.core_mask.device == q.device
        )
        if refresh_route or not route_valid:
            fine_start = _timing_start(timing_recorder, q.device)
            micro_scores = compute_micro_tile_scores(qh, kh)
            _timing_stop(timing_recorder, fine_start, "flashinfer_fine_score", step_idx, layer_idx, q.device)

            core_score_start = _timing_start(timing_recorder, q.device)
            macro_scores = aggregate_macro_scores(micro_scores)
            _timing_stop(timing_recorder, core_score_start, "flashinfer_core_score", step_idx, layer_idx, q.device)

            core_select_start = _timing_start(timing_recorder, q.device)
            core = _select_hyvideo_core_tiles(
                macro_scores, route_mode=route_mode, tile_top_p=tile_top_p,
                tile_top_ratio=tile_top_ratio, sequence=sequence, video_len=video_len,
            )
            _timing_stop(timing_recorder, core_select_start, "flashinfer_core_select", step_idx, layer_idx, q.device)

            residual_select_start = _timing_start(timing_recorder, q.device)
            residual = _select_residual_to_total_mass(
                micro_scores, core, total_top_p=token_top_p,
            )
            _timing_stop(timing_recorder, residual_select_start, "flashinfer_residual_select", step_idx, layer_idx, q.device)

            promotion_start = _timing_start(timing_recorder, q.device)
            core, residual, promoted = _promote_residual_microtiles(
                residual, core, promotion_threshold=promotion_threshold,
            )
            _timing_stop(timing_recorder, promotion_start, "flashinfer_promotion", step_idx, layer_idx, q.device)

            compact_start = _timing_start(timing_recorder, q.device)
            collect_route_stats = record_density or bool(
                timing_recorder is not None and getattr(timing_recorder, "enabled", False)
            )
            indices, indptr, buckets, residual_count_stats = _build_residual_csr(
                residual, collect_stats=collect_route_stats,
            )
            core_n, residual_n, residual_tiles = _count_route_interactions(
                core, residual, sequence
            )
            _timing_stop(timing_recorder, compact_start, "flashinfer_residual_compact", step_idx, layer_idx, q.device)
            # CUDA execution consumes only the compact K16 lists.  Keeping the
            # full bool matrix would cost about 180MB per layer at 480p.
            residual_mask_for_route = residual if not q.is_cuda else None
            self.route = FlashInfer64Route(
                core, residual_mask_for_route, indices, indptr, buckets,
                sequence, -1 if video_len is None else video_len,
                route_mode, tile_top_p, tile_top_ratio, token_top_p,
                promotion_threshold, promoted,
                core_n, residual_n, residual_tiles,
                *residual_count_stats,
            )
            del micro_scores, macro_scores, residual

        route = self.route
        if route is None:
            raise RuntimeError("failed to build hierarchical route")
        if timing_recorder is not None:
            timing_recorder.record_metrics({
                "residual_count_mean": route.residual_count_mean,
                "residual_count_p50": route.residual_count_p50,
                "residual_count_p95": route.residual_count_p95,
                "residual_count_max": route.residual_count_max,
                "residual_count_nonempty_ratio": route.residual_count_nonempty_ratio,
            })
        # Like SVG, plan is intentionally ephemeral and overwrites the prior
        # layer's expanded token indices.  Compact route reuse does not imply
        # expanded-plan reuse.
        plan_start = _timing_start(timing_recorder, q.device)
        wrapper = self._plan_core(qh, route.core_mask)
        _timing_stop(timing_recorder, plan_start, "flashinfer_plan", step_idx, layer_idx, q.device)

        core_output, core_lse = self._run_core(
            qh, kh, vh, route, wrapper,
            timing_recorder, step_idx, layer_idx,
        )
        residual_start = _timing_start(timing_recorder, q.device)
        residual_output, residual_lse = _run_residual_micro(
            qh, kh, vh, route, timing_recorder, step_idx, layer_idx,
        )
        _timing_stop(timing_recorder, residual_start, "flashinfer_residual_micro_run", step_idx, layer_idx, q.device)
        merge_start = _timing_start(timing_recorder, q.device)
        merged = _merge_lse_states(core_output, core_lse, residual_output, residual_lse)
        _timing_stop(timing_recorder, merge_start, "flashinfer_lse_merge", step_idx, layer_idx, q.device)

        density_start = _timing_start(timing_recorder, q.device)
        if record_density:
            padding = full_sequence - valid_sequence
            padding_n = qh.shape[0] * padding * padding
            self.last_stats = {
                "core_interactions": route.core_interactions + padding_n,
                "residual_token_interactions": route.residual_interactions,
                "residual_micro_tiles": route.residual_micro_tiles,
                "promoted_macro_tiles": route.promoted_tiles,
                "residual_count_mean": route.residual_count_mean,
                "residual_count_p50": route.residual_count_p50,
                "residual_count_p95": route.residual_count_p95,
                "residual_count_max": route.residual_count_max,
                "residual_count_nonempty_ratio": route.residual_count_nonempty_ratio,
                "total_possible": qh.shape[0] * (sequence * sequence + padding * padding),
            }
        _timing_stop(
            timing_recorder, density_start, "flashinfer_density_accounting",
            step_idx, layer_idx, q.device,
        )
        output_start = _timing_start(timing_recorder, q.device)
        output = restore(merged)
        _timing_stop(timing_recorder, output_start, "flashinfer_output_unpermute", step_idx, layer_idx, q.device)
        if not reuse_route:
            self.route = None
        return output


__all__ = [
    "FlashInfer64Attention", "FlashInfer64Route", "Q_MACRO", "K_MACRO", "MICRO",
    "compute_micro_tile_scores", "aggregate_macro_scores", "compute_64_tile_scores",
    "select_64_tiles", "select_64_tiles_from_scores", "select_64_tiles_topk_from_scores",
]
