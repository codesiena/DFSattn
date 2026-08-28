"""Hybrid 64x64/16x16 execution for a DFSAttn logical block mask.

This is deliberately an execution-backend experiment: it never promotes the
*logical* mask.  A promoted 64x64 tile computes its complete QK product, but
the unselected 16x16 microtiles are still assigned ``-inf`` before softmax.
Consequently, replacing ``fine_sparse_attention`` with
``hybrid_sparse_attention`` leaves the DFS selector unchanged.

The implementation groups all 64x64 tiles and all residual 16x16 tiles into
two batched matmuls.  On CUDA with FP16/BF16 inputs, the 64x64 matmul is a
Tensor-Core-eligible operation (the selected PyTorch backend still determines
the final kernel).  The code is also intentionally runnable on CPU, where it
serves as a correctness reference and planning-overhead benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from .kernels.triton.hybrid_core import core64_forward, merge_online_states
except ImportError:  # Optional Triton dependency.
    core64_forward = None
    merge_online_states = None


FINE_BLOCK = 16
MACRO_BLOCK = 64


@dataclass(frozen=True)
class HybridPlan:
    """GPU-resident partition of one DFS logical mask.

    ``core`` and ``residual`` contain ``(head, q_block, k_block)`` indices.
    Core block indices are in 64-token units, while residual indices are in
    16-token units.  ``core_micro_masks`` preserves the original 4x4 mask.
    """

    core: torch.Tensor
    residual: torch.Tensor
    residual_mask: torch.Tensor
    core_micro_masks: torch.Tensor
    fine_block: int
    macro_block: int
    threshold: int

    @property
    def core_tiles(self) -> int:
        return int(self.core.shape[0])

    @property
    def residual_tiles(self) -> int:
        return int(self.residual.shape[0])


def _validate_mask(mask: torch.Tensor, fine_block: int, macro_block: int) -> int:
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("block_mask must be a bool tensor shaped [heads, q16_blocks, k16_blocks]")
    if macro_block % fine_block:
        raise ValueError("macro_block must be an integer multiple of fine_block")
    ratio = macro_block // fine_block
    if ratio != 4:
        raise ValueError("the initial experiment is intentionally fixed to 4x4 16x16 -> 64x64 grouping")
    return ratio


def partition_block_mask(
    block_mask: torch.Tensor,
    *,
    threshold: int,
    fine_block: int = FINE_BLOCK,
    macro_block: int = MACRO_BLOCK,
) -> HybridPlan:
    """Split a DFS 16x16 logical mask into Core64 and Residual16 lists.

    All operations stay on ``block_mask.device``.  This is important when
    timing: the reported planning cost includes occupancy, thresholding, and
    index construction, but does not hide a GPU->CPU round trip.
    """

    ratio = _validate_mask(block_mask, fine_block, macro_block)
    if not 1 <= threshold <= ratio * ratio:
        raise ValueError(f"threshold must be in [1, {ratio * ratio}], got {threshold}")

    heads, q_blocks, k_blocks = block_mask.shape
    padded_q = math.ceil(q_blocks / ratio) * ratio
    padded_k = math.ceil(k_blocks / ratio) * ratio
    if padded_q != q_blocks or padded_k != k_blocks:
        padded = torch.zeros((heads, padded_q, padded_k), dtype=torch.bool, device=block_mask.device)
        padded[:, :q_blocks, :k_blocks] = block_mask
    else:
        padded = block_mask

    # [H, Q64, K64, 4(q16), 4(k16)]
    grouped = padded.view(heads, padded_q // ratio, ratio, padded_k // ratio, ratio)
    grouped = grouped.permute(0, 1, 3, 2, 4).contiguous()
    promote = grouped.sum(dim=(-1, -2)) >= threshold

    core = promote.nonzero(as_tuple=False)
    core_masks = grouped[core[:, 0], core[:, 1], core[:, 2]]

    residual_mask = grouped & ~promote[..., None, None]
    residual_coords = residual_mask.nonzero(as_tuple=False)
    residual = torch.stack(
        (
            residual_coords[:, 0],
            residual_coords[:, 1] * ratio + residual_coords[:, 3],
            residual_coords[:, 2] * ratio + residual_coords[:, 4],
        ),
        dim=1,
    )
    # Flatten grouped [H,Q64,K64,4,4] back to the native fine-mask layout.
    residual_mask_fine = residual_mask.permute(0, 1, 3, 2, 4).reshape(heads, padded_q, padded_k)
    return HybridPlan(core, residual, residual_mask_fine, core_masks, fine_block, macro_block, threshold)


def _pad_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError("q, k, v must have matching [1, heads, sequence, head_dim] shapes")
    sequence = q.shape[2]
    padded_sequence = math.ceil(sequence / block) * block
    if padded_sequence == sequence:
        return q, k, v, sequence
    pad = (0, 0, 0, padded_sequence - sequence)
    return F.pad(q, pad), F.pad(k, pad), F.pad(v, pad), sequence


def _merge_state(
    state_m: torch.Tensor,
    state_l: torch.Tensor,
    state_o: torch.Tensor,
    heads: torch.Tensor,
    starts: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_o: torch.Tensor,
    valid_rows: torch.Tensor,
) -> None:
    """Online-softmax merge of a batch of independently evaluated tiles."""

    for tile in range(heads.numel()):
        rows = valid_rows[tile]
        h = heads[tile]
        positions = starts[tile] + torch.arange(rows.numel(), device=heads.device)
        positions = positions[rows]
        old_m = state_m[h, positions]
        old_l = state_l[h, positions]
        old_o = state_o[h, positions]
        new_m = partial_m[tile, rows]
        new_l = partial_l[tile, rows]
        new_o = partial_o[tile, rows]
        merged_m = torch.maximum(old_m, new_m)
        old_scale = torch.exp(old_m - merged_m)
        new_scale = torch.exp(new_m - merged_m)
        state_l[h, positions] = old_scale * old_l + new_scale * new_l
        state_o[h, positions] = old_scale.unsqueeze(-1) * old_o + new_scale.unsqueeze(-1) * new_o
        state_m[h, positions] = merged_m


def _run_tiles(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tile_indices: torch.Tensor,
    *,
    tile_size: int,
    state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    logical_masks: torch.Tensor | None = None,
    original_sequence: int,
    return_tiles: bool = False,
):
    if tile_indices.numel() == 0:
        return None
    state_m, state_l, state_o = state
    _, heads, padded_sequence, head_dim = q.shape
    q_blocks = q[0].view(heads, padded_sequence // tile_size, tile_size, head_dim)
    k_blocks = k[0].view(heads, padded_sequence // tile_size, tile_size, head_dim)
    v_blocks = v[0].view(heads, padded_sequence // tile_size, tile_size, head_dim)
    h, q_block, k_block = tile_indices.unbind(dim=1)
    q_tile = q_blocks[h, q_block]
    k_tile = k_blocks[h, k_block]
    v_tile = v_blocks[h, k_block]

    # The CUDA path is a real 64x64 masked Core kernel.  It returns the
    # partial state required by the later joint LSE merge.  CPU or unsupported
    # dtypes use the reference implementation below.
    if logical_masks is not None and core64_forward is not None:
        positions = torch.arange(tile_size, device=q.device)
        q_valid = q_block[:, None] * tile_size + positions[None, :] < original_sequence
        k_valid = k_block[:, None] * tile_size + positions[None, :] < original_sequence
        triton_state = core64_forward(q_tile, k_tile, v_tile, logical_masks, q_valid, k_valid)
        if triton_state is not None:
            partial_m, partial_l, partial_o = triton_state
            valid_rows = logical_masks.repeat_interleave(FINE_BLOCK, dim=1).any(dim=-1)
            valid_rows = valid_rows & q_valid
            if return_tiles:
                return h, q_block * tile_size, partial_m, partial_l, partial_o, valid_rows
            _merge_state(
                state_m, state_l, state_o, h, q_block * tile_size,
                partial_m, partial_l, partial_o, valid_rows,
            )
            return

    # Keep the QK matmul in input precision so CUDA BF16/FP16 dispatch remains
    # Tensor-Core eligible.  Softmax state itself is accumulated in FP32.
    scores = torch.bmm(q_tile, k_tile.transpose(1, 2)).float() / math.sqrt(head_dim)
    token_mask = torch.ones_like(scores, dtype=torch.bool)
    if logical_masks is not None:
        ratio = tile_size // FINE_BLOCK
        token_mask = logical_masks.repeat_interleave(FINE_BLOCK, dim=1).repeat_interleave(FINE_BLOCK, dim=2)
        if ratio != 4:
            raise AssertionError("unexpected macro tile ratio")

    # Last partial blocks have padded tokens.  They must never contribute to a
    # real row's normalizer, even when their parent 16x16 microblock is active.
    positions = torch.arange(tile_size, device=q.device)
    q_valid = q_block[:, None] * tile_size + positions[None, :] < original_sequence
    k_valid = k_block[:, None] * tile_size + positions[None, :] < original_sequence
    token_mask = token_mask & q_valid[:, :, None] & k_valid[:, None, :]
    scores = scores.masked_fill(~token_mask, float("-inf"))
    valid_rows = token_mask.any(dim=-1)
    partial_m = scores.amax(dim=-1)
    weights = torch.where(valid_rows[:, :, None], torch.exp(scores - partial_m[:, :, None]), torch.zeros_like(scores))
    partial_l = weights.sum(dim=-1)
    partial_o = torch.bmm(weights.to(v_tile.dtype), v_tile).float()
    if return_tiles:
        return h, q_block * tile_size, partial_m, partial_l, partial_o, valid_rows
    _merge_state(state_m, state_l, state_o, h, q_block * tile_size, partial_m, partial_l, partial_o, valid_rows)


def _aggregate_tile_states(tile_state, heads: int, padded_sequence: int, head_dim: int):
    """Reduce per-tile partial states into one state per query row.

    This is the branch-local reduction before the final Core/Residual merge.
    It uses vectorized scatter operations instead of a Python loop over every
    tile, so the CUDA path remains a small number of GPU launches.
    """

    if tile_state is None:
        return _empty_state(heads, padded_sequence, head_dim, device=None)
    tile_heads, starts, pm, pl, pa, valid = tile_state
    device = pm.device
    row = starts[:, None] + torch.arange(pm.shape[1], device=device)[None, :]
    flat_valid = valid.reshape(-1) & (row.reshape(-1) < padded_sequence)
    index = tile_heads[:, None] * padded_sequence + row
    index = index.reshape(-1)[flat_valid]
    pm = pm.reshape(-1)[flat_valid]
    pl = pl.reshape(-1)[flat_valid]
    pa = pa.reshape(-1, head_dim)[flat_valid]
    total = heads * padded_sequence
    m = torch.full((total,), float("-inf"), device=device, dtype=torch.float32)
    m.scatter_reduce_(0, index, pm, reduce="amax", include_self=True)
    scale = torch.exp(pm - m[index])
    l = torch.zeros((total,), device=device, dtype=torch.float32)
    l.scatter_add_(0, index, scale * pl)
    a = torch.zeros((total, head_dim), device=device, dtype=torch.float32)
    a.index_add_(0, index, scale[:, None] * pa)
    return m.view(heads, padded_sequence), l.view(heads, padded_sequence), a.view(heads, padded_sequence, head_dim)


def _empty_state(heads: int, padded_sequence: int, head_dim: int, device=None):
    return (
        torch.full((heads, padded_sequence), float("-inf"), device=device, dtype=torch.float32),
        torch.zeros((heads, padded_sequence), device=device, dtype=torch.float32),
        torch.zeros((heads, padded_sequence, head_dim), device=device, dtype=torch.float32),
    )


def _init_state(q: torch.Tensor, padded_sequence: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads, head_dim = q.shape[1], q.shape[-1]
    return (
        torch.full((heads, padded_sequence), float("-inf"), device=q.device, dtype=torch.float32),
        torch.zeros((heads, padded_sequence), device=q.device, dtype=torch.float32),
        torch.zeros((heads, padded_sequence, head_dim), device=q.device, dtype=torch.float32),
    )


def _finalize_state(state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], original_sequence: int, dtype: torch.dtype) -> torch.Tensor:
    _, l, o = state
    if bool((l[:, :original_sequence] == 0).any()):
        raise ValueError("the logical mask leaves at least one query token without an active key block")
    return (o[:, :original_sequence] / l[:, :original_sequence, None]).to(dtype).unsqueeze(0)


def _native_residual_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    residual_mask: torch.Tensor,
    *,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run the original block-sparse CUDA kernel and expose its LSE state.

    The installed upstream extension requires m/n block dimensions to be
    multiples of 128, so it cannot execute this Hybrid backend's 16x16
    residual mask.  Returning ``None`` selects the local tiled reference
    implementation in the caller without modifying the upstream package.
    """

    if not q.is_cuda or block_size % 128 != 0:
        return None
    try:
        from block_sparse_attn.block_sparse_attn_interface import BlockSparseAttnFunc
    except ImportError as exc:
        raise ImportError("Hybrid CUDA residual path requires block_sparse_attn.") from exc

    _, heads, sequence, head_dim = q.shape
    q_bs = q[0].permute(1, 0, 2).contiguous()
    k_bs = k[0].permute(1, 0, 2).contiguous()
    v_bs = v[0].permute(1, 0, 2).contiguous()
    cu = torch.tensor([0, sequence], dtype=torch.int32, device=q.device)
    head_mask_type = torch.ones((heads,), dtype=torch.int32, device=q.device)
    base_mask = residual_mask.unsqueeze(0).contiguous()
    out, softmax_lse, _ = BlockSparseAttnFunc.apply(
        q_bs, k_bs, v_bs,
        cu, cu,
        block_size, block_size,
        head_mask_type,
        None,
        base_mask,
        sequence, sequence,
        0.0,
        head_dim ** (-0.5),
        False,
        False,
        True,
        -1, -1,
        False,
        False,
    )
    lse = softmax_lse[:, :, :sequence].squeeze(0)
    l = torch.exp(lse)
    a = out[:sequence].permute(1, 0, 2).float() * l[:, :, None]
    return lse, l, a


def hybrid_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    threshold: int = 8,
    fine_block: int = FINE_BLOCK,
    macro_block: int = MACRO_BLOCK,
    return_plan: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, HybridPlan]:
    """Execute the same 16x16 DFS mask through Core64 + Residual16 paths."""

    plan = partition_block_mask(block_mask, threshold=threshold, fine_block=fine_block, macro_block=macro_block)
    output = hybrid_sparse_attention_from_plan(q, k, v, plan)
    return (output, plan) if return_plan else output


def hybrid_sparse_attention_from_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: HybridPlan,
    *,
    timing_recorder=None,
    step_idx: int = 0,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Execute an already partitioned plan (Core64 + Residual16 + LSE merge).

    Keeping this separate from :func:`hybrid_sparse_attention` makes the
    benchmark's planning column exact: callers can time partition and execute
    independently, then also time the end-to-end wrapper.
    """

    q, k, v, original_sequence = _pad_qkv(q, k, v, plan.macro_block)
    core_state = _init_state(q, q.shape[2])
    core_start = timing_recorder.start(q.device) if timing_recorder is not None else None
    core_tiles = _run_tiles(q, k, v, plan.core, tile_size=plan.macro_block, state=core_state, logical_masks=plan.core_micro_masks, original_sequence=original_sequence, return_tiles=True)
    if core_tiles is not None:
        core_state = _aggregate_tile_states(core_tiles, q.shape[1], q.shape[2], q.shape[-1])
    if timing_recorder is not None:
        timing_recorder.stop(core_start, phase="hybrid_core64", step_idx=step_idx, layer_idx=layer_idx, device=q.device)

    residual_state = None
    if plan.residual_tiles:
        residual_start = timing_recorder.start(q.device) if timing_recorder is not None else None
        residual_state = _native_residual_state(
            q, k, v, plan.residual_mask, block_size=plan.fine_block
        )
    if residual_state is None:
        residual_state = _init_state(q, q.shape[2])
        residual_tiles = _run_tiles(q, k, v, plan.residual, tile_size=plan.fine_block, state=residual_state, original_sequence=original_sequence, return_tiles=True)
        if residual_tiles is not None:
            residual_state = _aggregate_tile_states(residual_tiles, q.shape[1], q.shape[2], q.shape[-1])
    if plan.residual_tiles and timing_recorder is not None:
        timing_recorder.stop(residual_start, phase="hybrid_residual16", step_idx=step_idx, layer_idx=layer_idx, device=q.device)

    merge_start = timing_recorder.start(q.device) if timing_recorder is not None else None
    if merge_online_states is not None:
        m, l, a = merge_online_states(core_state, residual_state)
    else:
        mc, lc, ac = core_state
        mr, lr, ar = residual_state
        m = torch.maximum(mc, mr)
        core_scale = torch.where(torch.isfinite(mc), torch.exp(mc - m), torch.zeros_like(m))
        residual_scale = torch.where(torch.isfinite(mr), torch.exp(mr - m), torch.zeros_like(m))
        l = core_scale * lc + residual_scale * lr
        a = core_scale[..., None] * ac + residual_scale[..., None] * ar
    if timing_recorder is not None:
        timing_recorder.stop(merge_start, phase="hybrid_lse_merge", step_idx=step_idx, layer_idx=layer_idx, device=q.device)
    output = _finalize_state((m, l, a), original_sequence, q.dtype)
    return output


def fine_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: torch.Tensor, *, fine_block: int = FINE_BLOCK) -> torch.Tensor:
    """Fine-16 execution reference used for same-mask correctness/latency tests."""

    _validate_mask(block_mask, fine_block, MACRO_BLOCK)
    q, k, v, original_sequence = _pad_qkv(q, k, v, fine_block)
    state = _init_state(q, q.shape[2])
    fine_indices = block_mask.nonzero(as_tuple=False)
    _run_tiles(q, k, v, fine_indices, tile_size=fine_block, state=state, original_sequence=original_sequence)
    return _finalize_state(state, original_sequence, q.dtype)


def plan_statistics(plan: HybridPlan) -> Dict[str, int]:
    """Return logical and actual QK counts for a partitioned mask."""

    logical_qk = (plan.residual_tiles + int(plan.core_micro_masks.sum().item())) * plan.fine_block**2
    actual_qk = plan.residual_tiles * plan.fine_block**2 + plan.core_tiles * plan.macro_block**2
    return {
        "core_tiles_64": plan.core_tiles,
        "residual_tiles_16": plan.residual_tiles,
        "logical_qk": logical_qk,
        "actual_qk": actual_qk,
    }
