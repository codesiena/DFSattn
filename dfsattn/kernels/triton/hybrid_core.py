"""Masked 64x64 Core kernel for the Hybrid-DFSAttn experiment.

The kernel computes one complete 64-token QK tile per program, applies the
original 4x4 (16x16 microblock) mask before softmax, and returns the partial
online-softmax state ``(m, l, a)``.  Keeping the mask at microblock level is
what makes threshold promotion semantics identical to DFSAttn.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in minimal installs
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _hybrid_core64_kernel(
        q_ptr, k_ptr, v_ptr, mask_ptr, qvalid_ptr, kvalid_ptr,
        m_ptr, l_ptr, a_ptr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        stride_q_tile: tl.constexpr,
        stride_k_tile: tl.constexpr,
        stride_v_tile: tl.constexpr,
        stride_mask_tile: tl.constexpr,
        stride_a_tile: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = tl.arange(0, 64)
        cols = tl.arange(0, 64)
        dims = tl.arange(0, BLOCK_D)

        q_offsets = pid * stride_q_tile + rows[:, None] * head_dim + dims[None, :]
        q = tl.load(q_ptr + q_offsets, mask=(dims[None, :] < head_dim), other=0.0)

        # K is loaded transposed directly from row-major [64, D] storage.
        k_offsets = pid * stride_k_tile + cols[None, :] * head_dim + dims[:, None]
        k_t = tl.load(k_ptr + k_offsets, mask=(dims[:, None] < head_dim), other=0.0)
        scores = tl.dot(q, k_t) * scale

        q_valid = tl.load(qvalid_ptr + pid * 64 + rows, mask=rows < 64, other=0)
        k_valid = tl.load(kvalid_ptr + pid * 64 + cols, mask=cols < 64, other=0)
        micro_q = rows // 16
        micro_k = cols // 16
        inner_mask = tl.load(
            mask_ptr + pid * stride_mask_tile + micro_q[:, None] * 4 + micro_k[None, :],
            mask=(micro_q[:, None] < 4) & (micro_k[None, :] < 4),
            other=0,
        )
        valid = inner_mask & q_valid[:, None] & k_valid[None, :]
        scores = tl.where(valid, scores, float("-inf"))

        row_m = tl.max(scores, axis=1)
        weights = tl.exp(scores - row_m[:, None])
        weights = tl.where(valid, weights, 0.0)
        row_l = tl.sum(weights, axis=1)

        v_offsets = pid * stride_v_tile + cols[:, None] * head_dim + dims[None, :]
        v = tl.load(v_ptr + v_offsets, mask=(dims[None, :] < head_dim), other=0.0)
        # Softmax weights are accumulated in FP32 for a stable LSE state.
        row_a = tl.dot(weights, v.to(tl.float32))

        tl.store(m_ptr + pid * 64 + rows, row_m)
        tl.store(l_ptr + pid * 64 + rows, row_l)
        tl.store(a_ptr + pid * stride_a_tile + rows[:, None] * head_dim + dims[None, :], row_a, mask=dims[None, :] < head_dim)

    @triton.jit
    def _hybrid_lse_merge_kernel(
        mc_ptr, lc_ptr, ac_ptr, mr_ptr, lr_ptr, ar_ptr,
        m_ptr, l_ptr, a_ptr,
        sequence, head_dim: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid // sequence
        row = pid - h * sequence
        dims = tl.arange(0, BLOCK_D)
        mc = tl.load(mc_ptr + h * sequence + row)
        lc = tl.load(lc_ptr + h * sequence + row)
        mr = tl.load(mr_ptr + h * sequence + row)
        lr = tl.load(lr_ptr + h * sequence + row)
        m = tl.maximum(mc, mr)
        sc = tl.where(mc != float("-inf"), tl.exp(mc - m), 0.0)
        sr = tl.where(mr != float("-inf"), tl.exp(mr - m), 0.0)
        tl.store(m_ptr + h * sequence + row, m)
        tl.store(l_ptr + h * sequence + row, sc * lc + sr * lr)
        ac = tl.load(ac_ptr + (h * sequence + row) * head_dim + dims, mask=dims < head_dim, other=0.0)
        ar = tl.load(ar_ptr + (h * sequence + row) * head_dim + dims, mask=dims < head_dim, other=0.0)
        tl.store(a_ptr + (h * sequence + row) * head_dim + dims, sc * ac + sr * ar, mask=dims < head_dim)


def core64_forward(
    q_tiles: torch.Tensor,
    k_tiles: torch.Tensor,
    v_tiles: torch.Tensor,
    core_masks: torch.Tensor,
    q_valid: torch.Tensor,
    k_valid: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run the Triton Core64 kernel, or return ``None`` when unavailable."""

    if triton is None or not q_tiles.is_cuda:
        return None
    if q_tiles.ndim != 3 or q_tiles.shape[1] != 64:
        raise ValueError("Core64 inputs must have shape [num_tiles, 64, head_dim]")
    if q_tiles.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Core64 Tensor-Core path requires FP16 or BF16 Q/K/V")
    n_tiles, _, head_dim = q_tiles.shape
    if head_dim > 256:
        raise ValueError("head_dim > 256 is not supported by the initial Core64 kernel")
    q_tiles = q_tiles.contiguous()
    k_tiles = k_tiles.contiguous()
    v_tiles = v_tiles.contiguous()
    core_masks = core_masks.contiguous().bool()
    q_valid = q_valid.contiguous().bool()
    k_valid = k_valid.contiguous().bool()
    m = torch.empty((n_tiles, 64), device=q_tiles.device, dtype=torch.float32)
    l = torch.empty_like(m)
    a = torch.empty((n_tiles, 64, head_dim), device=q_tiles.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    _hybrid_core64_kernel[(n_tiles,)](
        q_tiles, k_tiles, v_tiles, core_masks, q_valid, k_valid, m, l, a,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
        stride_q_tile=q_tiles.stride(0),
        stride_k_tile=k_tiles.stride(0),
        stride_v_tile=v_tiles.stride(0),
        stride_mask_tile=core_masks.stride(0),
        stride_a_tile=a.stride(0),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return m, l, a


def merge_online_states(
    core_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    residual_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint Core/Residual LSE merge; CUDA uses one small Triton kernel."""

    mc, lc, ac = core_state
    mr, lr, ar = residual_state
    if triton is None or not mc.is_cuda:
        m = torch.maximum(mc, mr)
        sc = torch.where(torch.isfinite(mc), torch.exp(mc - m), torch.zeros_like(m))
        sr = torch.where(torch.isfinite(mr), torch.exp(mr - m), torch.zeros_like(m))
        return m, sc * lc + sr * lr, sc[..., None] * ac + sr[..., None] * ar

    heads, sequence, head_dim = ac.shape
    m = torch.empty_like(mc)
    l = torch.empty_like(lc)
    a = torch.empty_like(ac)
    _hybrid_lse_merge_kernel[(heads * sequence,)](
        mc, lc, ac, mr, lr, ar, m, l, a,
        sequence=sequence,
        head_dim=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=2,
    )
    return m, l, a
