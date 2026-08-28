"""64x64 FlexAttention backend used by the tensor-tile-cover prototype.

The upstream ``block_sparse_attn`` extension accepts a public 128x128 block
mask only.  This module keeps the *execution* mask at 64x64 and converts the
per-query-block top-k indices directly into PyTorch FlexAttention's compact
``BlockMask`` representation.  It deliberately does not implement a
pair-level selector: that requires a separate streamed/quantized scorer.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on CPU-only installs.
    triton = None
    tl = None

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
except (ImportError, AttributeError):  # PyTorch < 2.5 or a build without FlexAttention.
    BlockMask = None
    flex_attention = None


FLEX_BLOCK_SIZE = 64

if triton is not None:

    @triton.jit
    def _flex64_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        counts_ptr, indices_ptr,
        sequence,
        stride_qh, stride_qm, stride_kh, stride_km,
        stride_vh, stride_vm, stride_oh, stride_om,
        stride_count_h, stride_index_h, stride_index_q,
        head_dim: tl.constexpr, block_d: tl.constexpr,
        max_k: tl.constexpr, scale: tl.constexpr,
    ):
        q_tile = tl.program_id(0)
        head = tl.program_id(1)
        rows = q_tile * 64 + tl.arange(0, 64)
        cols = tl.arange(0, 64)
        dims = tl.arange(0, block_d)
        q_valid = rows < sequence

        q_offsets = head * stride_qh + rows[:, None] * stride_qm + dims[None, :]
        q = tl.load(
            q_ptr + q_offsets,
            mask=q_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        m_i = tl.full((64,), float("-inf"), tl.float32)
        l_i = tl.zeros((64,), tl.float32)
        acc = tl.zeros((64, block_d), tl.float32)
        count = tl.load(counts_ptr + head * stride_count_h + q_tile)

        for slot in tl.range(0, max_k):
            key_block = tl.load(
                indices_ptr
                + head * stride_index_h
                + q_tile * stride_index_q
                + slot
            )
            key_rows = key_block * 64 + cols
            k_valid = (slot < count) & (key_rows < sequence)
            k_offsets = head * stride_kh + key_rows[None, :] * stride_km + dims[:, None]
            k_t = tl.load(
                k_ptr + k_offsets,
                mask=(dims[:, None] < head_dim) & k_valid[None, :],
                other=0.0,
            )
            scores = tl.dot(q, k_t) * scale
            scores = tl.where(q_valid[:, None] & k_valid[None, :], scores, float("-inf"))

            tile_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_m)
            alpha = tl.where(m_i != float("-inf"), tl.exp(m_i - m_new), 0.0)
            probabilities = tl.where(
                q_valid[:, None] & k_valid[None, :],
                tl.exp(scores - m_new[:, None]),
                0.0,
            )
            l_i = l_i * alpha + tl.sum(probabilities, axis=1)
            acc = acc * alpha[:, None]

            v_offsets = head * stride_vh + key_rows[:, None] * stride_vm + dims[None, :]
            v = tl.load(
                v_ptr + v_offsets,
                mask=k_valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            )
            acc += tl.dot(probabilities.to(q_ptr.dtype.element_ty), v)
            m_i = m_new

        output = acc / l_i[:, None]
        out_offsets = head * stride_oh + rows[:, None] * stride_om + dims[None, :]
        tl.store(out_ptr + out_offsets, output, mask=q_valid[:, None] & (dims[None, :] < head_dim))


def build_flex64_block_mask(topk_indices: torch.Tensor, *, kv_blocks: int) -> Any:
    """Create a 64x64 FlexAttention BlockMask from uniform per-row top-k IDs.

    Args:
        topk_indices: ``[H, Q64, k]`` or ``[B, H, Q64, k]`` key-block IDs.
        kv_blocks: Total number of 64-token key blocks, including a possible
            final partial block.

    Every selected block is marked as a *full* block.  FlexAttention still
    guards sequence boundaries in the final partial tile; marking it full only
    avoids invoking a Python ``mask_mod`` for the normal sparse tiles.
    """
    if BlockMask is None:
        raise ImportError(
            "flex64 execution requires PyTorch FlexAttention (PyTorch 2.5+)."
        )
    if topk_indices.ndim == 3:
        topk_indices = topk_indices.unsqueeze(0)
    if topk_indices.ndim != 4:
        raise ValueError(
            "topk_indices must have shape [H, Q64, k] or [B, H, Q64, k]."
        )
    if topk_indices.shape[-1] < 1:
        raise ValueError("flex64 requires at least one selected K64 block per Q64 block.")
    # Avoid a GPU->CPU synchronization in the cached inference path.  Indices
    # produced by ``torch.topk`` in DFS_Attention are in range by construction;
    # retain the inexpensive value check for CPU/unit-test callers.
    if not topk_indices.is_cuda and topk_indices.numel() and (
        int(topk_indices.min().item()) < 0 or int(topk_indices.max().item()) >= kv_blocks
    ):
        raise ValueError("topk_indices contains an out-of-range K64 block ID.")

    selected = topk_indices.to(dtype=torch.int32).contiguous()
    if selected.shape[-1] > kv_blocks:
        raise ValueError("topk_indices selects more K64 blocks than exist.")
    # FlexAttention infers the full K extent from the final metadata dimension,
    # so it must be ``kv_blocks`` rather than the selected count ``k``.
    indices = torch.zeros(
        (*selected.shape[:-1], kv_blocks), dtype=torch.int32, device=selected.device
    )
    indices[..., : selected.shape[-1]] = selected
    counts = torch.full(
        selected.shape[:-1], selected.shape[-1], dtype=torch.int32, device=selected.device
    )
    # ``from_kv_blocks`` requires both partial and full representations.  The
    # selected tiles are all full tiles, so the partial representation has zero
    # counts and inert indices of the same metadata shape.
    partial_counts = torch.zeros_like(counts)
    partial_indices = torch.zeros_like(indices)
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_kv_num_blocks=counts,
        full_kv_indices=indices,
        BLOCK_SIZE=(FLEX_BLOCK_SIZE, FLEX_BLOCK_SIZE),
    )


def topk_indices_from_mask(block_mask: torch.Tensor) -> torch.Tensor:
    """Recover uniform per-row selections from a bool ``[H, Q, K]`` mask.

    This is only a compatibility fallback for masks created before ``flex64``
    began caching indices.  The normal path retains the original ``torch.topk``
    result and never performs this extra operation.
    """
    if block_mask.ndim != 3 or block_mask.dtype != torch.bool:
        raise ValueError("block_mask must be a bool tensor shaped [H, Q64, K64].")
    counts = block_mask.sum(dim=-1)
    if not bool((counts == counts.flatten()[0]).all()):
        raise ValueError("flex64 currently requires the same k for every Q64 row.")
    k = int(counts.flatten()[0].item())
    if k < 1:
        raise ValueError("flex64 requires at least one selected K64 block per Q64 row.")
    return torch.topk(block_mask.to(torch.int8), k=k, dim=-1).indices.to(torch.int32)


def flex64_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Any,
    topk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run non-causal 64x64 block-sparse attention on ``[B,H,S,D]`` QKV.

    The production path is a direct Triton Q64/K64 online-softmax kernel.  The
    ``BlockMask`` argument is retained for metadata/debugging compatibility;
    execution uses the original top-k IDs so no BlockMask-to-index conversion
    is needed in the hot path.
    """
    if not q.is_cuda:
        raise ValueError("flex64 execution requires CUDA.")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have identical [B, H, S, D] shapes.")
    if triton is None:
        raise ImportError("flex64 execution requires Triton.")
    if topk_indices is None:
        raise ValueError("flex64 execution requires the cached top-k K64 indices.")
    if q.shape[0] != 1 or topk_indices.ndim != 3:
        raise ValueError("flex64 expects batch size 1 and topk_indices shaped [H,Q64,k].")
    _, heads, sequence, head_dim = q.shape
    if topk_indices.shape[0] != heads or topk_indices.shape[1] != (sequence + 63) // 64:
        raise ValueError("topk_indices does not match the QKV sequence shape.")
    counts = torch.full(
        topk_indices.shape[:-1], topk_indices.shape[-1], dtype=torch.int32, device=q.device
    )
    indices = topk_indices.to(device=q.device, dtype=torch.int32).contiguous()
    output = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    _flex64_kernel[(triton.cdiv(sequence, 64), heads)](
        q, k, v, output, counts, indices, sequence,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2),
        v.stride(1), v.stride(2), output.stride(1), output.stride(2),
        counts.stride(0), indices.stride(0), indices.stride(1),
        head_dim=head_dim, block_d=block_d, max_k=indices.shape[-1], scale=head_dim ** -0.5,
        num_warps=8, num_stages=3,
    )
    return output
