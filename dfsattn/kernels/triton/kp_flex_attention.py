"""Q-stationary Triton execution for DFSAttn KP masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


FINE_BLOCK = 16
COARSE_BLOCK = 128
QUERY_TILE = 64
KV_TILE = 64


@dataclass(frozen=True)
class KPExecutionPlan:
    kv_counts: torch.Tensor
    kv_indices: torch.Tensor
    fine_mask: torch.Tensor


if triton is not None:

    @triton.jit
    def _kp_attention_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        counts_ptr, indices_ptr, fine_mask_ptr,
        cu_seqlens_q_ptr, cu_seqlens_kv_ptr,
        sequence,
        stride_qh: tl.constexpr, stride_qm: tl.constexpr,
        stride_kh: tl.constexpr, stride_kn: tl.constexpr,
        stride_vh: tl.constexpr, stride_vn: tl.constexpr,
        stride_oh: tl.constexpr, stride_om: tl.constexpr,
        stride_count_h: tl.constexpr,
        stride_index_h: tl.constexpr, stride_index_q: tl.constexpr,
        stride_mask_h: tl.constexpr, stride_mask_q: tl.constexpr,
        head_dim: tl.constexpr, scale: tl.constexpr, BLOCK_D: tl.constexpr,
        HAS_SEQUENCE_SPLIT: tl.constexpr,
    ):
        q_tile = tl.program_id(0)
        head = tl.program_id(1)
        rows = q_tile * 64 + tl.arange(0, 64)
        cols = tl.arange(0, 64)
        dims = tl.arange(0, BLOCK_D)
        q_valid = rows < sequence

        q_offsets = head * stride_qh + rows[:, None] * stride_qm + dims[None, :]
        q = tl.load(
            q_ptr + q_offsets,
            mask=q_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        m_i = tl.full((64,), float("-inf"), tl.float32)
        l_i = tl.zeros((64,), tl.float32)
        acc = tl.zeros((64, BLOCK_D), tl.float32)

        coarse_q = (q_tile * 64) // 128
        coarse_count = tl.load(counts_ptr + head * stride_count_h + coarse_q)
        # Each selected K128 candidate is consumed as two K64 tiles.
        for kv_iter in tl.range(0, coarse_count * 2):
            candidate_slot = kv_iter // 2
            half = kv_iter - candidate_slot * 2
            coarse_k = tl.load(
                indices_ptr
                + head * stride_index_h
                + coarse_q * stride_index_q
                + candidate_slot
            )
            key_start = coarse_k * 128 + half * 64
            key_rows = key_start + cols
            k_valid = key_rows < sequence
            k_offsets = (
                head * stride_kh
                + key_rows[None, :] * stride_kn
                + dims[:, None]
            )
            k_t = tl.load(
                k_ptr + k_offsets,
                mask=(dims[:, None] < head_dim) & k_valid[None, :],
                other=0.0,
            )
            scores = tl.dot(q, k_t) * scale

            fine_q = rows // 16
            fine_k = key_rows // 16
            logical_mask = tl.load(
                fine_mask_ptr
                + head * stride_mask_h
                + fine_q[:, None] * stride_mask_q
                + fine_k[None, :],
                mask=q_valid[:, None] & k_valid[None, :],
                other=0,
            )
            valid = q_valid[:, None] & k_valid[None, :] & logical_mask
            if HAS_SEQUENCE_SPLIT:
                # HunyuanVideo packs the effective video+text prefix and the
                # padded text tail into one tensor.  Upstream DFSAttn passes
                # cu_seqlens=[0, effective_length, total_length] to keep those
                # two segments isolated.  Applying the same token-level
                # boundary here prevents every video query from attending to
                # padding K/V through the otherwise-dense text region.
                q_split = tl.load(cu_seqlens_q_ptr + 1)
                kv_split = tl.load(cu_seqlens_kv_ptr + 1)
                q_segment = rows >= q_split
                kv_segment = key_rows >= kv_split
                valid &= q_segment[:, None] == kv_segment[None, :]
            scores = tl.where(valid, scores, float("-inf"))

            tile_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_m)
            alpha = tl.where(m_i != float("-inf"), tl.exp(m_i - m_new), 0.0)
            probabilities = tl.where(valid, tl.exp(scores - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(probabilities, axis=1)
            acc = acc * alpha[:, None]

            v_offsets = (
                head * stride_vh
                + key_rows[:, None] * stride_vn
                + dims[None, :]
            )
            v = tl.load(
                v_ptr + v_offsets,
                mask=k_valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            )
            acc += tl.dot(probabilities.to(q_ptr.dtype.element_ty), v)
            m_i = m_new

        output = acc / l_i[:, None]
        out_offsets = head * stride_oh + rows[:, None] * stride_om + dims[None, :]
        tl.store(
            out_ptr + out_offsets,
            output,
            mask=q_valid[:, None] & (dims[None, :] < head_dim),
        )


def compact_block_mask(mask: torch.Tensor, block_size: int = COARSE_BLOCK) -> KPExecutionPlan:
    """Compact a fine bool mask into per-head/per-Q128 candidate lists."""
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("KP logical mask must be bool [heads, q16, k16]")
    if block_size != COARSE_BLOCK:
        raise ValueError("the initial KP kernel requires 128-token coarse blocks")

    heads, fine_q_blocks, fine_k_blocks = mask.shape
    ratio = COARSE_BLOCK // FINE_BLOCK
    coarse_q_blocks = math.ceil(fine_q_blocks / ratio)
    coarse_k_blocks = math.ceil(fine_k_blocks / ratio)
    if fine_q_blocks % ratio == 0 and fine_k_blocks % ratio == 0:
        # Reuse the cached logical mask.  At video scale one dense bool mask is
        # ~188 MB, so an unconditional padded copy per layer is prohibitive.
        padded = mask if mask.is_contiguous() else mask.contiguous()
    else:
        padded = torch.zeros(
            (heads, coarse_q_blocks * ratio, coarse_k_blocks * ratio),
            dtype=torch.bool, device=mask.device,
        )
        padded[:, :fine_q_blocks, :fine_k_blocks] = mask
    coarse = padded.view(
        heads, coarse_q_blocks, ratio, coarse_k_blocks, ratio
    ).any(dim=(2, 4))

    counts = coarse.sum(dim=-1, dtype=torch.int32)
    max_count = int(counts.max().item())
    if max_count == 0 or bool((counts == 0).any().item()):
        raise ValueError("KP logical mask leaves at least one Q128 block empty")
    head_idx, q_idx, k_idx = coarse.nonzero(as_tuple=True)
    row_id = head_idx * coarse_q_blocks + q_idx
    flat_counts = counts.flatten()
    row_offsets = flat_counts.cumsum(dim=0) - flat_counts
    rank = torch.arange(k_idx.numel(), device=mask.device) - row_offsets[row_id]
    packed = torch.zeros(
        (heads * coarse_q_blocks * max_count,), dtype=torch.int32, device=mask.device
    )
    packed[row_id * max_count + rank] = k_idx.to(torch.int32)
    indices = packed.view(heads, coarse_q_blocks, max_count)
    return KPExecutionPlan(counts.contiguous(), indices.contiguous(), padded.contiguous())


def kp_fine_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: KPExecutionPlan,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute KP with resident Q64 tiles and an online-softmax KV loop."""
    if triton is None:
        raise ImportError("KP execution requires Triton")
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("KP Tensor-Core execution requires CUDA FP16/BF16 Q/K/V")
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError("KP execution expects matching [1,H,S,D] Q/K/V")
    _, heads, sequence, head_dim = q.shape
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor(
            [0, sequence], dtype=torch.int32, device=q.device
        )
    if cu_seqlens_kv is None:
        cu_seqlens_kv = torch.tensor(
            [0, sequence], dtype=torch.int32, device=q.device
        )
    if cu_seqlens_q.device != q.device or cu_seqlens_kv.device != q.device:
        raise ValueError("KP cu_seqlens tensors must be on the same device as Q/K/V")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_kv.dtype != torch.int32:
        raise TypeError("KP cu_seqlens tensors must use torch.int32")
    if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
        raise ValueError("KP Q and KV cu_seqlens must contain the same number of segments")
    if cu_seqlens_q.numel() not in (2, 3):
        raise ValueError(
            "KP execution currently supports one sequence [0,S] or Hunyuan's "
            "effective/padding split [0,effective,S]"
        )
    has_sequence_split = cu_seqlens_q.numel() == 3
    if head_dim > 256:
        raise ValueError("KP execution currently supports head_dim <= 256")
    output = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    _kp_attention_kernel[(triton.cdiv(sequence, QUERY_TILE), heads)](
        q, k, v, output,
        plan.kv_counts, plan.kv_indices, plan.fine_mask,
        cu_seqlens_q, cu_seqlens_kv,
        sequence,
        stride_qh=q.stride(1), stride_qm=q.stride(2),
        stride_kh=k.stride(1), stride_kn=k.stride(2),
        stride_vh=v.stride(1), stride_vn=v.stride(2),
        stride_oh=output.stride(1), stride_om=output.stride(2),
        stride_count_h=plan.kv_counts.stride(0),
        stride_index_h=plan.kv_indices.stride(0),
        stride_index_q=plan.kv_indices.stride(1),
        stride_mask_h=plan.fine_mask.stride(0),
        stride_mask_q=plan.fine_mask.stride(1),
        head_dim=head_dim, scale=head_dim ** -0.5, BLOCK_D=block_d,
        HAS_SEQUENCE_SPLIT=has_sequence_split,
        num_warps=8, num_stages=3,
    )
    return output
