"""Independent 64x64 FlashInfer + token-TopK DFSAttn backend.

This module is intentionally separate from the historical ``native``,
``hybrid``, ``flex64`` and ``kp`` implementations.  Its execution contract is

    64x64 proxy selector -> FlashInfer block-sparse attention
    unselected tiles     -> per-query token Top-k Triton attention
    both paths            -> LSE merge

The proxy selector uses the same mean-pooled tile scores for both routes.  The
residual execution kernel is Triton; residual token candidates are obtained by
expanding those existing tile scores, so token-level QK is not recomputed by
the selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

try:  # FlashInfer is optional for import-time and CPU reference tests.
    import flashinfer
    from flashinfer.sparse import (
        VariableBlockSparseAttentionWrapper as FlashInferVariableBlockSparseAttention,
    )
except Exception:  # pragma: no cover - depends on the deployment image.
    flashinfer = None
    FlashInferVariableBlockSparseAttention = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installs.
    triton = None
    tl = None


BLOCK_SIZE = 64


@dataclass
class FlashInfer64Route:
    """Cached route for one layer/shape.

    ``core_mask`` is indexed as ``[head, q64_block, k64_block]``.  The route
    is kept on the GPU and can be reused while the caller's diffusion cache
    interval is active.
    """

    core_mask: torch.Tensor
    sequence: int
    padded_sequence: int
    video_len: int
    route_mode: str
    top_p: float
    tile_top_ratio: float
    token_top_k: int
    token_top_ratio: float
    token_top_p: float


# VariableBlockSparseAttentionWrapper expands the block route into token-level
# indices during plan().  Keeping one wrapper per transformer layer therefore
# retains one large expanded route per layer.  The model visits layers
# sequentially, so one official wrapper can be reused safely; only the small
# selector route remains cached per layer.
_shared_core_wrapper: Any = None
_shared_core_wrapper_signature = None
_shared_core_workspace: Optional[torch.Tensor] = None


def _timing_start(timing_recorder: Any, device: torch.device) -> Any:
    return None if timing_recorder is None else timing_recorder.start(device)


def _timing_stop(
    timing_recorder: Any,
    start: Any,
    phase: str,
    step_idx: int,
    layer_idx: int,
    device: torch.device,
) -> None:
    if timing_recorder is not None:
        timing_recorder.stop(
            start,
            phase=phase,
            step_idx=step_idx,
            layer_idx=layer_idx,
            device=device,
        )


def _ensure_flashinfer_vector_workspace(wrapper: Any) -> None:
    """Size FA3's vector-sparse scratch buffers for the current plan.

    FlashInfer 0.2.10 hard-codes space for 128M token indices and 32768 row
    pointers. A per-head 64x64 route for a video sequence can legitimately
    exceed either limit. ``plan`` has already materialized the exact CSR
    tensors, so use their sizes instead of guessing from a fixed MB value.

    The implementation checks ``capacity <= required`` (rather than ``<``),
    hence the deliberate one-element guard in both allocations.
    """
    if getattr(wrapper, "_backend", None) != "fa3":
        return

    indices = getattr(wrapper, "_paged_kv_indices_buf", None)
    indptr = getattr(wrapper, "_paged_kv_indptr_buf", None)
    vector_indices = getattr(wrapper, "_vector_sparse_indices_buffer", None)
    vector_indptr = getattr(wrapper, "_vector_sparse_indptr_buffer", None)
    if any(x is None for x in (indices, indptr, vector_indices, vector_indptr)):
        raise RuntimeError(
            "The installed FlashInfer FA3 variable-block wrapper does not expose "
            "the vector-sparse workspaces required by flashinfer64"
        )

    required_indices = indices.numel() + 1
    required_indptr = indptr.numel() + 1
    if vector_indices.numel() < required_indices:
        vector_indices = torch.empty(
            required_indices, dtype=torch.int32, device=indices.device
        )
    if vector_indptr.numel() < required_indptr:
        vector_indptr = torch.empty(
            required_indptr, dtype=torch.int32, device=indptr.device
        )
    wrapper.reset_workspace_buffer(
        float_workspace_buffer=wrapper._float_workspace_buffer,
        int_workspace_buffer=wrapper._int_workspace_buffer,
        vector_sparse_indices_buffer=vector_indices,
        vector_sparse_indptr_buffer=vector_indptr,
    )


if triton is not None:

    @triton.jit
    def _residual_token_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        index_ptr,
        out_ptr,
        lse_ptr,
        query_sequence,
        key_sequence,
        stride_qh,
        stride_qs,
        stride_kh,
        stride_ks,
        stride_vh,
        stride_vs,
        stride_ih,
        stride_is,
        stride_ik,
        stride_oh,
        stride_os,
        head_dim: tl.constexpr,
        block_d: tl.constexpr,
        top_k: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Attention over a fixed per-row list of token indices.

        One program owns one ``(head, query_token)`` row.  The index list is
        produced by the selector and contains only tokens from unselected
        64x64 tiles.  The loop is deliberately online-softmax so no residual
        score matrix is materialized by this execution path.
        """

        pid = tl.program_id(0)
        head = pid // query_sequence
        row = pid - head * query_sequence
        rows = row
        dims = tl.arange(0, block_d)
        q_offsets = head * stride_qh + rows * stride_qs + dims
        q_vec = tl.load(
            q_ptr + q_offsets,
            mask=dims < head_dim,
            other=0.0,
        )

        m_i = tl.full((), float("-inf"), tl.float32)
        l_i = tl.zeros((), tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        for slot in tl.range(0, top_k):
            key_index = tl.load(
                index_ptr + head * stride_ih + row * stride_is + slot * stride_ik
            )
            valid = key_index < key_sequence
            k_offsets = head * stride_kh + key_index * stride_ks + dims
            k_vec = tl.load(k_ptr + k_offsets, mask=(dims < head_dim) & valid, other=0.0)
            score = tl.sum(q_vec.to(tl.float32) * k_vec.to(tl.float32), axis=0) * scale
            score = tl.where(valid, score, float("-inf"))

            m_new = tl.maximum(m_i, score)
            alpha = tl.where(m_i != float("-inf"), tl.exp(m_i - m_new), 0.0)
            weight = tl.where(valid, tl.exp(score - m_new), 0.0)
            l_i = l_i * alpha + weight
            acc = acc * alpha

            v_offsets = head * stride_vh + key_index * stride_vs + dims
            v_vec = tl.load(v_ptr + v_offsets, mask=(dims < head_dim) & valid, other=0.0)
            acc += weight * v_vec.to(tl.float32)
            m_i = m_new

        output = acc / tl.maximum(l_i, 1.0e-20)
        out_offsets = head * stride_oh + rows * stride_os + dims
        tl.store(out_ptr + out_offsets, output, mask=dims < head_dim)
        tl.store(lse_ptr + head * query_sequence + row, m_i + tl.log(l_i))


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def _make_permutation(
    seq_len: int,
    video_perm: Optional[torch.Tensor],
    video_len: Optional[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if video_perm is None:
        perm = torch.arange(seq_len, device=device, dtype=torch.long)
    else:
        vp = video_perm.to(device=device, dtype=torch.long)
        if video_len is None or video_len == seq_len:
            perm = vp
        else:
            if vp.numel() != video_len:
                raise ValueError(
                    f"video_perm has {vp.numel()} entries, expected video_len={video_len}"
                )
            tail = torch.arange(video_len, seq_len, device=device, dtype=torch.long)
            perm = torch.cat((vp, tail))
    if perm.numel() != seq_len:
        raise ValueError(f"permutation has {perm.numel()} entries, expected {seq_len}")
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(seq_len, device=device, dtype=torch.long)
    return perm, inverse


def _permute_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_perm: Optional[torch.Tensor],
    video_len: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError("q, k, v must have matching [1, heads, sequence, head_dim] shapes")
    perm, inverse = _make_permutation(q.shape[2], video_perm, video_len, q.device)
    return q[0, :, perm, :].contiguous(), k[0, :, perm, :].contiguous(), v[0, :, perm, :].contiguous(), inverse


def _masked_tile_means(x: torch.Tensor, padded_sequence: int) -> torch.Tensor:
    """Return valid-token means with shape ``[H, num_64_tiles, D]``."""
    heads, sequence, head_dim = x.shape
    if padded_sequence != sequence:
        x = F.pad(x, (0, 0, 0, padded_sequence - sequence))
    tiles = x.view(heads, padded_sequence // BLOCK_SIZE, BLOCK_SIZE, head_dim)
    valid = torch.arange(padded_sequence, device=x.device) < sequence
    valid = valid.view(1, padded_sequence // BLOCK_SIZE, BLOCK_SIZE, 1)
    counts = valid.sum(dim=2).clamp_min(1).to(x.dtype)
    return (tiles * valid).sum(dim=2) / counts


def compute_64_tile_scores(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Compute the DFS-style mean-pooled score for every 64x64 tile pair."""
    sequence = q.shape[1]
    padded_sequence = math.ceil(sequence / BLOCK_SIZE) * BLOCK_SIZE
    q_mean = _masked_tile_means(q, padded_sequence)
    k_mean = _masked_tile_means(k, padded_sequence)
    logits = torch.bmm(q_mean, k_mean.transpose(1, 2)).float() / math.sqrt(q.shape[-1])
    key_valid = torch.arange(padded_sequence, device=q.device) < sequence
    key_valid = key_valid.view(1, -1, BLOCK_SIZE).any(dim=-1)
    logits = logits.masked_fill(~key_valid[:, None, :], float("-inf"))
    return torch.softmax(logits, dim=-1)


def select_64_tiles_from_scores(
    tile_scores: torch.Tensor,
    *,
    top_p: float,
) -> torch.Tensor:
    """Select 64x64 KV tiles from already-computed DFS tile scores."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    probs = tile_scores
    # A probability can underflow to exactly zero after the softmax.  With
    # the usual cumulative-mass rule, those zero-probability tiles would be
    # dropped even when top_p=1.0.  At p=1 the contract is all tiles, so make
    # that identity explicit.  The final partial tile is handled by
    # row_sizes/col_sizes in the FlashInfer plan.
    if top_p >= 1.0:
        return torch.ones_like(probs, dtype=torch.bool)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    keep_sorted = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) < top_p
    # The first valid tile is always retained, including under numerical
    # underflow or a very small top_p.
    keep_sorted[..., 0] = True
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, keep_sorted)
    return mask


def select_64_tiles_topk_from_scores(
    tile_scores: torch.Tensor,
    *,
    top_ratio: float,
) -> torch.Tensor:
    """Select the top-ranked fraction of KV tiles for each head/query tile."""
    if not 0.0 < top_ratio <= 1.0:
        raise ValueError(f"tile top_ratio must be in (0, 1], got {top_ratio}")
    keep = max(1, math.ceil(tile_scores.shape[-1] * top_ratio))
    indices = torch.topk(tile_scores, k=keep, dim=-1, sorted=False).indices
    mask = torch.zeros_like(tile_scores, dtype=torch.bool)
    return mask.scatter_(-1, indices, True)


def _select_hyvideo_core_tiles(
    tile_scores: torch.Tensor,
    *,
    route_mode: str,
    tile_top_p: float,
    tile_top_ratio: float,
    sequence: int,
    video_len: Optional[int],
) -> torch.Tensor:
    """Select video tiles while preserving DFSAttn's dense text policy.

    HunyuanVideo stores video tokens first and text tokens in the tail.  The
    upstream DFSAttn selector never sparsifies text KV blocks and evaluates
    every text-query block densely.  A 64-token block straddling the boundary
    is conservatively treated as text/dense, matching the upstream convention.
    """
    if video_len is None or video_len >= sequence:
        if route_mode == "topp_topk":
            return select_64_tiles_from_scores(tile_scores, top_p=tile_top_p)
        return select_64_tiles_topk_from_scores(
            tile_scores, top_ratio=tile_top_ratio
        )
    if video_len < 0:
        raise ValueError(f"video_len must be non-negative, got {video_len}")

    blocks = tile_scores.shape[-1]
    dense_start_block = video_len // BLOCK_SIZE
    num_dense_text_blocks = blocks - dense_start_block
    core_mask = torch.zeros_like(tile_scores, dtype=torch.bool)

    # Only fully-video KV blocks participate in ranking.  Text blocks are
    # forced on below and, as in upstream DFSAttn, consume the total Top-k
    # budget rather than competing by their pooled proxy scores.
    if dense_start_block > 0:
        video_scores = tile_scores[..., :dense_start_block]
        if route_mode == "topp_topk":
            video_probs = video_scores / video_scores.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(video_scores.dtype).tiny)
            core_mask[..., :dense_start_block] = select_64_tiles_from_scores(
                video_probs, top_p=tile_top_p
            )
        else:
            total_keep = max(1, math.ceil(blocks * tile_top_ratio))
            video_keep = min(
                dense_start_block,
                max(0, total_keep - num_dense_text_blocks),
            )
            if video_keep > 0:
                indices = torch.topk(
                    video_scores, k=video_keep, dim=-1, sorted=False
                ).indices
                core_mask[..., :dense_start_block].scatter_(-1, indices, True)

    # All video queries attend every text key.  All query blocks containing
    # any text token attend every key.  The latter assignment also covers the
    # mixed video/text boundary block.
    core_mask[..., dense_start_block:] = True
    core_mask[:, dense_start_block:, :] = True
    return core_mask


def select_64_tiles(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    top_p: float,
) -> torch.Tensor:
    """Compute DFS tile scores and select 64x64 tiles by Top-p."""
    return select_64_tiles_from_scores(
        compute_64_tile_scores(q, k), top_p=top_p
    )


def _build_residual_topk(
    tile_scores: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    sequence: int,
    token_top_k: Optional[int],
    token_top_ratio: float,
    token_top_p: Optional[float] = None,
    q_block_start: int = 0,
    q_block_end: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Expand DFS tile scores and select residual tokens without token QK."""
    heads, q_score_blocks, k_blocks = tile_scores.shape
    q_blocks = core_mask.shape[1]
    if q_score_blocks != q_blocks or core_mask.shape[2] != k_blocks:
        raise ValueError("tile_scores and core_mask must have matching block dimensions")
    if q_block_end is None:
        q_block_end = q_blocks
    if not 0 <= q_block_start < q_block_end <= q_blocks:
        raise ValueError(
            f"invalid q block range [{q_block_start}, {q_block_end}) for {q_blocks} blocks"
        )
    if token_top_k is not None and token_top_k < 0:
        raise ValueError("token_top_k must be non-negative")
    if not 0.0 < token_top_ratio <= 1.0:
        raise ValueError("token_top_ratio must be in (0, 1]")
    if token_top_p is not None and not 0.0 <= token_top_p <= 1.0:
        raise ValueError("token_top_p must be in [0, 1]")

    selected_blocks_all = ~core_mask[:, q_block_start:q_block_end, :]
    candidate_counts = selected_blocks_all.sum(dim=-1) * BLOCK_SIZE
    if sequence % BLOCK_SIZE:
        candidate_counts = candidate_counts - selected_blocks_all[..., -1].to(candidate_counts.dtype) * (
            BLOCK_SIZE - sequence % BLOCK_SIZE
        )
    candidate_counts = candidate_counts.clamp_min(0)
    if token_top_p == 0.0:
        desired_counts = torch.zeros_like(candidate_counts)
    elif token_top_p is not None:
        # Convert each tile's proxy probability mass to equal per-token mass,
        # then apply Top-p only within tiles rejected by the core Top-k route.
        # This stays independent of the output QK calculation and preserves
        # the selector's bounded q64-chunk memory contract.
        valid_per_block = torch.full(
            (k_blocks,), BLOCK_SIZE, dtype=torch.long, device=tile_scores.device
        )
        if sequence % BLOCK_SIZE:
            valid_per_block[-1] = sequence % BLOCK_SIZE
        per_token_mass = (
            tile_scores[:, q_block_start:q_block_end, :, None]
            / valid_per_block[None, None, :, None]
        ).expand(-1, -1, -1, BLOCK_SIZE)
        token_valid = selected_blocks_all[:, :, :, None].expand_as(per_token_mass)
        offsets = torch.arange(BLOCK_SIZE, device=tile_scores.device)
        token_valid = token_valid & (
            offsets[None, None, None, :] < valid_per_block[None, None, :, None]
        )
        masses = per_token_mass.masked_fill(~token_valid, 0.0).flatten(2)
        masses = masses / masses.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(masses.dtype).tiny
        )
        sorted_masses = torch.sort(masses, dim=-1, descending=True, stable=True).values
        keep = (torch.cumsum(sorted_masses, dim=-1) - sorted_masses) < token_top_p
        desired_counts = (keep & (sorted_masses > 0)).sum(dim=-1)
    elif token_top_k is not None and token_top_k > 0:
        desired_counts = torch.full_like(candidate_counts, int(token_top_k))
    else:
        desired_counts = torch.ceil(candidate_counts.float() * token_top_ratio).to(torch.long)
        desired_counts = desired_counts.clamp_min(1)
    desired_counts = torch.minimum(desired_counts, candidate_counts)
    max_top_k = int(desired_counts.max().item())

    # Process one q64 block at a time, but batch all heads.  Tile scores are
    # repeated over the 64 tokens in each tile; no residual token QK is used
    # for selection.  The residual Triton kernel computes QK only once for the
    # selected tokens needed by the final attention output.
    candidate_lists = []
    token_offsets = torch.arange(BLOCK_SIZE, device=tile_scores.device, dtype=torch.long)
    for local_block, q_block in enumerate(range(q_block_start, q_block_end)):
        selected_blocks = selected_blocks_all[:, local_block, :]
        candidate_tokens = (
            torch.arange(k_blocks, device=tile_scores.device, dtype=torch.long)[None, :, None] * BLOCK_SIZE
            + token_offsets[None, None, :]
        ).expand(heads, -1, -1).reshape(heads, -1)
        candidate_valid = selected_blocks[:, :, None].expand(-1, -1, BLOCK_SIZE).reshape(heads, -1)
        candidate_valid &= candidate_tokens < sequence
        safe_candidates = torch.where(candidate_valid, candidate_tokens, torch.zeros_like(candidate_tokens))
        block_scores = tile_scores[:, q_block, :]
        if token_top_p is not None and token_top_p > 0.0:
            block_scores = block_scores / valid_per_block[None, :]
        token_scores = block_scores[:, :, None].expand(-1, -1, BLOCK_SIZE).reshape(heads, -1)
        token_scores = token_scores.masked_fill(~candidate_valid, float("-inf"))
        # Stable tie break for the 64 equal token scores within a tile.
        ranking_scores = token_scores - safe_candidates.to(token_scores.dtype) * 1.0e-8
        q_start = q_block * BLOCK_SIZE
        q_end = min(sequence, q_start + BLOCK_SIZE)
        _, local = torch.sort(ranking_scores, dim=-1, descending=True, stable=True)
        local = local[:, :max_top_k]
        selected = safe_candidates.gather(1, local)[:, None, :].expand(
            -1, q_end - q_start, -1
        )
        slot_valid = torch.arange(max_top_k, device=tile_scores.device)[None, :] < desired_counts[:, local_block, None]
        slot_valid &= candidate_counts[:, local_block, None] > 0
        valid = slot_valid[:, None, :].expand(-1, q_end - q_start, -1).contiguous()
        candidate_lists.append((selected, valid))

    row_start = q_block_start * BLOCK_SIZE
    row_end = min(sequence, q_block_end * BLOCK_SIZE)
    indices = torch.zeros(
        (heads, row_end - row_start, max_top_k), dtype=torch.long, device=tile_scores.device
    )
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for local_block, (selected, selected_valid) in enumerate(candidate_lists):
        start = (q_block_start + local_block) * BLOCK_SIZE
        end = min(sequence, start + BLOCK_SIZE)
        local_start = start - row_start
        local_end = end - row_start
        indices[:, local_start:local_end, :] = selected
        valid[:, local_start:local_end, :] = selected_valid
    return indices, valid, max_top_k


def _run_residual_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    top_k: int,
    key_sequence: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if key_sequence is None:
        key_sequence = k.shape[1]
    if top_k == 0:
        heads, query_sequence, head_dim = q.shape
        return (
            torch.zeros_like(q),
            torch.full(
                (heads, query_sequence),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            ),
        )
    safe_indices = torch.where(valid, indices, torch.full_like(indices, key_sequence))
    if triton is None or not q.is_cuda:
        safe_indices = safe_indices.clamp(max=max(0, key_sequence - 1))
        scores = torch.gather(
            torch.matmul(q, k.transpose(1, 2)),
            2,
            safe_indices,
        ).float() * (q.shape[-1] ** -0.5)
        scores = scores.masked_fill(~valid, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        has_valid = valid.any(dim=-1, keepdim=True)
        weights = torch.softmax(torch.where(has_valid, scores, torch.zeros_like(scores)), dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        selected_v = torch.gather(
            v[:, None, :, :].expand(-1, q.shape[1], -1, -1),
            2,
            safe_indices[..., None].expand(-1, -1, -1, q.shape[-1]),
        )
        output = (weights[..., None] * selected_v).sum(dim=2)
        return torch.where(has_valid, output, torch.zeros_like(output)), lse

    heads, query_sequence, head_dim = q.shape
    output = torch.empty_like(q)
    lse = torch.empty((heads, query_sequence), dtype=torch.float32, device=q.device)
    block_d = _next_power_of_two(head_dim)
    _residual_token_attention_kernel[(heads * query_sequence,)](
        q,
        k,
        v,
        safe_indices,
        output,
        lse,
        query_sequence,
        key_sequence,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        output.stride(0),
        output.stride(1),
        head_dim=head_dim,
        block_d=block_d,
        top_k=top_k,
        scale=head_dim ** -0.5,
    )
    return output, lse


def _plan_shared_flashinfer_core_wrapper(
    q: torch.Tensor,
    core_mask: torch.Tensor,
):
    """Plan the one shared official FlashInfer variable-block wrapper."""
    global _shared_core_wrapper, _shared_core_wrapper_signature, _shared_core_workspace
    if not q.is_cuda:
        return None
    if flashinfer is None:
        raise ImportError("flashinfer64 requires FlashInfer to be installed for CUDA execution")
    variable_wrapper = FlashInferVariableBlockSparseAttention
    if variable_wrapper is None:
        raise ImportError(
            "This backend requires FlashInfer.VariableBlockSparseAttentionWrapper "
            "for per-head 64x64 routes"
        )
    heads, sequence, head_dim = q.shape
    signature = (q.device.index, q.dtype, heads, sequence, head_dim)
    if _shared_core_wrapper is None or _shared_core_wrapper_signature != signature:
        # Let FlashInfer select the backend supported by the installed build.
        # The reference HunyuanVideo integration uses ``auto``; forcing FA2
        # here can select a different variable-block implementation from the
        # one that was validated with this FlashInfer checkout.
        workspace_mb = int(os.environ.get("FLASHINFER64_WORKSPACE_MB", "128"))
        if workspace_mb < 16:
            raise ValueError("FLASHINFER64_WORKSPACE_MB must be at least 16")
        _shared_core_workspace = torch.empty(
            workspace_mb * 1024 * 1024, dtype=torch.uint8, device=q.device
        )
        _shared_core_wrapper = variable_wrapper(_shared_core_workspace, backend="auto")
        _shared_core_wrapper_signature = signature
    wrapper = _shared_core_wrapper
    blocks = core_mask.shape[1]
    row_sizes = torch.full((heads, blocks), BLOCK_SIZE, dtype=torch.int32, device=q.device)
    col_sizes = torch.full((heads, blocks), BLOCK_SIZE, dtype=torch.int32, device=q.device)
    if sequence % BLOCK_SIZE:
        tail = sequence % BLOCK_SIZE
        row_sizes[:, -1] = tail
        col_sizes[:, -1] = tail
    wrapper.plan(
        core_mask,
        row_sizes,
        col_sizes,
        heads,
        heads,
        head_dim,
        causal=False,
        pos_encoding_mode="NONE",
        sm_scale=head_dim ** -0.5,
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
    )
    _ensure_flashinfer_vector_workspace(wrapper)
    return wrapper


def _run_flashinfer_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    wrapper: Any = None,
    timing_recorder: Any = None,
    step_idx: int = -1,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not q.is_cuda:
        # Reference-only path for CPU tests and environments without the
        # optional FlashInfer extension.  Production CUDA execution remains
        # fail-fast below rather than silently falling back to this O(N^2)
        # implementation.
        heads, sequence, head_dim = q.shape
        block_mask = core_mask.repeat_interleave(BLOCK_SIZE, dim=1).repeat_interleave(BLOCK_SIZE, dim=2)
        block_mask = block_mask[:, :sequence, :sequence]
        run_start = _timing_start(timing_recorder, q.device)
        scores = torch.bmm(q, k.transpose(1, 2)).float() * (head_dim ** -0.5)
        scores = scores.masked_fill(~block_mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        output = torch.bmm(torch.softmax(scores, dim=-1).to(v.dtype), v)
        _timing_stop(
            timing_recorder, run_start, "flashinfer_run", step_idx, layer_idx, q.device
        )
        return output, lse
    if flashinfer is None or FlashInferVariableBlockSparseAttention is None:
        raise ImportError("flashinfer64 requires FlashInfer to be installed for CUDA execution")

    if wrapper is None:
        plan_start = _timing_start(timing_recorder, q.device)
        wrapper = _plan_shared_flashinfer_core_wrapper(q, core_mask)
        _timing_stop(
            timing_recorder,
            plan_start,
            "flashinfer_plan",
            step_idx,
            layer_idx,
            q.device,
        )
    if wrapper is None:
        raise ImportError("flashinfer64 requires FlashInfer to be installed for CUDA execution")
    run_start = _timing_start(timing_recorder, q.device)
    output, lse = wrapper.run(q, k, v, return_lse=True)
    _timing_stop(
        timing_recorder, run_start, "flashinfer_run", step_idx, layer_idx, q.device
    )
    # FlashInfer's prefill sparse kernel returns the base-2 log normalizer
    # used by its exp2 softmax implementation.  The residual Triton path and
    # the merge below use natural-log LSE, so convert at this boundary.
    return output, lse * math.log(2.0)


def _merge_lse_states(
    core_output: torch.Tensor,
    core_lse: torch.Tensor,
    residual_output: torch.Tensor,
    residual_lse: torch.Tensor,
) -> torch.Tensor:
    merged_lse = torch.maximum(core_lse, residual_lse)
    core_weight = torch.where(torch.isfinite(core_lse), torch.exp(core_lse - merged_lse), torch.zeros_like(merged_lse))
    residual_weight = torch.where(torch.isfinite(residual_lse), torch.exp(residual_lse - merged_lse), torch.zeros_like(merged_lse))
    normalizer = (core_weight + residual_weight).clamp_min(torch.finfo(core_weight.dtype).tiny)
    return ((core_weight[..., None] * core_output.float() + residual_weight[..., None] * residual_output.float()) / normalizer[..., None])


class FlashInfer64Attention:
    """Stateful independent backend used by the DFSAttn wrapper."""

    def __init__(self):
        self.route: Optional[FlashInfer64Route] = None
        self.last_stats = None

    @torch.no_grad()
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        video_perm: Optional[torch.Tensor],
        video_len: Optional[int],
        route_mode: str = "topp_topk",
        tile_top_p: float,
        tile_top_ratio: float = 0.25,
        token_top_k: Optional[int],
        token_top_ratio: float,
        token_top_p: float = 0.9,
        valid_sequence: Optional[int] = None,
        refresh_route: bool,
        record_density: bool = False,
        timing_recorder: Any = None,
        step_idx: int = -1,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        permute_start = _timing_start(timing_recorder, q.device)
        qh, kh, vh, inverse = _permute_qkv(q, k, v, video_perm, video_len)
        _timing_stop(
            timing_recorder,
            permute_start,
            "flashinfer_qkv_permute",
            step_idx,
            layer_idx,
            q.device,
        )
        full_sequence = qh.shape[1]
        if valid_sequence is None:
            valid_sequence = full_sequence
        if not 0 < valid_sequence <= full_sequence:
            raise ValueError(
                f"valid_sequence must be in [1, {full_sequence}], got {valid_sequence}"
            )

        # Dense HunyuanVideo uses FlashAttention varlen with two independent
        # sequences: the valid video+text prefix and the padded-text suffix.
        # The old FlashInfer64 path incorrectly joined both segments, allowing
        # real tokens to attend padding and padding queries to attend video.
        padding_q = qh[:, valid_sequence:, :]
        padding_k = kh[:, valid_sequence:, :]
        padding_v = vh[:, valid_sequence:, :]
        qh = qh[:, :valid_sequence, :].contiguous()
        kh = kh[:, :valid_sequence, :].contiguous()
        vh = vh[:, :valid_sequence, :].contiguous()
        sequence = valid_sequence

        def restore_segments(main_output: torch.Tensor) -> torch.Tensor:
            if valid_sequence < full_sequence:
                # Padding is a separate, small dense sequence in the original
                # varlen call. It must never be merged into the sparse prefix.
                padding_output = F.scaled_dot_product_attention(
                    padding_q[None, ...],
                    padding_k[None, ...],
                    padding_v[None, ...],
                    dropout_p=0.0,
                    is_causal=False,
                )[0]
                main_output = torch.cat((main_output, padding_output), dim=1)
            return main_output.to(q.dtype)[None, :, inverse, :]

        padded_sequence = math.ceil(sequence / BLOCK_SIZE) * BLOCK_SIZE
        expected_k = token_top_k if token_top_k is not None and token_top_k > 0 else -1
        if route_mode not in ("topp_topk", "topk_topp"):
            raise ValueError(
                f"route_mode must be 'topp_topk' or 'topk_topp', got {route_mode!r}"
            )
        # Compute the DFS-style tile score once.  It is reused both for the
        # 64x64 Top-p core selector and for residual token selection.
        tile_score_start = _timing_start(timing_recorder, q.device)
        tile_scores = compute_64_tile_scores(qh, kh)
        _timing_stop(
            timing_recorder,
            tile_score_start,
            "flashinfer_tile_score",
            step_idx,
            layer_idx,
            q.device,
        )
        route_valid = (
            self.route is not None
            and self.route.sequence == sequence
            and self.route.video_len == (-1 if video_len is None else video_len)
            and self.route.route_mode == route_mode
            and self.route.top_p == tile_top_p
            and self.route.tile_top_ratio == tile_top_ratio
            and (token_top_k is None or self.route.token_top_k == expected_k)
            and self.route.token_top_ratio == token_top_ratio
            and self.route.token_top_p == token_top_p
            and self.route.core_mask.device == q.device
        )
        if refresh_route or not route_valid:
            route_start = _timing_start(timing_recorder, q.device)
            core_mask = _select_hyvideo_core_tiles(
                tile_scores,
                route_mode=route_mode,
                tile_top_p=tile_top_p,
                tile_top_ratio=tile_top_ratio,
                sequence=sequence,
                video_len=video_len,
            )
            _timing_stop(
                timing_recorder,
                route_start,
                "flashinfer_route_select",
                step_idx,
                layer_idx,
                q.device,
            )
            if route_mode == "topp_topk" and tile_top_p >= 1.0 and not bool(core_mask.all().item()):
                raise RuntimeError("top_p=1.0 must select every FlashInfer64 tile")
            self.route = FlashInfer64Route(
                core_mask=core_mask,
                sequence=sequence,
                padded_sequence=padded_sequence,
                video_len=-1 if video_len is None else video_len,
                route_mode=route_mode,
                top_p=tile_top_p,
                tile_top_ratio=tile_top_ratio,
                token_top_k=expected_k,
                token_top_ratio=token_top_ratio,
                token_top_p=token_top_p,
            )
        route = self.route
        if route is None:
            raise RuntimeError("failed to build FlashInfer64 route")

        # VariableBlockSparseAttentionWrapper handles the final partial 64
        # block through row_sizes/col_sizes.  Do not zero-pad Q/K/V here:
        # zero padding would create real, non-masked keys in the last tile.
        core_output, core_lse = _run_flashinfer_core(
            qh,
            kh,
            vh,
            route.core_mask,
            timing_recorder=timing_recorder,
            step_idx=step_idx,
            layer_idx=layer_idx,
        )
        density_start = _timing_start(timing_recorder, q.device)
        if record_density:
            num_blocks = route.core_mask.shape[1]
            row_sizes = torch.full(
                (num_blocks,), BLOCK_SIZE, dtype=torch.long, device=qh.device
            )
            col_sizes = row_sizes.clone()
            if sequence % BLOCK_SIZE:
                row_sizes[-1] = sequence % BLOCK_SIZE
                col_sizes[-1] = sequence % BLOCK_SIZE
            core_interactions = int(
                (
                    route.core_mask.to(torch.int64)
                    * row_sizes[None, :, None]
                    * col_sizes[None, None, :]
                )
                .sum()
                .item()
            )
            padding_sequence = full_sequence - valid_sequence
            padding_interactions = qh.shape[0] * padding_sequence * padding_sequence
            core_interactions += padding_interactions
            total_possible = qh.shape[0] * (
                sequence * sequence + padding_sequence * padding_sequence
            )
        else:
            core_interactions = 0
            total_possible = 0
        _timing_stop(
            timing_recorder,
            density_start,
            "flashinfer_density_accounting",
            step_idx,
            layer_idx,
            q.device,
        )
        if route_mode == "topp_topk" and token_top_k is not None and token_top_k == 0:
            # Explicit zero means core-only mode.  Do not let zero fall
            # through to the ratio selector below.
            residual_output = torch.zeros_like(qh)
            residual_lse = torch.full(
                (qh.shape[0], sequence),
                float("-inf"),
                dtype=torch.float32,
                device=qh.device,
            )
            merge_start = _timing_start(timing_recorder, q.device)
            merged = _merge_lse_states(
                core_output, core_lse, residual_output, residual_lse
            )
            _timing_stop(
                timing_recorder,
                merge_start,
                "flashinfer_lse_merge",
                step_idx,
                layer_idx,
                q.device,
            )
            if record_density:
                self.last_stats = {
                    "core_interactions": core_interactions,
                    "residual_token_interactions": 0,
                    "total_possible": total_possible,
                }
            unpermute_start = _timing_start(timing_recorder, q.device)
            output = restore_segments(merged)
            _timing_stop(
                timing_recorder,
                unpermute_start,
                "flashinfer_output_unpermute",
                step_idx,
                layer_idx,
                q.device,
            )
            return output
        # Do not cache [head, sequence, top-k] residual indices.  At long
        # video lengths that tensor is several GB per layer and was the main
        # source of the observed OOM.  Build and execute a bounded number of
        # q64 blocks at a time instead.
        residual_output = torch.zeros_like(qh)
        residual_lse = torch.full(
            (qh.shape[0], sequence),
            float("-inf"),
            dtype=torch.float32,
            device=qh.device,
        )
        q_blocks = route.core_mask.shape[1]
        q_chunk_blocks = 8
        profile_residual_chunks = (
            timing_recorder is not None
            and os.environ.get("FLASHINFER64_PROFILE_CHUNKS", "0") == "1"
        )
        residual_total_start = _timing_start(timing_recorder, q.device)
        residual_interactions_device = torch.zeros(
            (), dtype=torch.int64, device=qh.device
        ) if record_density else None
        for q_block_start in range(0, q_blocks, q_chunk_blocks):
            q_block_end = min(q_blocks, q_block_start + q_chunk_blocks)
            residual_select_start = (
                _timing_start(timing_recorder, q.device)
                if profile_residual_chunks
                else None
            )
            residual_indices, residual_valid, residual_top_k = _build_residual_topk(
                tile_scores,
                route.core_mask,
                sequence=sequence,
                token_top_k=token_top_k,
                token_top_ratio=token_top_ratio,
                token_top_p=(token_top_p if route_mode == "topk_topp" else None),
                q_block_start=q_block_start,
                q_block_end=q_block_end,
            )
            if profile_residual_chunks:
                _timing_stop(
                    timing_recorder,
                    residual_select_start,
                    "flashinfer_residual_select",
                    step_idx,
                    layer_idx,
                    q.device,
                )
            row_start = q_block_start * BLOCK_SIZE
            row_end = min(sequence, q_block_end * BLOCK_SIZE)
            residual_run_start = (
                _timing_start(timing_recorder, q.device)
                if profile_residual_chunks
                else None
            )
            chunk_output, chunk_lse = _run_residual_triton(
                qh[:, row_start:row_end, :],
                kh,
                vh,
                residual_indices,
                residual_valid,
                top_k=residual_top_k,
                key_sequence=sequence,
            )
            if profile_residual_chunks:
                _timing_stop(
                    timing_recorder,
                    residual_run_start,
                    "flashinfer_residual_run",
                    step_idx,
                    layer_idx,
                    q.device,
                )
            residual_output[:, row_start:row_end, :] = chunk_output
            residual_lse[:, row_start:row_end] = chunk_lse
            if residual_interactions_device is not None:
                residual_interactions_device += residual_valid.sum()
        _timing_stop(
            timing_recorder,
            residual_total_start,
            "flashinfer_residual_total",
            step_idx,
            layer_idx,
            q.device,
        )
        density_start = _timing_start(timing_recorder, q.device)
        residual_interactions = (
            int(residual_interactions_device.item())
            if residual_interactions_device is not None
            else 0
        )
        _timing_stop(
            timing_recorder,
            density_start,
            "flashinfer_density_accounting",
            step_idx,
            layer_idx,
            q.device,
        )
        if record_density:
            self.last_stats = {
                "core_interactions": core_interactions,
                "residual_token_interactions": residual_interactions,
                "total_possible": total_possible,
            }
        merge_start = _timing_start(timing_recorder, q.device)
        merged = _merge_lse_states(core_output, core_lse, residual_output, residual_lse)
        _timing_stop(
            timing_recorder,
            merge_start,
            "flashinfer_lse_merge",
            step_idx,
            layer_idx,
            q.device,
        )
        unpermute_start = _timing_start(timing_recorder, q.device)
        output = restore_segments(merged)
        _timing_stop(
            timing_recorder,
            unpermute_start,
            "flashinfer_output_unpermute",
            step_idx,
            layer_idx,
            q.device,
        )
        return output


__all__ = [
    "FlashInfer64Attention",
    "FlashInfer64Route",
    "compute_64_tile_scores",
    "select_64_tiles",
    "select_64_tiles_from_scores",
    "select_64_tiles_topk_from_scores",
]
