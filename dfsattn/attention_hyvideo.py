import torch
from typing import Optional

# Cache for DFS_Attention instances per layer
_dfs_attention_instances = {}


import math
import os
import csv
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from block_sparse_attn import block_sparse_attn_func
except ImportError:
    block_sparse_attn_func = None

from .hybrid_block_attention import HybridPlan, hybrid_sparse_attention_from_plan, partition_block_mask
from .utils.timing import AttentionTimingRecorder
from .utils.subblock_retention import SubblockRetentionProfiler
from .kernels.triton.kp_flex_attention import (
    compact_block_mask,
    kp_fine_sparse_attention,
)
from .flashinfer64_attention import FlashInfer64Attention

_flashinfer64_instances = {}


def _compute_cache_schedule(
    step_idx: int,
    skip_steps: int,
    cache_interval: int,
    sparsity: float,
    sparsity_dcrt: float,
) -> Tuple[bool, float]:
    if cache_interval <= 0:
        raise ValueError("cache_interval must be greater than 0.")

    if step_idx < skip_steps:
        return False, sparsity

    raw_interval_idx = (step_idx - skip_steps) // cache_interval
    if sparsity_dcrt > 0:
        max_valid_interval = max(0, math.ceil(sparsity / sparsity_dcrt) - 1)
    else:
        max_valid_interval = raw_interval_idx
    interval_idx = min(raw_interval_idx, max_valid_interval)
    is_cache_step = (
        raw_interval_idx <= max_valid_interval
        and (step_idx - skip_steps) % cache_interval == 0
    )
    current_sparsity = sparsity - interval_idx * sparsity_dcrt
    current_sparsity = min(1.0, max(0.0, current_sparsity))
    return is_cache_step, current_sparsity


class DFS_Attention(nn.Module):
    # Mask granularity is part of the cache identity. Native uses 128x128,
    # while Hybrid consumes the same selector at 16x16.
    _cached_metadata: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    mask_output_dir: Optional[str] = None
    mask_head_indices: Optional[Tuple[int, ...]] = (0,)
    mask_layer_interval: int = 15
    mask_save_bool: bool = False
    # Toggle for recording the actual density of sparse attention masks
    record_density: bool = False
    # Recorded actual densities keyed by step_idx -> {layer_idx: density}
    density_records: Dict[int, Dict[int, float]] = {}
    # Interaction statistics for block Top-p + residual token Top-k.
    # Values are keyed by step_idx -> {layer_idx: statistics}.
    sparsity_records: Dict[int, Dict[int, Dict[str, float]]] = {}
    timing_recorder = AttentionTimingRecorder()
    subblock_profiler = SubblockRetentionProfiler()

    def __init__(
        self, 
        step_idx: int,
        layer_idx: int,
        sparse_ratio: float = 0.5,
        cache_flag: bool = True,
        video_len: int = 0,
        video_perm: Optional[torch.Tensor] = None,
        block_size: int = 128,
        q_tile_size: int = 32,
        k_tile_size: int = 32,
        sparse_execution: str = "native",
        hybrid_threshold: int = 8,
        block_top_p: Optional[float] = None,
        token_top_k: int = 0,
        residual_candidate_blocks: int = 4,
        selector_mode: str = "topk",
        fine_top_p: float = 0.9,
    ):
        super(DFS_Attention, self).__init__()

        self.step_idx = step_idx
        self.layer_idx = layer_idx
        self.sparse_ratio = sparse_ratio
        self.cache_flag = cache_flag
        self.video_len = video_len
        self.video_perm = video_perm
        self.block_size = block_size
        self.q_tile_size = q_tile_size
        self.k_tile_size = k_tile_size
        self.sparse_execution = sparse_execution
        self.hybrid_threshold = hybrid_threshold
        self.block_top_p = block_top_p
        self.token_top_k = token_top_k
        self.residual_candidate_blocks = residual_candidate_blocks
        self.selector_mode = selector_mode
        self.fine_top_p = fine_top_p

    def _compute_permutation(self, video_perm: Optional[torch.Tensor], seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # If permutation is disabled, return identity permutation
        if video_perm is None:
            identity_perm = torch.arange(seq_len, device=device, dtype=torch.long)
            return identity_perm, identity_perm
        

        text_len = seq_len - self.video_len
        
        # Build full permutation (video perm + identity for text tokens)
        if text_len > 0:
            tail_indices = torch.arange(self.video_len, seq_len, device=device)
            perm = torch.cat([video_perm, tail_indices])
        else:
            perm = video_perm
        
        # Create inverse permutation efficiently
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=inv_perm.dtype)
        
        return perm.to(device), inv_perm.to(device)
    

    def _apply_permutation_in_place(
        self, tensor: torch.Tensor, permutation: torch.Tensor
    ) -> torch.Tensor:

        if permutation.device != tensor.device:
            permutation = permutation.to(tensor.device, non_blocking=True)
        result = tensor[:, :, permutation, :]
        return result if result.is_contiguous() else result.contiguous()


    def _apply_inverse_permutation(
        self, tensor: torch.Tensor, inverse_permutation: torch.Tensor
    ) -> torch.Tensor:

        if inverse_permutation.device != tensor.device:
            inverse_permutation = inverse_permutation.to(tensor.device, non_blocking=True)
        result = tensor[:, :, inverse_permutation, :]
        return result if result.is_contiguous() else result.contiguous()

    def _mask_cache_key(self) -> Tuple[Any, ...]:
        return (
            self.layer_idx,
            self.block_size,
            self.q_tile_size,
            self.k_tile_size,
            self.sparse_ratio,
            self.block_top_p,
            self.token_top_k,
            self.residual_candidate_blocks,
            self.selector_mode,
            self.fine_top_p,
        )

    @torch.no_grad()
    def _refine_topk_mask_with_subblock_top_p(
        self,
        tile_score: torch.Tensor,
        coarse_video_mask: torch.Tensor,
        dense_start_block: int,
        seq_len: int,
        tiles_per_q_block: int,
        tiles_per_k_block: int,
        pair_chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Refine each selected 128x128 coarse block with local fine Top-p.

        The 8x8 scores inside one selected coarse pair are jointly ranked.
        The first score crossing ``fine_top_p`` is retained.  The boundary
        block containing the video tail/text and all text-query blocks retain
        DFSAttn's existing dense policy.
        """
        if tiles_per_q_block != 8 or tiles_per_k_block != 8:
            raise ValueError("kp mode requires 128x128 coarse blocks and 16x16 sub-blocks")

        _, heads, _, _ = coarse_video_mask.shape
        fine_block_size = self.q_tile_size
        fine_block_num = math.ceil(seq_len / fine_block_size)
        dense_fine_start = min(
            dense_start_block * tiles_per_k_block, fine_block_num
        )
        fine_mask = torch.zeros(
            (heads, fine_block_num, fine_block_num),
            dtype=torch.bool,
            device=tile_score.device,
        )

        if dense_start_block > 0:
            fine_scores = tile_score[
                0, :, :dense_fine_start, :dense_fine_start
            ].view(
                heads,
                dense_start_block,
                tiles_per_q_block,
                dense_start_block,
                tiles_per_k_block,
            )
            fine_video_mask = fine_mask[
                :, :dense_fine_start, :dense_fine_start
            ].view(
                heads,
                dense_start_block,
                tiles_per_q_block,
                dense_start_block,
                tiles_per_k_block,
            )
            selected_coarse = coarse_video_mask[
                0, :, :dense_start_block, :dense_start_block
            ]
            fine_count = tiles_per_q_block * tiles_per_k_block

            if self.fine_top_p >= 1.0:
                # This is an exact identity path, rather than relying on a
                # floating-point cumulative sum to retain the final entries.
                # Every selected Q128 x K128 pair expands to all 8 x 8 fine
                # blocks and every unselected pair remains empty.
                fine_video_mask.copy_(
                    selected_coarse[:, :, None, :, None].expand(
                        -1,
                        -1,
                        tiles_per_q_block,
                        -1,
                        tiles_per_k_block,
                    )
                )
            else:
                for head_idx in range(heads):
                    q_idx, k_idx = selected_coarse[head_idx].nonzero(as_tuple=True)
                    for start in range(0, q_idx.numel(), pair_chunk_size):
                        q_chunk = q_idx[start : start + pair_chunk_size]
                        k_chunk = k_idx[start : start + pair_chunk_size]
                        values = fine_scores[
                            head_idx, q_chunk, :, k_chunk, :
                        ].reshape(-1, fine_count).float()
                        sorted_values, sorted_indices = torch.sort(
                            values, dim=-1, descending=True
                        )
                        totals = sorted_values.sum(dim=-1, keepdim=True)
                        keep_sorted = (
                            sorted_values.cumsum(dim=-1) - sorted_values
                        ) < totals * self.fine_top_p
                        keep = torch.zeros_like(keep_sorted)
                        keep.scatter_(1, sorted_indices, keep_sorted)
                        fine_video_mask[
                            head_idx, q_chunk, :, k_chunk, :
                        ] = keep.view(-1, tiles_per_q_block, tiles_per_k_block)

        # Preserve upstream DFSAttn's dense tail/text convention.  It also
        # guarantees every query row has at least one active key block.
        fine_mask[:, :dense_fine_start, dense_fine_start:] = True
        fine_mask[:, dense_fine_start:, :] = True
        return fine_mask


    def _compute_tile_score(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_tile_size: int,
        k_tile_size: int,
        block_dim: int,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_heads, seq_len, head_dim = q.shape
        scale_factor = 1.0 / math.sqrt(head_dim)

        text_len = seq_len - self.video_len
        video_blocks_num = math.ceil(self.video_len / block_dim)
        q_block_num = math.ceil(seq_len / block_dim)
        k_block_num = math.ceil(seq_len / block_dim)
        
        # Apply permutation
        q_perm = self._apply_permutation_in_place(q, permutation)
        k_perm = self._apply_permutation_in_place(k, permutation)
        
        # Pad video tokens to align to block boundaries
        video_padded_len = video_blocks_num * block_dim
        video_pad_size = video_padded_len - self.video_len
        
        if video_pad_size > 0:
            q_video = q_perm[:, :, :self.video_len, :]
            k_video = k_perm[:, :, :self.video_len, :]
            q_text = q_perm[:, :, self.video_len:, :] if text_len > 0 else None
            k_text = k_perm[:, :, self.video_len:, :] if text_len > 0 else None
            
            q_pad = torch.zeros((bsz, num_heads, video_pad_size, head_dim), 
                                device=q.device, dtype=q.dtype)
            k_pad = torch.zeros((bsz, num_heads, video_pad_size, head_dim), 
                                device=k.device, dtype=k.dtype)
            
            if text_len > 0:
                q_padded = torch.cat([q_video, q_pad, q_text], dim=2)
                k_padded = torch.cat([k_video, k_pad, k_text], dim=2)
            else:
                q_padded = torch.cat([q_video, q_pad], dim=2)
                k_padded = torch.cat([k_video, k_pad], dim=2)
        else:
            q_padded = q_perm
            k_padded = k_perm
        
        # Pad at the end if needed, no need when text_len % block_dim == 0
        q_padded_len = q_block_num * block_dim
        k_padded_len = k_block_num * block_dim
        
        total_q_padded_len = q_padded.shape[2]
        if total_q_padded_len < q_padded_len:
            q_padded = F.pad(q_padded, (0, 0, 0, q_padded_len - total_q_padded_len, 0, 0, 0, 0), value=0)
        total_k_padded_len = k_padded.shape[2]
        if total_k_padded_len < k_padded_len:
            k_padded = F.pad(k_padded, (0, 0, 0, k_padded_len - total_k_padded_len, 0, 0, 0, 0), value=0)
        
        # Compute the tile score of the video query and entire key
        q_tiles_num = video_padded_len // q_tile_size
        k_tiles_num = k_padded_len // k_tile_size
        
        # Mean pooling
        q_video_tiles = (
            q_padded[:, :, :video_padded_len, :]
            .view(bsz, num_heads, q_tiles_num, self.q_tile_size, head_dim)
            .mean(dim=3)
        )
        k_tiles = (
            k_padded[:, :, :, :]
            .view(bsz, num_heads, k_tiles_num, self.k_tile_size, head_dim)
            .mean(dim=3)
        )
        
        tile_score = (q_video_tiles @ k_tiles.transpose(-2, -1) * scale_factor).softmax(dim=-1) # (1, num_heads, video_padded_len // tile_size, k_padded_len // tile_size)
        
        return tile_score
    
    
    def _compute_block_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        permutations: torch.Tensor,
        sparse_ratio: Optional[float] = None,
        block_top_p: Optional[float] = None,
        return_frontier: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        
        bsz, num_heads, seq_len, _ = q.shape
        assert bsz == 1, "DFS Attention with block sparse currently supports batch size 1."
        text_len = seq_len - self.video_len

        assert self.block_size % self.q_tile_size == 0, f"block_dim ({self.block_size}) must be divisible by q_tile_size ({self.q_tile_size})"
        assert self.block_size % self.k_tile_size == 0, f"block_dim ({self.block_size}) must be divisible by k_tile_size ({self.k_tile_size})"

        block_dim = self.block_size
        q_block_num = k_block_num = math.ceil(seq_len / block_dim)
        tiles_per_q_block = block_dim // self.q_tile_size
        tiles_per_k_block = block_dim // self.k_tile_size
        video_blocks_num = math.ceil(self.video_len / block_dim)

        # Compute tile scores
        tile_score = self._compute_tile_score(
                q, k, self.q_tile_size, self.k_tile_size, block_dim, permutations, 
        )

        q_tiles_num, k_tiles_num = tile_score.shape[2], tile_score.shape[3]
        assert q_tiles_num % tiles_per_q_block == 0 and k_tiles_num % tiles_per_k_block == 0, "q_tiles_num and k_tiles_num must be divisible by tiles_per_q_block and tiles_per_k_block"

        block_value_sums = (
            tile_score.view(1, num_heads, video_blocks_num, tiles_per_q_block, k_block_num, tiles_per_k_block)
            .sum(dim=(3, 5))  # Sum over tiles in each block
        )
        
        dense_start_block = self.video_len // block_dim
        num_text_blocks = max(0, k_block_num - dense_start_block)
            
        # Create final mask with in-place operations
        selected_mask = torch.zeros((bsz, num_heads, q_block_num, k_block_num), dtype=torch.bool, device=q.device)
        video_mask = torch.zeros((1, num_heads, video_blocks_num, k_block_num), dtype=torch.bool, device=q.device)

        if block_top_p is None:
            topk_block_num = max(1, int(sparse_ratio * k_block_num))
            if num_text_blocks > 0:
                video_block_sums = block_value_sums[:, :, :, :dense_start_block]
                num_video_blocks_to_select = topk_block_num - num_text_blocks
                assert num_video_blocks_to_select > 0, "num_video_blocks_to_select must be non-negative"
                _, video_top_indices = torch.topk(
                    video_block_sums, k=num_video_blocks_to_select, dim=-1
                )
                video_mask.scatter_(-1, video_top_indices, True)
            else:
                _, topk_indices = torch.topk(block_value_sums, k=topk_block_num, dim=-1)
                video_mask.scatter_(-1, topk_indices, True)
        else:
            # Replace only DFSAttn's original *video-block* Top-k selector.
            # Text blocks are made dense below and must not consume Top-p
            # probability mass; otherwise changing Top-k -> Top-p would also
            # change the text/video routing policy.
            if num_text_blocks > 0:
                video_scores = block_value_sums[..., :dense_start_block]
            else:
                video_scores = block_value_sums
            video_prob = video_scores / (
                video_scores.sum(dim=-1, keepdim=True) + 1e-8
            )
            sorted_prob, sorted_idx = torch.sort(
                video_prob, dim=-1, descending=True
            )
            keep_sorted = (
                torch.cumsum(sorted_prob, dim=-1) - sorted_prob
            ) < block_top_p
            video_mask.scatter_(-1, sorted_idx, keep_sorted)

        if DFS_Attention.subblock_profiler.enabled:
            if block_top_p is not None:
                raise ValueError("sub-block retention profiling requires DFSAttn coarse Top-k")
            # Exclude the forced-dense text region.  Only coarse blocks that
            # were genuinely selected by the video-block Top-k are profiled.
            selected_video_topk = video_mask[
                :, :, :video_blocks_num, :dense_start_block
            ]
            DFS_Attention.subblock_profiler.record(
                tile_score[:, :, :, : dense_start_block * tiles_per_k_block],
                selected_video_topk,
                tiles_per_q_block,
                tiles_per_k_block,
                self.step_idx,
                self.layer_idx,
            )

        if self.selector_mode == "kp":
            return self._refine_topk_mask_with_subblock_top_p(
                tile_score,
                video_mask,
                dense_start_block,
                seq_len,
                tiles_per_q_block,
                tiles_per_k_block,
            )

        # Preserve the current DFSAttn convention: text keys are dense for
        # every video query block, irrespective of the video-block selector.
        if num_text_blocks > 0:
            video_mask[..., dense_start_block:] = True
        selected_mask[:, :, :video_blocks_num, :] = video_mask

        assert torch.all(selected_mask[0, 0, :dense_start_block, dense_start_block:] == True), "selected_mask should be True for text tokens"

        # Handle text query blocks
        if text_len > 0:
            selected_mask[:, :, dense_start_block:, :] = True

        frontier_mask = torch.zeros_like(selected_mask)
        if return_frontier and self.residual_candidate_blocks > 0:
            candidate_scores = block_value_sums.masked_fill(
                video_mask, float("-inf")
            )
            frontier_k = min(self.residual_candidate_blocks, k_block_num)
            frontier_values, frontier_indices = torch.topk(
                candidate_scores, k=frontier_k, dim=-1
            )
            frontier_valid = torch.isfinite(frontier_values)
            frontier_video_mask = torch.zeros_like(video_mask)
            frontier_video_mask.scatter_(-1, frontier_indices, frontier_valid)
            frontier_mask[:, :, :video_blocks_num, :] = frontier_video_mask

        selected_mask = selected_mask.squeeze(0)
        frontier_mask = frontier_mask.squeeze(0)
        return (selected_mask, frontier_mask) if return_frontier else selected_mask

    def _block_interaction_count(
        self, block_mask: torch.Tensor, seq_len: int, block_size: Optional[int] = None
    ) -> int:
        """Count selected QK pairs exactly, excluding block-padding tokens."""
        block_size = self.block_size if block_size is None else block_size
        q_blocks, k_blocks = block_mask.shape[-2:]
        q_last = seq_len - block_size * (q_blocks - 1)
        k_last = seq_len - block_size * (k_blocks - 1)

        # Do not cast the full fine mask to float64.  At 720p a KP bool mask is
        # about 188 MB and the old weighted expression allocated another
        # ~1.4 GiB per layer merely to record density.  Only the final Q/K
        # blocks can be partial, so four integer reductions are sufficient.
        # count_nonzero reduces bool input directly; bool.sum(dtype=int64)
        # materializes an int64 conversion buffer of the input's full size.
        interior = torch.count_nonzero(block_mask[:, :-1, :-1])
        last_q = torch.count_nonzero(block_mask[:, -1:, :-1])
        last_k = torch.count_nonzero(block_mask[:, :-1, -1:])
        corner = torch.count_nonzero(block_mask[:, -1:, -1:])
        interactions = (
            interior * (block_size * block_size)
            + last_q * (q_last * block_size)
            + last_k * (block_size * k_last)
            + corner * (q_last * k_last)
        )
        return int(interactions.item())

    def _native_block_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run selected blocks through block_sparse_attn.

        ``softmax_lse`` returned by the extension is the *log-normalizer*
        (``logsumexp``), rather than the row maximum.  Keeping the native
        output normalized and carrying this log-normalizer makes it possible
        to merge it with the separately evaluated frontier without materializing
        a dense score matrix.
        """
        if block_sparse_attn_func is None:
            raise ImportError("block_sparse_attn is unavailable for the native frontier path.")
        from block_sparse_attn.block_sparse_attn_interface import BlockSparseAttnFunc

        _, num_heads, sequence, head_dim = q.shape
        q_bs = q[0].permute(1, 0, 2).contiguous()
        k_bs = k[0].permute(1, 0, 2).contiguous()
        v_bs = v[0].permute(1, 0, 2).contiguous()
        cu = torch.tensor([0, sequence], dtype=torch.int32, device=q.device)
        # BlockSparseAttnFunc expects the cumulative head-type encoding used
        # by block_sparse_attn_func (1, 2, ..., num_heads), not all ones.
        head_mask_type = torch.arange(
            1, num_heads + 1, dtype=torch.int32, device=q.device
        )
        base_mask = block_mask.unsqueeze(0).contiguous()
        out, softmax_lse, _ = BlockSparseAttnFunc.apply(
            q_bs,
            k_bs,
            v_bs,
            cu,
            cu,
            self.block_size,
            self.block_size,
            head_mask_type,
            None,
            base_mask,
            sequence,
            sequence,
            0.0,
            head_dim ** -0.5,
            False,
            False,
            True,
            -1,
            -1,
            False,
            False,
        )
        block_logz = softmax_lse[:, :, :sequence].squeeze(0)
        block_output = out[:sequence].permute(1, 0, 2).float()
        return block_logz, block_output

    def _frontier_residual_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frontier_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute exact residual state inside frontier blocks in chunks.

        The earlier implementation packed and scored one query block at a
        time.  That is mathematically fine but launches hundreds of tiny
        kernels at 720p.  Here block ids and token ids are formed in one
        vectorized operation and a small fixed number of query blocks are
        processed together, keeping peak memory bounded while substantially
        reducing Python/CUDA synchronization overhead.
        """
        _, num_heads, sequence, head_dim = q.shape
        q_blocks, k_blocks = frontier_mask.shape[1:]
        residual_m = torch.full(
            (num_heads, sequence), float("-inf"), device=q.device, dtype=torch.float32
        )
        residual_l = torch.zeros((num_heads, sequence), device=q.device, dtype=torch.float32)
        residual_o = torch.zeros(
            (num_heads, sequence, head_dim), device=q.device, dtype=torch.float32
        )
        residual_interactions = 0
        if self.token_top_k <= 0 or q_blocks == 0 or k_blocks == 0:
            return (
                residual_m,
                torch.zeros_like(residual_o),
                residual_interactions,
            )

        # Frontier construction normally contains exactly R blocks for video
        # query blocks and no blocks for text query blocks.  ``topk`` gives a
        # rectangular representation without a per-head ``nonzero``/packing
        # synchronization.  Invalid slots are masked below.
        frontier_block_k = min(max(1, self.residual_candidate_blocks), k_blocks)
        frontier_block_values, frontier_block_idx = torch.topk(
            frontier_mask.to(torch.int8), k=frontier_block_k, dim=-1
        )
        frontier_block_valid = frontier_block_values.to(torch.bool)
        token_offsets = torch.arange(self.block_size, device=q.device, dtype=torch.long)
        frontier_idx = (
            frontier_block_idx[..., None] * self.block_size + token_offsets
        ).flatten(-2)
        frontier_valid = frontier_block_valid[..., None].expand(
            -1, -1, -1, self.block_size
        ).flatten(-2)
        frontier_valid = frontier_valid & (frontier_idx < sequence)
        safe_frontier_idx = frontier_idx.clamp(max=max(0, sequence - 1))
        frontier_count = frontier_idx.shape[-1]
        residual_k = min(self.token_top_k, frontier_count)
        if residual_k == 0:
            return residual_m, torch.zeros_like(residual_o), residual_interactions

        # Video query blocks precede text query blocks in the DFS permutation;
        # skip the latter entirely, where the frontier is empty.
        active_q = frontier_mask.any(dim=(0, 2))
        if not bool(active_q.any().item()):
            return residual_m, torch.zeros_like(residual_o), residual_interactions
        last_active_block = int(torch.nonzero(active_q, as_tuple=False)[-1].item()) + 1
        q_chunk_blocks = 8
        q0, k0, v0 = q[0], k[0], v[0]
        scale = head_dim ** -0.5

        for block_start in range(0, last_active_block, q_chunk_blocks):
            block_end = min(block_start + q_chunk_blocks, last_active_block)
            token_start = block_start * self.block_size
            token_end = min(block_end * self.block_size, sequence)
            actual_tokens = token_end - token_start
            padded_tokens = (block_end - block_start) * self.block_size

            q_chunk = q0[:, token_start:token_end, :]
            if actual_tokens < padded_tokens:
                q_chunk = F.pad(q_chunk, (0, 0, 0, padded_tokens - actual_tokens))
            q_chunk = q_chunk.view(num_heads, block_end - block_start, self.block_size, head_dim)
            chunk_idx = safe_frontier_idx[:, block_start:block_end]
            chunk_valid = frontier_valid[:, block_start:block_end]
            frontier_k = torch.gather(
                k0[:, None, :, :].expand(-1, block_end - block_start, -1, -1),
                2,
                chunk_idx[..., None].expand(-1, -1, -1, head_dim),
            )
            frontier_score = torch.matmul(
                q_chunk, frontier_k.transpose(-2, -1)
            ).float() * scale
            frontier_score = frontier_score.masked_fill(
                ~chunk_valid[:, :, None, :], float("-inf")
            )
            # Use a stable, index-based tie break for quantized QK scores.
            # Batched GEMM and one-block GEMM can otherwise choose different
            # tokens when fp16/bf16 scores are exactly tied, causing
            # unnecessary diffusion drift even though the attention scores
            # are mathematically identical.
            ranking_score = frontier_score - chunk_idx[:, :, None, :].to(
                frontier_score.dtype
            ) * 1.0e-8
            _, residual_local_idx = torch.topk(
                ranking_score, k=residual_k, dim=-1
            )
            residual_score = torch.gather(frontier_score, -1, residual_local_idx)
            selected_valid = torch.gather(
                chunk_valid[:, :, None, :].expand(
                    -1, -1, self.block_size, -1
                ),
                -1,
                residual_local_idx,
            )
            selected_idx = torch.gather(
                chunk_idx[:, :, None, :].expand(
                    -1, -1, self.block_size, -1
                ),
                -1,
                residual_local_idx,
            )
            residual_v = torch.gather(
                v0[:, None, None, :, :].expand(
                    -1, block_end - block_start, self.block_size, -1, -1
                ),
                3,
                selected_idx[..., None].expand(-1, -1, -1, -1, head_dim),
            )
            chunk_m = residual_score.masked_fill(
                ~selected_valid, float("-inf")
            ).amax(dim=-1)
            weights = torch.where(
                selected_valid,
                torch.exp(residual_score - chunk_m[..., None]),
                torch.zeros_like(residual_score),
            )
            chunk_l = weights.sum(dim=-1)
            chunk_o = (
                weights.to(residual_v.dtype).unsqueeze(-1) * residual_v
            ).sum(dim=3).float()
            residual_m[:, token_start:token_end] = chunk_m.reshape(
                num_heads, -1
            )[:, :actual_tokens]
            residual_l[:, token_start:token_end] = chunk_l.reshape(
                num_heads, -1
            )[:, :actual_tokens]
            residual_o[:, token_start:token_end] = chunk_o.reshape(
                num_heads, -1, head_dim
            )[:, :actual_tokens]
            residual_interactions += int(
                selected_valid.reshape(num_heads, -1, residual_k)[:, :actual_tokens]
                .sum()
                .item()
            )

        residual_logz = residual_m + torch.log(residual_l.clamp_min(torch.finfo(residual_l.dtype).tiny))
        residual_output = residual_o / residual_l.clamp_min(torch.finfo(residual_l.dtype).tiny)[..., None]
        # Rows with no frontier tokens must remain neutral in the merge.  The
        # explicit zero avoids propagating the arbitrary 0/0 value above.
        residual_output = torch.where(
            residual_l[..., None] > 0,
            residual_output,
            torch.zeros_like(residual_output),
        )
        return residual_logz, residual_output, residual_interactions

    @staticmethod
    def _merge_attention_states(
        first: Tuple[torch.Tensor, torch.Tensor],
        second: Tuple[torch.Tensor, torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        first_logz, first_output = first
        second_logz, second_output = second
        merged_logz = torch.maximum(first_logz, second_logz)
        first_weight = torch.where(
            torch.isfinite(first_logz),
            torch.exp(first_logz - merged_logz),
            torch.zeros_like(merged_logz),
        )
        second_weight = torch.where(
            torch.isfinite(second_logz),
            torch.exp(second_logz - merged_logz),
            torch.zeros_like(merged_logz),
        )
        normalizer = (first_weight + second_weight).clamp_min(torch.finfo(first_weight.dtype).tiny)
        merged_output = (
            first_weight[..., None] * first_output
            + second_weight[..., None] * second_output
        ) / normalizer[..., None]
        return merged_output.to(dtype).unsqueeze(0)

    def _run_oracle_hybrid_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        permutation: torch.Tensor,
        inverse_permutation: torch.Tensor,
        block_mask: torch.Tensor,
        frontier_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Run block attention plus Top-k inside a bounded frontier.

        The residual selector only scores the unselected frontier blocks.  The
        final softmax is packed over selected block tokens and residual tokens,
        so rejected keys never enter a dense ``[heads, query, key]`` tensor.
        """
        bsz, num_heads, seq_len, head_dim = q.shape
        if bsz != 1:
            raise ValueError("The phase-A oracle currently supports batch_size=1.")
        q_perm = self._apply_permutation_in_place(q, permutation)
        k_perm = self._apply_permutation_in_place(k, permutation)
        v_perm = self._apply_permutation_in_place(v, permutation)

        # The native 128x128 extension supplies the regular block path and its
        # log-sum-exp state.  Only the small frontier is evaluated by PyTorch,
        # then both states are merged with one softmax normalization.
        if q.is_cuda and self.block_size == 128 and block_sparse_attn_func is not None:
            block_state = self._native_block_state(q_perm, k_perm, v_perm, block_mask)
            residual_logz, residual_output, residual_interactions = self._frontier_residual_state(
                q_perm, k_perm, v_perm, frontier_mask
            )
            output = self._merge_attention_states(
                block_state,
                (residual_logz, residual_output),
                q.dtype,
            )
            return self._apply_inverse_permutation(output, inverse_permutation), residual_interactions

        output = torch.empty_like(q_perm)
        residual_interactions = 0
        scale = head_dim ** -0.5

        def pack_indices(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Pack each head's true token indices into one rectangular tensor."""
            counts = mask.sum(dim=-1)
            max_count = int(counts.max().item())
            if max_count == 0:
                empty_idx = torch.empty((num_heads, 0), dtype=torch.long, device=mask.device)
                empty_valid = torch.empty((num_heads, 0), dtype=torch.bool, device=mask.device)
                return empty_idx, empty_valid
            head_idx, token_idx = mask.nonzero(as_tuple=True)
            rank = torch.cumsum(mask.to(torch.int32), dim=-1) - 1
            flat_index = head_idx * max_count + rank[head_idx, token_idx]
            packed = torch.zeros((num_heads * max_count,), dtype=torch.long, device=mask.device)
            packed.scatter_(0, flat_index, token_idx)
            valid = torch.arange(max_count, device=mask.device)[None, :] < counts[:, None]
            return packed.view(num_heads, max_count), valid

        for q_block in range(block_mask.shape[1]):
            start = q_block * self.block_size
            end = min(start + self.block_size, seq_len)
            if start >= end:
                break
            q_chunk = q_perm[0, :, start:end, :]
            selected_keys = block_mask[:, q_block].repeat_interleave(
                self.block_size, dim=-1
            )[:, :seq_len]

            selected_idx, selected_valid = pack_indices(selected_keys)
            selected_k = torch.gather(
                k_perm[0], 1, selected_idx[:, :, None].expand(-1, -1, head_dim)
            )
            selected_v = torch.gather(
                v_perm[0], 1, selected_idx[:, :, None].expand(-1, -1, head_dim)
            )
            selected_score = torch.matmul(
                q_chunk, selected_k.transpose(-2, -1)
            ).float() * scale
            selected_score = selected_score.masked_fill(
                ~selected_valid[:, None, :], float("-inf")
            )

            frontier_keys = frontier_mask[:, q_block].repeat_interleave(
                self.block_size, dim=-1
            )[:, :seq_len]
            frontier_idx, frontier_valid = pack_indices(frontier_keys)
            frontier_count = frontier_idx.shape[1]
            residual_k = min(self.token_top_k, frontier_count)
            if residual_k > 0:
                frontier_k = torch.gather(
                    k_perm[0], 1, frontier_idx[:, :, None].expand(-1, -1, head_dim)
                )
                frontier_score = torch.matmul(
                    q_chunk, frontier_k.transpose(-2, -1)
                ).float() * scale
                frontier_score = frontier_score.masked_fill(
                    ~frontier_valid[:, None, :], float("-inf")
                )
                ranking_score = frontier_score - frontier_idx[:, None, :].to(
                    frontier_score.dtype
                ) * 1.0e-8
                _, residual_local_idx = torch.topk(
                    ranking_score, k=residual_k, dim=-1
                )
                residual_score = torch.gather(frontier_score, -1, residual_local_idx)
                residual_valid = torch.gather(
                    frontier_valid[:, None, :].expand(-1, end - start, -1),
                    dim=-1,
                    index=residual_local_idx,
                )
                residual_idx = torch.gather(
                    frontier_idx[:, None, :].expand(-1, end - start, -1),
                    dim=-1,
                    index=residual_local_idx,
                )
                residual_v = torch.gather(
                    v_perm[0][:, None, :, :].expand(-1, end - start, -1, -1),
                    2,
                    residual_idx[:, :, :, None].expand(-1, -1, -1, head_dim),
                )
                residual_score = residual_score.masked_fill(
                    ~residual_valid, float("-inf")
                )
                residual_interactions += int(residual_valid.sum().item())
            else:
                residual_score = None
                residual_v = None
                residual_valid = None

            if residual_score is not None:
                active_score = torch.cat((selected_score, residual_score), dim=-1)
            else:
                active_score = selected_score
            probability = torch.softmax(active_score, dim=-1)
            selected_probability = probability[..., : selected_v.shape[1]]
            selected_output = torch.bmm(
                selected_probability.to(selected_v.dtype), selected_v
            )
            if residual_score is not None:
                residual_probability = probability[..., selected_v.shape[1] :]
                residual_output = (
                    residual_probability.to(residual_v.dtype).unsqueeze(-1)
                    * residual_v
                ).sum(dim=2)
                selected_output = selected_output + residual_output
            output[0, :, start:end, :] = selected_output

        return self._apply_inverse_permutation(output, inverse_permutation), residual_interactions


    def _run_block_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        permutation: torch.Tensor,
        inverse_permutation: torch.Tensor,
        block_mask: torch.Tensor,
        m_block_dim: int,
        n_block_dim: int,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_kv: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_kv: Optional[int],
        attn_mask: Optional[torch.Tensor],
        causal: bool,
        drop_rate: float,
        batch_size: int,
        hybrid_plan: Optional[HybridPlan] = None,
        kp_execution_mask=None,
    ) -> torch.Tensor:


        assert batch_size == 1, "DFS Attention with block sparse currently supports batch size 1."

        force_native_kp_identity = False
        if self.selector_mode == "kp" and self.fine_top_p < 1.0:
            if m_block_dim != 16 or n_block_dim != 16 or kp_execution_mask is None:
                raise ValueError("KP execution requires a compact 16x16 FlexAttention mask")
            q_perm = self._apply_permutation_in_place(q, permutation)
            k_perm = self._apply_permutation_in_place(k, permutation)
            v_perm = self._apply_permutation_in_place(v, permutation)
            x = kp_fine_sparse_attention(
                q_perm,
                k_perm,
                v_perm,
                kp_execution_mask,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
            )
            return self._apply_inverse_permutation(x, inverse_permutation)

        if self.selector_mode == "kp":
            # p=1 is the identity/golden path.  Execute it with upstream
            # DFSAttn's original 128x128 native kernel instead of accepting
            # any accumulated numerical drift from the custom KP kernel.
            if self.fine_top_p < 1.0 or kp_execution_mask is None:
                raise ValueError("KP p=1 identity execution requires a compact fine mask")
            padded_fine_mask = kp_execution_mask.fine_mask
            ratio = self.block_size // self.q_tile_size
            heads, fine_q_blocks, fine_k_blocks = padded_fine_mask.shape
            if fine_q_blocks % ratio != 0 or fine_k_blocks % ratio != 0:
                raise ValueError("KP identity mask must be padded to 128x128 boundaries")
            block_mask = padded_fine_mask.view(
                heads,
                fine_q_blocks // ratio,
                ratio,
                fine_k_blocks // ratio,
                ratio,
            ).any(dim=(2, 4))
            m_block_dim = self.block_size
            n_block_dim = self.block_size
            force_native_kp_identity = True

        if self.sparse_execution == "hybrid" and not force_native_kp_identity:
            if m_block_dim != 16 or n_block_dim != 16:
                raise ValueError(
                    "Hybrid DFSAttn requires a 16x16 logical mask; pass --block_size 16. "
                    "Its Core path then groups 4x4 microblocks into 64x64 tiles."
                )
            if causal or drop_rate != 0.0 or attn_mask is not None:
                raise ValueError("The initial Hybrid DFSAttn backend supports only non-causal attention without dropout or an extra token mask.")
            q_perm = self._apply_permutation_in_place(q, permutation)
            k_perm = self._apply_permutation_in_place(k, permutation)
            v_perm = self._apply_permutation_in_place(v, permutation)
            if hybrid_plan is None:
                hybrid_plan = partition_block_mask(block_mask.bool(), threshold=self.hybrid_threshold)
            x = hybrid_sparse_attention_from_plan(
                q_perm, k_perm, v_perm, hybrid_plan,
                timing_recorder=DFS_Attention.timing_recorder,
                step_idx=self.step_idx,
                layer_idx=self.layer_idx,
            )
            return self._apply_inverse_permutation(x, inverse_permutation)
        if self.sparse_execution != "native" and not force_native_kp_identity:
            raise ValueError(f"Unknown sparse_execution={self.sparse_execution!r}; expected 'native' or 'hybrid'.")
        if block_sparse_attn_func is None:
            raise ImportError("block_sparse_attn is unavailable; install its CUDA extension for native DFSAttn.")

        device = q.device
        _, num_heads, seq_len, head_dim = q.shape

        q_perm = self._apply_permutation_in_place(q, permutation)
        k_perm = self._apply_permutation_in_place(k, permutation)
        v_perm = self._apply_permutation_in_place(v, permutation)

        q_block_num = math.ceil(seq_len / m_block_dim)
        k_block_num = math.ceil(seq_len / n_block_dim)

        # Ensure block_mask is on correct device (should already be on GPU from cache)
        if block_mask.device != device:
            block_mask = block_mask.to(device, non_blocking=True)
        if (
            block_mask.shape[1] != q_block_num
            or block_mask.shape[2] != k_block_num
        ):
            padded_mask = torch.zeros(
                (block_mask.shape[0], q_block_num, k_block_num),
                dtype=torch.bool,
                device=device,
            )
            padded_mask[:, : block_mask.shape[1], : block_mask.shape[2]] = block_mask
            block_mask = padded_mask
        base_blockmask = block_mask.unsqueeze(0).repeat(cu_seqlens_q.numel() - 1, 1, 1, 1).bool()

        q_bs = q_perm.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_heads, head_dim)
        k_bs = k_perm.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_heads, head_dim)
        v_bs = v_perm.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_heads, head_dim)

        # Prepare cu_seqlens if not provided
        if cu_seqlens_q is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
            cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            cu[1:] = torch.cumsum(lengths, dim=0)
            cu_seqlens_q = cu
        if cu_seqlens_kv is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
            cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            cu[1:] = torch.cumsum(lengths, dim=0)
            cu_seqlens_kv = cu

        head_mask_type = torch.tensor([1] * num_heads, device=device, dtype=torch.int32)

        x = block_sparse_attn_func(
            q_bs, k_bs, v_bs, cu_seqlens_q, cu_seqlens_kv,
            head_mask_type=head_mask_type, streaming_info=None,
            base_blockmask=base_blockmask, max_seqlen_q_=max_seqlen_q,
            max_seqlen_k_=max_seqlen_kv, p_dropout=drop_rate,
            deterministic=False, softmax_scale=None, is_causal=causal,
            exact_streaming=False, return_attn_probs=False,
        )

        x = x.view(batch_size, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3)
        x = self._apply_inverse_permutation(x, inverse_permutation)
        return x

    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                causal: bool = False,
                drop_rate: float = 0.0,
                cu_seqlens_q: Optional[torch.Tensor] = None,
                cu_seqlens_kv: Optional[torch.Tensor] = None,
                max_seqlen_q: Optional[int] = None,
                max_seqlen_kv: Optional[int] = None,
                batch_size: int = 1) -> torch.Tensor:
        """
        Forward pass for DFS Attention
        
        Args:
            q, k, v: Query, key, value tensors
            attn_mask: Optional attention mask
            causal: Whether to use causal attention
            drop_rate: Dropout rate
            cu_seqlens_q, cu_seqlens_kv: Cumulative sequence lengths for flash attention
            max_seqlen_q, max_seqlen_kv: Maximum sequence lengths
            batch_size: Batch size
            
        Returns:
            Output tensor after attention
        """
        _, _, seq_len, _ = q.shape
        device = q.device
        if self.video_perm is not None:
            video_perm = self.video_perm.to(device)
        else:
            video_perm = None
        permutation, inverse_permutation = self._compute_permutation(video_perm, seq_len, device)
        
        block_mask = None
        frontier_mask = None
        kp_execution_mask = None
        mask_recomputed = False

        # Get cached block mask if available
        cache_entry = DFS_Attention._cached_metadata.get(self._mask_cache_key())

        if cache_entry is not None:
            block_mask = cache_entry["block_mask"]
            if block_mask.device != device:
                block_mask = block_mask.to(device, non_blocking=True)
            frontier_mask = cache_entry.get("frontier_mask")
            if frontier_mask is not None and frontier_mask.device != device:
                frontier_mask = frontier_mask.to(device, non_blocking=True)
            kp_execution_mask = cache_entry.get("kp_execution_mask")


        selector_start = None
        if block_mask is None or self.cache_flag or (
            self.token_top_k > 0 and frontier_mask is None
        ):
            selector_start = DFS_Attention.timing_recorder.start(device)
            topk_start = DFS_Attention.timing_recorder.start(device)
            with torch.no_grad():
                block_result = self._compute_block_mask(
                    q,
                    k,
                    permutation,
                    self.sparse_ratio,
                    self.block_top_p,
                    return_frontier=self.token_top_k > 0,
                )
                if self.token_top_k > 0:
                    block_mask, frontier_mask = block_result
                else:
                    block_mask = block_result
            DFS_Attention.timing_recorder.stop(
                topk_start, phase="topk_mask", step_idx=self.step_idx,
                layer_idx=self.layer_idx, device=device,
            )
            mask_recomputed = True

            if self.cache_flag:
                if self.selector_mode == "kp":
                    # KP masks are much larger than upstream 128x128 masks.
                    # Only the current sparsity schedule entry is useful for a
                    # layer; retaining old .3/.2/.1 entries would consume tens
                    # of GiB across 60 layers.
                    stale_keys = [
                        key
                        for key in DFS_Attention._cached_metadata
                        if key[0] == self.layer_idx and key != self._mask_cache_key()
                    ]
                    for key in stale_keys:
                        del DFS_Attention._cached_metadata[key]
                DFS_Attention._cached_metadata[self._mask_cache_key()] = {
                    "block_mask": (
                        block_mask if self.selector_mode == "kp" else block_mask.clone()
                    ),
                    "frontier_mask": None if frontier_mask is None else frontier_mask.clone(),
                }

        if self.token_top_k > 0 and frontier_mask is None:
            raise RuntimeError("Residual token Top-k requires a cached frontier mask.")

        hybrid_plan = None
        if self.selector_mode == "kp" and kp_execution_mask is None:
            routing_start = DFS_Attention.timing_recorder.start(device)
            kp_execution_mask = compact_block_mask(block_mask, block_size=128)
            DFS_Attention.timing_recorder.stop(
                routing_start, phase="routing", step_idx=self.step_idx,
                layer_idx=self.layer_idx, device=device,
            )
            if self.cache_flag:
                DFS_Attention._cached_metadata[self._mask_cache_key()][
                    "kp_execution_mask"
                ] = kp_execution_mask
        elif self.sparse_execution == "hybrid":
            routing_start = DFS_Attention.timing_recorder.start(device)
            hybrid_plan = partition_block_mask(block_mask.bool(), threshold=self.hybrid_threshold)
            DFS_Attention.timing_recorder.stop(
                routing_start, phase="routing", step_idx=self.step_idx,
                layer_idx=self.layer_idx, device=device,
            )
        # This is the requested Top-k + dense/sparse routing timing.  On a
        # cached mask it is intentionally not emitted because no Top-k ran.
        if mask_recomputed:
            DFS_Attention.timing_recorder.stop(
                selector_start, phase="topk_routing", step_idx=self.step_idx,
                layer_idx=self.layer_idx, device=device,
            )

        should_export_mask = (
            mask_recomputed
            and DFS_Attention.mask_output_dir is not None
            and DFS_Attention.mask_layer_interval > 0
            and self.layer_idx % DFS_Attention.mask_layer_interval == 0
        )
        if should_export_mask:
            from .utils.visualization import export_block_mask

            bool_path, png_paths = export_block_mask(
                block_mask,
                DFS_Attention.mask_output_dir,
                self.step_idx,
                self.layer_idx,
                DFS_Attention.mask_head_indices,
                DFS_Attention.mask_save_bool,
            )
            logger.info(
                "Saved top-k block heatmaps for step {} layer {} ({} heatmap(s)){}",
                self.step_idx,
                self.layer_idx,
                len(png_paths),
                " and bool mask to " + bool_path if bool_path else "",
            )

        # Record the actual density of the sparse mask for this step if enabled
        if DFS_Attention.record_density:
            with torch.no_grad():
                logical_block_size = 16 if self.selector_mode == "kp" else self.block_size
                block_interactions = self._block_interaction_count(
                    block_mask, seq_len, logical_block_size
                )
                block_density = block_interactions / (block_mask.shape[0] * seq_len * seq_len)
            DFS_Attention.density_records.setdefault(self.step_idx, {})[self.layer_idx] = block_density

        attention_start = DFS_Attention.timing_recorder.start(device)
        if self.token_top_k > 0:
            output, residual_interactions = self._run_oracle_hybrid_attention(
                q, k, v, permutation, inverse_permutation, block_mask, frontier_mask
            )
        else:
            execution_block_size = 16 if self.selector_mode == "kp" else self.block_size
            output = self._run_block_sparse_attention(
                q, k, v, permutation, inverse_permutation,
                block_mask, execution_block_size, execution_block_size,
                cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                attn_mask, causal, drop_rate, batch_size,
                hybrid_plan,
                kp_execution_mask,
            )
            residual_interactions = 0
        DFS_Attention.timing_recorder.stop(
            attention_start, phase="attention_execution", step_idx=self.step_idx,
            layer_idx=self.layer_idx, device=device,
        )

        if DFS_Attention.record_density:
            total_interactions = block_interactions + residual_interactions
            total_possible = block_mask.shape[0] * seq_len * seq_len
            DFS_Attention.sparsity_records.setdefault(self.step_idx, {})[self.layer_idx] = {
                "p_mass": float("nan") if self.block_top_p is None else self.block_top_p,
                "fine_top_p": self.fine_top_p if self.selector_mode == "kp" else float("nan"),
                "token_top_k": float(self.token_top_k),
                "residual_candidate_blocks": float(self.residual_candidate_blocks),
                "block_density": block_interactions / total_possible,
                "realized_block_sparsity": 1.0 - block_interactions / total_possible,
                "residual_token_interactions": float(residual_interactions),
                "final_density": total_interactions / total_possible,
                "final_hybrid_sparsity": 1.0 - total_interactions / total_possible,
            }

        return output

    @classmethod
    def clear_cache(cls):
        """Clear the sparse indices cache and all device-specific permutation caches"""
        cls._cached_metadata.clear()
        cls.density_records.clear()
        cls.sparsity_records.clear()
        _flashinfer64_instances.clear()

    @classmethod
    def configure_mask_export(
        cls,
        output_dir: Optional[str],
        head_indices: Optional[Tuple[int, ...]] = (0,),
        layer_interval: int = 15,
        save_bool: bool = False,
    ):
        """Configure heatmap export for newly computed top-k masks."""
        if layer_interval < 1:
            raise ValueError("layer_interval must be at least 1")
        cls.mask_output_dir = output_dir
        cls.mask_head_indices = head_indices
        cls.mask_layer_interval = layer_interval
        cls.mask_save_bool = bool(save_bool)

    @classmethod
    def configure_subblock_retention_profile(
        cls,
        output_dir: Optional[str],
        masses: Tuple[float, ...] = (0.9,),
    ):
        cls.subblock_profiler.configure(output_dir, masses)

    @classmethod
    def dump_subblock_retention_profile(cls) -> Optional[str]:
        return cls.subblock_profiler.dump_csv()

    @classmethod
    def set_record_density(cls, enabled: bool):
        """Toggle recording of the actual density of sparse attention masks"""
        cls.record_density = bool(enabled)
        if not cls.record_density:
            cls.density_records.clear()
            cls.sparsity_records.clear()

    @classmethod
    def set_record_timing(cls, enabled: bool):
        """Enable CUDA-event timing without synchronizing every attention call."""
        cls.timing_recorder.reset(bool(enabled))

    @classmethod
    def dump_timing_records(
        cls,
        output_path: str,
        extra_totals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        return cls.timing_recorder.dump_csv(output_path, extra_totals=extra_totals)

    @classmethod
    def get_density_records(cls) -> Dict[int, Dict[int, float]]:
        """Return recorded actual densities: {step_idx: {layer_idx: density}}"""
        return cls.density_records

    @classmethod
    def dump_density_records(cls, output_path: Optional[str] = None) -> Optional[str]:
        """Print recorded densities and optionally save them to a CSV file.

        CSV includes density plus Residual CSR row-length statistics when the
        hierarchical FlashInfer backend supplied them.
        """
        records = cls.density_records
        if not records:
            logger.warning("No density records found. Make sure record_density was enabled during generation.")
            return None

        flat = []
        for step in sorted(records):
            for layer, density in sorted(records[step].items()):
                flat.append((step, layer, density))

        header = f"{'step_idx':>9} {'layer_idx':>10} {'density':>9} {'density%':>9}"
        print("\n===== DFS Attention actual density records =====")
        print(header)
        print("-" * len(header))
        step_sums: Dict[int, float] = {}
        step_counts: Dict[int, int] = {}
        for step, layer, density in flat:
            step_sums[step] = step_sums.get(step, 0.0) + density
            step_counts[step] = step_counts.get(step, 0) + 1
            print(f"{step:>9} {layer:>10} {density:>9.4f} {density * 100:>9.2f}")
        all_mean = sum(d for _, _, d in flat) / len(flat)
        print(f"\nMean density over all layers/steps: {all_mean:.4f} ({all_mean * 100:.2f}%)")
        step_mean = sum(
            step_sums[step] / step_counts[step]
            for step in step_sums
        ) / len(step_sums)
        print(f"Mean density per step (averaged over layers): {step_mean:.4f} ({step_mean * 100:.2f}%)")
        print("================================================")

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", newline="") as f:
                metric_fields = [
                    "residual_count_mean", "residual_count_p50", "residual_count_p95",
                    "residual_count_max", "residual_count_nonempty_ratio",
                ]
                writer = csv.DictWriter(
                    f, fieldnames=["step_idx", "layer_idx", "density", *metric_fields]
                )
                writer.writeheader()
                for step, layer, density in flat:
                    sparse = cls.sparsity_records.get(step, {}).get(layer, {})
                    writer.writerow({
                        "step_idx": step,
                        "layer_idx": layer,
                        "density": density,
                        **{
                            field: sparse.get(field, float("nan"))
                            for field in metric_fields
                        },
                    })
            print(f"Density records saved to: {output_path}")
        return output_path

    @classmethod
    def dump_density_summary(
        cls,
        output_path: Optional[str] = None,
        prompt_idx: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Append one prompt's density averaged over sparse steps and layers.

        ``mean_final_density_sparse_steps_avg_over_layers`` is the canonical
        density for experiment comparisons.  For each sparse denoising step we
        average the final selected QK density over its recorded layers, then
        average those step means.  Dense warmup steps are not recorded and
        therefore cannot affect this number.

        The summary is intentionally one row per prompt so a multi-prompt shell
        sweep can append to the same CSV without overwriting earlier prompts.
        A legacy two-column ``metric,value`` file is migrated on first append.
        """
        records = cls.density_records
        if not records:
            logger.warning("No density records found. Make sure record_density was enabled during generation.")
            return None

        values = [density for step in sorted(records) for density in records[step].values()]
        step_means = [
            sum(records[step].values()) / len(records[step])
            for step in sorted(records)
            if records[step]
        ]
        mean_final_density_sparse_steps_avg_over_layers = sum(step_means) / len(step_means)
        summary = {
            "prompt_idx": "" if prompt_idx is None else prompt_idx,
            "prompt": "" if prompt is None else prompt,
            "density_definition": (
                "final_selected_qk / all_qk; "
                "mean_over_sparse_steps_of_mean_over_recorded_layers"
            ),
            # Canonical, unambiguous comparison value.  In particular, this
            # is not the sum over layers (which would be 60x larger here).
            "mean_final_density_sparse_steps_avg_over_layers": (
                mean_final_density_sparse_steps_avg_over_layers
            ),
            "mean_density_over_sparse_step_layer_records": sum(values) / len(values),
            # Keep the original field name for compatibility with existing
            # analysis scripts, and expose the printed metric explicitly.
            "mean_density_per_sparse_step": mean_final_density_sparse_steps_avg_over_layers,
            "mean_density_per_step_averaged_over_layers": mean_final_density_sparse_steps_avg_over_layers,
            "num_sparse_step_layer_records": len(values),
            "num_sparse_steps": len(step_means),
            "num_layers_with_records": len({layer for step in records.values() for layer in step}),
        }
        route_metrics = (
            "residual_count_mean", "residual_count_p50", "residual_count_p95",
            "residual_count_max", "residual_count_nonempty_ratio",
        )
        sparse_values = [
            values
            for step in sorted(cls.sparsity_records)
            for values in cls.sparsity_records[step].values()
        ]
        for metric in route_metrics:
            metric_values = [
                float(values[metric]) for values in sparse_values
                if metric in values and math.isfinite(float(values[metric]))
            ]
            summary[f"mean_{metric}"] = (
                sum(metric_values) / len(metric_values) if metric_values else float("nan")
            )
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fieldnames = list(summary.keys())
            existing_rows = []
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path, "r", newline="") as existing_file:
                    reader = csv.DictReader(existing_file)
                    existing_fieldnames = reader.fieldnames or []
                    if existing_fieldnames == ["metric", "value"]:
                        legacy_values = {
                            row.get("metric", ""): row.get("value", "")
                            for row in reader
                            if row.get("metric")
                        }
                        existing_rows.append({
                            name: legacy_values.get(name, "")
                            for name in fieldnames
                        })
                    elif set(existing_fieldnames).issubset(fieldnames):
                        existing_rows = [
                            {name: row.get(name, "") for name in fieldnames}
                            for row in reader
                        ]
                    else:
                        raise ValueError(
                            f"Unexpected density summary schema in {output_path}: "
                            f"{existing_fieldnames}"
                        )
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(summary)
            logger.info("Sparse density summary saved to {}", output_path)
        return output_path

    @classmethod
    def dump_sparsity_records(cls, output_path: Optional[str] = None) -> Optional[str]:
        """Write phase-A block/residual interaction statistics to CSV."""
        records = cls.sparsity_records
        if not records:
            logger.warning("No sparsity records found. Make sure --record_density true was enabled.")
            return None

        flat = []
        for step in sorted(records):
            for layer, values in sorted(records[step].items()):
                flat.append((step, layer, values))
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fields = [
                "p_mass", "fine_top_p", "token_top_k", "token_top_ratio", "residual_candidate_blocks", "block_density", "realized_block_sparsity",
                "residual_token_interactions", "residual_micro_tiles",
                "promoted_macro_tiles",
                "residual_count_mean", "residual_count_p50", "residual_count_p95",
                "residual_count_max", "residual_count_nonempty_ratio",
                "final_density", "final_hybrid_sparsity",
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step_idx", "layer_idx", *fields])
                writer.writeheader()
                for step, layer, values in flat:
                    writer.writerow({
                        "step_idx": step,
                        "layer_idx": layer,
                        **{field: values.get(field, float("nan")) for field in fields},
                    })
            logger.info("Phase-A sparsity records saved to {}", output_path)
        return output_path

    @classmethod
    def get_cache_info(cls) -> Dict:
        """Get information about the current cache"""
        return {
            'num_cached_steps': len(cls._cached_metadata),
            'cached_steps': list(cls._cached_metadata.keys()),
            'total_cached_layers': sum(len(layers) for layers in cls._cached_metadata.values()),
        }


def dfs_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    step_idx: int = 0,
    skip_steps: int = 12,
    cache_interval: int = 12,
    layer_idx: int = 0,
    sparsity: float = 0.25,
    sparsity_dcrt: float = 0.1,
    tile_size: int = 32,
    block_size: int = 128,
    video_len: int = 0,
    video_perm: Optional[torch.Tensor] = None,
    cache_flag: bool = True,
    record_density: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    sparse_execution: str = "native",
    hybrid_threshold: int = 8,
    block_top_p: Optional[float] = None,
    token_top_k: int = 0,
    residual_candidate_blocks: int = 4,
    selector_mode: str = "topk",
    fine_top_p: float = 0.9,
    flashinfer64_top_p: float = 0.25,
    flashinfer64_token_top_k: Optional[int] = None,
    flashinfer64_token_top_ratio: float = 0.10,
    flashinfer64_route_mode: str = "topk_topp",
    flashinfer64_tile_top_ratio: float = 0.25,
    flashinfer64_token_top_p: float = 0.9,
    flashinfer64_promotion_threshold: int = 24,
    flashinfer64_route_cache: bool = False,
    flashinfer64_core_only: bool = False,
    flashinfer64_direct_macro_csr: bool = True,
    flashinfer64_valid_sequence: Optional[int] = None,
) -> torch.Tensor:
    """
    DFS Attention wrapper function adapted for replace_hyvideo.py structure.
    This function calls dfs_attention from attenion.py.
    
    Args:
        q (torch.Tensor): Query tensor with shape [B, H, L, D] (batch, heads, length, dim)
        k (torch.Tensor): Key tensor with shape [B, H, L, D]
        v (torch.Tensor): Value tensor with shape [B, H, L, D]
        step_idx (int): Current step index
        skip_steps (int): Number of diffusion steps to run with full attention
        cache_interval (int): Diffusion-step interval between sparse mask refreshes
        layer_idx (int): Current layer index
        sparsity (float): Initial sparsity ratio (0.0-1.0)
        sparsity_dcrt (float): Sparsity decrement applied each cache interval
        tile_size (int): Tile size for both q and k
        block_size (int): Block dimension for sparse attention
        video_perm (Optional[torch.Tensor]): Video permutation tensor
        cache_flag (bool): Whether to cache the sparse block mask
        record_density (bool): Whether to record the actual density of sparse masks (stored in DFS_Attention.density_records)
        cu_seqlens_q (Optional[torch.Tensor]): Cumulative sequence lengths for q
        cu_seqlens_kv (Optional[torch.Tensor]): Cumulative sequence lengths for k and v
        max_seqlen_q (Optional[int]): Maximum sequence length for q
        max_seqlen_kv (Optional[int]): Maximum sequence length for k and v
        
    Returns:
        torch.Tensor: Output tensor with shape [B, H, L, D]
    """
    
    if sparse_execution == "flashinfer64":
        backend = _flashinfer64_instances.setdefault(layer_idx, FlashInfer64Attention())
        if record_density and not DFS_Attention.record_density:
            DFS_Attention.set_record_density(True)
        if cache_interval <= 0:
            raise ValueError("cache_interval must be greater than 0.")
        # FlashInfer64 has fixed Top-p/Top-k-ratio parameters and therefore
        # must not inherit native DFSAttn's sparsity_dcrt schedule cutoff.
        # Refresh exactly every cache_interval sparse steps.
        is_cache_step = (
            step_idx >= skip_steps
            and (step_idx - skip_steps) % cache_interval == 0
        )
        attention_start = DFS_Attention.timing_recorder.start(q.device)
        output = backend(
            q,
            k,
            v,
            video_perm=video_perm,
            video_len=video_len,
            route_mode=flashinfer64_route_mode,
            tile_top_p=flashinfer64_top_p,
            tile_top_ratio=flashinfer64_tile_top_ratio,
            token_top_k=flashinfer64_token_top_k,
            token_top_ratio=flashinfer64_token_top_ratio,
            token_top_p=flashinfer64_token_top_p,
            promotion_threshold=flashinfer64_promotion_threshold,
            reuse_route=flashinfer64_route_cache,
            core_only=flashinfer64_core_only,
            direct_macro_csr=flashinfer64_direct_macro_csr,
            valid_sequence=flashinfer64_valid_sequence,
            refresh_route=cache_flag and is_cache_step,
            record_density=record_density,
            timing_recorder=DFS_Attention.timing_recorder,
            step_idx=step_idx,
            layer_idx=layer_idx,
        )
        DFS_Attention.timing_recorder.stop(
            attention_start,
            phase="attention_execution",
            step_idx=step_idx,
            layer_idx=layer_idx,
            device=q.device,
        )
        if record_density and backend.last_stats is not None:
            stats = backend.last_stats
            total_possible = stats["total_possible"]
            core_density = stats["core_interactions"] / total_possible
            final_interactions = stats["core_interactions"] + stats["residual_token_interactions"]
            final_density = final_interactions / total_possible
            DFS_Attention.density_records.setdefault(step_idx, {})[layer_idx] = final_density
            DFS_Attention.sparsity_records.setdefault(step_idx, {})[layer_idx] = {
                "p_mass": float(flashinfer64_top_p) if flashinfer64_route_mode == "topp_topk" else float("nan"),
                "fine_top_p": float(flashinfer64_token_top_p),
                "token_top_k": float(-1),
                "token_top_ratio": float(flashinfer64_tile_top_ratio) if flashinfer64_route_mode == "topk_topp" else float(flashinfer64_token_top_ratio),
                "residual_candidate_blocks": float("nan"),
                "block_density": core_density,
                "realized_block_sparsity": 1.0 - core_density,
                "residual_token_interactions": float(stats["residual_token_interactions"]),
                "residual_micro_tiles": float(stats.get("residual_micro_tiles", 0)),
                "promoted_macro_tiles": float(stats.get("promoted_macro_tiles", 0)),
                "residual_count_mean": float(stats.get("residual_count_mean", float("nan"))),
                "residual_count_p50": float(stats.get("residual_count_p50", float("nan"))),
                "residual_count_p95": float(stats.get("residual_count_p95", float("nan"))),
                "residual_count_max": float(stats.get("residual_count_max", float("nan"))),
                "residual_count_nonempty_ratio": float(stats.get("residual_count_nonempty_ratio", float("nan"))),
                "final_density": final_density,
                "final_hybrid_sparsity": 1.0 - final_density,
            }
        return output

    instance_key = layer_idx
    
    # Get or create DFS_Attention instance for this layer
    if instance_key not in _dfs_attention_instances:
        _dfs_attention_instances[instance_key] = DFS_Attention(
            step_idx,
            layer_idx,
            sparsity,
            cache_flag,
            video_len,
            video_perm,
            block_size,
            tile_size,
            tile_size,
            sparse_execution,
            hybrid_threshold,
            block_top_p,
            token_top_k,
            residual_candidate_blocks,
            selector_mode,
            fine_top_p,
        )
    
    # Get the instance and update step_idx
    dfs_attn = _dfs_attention_instances[instance_key]
    dfs_attn.step_idx = step_idx

    is_cache_step, current_sparsity = _compute_cache_schedule(
        step_idx,
        skip_steps,
        cache_interval,
        sparsity,
        sparsity_dcrt,
    )
    dfs_attn.sparse_ratio = current_sparsity
    dfs_attn.cache_flag = cache_flag and is_cache_step
    dfs_attn.sparse_execution = sparse_execution
    dfs_attn.hybrid_threshold = hybrid_threshold
    dfs_attn.block_top_p = block_top_p
    dfs_attn.token_top_k = token_top_k
    dfs_attn.residual_candidate_blocks = residual_candidate_blocks
    dfs_attn.selector_mode = selector_mode
    dfs_attn.fine_top_p = fine_top_p

    # Enable global recording of actual densities (latch on once enabled)
    if record_density and not DFS_Attention.record_density:
        DFS_Attention.set_record_density(True)

    # Call forward method
    # Input is already in [B, H, L, D] format which DFS_Attention expects
    output = dfs_attn.forward(
        q=q,
        k=k,
        v=v,
        attn_mask=None,  # DFS_Attention handles attention mask internally if needed
        causal=False,  # Non-causal for video generation
        drop_rate=0.0,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=q.shape[0],
    )
    
    # Output is already in [B, H, L, D] format
    return output
