import torch
from typing import Optional

# Cache for DFS_Attention instances per layer
_dfs_attention_instances = {}


import math
import os
import csv
from typing import List, Tuple, Optional, Dict, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from block_sparse_attn import block_sparse_attn_func


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
    _cached_metadata: Dict[int, Dict[str, Any]] = {}
    mask_output_dir: Optional[str] = None
    mask_head_indices: Optional[Tuple[int, ...]] = (0,)
    mask_layer_interval: int = 15
    mask_save_bool: bool = False
    # Toggle for recording the actual density of sparse attention masks
    record_density: bool = False
    # Recorded actual densities keyed by step_idx -> {layer_idx: density}
    density_records: Dict[int, Dict[int, float]] = {}

    def __init__(
        self, 
        step_idx: int,
        layer_idx: int,
        sparse_ratio: float = 0.5,
        cache_flag: bool = True,
        video_perm: Optional[torch.Tensor] = None,
        block_size: int = 128,
        q_tile_size: int = 32,
        k_tile_size: int = 32,
    ):
        super(DFS_Attention, self).__init__()

        self.step_idx = step_idx
        self.layer_idx = layer_idx
        self.sparse_ratio = sparse_ratio
        self.cache_flag = cache_flag
        self.video_perm = video_perm
        self.block_size = block_size
        self.q_tile_size = q_tile_size
        self.k_tile_size = k_tile_size
    

    def _compute_permutation(self, video_perm: Optional[torch.Tensor], seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # If permutation is disabled, return identity permutation
        if video_perm is None:
            identity_perm = torch.arange(seq_len, device=device, dtype=torch.long)
            return identity_perm, identity_perm

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

        q_block_num = math.ceil(seq_len / block_dim)
        k_block_num = math.ceil(seq_len / block_dim)
        
        # Apply permutation
        q_perm = self._apply_permutation_in_place(q, permutation)
        k_perm = self._apply_permutation_in_place(k, permutation)
        
        # Pad video tokens to align to block boundaries
        video_padded_len = q_block_num * block_dim
        video_pad_size = video_padded_len - seq_len
        
        if video_pad_size > 0:
            q_pad = torch.zeros((bsz, num_heads, video_pad_size, head_dim), 
                                device=q.device, dtype=q.dtype)
            k_pad = torch.zeros((bsz, num_heads, video_pad_size, head_dim), 
                                device=k.device, dtype=k.dtype)
            q_padded = torch.cat([q_perm, q_pad], dim=2)
            k_padded = torch.cat([k_perm, k_pad], dim=2)
        else:
            q_padded = q_perm
            k_padded = k_perm
        
        # Compute the tile score of the video query and entire key
        q_tiles_num = video_padded_len // q_tile_size
        k_tiles_num = video_padded_len // k_tile_size
        
        # Mean pooling
        q_video_tiles = (
            q_padded[:, :, :, :]
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
    
    
    def _compute_block_mask(self, q: torch.Tensor, k: torch.Tensor, permutations: torch.Tensor, sparse_ratio: Optional[float] = None) -> torch.Tensor:
        
        bsz, num_heads, seq_len, _ = q.shape
        assert bsz == 1, "DFS Attention with block sparse currently supports batch size 1."


        assert self.block_size % self.q_tile_size == 0, f"block_dim ({self.block_size}) must be divisible by q_tile_size ({self.q_tile_size})"
        assert self.block_size % self.k_tile_size == 0, f"block_dim ({self.block_size}) must be divisible by k_tile_size ({self.k_tile_size})"

        block_dim = self.block_size
        q_block_num = k_block_num = math.ceil(seq_len / block_dim)
        tiles_per_q_block = block_dim // self.q_tile_size
        tiles_per_k_block = block_dim // self.k_tile_size

        # Compute tile scores
        tile_score = self._compute_tile_score(
                q, k, self.q_tile_size, self.k_tile_size, block_dim, permutations, 
        )

        q_tiles_num, k_tiles_num = tile_score.shape[2], tile_score.shape[3]
        assert q_tiles_num % tiles_per_q_block == 0 and k_tiles_num % tiles_per_k_block == 0, "q_tiles_num and k_tiles_num must be divisible by tiles_per_q_block and tiles_per_k_block"

        block_value_sums = (
            tile_score.view(1, num_heads, q_block_num, tiles_per_q_block, k_block_num, tiles_per_k_block)
            .sum(dim=(3, 5))  # Sum over tiles in each block
        )
        
        topk_block_num = max(1, int(sparse_ratio * k_block_num))
        _, topk_indices = torch.topk(block_value_sums, k=topk_block_num, dim=-1)  # Shape: (1, num_heads, video_blocks_num, topk_block_num)
            
        # Create final mask with in-place operations
        selected_mask = torch.zeros((bsz, num_heads, q_block_num, k_block_num), dtype=torch.bool, device=q.device)
        selected_mask.scatter_(-1, topk_indices, True)

        return selected_mask.squeeze(0)


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
    ) -> torch.Tensor:


        assert batch_size == 1, "DFS Attention with block sparse currently supports batch size 1."

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
            q_bs,
            k_bs,
            v_bs,
            cu_seqlens_q,
            cu_seqlens_kv,
            # m_block_dim=m_block_dim,
            # n_block_dim=n_block_dim,
            head_mask_type=head_mask_type,
            streaming_info=None,
            base_blockmask=base_blockmask,
            max_seqlen_q_=max_seqlen_q,
            max_seqlen_k_=max_seqlen_kv,
            p_dropout=drop_rate,
            deterministic=False,
            softmax_scale=None,
            is_causal=causal,
            exact_streaming=False,
            return_attn_probs=False,
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
        mask_recomputed = False

        # Get cached block mask if available
        cache_entry = (
            DFS_Attention._cached_metadata
            .get(self.layer_idx)
        )

        if cache_entry is not None:
            block_mask = cache_entry["block_mask"]
            if block_mask.device != device:
                block_mask = block_mask.to(device, non_blocking=True)


        if block_mask is None or self.cache_flag:
            with torch.no_grad():
                block_mask = self._compute_block_mask(
                    q,
                    k,
                    permutation,
                    self.sparse_ratio,
                )
            mask_recomputed = True

            if self.cache_flag:
                DFS_Attention._cached_metadata[
                    self.layer_idx
                ] = {
                    "block_mask": block_mask.clone(),  # Clone but keep on same device
                }

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
                actual_density = block_mask.float().mean().item()
            DFS_Attention.density_records.setdefault(self.step_idx, {})[self.layer_idx] = actual_density

        output = self._run_block_sparse_attention(
            q, k, v, permutation, inverse_permutation,
            block_mask, self.block_size, self.block_size,
            cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
            attn_mask, causal, drop_rate, batch_size,
        )

        return output

    @classmethod
    def clear_cache(cls):
        """Clear the sparse indices cache and all device-specific permutation caches"""
        cls._cached_metadata.clear()
        cls.density_records.clear()

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
    def set_record_density(cls, enabled: bool):
        """Toggle recording of the actual density of sparse attention masks"""
        cls.record_density = bool(enabled)
        if not cls.record_density:
            cls.density_records.clear()

    @classmethod
    def get_density_records(cls) -> Dict[int, Dict[int, float]]:
        """Return recorded actual densities: {step_idx: {layer_idx: density}}"""
        return cls.density_records

    @classmethod
    def dump_density_records(cls, output_path: Optional[str] = None) -> Optional[str]:
        """Print recorded densities and optionally save them to a CSV file.

        CSV columns: step_idx, layer_idx, density (fraction of kept blocks).
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
        for step, layer, density in flat:
            step_sums[step] = step_sums.get(step, 0.0) + density
            print(f"{step:>9} {layer:>10} {density:>9.4f} {density * 100:>9.2f}")
        all_mean = sum(d for _, _, d in flat) / len(flat)
        print(f"\nMean density over all layers/steps: {all_mean:.4f} ({all_mean * 100:.2f}%)")
        step_mean = sum(step_sums.values()) / len(step_sums)
        print(f"Mean density per step (averaged over layers): {step_mean:.4f} ({step_mean * 100:.2f}%)")
        print("================================================")

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step_idx", "layer_idx", "density"])
                writer.writerows(flat)
            print(f"Density records saved to: {output_path}")
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
    video_perm: Optional[torch.Tensor] = None,
    cache_flag: bool = True,
    record_density: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
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
    
    instance_key = layer_idx
    
    # Get or create DFS_Attention instance for this layer
    if instance_key not in _dfs_attention_instances:
        _dfs_attention_instances[instance_key] = DFS_Attention(
            step_idx,
            layer_idx,
            sparsity,
            cache_flag,
            video_perm,
            block_size,
            tile_size,
            tile_size,
        )
    
    # Get the instance and update step_idx
    dfs_attn = _dfs_attention_instances[instance_key]
    dfs_attn.step_idx = step_idx
    
    is_cache_step, current_sparsity = _compute_cache_schedule(
        step_idx,
        skip_steps * 2,
        cache_interval * 2,
        sparsity,
        sparsity_dcrt,
    )
    dfs_attn.sparse_ratio = current_sparsity
    dfs_attn.cache_flag = cache_flag and is_cache_step

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
