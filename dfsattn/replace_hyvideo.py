import json
import os

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from typing import Optional

from .fullattention import full_attention
from .attention_hyvideo import DFS_Attention, dfs_attention
from dfsattn.utils.logger import logger

# Try to import fast kernels
try:
    from .kernels import ENABLE_FAST_KERNEL, apply_qk_norm, apply_qk_rope_single, apply_qk_rope_double
    if ENABLE_FAST_KERNEL:
        logger.info("Fast CUDA/Triton kernels enabled for QK normalization and RoPE")
except ImportError:
    ENABLE_FAST_KERNEL = False
    apply_qk_norm = None
    apply_qk_rope_single = None
    apply_qk_rope_double = None
    logger.info("Fast CUDA/Triton kernels not available, using PyTorch fallback")


class AttentionDebugComplete(RuntimeError):
    """Internal clean-stop signal after requested debug layers are dumped."""


class HunyuanVideo_DFSAttn_Processor2_0:
    def __init__(
        self,
        mode,
        sparsity,
        tile_size,
        block_size,
        video_len,
        video_perm,
        processor_id,
        skip_layers,
        skip_steps,
        cache_interval,
        sparsity_dcrt,
        cache_flag,
        dense_interval=0,
        rest_steps=0,
        skip_steps2=0,
        record_density=False,
        sparse_execution="native",
        hybrid_threshold=8,
        block_top_p=None,
        token_top_k=0,
        residual_candidate_blocks=4,
        selector_mode="topk",
        fine_top_p=0.9,
        flashinfer64_top_p=0.25,
        flashinfer64_token_top_k=None,
        flashinfer64_token_top_ratio=0.10,
        flashinfer64_route_mode="topk_topp",
        flashinfer64_tile_top_ratio=0.25,
        flashinfer64_dynamic_tile_ratio=False,
        flashinfer64_fine_top_ratio=0.2,
        flashinfer64_fine_top_k=None,
        flashinfer64_token_top_p=0.9,
        flashinfer64_promotion_threshold=24,
        flashinfer64_dense_layer=-1,
        flashinfer64_dense_heads=(),
        flashinfer64_high_omission_heads_file=None,
        flashinfer64_residual_scorer="proxy",
        flashinfer64_residual_temperature=1.0,
        flashinfer64_residual_min_top_k=20,
        flashinfer64_residual_max_top_k=32,
        flashinfer64_route_cache=False,
        flashinfer64_core_only=False,
        flashinfer64_direct_macro_csr=True,
        attention_debug_dir=None,
        attention_debug_step=-1,
        attention_debug_layers=(0,),
        attention_debug_stop=True,
        attention_debug_steps=None,
    ):
        self.mode = mode
        self.sparsity = sparsity
        self.tile_size = tile_size
        self.block_size = block_size
        self.video_len = video_len
        self.video_perm = video_perm
        self.step_idx = 0
        self.layer_idx = processor_id
        self.skip_layers = skip_layers
        self.skip_steps = skip_steps
        self.cache_interval = cache_interval
        self.sparsity_dcrt = sparsity_dcrt
        self.cache_flag = cache_flag
        self.dense_interval = dense_interval
        self.rest_steps = rest_steps
        self.skip_steps2 = skip_steps2
        self.record_density = record_density
        self.sparse_execution = sparse_execution
        self.hybrid_threshold = hybrid_threshold
        self.block_top_p = block_top_p
        self.token_top_k = token_top_k
        self.residual_candidate_blocks = residual_candidate_blocks
        self.selector_mode = selector_mode
        self.fine_top_p = fine_top_p
        self.flashinfer64_top_p = flashinfer64_top_p
        self.flashinfer64_token_top_k = flashinfer64_token_top_k
        self.flashinfer64_token_top_ratio = flashinfer64_token_top_ratio
        self.flashinfer64_route_mode = flashinfer64_route_mode
        self.flashinfer64_tile_top_ratio = flashinfer64_tile_top_ratio
        self.flashinfer64_dynamic_tile_ratio = flashinfer64_dynamic_tile_ratio
        self.flashinfer64_fine_top_ratio = flashinfer64_fine_top_ratio
        self.flashinfer64_fine_top_k = flashinfer64_fine_top_k
        self.flashinfer64_token_top_p = flashinfer64_token_top_p
        self.flashinfer64_promotion_threshold = flashinfer64_promotion_threshold
        self.flashinfer64_dense_layer = flashinfer64_dense_layer
        self.flashinfer64_dense_heads = flashinfer64_dense_heads
        self.flashinfer64_high_omission_heads_file = flashinfer64_high_omission_heads_file
        self.flashinfer64_residual_scorer = flashinfer64_residual_scorer
        self.flashinfer64_residual_temperature = flashinfer64_residual_temperature
        self.flashinfer64_residual_min_top_k = flashinfer64_residual_min_top_k
        self.flashinfer64_residual_max_top_k = flashinfer64_residual_max_top_k
        self.flashinfer64_route_cache = flashinfer64_route_cache
        self.flashinfer64_core_only = flashinfer64_core_only
        self.flashinfer64_direct_macro_csr = flashinfer64_direct_macro_csr
        self.attention_debug_dir = attention_debug_dir
        resolved_debug_step = skip_steps if attention_debug_step < 0 else attention_debug_step
        self.attention_debug_steps = (
            (resolved_debug_step,)
            if attention_debug_steps is None
            else tuple(dict.fromkeys(int(step) for step in attention_debug_steps))
        )
        if not self.attention_debug_steps or any(step < 0 for step in self.attention_debug_steps):
            raise ValueError("attention_debug_steps must contain non-negative steps")
        self.attention_debug_step = self.attention_debug_steps[0]
        self.attention_debug_layers = tuple(attention_debug_layers)
        self.attention_debug_stop = bool(attention_debug_stop)

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def _dump_attention_debug(
        self,
        query,
        key,
        value,
        sparse_output,
        dense_output,
        valid_sequence,
    ):
        """Persist exact first-divergence tensors plus compact error metrics."""
        os.makedirs(self.attention_debug_dir, exist_ok=True)
        if self.sparse_execution == "flashinfer64":
            backend = f"flashinfer64_{self.flashinfer64_route_mode}"
        else:
            backend = f"dfsattn_{self.sparse_execution}_{self.selector_mode}"
        stem = f"step{self.step_idx:03d}_layer{self.layer_idx:03d}_{backend}"
        tensor_path = os.path.join(self.attention_debug_dir, stem + ".pt")
        summary_path = os.path.join(self.attention_debug_dir, stem + ".json")

        diff = sparse_output.float() - dense_output.float()
        dense_norm = torch.linalg.vector_norm(dense_output.float()).clamp_min(1e-20)
        per_head_mean = diff.abs().mean(dim=(0, 2, 3))
        per_head_max = diff.abs().amax(dim=(0, 2, 3))
        summary = {
            "step_idx": self.step_idx,
            "layer_idx": self.layer_idx,
            "backend": backend,
            "shape": list(query.shape),
            "dtype": str(query.dtype),
            "video_len": int(self.video_len),
            "valid_sequence": int(valid_sequence),
            "padding_sequence": int(query.shape[2] - valid_sequence),
            "max_abs_error_vs_dense": float(diff.abs().max().item()),
            "mean_abs_error_vs_dense": float(diff.abs().mean().item()),
            "relative_l2_error_vs_dense": float(
                (torch.linalg.vector_norm(diff) / dense_norm).item()
            ),
            "per_head_mean_abs_error": per_head_mean.cpu().tolist(),
            "per_head_max_abs_error": per_head_max.cpu().tolist(),
        }
        if self.sparse_execution == "flashinfer64":
            # Keep the routing configuration next to the tensors so offline
            # macro add-back analysis can reject a non-Core baseline instead
            # of silently measuring a different selector.
            summary.update({
                "flashinfer64_route_mode": self.flashinfer64_route_mode,
                "flashinfer64_tile_top_ratio": float(self.flashinfer64_tile_top_ratio),
                "flashinfer64_fine_top_ratio": float(self.flashinfer64_fine_top_ratio),
                "flashinfer64_fine_top_k": self.flashinfer64_fine_top_k,
                "flashinfer64_token_top_p": float(self.flashinfer64_token_top_p),
                "flashinfer64_core_only": bool(self.flashinfer64_core_only),
                "flashinfer64_promotion_threshold": int(self.flashinfer64_promotion_threshold),
                "flashinfer64_dense_layer": int(self.flashinfer64_dense_layer),
                "flashinfer64_dense_heads": (
                    None if self.flashinfer64_dense_heads is None
                    else [int(head) for head in self.flashinfer64_dense_heads]
                ),
            })
        del diff, dense_norm, per_head_mean, per_head_max

        # CPU tensors make the dump portable and prevent torch.load from
        # allocating GPU memory during the cross-backend comparison.
        payload = {
            "metadata": summary,
            "query": query.detach().cpu(),
            "key": key.detach().cpu(),
            "value": value.detach().cpu(),
            "sparse_output": sparse_output.detach().cpu(),
            "dense_output": dense_output.detach().cpu(),
        }
        temporary_path = tensor_path + ".tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, tensor_path)
        with open(summary_path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        os.replace(summary_path + ".tmp", summary_path)
        logger.info("Attention debug dump saved to {}", tensor_path)
    
    def get_cu_max_seqlen(self, attention_mask, device):
        cu_seqlens_q = torch.tensor([0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.tensor(
            [0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=device
        )
        max_seqlen_q = attention_mask.numel()
        max_seqlen_kv = attention_mask.numel()
        return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

        # 2. QK normalization - use fast kernel if available (skip if mode is "flash")
        if ENABLE_FAST_KERNEL and apply_qk_norm is not None and self.mode != "flash":
            query, key = apply_qk_norm(attn.norm_q, attn.norm_k, query, key)
        else:
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream - use fast kernel if available (skip if mode is "flash")
        if image_rotary_emb is not None:
            if ENABLE_FAST_KERNEL and apply_qk_rope_single is not None and apply_qk_rope_double is not None and self.mode != "flash":
                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    # Single stream: text after video
                    query, key = apply_qk_rope_single(query, key, image_rotary_emb, encoder_hidden_states)
                else:
                    # Double stream: no text
                    query, key = apply_qk_rope_double(query, key, image_rotary_emb)
            else:
                # Fallback to PyTorch implementation
                from diffusers.models.embeddings import apply_rotary_emb

                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    query = torch.cat(
                        [
                            apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                            query[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                    key = torch.cat(
                        [
                            apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                            key[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                else:
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2) # [B, H, L, D]
            key = torch.cat([key, encoder_key], dim=2) # [B, H, L, D]
            value = torch.cat([value, encoder_value], dim=2) # [B, H, L, D]

        # 5. Attention
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = self.get_cu_max_seqlen(attention_mask, query.device)
        ran_sparse_attention = False
        if self.mode == "dfs":
            phase_rest_end = self.skip_steps + self.rest_steps
            phase_sparse2_start = phase_rest_end + self.skip_steps2
            in_rest_phase = self.skip_steps <= self.step_idx < phase_rest_end
            in_sparse2_phase = self.step_idx >= phase_sparse2_start
            in_sparse_phase = in_rest_phase or in_sparse2_phase
            is_periodic_dense = (
                self.dense_interval > 0
                and self.step_idx >= self.skip_steps
                and (self.step_idx - self.skip_steps + 1) % self.dense_interval == 0
            )
            if self.layer_idx not in self.skip_layers and in_sparse_phase and not is_periodic_dense:
                ran_sparse_attention = True
                hidden_states = dfs_attention(
                    query,
                    key,
                    value,
                    step_idx=self.step_idx,
                    skip_steps=self.skip_steps,
                    cache_interval=self.cache_interval,
                    layer_idx=self.layer_idx,
                    sparsity=self.sparsity,
                    sparsity_dcrt=self.sparsity_dcrt,
                    tile_size=self.tile_size,
                    block_size=self.block_size,
                    video_len=self.video_len,
                    video_perm=self.video_perm,
                    cache_flag=self.cache_flag,
                    record_density=self.record_density,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    sparse_execution=self.sparse_execution,
                    hybrid_threshold=self.hybrid_threshold,
                    block_top_p=self.block_top_p,
                    token_top_k=self.token_top_k,
                    residual_candidate_blocks=self.residual_candidate_blocks,
                    selector_mode=self.selector_mode,
                    fine_top_p=self.fine_top_p,
                    flashinfer64_top_p=self.flashinfer64_top_p,
                    flashinfer64_token_top_k=self.flashinfer64_token_top_k,
                    flashinfer64_token_top_ratio=self.flashinfer64_token_top_ratio,
                    flashinfer64_route_mode=self.flashinfer64_route_mode,
                    flashinfer64_tile_top_ratio=self.flashinfer64_tile_top_ratio,
                    flashinfer64_dynamic_tile_ratio=self.flashinfer64_dynamic_tile_ratio,
                    flashinfer64_fine_top_ratio=self.flashinfer64_fine_top_ratio,
                    flashinfer64_fine_top_k=self.flashinfer64_fine_top_k,
                    flashinfer64_token_top_p=self.flashinfer64_token_top_p,
                    flashinfer64_promotion_threshold=self.flashinfer64_promotion_threshold,
                    flashinfer64_dense_layer=self.flashinfer64_dense_layer,
                    flashinfer64_dense_heads=self.flashinfer64_dense_heads,
                    flashinfer64_high_omission_heads_file=self.flashinfer64_high_omission_heads_file,
                    flashinfer64_residual_scorer=self.flashinfer64_residual_scorer,
                    flashinfer64_residual_temperature=self.flashinfer64_residual_temperature,
                    flashinfer64_residual_min_top_k=self.flashinfer64_residual_min_top_k,
                    flashinfer64_residual_max_top_k=self.flashinfer64_residual_max_top_k,
                    flashinfer64_route_cache=self.flashinfer64_route_cache,
                    flashinfer64_core_only=self.flashinfer64_core_only,
                    flashinfer64_direct_macro_csr=self.flashinfer64_direct_macro_csr,
                    flashinfer64_valid_sequence=int(cu_seqlens_q[1].item()),
                )
            else:
                attention_start = DFS_Attention.timing_recorder.start(query.device)
                hidden_states = full_attention(
                    query, key, value, 
                    mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, \
                    max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=query.shape[0])
                DFS_Attention.timing_recorder.stop(
                    attention_start, phase="attention_execution", step_idx=self.step_idx,
                    layer_idx=self.layer_idx, device=query.device,
                )
            
        elif self.mode in ["flash", "torch", "vanilla"]:
            attention_start = DFS_Attention.timing_recorder.start(query.device)
            hidden_states = full_attention(
                    query, key, value, 
                    mode=self.mode, drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, \
                    max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=query.shape[0])
            DFS_Attention.timing_recorder.stop(
                attention_start, phase="attention_execution", step_idx=self.step_idx,
                layer_idx=self.layer_idx, device=query.device,
            )

        debug_this_attention = (
            ran_sparse_attention
            and self.attention_debug_dir is not None
            and self.step_idx in self.attention_debug_steps
            and self.layer_idx in self.attention_debug_layers
        )
        if debug_this_attention:
            # Compute the reference on the exact same normalized/RoPE-applied
            # Q/K/V. This is the first point at which the sparse trajectories
            # can diverge, so no model-level effects are mixed into the error.
            dense_debug_output = full_attention(
                query,
                key,
                value,
                mode="flash",
                drop_rate=0.0,
                attn_mask=attention_mask,
                causal=False,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=query.shape[0],
            )
            self._dump_attention_debug(
                query,
                key,
                value,
                hidden_states,
                dense_debug_output,
                int(cu_seqlens_q[1].item()),
            )
            del dense_debug_output
            if (
                self.attention_debug_stop
                and self.step_idx == max(self.attention_debug_steps)
                and self.layer_idx == max(self.attention_debug_layers)
            ):
                raise AttentionDebugComplete(
                    f"completed attention debug dump at step {self.step_idx}, "
                    f"layer {self.layer_idx}"
                )
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        
        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        self.step_idx += 1
        if self.step_idx == 50:
            self.step_idx = 0

        return hidden_states, encoder_hidden_states
