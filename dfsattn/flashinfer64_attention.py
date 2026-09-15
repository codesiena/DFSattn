"""Hardware-aligned hierarchical sparse attention for HunyuanVideo.

The historical module/backend name is retained for CLI compatibility.  The
implementation is now Q128xK96 FlashInfer Core plus Q16xK16 grouped Triton
Residual, with macro Top-k/Top-p, global complement micro Top-p, occupancy promotion,
exact LSE merge, optional compact-route reuse, and cached per-layer FA3 plans.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from typing import Any, Dict, Optional, Tuple

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
    from flashinfer.page import block_sparse_indices_to_vector_sparse_offsets
    from flashinfer.prefill import get_batch_prefill_module
    from flashinfer.utils import (
        MaskMode,
        PosEncodingMode,
        TensorLayout,
        determine_attention_backend,
        device_support_pdl,
    )
except Exception:  # pragma: no cover - private API varies across FlashInfer releases
    block_sparse_indices_to_vector_sparse_offsets = None
    get_batch_prefill_module = None
    MaskMode = PosEncodingMode = TensorLayout = None
    determine_attention_backend = None
    device_support_pdl = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

try:
    from .rode_backend import load_rode_extension
except ImportError:  # pragma: no cover
    load_rode_extension = None


Q_MACRO = 128
K_MACRO = 96
MICRO = 16
Q_MICROS_PER_MACRO = Q_MACRO // MICRO
K_MICROS_PER_MACRO = K_MACRO // MICRO
BLOCK_SIZE = Q_MACRO  # historical compatibility only
RESIDUAL_BUCKET_CAPS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
OCCUPANCY_TAU_SWEEP = (8, 16, 24, 32, 40, 48)

# The compatibility VariableBlock path keeps one model-wide ephemeral wrapper.
# The direct FA3 path instead stores one compact scheduler plan per layer so a
# route cached for 12 denoising steps also avoids 11 repeated plan calls.
_shared_core_wrapper: Any = None
_shared_core_workspace: Optional[torch.Tensor] = None
_shared_core_signature = None
_shared_direct_vector_indices: Optional[torch.Tensor] = None
_shared_direct_pin_workspace: Optional[torch.Tensor] = None
_replay_mask_cache: dict[str, tuple[float, dict[str, list[list[int]]]]] = {}
_high_omission_heads_cache: dict[
    str, tuple[tuple[int, int], dict[int, tuple[int, ...]]]
] = {}

_HIGH_OMISSION_HEAD_LINE = re.compile(
    r"^\s*Layer\s*(\d+)\s*/\s*Head\s*(\d+)\s*(?:#.*)?$",
    re.IGNORECASE,
)


def _default_high_omission_heads_file() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "importanthead",
            "high_omission_heads.txt",
        )
    )


def _load_high_omission_heads(path: Optional[str] = None):
    """Load the editable Layer/Head risk prior used by hierarchical routes.

    The file intentionally uses the same 0-based Layer/Head numbering as the
    model code.  It is re-read when its mtime/size changes so an experiment
    can update the prior without restarting the Python process.
    """
    selected_path = path or os.environ.get("FLASHINFER64_HIGH_OMISSION_HEADS_FILE")
    selected_path = os.path.abspath(selected_path or _default_high_omission_heads_file())
    try:
        stat = os.stat(selected_path)
    except OSError as exc:
        raise FileNotFoundError(
            "high-omission head file not found: " + selected_path
        ) from exc
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _high_omission_heads_cache.get(selected_path)
    if cached is not None and cached[0] == fingerprint:
        return cached[1], (selected_path, *fingerprint)

    by_layer: dict[int, set[int]] = {}
    with open(selected_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            match = _HIGH_OMISSION_HEAD_LINE.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"invalid high-omission head entry at {selected_path}:{line_no}; "
                    "expected 'Layer<integer> / Head<integer>'"
                )
            layer, head = (int(value) for value in match.groups())
            if layer < 0 or head < 0:
                raise ValueError(
                    f"Layer/Head indices must be non-negative at "
                    f"{selected_path}:{line_no}"
                )
            by_layer.setdefault(layer, set()).add(head)
    normalized = {
        layer: tuple(sorted(heads)) for layer, heads in sorted(by_layer.items())
    }
    _high_omission_heads_cache[selected_path] = (fingerprint, normalized)
    return normalized, (selected_path, *fingerprint)


def _load_replay_mask(path: str) -> dict[str, list[list[int]]]:
    """Load a prompt-specific Q16/K16 replay mask once per process."""
    if not path:
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise FileNotFoundError(
            f"FLASHINFER64_REPLAY_MASK_FILE not found: {path}"
        ) from exc
    cached = _replay_mask_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", payload)
    if not isinstance(rows, dict):
        raise ValueError(f"Replay mask must be a JSON object: {path}")
    normalized: dict[str, list[list[int]]] = {}
    for key, values in rows.items():
        if not isinstance(values, list):
            raise ValueError(f"Replay mask rows for {key} must be a list: {path}")
        normalized[str(key)] = [
            [int(item[0]), int(item[1]), int(item[2])]
            for item in values
            if isinstance(item, (list, tuple)) and len(item) == 3
        ]
    _replay_mask_cache[path] = (mtime, normalized)
    return normalized


def _replay_residual_mask(
    micro_scores: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    replay_file: str,
    step_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    """Replay the latest available anchor mask and remove current-Core overlap.

    Replay masks are measured only at sparse anchor steps (currently 12/24/36),
    while the no-cache main route is rebuilt from the current Q/K on every
    diffusion step.  Holding only the external add-back coordinates until the
    next anchor keeps that main routing policy unchanged.
    """
    mask = torch.zeros_like(micro_scores, dtype=torch.bool)
    replay_rows = _load_replay_mask(replay_file)
    rows = replay_rows.get(f"{step_idx}:{layer_idx}")
    if rows is None:
        anchors = []
        for key, candidate_rows in replay_rows.items():
            try:
                anchor_step_text, anchor_layer_text = key.split(":", 1)
                anchor_step = int(anchor_step_text)
                anchor_layer = int(anchor_layer_text)
            except (TypeError, ValueError):
                continue
            if anchor_layer == layer_idx and anchor_step <= step_idx:
                anchors.append((anchor_step, candidate_rows))
        rows = max(anchors, key=lambda item: item[0])[1] if anchors else ()
    heads, q_blocks, k_blocks = micro_scores.shape
    for head, q16, k16 in rows:
        if 0 <= head < heads and 0 <= q16 < q_blocks and k16 == -1:
            # A negative K16 sentinel denotes full-row add-back.  The current
            # Core mask is removed below, so this remains a residual-only
            # intervention and never changes the main Core selector.
            mask[head, q16, :] = True
        elif 0 <= head < heads and 0 <= q16 < q_blocks and 0 <= k16 < k_blocks:
            mask[head, q16, k16] = True
    q_parent = torch.arange(q_blocks, device=core_mask.device) // Q_MICROS_PER_MACRO
    k_parent = torch.arange(k_blocks, device=core_mask.device) // K_MICROS_PER_MACRO
    selected_core = core_mask[:, q_parent[:, None], k_parent[None, :]]
    return mask & ~selected_core


@dataclass
class FlashInfer64Route:
    core_mask: torch.Tensor          # [H, ceil(S/128), ceil(S/96)]
    core_execution_mask: torch.Tensor  # includes anchors for empty Q128 rows
    core_empty_qblocks: torch.Tensor  # bool [H, ceil(S/128)]
    residual_mask: Optional[torch.Tensor]  # CPU reference only; never retained on CUDA
    residual_indices: torch.Tensor   # CSR column indices [nnz]
    residual_indptr: torch.Tensor    # CSR row pointers [H*ceil(S/16)+1]
    residual_buckets: Tuple[Tuple[int, torch.Tensor], ...]  # (capacity, CSR row ids)
    residual_active_rows: torch.Tensor  # flattened (head, Q16) CSR rows with non-zero residual
    sequence: int
    video_len: int
    route_mode: str
    top_p: float
    tile_top_ratio: float
    token_top_p: float
    promotion_threshold: int
    core_only: bool
    promoted_tiles: int
    core_interactions: int
    residual_interactions: int
    residual_micro_tiles: int
    residual_count_mean: float
    residual_count_p50: float
    residual_count_p95: float
    residual_count_max: float
    residual_count_nonempty_ratio: float
    fine_top_k: int
    fine_top_ratio: float
    occupancy_threshold: int
    macro_occupancy: torch.Tensor  # uint8, [H, ceil(S/128), ceil(S/96)]
    occupancy_histogram: Tuple[int, ...]
    head_occupancy_stats: Tuple[Dict[str, Any], ...]
    head_tau_stats: Tuple[Dict[str, Any], ...]


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


def _block_representatives(x: torch.Tensor, block: int = MICRO) -> torch.Tensor:
    """Return mean + three largest-deviation representatives per block."""
    heads, sequence, dim = x.shape
    padded = _ceil_div(sequence, block) * block
    if padded != sequence:
        x = F.pad(x, (0, 0, 0, padded - sequence))
    grouped = x.view(heads, padded // block, block, dim)
    valid = (
        torch.arange(padded, device=x.device) < sequence
    ).view(1, padded // block, block)
    valid_float = valid.unsqueeze(-1).to(x.dtype)
    means = (grouped * valid_float).sum(2) / valid_float.sum(2).clamp_min(1).to(x.dtype)
    deviation = (grouped.float() - means.float().unsqueeze(2)).square().sum(-1)
    deviation = deviation.masked_fill(~valid, float("-inf"))
    representative_count = min(3, block)
    top_indices = deviation.topk(representative_count, dim=2, largest=True).indices
    gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    representatives = torch.gather(grouped, 2, gather_indices)
    selected_valid = torch.gather(
        valid.expand(heads, -1, -1), 2, top_indices
    ).unsqueeze(-1)
    representatives = torch.where(
        selected_valid, representatives, means.unsqueeze(2).expand_as(representatives)
    )
    return torch.cat((means.unsqueeze(2), representatives), dim=2)


def compute_peak_aware_micro_tile_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    temperature: float = 1.0,
    q_chunk_size: int = 128,
) -> torch.Tensor:
    """Sampled-LSE Q16/K16 probability mass from the method specification.

    Each block contributes its mean and the three tokens furthest from that
    mean.  The 16 representative pair logits are reduced with log-mean-exp,
    and the result is softmaxed over K16 blocks.  Query blocks are processed
    in chunks so the temporary ``[heads, q_chunk, k16, 4, 4]`` tensor is not
    materialized for the whole sequence at once.
    """
    if q.ndim != 3 or k.ndim != 3 or q.shape != k.shape:
        raise ValueError("q and k must match [heads, sequence, head_dim]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if q_chunk_size < 1:
        raise ValueError("q_chunk_size must be positive")
    q_representatives = _block_representatives(q)
    k_representatives = _block_representatives(k)
    heads, q_blocks, representative_count, dim = q_representatives.shape
    k_blocks = k_representatives.shape[1]
    scale = dim ** -0.5
    log_representative_count = math.log(representative_count * representative_count)
    scores = []
    k_representatives = k_representatives.float()
    for q_start in range(0, q_blocks, q_chunk_size):
        q_chunk = q_representatives[:, q_start:q_start + q_chunk_size].float()
        logits = torch.einsum(
            "hqad,hkbd->hqkab", q_chunk, k_representatives
        ) * scale
        logits = logits.reshape(
            heads, q_chunk.shape[1], k_blocks, representative_count * representative_count
        )
        block_scores = torch.logsumexp(logits, dim=-1) - log_representative_count
        scores.append(torch.softmax(block_scores / temperature, dim=-1))
    return torch.cat(scores, dim=1)


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


def select_fine_topk_from_scores(
    micro_scores: torch.Tensor, *, top_k: Optional[int] = None,
    top_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact Q16/K16 Top-k indices and their valid-slot mask.

    The second tensor is needed for short video/text tails where the requested
    ``top_k`` can be larger than the number of video K16 tiles.  Keeping the
    selection compact is also what lets the CUDA route avoid materializing a
    complete ``[head, Q16, K16]`` bool mask.
    """
    if micro_scores.ndim != 3:
        raise ValueError("micro_scores must have shape [heads, q16, k16]")
    k_micro = micro_scores.shape[-1]
    if top_ratio is not None:
        if not 0.0 < top_ratio <= 1.0:
            raise ValueError(f"fine top_ratio must be in (0, 1], got {top_ratio}")
        ratio_top_k = max(1, math.ceil(k_micro * top_ratio))
        if top_k is not None and top_k != ratio_top_k:
            raise ValueError("top_k and top_ratio resolve to different selections")
        top_k = ratio_top_k
    if top_k is None:
        raise ValueError("one of top_k or top_ratio must be provided")
    if not 1 <= top_k <= k_micro:
        raise ValueError(f"fine top_k must be in [1, {k_micro}], got {top_k}")
    values, indices = torch.topk(
        micro_scores, top_k, dim=-1, largest=True, sorted=True
    )
    return indices.to(torch.int32).contiguous(), values > 0


def _fine_topk_occupancy_route(
    micro_scores: torch.Tensor,
    *,
    fine_top_k: int,
    fine_top_ratio: Optional[float] = None,
    occupancy_threshold: int,
    sequence: int,
    video_len: Optional[int],
    fine_indices: Optional[torch.Tensor] = None,
    fine_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the new Fine Top-k -> uint8 occupancy -> Core/Residual route."""
    if not 1 <= occupancy_threshold <= Q_MICROS_PER_MACRO * K_MICROS_PER_MACRO:
        raise ValueError(
            "occupancy threshold must be in [1, 48], "
            f"got {occupancy_threshold}"
        )
    if fine_indices is None or fine_valid is None:
        if fine_top_k is not None:
            fine_indices, fine_valid = select_fine_topk_from_scores(
                micro_scores, top_k=fine_top_k
            )
        else:
            fine_indices, fine_valid = select_fine_topk_from_scores(
                micro_scores, top_ratio=fine_top_ratio
            )
    heads, q_micro, k_micro = micro_scores.shape
    q_blocks = _ceil_div(sequence, Q_MACRO)
    k_blocks = _ceil_div(sequence, K_MACRO)
    q_parent = torch.arange(q_micro, device=micro_scores.device) // Q_MICROS_PER_MACRO
    k_parent = fine_indices // K_MICROS_PER_MACRO

    # Only fully-video macro regions participate in Fine Top-k.  The existing
    # HunyuanVideo policy makes the macro containing the text boundary dense.
    if video_len is None or video_len >= sequence:
        video_q_blocks, video_k_blocks = q_blocks, k_blocks
    else:
        video_len = max(0, video_len)
        video_q_blocks = min(q_blocks, video_len // Q_MACRO)
        video_k_blocks = min(k_blocks, video_len // K_MACRO)
    route_q_valid = q_parent[:, None] < video_q_blocks
    route_k_valid = k_parent < video_k_blocks
    fine_valid = fine_valid & route_q_valid[None, :, :] & route_k_valid

    # uint8 is sufficient because one macro has at most 8*6=48 selected
    # microtiles.  scatter_add is performed directly into the compact macro
    # occupancy buffer; no full fine-grained bool mask is retained.
    occupancy = torch.zeros(
        (heads, q_blocks, k_blocks), dtype=torch.uint8, device=micro_scores.device
    )
    macro_linear = (
        q_parent[None, :, None] * k_blocks + k_parent
    ).expand(heads, -1, -1)
    occupancy_flat = occupancy.view(heads, -1)
    occupancy_flat.scatter_add_(
        1,
        macro_linear.reshape(heads, -1).long(),
        fine_valid.to(torch.uint8).reshape(heads, -1),
    )

    core = occupancy >= occupancy_threshold
    if video_len is not None and video_len < sequence:
        q_dense_start = min(q_blocks, video_len // Q_MACRO)
        k_dense_start = min(k_blocks, video_len // K_MACRO)
        core[:, q_dense_start:, :] = True
        core[:, :, k_dense_start:] = True

    selected_macro = core[:, q_parent, :].gather(2, k_parent.long())
    residual_valid = fine_valid & ~selected_macro
    return core, fine_indices, residual_valid, occupancy


def _build_residual_csr_from_fine_indices(
    fine_indices: torch.Tensor,
    fine_valid: torch.Tensor,
    *,
    q_micro_blocks: int,
    k_micro_blocks: int,
    core_mask: torch.Tensor,
    build_mask: bool,
):
    """Pack compact Fine Top-k output into the existing Residual CSR format."""
    heads = fine_indices.shape[0]
    q_parent = torch.arange(q_micro_blocks, device=fine_indices.device) // Q_MICROS_PER_MACRO
    k_parent = fine_indices // K_MICROS_PER_MACRO
    selected_macro = core_mask[:, q_parent, :].gather(2, k_parent.long())
    keep = fine_valid & ~selected_macro
    flat_keep = keep.reshape(heads * q_micro_blocks, -1)
    counts = flat_keep.sum(-1, dtype=torch.int32)
    indptr = torch.empty(counts.numel() + 1, dtype=torch.int32, device=fine_indices.device)
    indptr[0] = 0
    torch.cumsum(counts, dim=0, out=indptr[1:])
    indices = fine_indices.reshape(heads * q_micro_blocks, -1).masked_select(flat_keep).contiguous()

    caps = list(RESIDUAL_BUCKET_CAPS)
    while caps[-1] < k_micro_blocks:
        caps.append(caps[-1] * 2)
    all_rows = torch.arange(counts.numel(), dtype=torch.int32, device=fine_indices.device)
    buckets = []
    lower = 0
    for cap in caps:
        row_ids = all_rows.masked_select((counts > lower) & (counts <= cap))
        if row_ids.numel():
            buckets.append((cap, row_ids.contiguous()))
        lower = cap

    if counts.numel():
        count_float = counts.float()
        p50, p95 = torch.quantile(
            count_float, torch.tensor((0.50, 0.95), device=fine_indices.device)
        ).tolist()
        stats = (
            float(count_float.mean().item()), float(p50), float(p95),
            float(counts.max().item()), float((counts > 0).float().mean().item()),
        )
    else:
        stats = (0.0, 0.0, 0.0, 0.0, 0.0)

    residual_mask = None
    if build_mask:
        residual_mask = torch.zeros(
            (heads, q_micro_blocks, k_micro_blocks),
            dtype=torch.bool, device=fine_indices.device,
        )
        # The compact indices are row-concatenated, so fill the reference mask
        # row by row only on the CPU/reference path.
        flat_mask = residual_mask.view(heads * q_micro_blocks, k_micro_blocks)
        for row in range(flat_mask.shape[0]):
            start, end = indptr[row].item(), indptr[row + 1].item()
            if end > start:
                flat_mask[row, indices[start:end].long()] = True
    active_rows = torch.cat(tuple(rows for _, rows in buckets), dim=0) if buckets else torch.empty(
        0, dtype=torch.int32, device=fine_indices.device
    )
    return residual_mask, indices, indptr, tuple(buckets), active_rows, stats


def _fine_route_valid_mask(
    fine_indices: torch.Tensor,
    fine_valid: torch.Tensor,
    *,
    q_micro_blocks: int,
    sequence: int,
    video_len: Optional[int],
) -> torch.Tensor:
    """Keep Fine Top-k slots belonging to fully-video macro regions."""
    q_parent = torch.arange(
        q_micro_blocks, device=fine_indices.device
    ) // Q_MICROS_PER_MACRO
    k_parent = fine_indices // K_MICROS_PER_MACRO
    q_blocks = _ceil_div(sequence, Q_MACRO)
    k_blocks = _ceil_div(sequence, K_MACRO)
    if video_len is None or video_len >= sequence:
        video_q_blocks, video_k_blocks = q_blocks, k_blocks
    else:
        video_q_blocks = min(q_blocks, max(0, video_len) // Q_MACRO)
        video_k_blocks = min(k_blocks, max(0, video_len) // K_MACRO)
    return fine_valid & (
        (q_parent[None, :, None] < video_q_blocks)
        & (k_parent < video_k_blocks)
    )


def _compute_head_occupancy_tau_stats(
    micro_scores: torch.Tensor,
    fine_indices: torch.Tensor,
    fine_valid: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    sequence: int,
    video_len: Optional[int],
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    """Analyze all tau values from one unchanged Fine Top-k support.

    This is intentionally statistics-only.  It builds no FlashInfer plan and
    launches no Core/Residual kernel, so tau sweeps do not add runtime to the
    actual attention path.
    """
    heads, q_micro, k_micro = micro_scores.shape
    q_blocks, k_blocks = occupancy.shape[1:]
    q_sizes = torch.full((q_blocks,), Q_MACRO, dtype=torch.float32, device=occupancy.device)
    k_sizes = torch.full((k_blocks,), K_MACRO, dtype=torch.float32, device=occupancy.device)
    q_sizes[-1] = sequence - (q_blocks - 1) * Q_MACRO
    k_sizes[-1] = sequence - (k_blocks - 1) * K_MACRO
    q16_sizes = torch.minimum(
        torch.full((q_micro,), MICRO, dtype=torch.float32, device=occupancy.device),
        torch.as_tensor(sequence, device=occupancy.device, dtype=torch.float32)
        - torch.arange(q_micro, device=occupancy.device, dtype=torch.float32) * MICRO,
    )
    k16_ids = fine_indices.long()
    k16_sizes = torch.minimum(
        torch.full_like(k16_ids, MICRO, dtype=torch.float32),
        torch.as_tensor(sequence, device=occupancy.device, dtype=torch.float32)
        - k16_ids.to(torch.float32) * MICRO,
    )
    route_valid = _fine_route_valid_mask(
        fine_indices, fine_valid, q_micro_blocks=q_micro,
        sequence=sequence, video_len=video_len,
    )
    q_parent = torch.arange(q_micro, device=occupancy.device) // Q_MICROS_PER_MACRO
    k_parent = fine_indices.long() // K_MICROS_PER_MACRO
    entropy_rows = micro_scores
    q_valid = q_parent < (q_blocks if video_len is None or video_len >= sequence else video_len // Q_MACRO)
    entropy = -(entropy_rows.clamp_min(torch.finfo(entropy_rows.dtype).tiny).log() * entropy_rows).sum(-1)
    entropy = entropy[:, q_valid].mean(-1)

    occupancy_long = occupancy.long()
    head_occupancy = []
    for head in range(heads):
        values = occupancy_long[head].reshape(-1)
        histogram = torch.bincount(values, minlength=49).tolist()
        nonzero = values[values > 0].float()
        quantiles = torch.quantile(values.float(), torch.tensor((0.50, 0.95), device=values.device)).tolist()
        weighted = (
            (values.float() * values.float()).sum() / values.float().sum().clamp_min(1.0)
        )
        head_occupancy.append({
            "head": head,
            "occupancy_histogram": tuple(int(x) for x in histogram),
            "mean": float(values.float().mean().item()),
            "p50": float(quantiles[0]),
            "p95": float(quantiles[1]),
            "max": float(values.max().item()),
            "weighted_occupancy": float(weighted.item()),
            "fine_entropy_mean": float(entropy[head].item()),
            "macro_count": int(values.numel()),
        })

    head_tau = []
    total_possible = float(sequence * sequence)
    for tau in OCCUPANCY_TAU_SWEEP:
        core = occupancy >= tau
        if video_len is not None and video_len < sequence:
            q_dense_start = min(q_blocks, max(0, video_len) // Q_MACRO)
            k_dense_start = min(k_blocks, max(0, video_len) // K_MACRO)
            core = core.clone()
            core[:, q_dense_start:, :] = True
            core[:, :, k_dense_start:] = True
        selected_core = core[:, q_parent, :].gather(2, k_parent)
        residual = route_valid & ~selected_core
        residual_row_counts = residual.sum(-1).float()
        residual_qk = (
            residual.float()
            * q16_sizes[None, :, None]
            * k16_sizes
        ).sum((1, 2))
        core_qk = (
            core.float()
            * q_sizes[None, :, None]
            * k_sizes[None, None, :]
        ).sum((1, 2))
        p50_p95 = torch.quantile(
            residual_row_counts,
            torch.tensor((0.50, 0.95), device=occupancy.device), dim=1,
        ).transpose(0, 1)
        for head in range(heads):
            head_tau.append({
                "head": head,
                "tau": tau,
                "promoted_macro_count": int((occupancy[head] >= tau).sum().item()),
                "core_density": float((core_qk[head] / total_possible).item()),
                "residual_density": float((residual_qk[head] / total_possible).item()),
                "residual_microtiles": int(residual[head].sum().item()),
                "residual_row_mean": float(residual_row_counts[head].mean().item()),
                "residual_row_p50": float(p50_p95[head, 0].item()),
                "residual_row_p95": float(p50_p95[head, 1].item()),
                "residual_row_max": float(residual_row_counts[head].max().item()),
                "residual_row_nonempty_ratio": float((residual_row_counts[head] > 0).float().mean().item()),
            })
    return tuple(head_occupancy), tuple(head_tau)


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


def _force_dense_heads(core_mask: torch.Tensor, dense_heads) -> None:
    """Make selected heads attend every valid macro K block.

    This is deliberately applied only to the legacy ``topp_topk`` route by
    the caller.  Keeping the override at macro-mask level preserves the
    existing FlashInfer execution path while making the selected heads
    functionally dense (all Q128 x K96 tiles are present and no Residual
    complement remains).
    """
    if dense_heads is None:
        core_mask[...] = True
        return
    if not dense_heads:
        return
    heads = core_mask.shape[0]
    indices = tuple(sorted(set(int(head) for head in dense_heads)))
    if any(head < 0 or head >= heads for head in indices):
        raise ValueError(
            f"dense head indices {indices} are outside the available range [0, {heads})"
        )
    core_mask[list(indices)] = True


def select_64_tiles(q, k, *, top_p):
    return select_64_tiles_from_scores(compute_64_tile_scores(q, k), top_p=top_p)


def _select_residual_to_total_mass(
    micro_scores,
    core_mask,
    *,
    total_top_p,
    head_mask: Optional[torch.Tensor] = None,
    min_top_k: int = 0,
    max_top_k: Optional[int] = None,
    score_overrides: Optional[Dict[int, torch.Tensor]] = None,
):
    """Select rejected microtiles until Core+Residual reaches total Top-p.

    ``micro_scores`` is already normalized over all K16 tiles for each
    (head,Q16).  The complement is deliberately *not* renormalized: Core mass
    is subtracted from the requested total mass, and raw rejected mass fills
    only the remainder.  ``head_mask`` and the optional bounds let the
    ``topp_topk`` route spend this extra check only on configured risk heads.
    """
    if not 0.0 <= total_top_p <= 1.0:
        raise ValueError("total_top_p must be in [0, 1]")
    if min_top_k < 0 or (max_top_k is not None and max_top_k < 0):
        raise ValueError("residual Top-k bounds must be non-negative")
    if max_top_k is not None and min_top_k > max_top_k:
        raise ValueError("residual min_top_k cannot exceed max_top_k")
    _, q_micro, k_micro = micro_scores.shape
    if head_mask is not None:
        if head_mask.ndim != 1 or head_mask.shape[0] != micro_scores.shape[0]:
            raise ValueError("head_mask must have shape [heads]")
        selected_heads = torch.nonzero(head_mask, as_tuple=False).flatten().tolist()
        residual = torch.zeros_like(micro_scores, dtype=torch.bool)
        for head in selected_heads:
            head_scores = micro_scores[head:head + 1]
            if score_overrides is not None and head in score_overrides:
                head_scores = score_overrides[head]
                if head_scores.shape != micro_scores[head:head + 1].shape:
                    raise ValueError(
                        f"score override for head {head} has shape "
                        f"{tuple(head_scores.shape)}, expected "
                        f"{tuple(micro_scores[head:head + 1].shape)}"
                    )
            residual[head:head + 1] = _select_residual_to_total_mass(
                head_scores,
                core_mask[head:head + 1],
                total_top_p=total_top_p,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
            )
        return residual
    q_parent = torch.arange(q_micro, device=micro_scores.device) // Q_MICROS_PER_MACRO
    k_parent = torch.arange(k_micro, device=micro_scores.device) // K_MICROS_PER_MACRO
    covered_by_core = core_mask[:, q_parent[:, None], k_parent[None, :]]
    core_mass = micro_scores.masked_fill(~covered_by_core, 0.0).sum(-1)
    required_mass = (total_top_p - core_mass).clamp_min(0.0)
    rejected_mass = micro_scores.masked_fill(covered_by_core, 0.0)
    values, indices = torch.sort(rejected_mass, dim=-1, descending=True, stable=True)
    keep_by_mass = (
        ((torch.cumsum(values, -1) - values) < required_mass[..., None])
        & (required_mass[..., None] > 0)
        & (values > 0)
    )
    if min_top_k or max_top_k is not None:
        selected_count = keep_by_mass.sum(-1)
        target_count = selected_count.clamp_min(min_top_k)
        if max_top_k is not None:
            target_count = target_count.clamp_max(max_top_k)
        rank = torch.arange(k_micro, device=micro_scores.device)
        keep = (rank < target_count[..., None]) & (values > 0)
    else:
        keep = keep_by_mass
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
    if residual_mask is None:
        return core_n, 0, 0
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


def _count_residual_csr_interactions(
    residual_indices: torch.Tensor, residual_indptr: torch.Tensor,
    sequence: int, q_micro_blocks: int,
) -> Tuple[int, int]:
    """Count exact residual token interactions without rebuilding a bool mask."""
    counts = residual_indptr[1:] - residual_indptr[:-1]
    if residual_indices.numel() == 0:
        return 0, 0
    row_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=residual_indices.device), counts.long()
    )
    q_ids = row_ids % q_micro_blocks
    q_sizes = torch.minimum(
        torch.full_like(q_ids, MICRO),
        torch.as_tensor(sequence, device=q_ids.device) - q_ids * MICRO,
    )
    k_sizes = torch.minimum(
        torch.full_like(residual_indices, MICRO),
        torch.as_tensor(sequence, device=residual_indices.device) - residual_indices * MICRO,
    )
    interactions = (q_sizes * k_sizes).sum()
    return int(interactions.item()), int(residual_indices.numel())


def _make_core_execution_mask(
    core_mask: torch.Tensor, occupancy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Give FlashInfer one harmless anchor for a Q128 row with no Core tile.

    Fine Top-k with a high occupancy threshold can legitimately leave a whole
    Q128 row to Residual.  FlashInfer's paged sparse ABI expects a non-empty
    KV range per query row, so it gets an anchor tile which is discarded from
    the logical Core state immediately after the Core run.
    """
    empty = ~core_mask.any(dim=-1)
    if not bool(empty.any().item()):
        return core_mask, empty
    execution = core_mask.clone()
    anchor = occupancy.to(torch.int16).argmax(dim=-1)
    execution[empty] = F.one_hot(
        anchor[empty].long(), num_classes=core_mask.shape[-1]
    ).to(torch.bool)
    return execution, empty


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
    changed = False
    if vector_indices.numel() <= indices.numel():
        vector_indices = torch.empty(indices.numel() + 1, dtype=torch.int32, device=indices.device)
        changed = True
    if vector_indptr.numel() <= indptr.numel():
        vector_indptr = torch.empty(indptr.numel() + 1, dtype=torch.int32, device=indptr.device)
        changed = True
    # reset_workspace_buffer also allocates a new pinned 8MB host buffer.  Do
    # not pay that cost on every layer-call when the existing buffers fit.
    if changed:
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
    vector_indices = getattr(wrapper, "_vector_sparse_indices_buffer", None)
    vector_indptr = getattr(wrapper, "_vector_sparse_indptr_buffer", None)
    changed = False
    if vector_indices is None or vector_indices.numel() < required_indices:
        vector_indices = torch.empty(
            required_indices, dtype=torch.int32, device=core_mask.device
        )
        changed = True
    if vector_indptr is None or vector_indptr.numel() < required_indptr:
        vector_indptr = torch.empty(
            required_indptr, dtype=torch.int32, device=core_mask.device
        )
        changed = True
    # The first compact plan intentionally replaces FlashInfer's 512MB default
    # vector buffer.  Later plans reuse the largest capacity seen so far.
    if changed or vector_indices.numel() >= 128 * 1024 * 1024:
        if vector_indices.numel() >= 128 * 1024 * 1024:
            vector_indices = torch.empty(
                required_indices, dtype=torch.int32, device=core_mask.device
            )
        wrapper.reset_workspace_buffer(
            float_workspace_buffer=wrapper._float_workspace_buffer,
            int_workspace_buffer=wrapper._int_workspace_buffer,
            vector_sparse_indices_buffer=vector_indices,
            vector_sparse_indptr_buffer=vector_indptr,
        )


def _direct_macro_csr_available(q: torch.Tensor) -> bool:
    """Whether the installed FlashInfer can run the SM90 macro-CSR fast path."""
    if not q.is_cuda or any(
        item is None
        for item in (
            block_sparse_indices_to_vector_sparse_offsets,
            get_batch_prefill_module,
            determine_attention_backend,
        )
    ):
        return False
    try:
        backend = determine_attention_backend(
            q.device,
            PosEncodingMode.NONE.value,
            False,  # use_fp16_qk_reduction
            False,  # use_custom_mask
            q.dtype,
            q.dtype,
        )
        return backend == "fa3"
    except Exception:
        # Private FlashInfer APIs have changed across releases.  Falling back
        # to VariableBlock is slower but preserves compatibility.
        return False


def _ensure_shared_direct_vector_workspace(required: int, device: torch.device) -> torch.Tensor:
    global _shared_direct_vector_indices
    # Hopper's sparse gather may issue a full CTA_KV access at the end.  Keep
    # one extra native K96 tile, even though predicates discard the tail.
    required = required + K_MACRO
    if (
        _shared_direct_vector_indices is None
        or _shared_direct_vector_indices.device != device
        or _shared_direct_vector_indices.numel() < required
    ):
        _shared_direct_vector_indices = torch.empty(
            required, dtype=torch.int32, device=device
        )
    return _shared_direct_vector_indices


class _DirectMacroCSRPlan:
    """Cached per-layer FA3 plan backed by compact K96 macro CSR.

    VariableBlockSparseAttentionWrapper expands every selected macro into 96
    token indices and rebuilds the scheduler on every call.  This plan keeps
    only K96 base offsets.  Expansion is a small CUDA kernel at ``run`` time,
    while the expensive SM90 scheduler plan is rebuilt only when the route is
    refreshed.
    """

    def __init__(self, float_workspace: torch.Tensor, q: torch.Tensor):
        global _shared_direct_pin_workspace
        plan_workspace_mb = int(os.environ.get("FLASHINFER64_PLAN_WORKSPACE_MB", "8"))
        if plan_workspace_mb < 1:
            raise ValueError("FLASHINFER64_PLAN_WORKSPACE_MB must be at least 1")
        self.float_workspace = float_workspace
        self.int_workspace = torch.empty(
            plan_workspace_mb * 1024 * 1024, dtype=torch.uint8, device=q.device
        )
        if (
            _shared_direct_pin_workspace is None
            or _shared_direct_pin_workspace.numel() < self.int_workspace.numel()
        ):
            _shared_direct_pin_workspace = torch.empty(
                self.int_workspace.numel(), dtype=torch.uint8,
                device="cpu", pin_memory=True,
            )
        self.pin_workspace = _shared_direct_pin_workspace
        self.route_identity: Any = None
        self.module = None
        self.plan_info = None

    def plan(self, q: torch.Tensor, core_mask: torch.Tensor, route_identity: Any) -> None:
        heads, sequence, dim = q.shape
        q_blocks, k_blocks = core_mask.shape[1:]

        counts = core_mask.sum(-1, dtype=torch.int32).reshape(-1)
        macro_indptr = torch.empty(
            counts.numel() + 1, dtype=torch.int32, device=q.device
        )
        macro_indptr[0] = 0
        torch.cumsum(counts, 0, out=macro_indptr[1:])
        selected = core_mask.nonzero(as_tuple=False)
        # Store token-base offsets rather than token-expanded indices.  Rows are
        # ordered (head, Q128, K96), so a partial final K96 block is always the
        # final item of that CSR row and fixed-96 expansion remains valid.
        macro_bases = (
            selected[:, 0].to(torch.int64) * sequence
            + selected[:, 2].to(torch.int64) * K_MACRO
        ).to(torch.int32).contiguous()

        col_sizes = torch.full(
            (k_blocks,), K_MACRO, dtype=torch.int32, device=q.device
        )
        col_sizes[-1] = sequence - (k_blocks - 1) * K_MACRO
        kv_lens = (
            core_mask.to(torch.int32) * col_sizes[None, None, :]
        ).sum(-1, dtype=torch.int32).reshape(-1).contiguous()
        vector_indptr = torch.empty(
            kv_lens.numel() + 1, dtype=torch.int32, device=q.device
        )
        vector_indptr[0] = 0
        torch.cumsum(kv_lens, 0, out=vector_indptr[1:])

        q_sizes = torch.full(
            (heads, q_blocks), Q_MACRO, dtype=torch.int32, device=q.device
        )
        q_sizes[:, -1] = sequence - (q_blocks - 1) * Q_MACRO
        qo_indptr = torch.empty(
            q_sizes.numel() + 1, dtype=torch.int32, device=q.device
        )
        qo_indptr[0] = 0
        torch.cumsum(q_sizes.reshape(-1), 0, out=qo_indptr[1:])
        last_page_len = torch.ones_like(kv_lens)

        qo_indptr_host = qo_indptr.to("cpu")
        vector_indptr_host = vector_indptr.to("cpu")
        kv_lens_host = kv_lens.to("cpu")
        required_vector = int(vector_indptr_host[-1].item())
        _ensure_shared_direct_vector_workspace(required_vector, q.device)

        self.module = get_batch_prefill_module(
            "fa3", q.dtype, q.dtype, q.dtype, torch.int32,
            dim, dim, PosEncodingMode.NONE.value,
            False,  # use_sliding_window
            False,  # use_logits_soft_cap
            False,  # use_fp16_qk_reduction
        )
        self.plan_info = self.module.plan(
            self.float_workspace,
            self.int_workspace,
            self.pin_workspace,
            qo_indptr_host,
            vector_indptr_host,
            kv_lens_host,
            heads * sequence,
            heads * q_blocks,
            1,  # each logical head is represented as one independent batch
            1,
            1,  # vector-sparse page size
            False,
            dim,
            dim,
            False,
        )
        self.qo_indptr = qo_indptr
        self.macro_indptr = macro_indptr
        self.macro_bases = macro_bases
        self.vector_indptr = vector_indptr
        self.kv_lens = kv_lens
        self.last_page_len = last_page_len
        self.required_vector = required_vector
        self.route_identity = route_identity

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, return_lse: bool,
        recorder=None, step=-1, layer=-1,
    ):
        if self.module is None or self.plan_info is None:
            raise RuntimeError("direct macro-CSR plan has not been initialized")
        heads, sequence, dim = q.shape
        vector_indices = _ensure_shared_direct_vector_workspace(
            self.required_vector, q.device
        )
        expand_start = _timing_start(recorder, q.device)
        block_sparse_indices_to_vector_sparse_offsets(
            self.macro_bases,
            self.macro_indptr,
            vector_indices,
            self.vector_indptr,
            self.kv_lens,
            1,  # macro_bases already contain absolute token offsets
            1,
            K_MACRO,
        )
        _timing_stop(
            recorder, expand_start, "flashinfer_core_csr_expand",
            step, layer, q.device,
        )
        q_flat = q.reshape(heads * sequence, 1, dim)
        k_flat = k.reshape(heads * sequence, 1, 1, dim)
        v_flat = v.reshape(heads * sequence, 1, 1, dim)
        output = torch.empty_like(q_flat)
        lse = torch.empty(
            (heads * sequence, 1), dtype=torch.float32, device=q.device
        ) if return_lse else None
        enable_pdl = device_support_pdl(q.device) if device_support_pdl is not None else False
        self.module.paged_run(
            self.float_workspace,
            self.int_workspace,
            self.plan_info,
            q_flat,
            k_flat,
            v_flat,
            self.qo_indptr,
            self.vector_indptr,
            vector_indices,
            self.last_page_len,
            output,
            lse,
            MaskMode.NON_CAUSAL.value,
            TensorLayout.NHD.value,
            -1,
            enable_pdl,
            None,  # packed custom mask
            None,  # mask indptr
            None,  # alibi slopes
            None,  # maybe_prefix_len_ptr
            None,  # maybe_token_pos_in_items_ptr
            None,  # maybe_max_item_len_ptr
            0.0,  # logits_soft_cap
            dim ** -0.5,
            None, None, None,  # fp8 scales
            1.0, 1.0e4,  # rope scale/theta (unused with NONE)
            0,  # token_pos_in_items_len
        )
        output = output.reshape(heads, sequence, dim)
        if not return_lse:
            return output
        return output, lse.reshape(heads, sequence)


if triton is not None:
    @triton.jit
    def _residual_micro_attention_kernel(
        q_ptr, k_ptr, v_ptr, index_ptr, indptr_ptr, row_ids_ptr, out_ptr, lse_ptr,
        sequence, q_micro_blocks, k_micro_blocks, active_offset,
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
        active_row = active_offset + pid
        tl.store(
            out_ptr + active_row * stride_oh + rows[:, None] * stride_os + dims[None, :], output,
            mask=q_valid[:, None] & (dims[None, :] < head_dim),
        )
        row_lse = tl.where(l > 0, m + tl.log(l), float("-inf"))
        tl.store(lse_ptr + active_row * micro_size + rows, row_lse, mask=q_valid)

    @triton.jit
    def _merge_lse_active_rows_kernel(
        core_out_ptr, core_lse_ptr, residual_out_ptr, residual_lse_ptr,
        active_rows_ptr, sequence, q_micro_blocks,
        stride_oh, stride_os, stride_roh, stride_ros,
        head_dim: tl.constexpr, block_d: tl.constexpr,
        micro_size: tl.constexpr,
    ):
        """Merge only Q16 rows which actually contain Residual support."""
        pid = tl.program_id(0)
        csr_row = tl.load(active_rows_ptr + pid)
        head, qb = csr_row // q_micro_blocks, csr_row % q_micro_blocks
        rows = tl.arange(0, micro_size)
        dims = tl.arange(0, block_d)
        q_pos = qb * micro_size + rows
        valid = q_pos < sequence
        flat = head * sequence + q_pos
        core_lse = tl.load(core_lse_ptr + flat, mask=valid, other=float("-inf"))
        residual_lse = tl.load(
            residual_lse_ptr + pid * micro_size + rows,
            mask=valid, other=float("-inf"),
        )
        m = tl.maximum(core_lse, residual_lse)
        wc = tl.where(core_lse != float("-inf"), tl.exp(core_lse - m), 0.0)
        wr = tl.where(residual_lse != float("-inf"), tl.exp(residual_lse - m), 0.0)
        z = tl.maximum(wc + wr, 1.0e-20)
        offsets = head * stride_oh + q_pos[:, None] * stride_os + dims[None, :]
        mask = valid[:, None] & (dims[None, :] < head_dim)
        core = tl.load(core_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(
            residual_out_ptr
            + pid * stride_roh + rows[:, None] * stride_ros + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < head_dim), other=0.0,
        ).to(tl.float32)
        merged = (wc[:, None] * core + wr[:, None] * residual) / z[:, None]
        # Core output is dead after the merge, so update it in place and avoid
        # allocating/copying a full [H,S,D] output tensor.
        tl.store(core_out_ptr + offsets, merged, mask=mask)

    @triton.jit
    def _merge_lse_center_rows_kernel(
        core_out_ptr, core_lse_ptr, residual_out_ptr, residual_lse_ptr,
        active_rows_ptr, sequence, q_micro_blocks,
        stride_oh, stride_os, stride_ro,
        head_dim: tl.constexpr, block_d: tl.constexpr,
        micro_size: tl.constexpr, center_offset: tl.constexpr,
    ):
        """Merge RoDe output for one center token per selected Q16 row."""
        pid = tl.program_id(0)
        csr_row = tl.load(active_rows_ptr + pid)
        head = csr_row // q_micro_blocks
        qb = csr_row % q_micro_blocks
        q_pos = qb * micro_size + center_offset
        valid = q_pos < sequence
        rows = tl.arange(0, block_d)
        dims = rows < head_dim
        core_lse = tl.load(
            core_lse_ptr + head * sequence + q_pos,
            mask=valid, other=float("-inf"),
        )
        residual_lse = tl.load(residual_lse_ptr + pid, mask=valid, other=float("-inf"))
        m = tl.maximum(core_lse, residual_lse)
        wc = tl.where(core_lse != float("-inf"), tl.exp(core_lse - m), 0.0)
        wr = tl.where(residual_lse != float("-inf"), tl.exp(residual_lse - m), 0.0)
        z = tl.maximum(wc + wr, 1.0e-20)
        offsets = head * stride_oh + q_pos * stride_os + rows
        core = tl.load(core_out_ptr + offsets, mask=valid & dims, other=0.0).to(tl.float32)
        residual = tl.load(
            residual_out_ptr + pid * stride_ro + rows,
            mask=valid & dims, other=0.0,
        ).to(tl.float32)
        merged = (wc * core + wr * residual) / z
        tl.store(core_out_ptr + offsets, merged, mask=valid & dims)


def _build_rode_center_csr(route, *, heads: int, sequence: int):
    """Expand selected Q16/K16 tiles into one center-token CSR edge each."""
    cache = getattr(route, "_rode_center_csr", None)
    if cache is not None:
        return cache

    device = route.residual_indices.device
    q_micro_blocks = _ceil_div(sequence, MICRO)
    micro_rows = torch.arange(
        heads * q_micro_blocks, device=device, dtype=torch.long
    )
    counts = (route.residual_indptr[1:] - route.residual_indptr[:-1]).to(torch.long)
    edge_micro_rows = torch.repeat_interleave(micro_rows, counts)
    key_micro = route.residual_indices.to(torch.long)
    center_q = (edge_micro_rows % q_micro_blocks) * MICRO + (MICRO // 2)
    center_k = key_micro * MICRO + (MICRO // 2)
    valid = (center_q < sequence) & (center_k < sequence)
    edge_micro_rows = edge_micro_rows[valid]
    center_k = center_k[valid]
    heads_for_edge = edge_micro_rows // q_micro_blocks
    center_rows = heads_for_edge * sequence + center_q[valid]
    center_cols = heads_for_edge * sequence + center_k

    row_count = torch.bincount(
        center_rows, minlength=heads * sequence
    ).to(torch.int32)
    row_ptr = torch.empty(heads * sequence + 1, dtype=torch.int32, device=device)
    row_ptr[0] = 0
    torch.cumsum(row_count, dim=0, dtype=torch.int32, out=row_ptr[1:])
    active = row_count > 0
    active_token_rows = torch.nonzero(active, as_tuple=False).flatten().to(torch.int32)
    active_q16_rows = (
        (active_token_rows.to(torch.long) // sequence) * q_micro_blocks
        + ((active_token_rows.to(torch.long) % sequence) // MICRO)
    ).to(torch.int32)

    # edge_micro_rows is generated in CSR row order, so center_cols already
    # matches the row_ptr order expected by RoDe.
    cache = {
        "row_ptr_cpu": row_ptr.cpu().contiguous(),
        "columns_cpu": center_cols.to(torch.int32).cpu().contiguous(),
        "row_ptr": row_ptr,
        "edge_rows": center_rows.to(torch.int32),
        "active_q16_rows": active_q16_rows,
        "active_token_rows": active_token_rows,
        "nnz": int(center_cols.numel()),
    }
    route._rode_center_csr = cache
    return cache


def _run_residual_rode_center(q, k, v, route, recorder=None, step=-1, layer=-1):
    """Run downloaded RoDe on center-token CSR edges from residual tiles."""
    if load_rode_extension is None:
        raise RuntimeError("RoDe backend loader is unavailable")
    heads, sequence, dim = q.shape
    cache = _build_rode_center_csr(route, heads=heads, sequence=sequence)
    if cache["nnz"] == 0:
        return None

    plan = getattr(route, "_rode_center_plan", None)
    if plan is None:
        plan_start = _timing_start(recorder, q.device)
        rode = load_rode_extension()
        plan = rode.RoDeCenterPlan(
            cache["row_ptr_cpu"], cache["columns_cpu"], heads * sequence, 32, 512
        )
        route._rode_center_plan = plan
        _timing_stop(
            recorder, plan_start, "flashinfer_residual_rode_plan",
            step, layer, q.device,
        )

    fp32_start = _timing_start(recorder, q.device)
    qf = q.float().reshape(heads * sequence, dim).contiguous()
    kf = k.float().reshape(heads * sequence, dim).contiguous()
    vf = v.float().reshape(heads * sequence, dim).contiguous()
    _timing_stop(
        recorder, fp32_start, "flashinfer_residual_rode_fp32",
        step, layer, q.device,
    )

    sddmm_start = _timing_start(recorder, q.device)
    scores = plan.sddmm(qf, kf) * (dim ** -0.5)
    _timing_stop(
        recorder, sddmm_start, "flashinfer_residual_rode_sddmm",
        step, layer, q.device,
    )

    softmax_start = _timing_start(recorder, q.device)
    edge_rows = cache["edge_rows"].to(torch.long)
    max_rows = torch.full(
        (heads * sequence,), float("-inf"), device=q.device, dtype=torch.float32
    )
    max_rows.scatter_reduce_(0, edge_rows, scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(scores - max_rows[edge_rows])
    norm = torch.zeros_like(max_rows)
    norm.scatter_add_(0, edge_rows, exp_scores)
    probs = exp_scores / norm[edge_rows].clamp_min(torch.finfo(torch.float32).tiny)
    lse = max_rows + torch.log(norm.clamp_min(torch.finfo(torch.float32).tiny))
    _timing_stop(
        recorder, softmax_start, "flashinfer_residual_rode_softmax",
        step, layer, q.device,
    )

    spmm_start = _timing_start(recorder, q.device)
    output_full = plan.spmm(probs, vf)
    center_output = output_full[cache["active_token_rows"].to(torch.long)].contiguous()
    center_lse = lse[cache["active_token_rows"].to(torch.long)].contiguous()
    _timing_stop(
        recorder, spmm_start, "flashinfer_residual_rode_spmm",
        step, layer, q.device,
    )
    return center_output, center_lse, cache["active_q16_rows"]


def _merge_lse_center_rows(
    core_output, core_lse, residual_output, residual_lse, active_q16_rows, sequence,
):
    if triton is None:
        raise RuntimeError("RoDe center merge requires Triton")
    if active_q16_rows.numel() == 0:
        return core_output
    _merge_lse_center_rows_kernel[(active_q16_rows.numel(),)](
        core_output, core_lse, residual_output, residual_lse,
        active_q16_rows, sequence, _ceil_div(sequence, MICRO),
        core_output.stride(0), core_output.stride(1), residual_output.stride(0),
        head_dim=core_output.shape[-1],
        block_d=triton.next_power_of_2(core_output.shape[-1]),
        micro_size=MICRO,
        center_offset=MICRO // 2,
        num_warps=4,
    )
    return core_output


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
    active_count = route.residual_active_rows.numel()
    # Compact outputs avoid memset/write traffic for empty Q16 rows.  The
    # active-row ordering is exactly the concatenated bucket ordering below.
    output = torch.empty(
        (active_count, MICRO, dim), dtype=q.dtype, device=q.device
    )
    lse = torch.empty(
        (active_count, MICRO), dtype=torch.float32, device=q.device
    )
    active_offset = 0
    for bucket_capacity, row_ids in route.residual_buckets:
        bucket_start = _timing_start(recorder, q.device)
        _residual_micro_attention_kernel[(row_ids.numel(),)](
            q, k, v, indices, route.residual_indptr, row_ids, output, lse,
            sequence, q_micro_blocks, _ceil_div(sequence, MICRO), active_offset,
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
        active_offset += row_ids.numel()
    return output, lse


def _merge_lse_states(core_output, core_lse, residual_output, residual_lse):
    m = torch.maximum(core_lse, residual_lse)
    wc = torch.where(torch.isfinite(core_lse), torch.exp(core_lse - m), torch.zeros_like(m))
    wr = torch.where(torch.isfinite(residual_lse), torch.exp(residual_lse - m), torch.zeros_like(m))
    z = (wc + wr).clamp_min(torch.finfo(wc.dtype).tiny)
    return (wc[..., None] * core_output.float() + wr[..., None] * residual_output.float()) / z[..., None]


def _merge_lse_active_rows(
    core_output, core_lse, residual_output, residual_lse, route,
):
    """Exact in-place LSE merge restricted to non-empty Residual Q16 rows."""
    active_rows = route.residual_active_rows
    if active_rows.numel() == 0:
        return core_output
    if triton is None or not core_output.is_cuda:
        merged = _merge_lse_states(
            core_output, core_lse, residual_output, residual_lse
        ).to(core_output.dtype)
        q_micro_blocks = _ceil_div(core_output.shape[1], MICRO)
        row_mask = torch.zeros(
            core_output.shape[0] * q_micro_blocks,
            dtype=torch.bool, device=core_output.device,
        )
        row_mask[active_rows.long()] = True
        token_mask = row_mask.view(core_output.shape[0], q_micro_blocks).repeat_interleave(
            MICRO, dim=1
        )[:, :core_output.shape[1]]
        return torch.where(token_mask[..., None], merged, core_output)
    if residual_output.ndim != 3 or residual_output.shape[1] != MICRO:
        raise RuntimeError("CUDA Residual output must use compact active-row layout")
    _merge_lse_active_rows_kernel[(active_rows.numel(),)](
        core_output, core_lse, residual_output, residual_lse,
        active_rows, core_output.shape[1], _ceil_div(core_output.shape[1], MICRO),
        core_output.stride(0), core_output.stride(1),
        residual_output.stride(0), residual_output.stride(1),
        head_dim=core_output.shape[2],
        block_d=triton.next_power_of_2(core_output.shape[2]),
        micro_size=MICRO,
        num_warps=4,
    )
    return core_output


class FlashInfer64Attention:
    """Per-layer compact route; expanded FlashInfer plan is model-wide."""

    def __init__(self):
        self.route: Optional[FlashInfer64Route] = None
        # Optional lightweight cache used only by the RoDe center experiment.
        # It keeps the support/CSR needed to reuse a RoDe plan, but drops the
        # micro-residual buckets and other transient route data.
        self._rode_route_cache: Optional[FlashInfer64Route] = None
        self.last_stats = None
        self._direct_core_plan: Optional[_DirectMacroCSRPlan] = None
        self._route_dense_heads_key = ()
        self._route_high_omission_heads_key = ()
        self._route_residual_scorer_key = ()
        self._parallel_residual_stream: Optional[torch.cuda.Stream] = None
        self._parallel_residual_device = None

    @staticmethod
    def _compact_rode_route(route: FlashInfer64Route) -> FlashInfer64Route:
        """Drop data needed only by the ordinary micro residual backend."""
        route.residual_mask = None
        route.residual_buckets = ()
        return route

    def _get_parallel_residual_stream(self, device: torch.device) -> torch.cuda.Stream:
        if (
            self._parallel_residual_stream is None
            or self._parallel_residual_device != device
        ):
            self._parallel_residual_stream = torch.cuda.Stream(device=device)
            self._parallel_residual_device = device
        return self._parallel_residual_stream

    def _plan_core(self, q, core_mask, *, route_identity=None, direct_macro_csr=True):
        global _shared_core_wrapper, _shared_core_workspace, _shared_core_signature
        if not q.is_cuda:
            return None, False
        if flashinfer is None or FlashInferVariableBlockSparseAttention is None:
            raise ImportError("hierarchical backend requires FlashInfer")
        heads, sequence, dim = q.shape
        signature = (q.device.index, q.dtype, heads, sequence, dim)
        if _shared_core_workspace is None or _shared_core_signature != signature:
            workspace_mb = int(os.environ.get("FLASHINFER64_WORKSPACE_MB", "128"))
            if workspace_mb < 16:
                raise ValueError("FLASHINFER64_WORKSPACE_MB must be at least 16")
            _shared_core_workspace = torch.empty(
                workspace_mb * 1024 * 1024, device=q.device, dtype=torch.uint8
            )
            _shared_core_wrapper = None
            _shared_core_signature = signature
        if direct_macro_csr and _direct_macro_csr_available(q):
            if (
                self._direct_core_plan is None
                or self._direct_core_plan.float_workspace is not _shared_core_workspace
            ):
                self._direct_core_plan = _DirectMacroCSRPlan(_shared_core_workspace, q)
            if self._direct_core_plan.route_identity is not route_identity:
                self._direct_core_plan.plan(q, core_mask, route_identity)
                return self._direct_core_plan, True
            return self._direct_core_plan, False
        if _shared_core_wrapper is None:
            _shared_core_wrapper = FlashInferVariableBlockSparseAttention(
                _shared_core_workspace, backend="auto"
            )
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
        return _shared_core_wrapper, True

    def _run_core(self, q, k, v, route, wrapper, recorder, step, layer):
        start = _timing_start(recorder, q.device)
        if not q.is_cuda:
            mask = route.core_mask.repeat_interleave(Q_MACRO, 1).repeat_interleave(K_MACRO, 2)
            mask = mask[:, :q.shape[1], :k.shape[1]]
            scores = torch.bmm(q, k.transpose(1, 2)).float() * (q.shape[-1] ** -0.5)
            scores.masked_fill_(~mask, float("-inf"))
            lse = torch.logsumexp(scores, -1)
            valid = mask.any(dim=-1, keepdim=True)
            weights = torch.softmax(torch.where(valid, scores, torch.zeros_like(scores)), -1)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            output = torch.bmm(weights.to(v.dtype), v)
        else:
            if wrapper is None:
                raise RuntimeError("shared FlashInfer plan is missing")
            if isinstance(wrapper, _DirectMacroCSRPlan):
                output, lse = wrapper.run(
                    q, k, v, return_lse=True,
                    recorder=recorder, step=step, layer=layer,
                )
            else:
                output, lse = wrapper.run(q, k, v, return_lse=True)
            lse = lse * math.log(2.0)
        if bool(route.core_empty_qblocks.any().item()):
            # The execution-only anchor keeps the FlashInfer ABI valid, but
            # must not contribute to the logical union with Residual.
            for head, qblock in route.core_empty_qblocks.nonzero(as_tuple=False).tolist():
                start = qblock * Q_MACRO
                end = min(start + Q_MACRO, q.shape[1])
                output[head, start:end] = 0
                lse[head, start:end] = float("-inf")
        _timing_stop(recorder, start, "flashinfer_core_run", step, layer, q.device)
        return output, lse

    @torch.no_grad()
    def __call__(
        self, q, k, v, *, video_perm, video_len,
        route_mode="topk_topp", tile_top_p: float, tile_top_ratio: float = 0.25,
        fine_top_ratio: float = 0.2,
        fine_top_k: Optional[int] = None,
        token_top_k: Optional[int], token_top_ratio: float, token_top_p: float = 0.9,
        promotion_threshold: int = 24,
        dense_layer: int = -1,
        dense_heads=(),
        high_omission_heads_file: Optional[str] = None,
        residual_scorer: str = "proxy",
        residual_temperature: float = 1.0,
        residual_min_top_k: int = 20,
        residual_max_top_k: int = 32,
        reuse_route: bool = False,
        core_only: bool = False,
        direct_macro_csr: bool = True,
        valid_sequence: Optional[int] = None, refresh_route: bool,
        record_density: bool = False, timing_recorder: Any = None,
        step_idx: int = -1, layer_idx: int = -1,
    ):
        del token_top_k, token_top_ratio  # retained only for old launch scripts
        if residual_scorer not in ("proxy", "sampled_lse"):
            raise ValueError(
                "residual_scorer must be 'proxy' or 'sampled_lse'"
            )
        if residual_temperature <= 0:
            raise ValueError("residual_temperature must be positive")
        if residual_min_top_k < 0 or residual_max_top_k < 0:
            raise ValueError("residual Top-k bounds must be non-negative")
        if residual_min_top_k > residual_max_top_k:
            raise ValueError("residual_min_top_k cannot exceed residual_max_top_k")
        residual_backend = os.environ.get(
            "FLASHINFER64_RESIDUAL_BACKEND", "micro"
        ).lower()
        if residual_backend not in ("micro", "rode_center"):
            raise ValueError(
                "FLASHINFER64_RESIDUAL_BACKEND must be 'micro' or 'rode_center'"
            )
        if residual_backend == "rode_center" and route_mode != "topk_topp":
            raise ValueError(
                "rode_center is isolated to the already2.md topk_topp route; "
                f"got route_mode={route_mode!r}"
            )
        rode_cache_value = os.environ.get(
            "FLASHINFER64_RODE_CACHE", "False"
        ).strip().lower()
        if rode_cache_value not in ("0", "1", "false", "true", "no", "yes", "off", "on"):
            raise ValueError(
                "FLASHINFER64_RODE_CACHE must be a boolean string"
            )
        rode_cache_enabled = (
            residual_backend == "rode_center"
            and not reuse_route
            and rode_cache_value in ("1", "true", "yes", "on")
        )
        if not rode_cache_enabled:
            self._rode_route_cache = None
        replay_file = os.environ.get("FLASHINFER64_REPLAY_MASK_FILE", "")
        if replay_file and route_mode != "topk_topp":
            raise ValueError("FLASHINFER64_REPLAY_MASK_FILE requires route_mode=topk_topp")
        high_omission_heads_key = ()
        high_omission_heads_by_layer = {}
        if route_mode in ("topp_topk", "topk_topp"):
            high_omission_heads_by_layer, high_omission_heads_key = (
                _load_high_omission_heads(high_omission_heads_file)
            )
        high_omission_heads = high_omission_heads_by_layer.get(layer_idx, ())
        residual_scorer_key = (
            residual_scorer,
            float(residual_temperature),
            int(residual_min_top_k),
            int(residual_max_top_k),
        ) if route_mode in ("topp_topk", "topk_topp") else ()
        dense_heads_key = ()
        if route_mode == "topp_topk" and dense_layer == layer_idx:
            dense_heads_key = (
                None
                if dense_heads is None
                else tuple(sorted(set(int(head) for head in dense_heads)))
            )
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
            and (fine_top_k is None or self.route.fine_top_k == fine_top_k)
            and self.route.fine_top_ratio == fine_top_ratio
            and self.route.core_only == core_only
            and self._route_dense_heads_key == dense_heads_key
            and self._route_high_omission_heads_key == high_omission_heads_key
            and self._route_residual_scorer_key == residual_scorer_key
            and self.route.core_mask.device == q.device
        )
        if rode_cache_enabled and not refresh_route:
            cached_route = self._rode_route_cache
            cached_route_valid = (
                cached_route is not None
                and cached_route.sequence == sequence
                and cached_route.video_len == (-1 if video_len is None else video_len)
                and cached_route.route_mode == route_mode
                and cached_route.top_p == tile_top_p
                and cached_route.tile_top_ratio == tile_top_ratio
                and cached_route.token_top_p == token_top_p
                and cached_route.promotion_threshold == promotion_threshold
                and (fine_top_k is None or cached_route.fine_top_k == fine_top_k)
                and cached_route.fine_top_ratio == fine_top_ratio
                and cached_route.core_only == core_only
                and self._route_dense_heads_key == dense_heads_key
                and self._route_high_omission_heads_key == high_omission_heads_key
                and self._route_residual_scorer_key == residual_scorer_key
                and cached_route.core_mask.device == q.device
            )
            if cached_route_valid:
                # Keep the same route object identity so the direct Core plan
                # and the RoDe plan both hit their per-layer caches.
                self.route = cached_route
                route_valid = True
        if refresh_route or not route_valid:
            fine_start = _timing_start(timing_recorder, q.device)
            micro_scores = compute_micro_tile_scores(qh, kh)
            _timing_stop(timing_recorder, fine_start, "flashinfer_fine_score", step_idx, layer_idx, q.device)

            macro_scores = None
            if route_mode != "fine_topk_occupancy":
                core_score_start = _timing_start(timing_recorder, q.device)
                macro_scores = aggregate_macro_scores(micro_scores)
                _timing_stop(
                    timing_recorder, core_score_start, "flashinfer_core_score",
                    step_idx, layer_idx, q.device,
                )

            if route_mode == "fine_topk_occupancy":
                fine_select_start = _timing_start(timing_recorder, q.device)
                fine_selection_scores = micro_scores
                if video_len is not None and video_len < sequence:
                    video_k_blocks = min(
                        micro_scores.shape[2] // K_MICROS_PER_MACRO,
                        video_len // K_MACRO,
                    )
                    video_k_micro = video_k_blocks * K_MICROS_PER_MACRO
                    fine_selection_scores = micro_scores.masked_fill(
                        torch.arange(micro_scores.shape[2], device=q.device)[None, None, :]
                        >= video_k_micro,
                        float("-inf"),
                    )
                fine_indices, fine_valid = select_fine_topk_from_scores(
                    fine_selection_scores,
                    top_k=fine_top_k,
                    top_ratio=None if fine_top_k is not None else fine_top_ratio,
                )
                resolved_fine_top_k = fine_indices.shape[-1]
                _timing_stop(
                    timing_recorder, fine_select_start, "flashinfer_fine_topk",
                    step_idx, layer_idx, q.device,
                )
                # Preserve the unchanged Fine Top-k support for the
                # statistics-only per-head occupancy/tau analysis.  The
                # residual builder below reuses ``fine_valid`` for its
                # compact route and must not destroy this input.
                selected_fine_valid = _fine_route_valid_mask(
                    fine_indices,
                    fine_valid,
                    q_micro_blocks=micro_scores.shape[1],
                    sequence=sequence,
                    video_len=video_len,
                )
                occupancy_start = _timing_start(timing_recorder, q.device)
                core, fine_indices, fine_valid, occupancy = _fine_topk_occupancy_route(
                    micro_scores,
                    fine_top_k=resolved_fine_top_k,
                    fine_top_ratio=fine_top_ratio,
                    occupancy_threshold=promotion_threshold,
                    sequence=sequence,
                    video_len=video_len,
                    fine_indices=fine_indices,
                    fine_valid=fine_valid,
                )
                _timing_stop(
                    timing_recorder, occupancy_start, "flashinfer_occupancy",
                    step_idx, layer_idx, q.device,
                )
                if record_density:
                    head_occupancy_stats, head_tau_stats = _compute_head_occupancy_tau_stats(
                        micro_scores,
                        fine_indices,
                        selected_fine_valid,
                        occupancy,
                        sequence=sequence,
                        video_len=video_len,
                    )
                else:
                    head_occupancy_stats, head_tau_stats = (), ()
                promoted = int((occupancy >= promotion_threshold).sum().item())
                if core_only:
                    indices = torch.empty(0, dtype=torch.int32, device=q.device)
                    indptr = torch.zeros(
                        micro_scores.shape[0] * micro_scores.shape[1] + 1,
                        dtype=torch.int32, device=q.device,
                    )
                    buckets = ()
                    active_rows = torch.empty(0, dtype=torch.int32, device=q.device)
                    residual_mask = None if q.is_cuda else torch.zeros_like(micro_scores, dtype=torch.bool)
                    residual_count_stats = (0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    partition_start = _timing_start(timing_recorder, q.device)
                    residual_mask, indices, indptr, buckets, active_rows, residual_count_stats = (
                        _build_residual_csr_from_fine_indices(
                            fine_indices, fine_valid,
                            q_micro_blocks=micro_scores.shape[1],
                            k_micro_blocks=micro_scores.shape[2],
                            core_mask=core,
                            build_mask=not q.is_cuda,
                        )
                    )
                    _timing_stop(
                        timing_recorder, partition_start, "flashinfer_partition",
                        step_idx, layer_idx, q.device,
                    )
                residual = None
            else:
                head_occupancy_stats, head_tau_stats = (), ()
                core_select_start = _timing_start(timing_recorder, q.device)
                core = _select_hyvideo_core_tiles(
                    macro_scores, route_mode=route_mode, tile_top_p=tile_top_p,
                    tile_top_ratio=tile_top_ratio, sequence=sequence, video_len=video_len,
                )
                if dense_heads_key != ():
                    _force_dense_heads(core, dense_heads_key)
                _timing_stop(timing_recorder, core_select_start, "flashinfer_core_select", step_idx, layer_idx, q.device)

                if core_only:
                    residual = torch.zeros_like(micro_scores, dtype=torch.bool)
                    promoted = 0
                    indices = torch.empty(0, dtype=torch.int32, device=q.device)
                    indptr = torch.zeros(
                        residual.shape[0] * residual.shape[1] + 1,
                        dtype=torch.int32, device=q.device,
                    )
                    buckets = ()
                    active_rows = torch.empty(0, dtype=torch.int32, device=q.device)
                    residual_count_stats = (0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    residual_select_start = _timing_start(timing_recorder, q.device)
                    if replay_file:
                        residual = _replay_residual_mask(
                            micro_scores,
                            core,
                            replay_file=replay_file,
                            step_idx=step_idx,
                            layer_idx=layer_idx,
                        )
                    else:
                        risk_head_mask = torch.zeros(
                            micro_scores.shape[0], dtype=torch.bool, device=q.device
                        )
                        valid_risk_heads = tuple(
                            head for head in high_omission_heads
                            if head < micro_scores.shape[0]
                        )
                        if valid_risk_heads:
                            risk_head_mask[list(valid_risk_heads)] = True
                        score_overrides = None
                        if residual_scorer == "sampled_lse" and valid_risk_heads:
                            peak_score_start = _timing_start(timing_recorder, q.device)
                            peak_scores = compute_peak_aware_micro_tile_scores(
                                qh[list(valid_risk_heads)],
                                kh[list(valid_risk_heads)],
                                temperature=residual_temperature,
                                q_chunk_size=int(os.environ.get(
                                    "FLASHINFER64_SAMPLED_LSE_Q_CHUNK", "128"
                                )),
                            )
                            score_overrides = {
                                head: peak_scores[index:index + 1]
                                for index, head in enumerate(valid_risk_heads)
                            }
                            _timing_stop(
                                timing_recorder,
                                peak_score_start,
                                "flashinfer_peak_score",
                                step_idx,
                                layer_idx,
                                q.device,
                            )
                        residual = _select_residual_to_total_mass(
                            micro_scores,
                            core,
                            total_top_p=token_top_p,
                            head_mask=risk_head_mask,
                            min_top_k=residual_min_top_k if high_omission_heads else 0,
                            max_top_k=residual_max_top_k if high_omission_heads else None,
                            score_overrides=score_overrides,
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
                    active_rows = torch.cat(
                        tuple(rows for _, rows in buckets), dim=0
                    ) if buckets else torch.empty(0, dtype=torch.int32, device=q.device)
                    _timing_stop(timing_recorder, compact_start, "flashinfer_residual_compact", step_idx, layer_idx, q.device)
                occupancy = torch.zeros_like(core, dtype=torch.uint8)
                residual_mask = residual if not q.is_cuda else None
            core_execution_mask, core_empty_qblocks = _make_core_execution_mask(
                core, occupancy
            )
            core_n, _unused_residual_n, _unused_residual_tiles = _count_route_interactions(
                core, residual, sequence,
            )
            if residual is None:
                residual_n, residual_tiles = _count_residual_csr_interactions(
                    indices, indptr, sequence, micro_scores.shape[1]
                )
            else:
                _, residual_n, residual_tiles = _count_route_interactions(
                    core, residual, sequence
                )
            # CUDA execution consumes only the compact K16 lists.  Keeping the
            # full bool matrix would cost about 180MB per layer at 480p.
            residual_mask_for_route = residual_mask
            occupancy_histogram = tuple(
                torch.bincount(occupancy.reshape(-1).long(), minlength=49).tolist()
            )
            self.route = FlashInfer64Route(
                core, core_execution_mask, core_empty_qblocks,
                residual_mask_for_route, indices, indptr, buckets, active_rows,
                sequence, -1 if video_len is None else video_len,
                route_mode, tile_top_p, tile_top_ratio, token_top_p,
                promotion_threshold, core_only, promoted,
                core_n, residual_n, residual_tiles,
                *residual_count_stats,
                fine_indices.shape[-1] if route_mode == "fine_topk_occupancy" else (
                    fine_top_k if fine_top_k is not None else 0
                ),
                fine_top_ratio, promotion_threshold, occupancy, occupancy_histogram,
                head_occupancy_stats, head_tau_stats,
            )
            self._route_dense_heads_key = dense_heads_key
            self._route_high_omission_heads_key = high_omission_heads_key
            self._route_residual_scorer_key = residual_scorer_key
            del micro_scores, macro_scores, residual

        route = self.route
        if route is None:
            raise RuntimeError("failed to build hierarchical route")
        if timing_recorder is not None:
            occupancy_total = max(1, sum(route.occupancy_histogram))
            timing_recorder.record_metrics({
                "residual_count_mean": route.residual_count_mean,
                "residual_count_p50": route.residual_count_p50,
                "residual_count_p95": route.residual_count_p95,
                "residual_count_max": route.residual_count_max,
                "residual_count_nonempty_ratio": route.residual_count_nonempty_ratio,
                "fine_top_k": route.fine_top_k,
                "fine_top_ratio": route.fine_top_ratio,
                "occupancy_threshold": route.occupancy_threshold,
                "occupancy_mean": sum(
                    i * count for i, count in enumerate(route.occupancy_histogram)
                ) / occupancy_total,
            })
        # On FA3, each layer retains its compact scheduler plan and cache12
        # skips the next 11 plan calls.  The VariableBlock compatibility path
        # remains ephemeral and is replanned because its expanded buffers are
        # shared model-wide.
        direct_hit = bool(
            qh.is_cuda and direct_macro_csr and self._direct_core_plan is not None
            and self._direct_core_plan.route_identity is route
        )
        if direct_hit:
            wrapper = self._direct_core_plan
        else:
            plan_start = _timing_start(timing_recorder, q.device)
            wrapper, _ = self._plan_core(
                qh, route.core_execution_mask, route_identity=route,
                direct_macro_csr=direct_macro_csr,
            )
            _timing_stop(timing_recorder, plan_start, "flashinfer_plan", step_idx, layer_idx, q.device)

        has_residual = route.residual_indices.numel() != 0
        parallel_requested = os.environ.get(
            "FLASHINFER64_PARALLEL_CORE_RESIDUAL", "False"
        ).strip().lower() in ("1", "true", "yes", "on")
        parallel_core_residual = bool(
            parallel_requested and qh.is_cuda and has_residual and not core_only
        )
        current_stream = None
        residual_stream = None
        if parallel_core_residual:
            current_stream = torch.cuda.current_stream(qh.device)
            residual_stream = self._get_parallel_residual_stream(qh.device)
            # Capture all preprocessing and route/plan work already queued on
            # the current stream.  Core is launched below on the current
            # stream, while Residual starts after this snapshot and can run
            # concurrently with Core.
            residual_stream.wait_stream(current_stream)

        core_output, core_lse = self._run_core(
            qh, kh, vh, route, wrapper,
            timing_recorder, step_idx, layer_idx,
        )

        def run_residual_backend():
            if residual_backend == "rode_center":
                residual_start = _timing_start(timing_recorder, q.device)
                result = _run_residual_rode_center(
                    qh, kh, vh, route, timing_recorder, step_idx, layer_idx,
                )
                _timing_stop(
                    timing_recorder, residual_start, "flashinfer_residual_rode_run",
                    step_idx, layer_idx, q.device,
                )
                return result

            residual_start = _timing_start(timing_recorder, q.device)
            result = _run_residual_micro(
                qh, kh, vh, route, timing_recorder, step_idx, layer_idx,
            )
            _timing_stop(
                timing_recorder, residual_start, "flashinfer_residual_micro_run",
                step_idx, layer_idx, q.device,
            )
            return result

        if route.residual_indices.numel() == 0:
            merged = core_output
        else:
            if parallel_core_residual:
                assert residual_stream is not None and current_stream is not None
                with torch.cuda.stream(residual_stream):
                    residual_result = run_residual_backend()
                # Merge executes on the caller's stream and must observe all
                # Residual writes before reading its compact output/LSE.
                current_stream.wait_stream(residual_stream)
            else:
                residual_result = run_residual_backend()

            if residual_result is None:
                merged = core_output
            elif residual_backend == "rode_center":
                residual_output, residual_lse, active_q16_rows = residual_result
                merge_start = _timing_start(timing_recorder, q.device)
                merged = _merge_lse_center_rows(
                    core_output, core_lse, residual_output, residual_lse,
                    active_q16_rows, sequence,
                )
                _timing_stop(
                    timing_recorder, merge_start, "flashinfer_lse_merge",
                    step_idx, layer_idx, q.device,
                )
            else:
                residual_output, residual_lse = residual_result
                merge_start = _timing_start(timing_recorder, q.device)
                merged = _merge_lse_active_rows(
                    core_output, core_lse, residual_output, residual_lse, route,
                )
                _timing_stop(
                    timing_recorder, merge_start, "flashinfer_lse_merge",
                    step_idx, layer_idx, q.device,
                )

        density_start = _timing_start(timing_recorder, q.device)
        if record_density:
            padding = full_sequence - valid_sequence
            padding_n = qh.shape[0] * padding * padding
            occupancy_total = max(1, sum(route.occupancy_histogram))
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
                "fine_top_k": route.fine_top_k,
                "occupancy_threshold": route.occupancy_threshold,
                "occupancy_mean": sum(
                    i * count for i, count in enumerate(route.occupancy_histogram)
                ) / occupancy_total,
                "occupancy_histogram": route.occupancy_histogram,
                "head_occupancy_stats": route.head_occupancy_stats,
                "head_tau_stats": route.head_tau_stats,
                "total_possible": qh.shape[0] * (sequence * sequence + padding * padding),
            }
        _timing_stop(
            timing_recorder, density_start, "flashinfer_density_accounting",
            step_idx, layer_idx, q.device,
        )
        if rode_cache_enabled:
            # Preserve only the support and the already-built RoDe/center
            # plans.  The ordinary micro backend's buckets are not needed by
            # this isolated path and can be released before the next step.
            self._rode_route_cache = self._compact_rode_route(route)
        output_start = _timing_start(timing_recorder, q.device)
        output = restore(merged)
        _timing_stop(timing_recorder, output_start, "flashinfer_output_unpermute", step_idx, layer_idx, q.device)
        if not reuse_route:
            self.route = None
            if self._direct_core_plan is not None and not rode_cache_enabled:
                # Do not keep the full route alive solely through the cache
                # identity when the caller selected bounded-memory mode.
                self._direct_core_plan.route_identity = None
        return output


__all__ = [
    "FlashInfer64Attention", "FlashInfer64Route", "Q_MACRO", "K_MACRO", "MICRO",
    "compute_micro_tile_scores", "compute_peak_aware_micro_tile_scores",
    "aggregate_macro_scores", "compute_64_tile_scores",
    "select_64_tiles", "select_64_tiles_from_scores", "select_64_tiles_topk_from_scores",
    "select_fine_topk_from_scores",
]
