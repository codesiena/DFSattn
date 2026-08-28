import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import torch


HistogramKey = Tuple[int, int, int, float, int, int]


class SubblockRetentionProfiler:
    """Streaming profiler for fine scores inside selected coarse blocks.

    Records histograms instead of individual coarse-block observations, keeping
    host memory bounded even for a full HunyuanVideo denoising trajectory.
    """

    def __init__(self) -> None:
        self.output_dir: Optional[str] = None
        self.masses: Tuple[float, ...] = ()
        self.histograms: Dict[HistogramKey, int] = defaultdict(int)

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None and bool(self.masses)

    def configure(
        self,
        output_dir: Optional[str],
        masses: Iterable[float] = (0.9,),
    ) -> None:
        parsed = tuple(sorted(set(float(mass) for mass in masses)))
        if output_dir is not None and any(not 0.0 < mass <= 1.0 for mass in parsed):
            raise ValueError("sub-block attention masses must be in (0, 1]")
        self.output_dir = output_dir
        self.masses = parsed if output_dir is not None else ()
        self.histograms.clear()

    @torch.no_grad()
    def record(
        self,
        tile_score: torch.Tensor,
        selected_video_mask: torch.Tensor,
        tiles_per_q_block: int,
        tiles_per_k_block: int,
        step_idx: int,
        layer_idx: int,
        pair_chunk_size: int = 4096,
    ) -> None:
        """Accumulate minimum fine-tile counts reaching each target mass.

        ``tile_score`` has shape [1, H, Q_fine, K_fine].  The selected mask is
        restricted by the caller to genuine Top-k video-to-video coarse blocks,
        excluding forced-dense text blocks.
        """
        if not self.enabled:
            return
        if tile_score.shape[0] != 1 or selected_video_mask.ndim != 4:
            raise ValueError("sub-block profiling expects batch size 1")

        _, heads, q_blocks, k_blocks = selected_video_mask.shape
        fine = tile_score.view(
            1,
            heads,
            tile_score.shape[2] // tiles_per_q_block,
            tiles_per_q_block,
            tile_score.shape[3] // tiles_per_k_block,
            tiles_per_k_block,
        )
        fine_count = tiles_per_q_block * tiles_per_k_block

        for head_idx in range(heads):
            q_idx, k_idx = selected_video_mask[0, head_idx].nonzero(as_tuple=True)
            for start in range(0, q_idx.numel(), pair_chunk_size):
                q_chunk = q_idx[start : start + pair_chunk_size]
                k_chunk = k_idx[start : start + pair_chunk_size]
                values = fine[0, head_idx, q_chunk, :, k_chunk, :].reshape(-1, fine_count)
                values = torch.sort(values.float(), dim=-1, descending=True).values
                cumulative = values.cumsum(dim=-1)
                totals = cumulative[:, -1:]
                for mass in self.masses:
                    retained = (cumulative < totals * mass).sum(dim=-1) + 1
                    counts = torch.bincount(retained, minlength=fine_count + 1).cpu()
                    for retained_count in counts.nonzero(as_tuple=False).flatten().tolist():
                        key = (
                            int(step_idx),
                            int(layer_idx),
                            int(head_idx),
                            float(mass),
                            int(retained_count),
                            int(fine_count),
                        )
                        self.histograms[key] += int(counts[retained_count].item())

    def dump_csv(self, output_path: Optional[str] = None) -> Optional[str]:
        if not self.histograms:
            return None
        if output_path is None:
            if self.output_dir is None:
                return None
            output_path = os.path.join(self.output_dir, "subblock_retention_hist.csv")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "step_idx",
                    "layer_idx",
                    "head_idx",
                    "attention_mass",
                    "retained_subblocks",
                    "total_subblocks",
                    "retention_ratio",
                    "count",
                ]
            )
            for key, count in sorted(self.histograms.items()):
                step, layer, head, mass, retained, total = key
                writer.writerow(
                    [step, layer, head, mass, retained, total, retained / total, count]
                )
        return output_path

