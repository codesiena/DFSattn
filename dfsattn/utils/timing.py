"""Low-perturbation CUDA timing utilities for inference instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class _EventPair:
    phase: str
    step_idx: int
    layer_idx: int
    start: Union[torch.cuda.Event, float]
    end: Union[torch.cuda.Event, float]
    device: torch.device


class AttentionTimingRecorder:
    """Records CUDA events now and resolves their durations after inference.

    Calling ``elapsed_time`` during each attention call would synchronize the
    stream and substantially perturb latency.  This recorder instead emits two
    lightweight events around each region and resolves all durations only when
    ``dump_csv`` is called after the pipeline has completed.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._records: List[_EventPair] = []
        self._metric_records: Dict[str, List[float]] = {}

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._records.clear()
        self._metric_records.clear()

    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """Record non-timing route statistics for the aggregate timing CSV."""
        if not self.enabled:
            return
        for name, value in metrics.items():
            self._metric_records.setdefault(name, []).append(float(value))

    def start(self, device: torch.device) -> Optional[Union[torch.cuda.Event, float]]:
        if not self.enabled:
            return None
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(
        self,
        start: Optional[Union[torch.cuda.Event, float]],
        *,
        phase: str,
        step_idx: int,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        if start is None:
            return
        if device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        else:
            end = time.perf_counter()
        self._records.append(_EventPair(phase, step_idx, layer_idx, start, end, device))

    def rows(self) -> List[Dict[str, Any]]:
        cuda_devices = {record.device for record in self._records if record.device.type == "cuda"}
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        rows: List[Dict[str, Any]] = []
        for record in self._records:
            if record.device.type == "cuda":
                elapsed_ms = record.start.elapsed_time(record.end)  # type: ignore[union-attr]
            else:
                elapsed_ms = (record.end - record.start) * 1_000.0  # type: ignore[operator]
            rows.append(
                {
                    "phase": record.phase,
                    "step_idx": record.step_idx,
                    "layer_idx": record.layer_idx,
                    "elapsed_ms": float(elapsed_ms),
                }
            )
        return rows

    def dump_csv(
        self,
        output_path: str,
        extra_totals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Write one aggregate row per requested timing category.

        The recorder still keeps one CUDA-event pair per instrumented region so
        that event resolution does not synchronize the stream during inference.
        At dump time those rows are reduced to video-level totals; step/layer
        details are deliberately not written to disk.

        ``top_k_selection`` is the sum of ``topk_mask`` events (the historical
        DFS selector). FlashInfer64 sub-phases are also emitted individually.
        They intentionally overlap the enclosing ``attention_execution`` row:
        the detailed rows explain that total and must not be added to it.
        """
        rows = self.rows()
        detailed_phases = (
            "flashinfer_qkv_permute",
            "flashinfer_fine_score",
            "flashinfer_core_score",
            "flashinfer_core_select",
            "flashinfer_residual_select",
            "flashinfer_promotion",
            "flashinfer_residual_compact",
            "flashinfer_plan",
            "flashinfer_core_run",
            "flashinfer_residual_micro_run",
            "flashinfer_density_accounting",
            "flashinfer_lse_merge",
            "flashinfer_output_unpermute",
        )
        phase_totals: Dict[str, float] = {
            "top_k_selection": 0.0,
            "attention_execution": 0.0,
            **{phase: 0.0 for phase in detailed_phases},
        }
        phase_calls: Dict[str, float] = {
            phase: 0.0 for phase in phase_totals
        }
        aliases = {
            "topk_mask": "top_k_selection",
            "attention_execution": "attention_execution",
        }
        for row in rows:
            raw_phase = row["phase"]
            phase = aliases.get(raw_phase)
            if phase is None and raw_phase in detailed_phases:
                phase = raw_phase
            if phase is None and raw_phase.startswith("flashinfer_residual_bucket_le"):
                phase = raw_phase
                phase_totals.setdefault(phase, 0.0)
                phase_calls.setdefault(phase, 0.0)
            if phase is None:
                continue
            phase_totals[phase] += float(row["elapsed_ms"])
            phase_calls[phase] += 1.0

        if extra_totals:
            for phase, elapsed_ms in extra_totals.items():
                phase_totals[phase] = float(elapsed_ms)
                phase_calls[phase] = 1.0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        metric_fields = (
            "residual_count_mean", "residual_count_p50", "residual_count_p95",
            "residual_count_max", "residual_count_nonempty_ratio",
        )
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["phase", "total_ms", "calls", "mean_ms", *metric_fields],
            )
            writer.writeheader()
            for phase, total_ms in phase_totals.items():
                calls = phase_calls.get(phase, 0.0)
                writer.writerow(
                    {
                        "phase": phase,
                        "total_ms": total_ms,
                        "calls": int(calls),
                        "mean_ms": total_ms / calls if calls else 0.0,
                    }
                )
            metric_values = {
                name: self._metric_records.get(name, []) for name in metric_fields
            }
            metric_calls = max((len(values) for values in metric_values.values()), default=0)
            if metric_calls:
                writer.writerow({
                    "phase": "residual_route_stats",
                    "total_ms": "",
                    "calls": metric_calls,
                    "mean_ms": "",
                    **{
                        name: sum(values) / len(values) if values else float("nan")
                        for name, values in metric_values.items()
                    },
                })

        summary: Dict[str, Dict[str, float]] = {}
        for phase, total_ms in phase_totals.items():
            calls = phase_calls.get(phase, 0.0)
            summary[phase] = {
                "calls": calls,
                "total_ms": total_ms,
                "mean_ms": total_ms / calls if calls else 0.0,
            }
        return summary
