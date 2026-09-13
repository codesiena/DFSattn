#!/usr/bin/env python3
"""Summarize cross-video Hunyuan K96/Core independent K16 add-backs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["video", "step", "layer", "head", "q16"]


def _hit(values: pd.Series, ks=(1, 2, 5, 10, 20, 40)) -> dict[str, float]:
    return {f"top{k}": float((values <= k).mean()) for k in ks}


def _scope_summary(frame: pd.DataFrame) -> dict:
    positions = frame["best_k16"].value_counts().head(30)
    layer_heads = (
        frame.groupby(["layer", "head"])["best_delta"]
        .agg(["count", "mean", "median", "max"])
        .sort_values(["count", "mean"], ascending=False)
        .head(50)
        .reset_index()
    )
    return {
        "q16_count": int(len(frame)),
        "best_delta": {
            "mean": float(frame.best_delta.mean()),
            "median": float(frame.best_delta.median()),
            "p90": float(frame.best_delta.quantile(0.9)),
            "max": float(frame.best_delta.max()),
        },
        "true_best_k16_rank": {
            "proxy_tile": _hit(frame.proxy_rank),
            "dense_mass_oracle": _hit(frame.dense_rank),
            "proxy_macro_M": _hit(frame.proxy_M_rank),
            "proxy_macro_A": _hit(frame.proxy_A_rank),
            "dense_macro_M_oracle": _hit(frame.dense_M_rank),
            "dense_macro_A_oracle": _hit(frame.dense_A_rank),
        },
        "top_best_k16_positions": [
            {"k16_global": int(k), "count": int(n), "fraction": float(n / len(frame))}
            for k, n in positions.items()
        ],
        "top_layer_heads": layer_heads.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input_root.glob("*_prompt*/k16_deltaE_all.csv.gz"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least two video CSVs under {args.input_root}")

    columns = [
        "step", "layer", "head", "sample_type", "q16", "kv_macro",
        "k16_global", "delta_e_k16", "dense_tile_mass", "proxy_tile_score",
    ]
    parts = []
    for video, path in enumerate(files):
        frame = pd.read_csv(path, usecols=columns)
        frame.insert(0, "video", video)
        parts.append(frame)
    detail = pd.concat(parts, ignore_index=True)
    detail["group"] = detail.groupby(KEYS, sort=False).ngroup()
    best_index = detail.groupby("group", sort=False).delta_e_k16.idxmax()
    best = detail.loc[best_index].copy().rename(columns={
        "k16_global": "best_k16", "kv_macro": "best_macro",
        "delta_e_k16": "best_delta",
    })

    detail["proxy_rank"] = detail.groupby("group").proxy_tile_score.rank(method="min", ascending=False)
    detail["dense_rank"] = detail.groupby("group").dense_tile_mass.rank(method="min", ascending=False)
    best = best.merge(detail.loc[best_index, ["group", "proxy_rank", "dense_rank"]], on="group")
    macros = detail.groupby(["group", "kv_macro"], sort=False).agg(
        proxy_M=("proxy_tile_score", "sum"), proxy_A=("proxy_tile_score", "max"),
        dense_M=("dense_tile_mass", "sum"), dense_A=("dense_tile_mass", "max"),
    ).reset_index()
    for score in ("proxy_M", "proxy_A", "dense_M", "dense_A"):
        macros[f"{score}_rank"] = macros.groupby("group")[score].rank(method="min", ascending=False)
    true_macros = macros.merge(
        best[["group", "best_macro"]],
        left_on=["group", "kv_macro"], right_on=["group", "best_macro"],
    )
    best = best.merge(
        true_macros[["group", "proxy_M_rank", "proxy_A_rank", "dense_M_rank", "dense_A_rank"]],
        on="group",
    )

    high = best[best.sample_type.str.contains("high_error", na=False)]
    severe_threshold = float(high.best_delta.quantile(0.9))
    severe = high[high.best_delta >= severe_threshold]
    summary = {
        "schema_version": 1,
        "video_count": len(files),
        "input_files": [str(path) for path in files],
        "all_samples": _scope_summary(best),
        "high_error_samples": _scope_summary(high),
        "highest_recovery_10pct_of_high_error": _scope_summary(severe),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_video_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    severe_summary = summary["highest_recovery_10pct_of_high_error"]
    ranks = severe_summary["true_best_k16_rank"]
    positions = severe_summary["top_best_k16_positions"][:10]
    lines = [
        "# Hunyuan K96 Core：三视频逐K16独立Add-back", "",
        f"- 视频数：{len(files)}",
        f"- 采样Q16：{len(best)}",
        f"- 高误差Q16：{len(high)}",
        f"- 高误差中恢复收益最高10%阈值：{severe_threshold:.6g}", "",
        "## 严重遗漏的在线可发现性", "",
        "|信号|Top1|Top5|Top10|Top20|", "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("当前proxy K16", "proxy_tile"), ("当前proxy Macro M", "proxy_macro_M"),
        ("当前proxy Macro A", "proxy_macro_A"), ("Dense质量oracle", "dense_mass_oracle"),
    ):
        values = ranks[key]
        lines.append(
            f"|{label}|{values['top1']:.1%}|{values['top5']:.1%}|"
            f"{values['top10']:.1%}|{values['top20']:.1%}|"
        )
    lines.extend(["", "## 严重遗漏的高频真实最佳K16", ""])
    for row in positions:
        lines.append(
            f"- K16={row['k16_global']}: {row['count']} ({row['fraction']:.1%})"
        )
    lines.extend([
        "", "注意：每条记录均从同一Core独立补回一个K16；多个K16的DeltaE不能相加。",
    ])
    (args.output_dir / "cross_video_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"video_count": len(files), "q16_count": len(best), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
