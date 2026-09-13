#!/usr/bin/env python3
"""Build replay masks that restore every non-Core K16 for risky Q16 rows.

The risky Q16 rows are the same high-error Layer-12 Head-8/18/5 groups used
by the individual K16 analysis.  ``k16=-1`` is a compact replay sentinel for
the inference backend and means all valid K16 positions in that Q16 row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--heads", default="8,18,5")
    args = ap.parse_args()
    heads = {int(x) for x in args.heads.split(",") if x.strip()}
    paths = sorted(args.input_root.glob("*_prompt*/k16_deltaE_all.csv.gz"))
    if len(paths) != 3:
        raise FileNotFoundError(f"expected 3 prompt CSVs under {args.input_root}")

    cols = ["step", "layer", "head", "sample_type", "q16", "k16_global", "delta_e_k16"]
    parts = []
    for video, path in enumerate(paths):
        frame = pd.read_csv(path, usecols=cols)
        frame.insert(0, "video", video)
        frame["group"] = frame.groupby(
            ["video", "step", "layer", "head", "sample_type", "q16"],
            sort=False,
        ).ngroup() + video * 100000
        parts.append(frame)
    detail = pd.concat(parts, ignore_index=True)
    best = detail.loc[detail.groupby("group").delta_e_k16.idxmax()].copy()
    high = best[best.sample_type == "high_error"]
    threshold = float(high.delta_e_k16.quantile(0.9)) if args.threshold is None else float(args.threshold)
    selected = best[
        (best.sample_type == "high_error")
        & (best.layer == 12)
        & best["head"].isin(heads)
        & (best.delta_e_k16 >= threshold)
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "method": "q16_row_dense",
        "source": str(args.input_root),
        "threshold": threshold,
        "heads": sorted(heads),
        "selected_q16_rows": int(len(selected)),
        "videos": {},
    }
    for video, sub in selected.groupby("video", sort=True):
        rows: dict[str, list[list[int]]] = {}
        for row in sub.itertuples(index=False):
            key = f"{int(row.step)}:{int(row.layer)}"
            rows.setdefault(key, []).append([int(row.head), int(row.q16), -1])
        for values in rows.values():
            values.sort()
        out = args.output_dir / f"prompt_{int(video)}.json"
        out.write_text(
            json.dumps({"schema_version": 1, "method": "q16_row_dense", "rows": rows}, indent=2),
            encoding="utf-8",
        )
        manifest["videos"][str(int(video))] = {
            "q16_rows": int(len(sub)),
            "anchors": sorted(rows),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
