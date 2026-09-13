#!/usr/bin/env python3
"""Build equal-budget Oracle/Proxy/Random residual replay masks.

The masks are keyed by diffusion ``step:layer`` and contain ``[head, q16,
k16]`` entries.  Oracle entries are the high-recovery tail in Layer 12
Heads 8/18/5 from the ratio=0.20 independent-addback data.  Proxy and Random
choose one candidate for exactly the same (step, layer, head, q16) rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = ["video", "step", "layer", "head", "sample_type", "q16"]


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
    cols = [
        "step", "layer", "head", "sample_type", "q16", "k16_global",
        "delta_e_k16", "proxy_tile_score",
    ]
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
    threshold = (
        float(high.delta_e_k16.quantile(0.9))
        if args.threshold is None else float(args.threshold)
    )
    oracle = best[
        (best.sample_type == "high_error")
        & (best.layer == 12)
        & best["head"].isin(heads)
        & (best.delta_e_k16 >= threshold)
    ].copy()
    # Restrict candidate lookup to the same sampled Q16 group.  Every method
    # therefore has exactly one K16 event per oracle Q16 event.
    candidates = detail[detail.group.isin(oracle.group)].copy()
    proxy = candidates.loc[candidates.groupby("group").proxy_tile_score.idxmax()].copy()
    rng = np.random.default_rng(20260908)
    random_rows = []
    for group, frame in candidates.groupby("group", sort=False):
        random_rows.append(frame.iloc[int(rng.integers(len(frame)))])
    random = pd.DataFrame(random_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"oracle": oracle, "proxy": proxy, "random": random}
    manifest = {
        "schema_version": 1,
        "source": str(args.input_root),
        "threshold": threshold,
        "heads": sorted(heads),
        "event_count": int(len(oracle)),
        "methods": {},
    }
    for method, frame in outputs.items():
        by_key: dict[str, list[list[int]]] = {}
        for row in frame.itertuples(index=False):
            key = f"{int(row.step)}:{int(row.layer)}"
            by_key.setdefault(key, []).append(
                [int(row.head), int(row.q16), int(row.k16_global)]
            )
        for values in by_key.values():
            values.sort()
        payload = {"schema_version": 1, "method": method, "rows": by_key}
        (args.output_dir / f"prompt_{int(frame.video.iloc[0])}_{method}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        # Split per prompt so the inference shell can set one file per process.
        for video, sub in frame.groupby("video", sort=True):
            rows: dict[str, list[list[int]]] = {}
            for row in sub.itertuples(index=False):
                key = f"{int(row.step)}:{int(row.layer)}"
                rows.setdefault(key, []).append(
                    [int(row.head), int(row.q16), int(row.k16_global)]
                )
            for values in rows.values():
                values.sort()
            out = args.output_dir / method / f"prompt_{int(video)}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"schema_version": 1, "method": method, "rows": rows}, indent=2),
                encoding="utf-8",
            )
            manifest["methods"].setdefault(method, {})[str(int(video))] = int(len(sub))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
