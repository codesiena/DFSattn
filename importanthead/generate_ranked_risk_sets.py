#!/usr/bin/env python3
"""Generate nested Layer/Head risk sets from the full-layer K16 scan."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_TOP5 = (
    (43, 17),
    (30, 13),
    (48, 5),
    (8, 10),
    (12, 5),
)
# Breadth sweep sizes used by the priority experiment.  The generator still
# writes risk_all1440.txt and the random negative controls below.
SET_SIZES = (5, 100, 300, 600, 800, 1000)


def write_set(path: Path, pairs: list[tuple[int, int]], description: str) -> None:
    lines = [
        "# 0-based Layer/Head risk prior.",
        f"# {description}",
        "# Format: Layer<layer> / Head<head>",
    ]
    lines.extend(f"Layer{layer} / Head{head}" for layer, head in pairs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/work/liutt/Hunyuan_K16_DeltaE_3Videos_K96Core_Ratio016_20260910/"
            "01_prompt0/k96_k16_summary.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "risk_sets",
    )
    parser.add_argument("--random-seed", type=int, default=20260915)
    args = parser.parse_args()

    columns = ["layer", "head", "sample_type", "q16", "best_delta_e_k16"]
    partial = []
    for chunk in pd.read_csv(args.input, usecols=columns, chunksize=500_000):
        chunk = chunk[chunk.sample_type.eq("high_error")]
        partial.append(
            chunk.groupby(["layer", "head", "q16"], as_index=False)
            .best_delta_e_k16.max()
        )
    per_query = (
        pd.concat(partial, ignore_index=True)
        .groupby(["layer", "head", "q16"], as_index=False)
        .best_delta_e_k16.max()
    )
    ranking = (
        per_query.groupby(["layer", "head"])
        .best_delta_e_k16.agg(["count", "mean", "max", "sum"])
        .reset_index()
    )
    all_pairs = pd.MultiIndex.from_product(
        [range(60), range(24)], names=["layer", "head"]
    ).to_frame(index=False)
    ranking = all_pairs.merge(ranking, on=["layer", "head"], how="left").fillna(0)
    ranking = ranking.sort_values(
        ["sum", "count", "mean", "max", "layer", "head"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "risk_rank", np.arange(1, len(ranking) + 1))

    ranked_pairs = list(
        zip(ranking["layer"].astype(int), ranking["head"].astype(int))
    )
    nested_pairs = list(CURRENT_TOP5)
    nested_pairs.extend(pair for pair in ranked_pairs if pair not in CURRENT_TOP5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output_dir / "risk_ranking.csv", index=False)
    for size in SET_SIZES:
        write_set(
            args.output_dir / f"risk_top{size}.txt",
            nested_pairs[:size],
            f"Current Top-5 plus risk-sum-ranked candidates; N={size}.",
        )
    write_set(
        args.output_dir / "risk_all1440.txt",
        [(layer, head) for layer in range(60) for head in range(24)],
        "All 60x24 Layer/Head pairs.",
    )
    rng = np.random.default_rng(args.random_seed)
    for random_size in (300, 800):
        random_ids = rng.choice(60 * 24, size=random_size, replace=False)
        random_pairs = sorted((int(i // 24), int(i % 24)) for i in random_ids)
        write_set(
            args.output_dir / f"risk_random{random_size}_seed20260915.txt",
            random_pairs,
            f"Random N={random_size} negative control; seed=20260915.",
        )
    print(f"Wrote ranked risk sets to {args.output_dir}")


if __name__ == "__main__":
    main()
