"""Lazy loader for the unmodified RoDe CUDA compute kernels.

The extension compiles the SDDMM/SpMM sources downloaded from CRAFT-THU/RoDe
and adds only a PyTorch bridge plus CSR row-descriptor construction.  The
normal FlashInfer64 residual path never imports or builds this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch


_RODE_MODULE: Optional[Any] = None


def load_rode_extension() -> Any:
    global _RODE_MODULE
    if _RODE_MODULE is not None:
        return _RODE_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("RoDe center backend requires a CUDA device")

    from torch.utils.cpp_extension import load

    repo_root = Path(__file__).resolve().parents[1]
    rode_root = repo_root / "third_party" / "RoDe" / "RoDe"
    bridge = repo_root / "dfsattn" / "rode_bridge.cu"
    sddmm = rode_root / "RoDe_SDDMM" / "RoDeSddmm.cu"
    spmm = rode_root / "RoDe_SpMM" / "RoDeSpmm.cu"
    required = (bridge, sddmm, spmm, rode_root / "third_party" / "abseil-cpp")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "RoDe source tree is incomplete; missing: " + ", ".join(missing)
        )

    build_directory = os.environ.get("DFSATTN_RODE_BUILD_DIR")
    if build_directory is None:
        build_directory = str(repo_root / "dfsattn" / "rode_build")
    Path(build_directory).mkdir(parents=True, exist_ok=True)

    _RODE_MODULE = load(
        name="dfsattn_rode",
        sources=[str(bridge), str(sddmm), str(spmm)],
        extra_include_paths=[
            str(rode_root),
            str(rode_root / "utils"),
            str(rode_root / "RoDe_SDDMM"),
            str(rode_root / "RoDe_SpMM"),
            str(rode_root / "third_party" / "abseil-cpp"),
        ],
        extra_cflags=["-O3", "-Wno-sign-compare"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=build_directory,
        with_cuda=True,
        verbose=bool(int(os.environ.get("DFSATTN_RODE_VERBOSE_BUILD", "0"))),
    )
    return _RODE_MODULE
