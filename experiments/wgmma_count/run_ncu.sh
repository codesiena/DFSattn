#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
out_root="${1:-${repo_root}/experiments/wgmma_count/results}"
mkdir -p "$out_root"

ncu_bin="${NCU_BIN:-/opt/nvidia/nsight-compute/2025.3.1/ncu}"
python_bin="${PYTHON_BIN:-/work/liutt/miniconda3/envs/SVG/bin/python}"
metric="smsp__inst_executed_pipe_tensor_op_gmma.sum"

for tile in ${TILES:-128x96 64x64}; do
    cache_root="${out_root}/cache_${tile}"
    report="${out_root}/flashinfer_core_${tile}"
    log="${out_root}/flashinfer_core_${tile}.log"
    env \
        FLASHINFER_WORKSPACE_BASE="$cache_root" \
        PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
        TORCH_CUDA_ARCH_LIST="9.0a" \
        "$ncu_bin" \
        --target-processes all \
        --profile-from-start off \
        --kernel-name-base demangled \
        --kernel-name regex:PrefillWithKVCacheKernel \
        --metrics "$metric" \
        --csv \
        --force-overwrite \
        --export "$report" \
        "$python_bin" "${repo_root}/experiments/wgmma_count/bench_flashinfer_core.py" \
        --tile "$tile" --profile 2>&1 | tee "$log"
done
