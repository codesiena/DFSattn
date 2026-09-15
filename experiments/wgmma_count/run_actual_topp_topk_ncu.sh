#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
out_root="${1:-${repo_root}/experiments/wgmma_count/actual_topp_topk_results}"
input_dump="${2:-${INPUT_DUMP:-}}"
mkdir -p "$out_root"
ncu_bin="${NCU_BIN:-/opt/nvidia/nsight-compute/2025.3.1/ncu}"
python_bin="${PYTHON_BIN:-/work/liutt/miniconda3/envs/SVG/bin/python}"

dump_args=()
if [[ -n "$input_dump" ]]; then
    dump_args=(--input-dump "$input_dump")
fi

for tile in 128x96 64x64; do
    cache_root="${out_root}/cache_${tile}"
    env \
        FLASHINFER_WORKSPACE_BASE="$cache_root" \
        PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
        TORCH_CUDA_ARCH_LIST="9.0a" \
        "$ncu_bin" \
        --target-processes all \
        --profile-from-start off \
        --kernel-name-base demangled \
        --kernel-name regex:PrefillWithKVCacheKernel \
        --metrics smsp__inst_executed_pipe_tensor_op_gmma.sum \
        --section LaunchStats \
        --section Occupancy \
        --force-overwrite \
        --export "${out_root}/actual_topp_topk_${tile}" \
        "$python_bin" "${repo_root}/experiments/wgmma_count/bench_actual_topp_topk_core.py" \
        --tile "$tile" "${dump_args[@]}" --profile \
        --summary-json "${out_root}/actual_topp_topk_${tile}.json" \
        2>&1 | tee "${out_root}/actual_topp_topk_${tile}.log"
done
