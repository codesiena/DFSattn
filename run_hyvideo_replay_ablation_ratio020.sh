#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/../../res/Hunyuan_OracleReplay_Ratio020_PerStepCore_AnchorReplay_20260909}"
MASK_ROOT="${MASK_ROOT:-${REPO_ROOT}/../../res/Hunyuan_K16_DeltaE_3Videos_K96Core_Ratio020_20260908/replay_masks}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: CUDA/NVIDIA driver unavailable; replay ablation not started." >&2
    exit 2
fi
if [ ! -f "$MASK_ROOT/manifest.json" ]; then
    echo "ERROR: replay mask manifest not found: $MASK_ROOT/manifest.json" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"

run_group() {
    local name="$1"
    local replay_dir="$2"
    local out_dir="$RESULT_ROOT/$name"
    mkdir -p "$out_dir"
    echo "===== $name ====="
    START_IDX=0 END_IDX=2 \
    OUTPUT_DIR="$out_dir/generation" \
    SPARSE_EXECUTION=flashinfer64 \
    FLASHINFER64_ROUTE_MODE=topk_topp \
    FLASHINFER64_TILE_TOP_RATIO=0.20 \
    FLASHINFER64_TOKEN_TOP_P=0.90 \
    FLASHINFER64_PROMOTION_THRESHOLD=24 \
    FLASHINFER64_ROUTE_CACHE=False \
    FLASHINFER64_DIRECT_MACRO_CSR=True \
    FLASHINFER64_CORE_ONLY=False \
    FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER=0 \
    CACHE_INTERVAL=12 \
    PROMPT_SET=33 \
    HEIGHT=480 WIDTH=720 NUM_FRAMES=129 NUM_INFERENCE_STEPS=50 \
    RECORD_DENSITY=True RECORD_TIMING=True \
    FLASHINFER64_REPLAY_MASK_DIR="$replay_dir" \
    bash "$REPO_ROOT/hyvideo_t2v_720p_dfs.sh"
}

run_group "oracle_addback" "$MASK_ROOT/oracle"
run_group "proxy_addback" "$MASK_ROOT/proxy"
run_group "random_addback" "$MASK_ROOT/random"

echo "All ratio=0.20 replay ablations completed under $RESULT_ROOT"
