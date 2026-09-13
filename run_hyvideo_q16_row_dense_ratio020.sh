#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/../../res/Hunyuan_Q16RowDense_Ratio020_20260909}"
MASK_ROOT="${MASK_ROOT:-${REPO_ROOT}/../../res/Hunyuan_K16_DeltaE_3Videos_K96Core_Ratio020_20260908/replay_masks/q16_row_dense}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: CUDA/NVIDIA driver unavailable; Q16 row-dense replay not started." >&2
    exit 2
fi
if [ ! -f "$MASK_ROOT/manifest.json" ]; then
    echo "ERROR: row-dense replay manifest not found: $MASK_ROOT/manifest.json" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
for prompt_idx in 0 1 2; do
    out_dir="$RESULT_ROOT/prompt_${prompt_idx}"
    mkdir -p "$out_dir"
    START_IDX="$prompt_idx" END_IDX="$prompt_idx" \
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
    SKIP_STEPS=12 \
    PROMPT_SET=33 \
    HEIGHT=480 WIDTH=720 NUM_FRAMES=129 NUM_INFERENCE_STEPS=50 \
    RECORD_DENSITY=True RECORD_TIMING=True \
    FLASHINFER64_REPLAY_MASK_DIR="$MASK_ROOT" \
    bash "$REPO_ROOT/hyvideo_t2v_720p_dfs.sh"
done

echo "Q16 row-dense replay completed under $RESULT_ROOT"
