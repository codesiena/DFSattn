#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep large Q/K/V debug snapshots off the repository filesystem.  `/work`
# currently has enough free space for the one-video/one-step full-layer scan;
# callers can still override RESULT_ROOT explicitly.
RESULT_ROOT="${RESULT_ROOT:-/work/liutt/Hunyuan_K16_DeltaE_3Videos_K96Core_Ratio020_20260910}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${RESULT_ROOT}/qkv_snapshots}"
GENERATION_ROOT="${GENERATION_ROOT:-${RESULT_ROOT}/generation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MACRO_TOP_RATIO="${MACRO_TOP_RATIO:-0.20}"
TOP_ERROR="${TOP_ERROR:-64}"
CONTROLS="${CONTROLS:-64}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-2}"
DEBUG_STEPS="${ATTENTION_DEBUG_STEPS:-12,24,36}"
# HunyuanVideo has 20 double-stream + 40 single-stream attention layers.
# Keep an explicit override for smoke tests, but make the full calibration
# the default so layer coverage cannot silently regress to a hand-picked
# subset.
TOTAL_LAYERS="${TOTAL_LAYERS:-60}"
if [[ -z "${ATTENTION_DEBUG_LAYERS:-}" ]]; then
    ATTENTION_DEBUG_LAYERS="$(seq -s, 0 $((TOTAL_LAYERS - 1)))"
fi

mkdir -p "$RESULT_ROOT" "$SNAPSHOT_ROOT" "$GENERATION_ROOT"

START_IDX="$START_IDX" END_IDX="$END_IDX" \
OUTPUT_DIR="$GENERATION_ROOT" \
SPARSE_EXECUTION=flashinfer64 \
FLASHINFER64_ROUTE_MODE=topk_topp \
FLASHINFER64_TILE_TOP_RATIO="$MACRO_TOP_RATIO" \
FLASHINFER64_TOKEN_TOP_P=0 \
FLASHINFER64_CORE_ONLY=True \
FLASHINFER64_ROUTE_CACHE=False \
ATTENTION_DEBUG_DIR="$SNAPSHOT_ROOT" \
ATTENTION_DEBUG_STEPS="$DEBUG_STEPS" \
ATTENTION_DEBUG_LAYERS="$ATTENTION_DEBUG_LAYERS" \
ATTENTION_DEBUG_STOP=True \
HEIGHT=480 WIDTH=720 NUM_FRAMES=129 NUM_INFERENCE_STEPS=50 \
bash "$REPO_ROOT/hyvideo_t2v_720p_dfs.sh"

for prompt_idx in $(seq "$START_IDX" "$END_IDX"); do
    label=$(printf '%02d_prompt%d' "$((prompt_idx + 1))" "$prompt_idx")
    "$PYTHON_BIN" "$REPO_ROOT/analyze_hyvideo_k16_individual_addback.py" \
        --input-root "$SNAPSHOT_ROOT/prompt_${prompt_idx}" \
        --output-dir "$RESULT_ROOT/$label" \
        --macro-top-ratio "$MACRO_TOP_RATIO" \
        --order hilbert3d --height 480 --width 720 --num-frames 129 \
        --top-error "$TOP_ERROR" --controls "$CONTROLS" \
        --candidate-batch 128 --device cuda
done

# The cross-video summary requires at least two prompt directories.  A
# one-video calibration run intentionally stops after per-video analysis.
if (( END_IDX - START_IDX + 1 >= 2 )); then
    "$PYTHON_BIN" "$REPO_ROOT/summarize_hyvideo_k16_3videos.py" \
        --input-root "$RESULT_ROOT" --output-dir "$RESULT_ROOT"
fi
