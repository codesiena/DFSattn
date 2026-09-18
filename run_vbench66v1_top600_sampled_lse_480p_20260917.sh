#!/usr/bin/env bash
# Full VBench66 v1 main experiment: Top600 + sampled-LSE at 720x480.
# Existing non-empty videos are skipped, so this script is restart-safe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${HYVIDEO_MODEL_ID:-/work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5}"
PROMPT_FILE="/cnic/work/liutt/mywork/attention/Sparse-VideoGen/data/vbench_data/vbench_66_prompts.txt"
RISK_FILE="${REPO_ROOT}/importanthead/risk_sets/risk_top600.txt"
OUTPUT_DIR="${OUTPUT_DIR:-/cnic/work/liutt/mywork/attention_time/res/hymor_vbench66v1_top600_sampled_lse_480p_20260917/vbench_66/HunyuanVideo/seed0_tilekratio0.30_dynamicTrue_dcrt0.10_totaltopp0.90_scoresampled_lse_temp1.0_mink20_maxk32_promote24_resbackendmicro_rodecacheFalse_parallelFalse_hilbert3d_480_routecacheTrue_coreonlyFalse_directcsrTrue_ctamul0_cache12}"

if [ ! -f "$PROMPT_FILE" ] || [ "$(wc -l < "$PROMPT_FILE")" -ne 66 ]; then
    echo "ERROR: VBench66 v1 prompt file is missing or does not contain 66 lines: $PROMPT_FILE" >&2
    exit 1
fi
if [ ! -f "$RISK_FILE" ]; then
    echo "ERROR: missing Top600 risk set: $RISK_FILE" >&2
    exit 1
fi
if [ ! -f "$MODEL_ROOT/model_index.json" ] || [ ! -f "$MODEL_ROOT/transformer/config.json" ]; then
    echo "ERROR: HunyuanVideo model is not available at: $MODEL_ROOT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
echo "[$(date '+%F %T')] START Top600 + sampled-LSE, VBench66 v1 0--65, 720x480"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
SPARSE_EXECUTION=flashinfer64 \
FLASHINFER64_ROUTE_MODE=topk_topp \
FLASHINFER64_TILE_TOP_RATIO=0.30 \
FLASHINFER64_DYNAMIC_TILE_RATIO=True \
FLASHINFER64_TOKEN_TOP_P=0.90 \
FLASHINFER64_RESIDUAL_SCORER=sampled_lse \
FLASHINFER64_RESIDUAL_TEMPERATURE=1.0 \
FLASHINFER64_RESIDUAL_MIN_TOP_K=20 \
FLASHINFER64_RESIDUAL_MAX_TOP_K=32 \
FLASHINFER64_HIGH_OMISSION_HEADS_FILE="$RISK_FILE" \
FLASHINFER64_PROMOTION_THRESHOLD=24 \
FLASHINFER64_ROUTE_CACHE=True \
FLASHINFER64_CORE_ONLY=False \
FLASHINFER64_DIRECT_MACRO_CSR=True \
FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER=0 \
FLASHINFER64_RESIDUAL_BACKEND=micro \
FLASHINFER64_RODE_CACHE=False \
FLASHINFER64_PARALLEL_CORE_RESIDUAL=False \
SPARSITY=0.30 \
SPARSITY_DCRT=0.10 \
SKIP_STEPS=12 \
CACHE_INTERVAL=12 \
PROMPT_SET=66 \
PROMPT_FILE="$PROMPT_FILE" \
START_IDX=0 \
END_IDX=65 \
SEED=0 \
HYVIDEO_MODEL_ID="$MODEL_ROOT" \
HEIGHT=480 \
WIDTH=720 \
NUM_FRAMES=129 \
NUM_INFERENCE_STEPS=50 \
ORDER=hilbert3d \
RECORD_DENSITY=True \
RECORD_TIMING=True \
OUTPUT_DIR="$OUTPUT_DIR" \
bash "$REPO_ROOT/hyvideo_t2v_720p_dfs.sh"
echo "[$(date '+%F %T')] DONE Top600 + sampled-LSE, VBench66 v1 720x480"
