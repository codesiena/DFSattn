#!/usr/bin/env bash
# Run n600 + k20-32 on VBench-33 and VBench-66, then native DFSAttn on VBench-66.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${HYVIDEO_MODEL_ID:-/work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5}"
PROMPT66="/cnic/work/liutt/mywork/attention/Sparse-VideoGen/data/vbench_data/vbench_66_prompts.txt"
PROMPT33="${REPO_ROOT}/examples/vbench_33_prompts.txt"
RISK600="${REPO_ROOT}/importanthead/risk_sets/risk_top600.txt"
RESULT_ROOT="${RESULT_ROOT:-/cnic/work/liutt/mywork/attention_time/res/hymor_vbench_full_20260916}"

if [ ! -f "$PROMPT66" ] || [ "$(wc -l < "$PROMPT66")" -ne 66 ]; then
    echo "ERROR: VBench-66 prompt file is missing or does not contain 66 lines: $PROMPT66" >&2
    exit 1
fi
if [ ! -f "$PROMPT33" ] || [ "$(wc -l < "$PROMPT33")" -ne 33 ]; then
    echo "ERROR: VBench-33 prompt file is missing or does not contain 33 lines: $PROMPT33" >&2
    exit 1
fi
if [ ! -f "$RISK600" ]; then
    echo "ERROR: missing n600 risk-head file: $RISK600" >&2
    exit 1
fi
if [ ! -f "$MODEL_ROOT/model_index.json" ] || [ ! -f "$MODEL_ROOT/transformer/config.json" ]; then
    echo "ERROR: HunyuanVideo model is not available at: $MODEL_ROOT" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"

run_n600() {
    local prompt_set="$1"
    local prompt_file="$2"
    local output_dir="$RESULT_ROOT/n600_proxy_k20_32/vbench_${prompt_set}/HunyuanVideo/seed0_tilekratio0.30_dynamicTrue_dcrt0.10_totaltopp0.90_scoreproxy_temp1.0_mink20_maxk32_promote24_resbackendmicro_rodecacheFalse_parallelFalse_hilbert3d_480_routecacheTrue_coreonlyFalse_directcsrTrue_ctamul0_cache12"
    mkdir -p "$output_dir"
    echo "[$(date '+%F %T')] START n600_proxy_k20_32 vbench_${prompt_set}"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=. \
    SPARSE_EXECUTION=flashinfer64 \
    FLASHINFER64_ROUTE_MODE=topk_topp \
    FLASHINFER64_TILE_TOP_RATIO=0.30 \
    FLASHINFER64_DYNAMIC_TILE_RATIO=True \
    FLASHINFER64_TOKEN_TOP_P=0.90 \
    FLASHINFER64_RESIDUAL_SCORER=proxy \
    FLASHINFER64_RESIDUAL_TEMPERATURE=1.0 \
    FLASHINFER64_RESIDUAL_MIN_TOP_K=20 \
    FLASHINFER64_RESIDUAL_MAX_TOP_K=32 \
    FLASHINFER64_HIGH_OMISSION_HEADS_FILE="$RISK600" \
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
    PROMPT_SET="$prompt_set" \
    PROMPT_FILE="$prompt_file" \
    START_IDX=0 \
    END_IDX="$(( $(wc -l < "$prompt_file") - 1 ))" \
    SEED=0 \
    HYVIDEO_MODEL_ID="$MODEL_ROOT" \
    HEIGHT=480 \
    WIDTH=720 \
    NUM_FRAMES=129 \
    NUM_INFERENCE_STEPS=50 \
    ORDER=hilbert3d \
    RECORD_DENSITY=True \
    RECORD_TIMING=True \
    OUTPUT_DIR="$output_dir" \
    bash "$REPO_ROOT/hyvideo_t2v_720p_dfs.sh"
    echo "[$(date '+%F %T')] DONE n600_proxy_k20_32 vbench_${prompt_set}"
}

run_dfsattn_vbench66() {
    local output_dir="/cnic/work/liutt/mywork/attention_time/res/0.3/vbench_66/HunyuanVideo/dfs/ts16_128_seed0_hilbert3d_480_cache12"
    mkdir -p "$output_dir"
    echo "[$(date '+%F %T')] START native DFSAttn vbench_66"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=. \
    SPARSE_EXECUTION=native \
    SELECTOR_MODE=topk \
    SPARSITY=0.30 \
    TILE_SIZE=16 \
    BLOCK_SIZE=128 \
    SKIP_STEPS=12 \
    CACHE_INTERVAL=12 \
    SPARSITY_DCRT=0.10 \
    PROMPT_SET=66 \
    PROMPT_FILE="$PROMPT66" \
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
    OUTPUT_DIR="$output_dir" \
    bash "$REPO_ROOT/hyvideo_t2v_720p_dfs.sh"
    echo "[$(date '+%F %T')] DONE native DFSAttn vbench_66"
}

cd "$REPO_ROOT"
run_n600 33 "$PROMPT33"
run_n600 66 "$PROMPT66"
run_dfsattn_vbench66
echo "[$(date '+%F %T')] ALL VBench-33/VBench-66 runs complete"
