#!/usr/bin/env bash
# Two n300 controls on VBench-66 prompts 0--9. Existing videos are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${HYVIDEO_MODEL_ID:-/work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5}"
PROMPT66="/cnic/work/liutt/mywork/attention/Sparse-VideoGen/data/vbench_data/vbench_66_prompts.txt"
RISK_ROOT="${REPO_ROOT}/importanthead/risk_sets"
RESULT_ROOT="${RESULT_ROOT:-/cnic/work/liutt/mywork/attention_time/res/hymor_vbench_full_20260916}"

if [ ! -f "$PROMPT66" ] || [ "$(wc -l < "$PROMPT66")" -ne 66 ]; then
    echo "ERROR: VBench-66 prompt file is missing or does not contain 66 lines: $PROMPT66" >&2
    exit 1
fi
if [ ! -f "$MODEL_ROOT/model_index.json" ] || [ ! -f "$MODEL_ROOT/transformer/config.json" ]; then
    echo "ERROR: HunyuanVideo model is not available at: $MODEL_ROOT" >&2
    exit 1
fi

run_config() {
    local name="$1"
    local risk_file="$2"
    local scorer="$3"
    local output_dir="${RESULT_ROOT}/${name}/vbench_66/HunyuanVideo/seed0_tilekratio0.30_dynamicTrue_dcrt0.10_totaltopp0.90_score${scorer}_temp1.0_mink20_maxk32_promote24_resbackendmicro_rodecacheFalse_parallelFalse_hilbert3d_480_routecacheTrue_coreonlyFalse_directcsrTrue_ctamul0_cache12"
    mkdir -p "$output_dir"
    echo "[$(date '+%F %T')] START ${name}: VBench-66 prompts 0--9"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=. \
    SPARSE_EXECUTION=flashinfer64 \
    FLASHINFER64_ROUTE_MODE=topk_topp \
    FLASHINFER64_TILE_TOP_RATIO=0.30 \
    FLASHINFER64_DYNAMIC_TILE_RATIO=True \
    FLASHINFER64_TOKEN_TOP_P=0.90 \
    FLASHINFER64_RESIDUAL_SCORER="$scorer" \
    FLASHINFER64_RESIDUAL_TEMPERATURE=1.0 \
    FLASHINFER64_RANDOM_TOKEN_SEED=20260916 \
    FLASHINFER64_RESIDUAL_MIN_TOP_K=20 \
    FLASHINFER64_RESIDUAL_MAX_TOP_K=32 \
    FLASHINFER64_HIGH_OMISSION_HEADS_FILE="$risk_file" \
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
    PROMPT_FILE="$PROMPT66" \
    START_IDX=0 \
    END_IDX=9 \
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
    echo "[$(date '+%F %T')] DONE ${name}"
}

cd "$REPO_ROOT"
run_config n300_random_heads_k20_32 "$RISK_ROOT/risk_random300_seed20260915.txt" proxy
run_config n300_random_tokens_k20_32 "$RISK_ROOT/risk_top300.txt" random
echo "[$(date '+%F %T')] All n300 random-control runs complete"
