#!/usr/bin/env bash
# Three-stage sampled-LSE experiment suite. Existing non-empty videos are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${HYVIDEO_MODEL_ID:-/work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5}"
PROMPT_ROOT="/cnic/work/liutt/mywork/attention/Sparse-VideoGen/data/vbench_data"
PROMPT66="${PROMPT_ROOT}/vbench_66_prompts.txt"
PROMPT33="${PROMPT_ROOT}/vbench_33_prompts.txt"
RISK_ROOT="${REPO_ROOT}/importanthead/risk_sets"
RESULT_ROOT="${RESULT_ROOT:-/cnic/work/liutt/mywork/attention_time/res/hymor_sampled_lse_suite_20260917}"
REPRESENTATIVE_PROMPTS=(0 3 6 9 12 15 18 21 24 27 30)

check_lines() {
    local file="$1"
    local expected="$2"
    if [ ! -f "$file" ] || [ "$(wc -l < "$file")" -ne "$expected" ]; then
        echo "ERROR: expected ${expected} lines in ${file}" >&2
        exit 1
    fi
}

check_lines "$PROMPT66" 66
check_lines "$PROMPT33" 33
if [ ! -f "$MODEL_ROOT/model_index.json" ] || [ ! -f "$MODEL_ROOT/transformer/config.json" ]; then
    echo "ERROR: HunyuanVideo model is not available at: $MODEL_ROOT" >&2
    exit 1
fi
for risk_file in risk_top300.txt risk_top600.txt risk_random600_seed20260915.txt; do
    if [ ! -f "${RISK_ROOT}/${risk_file}" ]; then
        echo "ERROR: missing risk set: ${RISK_ROOT}/${risk_file}" >&2
        exit 1
    fi
done

run_range() {
    local phase="$1"
    local name="$2"
    local prompt_set="$3"
    local prompt_file="$4"
    local start_idx="$5"
    local end_idx="$6"
    local risk_file="$7"
    local scorer="$8"
    local total_top_p="$9"
    local output_dir="${RESULT_ROOT}/${phase}/${name}/vbench_${prompt_set}/HunyuanVideo/seed0_tilekratio0.30_dynamicTrue_dcrt0.10_totaltopp${total_top_p}_score${scorer}_temp1.0_mink20_maxk32_promote24_resbackendmicro_rodecacheFalse_parallelFalse_hilbert3d_480_routecacheTrue_coreonlyFalse_directcsrTrue_ctamul0_cache12"

    mkdir -p "$output_dir"
    echo "[$(date '+%F %T')] START ${phase}/${name}: VBench-${prompt_set} prompt ${start_idx}--${end_idx}"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=. \
    SPARSE_EXECUTION=flashinfer64 \
    FLASHINFER64_ROUTE_MODE=topk_topp \
    FLASHINFER64_TILE_TOP_RATIO=0.30 \
    FLASHINFER64_DYNAMIC_TILE_RATIO=True \
    FLASHINFER64_TOKEN_TOP_P="$total_top_p" \
    FLASHINFER64_RESIDUAL_SCORER="$scorer" \
    FLASHINFER64_RESIDUAL_TEMPERATURE=1.0 \
    FLASHINFER64_RANDOM_TOKEN_SEED=20260917 \
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
    PROMPT_SET="$prompt_set" \
    PROMPT_FILE="$prompt_file" \
    START_IDX="$start_idx" \
    END_IDX="$end_idx" \
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
    echo "[$(date '+%F %T')] DONE ${phase}/${name}: prompt ${start_idx}--${end_idx}"
}

cd "$REPO_ROOT"

echo "========== Phase 1: VBench66 top-p selection (Top600 sampled-LSE, prompts 0--9) =========="
for total_top_p in 0.80 0.85 0.90 0.95; do
    run_range phase1_parameter_selection "top600_sampled_lse_p${total_top_p/./}" \
        66 "$PROMPT66" 0 9 "$RISK_ROOT/risk_top600.txt" sampled_lse "$total_top_p"
done

echo "========== Phase 2: VBench33 main results (fixed top-p=0.90) =========="
run_range phase2_main_results top600_sampled_lse_p090 \
    33 "$PROMPT33" 0 32 "$RISK_ROOT/risk_top600.txt" sampled_lse 0.90
run_range phase2_main_results top300_sampled_lse_p090 \
    33 "$PROMPT33" 0 32 "$RISK_ROOT/risk_top300.txt" sampled_lse 0.90

echo "========== Phase 3: two mechanism ablations (11 matched VBench33 prompts) =========="
for prompt_idx in "${REPRESENTATIVE_PROMPTS[@]}"; do
    run_range phase3_mechanism_ablations random600_heads_sampled_lse_p090 \
        33 "$PROMPT33" "$prompt_idx" "$prompt_idx" \
        "$RISK_ROOT/risk_random600_seed20260915.txt" sampled_lse 0.90
done
for prompt_idx in "${REPRESENTATIVE_PROMPTS[@]}"; do
    run_range phase3_mechanism_ablations top600_sampled_lse_count_matched_random_token_p090 \
        33 "$PROMPT33" "$prompt_idx" "$prompt_idx" \
        "$RISK_ROOT/risk_top600.txt" sampled_lse_random 0.90
done

echo "[$(date '+%F %T')] All three sampled-LSE phases complete"
