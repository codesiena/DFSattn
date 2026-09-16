#!/usr/bin/env bash
# Fixed, non-adaptive risk breadth/depth sweep. Existing videos are skipped.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/cnic/work/liutt/mywork/attention_time/res/hymor_risk_breadth_sweep_20260915}"
RISK_ROOT="${REPO_ROOT}/importanthead/risk_sets"
GENERATOR="${REPO_ROOT}/importanthead/generate_ranked_risk_sets.py"
MODEL_ROOT="${HYVIDEO_MODEL_ID:-/work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5}"

cd "$REPO_ROOT"
if [ ! -f "$MODEL_ROOT/model_index.json" ] || [ ! -f "$MODEL_ROOT/transformer/config.json" ]; then
    echo "ERROR: HunyuanVideo model is not available at: $MODEL_ROOT"
    exit 1
fi
echo "Using HunyuanVideo model: $MODEL_ROOT"
python "$GENERATOR" --output-dir "$RISK_ROOT"

run_prompt() {
    local name="$1"
    local risk_file="$2"
    local min_k="$3"
    local max_k="$4"
    local scorer="$5"
    local promotion_tau="$6"
    local seed="$7"
    local prompt_idx="$8"
    local output_dir="${RESULT_ROOT}/${name}/seed${seed}_ctamul0"
    local output_file="${output_dir}/${prompt_idx}.mp4"

    mkdir -p "$output_dir"
    if [ -s "$output_file" ]; then
        echo "[$(date '+%F %T')] SKIP  name=${name} seed=${seed} prompt=${prompt_idx} (video exists)"
        return 0
    fi
    echo "[$(date '+%F %T')] START name=${name} seed=${seed} prompt=${prompt_idx} risk=$(basename "$risk_file") min=${min_k} max=${max_k} scorer=${scorer} tau=${promotion_tau}"
    if CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=. \
        SPARSE_EXECUTION=flashinfer64 \
        FLASHINFER64_ROUTE_MODE=topk_topp \
        FLASHINFER64_TILE_TOP_RATIO=0.30 \
        FLASHINFER64_DYNAMIC_TILE_RATIO=True \
        FLASHINFER64_TOKEN_TOP_P=0.90 \
        FLASHINFER64_RESIDUAL_SCORER="$scorer" \
        FLASHINFER64_RESIDUAL_TEMPERATURE=1.0 \
        FLASHINFER64_RESIDUAL_MIN_TOP_K="$min_k" \
        FLASHINFER64_RESIDUAL_MAX_TOP_K="$max_k" \
        FLASHINFER64_HIGH_OMISSION_HEADS_FILE="$risk_file" \
        FLASHINFER64_PROMOTION_THRESHOLD="$promotion_tau" \
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
        PROMPT_SET=33 \
        START_IDX="$prompt_idx" \
        END_IDX="$prompt_idx" \
        SEED="$seed" \
        HYVIDEO_MODEL_ID="$MODEL_ROOT" \
        HEIGHT=480 \
        WIDTH=720 \
        NUM_FRAMES=129 \
        NUM_INFERENCE_STEPS=50 \
        ORDER=hilbert3d \
        RECORD_DENSITY=True \
        RECORD_TIMING=True \
        OUTPUT_DIR="$output_dir" \
        /usr/bin/time -f "wall_seconds=%e user_seconds=%U sys_seconds=%S" \
        -o "${output_dir}/wall_prompt${prompt_idx}.txt" \
        bash "${REPO_ROOT}/hyvideo_t2v_720p_dfs.sh"; then
        if [ -f "${output_dir}/${prompt_idx}_timing.csv" ]; then
            cp "${output_dir}/${prompt_idx}_timing.csv" "${output_dir}/timing_prompt${prompt_idx}.csv"
        fi
        echo "[$(date '+%F %T')] DONE  name=${name} seed=${seed} prompt=${prompt_idx}"
    else
        status=$?
        echo "[$(date '+%F %T')] FAIL  name=${name} seed=${seed} prompt=${prompt_idx} status=${status}"
    fi
}

RISK_BREADTH_CONFIGS=(
    # Existing breadth baseline points; run_prompt skips completed videos.
    n5_proxy_k20_32
    n100_proxy_k20_32
    n300_proxy_k20_32
    n600_proxy_k20_32
    # New priority experiments: breadth, endpoint, matched-budget, random control.
    n800_proxy_k20_32
    n1000_proxy_k20_32
    n1000_proxy_k16_26
    random800_proxy_k20_32
)

run_config_prompt() {
    local config="$1"
    local prompt_idx="$2"
    case "$config" in
        n5_proxy_k20_32)    run_prompt "$config" "$RISK_ROOT/risk_top5.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        n100_proxy_k20_32)  run_prompt "$config" "$RISK_ROOT/risk_top100.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        n300_proxy_k20_32)  run_prompt "$config" "$RISK_ROOT/risk_top300.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        n600_proxy_k20_32)  run_prompt "$config" "$RISK_ROOT/risk_top600.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        n800_proxy_k20_32)  run_prompt "$config" "$RISK_ROOT/risk_top800.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        n1000_proxy_k20_32) run_prompt "$config" "$RISK_ROOT/risk_top1000.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        n1000_proxy_k16_26) run_prompt "$config" "$RISK_ROOT/risk_top1000.txt" 16 26 proxy 24 0 "$prompt_idx" ;;
        random800_proxy_k20_32) run_prompt "$config" "$RISK_ROOT/risk_random800_seed20260915.txt" 20 32 proxy 24 0 "$prompt_idx" ;;
        *) echo "Unknown config: $config"; return 2 ;;
    esac
}

echo "Priority phase: fixed risk-head breadth/depth sweep, 8 configurations x 3 prompts = 24 videos"
echo "Configurations: n5/n100/n300/n600/n800/n1000 (k20-32), n1000 (k16-26), random800 (k20-32)"
echo "Prompts: 0, 7, 18"
for config in "${RISK_BREADTH_CONFIGS[@]}"; do
    for prompt_idx in 0 7 18; do
        run_config_prompt "$config" "$prompt_idx"
    done
done

echo "[$(date '+%F %T')] Fixed HyMoR risk breadth priority sweep complete"
