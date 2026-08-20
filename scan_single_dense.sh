#!/usr/bin/env bash
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME="/work/liutt/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTN_TIME_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

model_id="${WAN_MODEL_ID:-}"
step_interval="${STEP_INTERVAL:-10}"
start_idx="${START_IDX:-0}"
end_idx="${END_IDX:-2}"
seed="${SEED:-42}"
height="${HEIGHT:-720}"
width="${WIDTH:-1280}"
num_frames="${NUM_FRAMES:-81}"
num_inference_steps="${NUM_INFERENCE_STEPS:-50}"
res_root="${ATTN_TIME_ROOT}/res"
output_dir="${OUTPUT_DIR:-${res_root}/wan/scan_single_dense}"
prompt_file="${PROMPT_FILE:-${REPO_ROOT}/examples/vbench_33_prompts.txt}"
prompt_source="${PROMPT_SOURCE:-T2V_Wan_VBench}"

sparsity="${SPARSITY:-0.3}"
tile_size="${TILE_SIZE:-16}"
block_size="${BLOCK_SIZE:-128}"
order="${ORDER:-hilbert3d}"
skip_steps="${SKIP_STEPS:-0}"
cache_interval="${CACHE_INTERVAL:-12}"
sparsity_dcrt="${SPARSITY_DCRT:-0.1}"

if [ -z "$model_id" ]; then
    echo "ERROR: Set WAN_MODEL_ID."
    exit 1
fi

if [ ! -f "$prompt_file" ]; then
    echo "ERROR: Prompt file not found: $prompt_file"
    exit 1
fi

num_prompts=$(wc -l < "$prompt_file")
if [ "$end_idx" -ge "$num_prompts" ]; then
    echo "WARNING: end_idx=$end_idx exceeds prompt count ($num_prompts), clamping."
    end_idx=$((num_prompts - 1))
fi

echo "============================================"
echo "Single Dense-Step Scanning Experiment"
echo "============================================"
echo "Model:           $model_id"
echo "Resolution:      ${height}x${width}, ${num_frames} frames"
echo "Steps:           $num_inference_steps (scan every $step_interval)"
echo "Prompts:         ${start_idx}-${end_idx} (from $prompt_file)"
echo "Output:          $output_dir"
echo "All-sparse config: skip_steps=$skip_steps, sparsity=$sparsity"
echo "============================================"

mkdir -p "$output_dir"

pushd "$REPO_ROOT" > /dev/null

python scan_single_dense.py \
    --model_id "$model_id" \
    --height "$height" \
    --width "$width" \
    --num_frames "$num_frames" \
    --num_inference_steps "$num_inference_steps" \
    --seed "$seed" \
    --prompt_file "$prompt_file" \
    --prompt_source "$prompt_source" \
    --start_idx "$start_idx" \
    --end_idx "$end_idx" \
    --output_dir "$output_dir" \
    --step_interval "$step_interval" \
    --sparsity "$sparsity" \
    --tile_size "$tile_size" \
    --block_size "$block_size" \
    --order "$order" \
    --skip_steps "$skip_steps" \
    --cache_interval "$cache_interval" \
    --sparsity_dcrt "$sparsity_dcrt"

popd > /dev/null

echo ""
echo "============================================"
echo "Experiment completed!"
echo "Results saved to: $output_dir"
echo "============================================"
