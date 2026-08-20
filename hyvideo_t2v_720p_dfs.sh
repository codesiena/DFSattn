#!/usr/bin/env bash
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME="/work/liutt/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

sparsity="${SPARSITY:-0.3}"
skip_steps="${SKIP_STEPS:-12}"
cache_interval="${CACHE_INTERVAL:-12}"
sparsity_dcrt="${SPARSITY_DCRT:-0.1}"
tile_size="${TILE_SIZE:-16}"
block_size="${BLOCK_SIZE:-128}"
order="${ORDER:-hilbert3d}"
prompt_set="${PROMPT_SET:-11}"
if [ "$prompt_set" = "33" ]; then
    default_prompt_file="${REPO_ROOT}/examples/vbench_33_prompts.txt"
else
    default_prompt_file="${REPO_ROOT}/examples/vbench_11_prompts.txt"
fi
prompt_file="${PROMPT_FILE:-$default_prompt_file}"
res_root="$(cd "${REPO_ROOT}/../.." && pwd)/res"
start_idx="${START_IDX:-0}"
seed="${SEED:-0}"
height="${HEIGHT:-480}"
width="${WIDTH:-720}"
num_frames="${NUM_FRAMES:-129}"
num_inference_steps="${NUM_INFERENCE_STEPS:-50}"
dense_interval="${DENSE_INTERVAL:-0}"
rest_steps="${REST_STEPS:-0}"
skip_steps2="${SKIP_STEPS2:-0}"
record_density="${RECORD_DENSITY:-False}"
block_mask_dir="${BLOCK_MASK_DIR:-}"
block_mask_heads="${BLOCK_MASK_HEADS:-0}"
output_dir="${OUTPUT_DIR:-${res_root}/hyvideo/dfs_${height}x${width}_skip${skip_steps}_di${dense_interval}_vbench${prompt_set}_sp${sparsity}_spd${sparsity_dcrt}}"
model_id="${HYVIDEO_MODEL_ID:-}"

if [ -z "$model_id" ]; then
    echo "ERROR: Set HYVIDEO_MODEL_ID."
    exit 1
fi

if [ ! -f "$prompt_file" ]; then
    echo "ERROR: Prompt file not found: $prompt_file"
    exit 1
fi

num_prompts=$(wc -l < "$prompt_file")
end_idx="${END_IDX:-$((num_prompts - 1))}"

echo "Found $num_prompts prompts in $prompt_file"
echo "Generating videos for prompts ${start_idx} to ${end_idx}"
mkdir -p "$output_dir"

for prompt_idx in $(seq "$start_idx" "$end_idx"); do
    out_file="${output_dir}/${prompt_idx}.mp4"
    if [ -s "$out_file" ]; then
        echo "Skipping prompt $prompt_idx ($out_file already exists)"
        continue
    fi

    echo "Processing prompt $prompt_idx..."

    block_mask_args=()
    if [ -n "$block_mask_dir" ]; then
        block_mask_args=(
            --block_mask_dir "$block_mask_dir"
            --block_mask_heads "$block_mask_heads"
        )
    fi

    python "${REPO_ROOT}/hyvideo_t2v_inference.py" \
        --model_id "$model_id" \
        --seed "$seed" \
        --height "$height" \
        --width "$width" \
        --num_frames "$num_frames" \
        --num_inference_steps "$num_inference_steps" \
        --prompt "$prompt_file" \
        --prompt_source "T2V_Hyv_VBench" \
        --prompt_idx "$prompt_idx" \
        --output_file "$out_file" \
        --mode "dfs" \
        --sparsity "$sparsity" \
        --tile_size "$tile_size" \
        --block_size "$block_size" \
        --skip_steps "$skip_steps" \
        --cache_interval "$cache_interval" \
        --sparsity_dcrt "$sparsity_dcrt" \
        --order "$order" \
        --cache_flag True \
        --dense_interval "$dense_interval" \
        --rest_steps "$rest_steps" \
        --skip_steps2 "$skip_steps2" \
        --record_density "$record_density" \
        "${block_mask_args[@]}"

    echo "Successfully generated video for prompt $prompt_idx"
done

echo "Finished processing all prompts"
