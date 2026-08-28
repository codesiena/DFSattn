#!/usr/bin/env bash
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME="/work/liutt/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# block_sparse_attn 在 import 阶段就需要 torch 动态库
export LD_LIBRARY_PATH="/work/liutt/miniconda3/envs/SVG/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

prompt_set="${PROMPT_SET:-11}"
if [ "$prompt_set" = "33" ]; then
    default_prompt_file="${REPO_ROOT}/examples/vbench_33_prompts.txt"
else
    default_prompt_file="${REPO_ROOT}/examples/vbench_11_prompts.txt"
fi
prompt_file="${PROMPT_FILE:-$default_prompt_file}"
res_root="$(cd "${REPO_ROOT}/../.." && pwd)/res"
output_dir="${OUTPUT_DIR:-${res_root}/wan/dense/vbench${prompt_set}}"
start_idx="${START_IDX:-0}"
seed="${SEED:-42}"
height="${HEIGHT:-480}"
width="${WIDTH:-832}"
num_frames="${NUM_FRAMES:-81}"
num_inference_steps="${NUM_INFERENCE_STEPS:-50}"
model_id="${WAN_MODEL_ID:-}"
record_timing="${RECORD_TIMING:-False}"

if [ -z "$model_id" ]; then
    echo "ERROR: Set WAN_MODEL_ID."
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

    python "${REPO_ROOT}/wan21_t2v_inference.py" \
        --model_id "$model_id" \
        --seed "$seed" \
        --height "$height" \
        --width "$width" \
        --num_frames "$num_frames" \
        --num_inference_steps "$num_inference_steps" \
        --prompt "$prompt_file" \
        --prompt_source "T2V_Wan_VBench" \
        --prompt_idx "$prompt_idx" \
        --output_file "$out_file" \
        --mode "flash" \
        --record_timing "$record_timing" \
        --timing_csv "${out_file%.mp4}_timing.csv"

    echo "Successfully generated video for prompt $prompt_idx"
done

echo "Finished processing all prompts"
