#!/usr/bin/env bash
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME="/work/liutt/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
set -euo pipefail

# block_sparse_attn loads libc10 at import time.
export LD_LIBRARY_PATH="/work/liutt/miniconda3/envs/SVG/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

sparsity="${SPARSITY:-0.3}"
skip_steps="${SKIP_STEPS:-12}"
cache_interval="${CACHE_INTERVAL:-12}"
sparsity_dcrt="${SPARSITY_DCRT:-0.1}"
tile_size="${TILE_SIZE:-16}"
block_size="${BLOCK_SIZE:-128}"
block_top_p="${BLOCK_TOP_P:-}"
token_top_k="${TOKEN_TOP_K:-0}"
residual_candidate_blocks="${RESIDUAL_CANDIDATE_BLOCKS:-4}"
selector_mode="${SELECTOR_MODE:-topk}"
fine_top_p="${FINE_TOP_P:-0.9}"
flashinfer64_top_p="${FLASHINFER64_TOP_P:-0.25}"
flashinfer64_token_top_ratio="${FLASHINFER64_TOKEN_TOP_RATIO:-0.10}"
flashinfer64_route_mode="${FLASHINFER64_ROUTE_MODE:-topk_topp}"
flashinfer64_tile_top_ratio="${FLASHINFER64_TILE_TOP_RATIO:-0.25}"
flashinfer64_token_top_p="${FLASHINFER64_TOKEN_TOP_P:-0.9}"
flashinfer64_promotion_threshold="${FLASHINFER64_PROMOTION_THRESHOLD:-24}"
flashinfer64_route_cache="${FLASHINFER64_ROUTE_CACHE:-False}"
flashinfer64_core_only="${FLASHINFER64_CORE_ONLY:-False}"
flashinfer64_direct_macro_csr="${FLASHINFER64_DIRECT_MACRO_CSR:-True}"
if [ "$selector_mode" = "kp" ]; then
    sparse_execution="${SPARSE_EXECUTION:-hybrid}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
else
    sparse_execution="${SPARSE_EXECUTION:-native}"
fi
if [ "$sparse_execution" = "flashinfer64" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
if [ -n "$block_top_p" ]; then
    selector_tag="topp${block_top_p}_tokenk${token_top_k}_cand${residual_candidate_blocks}"
elif [ "$token_top_k" -gt 0 ]; then
    selector_tag="topk${sparsity}_tokenk${token_top_k}_cand${residual_candidate_blocks}"
elif [ "$selector_mode" = "kp" ]; then
    selector_tag="kp_FINE_TOP_P${fine_top_p}_k${sparsity}"
else
    selector_tag="${sparsity}"
fi
order="${ORDER:-hilbert3d}"
prompt_set="${PROMPT_SET:-11}"
dataset_name="${DATASET_NAME:-vbench_${prompt_set}}"
model_name="${MODEL_NAME:-HunyuanVideo}"
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
record_density="${RECORD_DENSITY:-True}"
attention_debug_dir="${ATTENTION_DEBUG_DIR:-}"
attention_debug_step="${ATTENTION_DEBUG_STEP:--1}"
attention_debug_layers="${ATTENTION_DEBUG_LAYERS:-0}"
block_mask_dir="${BLOCK_MASK_DIR:-}"
block_mask_heads="${BLOCK_MASK_HEADS:-0}"
subblock_profile_dir="${SUBBLOCK_PROFILE_DIR:-}"
subblock_profile_masses="${SUBBLOCK_PROFILE_MASSES:-0.9}"
if [ "$sparse_execution" = "flashinfer64" ]; then
    if [ "$flashinfer64_route_mode" = "topk_topp" ]; then
        flashinfer64_route_tag="tilekratio${flashinfer64_tile_top_ratio}_totaltopp${flashinfer64_token_top_p}_promote${flashinfer64_promotion_threshold}"
    else
        flashinfer64_route_tag="topp${flashinfer64_top_p}_totaltopp${flashinfer64_token_top_p}_promote${flashinfer64_promotion_threshold}"
    fi
    output_dir="${OUTPUT_DIR:-${res_root}/flashinfer64/${dataset_name}/${model_name}/seed${seed}_${flashinfer64_route_tag}_${order}_${height}_routecache${flashinfer64_route_cache}_coreonly${flashinfer64_core_only}_directcsr${flashinfer64_direct_macro_csr}_cache${cache_interval}}"
else
    output_dir="${OUTPUT_DIR:-${res_root}/${selector_tag}/${dataset_name}/${model_name}/dfs/ts${tile_size}_${block_size}_seed${seed}_${order}_${height}_cache${cache_interval}}"
fi
model_id="${HYVIDEO_MODEL_ID:-/cnic/work/liutt/mywork/2a15b5574ee77888e51ae6f593b2ceed8ce813e5}"

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

    block_top_p_args=()
    if [ -n "$block_top_p" ]; then
        block_top_p_args=(--block_top_p "$block_top_p")
    fi

    subblock_profile_args=()
    if [ -n "$subblock_profile_dir" ]; then
        subblock_profile_args=(
            --subblock_profile_dir "$subblock_profile_dir"
            --subblock_profile_masses "$subblock_profile_masses"
        )
    fi

    flashinfer64_args=(
        --flashinfer64_route_mode "$flashinfer64_route_mode"
        --flashinfer64_top_p "$flashinfer64_top_p"
        --flashinfer64_tile_top_ratio "$flashinfer64_tile_top_ratio"
        --flashinfer64_token_top_p "$flashinfer64_token_top_p"
        --flashinfer64_token_top_ratio "$flashinfer64_token_top_ratio"
        --flashinfer64_promotion_threshold "$flashinfer64_promotion_threshold"
        --flashinfer64_route_cache "$flashinfer64_route_cache"
        --flashinfer64_core_only "$flashinfer64_core_only"
        --flashinfer64_direct_macro_csr "$flashinfer64_direct_macro_csr"
    )
    attention_debug_args=()
    if [ -n "$attention_debug_dir" ]; then
        attention_debug_args=(
            --attention_debug_dir "$attention_debug_dir"
            --attention_debug_step "$attention_debug_step"
            --attention_debug_layers "$attention_debug_layers"
            --attention_debug_stop True
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
        --sparse_execution "$sparse_execution" \
        --selector_mode "$selector_mode" \
        --fine_top_p "$fine_top_p" \
        --token_top_k "$token_top_k" \
        --residual_candidate_blocks "$residual_candidate_blocks" \
        --skip_steps "$skip_steps" \
        --cache_interval "$cache_interval" \
        --sparsity_dcrt "$sparsity_dcrt" \
        --order "$order" \
        --cache_flag True \
        --dense_interval "$dense_interval" \
        --rest_steps "$rest_steps" \
        --skip_steps2 "$skip_steps2" \
        --save_dense_warmup True \
        --dense_warmup_output "${out_file%.mp4}_dense_warmup.mp4" \
        --record_density "$record_density" \
        "${flashinfer64_args[@]}" \
        "${attention_debug_args[@]}" \
        "${block_top_p_args[@]}" \
        "${subblock_profile_args[@]}" \
        "${block_mask_args[@]}"

    echo "Successfully generated video for prompt $prompt_idx"
done

if [ -n "$subblock_profile_dir" ]; then
    python "${REPO_ROOT}/analyze_subblock_retention.py" \
        --input_root "$subblock_profile_dir" \
        --output_dir "${subblock_profile_dir}/aggregate"
fi

echo "Finished processing all prompts"
