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
flashinfer64_top_p="${FLASHINFER64_TOP_P:-0.16}"
flashinfer64_token_top_ratio="${FLASHINFER64_TOKEN_TOP_RATIO:-0.10}"
flashinfer64_route_mode="${FLASHINFER64_ROUTE_MODE:-fine_topk_occupancy}"
flashinfer64_tile_top_ratio="${FLASHINFER64_TILE_TOP_RATIO:-0.25}"
flashinfer64_dynamic_tile_ratio="${FLASHINFER64_DYNAMIC_TILE_RATIO:-False}"
flashinfer64_fine_top_ratio="${FLASHINFER64_FINE_TOP_RATIO:-0.2}"
flashinfer64_fine_top_k_override="${FLASHINFER64_FINE_TOP_K:-}"
flashinfer64_token_top_p="${FLASHINFER64_TOKEN_TOP_P:-0.9}"
flashinfer64_promotion_threshold="${FLASHINFER64_PROMOTION_THRESHOLD:-24}"
flashinfer64_dense_layer="${FLASHINFER64_DENSE_LAYER:--1}"
flashinfer64_dense_heads="${FLASHINFER64_DENSE_HEADS:-}"
flashinfer64_high_omission_heads_file="${FLASHINFER64_HIGH_OMISSION_HEADS_FILE:-}"
flashinfer64_residual_scorer="${FLASHINFER64_RESIDUAL_SCORER:-proxy}"
flashinfer64_residual_temperature="${FLASHINFER64_RESIDUAL_TEMPERATURE:-1.0}"
flashinfer64_residual_min_top_k="${FLASHINFER64_RESIDUAL_MIN_TOP_K:-20}"
flashinfer64_residual_max_top_k="${FLASHINFER64_RESIDUAL_MAX_TOP_K:-32}"
flashinfer64_route_cache="${FLASHINFER64_ROUTE_CACHE:-False}"
flashinfer64_core_only="${FLASHINFER64_CORE_ONLY:-True}"
flashinfer64_direct_macro_csr="${FLASHINFER64_DIRECT_MACRO_CSR:-True}"
flashinfer64_residual_backend="${FLASHINFER64_RESIDUAL_BACKEND:-micro}"
flashinfer64_rode_cache="${FLASHINFER64_RODE_CACHE:-False}"
flashinfer64_parallel_core_residual="${FLASHINFER64_PARALLEL_CORE_RESIDUAL:-False}"
flashinfer64_capped_residual_select="${FLASHINFER64_CAPPED_RESIDUAL_SELECT:-False}"
flashinfer64_batched_residual_select="${FLASHINFER64_BATCHED_RESIDUAL_SELECT:-False}"
flashinfer64_sampled_lse_gemm_dtype="${FLASHINFER64_SAMPLED_LSE_GEMM_DTYPE:-fp32}"
flashinfer64_residual_short_bucket_tuning="${FLASHINFER64_RESIDUAL_SHORT_BUCKET_TUNING:-}"
flashinfer64_fused_qkv_permute_tuning="${FLASHINFER64_FUSED_QKV_PERMUTE_TUNING:-}"
flashinfer64_fused_output_unpermute_tuning="${FLASHINFER64_FUSED_OUTPUT_UNPERMUTE_TUNING:-}"
export FLASHINFER64_RESIDUAL_BACKEND="$flashinfer64_residual_backend"
export FLASHINFER64_RODE_CACHE="$flashinfer64_rode_cache"
export FLASHINFER64_PARALLEL_CORE_RESIDUAL="$flashinfer64_parallel_core_residual"
export FLASHINFER64_CAPPED_RESIDUAL_SELECT="$flashinfer64_capped_residual_select"
export FLASHINFER64_BATCHED_RESIDUAL_SELECT="$flashinfer64_batched_residual_select"
export FLASHINFER64_SAMPLED_LSE_GEMM_DTYPE="$flashinfer64_sampled_lse_gemm_dtype"
export FLASHINFER64_RESIDUAL_SHORT_BUCKET_TUNING="$flashinfer64_residual_short_bucket_tuning"
export FLASHINFER64_FUSED_QKV_PERMUTE_TUNING="$flashinfer64_fused_qkv_permute_tuning"
export FLASHINFER64_FUSED_OUTPUT_UNPERMUTE_TUNING="$flashinfer64_fused_output_unpermute_tuning"
# The current best direct-CSR schedule is one CTA per CSR row.  Set this in
# the wrapper (and export it) so the effective value is reproducible even when
# the caller does not specify the variable explicitly.
flashinfer_csr_expand_cta_multiplier="${FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER:-0}"
export FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER="$flashinfer_csr_expand_cta_multiplier"
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
prompt_set="${PROMPT_SET:-33}"
dataset_name="${DATASET_NAME:-vbench_${prompt_set}}"
model_name="${MODEL_NAME:-HunyuanVideo}"
if [ "$prompt_set" = "33" ]; then
    default_prompt_file="${REPO_ROOT}/examples/vbench_33_prompts.txt"
elif [ "$prompt_set" = "66" ]; then
    default_prompt_file="/cnic/work/liutt/mywork/attention/Sparse-VideoGen/data/vbench_data/vbench_66_prompts.txt"
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
record_density="${RECORD_DENSITY:-True}"
record_timing="${RECORD_TIMING:-False}"
rest_steps="${REST_STEPS:-0}"
skip_steps2="${SKIP_STEPS2:-0}"
attention_debug_dir="${ATTENTION_DEBUG_DIR:-}"
attention_debug_step="${ATTENTION_DEBUG_STEP:--1}"
attention_debug_steps="${ATTENTION_DEBUG_STEPS:-}"
attention_debug_layers="${ATTENTION_DEBUG_LAYERS:-0}"
attention_debug_stop="${ATTENTION_DEBUG_STOP:-True}"
block_mask_dir="${BLOCK_MASK_DIR:-}"
block_mask_heads="${BLOCK_MASK_HEADS:-0}"
replay_mask_dir="${FLASHINFER64_REPLAY_MASK_DIR:-}"
subblock_profile_dir="${SUBBLOCK_PROFILE_DIR:-}"
subblock_profile_masses="${SUBBLOCK_PROFILE_MASSES:-0.9}"
if [ "$sparse_execution" = "flashinfer64" ]; then
    if [ "$flashinfer64_route_mode" = "fine_topk_occupancy" ]; then
        if [ -n "$flashinfer64_fine_top_k_override" ]; then
            flashinfer64_route_tag="finek${flashinfer64_fine_top_k_override}_tau${flashinfer64_promotion_threshold}"
        else
            flashinfer64_route_tag="fineratio${flashinfer64_fine_top_ratio}_tau${flashinfer64_promotion_threshold}"
        fi
    elif [ "$flashinfer64_route_mode" = "topk_topp" ]; then
        flashinfer64_route_tag="tilekratio${flashinfer64_tile_top_ratio}_dynamic${flashinfer64_dynamic_tile_ratio}_dcrt${sparsity_dcrt}_totaltopp${flashinfer64_token_top_p}_score${flashinfer64_residual_scorer}_temp${flashinfer64_residual_temperature}_mink${flashinfer64_residual_min_top_k}_maxk${flashinfer64_residual_max_top_k}_promote${flashinfer64_promotion_threshold}"
    else
        flashinfer64_route_tag="topp${flashinfer64_top_p}_totaltopp${flashinfer64_token_top_p}_score${flashinfer64_residual_scorer}_temp${flashinfer64_residual_temperature}_promote${flashinfer64_promotion_threshold}"
    fi
    if [ "$flashinfer64_dense_layer" -ge 0 ] 2>/dev/null && [ -n "$flashinfer64_dense_heads" ]; then
        flashinfer64_route_tag="${flashinfer64_route_tag}_denseL${flashinfer64_dense_layer}H${flashinfer64_dense_heads//,/x}"
    fi
    if [ "$flashinfer64_capped_residual_select" = "True" ] || [ "$flashinfer64_capped_residual_select" = "true" ]; then
        flashinfer64_route_tag="${flashinfer64_route_tag}_cappedselectTrue"
    fi
    if [ "$flashinfer64_batched_residual_select" = "True" ] || [ "$flashinfer64_batched_residual_select" = "true" ]; then
        flashinfer64_route_tag="${flashinfer64_route_tag}_batchedselectTrue"
    fi
    if [ "$flashinfer64_sampled_lse_gemm_dtype" != "fp32" ]; then
        flashinfer64_route_tag="${flashinfer64_route_tag}_sampledlsegemm${flashinfer64_sampled_lse_gemm_dtype}"
    fi
    if [ -n "$flashinfer64_residual_short_bucket_tuning" ]; then
        flashinfer64_bucket_tag="${flashinfer64_residual_short_bucket_tuning//:/-}"
        flashinfer64_bucket_tag="${flashinfer64_bucket_tag//,/x}"
        flashinfer64_route_tag="${flashinfer64_route_tag}_shortbuckets${flashinfer64_bucket_tag}"
    fi
    if [ -n "$flashinfer64_fused_qkv_permute_tuning" ]; then
        flashinfer64_permute_tag="${flashinfer64_fused_qkv_permute_tuning//:/-}"
        flashinfer64_route_tag="${flashinfer64_route_tag}_fusedqkvperm${flashinfer64_permute_tag}"
    fi
    if [ -n "$flashinfer64_fused_output_unpermute_tuning" ]; then
        flashinfer64_output_tag="${flashinfer64_fused_output_unpermute_tuning//:/-}"
        flashinfer64_route_tag="${flashinfer64_route_tag}_fusedoutperm${flashinfer64_output_tag}"
    fi
    default_output_dir="${res_root}/flashinfer64/${dataset_name}/${model_name}/seed${seed}_${flashinfer64_route_tag}_resbackend${flashinfer64_residual_backend}_rodecache${flashinfer64_rode_cache}_parallel${flashinfer64_parallel_core_residual}_${order}_${height}_routecache${flashinfer64_route_cache}_coreonly${flashinfer64_core_only}_directcsr${flashinfer64_direct_macro_csr}_ctamul${flashinfer_csr_expand_cta_multiplier}_cache${cache_interval}"
    if [ -n "${OUTPUT_DIR:-}" ]; then
        output_dir="${OUTPUT_DIR%/}"
        case "$output_dir" in
            *"ctamul${flashinfer_csr_expand_cta_multiplier}"*) ;;
            *) output_dir="${output_dir}_ctamul${flashinfer_csr_expand_cta_multiplier}" ;;
        esac
    else
        output_dir="$default_output_dir"
    fi
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
if [ "$prompt_set" = "66" ] && [ "$num_prompts" -ne 66 ]; then
    echo "ERROR: vbench66 prompt file must contain 66 lines, found $num_prompts: $prompt_file"
    exit 1
fi
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

    # Replay masks are prompt-specific and are consumed by the FlashInfer64
    # residual builder.  Each prompt is a separate Python process, so setting
    # the file just before launch keeps Oracle/Proxy/Random routes isolated.
    if [ -n "$replay_mask_dir" ]; then
        replay_mask_file="$replay_mask_dir/prompt_${prompt_idx}.json"
        if [ ! -f "$replay_mask_file" ]; then
            echo "ERROR: replay mask not found: $replay_mask_file"
            exit 1
        fi
        export FLASHINFER64_REPLAY_MASK_FILE="$replay_mask_file"
    else
        unset FLASHINFER64_REPLAY_MASK_FILE || true
    fi

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
        --flashinfer64_dynamic_tile_ratio "$flashinfer64_dynamic_tile_ratio"
        --flashinfer64_fine_top_ratio "$flashinfer64_fine_top_ratio"
        --flashinfer64_token_top_p "$flashinfer64_token_top_p"
        --flashinfer64_token_top_ratio "$flashinfer64_token_top_ratio"
        --flashinfer64_promotion_threshold "$flashinfer64_promotion_threshold"
        --flashinfer64_dense_layer "$flashinfer64_dense_layer"
        --flashinfer64_route_cache "$flashinfer64_route_cache"
        --flashinfer64_core_only "$flashinfer64_core_only"
        --flashinfer64_direct_macro_csr "$flashinfer64_direct_macro_csr"
    )
    if [ -n "$flashinfer64_dense_heads" ]; then
        flashinfer64_args+=(--flashinfer64_dense_heads "$flashinfer64_dense_heads")
    fi
    if [ -n "$flashinfer64_high_omission_heads_file" ]; then
        flashinfer64_args+=(--flashinfer64_high_omission_heads_file "$flashinfer64_high_omission_heads_file")
    fi
    flashinfer64_args+=(
        --flashinfer64_residual_scorer "$flashinfer64_residual_scorer"
        --flashinfer64_residual_temperature "$flashinfer64_residual_temperature"
        --flashinfer64_residual_min_top_k "$flashinfer64_residual_min_top_k"
        --flashinfer64_residual_max_top_k "$flashinfer64_residual_max_top_k"
    )
    if [ -n "$flashinfer64_fine_top_k_override" ]; then
        flashinfer64_args+=(--flashinfer64_fine_top_k "$flashinfer64_fine_top_k_override")
    fi
    attention_debug_args=()
    if [ -n "$attention_debug_dir" ]; then
        # A multi-prompt diagnostic run must keep one Q/K/V dump per video;
        # otherwise every prompt writes the same step/layer filename.
        attention_debug_prompt_dir="${attention_debug_dir}/prompt_${prompt_idx}"
        attention_debug_args=(
            --attention_debug_dir "$attention_debug_prompt_dir"
            --attention_debug_step "$attention_debug_step"
            --attention_debug_layers "$attention_debug_layers"
            --attention_debug_stop "$attention_debug_stop"
        )
        if [ -n "$attention_debug_steps" ]; then
            attention_debug_args+=(--attention_debug_steps "$attention_debug_steps")
        fi
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
        --record_density "$record_density" \
        --record_timing "$record_timing" \
        --timing_csv "${out_file%.mp4}_timing.csv" \
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
