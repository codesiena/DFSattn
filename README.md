# DFSAttn: Dynamic Fine-grained Sparse Attention for Efficient Video Generation

DFSAttn is a training-free sparse attention framework for efficient video
diffusion inference. It exploits dynamic and fine-grained sparsity in diffusion
transformers through 3D Hilbert token reordering, hierarchical block scoring,
and adaptive sparse mask caching, achieving up to 2.1x end-to-end speedup while
preserving generation quality.

[Paper](https://arxiv.org/abs/2605.23445) | [Examples](#examples) | [Installation](#installation) | [Citation](#citation)

## Highlights

- Training-free acceleration for video diffusion transformers.
- Fine-grained sparsity with GPU-friendly block-sparse execution.
- 3D Hilbert token reordering to preserve spatiotemporal locality.
- Hierarchical block scoring for more accurate sparse mask selection.
- Adaptive sparse mask caching across denoising steps.
- Supports HunyuanVideo-T2V-13B and Wan2.1-T2V-14B.

## What Is Included

```text
.
├── dfsattn/                    # DFSAttn attention processors and utilities
│   ├── attention_hyvideo.py     # DFS attention for HunyuanVideo
│   ├── attention_wan.py         # DFS attention for Wan
│   ├── replace_hyvideo.py       # HunyuanVideo attention replacement
│   ├── replace_wan.py           # Wan attention replacement
│   ├── fullattention.py         # Dense attention fallback
│   ├── utils/                   # Seed, logging, ordering, visualization helpers
│   └── kernels/                 # Optional CUDA/Triton acceleration kernels
├── hyvideo_t2v_inference.py     # HunyuanVideo text-to-video inference
├── wan21_t2v_inference.py       # Wan 2.1 text-to-video inference
├── *_720p_dfs.sh                # Batch DFSAttn launch examples
├── dataloader.py                # Prompt loading helpers
├── requirements.txt             # Reference Python dependencies
├── examples/prompts.txt         # Minimal prompt file for smoke tests
└── assets/examples/             # Example full-attention and DFSAttn videos
```

## Installation

The code targets Linux CUDA environments. The reference environment uses Python
3.10, PyTorch 2.6, CUDA 12.x, Diffusers, FlashAttention, and
`block_sparse_attn`.

### 1. Create Environment

Install the PyTorch/CUDA pair that matches your machine first, then install the
CUDA extension build tools:

```bash
conda create -n dfsattn python=3.10 -y
conda activate dfsattn
# Example only. Pick the PyTorch command that matches your CUDA runtime.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U setuptools wheel cmake ninja psutil packaging
```

### 2. Required: Block-Sparse-Attention

DFSAttn directly uses the block-sparse attention operator from
[MIT HAN Lab Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention).
Build it from the upstream repository after PyTorch is installed. `CUDA_HOME`
must point to the same CUDA toolkit version used by PyTorch; for example,
PyTorch `cu124` should use CUDA 12.4 `nvcc`.

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# A100: 80, A10/RTX 30xx: 86, H100/H200: 90. See the upstream repo for more architectures.
export BLOCK_SPARSE_ATTN_CUDA_ARCHS=80
pip install --no-build-isolation .
cd ..
```

### 3. Install DFSAttn Dependencies

```bash
pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned where Wan 2.1 depends on a
specific Diffusers/Transformers API. Upgrade those packages only after checking
the attention processor interfaces.

### 4. Optional: Fast QK-Norm and RoPE Kernels

DFSAttn can integrate the fast QK-Norm and RoPE CUDA/Triton kernels from
[Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen). These
kernels are optional, but recommended when reproducing the end-to-end speedups
reported in the paper. DFSAttn falls back to PyTorch implementations if they are
not built.

Prepare the kernel sources and third-party headers by following the upstream
Sparse-VideoGen customized-kernel setup first. This initializes the required
submodules and verifies that the upstream kernels build in your environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/svg-project/Sparse-VideoGen.git
cd Sparse-VideoGen
pip install -U setuptools
git submodule update --init --recursive
cd svg/kernels
pip install -U cmake
bash setup.sh
```

DFSAttn expects the initialized Sparse-VideoGen third-party layout at:

```text
dfsattn/kernels/3rdparty/
├── cutlass/
├── flashinfer/
└── pybind/
```

```bash
cd /path/to/DFSAttn
rm -rf dfsattn/kernels/3rdparty
cp -a /path/to/Sparse-VideoGen/svg/kernels/3rdparty dfsattn/kernels/3rdparty
```

Then build the DFSAttn extension:

```bash
cd dfsattn/kernels
bash setup.sh
python -c 'from dfsattn.kernels import ENABLE_FAST_KERNEL; print(ENABLE_FAST_KERNEL)'
```

## Model Checkpoints

Use either Hugging Face model IDs or local checkpoint directories:

```bash
export HYVIDEO_MODEL_ID=/path/to/HunyuanVideo
export WAN_MODEL_ID=/path/to/Wan2.1-T2V-14B
```

You can also pass `--model_id` directly to the inference scripts.

## Single Prompt Inference

Run HunyuanVideo with DFSAttn:

```bash
python hyvideo_t2v_inference.py \
  --model_id "$HYVIDEO_MODEL_ID" \
  --prompt "A cinematic city street at night with colorful neon lights" \
  --output_file output/hyvideo_dfs.mp4 \
  --mode dfs \
  --sparsity 0.3 \
  --tile_size 16 \
  --block_size 128 \
  --order hilbert3d \
  --skip_steps 12 \
  --cache_interval 12 \
  --sparsity_dcrt 0.1
```

Run Wan 2.1 with DFSAttn:

```bash
python wan21_t2v_inference.py \
  --model_id "$WAN_MODEL_ID" \
  --prompt "A cat walks on the grass, realistic" \
  --output_file output/wan_dfs.mp4 \
  --mode dfs \
  --sparsity 0.3 \
  --tile_size 16 \
  --block_size 128 \
  --order hilbert3d \
  --skip_steps 12 \
  --cache_interval 12 \
  --sparsity_dcrt 0.1
```

## Export Top-k Block Masks

Pass `--block_mask_dir` to either inference script to save the block matrix used
by block-sparse attention after top-k selection:

```bash
python wan21_t2v_inference.py \
  --model_id "$WAN_MODEL_ID" \
  --prompt "A cat walks on the grass, realistic" \
  --output_file output/wan_dfs.mp4 \
  --mode dfs \
  --block_mask_dir output/wan_block_masks \
  --block_mask_heads 0,1 \
  --block_mask_layer_interval 15
```

Masks are exported only when top-k is recomputed, not when a cached mask is
reused. Each refresh produces files such as:

```text
output/wan_block_masks/
└── prompt_000_a_cat_walks_on_the_grass_realistic/
    └── step_024/layer_15/
        ├── head_00_block_mask.png
        └── head_01_block_mask.png
```

The prompt directory always includes its zero-based prompt number and, when
possible, a filesystem-safe excerpt of the prompt text. This keeps masks from
different prompts separate and identifiable.

Only PNG heatmaps are saved by default, for layers `0, 15, 30, ...`. PNGs use
white for unselected blocks and blue for selected blocks. The default renders
head 0; use `--block_mask_heads all` to render every head. Coordinates refer to
the reordered block matrix actually passed to the sparse attention kernel. Use
`--block_mask_layer_interval 1` to export every layer, or
`--block_mask_save_bool true` if the raw bool array is also needed.

```python
import numpy as np

# Only available when --block_mask_save_bool true is used.
mask = np.load(
    "output/wan_block_masks/prompt_000_a_cat_walks_on_the_grass_realistic/"
    "step_024/layer_15/block_mask.npy"
)
print(mask.dtype, mask.shape)
print(np.argwhere(mask[0]))  # [query_block, key_block] for head 0
```

Compare one head at several diffusion steps while advancing through layers:

```bash
python make_block_mask_video.py \
  --mask_dir output/hyvideo_masks/prompt_000_gwen_stacy_reading_a_book_surrealism_style \
  --steps 12 24 36 \
  --head 0 \
  --fps 4 \
  --output output/hyvideo_masks/head_00_steps_12_24_36_by_layer.mp4
```

## Batch Inference

The launch scripts are configured through environment variables and do not depend on local absolute paths.

```bash
CUDA_VISIBLE_DEVICES=0 \
HYVIDEO_MODEL_ID=/path/to/HunyuanVideo \
PROMPT_FILE=examples/vbench_11_prompts.txt \
OUTPUT_DIR=output/hyvideo/dfs \
BLOCK_MASK_DIR=output/hyvideo/masks \
./hyvideo_t2v_720p_dfs.sh
```

Run coarse Top-k followed by local `16x16` sub-block Top-p (`kp` mode):

```bash
SELECTOR_MODE=kp \
FINE_TOP_P=0.9 \
START_IDX=0 \
END_IDX=1 \
bash hyvideo_t2v_720p_dfs.sh
```

`SPARSITY` continues to control the upstream `128x128` coarse Top-k budget.
Within every selected coarse block, KP jointly ranks its 64 fine scores and
keeps the smallest set whose cumulative score reaches `FINE_TOP_P`.  KP uses
the Q-stationary Triton backend automatically; its current supported geometry
is `BLOCK_SIZE=128` and `TILE_SIZE=16`.

```bash
CUDA_VISIBLE_DEVICES=0 \
WAN_MODEL_ID=/path/to/Wan2.1-T2V-14B \
PROMPT_FILE=examples/vbench_11_prompts.txt \
OUTPUT_DIR=output/wan/dfs \
BLOCK_MASK_DIR=output/wan/masks \
./wan21_t2v_720p_dfs.sh
```

Useful script variables:

- `PROMPT_FILE`: text file with one prompt per line.
- `OUTPUT_DIR`: directory for generated videos.
- `START_IDX` and `END_IDX`: inclusive prompt index range.
- `SPARSITY`, `SKIP_STEPS`, `CACHE_INTERVAL`, `SPARSITY_DCRT`, `TILE_SIZE`, `BLOCK_SIZE`, `ORDER`: DFSAttn parameters.
- `SELECTOR_MODE`: `topk` (default) or coarse-Top-k/fine-Top-p `kp`.
- `FINE_TOP_P`: fine cumulative mass for `kp` (default: `0.9`).
- `SPARSE_EXECUTION=flashinfer64`: hardware-aligned hierarchical routing. The paper path aggregates Q16xK16 fine scores into Q128xK96 macro scores, selects Core by ratio Top-k, and sends it through FlashInfer. Residual searches every non-Core Q16xK16 tile and uses its original, unnormalized mass only to bring Core+Residual coverage up to `FLASHINFER64_TOKEN_TOP_P`. `FLASHINFER64_PROMOTION_THRESHOLD` promotes a macro tile when enough of its 48 microtiles are active.
- `FLASHINFER64_ROUTE_CACHE=False` (default): rebuild and release the compact route on every sparse call. Set it to `True` only when route reuse is wanted; the expanded FlashInfer plan is still rebuilt every call and is never cached per layer.
- `HEIGHT`, `WIDTH`, `NUM_FRAMES`, `NUM_INFERENCE_STEPS`, `SEED`: generation settings.
- `BLOCK_MASK_DIR`: enable top-k mask export; inference creates a `prompt_NNN_<prompt-text>` subdirectory.
- `BLOCK_MASK_HEADS`: comma-separated heads to render, or `all` (default: `0`).

## Default Parameters

| Argument | Default | Description |
|---|---:|---|
| `SPARSITY` / `--sparsity` | `0.3` | Initial sparsity budget after skipped denoising steps. |
| `SKIP_STEPS` / `--skip_steps` | `12` | Number of early denoising steps using full attention. |
| `CACHE_INTERVAL` / `--cache_interval` | `12` | Sparse mask refresh interval. |
| `SPARSITY_DCRT` / `--sparsity_dcrt` | `0.1` | Sparsity decrease after each refresh. |
| `TILE_SIZE` / `--tile_size` | `16` | Token grouping granularity for hierarchical scoring. |
| `BLOCK_SIZE` / `--block_size` | `128` | Block size used by block-sparse attention. |
| `ORDER` / `--order` | `hilbert3d` | Token ordering strategy. |

The independent `flashinfer64` backend does not reuse the historical
Hybrid/KP route. It scores Q16/K16 interactions, aggregates them into Q128/K96
macro tiles, sends Core macro tiles to FlashInfer, and packs selected complement
microtiles into CSR length buckets for grouped Triton MMA. Empty Q16 rows are
not launched and no row is padded to the global Residual maximum. Dense-enough
residual regions are promoted to Core, and the disjoint states are merged with
exact LSE normalization. Following SVG's bounded-memory lifecycle, all layers
share one FlashInfer wrapper and each call replans into it, overwriting the
previous expanded plan. No layer retains an expanded plan or a full CUDA
Residual boolean mask. Compact route reuse is optional and is disabled by
default for initial bring-up. It requires CUDA and a FlashInfer build exposing
`VariableBlockSparseAttentionWrapper`; the CPU path is a correctness reference.

DFSAttn runs full attention for the first `SKIP_STEPS` diffusion steps. After
that, it starts from `SPARSITY` and refreshes the sparse mask every
`CACHE_INTERVAL` diffusion steps, decreasing sparsity by `SPARSITY_DCRT` each
interval. If a scheduled refresh would make sparsity non-positive, DFSAttn skips
that refresh and keeps using the previous cached mask and sparsity.

## Diagnose Macro Top-k Errors

To test whether coarse Macro Top-k misses a few concentrated Q16/K16
interactions, first make a same-input attention dump using only the fixed Core
route:

```bash
FLASHINFER64_ROUTE_MODE=topk_topp \
FLASHINFER64_TILE_TOP_RATIO=0.2 \
FLASHINFER64_TOKEN_TOP_P=0 \
FLASHINFER64_CORE_ONLY=True \
SPARSE_EXECUTION=flashinfer64 \
ATTENTION_DEBUG_DIR=../../res/attention_debug/core_topk02 \
ATTENTION_DEBUG_LAYERS=0 \
bash hyvideo_t2v_720p_dfs.sh
```

The dump from this run is the fixed `128x96` Macro Top-k baseline. Analyze the
highest-error Q128 regions and add back every omitted macro in those regions:

```bash
PYTHONPATH=. python analyze_macro_topk_error.py \
  --input-dump ../../res/attention_debug/core_topk02/step012_layer000_flashinfer64_topk_topp.pt \
  --base-dump ../../res/attention_debug/core_topk02/step012_layer000_flashinfer64_topk_topp.pt \
  --output-dir ../../res/macro_topk_error/core_topk02 \
  --macro-top-ratio 0.2 \
  --order hilbert3d \
  --height 480 --width 720 --num-frames 129 \
  --q-macro-count 32
```

The analyzer writes `query_errors.csv` (per-query `e_q`),
`q16_error_summary.csv`, `macro_addback.csv`, `top_addback_macros.csv`,
`summary.json`, and the key plot `macro_addback_entropy_max.png`. Each row in
`macro_addback.csv` is one `(head, Q128 macro, omitted K96 macro)` and reports
`M_g`, `A_g`, normalized `H_g`, `C_g`, plus exact one-macro error recovery.
The add-back uses the changed softmax denominator, so positive recovery is
evidence that the omitted macro itself fixes the local error. Use
`--q-macro-count 0` to run all Q128 regions; the default of 8 keeps the CPU
fallback practical for quick diagnosis.

For multiple videos, the launch script automatically places dumps under
`ATTENTION_DEBUG_DIR/prompt_<idx>/` so prompts do not overwrite one another.
After collecting prompts 0--2, aggregate them with:

```bash
PYTHONPATH=. python analyze_macro_topk_error_batch.py \
  --input-root ../../res/attention_debug/core_topk02 \
  --output-dir ../../res/macro_topk_error/core_topk02_batch \
  --macro-top-ratio 0.2 --q-macro-count 8
```

## Examples

<table>
  <tr>
    <th width="12%"></th>
    <th width="29%"></th>
    <th width="29%"></th>
    <th width="29%"></th>
  </tr>
  <tr>
    <th valign="top">Prompt</th>
    <td valign="top"><small>On the beach, the waves gently lap against the shore. Some people are sunbathing, surfers are sliding on the waves, and children are building sandcastles. The entire video presents a joyful atmosphere.</small></td>
    <td valign="top"><small>A female student in a gray coat slowly stands up in the rain. The entire video presents a melancholic atmosphere.</small></td>
    <td valign="top"><small>Under the azure sky, a polar bear stands in the snow, turning its head to look at its cub behind him.</small></td>
  </tr>
  <tr>
    <th valign="top">Full Attention</th>
    <td valign="top"><a href="assets/examples/dense_1.mp4"><img src="assets/examples/gifs/dense_1.gif" alt="Full attention beach example" width="256"></a></td>
    <td valign="top"><a href="assets/examples/dense_2.mp4"><img src="assets/examples/gifs/dense_2.gif" alt="Full attention rain example" width="256"></a></td>
    <td valign="top"><a href="assets/examples/dense_3.mp4"><img src="assets/examples/gifs/dense_3.gif" alt="Full attention polar bear example" width="256"></a></td>
  </tr>
  <tr>
    <th valign="top">DFSAttn</th>
    <td valign="top"><a href="assets/examples/dfs_1.mp4"><img src="assets/examples/gifs/dfs_1.gif" alt="DFSAttn beach example" width="256"></a></td>
    <td valign="top"><a href="assets/examples/dfs_2.mp4"><img src="assets/examples/gifs/dfs_2.gif" alt="DFSAttn rain example" width="256"></a></td>
    <td valign="top"><a href="assets/examples/dfs_3.mp4"><img src="assets/examples/gifs/dfs_3.gif" alt="DFSAttn polar bear example" width="256"></a></td>
  </tr>
</table>

## Citation

If you find DFSAttn useful for your research, please cite:

```bibtex
@article{hu2026dfsattn,
  title={DFSAttn: Dynamic Fine-grained Sparse Attention for Efficient Video Generation},
  author={Hu, Jie and Gao, Zixiang and He, Yutong and Yuan, Kun},
  journal={arXiv preprint arXiv:2605.23445},
  year={2026}
}
```

## Acknowledgements

- DFSAttn uses the block-sparse attention operator from [MIT HAN Lab Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), which exposes the `block_sparse_attn_func` interface. Please preserve the upstream BSD-3-Clause license notice when redistributing adapted code.
- For end-to-end acceleration, DFSAttn can integrate fast QK-Norm and RoPE kernels from [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen), which provides customized kernels for faster video diffusion inference.

## License

Please follow the license terms of this repository and the upstream projects
listed above.
