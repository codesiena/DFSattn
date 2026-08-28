# Hybrid-DFSAttn initial experiment

`dfsattn/hybrid_block_attention.py` implements the first Go/No-Go experiment
from `expdual.md` without changing DFSAttn's selector.  It requires the DFS
logical mask to use 16x16 microblocks.

For each 4x4 group of microblocks, an occupancy at least `n*` is promoted to a
64x64 Core tile.  The full 64x64 QK product is evaluated, but each unselected
16x16 microblock remains masked to `-inf` before softmax.  Low-occupancy groups
remain as 16x16 residual tiles.  The two paths produce `(max, normalizer,
unnormalized value)` partial states that are merged using online softmax; their
independent softmax outputs are never added directly.

On CUDA with BF16/FP16 QKV, Core tiles use the Triton 64x64 masked kernel in
`dfsattn/kernels/triton/hybrid_core.py`; its QK dot product is Tensor-Core
eligible and returns `(m_c, l_c, a_c)`.  The installed upstream
`block_sparse_attn` extension requires block dimensions that are multiples of
128, so 16x16 residual tiles use the local tiled reference backend instead.
A separate Triton merge kernel performs the joint LSE merge.

The current promotion condition and default threshold are intentionally left
unchanged (`n_B >= hybrid_threshold`, default `8`); this change only replaces
the execution kernels and makes the mask cache identity include its block/tile
granularity.

Run the deterministic correctness test:

```bash
python idea2/test_hybrid_attention.py
```

Run the threshold sweep on a CUDA host (the table separately includes mask
partition planning and Core+Residual+LSE-merge execution):

```bash
python idea2/benchmark_hybrid_attention.py \
  --device cuda --seq-len 4096 --heads 16 --head-dim 128 \
  --dtype bfloat16 --thresholds 4 6 8 10 12 14
```

`fine16_reference` is a same-mask PyTorch reference backend, not the existing
`block_sparse_attn` CUDA extension.  Use it first to find the occupancy
crossover and validate the merge.  The subsequent Go/No-Go comparison must
also time the native DFSAttn extension on identical captured Q/K/V/masks.

To benchmark a captured 16x16 logical mask, save it as `bool [heads, q16,
k16]` NumPy data and pass `--mask path/to/block_mask.npy`.  Do not pass this
repository's historical 128x128 masks: they are a different selector
granularity and cannot establish the same-mask claim.

For an integration run, select the backend explicitly:

```bash
python hyvideo_t2v_inference.py ... --mode dfs --tile_size 16 --block_size 16 \
  --sparse_execution hybrid --hybrid_threshold 8
```

`native` remains the default.  The hybrid backend currently supports the
non-causal, no-dropout, no-extra-token-mask video-attention path only.

## Run the real VBench prompts

The same Python entry point also has a video mode.  It reads all 11 lines from
`examples/vbench_11_prompts.txt` and writes `0.mp4` ... `10.mp4`, one timing CSV
per prompt, and exported masks under
`/cnic/work/liutt/mywork/attention_time/res/dual`:

```bash
python idea2/benchmark_hybrid_attention.py --run-vbench \
  --backend wan --output-dir /cnic/work/liutt/mywork/attention_time/res/dual \
  --skip-existing
```

Set `WAN_MODEL_ID` (or pass `--model-id`) for Wan.  Use
`--backend hyvideo` with `HYVIDEO_MODEL_ID` for HunyuanVideo.  To test one
prompt first, add `--start-idx 0 --end-idx 0`.

## Inference timing CSV

Both inference scripts enable `--record_timing true` by default and save one
video-level summary CSV beside the output video (or to `--timing_csv PATH`).
The file has only `phase,total_ms` rows, so it does not contain per-step or
per-layer details:

- `top_k_selection`: sum of all DFS `topk_mask` events (actual mask/Top-k
  computation; cached-mask calls contribute zero);
- `attention_execution`: sum of all attention execution events;
- `e2e_generation_wall`: synchronized wall-clock time for the complete
  `pipe(...)` generation call;
- `e2e_generation_gpu`: CUDA-event stream time for that call, when available.

Model loading and video encoding are excluded from the end-to-end generation
measurement.
