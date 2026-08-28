# Tensor-Tile-Cover v0：Wan / Triton 64 实验记录

本文档维护当前可运行实现、指标定义和实验缺口。本文中的 “kernel” 指的是
`flex64` execution backend；当前生产路径是直接 Triton 的 Q64/K64 online-softmax
kernel（`BlockMask` 仍保留用于 mask/debug 元数据），用于验证 64×64 execution
tile 的端到端可行性。它还不是最终手写 WGMMA kernel，Tensor-Core 指令映射需要
Nsight Compute 进一步确认。

## 当前实现

实现位于：

- `dfsattn/flex64_attention.py`：把每个 Q64 row 的 top-k K64 IDs 转为
  `BlockMask`，并由 Triton kernel 按 `BLOCK_M=BLOCK_N=64` 执行 online softmax。
- `dfsattn/attention_wan.py`：DFS selector 可直接生成 64×64 mask 和
  top-k IDs，并缓存 mask、IDs 与 `BlockMask`；`sparse_execution=flex64`
  时 Q/K/V 在 Hilbert 顺序下执行 Q64/K64 Triton attention，输出再 inverse-permute。
- `wan21_t2v_inference.py`：暴露 `--sparse_execution flex64`，并限制它
  必须与 `--block_size 64` 一起使用。

本版仍是 **Ours-v0**：selector 复用 DFSAttn 的 pooled QK score，直接聚合到
64×64 payload 并对每个 Q64 row 选 top-k K64 tiles。尚未实现 pair-level
quantized top-p scorer，也尚未实现 Q/K packing permutation。

## 建议的首次运行

```bash
cd /cnic/work/liutt/mywork/attention_time/code/DFSAttn

python wan21_t2v_inference.py \
  --model_id "$WAN_MODEL_ID" \
  --mode dfs \
  --sparse_execution flex64 \
  --block_size 64 \
  --tile_size 16 \
  --sparsity 0.3 \
  --order hilbert3d \
  --skip_steps 12 \
  --cache_interval 12 \
  --prompt "A cat walks on the grass, realistic" \
  --output_file output/wan_flex64.mp4 \
  --record_density true \
  --record_timing true
```

### GH200 480p 命令

Wan 480p 通常使用 `height=480, width=832`。在实际 GH200 节点上，将
`WAN_MODEL_ID` 设置为本地 Wan2.1-T2V-14B-Diffusers 权重目录后运行：

```bash
cd /cnic/work/liutt/mywork/attention_time/code/DFSAttn
export WAN_MODEL_ID=/path/to/Wan2.1-T2V-14B-Diffusers

python wan21_t2v_inference.py \
  --model_id "$WAN_MODEL_ID" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --num_inference_steps 50 \
  --mode dfs \
  --sparse_execution flex64 \
  --block_size 64 \
  --tile_size 16 \
  --sparsity 0.3 \
  --order hilbert3d \
  --skip_steps 12 \
  --cache_interval 12 \
  --record_density true \
  --record_timing true \
  --output_file output/wan_480p_flex64.mp4 \
  --timing_csv output/wan_480p_flex64_timing.csv
```

GH200 实验使用 `wan21_t2v_720p_dfs.sh` 的真实配置：`SPARSITY=0.3`、
`HEIGHT=480`、`WIDTH=832`、`NUM_FRAMES=81`、`NUM_INFERENCE_STEPS=50`、
`SKIP_STEPS=12`、`CACHE_INTERVAL=12`。输出目录为
`/cnic/work/liutt/mywork/attention_time/res/kkknernal`。第一次直接 Triton
编译产生的 warm-up 时间应从性能结论中单独标注。

首次 Triton 调用会触发 kernel 编译；本次三组样本均包含首次编译开销，正式论文
数据应另行做同 shape warm-up 后再重复测量。

输出文件：

- `output/wan_flex64.mp4`：生成视频；
- `output/density_records.csv`：每个 sparse step/layer 的 execution-tile 记录；
- `output/timing.csv`：全视频聚合的 selector、attention execution 和 E2E 时间；
- 若传入 `--block_mask_dir`：导出 bool mask / heatmap。

## 表格指标：当前实现状态

| 表格列 | 当前状态 | 定义 / 保存位置 | 说明 |
| --- | --- | --- | --- |
| Pair \(p\) | 未实现 | 无 | 本版没有 pair-level top-p；`--sparsity` 不是 p。 |
| Tile \(k\) | 已计算并保存 | `density_records.csv`: `tile_k_mean/min/max` | 每个 Q64 row 选择的 K64 tile 数。flex64 selector 中所有 row 的 k 相同，因此三者通常相等。 |
| HW density | 已计算并保存 | `density_records.csv`: `tiles_executed`, `tiles_dense`, `execution_tile_density` | flex64 下严格为 \(D_{HW}=N_{executed\;64\times64}/N_{dense\;64\times64}\)。最后一块 padding 仍按一个执行 tile 计数。 |
| Attention recall | 未实现 | 无 | 需要 dense attention mass/reference trace；不能由当前 pooled score 或 mask density 替代。 |
| Attn latency | 已计算并保存 | `timing.csv`: `attention_execution` | 包括 Q/K/V permutation、Triton Q64/K64 attention 和 inverse permutation；不含 selector，selector 单列为 `top_k_selection`。 |
| E2E latency | 已计算并保存 | `timing.csv`: `e2e_generation_gpu`, `e2e_generation_wall` | GPU 项包括整次 pipeline GPU 时间；wall 项包括 CPU/编码等开销。 |
| Video quality | 已做 fidelity sanity check | `quality_vs_full.csv`、`summary_vbench11.csv`: PSNR/SSIM | 使用本次 Full 输出作为同 prompt/seed reference；尚未运行 VBench 语义质量模型。 |

## GH200 480p 实测（VBench11 prompt 0--2）

运行日期为 2026-08-25，机器为 NVIDIA GH200 480GB；三组实验均使用
`SPARSITY=0.3`、`HEIGHT=480`、`WIDTH=832`、81 帧、50 steps、seed 和 prompt
保持一致。每个样本都保存了 MP4；sparse 两组另保存 density CSV，三组均保存 timing CSV。

| Method | block/tile | mean tile k | mean (D_{HW}) | Attn latency (s) | E2E GPU latency (s) | selector (s) | PSNR / SSIM vs Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full dense | dense | — | 100% | N/A | 477.80 | N/A | reference |
| DFSAttn native | 128 | 49.32 | 19.26% | 51.59 | 318.21 | 0.627 | 20.70 / 0.7275 |
| Ours-64 Triton | 64 | 99.32 | 19.40% | 117.21 | 393.58 | 2.068 | 20.82 / 0.7308 |

结果目录：

- `res/kkknernal/dfsattn_480p_vbench11/`
- `res/kkknernal/ours64_480p_vbench11/`
- `res/kkknernal/full_480p_vbench11/`

这里的 density 是实际执行 tile 数除以 dense tile 数，按 CSV 行逐条统计后取均值；
不是命令行 `--sparsity` 的补数。当前结果显示 Ours-64 的实际 (D_{HW}) 与 native
几乎相同，但在这版直接 Triton kernel 上 attention/E2E 更慢，说明仅降低 tile
边长并不能自动带来加速，下一步需要针对 CTA wave、tile packing 和 WGMMA 映射做
kernel 优化。PSNR/SSIM 是对当前 Full 输出的逐帧 fidelity sanity check（不是
VBench 语义质量分数）；Full 均值为三个样本的 E2E 均值，Attn latency 对 dense
路径没有单独 instrumentation。Pair (p) 和 attention recall 仍未计算。

## 当前实验可得出的结论

当前实验能够比较：在相同 DFS pooled-score budget 下，128×128 native mask 与
64×64 Triton mask 的 execution-tile density、selector 时间、attention execution
时间与 E2E 时间；同时产出后续离线质量评测所需的视频。

当前实验**不能**声称 pair-level top-p recall 更高，也不能声称已优化 Q/K
packing；这两项是下一版工作。PSNR/SSIM 仅用于与 Full 的 fidelity sanity check，
不等价于 VBench 语义质量。

## 下一版最小增量

1. 记录 dense/streamed attention mass，定义 selected-tile recall：
   \[R=\sum_{(i,j)\in M}A_{ij}/\sum_{i,j}A_{ij}.\]
2. 实现低成本的 pair-level score / top-p threshold，独立记录 p 与 k。
3. 在固定 k 下加入 Q/K atom packing permutation，报告 tile cover / recall 改善。
4. 使用 H100 Nsight Compute 验证 Tensor Pipe、CTA wave 与实际 kernel 时间；
   当前 Triton 结果只作为 v0 系统验证，不应直接等同于最终 WGMMA kernel 数字。
