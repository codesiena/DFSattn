# FlashInfer64 / RoDe / 原始 DFSAttn 时间对比

## 1. 实验设置

本次使用同一个 VBench prompt（prompt 0）：

- 分辨率：`480×720`
- 视频帧数：`129`
- 推理步数：`50`
- dense warmup：前 `12` 步
- 稀疏步数：`38`
- 层数：`60`
- `cache_interval=12`
- FlashInfer64 参数：`tile ratio=0.16`、`total top-p=0.5`
- FlashInfer64：`FLASHINFER64_CORE_ONLY=False`
- FlashInfer64 与 RoDe 对照均使用：`FLASHINFER64_ROUTE_CACHE=False`

三组配置为：

1. 原始 DFSAttn：脚本默认 `SPARSE_EXECUTION=native`、`SELECTOR_MODE=topk`。
2. FlashInfer64 + 原有 `micro` residual。
3. FlashInfer64 + `rode_center` residual。

原始 DFSAttn 实验使用新的输出目录，实际处理了 prompt 0，没有因为已有 mp4
而跳过。RoDe 扩展已提前编译，因此下面的 RoDe 结果不包含首次 CUDA 扩展编译
时间。

## 2. 端到端结果

| 配置 | GPU/e2e 时间 | 外部 wall time | 相对原始 DFSAttn | 相对 FlashInfer64 micro |
|---|---:|---:|---:|---:|
| 原始 DFSAttn | `200.957 s` | `214.23 s` | — | — |
| FlashInfer64 + micro | `288.659 s` | `302.12 s` | `+43.64%` | — |
| FlashInfer64 + rode_center | `294.446 s` | `308.06 s` | `+46.52%` | `+2.00%` |

对应文件：

- 原始 DFSAttn：[dfsattn_native_prompt0/timing.csv](../res/rode_ablation/dfsattn_native_prompt0/timing.csv)
- FlashInfer64 micro：[tile016_topp05_micro_ctamul0/timing.csv](../res/rode_ablation/tile016_topp05_micro_ctamul0/timing.csv)
- FlashInfer64 RoDe：[tile016_topp05_rode_ctamul0/timing.csv](../res/rode_ablation/tile016_topp05_rode_ctamul0/timing.csv)

当前端到端结论是：`rode_center` 没有带来加速，反而比 FlashInfer64 的
micro residual 慢约 `5.787 s`。更重要的是，当前两种 FlashInfer64 路径都
明显慢于原始 DFSAttn。

外部 wall time 与程序内部 GPU/e2e time 的差值约为 `13–14 s`，主要对应模型
加载、视频导出等非 attention 阶段。比较 attention 优化时应优先使用
`e2e_generation_gpu`，外部 wall time 作为端到端复核。

## 3. 为什么不能直接比较 `attention_execution`

三个配置中 `attention_execution` 的计时范围并不相同。

原始 DFSAttn 在
`dfsattn/attention_hyvideo.py` 中只将真正的
`_run_block_sparse_attention` 包在 `attention_execution` 内；Top-k mask
选择和 routing 是独立阶段。

FlashInfer64 的 `attention_execution` 则包住整个 backend 调用，包括：

```text
QKV permute
fine score
Core/Residual selection
CSR compact
FlashInfer plan
Core kernel
Residual kernel
LSE merge
output unpermute
```

所以日志中的：

```text
原始 DFSAttn attention_execution:  88.622 s
FlashInfer64 micro:                178.391 s
FlashInfer64 rode_center:          184.400 s
```

不能直接解释为 FlashInfer64 的 attention 数学 kernel 比原始 DFSAttn
计算量更多。这个差异主要包含了不同的计时边界和 FlashInfer64 额外的路由、
plan、格式转换及 merge 开销。

原始 DFSAttn 的 `timing.csv` 还只汇总了 `topk_mask` 到
`top_k_selection`，没有把 `topk_routing` 外层阶段作为独立汇总项；因此
原始路径的内部 phase 统计不适合作为完整端到端时间。

## 4. 实际 Core/Residual kernel 对比

FlashInfer64 micro 路径的主要执行阶段：

| 阶段 | 总时间 |
|---|---:|
| `flashinfer_core_run` | `15.915 s` |
| `flashinfer_residual_micro_run` | `15.857 s` |
| `flashinfer_lse_merge` | `0.078 s` |
| 合计 | `31.850 s` |

RoDe 路径：

| 阶段 | 总时间 |
|---|---:|
| `flashinfer_core_run` | `15.775 s` |
| `flashinfer_residual_rode_run` | `21.526 s` |
| `flashinfer_lse_merge` | `0.342 s` |
| 合计 | `37.643 s` |

但 `flashinfer_residual_rode_run` 中包含 plan、FP32 转换、RoDe SDDMM、
softmax 和 SpMM，不能只把它当作 RoDe 数学计算时间。

RoDe residual 内部拆分如下：

| 阶段 | 总时间 | 平均每次调用 |
|---|---:|---:|
| `flashinfer_residual_rode_plan` | `13.538 s` | `5.940 ms` |
| `flashinfer_residual_rode_fp32` | `2.751 s` | `1.207 ms` |
| `flashinfer_residual_rode_sddmm` | `0.541 s` | `0.237 ms` |
| `flashinfer_residual_rode_softmax` | `0.429 s` | `0.188 ms` |
| `flashinfer_residual_rode_spmm` | `1.111 s` | `0.487 ms` |

RoDe 的实际 FP32 转换、SDDMM、softmax、SpMM 合计约 `4.832 s`；其中
SDDMM+SpMM 本身很快。主要问题是 `RoDeCenterPlan` 被构建了 `2279` 次，
plan 构建总计 `13.538 s`，占 RoDe residual 总阶段的大部分。

因此目前的结论是：

> RoDe 的 residual 数学 kernel 比 micro residual 更有潜力，但当前每次
> attention 重建 plan，plan 开销抵消了 kernel 加速。

## 5. 路由和密度统计

FlashInfer64 两组的 residual workload 基本一致：

| 指标 | micro | rode_center |
|---|---:|---:|
| residual count mean | `23.790` | `23.853` |
| residual count p50 | `15.982` | `16.028` |
| residual count p95 | `79.187` | `79.429` |
| residual count max | `358.122` | `358.668` |
| residual nonempty ratio | `0.1372` | `0.1378` |

所以 RoDe 变慢不是因为它处理了更多 residual，而是因为 backend 实现开销
更高。两组运行路径会因为 RoDe 只计算 residual tile 的中心 token 而产生
不同的后续 hidden states，因而后续 step 的 route 统计有轻微漂移，但本次
workload 差异很小。

原始 DFSAttn 的平均最终 density 为约 `19.67%`，FlashInfer64 两组约为
`17.61%`。原始 DFSAttn 实际保留的 QK 支持还更多，却仍然更快，说明原始
DFSAttn 的 kernel 和执行调度效率明显更高；FlashInfer64 不能简单依靠更低
density 获得端到端优势。

## 6. 当前结论与下一步

当前实验可以得出三点：

1. FlashInfer64 的 Core/Residual 数学执行阶段本身比原始 DFSAttn 的大块
   稀疏执行更有潜力，但路由、CSR、plan 和数据重排开销很大。
2. `rode_center` 的 SDDMM/SpMM 本身不是主要瓶颈；当前最主要的问题是
   `RoDeCenterPlan` 每次调用都重新构建。
3. 在当前 `FLASHINFER64_ROUTE_CACHE=False` 设置下，RoDe 端到端没有收益。

下一步应优先实现独立的轻量级 RoDe residual cache，而不是直接打开完整的
`FLASHINFER64_ROUTE_CACHE`：

```text
保留：residual CSR、active rows、匹配的 core support、RoDe GPU plan
释放：Q/K/V、dense residual mask、micro bucket、非必要统计结构
```

缓存应按照现有 `cache_interval=12` 的 refresh step 更新，在中间 step 复用。
这样可以在控制峰值显存的同时，把 plan 调用次数从约 `2279` 降到约
`240` 次量级，再重新判断 RoDe 是否能超过 micro residual。

## 7. RoDe cache + Core/Residual 并行复测

随后启用：

```bash
FLASHINFER64_ROUTE_CACHE=False
FLASHINFER64_RODE_CACHE=True
FLASHINFER64_RESIDUAL_BACKEND=rode_center
FLASHINFER64_PARALLEL_CORE_RESIDUAL=True
```

其它参数仍为 `tile ratio=0.16`、`total top-p=0.5`、prompt 0、50 steps。

本次输出目录为：

```text
res/rode_ablation/tile016_topp05_rodecacheTrue_parallelTrue_prompt0_ctamul0/
```

结果：

| 配置 | GPU/e2e 时间 | 外部 wall time |
|---|---:|---:|
| RoDe，无 cache、无并行 | `294.446 s` | `308.06 s` |
| RoDe cache + 并行 | `649.527 s` | `662.98 s` |

新配置比之前的 RoDe 配置慢：

```text
GPU/e2e:  +355.081 s，+120.59%
wall:     +354.92 s，+115.21%
```

RoDe cache 本身已经生效：

```text
RoDe plan 调用：2279 次 → 240 次
RoDe plan 时间：13.538 s → 1.426 s
```

但是并行配置下 residual 计算出现异常膨胀：

| RoDe phase | 无 cache、无并行 | cache + 并行 |
|---|---:|---:|
| `flashinfer_residual_rode_fp32` | `2.751 s` | `310.768 s` |
| `flashinfer_residual_rode_sddmm` | `0.541 s` | `2.024 s` |
| `flashinfer_residual_rode_softmax` | `0.429 s` | `5.260 s` |
| `flashinfer_residual_rode_spmm` | `1.111 s` | `105.470 s` |
| `flashinfer_residual_rode_run` | `21.526 s` | `428.739 s` |

因此这次结果不能解释为 RoDe cache 无效；相反，cache 已经显著减少了
route/plan 开销。问题集中在当前 Core/Residual 双 CUDA stream 实现：Core
和 Residual 没有获得有效重叠，且跨 stream 的同步、allocator dependency 或
资源竞争让 FP32 转换和 SpMM 大幅退化。`flashinfer_core_run` 基本不变
（`15.775 s → 15.612 s`），说明主要异常发生在 Residual stream。

当前结论：

1. `FLASHINFER64_RODE_CACHE=True` 的轻量缓存方向是有效的；
2. `FLASHINFER64_PARALLEL_CORE_RESIDUAL=True` 当前实现不可用于性能结论，
   需要先单独关闭并行复测 `RODE_CACHE=True`；
3. 在修复 stream/allocator 同步前，不应把这次 `649.527 s` 作为 RoDe
   算法本身的性能结果。

## 8. RoDe cache-only 复测（关闭 Core/Residual 并行）

为隔离 RoDe cache 的作用，本次使用与前面相同的 prompt、分辨率、50 steps、
`tile ratio=0.16`、`total top-p=0.5`，但设置为：

```bash
FLASHINFER64_ROUTE_CACHE=False
FLASHINFER64_RODE_CACHE=True
FLASHINFER64_RESIDUAL_BACKEND=rode_center
FLASHINFER64_PARALLEL_CORE_RESIDUAL=False
```

输出目录：

```text
res/rode_ablation/tile016_topp05_rodecacheTrue_parallelFalse_prompt0_ctamul0/
```

### 端到端结果

| 配置 | GPU/e2e 时间 | 外部 wall time | 相对未缓存 RoDe | 相对 FlashInfer64 micro | 相对原始 DFSAttn |
|---|---:|---:|---:|---:|---:|
| RoDe，无 cache、无并行 | `294.446 s` | `308.06 s` | — | `+2.00%` | `+46.52%` |
| RoDe cache、无并行 | `199.514 s` | `215.35 s` | `-32.24%` | `-30.88%` | `-0.72%` |
| 原始 DFSAttn | `200.957 s` | `214.23 s` | — | — | — |

缓存-only 的 RoDe GPU/e2e 时间比未缓存版本减少 `94.932 s`，已经比原始
DFSAttn 快 `1.443 s`；考虑运行抖动，两者基本处于同一水平。外部 wall time
也从 `308.06 s` 降至 `215.35 s`，与原始 DFSAttn 的 `214.23 s` 接近。

### 缓存是否按预期生效

| phase | RoDe，无 cache | RoDe cache、无并行 |
|---|---:|---:|
| `flashinfer_fine_score` calls | `240` | `240` |
| `flashinfer_residual_select` calls | `240` | `240` |
| `flashinfer_residual_compact` calls | `240` | `240` |
| `flashinfer_plan` calls | `240` | `240` |
| `flashinfer_residual_rode_plan` calls | `2279` | `239` |
| `flashinfer_residual_rode_plan` time | `13.538 s` | `1.528 s` |
| `flashinfer_residual_rode_run` time | `21.526 s` | `7.154 s` |

RoDe plan 构建次数从约每个 layer/step 一次降为约每个 route refresh 一次，
plan 时间减少 `12.010 s`。更重要的是，RoDe residual 执行阶段从 `21.526 s`
降到 `7.154 s`，说明缓存不只是消除了显式计时的 plan，还减少了 plan 相关
的隐含准备和同步开销。

cache-only 的主要 phase 如下：

| phase | 总时间 |
|---|---:|
| `flashinfer_residual_select` | `4.223 s` |
| `flashinfer_residual_rode_plan` | `1.528 s` |
| `flashinfer_residual_rode_fp32` | `2.766 s` |
| `flashinfer_residual_rode_sddmm` | `0.550 s` |
| `flashinfer_residual_rode_softmax` | `0.442 s` |
| `flashinfer_residual_rode_spmm` | `1.274 s` |
| `flashinfer_residual_rode_run` | `7.154 s` |
| `flashinfer_core_run` | `15.721 s` |
| `flashinfer_output_unpermute` | `2.582 s` |

### 当前判断

这次实验确认：

1. 轻量级 `FLASHINFER64_RODE_CACHE` 是有效的，且不会引入上次并行 stream
   实验中的大幅退化。
2. 当前配置下，RoDe residual 的主要问题确实是重复 plan/准备，而不是
   SDDMM、softmax 或 SpMM 数学 kernel 本身。
3. `FLASHINFER64_PARALLEL_CORE_RESIDUAL=True` 仍应保持关闭。它在上一组实验
   中使 e2e 时间升至 `649.527 s`，而 cache-only 串行版本为 `199.514 s`；
   因此并行 stream 实现需要单独修复后才能重新评估。
4. 在本次单个 prompt 的测量误差范围内，cache-only RoDe 已达到原始 DFSAttn
   的端到端水平，并明显优于未缓存 RoDe 和 FlashInfer64 micro residual。

对应 timing 文件：

- [RoDe cache-only timing.csv](../res/rode_ablation/tile016_topp05_rodecacheTrue_parallelFalse_prompt0_ctamul0/timing.csv)
- [RoDe cache-only walltime.txt](../res/rode_ablation/rode_cache_serial.walltime.txt)


