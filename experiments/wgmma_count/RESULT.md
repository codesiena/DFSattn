# FlashInfer Core WGMMA count: 128x96 vs 64x64

Date: 2026-09-15

i## Main result: actual `topp_topk` Core CSR

This is the result relevant to the current method.  It is **not** a synthetic
dense matrix.  The input is the real HunyuanVideo prompt-0 dump captured at
step 12, layer 0 from a run whose backend metadata is
`flashinfer64_topp_topk`:

`/cnic/work/liutt/mywork/attention_time/res/wgmma_exact_topp_topk_20260915/snapshot/prompt_0/step012_layer000_flashinfer64_topp_topk.pt`

Layer 0 of the first sparse step was chosen so no earlier sparse attention
layer can make the 128x96 and 64x64 trajectories diverge.  Starting from the
same dumped Q/K/V, each variant independently recomputes the current method's
micro scores, macro scores, `topp_topk(top_p=0.16)` Core mask, and direct macro
CSR.  Residual is not run.  Nsight profiles only the resulting FlashInfer Core
`PrefillWithKVCacheKernel`; the CUDA-event `plan.run` timing also includes the
direct macro-CSR-to-vector-sparse index expansion launched immediately before
the attention kernel.

| Metric | 128x96 | 64x64 | 64x64 change |
|---|---:|---:|---:|
| Q macro blocks x K macro blocks | 349 x 465 | 697 x 697 | finer grid |
| Actually selected Core blocks | 187,682 | 507,243 | +170.27% |
| Logical Core QK pairs | 2,102,827,800 | 1,977,084,696 | -5.98% |
| Logical Core density | 4.41248% | 4.14862% | -0.26386 pp |
| Nsight GMMA warp instructions | 21,020,384 | 24,347,664 | +15.83% |
| WGMMA warp-group issues (raw / 4) | 5,255,096 | 6,086,916 | +15.83% |
| Core `plan.run` time (20 iterations) | 2.15578 ms | 3.82785 ms | +77.56% |

Nsight's GMMA metric counts each participating warp, so one source-level
four-warp WGMMA issue contributes four to the raw counter.  The raw counter is
the least ambiguous value and is retained in the table.

The 64x64 route covers 5.98% fewer logical QK pairs, but it represents them
with 2.70x as many selected sparse blocks.  On this actual route, that finer
work decomposition increases both dynamic WGMMA issues and scheduler/metadata
overhead.  Therefore the current 64x64 implementation is not a Core-time
optimization for this sample.

### Consistency check and interpretation

The 64x64 experiment changes the selector and CSR geometry together; it is not
a 128x96 CSR forcibly split into 64x64 tiles.  In
`bench_actual_topp_topk_core.py`, the 64x64 branch sets `Q_MACRO=K_MACRO=64`
and four Q16/K16 microtiles per macro.  The 128x96 branch uses eight Q16 and
six K16 microtiles.  The selector then recomputes macro scores and `topp_topk`,
and the direct CSR uses the corresponding Q/K macro sizes.  The resulting
grids (349x465 versus 697x697) verify that the upper algorithm changed as well
as the physical FlashInfer CTA.

For valid sequence length 44,561, tail capacity padding is actually smaller
for 64x64: Q/K tail padding is 47/47 tokens, versus 111/79 for 128x96.  Tail
padding is therefore not the explanation for the slowdown.  The main measured
structural change is more sparse rows and more selected blocks.  Each selected
block also causes CSR metadata handling, page/index expansion, address
calculation, K/V load setup, and softmax/synchronization work.  These costs do
not appear in the WGMMA counter.  At layer 20, for example, selected blocks
increase 93,054 -> 209,150 and the approximate expanded K-index work
(`selected_blocks * K_MACRO`) increases 8.93M -> 13.39M, while logical QK work
decreases.

The three layers show why WGMMA count alone is insufficient: layer 0 has more
64x64 WGMMA, whereas layers 9 and 20 have fewer, yet 64x64 is slower in all
three cases.  The current timing includes CSR expansion by design; a separate
future breakdown of expansion versus the pure prefill kernel is needed to
assign an exact percentage to each overhead source.

The production module's default constants remain `Q_MACRO=128` and
`K_MACRO=96`.  Enabling only the low-level
`FLASHINFER_FA3_FORCE_64X64=1` switch in a full application would be an
upper/lower-layer mismatch.  The benchmark avoids that mismatch by overriding
the Python selector/CSR constants before computing the 64x64 route; a
production 64x64 mode should expose the same synchronized configuration
explicitly.

### Cross-layer check: step 12, layer 9

The same experiment was repeated on a separately captured layer-9 Q/K/V dump
from the same prompt, seed, model, and `topp_topk` run:

`/cnic/work/liutt/mywork/attention_time/res/wgmma_exact_topp_topk_20260915/snapshot_layer9/prompt_0/step012_layer009_flashinfer64_topp_topk.pt`

| Metric | 128x96 | 64x64 | 64x64 change |
|---|---:|---:|---:|
| Actually selected Core blocks | 63,019 | 133,089 | +111.19% |
| Logical Core QK pairs | 570,968,856 | 444,549,912 | -22.14% |
| Nsight GMMA warp instructions | 7,058,128 | 6,388,272 | -9.49% |
| WGMMA warp-group issues (raw / 4) | 1,764,532 | 1,597,068 | -9.49% |
| Core `plan.run` time (20 iterations) | 0.95611 ms | 1.40932 ms | +47.40% |

Layer 9 confirms the timing conclusion but shows that WGMMA count alone is not
predictive: 64x64 uses fewer WGMMA instructions there, yet remains slower due
to the finer sparse decomposition and associated scheduling/metadata/irregular
memory overhead.  Thus the robust conclusion across the two layers is that
64x64 is slower for this current Core implementation; the WGMMA direction can
vary with the layer's selected CSR pattern.

### Third-layer check: step 12, layer 20

For an additional later layer, the same procedure was run on:

`/cnic/work/liutt/mywork/attention_time/res/wgmma_exact_topp_topk_20260915/snapshot_layer20/prompt_0/step012_layer020_flashinfer64_topp_topk.pt`

| Metric | 128x96 | 64x64 | 64x64 change |
|---|---:|---:|---:|
| Actually selected Core blocks | 93,054 | 209,150 | +124.76% |
| Logical Core QK pairs | 940,038,936 | 756,095,768 | -19.57% |
| Nsight GMMA warp instructions | 10,422,048 | 10,039,200 | -3.67% |
| WGMMA warp-group issues (raw / 4) | 2,605,512 | 2,509,800 | -3.67% |
| Core `plan.run` time (20 iterations) | 1.23876 ms | 1.90682 ms | +53.93% |

Layer 20 again has fewer WGMMA instructions with 64x64, but is substantially
slower.  Across layers 0, 9, and 20, 64x64 is slower every time; its WGMMA
count is layer-dependent, while the finer CSR decomposition consistently
increases the measured Core time.

Timing JSON and Nsight reports are under:

`/cnic/work/liutt/mywork/attention_time/res/wgmma_exact_topp_topk_20260915`

Reproduce both NCU runs with:

```bash
bash experiments/wgmma_count/run_actual_topp_topk_ncu.sh \
  /cnic/work/liutt/mywork/attention_time/res/wgmma_exact_topp_topk_20260915 \
  /cnic/work/liutt/mywork/attention_time/res/wgmma_exact_topp_topk_20260915/snapshot/prompt_0/step012_layer000_flashinfer64_topp_topk.pt
```

## Synthetic dense mechanism check (not the current route)

The remainder of this document records the earlier dense microbenchmark.  It
is useful for isolating the instruction decomposition of a fully dense support,
but it must not be used as the performance conclusion for `topp_topk`.

## Setup

- GPU: NVIDIA GH200 480GB (GH100, compute capability 9.0)
- Nsight Compute: 2025.3.1
- Data type: BF16; QK/VO head dimension: 128
- Attention: non-causal, one head, sequence length 3072
- Support: dense 3072x3072 in both runs (9,437,184 logical QK pairs)
- Profiled call: one `PrefillWithKVCacheKernel` after three warmups
- Metric: `smsp__inst_executed_pipe_tensor_op_gmma.sum`

The two sequence dimensions are divisible by 128, 96, and 64, so neither run
contains a partial Q/K tile. The Core CSR and physical FA3 CTA tile are changed
together. Nsight's demangled kernel names confirm traits `(CTA_Q, CTA_KV) =
(128, 96)` and `(64, 64)` respectively.

## Result

| CTA tile | Nsight GMMA warp-instruction count | WGMMA warp-group issues | Change |
|---|---:|---:|---:|
| 128x96 | 86,016 | 21,504 | baseline |
| 64x64 | 110,592 | 27,648 | +28.571% |

Nsight reports this metric per participating warp. A Hopper WGMMA/HGMMA is
issued by one four-warp warp group, so the source-level warp-group issue count is
the raw value divided by four.

The count decomposition is:

| CTA tile | QK WGMMA issues | PV WGMMA issues | Total |
|---|---:|---:|---:|
| 128x96 | `24*32*2*8 = 12,288` | `24*32*2*6 = 9,216` | 21,504 |
| 64x64 | `48*48*1*8 = 18,432` | `48*48*1*4 = 9,216` | 27,648 |

Thus PV uses the same number of WGMMA instructions for fixed logical work, but
QK increases by 50% because its SASS instruction changes from
`HGMMA.64x96x16.F32.BF16` to `HGMMA.64x64x16.F32.BF16`. This raises the total
QK+PV WGMMA issue count by 28.571%.

## Numerical and timing checks

Both variants used the same random seed and input shape. Comparing the BF16
outputs:

- max absolute difference: 0.00048828125
- mean absolute difference: 8.24543e-06
- relative L2 difference: 0.00111152
- exactly equal BF16 elements: 85.3500%

The difference is consistent with the changed reduction grouping/order. A
100-iteration CUDA-event check measured 0.04946 ms/run for 128x96 and 0.04390
ms/run for 64x64 for this one-head microbenchmark. This timing includes the
direct-CSR expansion inside `plan.run` and is not an end-to-end HunyuanVideo
claim; the instruction-count comparison is the primary result.

## Kernel resources

Nsight Compute `LaunchStats` and `Occupancy` were collected in a follow-up run:

| CTA tile | Registers/thread | Threads/CTA | Registers/CTA | Dynamic shared memory | Theoretical occupancy | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|
| 128x96 | 168 | 384 | 64,512 | 131.15 KB | 18.75% | 18.35% |
| 64x64 | 144 | 256 | 36,864 | 82.00 KB | 12.50% | 12.29% |

Both variants are limited to one resident CTA per SM by register and shared
memory limits. The 64x64 speedup therefore does not come from using more
registers: it uses 24 fewer registers per thread. It has lower occupancy because
its one resident CTA contains eight warps instead of twelve. Static cubin
resource inspection also reports a 72-byte stack frame for the profiled 64x64
non-causal kernel versus zero for 128x96; local-memory instruction counters were
not collected in this run.

The resource reports are `flashinfer_core_128x96_resources.ncu-rep` and
`flashinfer_core_64x64_resources.ncu-rep` in the artifact directory below.

## Artifacts

The Nsight reports and saved outputs are under:

`/cnic/work/liutt/mywork/attention_time/res/wgmma_flashinfer_20260914_v2`

Reproduce with:

```bash
bash experiments/wgmma_count/run_ncu.sh \
  /cnic/work/liutt/mywork/attention_time/res/wgmma_flashinfer_20260914_v2
```

The local FlashInfer source now accepts `FLASHINFER_FA3_FORCE_64X64=1`, gives
the JIT module a distinct cache URI, and disables the inter-consumer-warp-group
software barrier when the 64-row tile has only one consumer warp group.

## HunyMOR risk-breadth sweep: video quality and density

The following results are for the requested `n5/n100/n300/n600_proxy_k20_32`
experiments. Each candidate video was compared with the corresponding dense
VBench reference using `videometric.py`; all comparisons used 129 aligned
frames. Density is the corresponding `mean_final_density_sparse_steps_avg_over_layers`
record in `density_summary.csv`.

| Dangerous heads | Video | Dense reference | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Density |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 0 | `appearance_style_0` | 28.793885 | 0.895655 | 0.084162 | 0.194412478 |
| 5 | 7 | `human_action_1` | 23.845418 | 0.843256 | 0.169195 | 0.194232869 |
| 5 | 18 | `subject_consistency_0` | 11.297037 | 0.440494 | 0.697810 | 0.194161882 |
| 100 | 0 | `appearance_style_0` | 28.780915 | 0.895640 | 0.083916 | 0.195068352 |
| 100 | 7 | `human_action_1` | 23.886935 | 0.843981 | 0.168434 | 0.194928819 |
| 100 | 18 | `subject_consistency_0` | 11.299708 | 0.440813 | 0.697506 | 0.194874712 |
| 300 | 0 | `appearance_style_0` | 28.917843 | 0.897906 | 0.080747 | 0.196454137 |
| 300 | 7 | `human_action_1` | 23.724537 | 0.843052 | 0.167157 | 0.196365580 |
| 300 | 18 | `subject_consistency_0` | 11.300783 | 0.440628 | 0.697440 | 0.196335639 |
| 600 | 0 | `appearance_style_0` | 28.936576 | 0.897740 | 0.080401 | 0.198189253 |
| 600 | 7 | `human_action_1` | 23.972725 | 0.845191 | 0.164628 | 0.198148852 |
| 600 | 18 | `subject_consistency_0` | 11.294898 | 0.439865 | 0.697354 | 0.198226534 |

### 结论

1. **危险头数量增加确实提高了密度，但增幅有限。** 从 `n5` 增加到 `n600`，三个视频的密度分别增加：
   - 视频 0：`0.194412478 → 0.198189253`，增加 `0.003776775`；
   - 视频 7：`0.194232869 → 0.198148852`，增加 `0.003915983`；
   - 视频 18：`0.194161882 → 0.198226534`，增加 `0.004064652`。

2. **视频 0 的质量随危险头数量增加总体改善。** `n600` 相比 `n5`，PSNR 从 `28.793885` 提升到 `28.936576`，SSIM 从 `0.895655` 提升到 `0.897740`，LPIPS 从 `0.084162` 降至 `0.080401`。其中 `n300` 和 `n600` 明显优于 `n5/n100`。

3. **视频 7 的感知质量持续改善，但 PSNR/SSIM 有小幅波动。** `n600` 的 LPIPS 最低（`0.164628`），SSIM 最高（`0.845191`），PSNR 也最高（`23.972725`）；因此综合来看 `n600` 最好。

4. **视频 18 对危险头数量不敏感，且整体质量明显较差。** 四组实验的 PSNR 约为 `11.30`、SSIM 约为 `0.44`、LPIPS 约为 `0.697`，质量变化很小。`n600` 的 LPIPS 略低，但 SSIM/PSNR 略有下降，说明继续增加危险头没有带来明显视觉收益。

5. **总体结论：** 在本次 `proxy_k20_32` 扩展实验中，增加危险头数量可以将密度从约 `0.194` 提高到约 `0.198`，同时视频 0 和视频 7 的质量基本不降并略有改善；但视频 18 的质量瓶颈并未通过增加危险头解决。若以质量与密度的折中为目标，`n300` 已能取得较好的结果，`n600` 提供了最高密度和总体上最好的质量，但边际收益较小。

### 运行时间变化

`timing.csv` 中的 `e2e_generation_wall` 显示，危险头数量增加后端到端生成时间反而下降：

| 危险头数量 | E2E wall time | 相对 n5 |
|---:|---:|---:|
| 5 | 216.171 s | 基准 |
| 100 | 212.218 s | -1.83% |
| 300 | 199.282 s | -7.81% |
| 600 | 195.664 s | -9.49% |

从 `n5` 到 `n600`，端到端时间减少约 `20.507 s`。主要原因是 `flashinfer_core_run` 时间从 `20.259 s` 降至 `17.991 s`，而残差相关开销同时上升：`flashinfer_residual_micro_run` 从 `0.434 s` 增加到 `4.795 s`，`flashinfer_residual_select` 从 `0.076 s` 增加到 `1.822 s`。在当前配置下，Core 执行时间的下降超过了残差路径新增开销，因此总生成时间仍然下降。

需要注意，`n5` 与 `n100/n300/n600` 的残差路由形态不同，GPU 计时中的 `attention_execution` 并不严格单调（分别为 `84.113/87.155/84.152/85.186 s`）；因此这里的结论是端到端 wall time 的实测趋势，而不是单纯由密度变化推导出的理论加速。

### 运行时间结论修正与原因

上面的 `timing.csv` 实际是脚本最后运行的 prompt 18 的结果，不是三个 prompt
的平均值。脚本为每个 prompt 另外保存了 `timing_prompt0.csv`、
`timing_prompt7.csv` 和 `timing_prompt18.csv`。因此，`n600` 比 `n5` 快约
`9.49%` 只适用于 prompt 18，不能概括为“危险头越多，运行越快”。三个
prompt 的 E2E wall time 相对各自 `n5` 如下：

| Prompt | n5 | n100 | n300 | n600 |
|---:|---:|---:|---:|---:|
| 0 | 189.680 s | 191.704 s (+1.07%) | 192.660 s (+1.57%) | 195.937 s (+3.30%) |
| 7 | 189.918 s | 288.969 s (+52.15%) | 241.296 s (+27.05%) | 195.776 s (+3.08%) |
| 18 | 216.171 s | 212.218 s (-1.83%) | 199.282 s (-7.81%) | 195.664 s (-9.49%) |

原因主要有三点：

1. `n` 改变的是风险头的路由结构，而不是简单地把同一个 kernel 的工作量按比例放大。风险头会额外进入 Residual micro 路径；例如 prompt 7 的平均 Residual 行数从 `0.096`（n5）增加到 `10.994`（n600），非空 Residual 行比例从 `0.00347` 增加到 `0.41652`。总密度只表示 Core 与 Residual 的联合支持，不能说明工作在两个路径之间如何分配，也不能反映 CSR 行长度、bucket 分布和 kernel 形状。

2. 路由形状会影响 GPU kernel 的效率，而且不是单调关系。prompt 7 的 n100/n300 出现了明显的 route/plan 开销：`flashinfer_core_run` 分别为 `26.642/22.577 s`，而 n5/n600 约为 `18.030/18.089 s`；n300 的 `flashinfer_residual_compact` 也达到 `3.546 s`，明显高于其他配置。这解释了 prompt 7 的异常变慢。prompt 0 则随 n 增加而逐步变慢，prompt 18 恰好表现为逐步变快。

3. 每个配置只有一次生成运行，GPU 时钟、缓存、内存分配和调度状态都会造成波动。`e2e_generation_wall` 与 `e2e_generation_gpu` 几乎一致，所以本次差异不是模型加载或视频编码造成的；但它仍包含整个 pipeline 中未拆分的操作。详细 phase 之间还存在嵌套（例如 `flashinfer_core_run` 包含 CSR expand），不能将所有 phase 直接相加。

因此更准确的结论是：**危险头数量增加会改变 Core/Residual 的工作分配和 kernel 路由，运行时间呈 prompt 依赖的非单调变化；本次 prompt 18 的 n600 加速不是普遍规律。** 要判断平均趋势，应对每个 prompt 和每个 n 重复多次，并报告均值与方差。
