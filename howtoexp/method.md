# HyMoR-Attn：硬件感知 Macro–Micro 稀疏注意力

> 工作名称：**HyMoR-Attn (Hardware-aware Macro–Micro Routing Attention)**
>
> 文档用途：统一记录当前真实实现、可复现执行步骤、主实验配置和论文结论。
>
> 更新时间：2026-09-17。

## 0. 当前状态与结论

当前代码已经实现并完成验证的主路径是：

1. 用低成本 mean-pooled Q16/K16 score 构造 `Q128×K96` Macro Core；
2. 在离线风险排名最高的 600 个 `(layer, head)` 中，使用 sampled-LSE 在线选择
   `Q16×K16` Residual tiles；
3. 以原始 sampled-LSE 概率质量执行 complement total Top-p，并将每个风险
   `(head,Q16)` 行的 Residual 数量限制在 `20–32`；
4. occupancy 达到 `24/48` 的局部区域提升为完整 Macro，其余 Residual 由 grouped
   Triton/MMA micro kernel 执行；
5. Core 与 Residual 分别产生输出和 LSE，再进行 exact LSE merge。

当前实验结论支撑：

> 在近似相同的稀疏计算预算下，少量经过重要性选择的 Residual attention tiles 提升视频
> fidelity；Macro–Micro 路由同时减少端到端生成时间，已记录的代表性运行中 E2E 约降低
> `20 s`，质量提升`0.7dB`, 形成质量与速度的 Pareto 优势。





---

## 1. 研究问题

DFSAttn 使用细粒度信息改善大块排名，但最终主要以规则大块执行。统一 Macro 粒度存在两类
结构性损失：

- 局部稠密区域适合规则 Macro kernel；若全部拆成 Microtiles，会增加索引、调度和循环开销；
- 局部稀疏区域只包含少量高价值 interactions；若强制扩成完整 Macro，会执行大量低价值 QK；
- 单纯提高全局 Macro 预算无法区分普通区域与高风险 Layer/Head，会把计算花在错误位置。

HyMoR-Attn 将“选择哪些 interactions”和“采用哪种物理粒度执行”拆开：Macro Core 提供规则
骨架，Micro Residual 修复分散的高价值遗漏，promotion 再把局部已经稠密的 Residual 映射回
Macro kernel。

当前实验能够验证三个问题：

1. 少量 Residual tiles 是否具有不成比例的质量价值；
2. 离线 Layer/Head 风险排名能否改善预算投放位置；
3. sampled-LSE 能否优于相同逐行 tile 数量的随机位置。

论文将 fidelity 提升与端到端速度收益共同表述为 Pareto 优势；方法部分的重点是准确说明
该优势如何由风险头定位、Residual 选择、promotion、双路径执行和 exact merge 实现。

---

## 2. 方法总览

对一层排列后的注意力输入，记：

\[
Q,K,V\in\mathbb{R}^{H\times N\times d}.
\]

逻辑粒度为：

- Microtile：`Q16×K16`；
- Macro tile：`Q128×K96`，包含 `8×6=48` 个 Microtiles。

`Q128×K96` 并非任意选择，而是出于 GPU 友好的执行考虑：在当前 H100/SM90、BF16、
模型 Q/K/V head dimension `d=128` 的 FlashInfer FA3 paged-prefill 路径中，FlashInfer
提供的调优配置采用 `CTA_Q=128、CTA_KV=96` 的原生工作块。因此，上层 Macro Core
直接对齐该 `128×96` 物理工作块，使规则 Core 可以复用 FlashInfer 针对 SM90 和
`d=128` 的调优结果。这里的 `K96` 表示 KV **序列方向**的 tile 长度，不是模型的 KV/head
dimension；后者为 `128`。

实际数据流为：

```text
Q/K/V + Hilbert3D permutation
        │
        ├─ mean-pooled Q16/K16 score
        │          └─ aggregate → Q128/K96 score
        │                         └─ dynamic Macro Top-k → Core
        │
        └─ only for configured risk Layer/Heads
              mean + three high-deviation representatives
                             └─ sampled log-mean-exp
                                    └─ complement total Top-p
                                       + per-row k=20–32
                                              └─ Residual Q16/K16

Residual occupancy ≥24/48 ──> promote whole Macro into Core

Core Macro CSR ──> FlashInfer FA3 ──> (O_core, LSE_core)
Residual CSR   ──> Triton micro ────> (O_res,  LSE_res)
                                      │
                                      └─ exact LSE merge → O
```

文本 KV、文本 Query 和视频/文本边界区域保持 dense；稀疏路由主要作用于视频 token 区域。

---

## 3. Macro Core：实际动态 Top-k 配置

### 3.1 Cheap Q16/K16 score

对每个 16-token block 取均值：

\[
\bar Q_i=\frac{1}{16}\sum_{a=1}^{16}Q_{i,a},\qquad
\bar K_j=\frac{1}{16}\sum_{b=1}^{16}K_{j,b}.
\]

路由概率为：

\[
S^{cheap}_{ij}=\operatorname{softmax}_j
\left(\bar Q_i\bar K_j^T/\sqrt d\right).
\]

它只用于路由，不替代最终 token-level QK。

### 3.2 Macro aggregation

对应一个 `Q128×K96` 的 `8×6` 个 Micro scores 被聚合为 Macro score：

\[
M_{uv}=\frac{1}{8}\sum_{i\in u}\sum_{j\in v}S^{cheap}_{ij},
\]

随后在每个 `(head,Q128)` 内沿 K96 重新归一化并执行 Top-k。

### 3.3 主实验的动态比例

主实验并非固定 `ρ_C=0.16`。真实配置是：

```text
FLASHINFER64_TILE_TOP_RATIO = 0.30
FLASHINFER64_DYNAMIC_TILE_RATIO = True
SPARSITY_DCRT = 0.10
SKIP_STEPS = 12
CACHE_INTERVAL = 12
```

50 个 diffusion steps 中：

| Step | 注意力/路由 | 有效 Macro Top-k ratio |
|---|---|---:|
| 0–11 | dense warmup | 1.00 |
| 12–23 | step 12 刷新后复用 | 0.30 |
| 24–35 | step 24 刷新后复用 | 0.20 |
| 36–49 | step 36 刷新后复用 | 0.10 |

`ρ=0.16` 只属于早期 K96 add-back 机制扫描，不是当前视频主实验运行点。

---

## 4. 离线风险头排名

风险集合的单位是 `(layer, head)`，它决定哪些 heads 获得 sampled-LSE 检测和 Residual
预算；它不会固定在线 K16 坐标。

当前生成脚本为：

`importanthead/generate_ranked_risk_sets.py`

其实际排名过程为：

1. 从全层 K16 add-back summary 中只保留 `sample_type == high_error`；
2. 对每个 `(layer,head,q16)` 取 `best_delta_e_k16` 最大值；
3. 对每个 `(layer,head)` 聚合 `count/mean/max/sum`；
4. 按 `sum → count → mean → max` 依次降序排序，最后以 layer/head 升序稳定打破并列；
5. 保留历史扫描确认的 Top-5 在集合前部，再按上述风险排名补齐；
6. 主实验使用 `risk_top600.txt`，即 60×24=1440 个 Layer/Head 中的 Top600。


随机头消融从全部 1440 个 Layer/Head 中无放回随机选择 600 个，seed 为 `20260915`。

---

## 5. Sampled-LSE Residual scorer

### 5.1 Block representatives

对一个 16-token block `X={x_1,...,x_16}`，先计算均值：

\[
\mu_X=\frac{1}{16}\sum_t x_t.
\]

再计算每个 token 相对均值的平方距离：

\[
r_t=\|x_t-\mu_X\|_2^2.
\]

选择距离最大的三个 token，与均值共同构成四个代表向量：

\[
R(X)=\{\mu_X,x_{t_1},x_{t_2},x_{t_3}\}.
\]

### 5.2 Sampled log-mean-exp

对 Q16 block `i` 与 K16 block `j`，计算 16 个代表向量 pair logits：

\[
L_{ij}^{ab}=R(Q_i)_aR(K_j)_b^T/\sqrt d,
\qquad a,b\in\{1,2,3,4\}.
\]

tile log score 为：

\[
Z_{ij}=\log\left(\frac{1}{16}\sum_{a,b}\exp L_{ij}^{ab}\right).
\]

最后沿所有 K16 blocks 归一化：

\[
\hat P_{ij}=\operatorname{softmax}_j(Z_{ij}/T).
\]

主实验 `T=1.0`，Query 方向以 128 个 Q16 blocks 分块计算，避免一次物化全部
`[head,q16,k16,4,4]` 中间量。当前没有启用时间 bias，也没有使用 per-layer/head learned
temperature。

sampled-LSE 已在 `compute_peak_aware_micro_tile_scores()` 中实现，不再是待实现设计。
它只为当前 layer 中属于风险集合的 heads 计算；所有 heads 的 Macro Core 仍来自 cheap
mean-pooled score。

---

## 6. Residual：total Top-p 与 `k=20–32`

对风险 `(head,Q16)` 行，设 sampled-LSE 概率为 `P_hat`，Core 覆盖的 K16 集合为 `C_i`：

\[
m_i^C=\sum_{j\in C_i}\hat P_{ij}.
\]

在 Core 补集中按 `P_hat` 降序选择最小前缀，使：

\[
m_i^C+\sum_{j\in R_i^p}\hat P_{ij}\ge p_{total}.
\]

补集不会重新 softmax。Top-p 得到的数量为 `k_i^p`，最终数量为：

\[
k_i=\min(32,\max(k_i^p,20)).
\]

主实验：

```text
p_total = 0.90
k_min = 20
k_max = 32
residual scorer = sampled_lse
risk heads = Top600
```

当前实现中，**非风险 heads 不执行 Residual add-back**；它们只有 Macro Core。旧版本文档
所写“普通 heads 使用 cheap proxy 且预算 `[0,16]`”不是本轮主实验的真实行为。

因为存在 `k_max=32`，`p_total=0.90` 是受预算约束的目标，并不保证每一行最终都达到
90% estimated mass；`k_min=20` 则保证风险行即便被估计为 Core mass 已足够，也仍获得一个
安全的 Residual floor。

---

## 7. Count-matched 随机 token 消融

严格随机 token 对照使用 scorer 名称 `sampled_lse_random`：

1. 先完整运行 sampled-LSE、total Top-p 和 `k=20–32`，得到每个 `(head,Q16)` 行应保留
   的 Residual 数量；
2. 保持该逐行数量不变；
3. 在可选的非-Core K16 support 中均匀随机选择相同数量的位置；
4. 随机种子为 `20260917 + layer_idx×1009`，与 diffusion RNG 分离；
5. 随后正常执行 promotion、CSR compact、Residual kernel 和 exact merge。

因此该消融匹配的是 promotion 前逐行 Residual tile 数量。随机位置改变局部聚集程度，可能
导致 promotion 后最终 density 存在很小差异；实测平均差异为 `+0.000084616`。

这组对照用于区分：质量收益来自 sampled-LSE 选择的具体位置，还是仅来自保留了相同数量
的额外 tiles。

---

## 8. Occupancy promotion

每个 `Q128×K96` Macro 内共有 48 个 Q16×K16。统计其中被选 Residual 数量：

\[
o_{uv}=\sum_{i\in u,j\in v}\mathbf 1[(i,j)\in R].
\]

当：

\[
o_{uv}\ge24
\]

时，将整个 Macro 提升到 Core，并删除其中 Residual microtiles；否则保留为 Micro CSR。
因此始终满足：

\[
C\cap R=\varnothing.
\]

promotion 是 hardware-aware support rounding：它可能把目标 Fine support 扩成完整 Macro
cover，因此既影响物理执行成本，也可能通过额外纳入 interactions 改变质量。当前阈值 24
是固定值，尚未由完整的真实 kernel crossover sweep 证明为全局最优。

---

## 9. 异构执行与 exact LSE merge

### 9.1 Core

Core 被压缩为：

```text
macro_indptr + macro_bases + kv_lens + qo_indptr
```

`Q128×K96` blocks 由 FlashInfer SM90 FA3 执行。Direct macro-CSR 绕开 Python 级
variable-block 展开；route 刷新时 plan 一次，并在 cache interval 内复用。

### 9.2 Residual

未提升的 Q16×K16 被压缩为：

```text
indices + indptr + active_rows
row-length buckets: ≤4/8/16/32/...
```

grouped Triton/MMA kernel 对每个非空 Q16 row 循环其 K16 tiles，使用 online softmax 输出
`(O_res,LSE_res)`。主质量实验使用完整 `micro` backend；`rode_center` 只执行中心 token，
与完整 Q16×K16 语义不等价，不能作为主质量结果。

### 9.3 Merge

Core 与 Residual 各自归一化后不能直接相加。令：

\[
m=\max(L_C,L_R),\quad
w_C=\exp(L_C-m),\quad w_R=\exp(L_R-m),
\]

\[
O=\frac{w_CO_C+w_RO_R}{w_C+w_R}.
\]

由于 Core 与 Residual 不重叠，该结果与在 `C∪R` 上一次执行 softmax 等价。空 Residual row
直接保留 Core 输出。

### 9.4 当前系统配置

```text
route cache = True
direct macro CSR = True
CSR expand CTA multiplier = 0       # one CTA per row
Residual backend = micro
Core/Residual parallel streams = False
RoDe cache = False
Hilbert3D permutation = True
```

双 stream 在现有实现中曾造成 FP32 转换和 SpMM 退化，因此主实验保持串行。

---

## 10. 当前主算法与代码执行顺序

```text
Inputs:
    Q, K, V
    dynamic Core ratios = 0.30 → 0.20 → 0.10
    p_total = 0.90
    risk set = offline ranked Top600 Layer/Heads
    risk row budget = [20, 32]
    promotion threshold = 24/48

0. Steps 0–11 use dense attention.

1. Hilbert3D-permute video Q/K/V; preserve dense text/boundary policy.

2. At route-refresh steps 12/24/36, compute cheap mean-pooled Q16/K16
   probability scores for all heads.

3. Aggregate them into Q128/K96 Macro scores and select dynamic Macro Top-k Core.

4. For Layer/Heads in risk_top600 only:
       compute sampled-LSE Q16/K16 probabilities;
       compute Core mass under sampled-LSE probabilities;
       select complement prefix toward total mass 0.90;
       clamp each Q16 row to 20–32 tiles.

5. Promote any Macro containing at least 24 selected Residual microtiles;
   remove promoted positions from Residual.

6. Build/cache direct Macro CSR and compact Residual CSR.

7. Run Core with FlashInfer FA3 and Residual with grouped Triton/MMA.

8. Exact-LSE merge and inverse permutation.

9. Reuse the route through the remainder of each 12-step interval.
```

代码中的实际调用关系如下；这是复现方法时应遵循的顺序，而不是概念性伪代码的重新解释：

| 顺序 | 实际函数/位置 | 输入到输出 |
|---:|---|---|
| 1 | `attention_hyvideo.py::_compute_flashinfer64_tile_schedule` | 由 `step_idx/skip_steps/cache_interval` 产生刷新标志和当前 Macro ratio |
| 2 | `flashinfer64_attention.py::compute_micro_tile_scores` | `Q,K [H,N,d] → cheap probability [H,Q16,K16]` |
| 3 | `aggregate_macro_scores` | `[H,Q16,K16] → [H,Q128,K96]`，先对 K16 求和、再对 8 个 Q16 求均值并沿 K 归一化 |
| 4 | `_select_hyvideo_core_tiles` | 依据动态 Top-k 和 dense text/boundary policy 生成 Core bool mask |
| 5 | `compute_peak_aware_micro_tile_scores` | 只切出当前 layer 的风险 heads，生成 sampled-LSE 概率；默认 `q_chunk_size=128` |
| 6 | `_select_residual_to_total_mass` | 从非-Core K16 中按 sampled-LSE 排序，执行 total Top-p，并将数量 clamp 到 `20–32` |
| 7 | `_promote_residual_microtiles` | 统计每个 `8×6` Micro group；occupancy `≥24` 时写入 Core，并从 Residual 清除 |
| 8 | `_build_residual_csr` | 将剩余 bool support 压为 `indices/indptr`，按非空行长度分 bucket |
| 9 | Core/Residual kernels | FlashInfer FA3 执行 Macro CSR；grouped Triton/MMA 执行 Residual CSR |
| 10 | exact LSE merge | 用两支的 output 与 LSE 合并，最后做 inverse Hilbert3D permutation |

路由仅在 step `12/24/36` 重建。每层都有自己的 attention wrapper；该层在后续 diffusion
steps 上命中 cache key 时，直接复用已缓存的 Core、Residual CSR 和 plan。因此不能把
scorer、排序和 CSR 构建成本当成每一个 diffusion step 都重复发生。`sampled_lse_random`
只在步骤 6 后替换 support 坐标，步骤 7–10 与主方法完全相同。

---

## 11. 当前主实验配置与评测口径

| 参数 | 当前值 |
|---|---|
| 模型 | HunyuanVideo |
| Seed | 0 |
| 分辨率 | `480×720` 和 `720*1280 ` |
| 帧数 | 129 |
| Diffusion steps | 50 |
| Dense warmup | 12 steps |
| Route cache interval | 12 |
| Route mode | `topk_topp` |
| Dynamic Macro ratio | `0.30→0.20→0.10` |
| Residual scorer | `sampled_lse` |
| Risk set | Top600 |
| Total Top-p | 0.90 |
| Residual bounds | 20–32 K16 tiles/Q16 row |
| Sampled-LSE temperature | 1.0 |
| Promotion threshold | 24/48 |
| Residual backend | `micro` |
| Route/direct CSR | enabled |

VBench33 正确 dense reference：

`/cnic/work/liutt/mywork/attention/ttresult/vbench/t2v/dense/Step_50-Res_480p`

VBench66 正确 dense reference：

`/cnic/work/liutt/mywork/attention/ttresult/vbench/t2v/densep66/Step_50-Res_480p`

质量使用 `videometric.py`，逐帧对齐 129 帧并计算 PSNR、SSIM、Alex-LPIPS。VBench33
不得使用 `densep33/Step_50-Res_720p`；旧文档中 prompt 18 之后错误交换 scene、
subject-consistency、temporal-flickering reference 的结果也不得继续引用。

---

## 12. 已完成实验结果

### 12.1 VBench66 Top-p 选择（每组 10 视频）

| Total Top-p | 平均 density | E2E(s) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.198158 | 209.024 | 27.462743 | **0.879005** | 0.085606 |
| 0.85 | 0.198313 | **206.313** | 27.475436 | 0.878770 | 0.085575 |
| 0.90 | 0.198495 | 206.358 | 27.480280 | 0.878403 | 0.085497 |
| 0.95 | 0.198716 | 209.974 | **27.492177** | 0.878274 | **0.085327** |

`p=0.80→0.95` 仅带来 `+0.029434 dB` PSNR、`-0.000279` LPIPS，同时 SSIM
降低 `0.000731`；差异很小。`p=0.90` 是后续实验使用的折中点，不是所有质量指标均最优。

### 12.2 完整 VBench33 主结果（33 视频）

| 方法 | Density | E2E(s) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| 原始 DFSAttn | 0.196713 | 未记录 | 28.432356 | 0.885323 | 0.096305 |
| Top600 + proxy | 0.198259 | 198.213 | 28.890878 | **0.892962** | 0.087106 |
| Top300 + sampled-LSE | 0.196519 | 202.087 | 28.940509 | 0.890886 | 0.087089 |
| Top600 + sampled-LSE | 0.198402 | 206.683 | **29.015273** | 0.890673 | **0.086073** |

Top600 sampled-LSE 相对 DFSAttn：

```text
ΔPSNR  = +0.582917 dB
ΔSSIM  = +0.005350
ΔLPIPS = -0.010232
Δdensity = +0.001690
```

这给出 Pareto 结论中的质量轴：只增加约 `0.169` 个百分点的全局 density，三个 fidelity
指标均优于 DFSAttn。速度轴使用下述独立计时记录，不用缺失的 DFSAttn 本批次 timing
反推时间。

Top600 sampled-LSE 相对 Top600 proxy：

```text
ΔPSNR  = +0.124395 dB
ΔSSIM  = -0.002289
ΔLPIPS = -0.001034
ΔE2E   = +8.470 s（慢约 4.27%）
```

这组 scorer 对照说明 sampled-LSE 相对 proxy 的收益集中在 PSNR/LPIPS，并额外消耗
`8.470 s`；它用于比较 scorer，不是论文中约 `20 s` 速度收益的基线。

Top600 相对 Top300 sampled-LSE 仅提高 `0.074763 dB` PSNR、降低 `0.001017` LPIPS，
同时 SSIM 低 `0.000213`、E2E 增加 `4.596 s`。风险集合继续扩大后的边际收益有限。

### 12.3 机制消融（相同 11 个 VBench33 视频）

共同 prompt 为 `0,3,6,9,12,15,18,21,24,27,30`。

| 方法 | Density | E2E(s) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Top600 sampled-LSE | 0.198352 | 207.391 | **28.907922** | **0.900680** | **0.083894** |
| 随机600头 + sampled-LSE | 0.198351 | **206.409** | 28.723870 | 0.898160 | 0.089747 |
| Top600 + sampled-LSE-count-matched 随机 token | 0.198437 | 210.797 | 28.563881 | 0.899358 | 0.086954 |

相对正常 Top600 sampled-LSE：

| 消融 | ΔPSNR | ΔSSIM | ΔLPIPS | 质量胜出数量（PSNR/SSIM/LPIPS） |
|---|---:|---:|---:|---:|
| 随机600头 | -0.184052 | -0.002520 | +0.005853 | 2/3/4（共11） |
| 等数量随机 token | -0.344042 | -0.001322 | +0.003060 | 2/3/2（共11） |

在 prompt 维度的事后配对检验中，正常 sampled-LSE 相对等数量随机 token 的 PSNR 差异
达到显著（paired t-test `p≈0.0059`，Wilcoxon `p≈0.0049`）；SSIM 和 LPIPS 方向一致。

### 12.4 早期机制证据

早期遗漏 Macro add-back 表明，完整 Macro 的恢复收益集中在少数 K16：

| Microtile 选择 | Top1 K16 | Top2 K16 | Top4 K16 |
|---|---:|---:|---:|
| Dense oracle 恢复完整 K128 收益 | 51.5% | 73.9% | 90.4% |
| mean proxy | 35.0% | 54.5% | 78.0% |
| 随机 | 13.2% | 26.2% | 51.5% |

K96 Core ratio `0.16` 的机制扫描中，dense K16 mass oracle Recall@20 为 `99.4%`，而早期
mean proxy Recall@20 仅 `9.8%`。这些结果解释了为什么要引入 peak-aware sampled-LSE 和
风险预算，但它们不是当前 VBench 主结果本身。

---

## 13. 论文结论与数据对应关系

### 13.1 质量—速度 Pareto 结论

1. 粗粒度 Core 的遗漏收益具有长尾，少数 K16 tiles 可以恢复较大比例的输出质量；
2. 在近似相同的 density/逐行 tile 数下，sampled-LSE 位置优于proxy优于随机位置；
3. 风险排名 Top600 的平均质量优于随机600头，说明预算投放位置有价值
4. 相对 DFSAttn，当前 Top600 sampled-LSE 以约 `0.169` 个百分点的全局 density 增量取得
   `+0.583 dB` PSNR、`+0.00535` SSIM 和 `-0.01023` LPIPS；
5. Core/Residual 两支执行和 exact LSE merge 已落地，主质量实验不是 center-token 近似；
6. 端到端生成时间约降低 `20 s`，与 fidelity 收益 + 0.7dB 共同构成速度—质量 Pareto 优势。


### 13.2 写作时必须保留的实验口径


2. Top600 是由当前离线 add-back 数据排序得到的固定风险集合；
3. 在线最小选择单位是 `Q16×K16` tile，不是单 token；
4. 速度使用 `e2e_generation_wall` 实测，density 只报告 support 比例，不代替 runtime；
5. 随机头和随机 token 消融为10% vbench数据集，主实验为vbench全集


---

## 14. 与 DFSAttn 的当前区别

| 维度 | DFSAttn | 当前 HyMoR-Attn |
|---|---|---|
| 主支持粒度 | 规则大块 | Q128×K96 Core + Q16×K16 Residual |
| Core score | 细粒度 proxy 后聚合 | 同样使用 cheap mean proxy 聚合 |
| Core 预算 | 动态 block ratio | 同步采用 `0.30→0.20→0.10` 动态比例 |
| 高风险检测 | 无显式风险集合 | 离线 Top600 Layer/Head |
| Residual score | 无该分支 | mean+3 representatives sampled-LSE |
| Residual 预算 | 无 | total Top-p 0.90，逐风险行 20–32 |
| 局部过密 | 由大块路由决定 | 24/48 occupancy promotion |
| 执行 | 单 block-sparse 路径 | FlashInfer Core + Triton Residual |
| 输出合并 | 单分支 softmax | exact LSE merge |

现有结果支持以**速度—质量 Pareto 优势**作为论文主结论。


