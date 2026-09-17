# 分层 Top-k/Top-p 与异构 Kernel 稀疏注意力：当前实现概述

## 0. 相较原生 FlashInfer 的改动清单

这里的“原生 FlashInfer”特指原来的 `VariableBlockSparseAttentionWrapper`/FA3 sparse 路径：由调用方提供 variable-block mask，FlashInfer 负责 planning、必要的 token-level index 展开，以及 block-sparse attention 执行。本项目仍然复用 FlashInfer 的 FA3 scheduler 和 Core attention kernel；下面列出在其上新增或替换的部分。

| 层次 | 原生 FlashInfer | 当前实现的改变 |
|---|---|---|
| 稀疏决策 | 不负责 HunyuanVideo 的质量路由 | 新增 Q16×K16 fine score、Q128×K96 macro score、Core macro Top-k/Top-p、Residual total Top-p 和 occupancy promotion |
| 支持集 | 一个 sparse mask/一个 attention 分支 | 拆成互不重叠的 Core 与 Residual 支持集，分别执行后 exact LSE merge |
| Core 粒度 | 由 wrapper 的 variable blocks 处理 | 将上层逻辑固定到 `Q128 × K96`，与 SM90 FA3 工作块对齐；当前实验的有效序列按该布局对齐，代码保留尾块处理仅作为通用防御路径 |
| Core 表示 | 由 bool/variable-block mask 进入 wrapper，再由 wrapper 展开 | 直接从 Core mask 构造 `macro_indptr + macro_bases + kv_lens + qo_indptr` 的紧凑 macro-CSR，绕过 Python 级 variable-block 展开 |
| FA3 plan 生命周期 | 原路径通常在 sparse forward 中重新建立 wrapper/workspace 并 plan | route 刷新时 plan 一次，按 route 和 layer 保留；cache interval 内复用 plan |
| CSR 到 token offsets | 使用 FlashInfer page-op 展开 | 保留 page-op 以满足 FA3 paged-attention ABI，但修复原 kernel 的重复 row 处理/写入竞争，并增加 CTA 数运行时配置 |
| Residual 执行 | 没有本项目的细粒度补偿分支 | 新增 grouped Triton/MMA Q16×K16 online-softmax kernel，按实际 CSR 长度分桶执行 |
| Residual 存储 | 不适用 | bool mask 压缩为 `indices + indptr`，只保留非空 row；按 `≤4/8/16/32/64/...` K16 数量分桶，避免全局最大长度 padding |
| Residual 输出与 merge | 不适用 | 只为 active rows 分配 compact output/LSE；新增 exact LSE merge，并原地更新 Core output |
| 视频 token 布局 | 不包含本项目的视频 Hilbert/边界处理 | 新增 Q/K/V 视频 token permutation 与 inverse permutation；文本区域、跨模态边界块保留 dense |
| 选择确定性 | 不针对本项目的量化 score tie 做规则约束 | Residual/需要完整排序的路径使用 stable sort，使输入 K index 顺序成为 tie-break，降低 BF16/FP16 并列时因 batch/GEMM 形状改变选择结果的风险 |
| route 与 workspace | 由 wrapper 管理其自身临时 buffer | 全模型共享 float/vector-offset workspace，按最大实际需求增长复用；每层保留紧凑 plan 和可配置 integer plan workspace |
| 显存占用 | 可能保留完整 mask、expanded buffer 或反复 reset workspace | CUDA 路径不长期保留完整 Residual bool matrix，不为 Residual/merge 分配完整 `[head, sequence, dim]` 临时输出 |
| 兼容性 | 使用 FlashInfer 原生路径 | direct macro-CSR/FA3 不可用时回退到 VariableBlock wrapper；因此该改动不是对所有设备的强制替换 |
| 观测与消融 | 只有基础调用计时 | 新增 route、plan、CSR expand、Core/Residual kernel、bucket、merge、density 和 residual 长尾统计 |

因此，当前方法不是替换 FlashInfer 的 FA3 attention，而是：**用新的质量感知路由和数据布局决定“算哪些块”，用 direct macro-CSR 和缓存减少 wrapper 开销，用自定义 Residual kernel 补充细粒度区域，最后仍由 FlashInfer FA3 完成规则 Core 的主要计算。**

### 0.1 改动边界与不应重复归因的部分

- `Q128 × K96`、FA3 `paged_run`、SM90 scheduler 和 online-softmax Core kernel 仍是 FlashInfer 提供的能力；我们的改动主要在路由、输入表示、生命周期和 Residual 分支。
- token-level vector offsets 仍然会生成，因为这是当前 FA3 paged-attention 的输入 ABI 要求；direct macro-CSR 只是把生成点从 wrapper 的 variable-block planning 中移到可控的 page-op，并通过 workspace/cache/CTA 调度降低代价。
- `flashinfer_core_run` 的计时包含 `flashinfer_core_csr_expand`，`flashinfer_residual_micro_run` 的计时包含各个 `flashinfer_residual_bucket_leN`；分析时不能把嵌套项再次累加。
- 旧 DFS packed 路径中曾出现的 selected-value 四维临时张量 OOM，则通过“分别计算两支的输出与 LSE、最后 merge”这一状态分解规避；当前 FlashInfer64 路径进一步使用 compact Residual output，不保留完整 Residual bool/output 临时张量。

### 0.2 Page-op 修复与 CTA 配置

本地 FlashInfer 的 `BlockSparseIndicesToVectorSparseOffsetsKernel` 原先启动 `num_sms` 个 CTA，但每个 CTA 都从自己的 `blockIdx.x` 开始、以步长 1 处理后续 rows，导致多个 CTA 重复处理同一 CSR row，并对相同输出位置发生竞争写入。当前改为 grid-stride row 分配：

```cpp
for (int b = blockIdx.x; b < batch_size; b += gridDim.x)
```

同时通过 `FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER` 配置 CTA 数：正整数 `N` 使用 `min(batch_size, N × SM数)`，`0` 使用 `batch_size`（一行一个 CTA）。在 HunyuanVideo 480p、相同 route 配置下，实测结果为：

| 配置 | CSR expand | Attention | E2E GPU |
|---|---:|---:|---:|
| `1 × SM` | 5.023 ms | 32.932 ms | 219.934 s |
| `4 × SM` | 4.731 ms | 32.684 ms | 219.158 s |
| 一行一个 CTA | **0.133 ms** | **29.256 ms** | **209.347 s** |

三种配置的 `sparsity_records.csv`、`density_records.csv` 和输出视频一致；减去 CSR expand 后，FA3 Core 实际时间均约 7.30 ms，说明这部分收益来自 CSR 调度而非改变了 attention 计算。当前建议将“一行一个 CTA”作为 direct CSR 的默认实验配置。

## 1. 核心思路

当前新增的方法将一次注意力拆成两个互不重叠的计算分支：

- **Core**：以逻辑块 `Q128 × K96` 为单位选择规则、密集的大块，交给 FlashInfer block-sparse attention。
- **Residual**：以 `Q16 × K16` 为单位补充 Core 没覆盖但质量较高的细粒度区域，交给 grouped Triton/MMA kernel。
- **Merge**：两个分支分别计算输出和 LSE，最后用 exact LSE merge 得到与二者并集完全等价的 softmax 结果。

这样做的目的，是让规则的大块利用 FlashInfer 的高吞吐，同时用较小的 microtile 避免大块选择造成的精度损失。在当前 H100、head dimension 128、FA3 paged-prefill 路径中，FlashInfer SM90 kernel 的原生工作块就是 `CTA_Q=128, CTA_KV=96`；因此上层 Core 的 `Q128 × K96` 与该物理工作块对齐。当前实验的有效序列由上层布局保证按 `128 × 96` 对齐，不会产生 partial K block；实现中的尾块 predication 仅保留给其他序列长度的兼容场景。

整体数据流如下：

```text
Q/K
 └─ Q16×K16 fine score
     ├─ 每 8×6 个 fine score 聚合为 Q128×K96 macro score
     │   └─ macro Top-k/Top-p → Core mask → FlashInfer
     └─ 所有非 Core microtiles 按原始质量补足 total Top-p
         └─ occupancy promotion → compact indices → Triton Residual

FlashInfer (O_core, LSE_core) + Triton (O_res, LSE_res)
                         └─ exact LSE merge → 最终输出
```

## 2. 选择算法

### 2.1 Fine score：已有的 Q16×K16 tile score

对每个 head，先分别对 Q 和 K 的每 16 个 token 做均值池化：

```text
Q̄_i = mean(Q[16i : 16(i+1)])
K̄_j = mean(K[16j : 16(j+1)])
```

再计算代理注意力并沿 K16 维归一化：

```text
S_fine(i,j) = softmax_j(Q̄_i K̄_j^T / sqrt(d))
```

因此 `S_fine` 的形状为 `[head, ceil(S/16), ceil(S/16)]`。它只用于路由，不代替最终 token-level QK 计算。

### 2.2 Core score：8×6 个 fine scores 聚合

一个 `Q128 × K96` macro tile 正好包含：

```text
Q 方向：128 / 16 = 8 个 microtiles
K 方向： 96 / 16 = 6 个 microtiles
总计：8 × 6 = 48 个 Q16×K16 microtiles
```

实现中将对应的 48 个 fine scores 聚合成 macro score，并在每个 head、每个 Q128 block 内沿 K96 方向重新归一化。

### 2.3 Core：macro Top-k 或 Top-p

论文主路径使用 `topk_topp`：

- 对每个 head、每个 Q128 block，根据 macro score 排序。
- 按 `FLASHINFER64_TILE_TOP_RATIO` 的比例选择 K96 blocks，数量为 `ceil(K_blocks × ratio)`，至少选 1 个。
- 被选中的 macro tiles 构成 Core mask，并进入 FlashInfer。

兼容路径 `topp_topk` 则按 `FLASHINFER64_TOP_P` 对 macro score 做累计质量 Top-p。

HunyuanVideo 中，纯视频区域采用上述稀疏选择；文本 KV blocks、文本 query blocks，以及视频/文本边界块保持 dense，避免跨模态边界被错误裁剪。

### 2.4 Residual：非 Core microtiles 按绝对质量补足总 Top-p

Residual 不对 Core 的补集重新归一化。对每个 head、每个 Q16 query：

1. 计算该 query 已被 Core 覆盖的原始 fine-score 质量 `m_core`。
2. 目标总覆盖质量为 `p_total = FLASHINFER64_TOKEN_TOP_P`。
3. 只需补充 `max(p_total - m_core, 0)` 的质量。
4. 在所有非 Core 的 Q16×K16 microtiles 中，按原始 `S_fine` 从高到低选择，直到补足该差值。

也就是说，Residual 的 Top-p 是对**原始概率质量**的补足，而不是对剩余区域再次做 softmax。这样 Core 已覆盖得越多，Residual 自动选择得越少。

### 2.5 Promotion：Residual 过密时提升为 Core

如果同一个 `Q128 × K96` macro tile 内被 Residual 选中的 microtiles 数量达到：

```text
FLASHINFER64_PROMOTION_THRESHOLD
```

则整个 macro tile 被提升为 Core，并从 Residual 中删除其 48 个 microtile 位置。默认阈值为 24，即 occupancy 达到 50% 时提升。

Promotion 的作用是避免 Triton 分支在局部已经很密集时仍逐个处理 microtiles；此时直接交给 FlashInfer 更适合。Promotion 后 Core 与 Residual 始终不重叠。

### 2.6 遗漏 K/V tile 的选择与评分（以及 layer/head 排名）

这里把“遗漏头”统一解释为：某个 `(layer, head, Q16)` 行中，未进入
Core 的 K96 macro 内的 K16 microtile；不是删除整个 attention head。整个
attention head 的重要性只用于离线确定预算优先级，不能替代运行时的 tile 选择。

#### 2.6.1 当前已实现的遗漏选择（`topk_topp` 基线）

对每个 `(layer, head, q16=i)`，令 `C_i` 为 Core 已选中的 K96 macro 集合，
`G_i` 为其余 macro 中的全部 K16 候选。`S_fine(i,j)` 已经在**全部** K16
上归一化，因此：

```text
m_core(i)   = sum_{j in C_i} S_fine(i,j)
r_need(i)   = max(p_total - m_core(i), 0)
```

其中 `p_total = FLASHINFER64_TOKEN_TOP_P`。随后只在 `G_i` 中按原始
`S_fine(i,j)` 降序稳定排序，若第 `t` 个候选加入前的累计质量仍小于
`r_need(i)`，就选入该 K16；已在 Core 中的 K16 不会再次进入 Residual。
补集不重新 softmax，也不做补集内归一化。`min_top_k/max_top_k`（若配置）
只对最终数量作下限/上限约束，稳定排序时以 K16 原始索引作为 tie-break。

因此当前评分只有一个含义：**原始 fine 概率质量**。它保证 Core+Residual
达到目标总质量，但不直接估计“加入这个 tile 能减少多少输出误差”。选完后，
若同一 Q128×K96 macro 中 Residual 数量达到
`FLASHINFER64_PROMOTION_THRESHOLD`，整块提升到 Core，并从 Residual 删除，
最后仍用 exact LSE merge 合并两支。

#### 2.6.2 为什么需要第二个“恢复性价比”分数

实验中的离线标签是独立 addback 的误差下降：

```text
DeltaE(i,g,j) = E_sparse(i) - E_sparse+K16(i,g,j)
```

它只在离线拥有 dense 结果时可计算，线上不可直接使用。`S_fine` 大只能说明
该 tile 可能有注意力质量，不能说明它对当前 Core 输出是否提供了新的 V 信息；
反之，局部峰值被均值池化稀释时，真正高 `DeltaE` 的 tile 可能被遗漏。因此线上
应把 `S_fine` 当作召回分数，再用一个便宜的、以误差恢复为目标的代理分数排序。

#### 2.6.3 建议的两阶段在线方法（具体可实现版本）

**阶段 A：候选召回。** 对每个遗漏 macro `g`，保留现有均值分数，同时计算
三个诊断量：

```text
M_g = sum_{j in g} S_fine(i,j)       # macro 总质量
A_g = max_{j in g} S_fine(i,j)       # macro 内峰值
H_g = -sum_j (S_fine(i,j)/M_g) log(S_fine(i,j)/M_g)
C_g = A_g / (M_g + eps)             # 峰值集中度
```

不要用 `M` 或低熵作硬门限；它们只做召回特征。候选 macro 取
`Top-B(M_g) ∪ Top-B(A_g)`，再从每个入选 macro 保留其 `Top-b(S_fine)` K16。
为减少均值池化漏峰，可增加一个低开销的 `S_peak`：将 Q16、K16 各分成
2×2 个 8-token 子块，取 16 个 Q/K 子块组合的池化点积最大值；再加入
`Top-B(S_peak)` 的 macro。`B`、`b` 是预算参数，不是误差标签；建议先用
`B=16, b=4` 做消融，再按 Residual interaction 预算调节。

**阶段 B：恢复性价比排序。** 这是对当前代码的改进方案：需要先执行 Core，
得到 `(O_core,LSE_core)`，再对阶段 A 的少量候选排序；当前实现仍在 Core
执行前完成 Residual 选择，因此尚未使用这个 `U_hat`。

对阶段 A 的候选计算一个不需要 dense attention 的估计值。令
`v_j` 为 K16 的 V 均值，`var_j` 为该 K16 内 V 的方差，`o_i` 为当前 Q16
的 Core 输出均值；令 `r_ij` 为 `S_fine` 与 `S_peak` 的归一化 rank 融合
（例如各占 0.5）。定义：

```text
d2_ij      = ||v_j - o_i||_2^2 + var_j
alpha_ij   = r_ij / (m_core(i) + r_ij + eps)
U_hat(i,j) = alpha_ij * sqrt(d2_ij)
```

`d2` 表示该 tile 的 V 内容相对当前 Core 的潜在新信息，`alpha` 抑制只有
极小注意力质量的候选，`U_hat` 是“预计误差恢复量”的代理，而不是严格的
`DeltaE`。若要比较不同 head，可再除以
`sqrt(mean(||o_i||_2^2))+eps` 做尺度归一化。

**阶段 C：按成本选入。** 对每个候选定义增量成本 `DeltaCost`（至少按一个
K16 interaction 计，若会跨 length bucket 或触发 promotion，则加入对应成本）。
在每个 Q16 的 `K16` 上限和整层 Residual interaction 总预算下，按

```text
Score_select(i,j) = U_hat(i,j) / (DeltaCost(i,j) + eps)
```

贪心选取。可选地先用原始质量规则满足一个较低的 `p_floor`，剩余预算全部按
`Score_select` 补充；这样不会完全丢失质量覆盖，同时把额外预算给恢复性价比
最高的 tile。选完后沿用现有 promotion、CSR 分桶、Residual kernel 和 exact
LSE merge，计算结果仍严格对应最终支持集。若阶段 B 的额外读取成本暂时不能
接受，阶段 A 的 rank 融合可以作为单阶段退化版本，但应明确它只是 recall proxy。

#### 2.6.4 离线如何给 `(layer, head)` 排名

对每个采样到的高误差 Q16，先在所有遗漏 macro/K16 中取最佳独立恢复量：

```text
bestDeltaE(i) = max_{g not in C_i} max_{j in g} DeltaE(i,g,j)
```

然后以 `(layer, head)` 分组，主排名分数取：

```text
B(l,h) = mean_{i in sampled_high_error(l,h)} bestDeltaE(i)
```

同时报告 `n`、median、p90、max；`n` 很小时不能只看均值，部署前应设置最小
样本数或使用收缩均值
`B_shrink=(n/(n+n0))*B + (n0/(n+n0))*B_global`。这个排名是**预算优先级**，
不是运行时直接选择 K16 的分数，也不是“该 head 在所有 token 中发生的概率”。

当前全层 Step12、单视频、`ratio=0.16` 的初步候选（详见
`importanthead/headv1.md`）为：

| layer | head | 高误差样本 n | `mean(bestDeltaE)` | 说明 |
|---:|---:|---:|---:|---|
| 43 | 17 | 40 | 0.871153 | 当前最稳定的高恢复收益候选 |
| 30 | 13 | 30 | 0.836940 | 高收益候选 |
| 48 | 5 | 7 | 0.785461 | 样本少，需复测 |
| 8 | 10 | 3 | 0.607010 | 样本很少，不能单独据此固化 |
| 12 | 5 | 5 | 0.567704 | 候选，需跨视频/step 验证 |

这些数值来自“每个 layer/head 先取高误差 Q16”的条件样本，只能用于当前
实验的候选排序；最终固定 head 白名单或概率模型前，必须在所有 layer、多个
视频和多个 step 上复测，并同时考察覆盖率、误差下降和实际 kernel 成本。

## 3. 底层执行

### 3.1 Core：直接 macro-CSR FA3 路径

Core mask 的形状为：

```text
[head, ceil(S/128), ceil(S/96)]
```

每个 block 的 row size 为 128、column size 为 96；当前实验的有效序列已经对齐，因此实际运行没有 partial K block。代码仍保留尾块实际长度的处理，以支持其他序列长度。默认 direct 路径不再让 `VariableBlockSparseAttentionWrapper` 每次从三维 bool mask 展开和 plan，而是直接从 Core mask 构造紧凑 macro CSR：

```text
macro_indptr : 每个 (head,Q128) 行所选 K96 block 的范围
macro_bases  : 所选 K96 block 在按 head 展平的 K/V 中的 token 起始偏移
kv_lens      : 每行实际 token 数；当前对齐配置下为选中 K96 blocks 数量乘 96，通用路径可容纳尾块实际长度
qo_indptr    : 每个 (head,Q128) 的实际 query 行范围
```

route 刷新时只执行一次 FA3 scheduler `plan()`，plan 与该层 route 一起保留；随后 cache interval 内直接复用。每次 `run()` 前仍需调用 FlashInfer 的 CUDA page-op，把 compact K96 bases 展开为底层 paged FA3 接口所需的 vector offsets，但这一步不再经过 Python 级 variable-block 展开，也不会重建 scheduler plan。随后底层直接调用 FA3 `paged_run()` 计算：

```text
(O_core, LSE_core)
```

为解决此前的重复 plan、OOM 和 `_vector_sparse_indices_buffer is not large enough`：

- 全模型共享一个 FlashInfer float workspace 和 vector-offset workspace。
- 每层只保留自己的 compact scheduler plan 及较小的 integer plan workspace；`FLASHINFER64_ROUTE_CACHE=True` 且 cache interval 为 12 时，刷新步 plan 一次，后续 11 步不再 plan。
- vector-offset workspace 按运行中最大实际展开规模增长并复用，不再每次 reset wrapper 或重新分配默认 512MB buffer。
- direct API 不可用、设备不是 FA3 时，自动回退到原 VariableBlock wrapper；回退路径仍复用已足够大的 vector buffers，但必须逐次 plan。
- CUDA 路径不在每层保留完整 Residual bool matrix。
- `FLASHINFER64_ROUTE_CACHE=False` 仍是低峰值显存模式；此时 route identity 在调用后释放，下一次重新选路和 plan。

direct 路径会额外保留每层的 integer scheduler workspace，其大小由 `FLASHINFER64_PLAN_WORKSPACE_MB` 控制，默认 8MB。它用显存换掉重复 plan；若显存不足，可降低该值验证 FlashInfer 是否仍能成功 plan，或关闭 direct 路径回退到共享 wrapper。

### 3.2 Residual：Q16×K16 grouped Triton/MMA

Residual bool mask 会先压缩成真正的 CSR：

```text
indices : 所有被选 K16 block 编号的连续一维数组
indptr  : 每个 (head,Q16) 行在 indices 中的起止位置
```

CUDA 上不再保留原始三维 bool mask，也不再把所有行 padding 到全局最大 Residual 长度。非空 CSR 行按照实际 K16 数量进入 `≤4/8/16/32/64/128/192/256/384/512/...` 的 length buckets。

Triton kernel 的执行方式为：

- `≤64` 的短行仍由一个 program 对应一个非空 `(head, Q16 block)`；空行不启动 program。
- 从 `≤128` bucket 开始，默认将一行拆成多个 program，每个 program 负责 `64/96/128` 个 K16 slot，分别生成 FP32 partial `(O,LSE)`，随后用分层 LSE merge 合成该行结果。可用 `FLASHINFER64_RESIDUAL_SPLIT_MIN_CAPACITY=0` 关闭拆分做 A/B。
- 每个 program 将该 Q16 block 的 query 载入一次，并循环自己负责的 CSR K16 片段，逐块执行 `16×d` 与 `d×16` 的 QK MMA；更紧的 `192/384` bucket 同时降低长行尾部 padding。
- 采用 online softmax，持续更新行最大值、指数和以及 value accumulator。
- 最终输出 `(O_res, LSE_res)`。

该 kernel 的优势是查询粒度小、选择灵活，并且同一 Q16 的多个 K16 tiles 在一个 program 内完成，减少逐 microtile launch 的开销。当前 SM90、BF16、head dimension 128 的编译路径已经验证。

### 3.3 Residual 快速路径与 Exact LSE merge

若 Residual 完全为空，代码直接返回 Core 输出，不启动 Residual kernel，也不执行 merge。若仅少数 Q16 rows 非空，Residual kernel 只分配 `[active_rows,16,head_dim]` 和 `[active_rows,16]` 的 compact 输出/LSE；merge kernel 同样只遍历 compact `residual_active_rows`，并原地更新 Core 输出。不再为 Residual 或 merged output 分配、清零完整 `[head,sequence,head_dim]` tensor。

Core 与 Residual 的支持集互不重叠，但两边各自做了局部 softmax，因此不能直接相加。设两边输出和自然对数域 LSE 分别为：

```text
(O_c, L_c), (O_r, L_r)
```

令：

```text
m   = max(L_c, L_r)
w_c = exp(L_c - m)
w_r = exp(L_r - m)
O   = (w_c O_c + w_r O_r) / (w_c + w_r)
```

这等价于在 `Core ∪ Residual` 上一次性执行 softmax，不是近似 merge。FlashInfer 返回的 LSE 若为 log2 域，会先乘 `ln(2)` 转成自然对数域再合并。

## 4. 关键参数

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `FLASHINFER64_ROUTE_MODE` | `topk_topp` | Core 使用 macro ratio Top-k，Residual 使用 total Top-p |
| `FLASHINFER64_TILE_TOP_RATIO` | `0.25` | 每个 head/Q128 选择的 K96 block 比例 |
| `FLASHINFER64_TOKEN_TOP_P` | `0.9` | Core 与 Residual 的目标总 fine-score 质量 |
| `FLASHINFER64_PROMOTION_THRESHOLD` | `24` | 一个 macro 中达到多少个 residual microtiles 后提升为 Core，范围 1–48 |
| `FLASHINFER64_ROUTE_CACHE` | `False` | 是否复用 compact route；direct FA3 路径同时复用对应 plan，回退路径仍逐次 plan |
| `FLASHINFER64_WORKSPACE_MB` | `128` | 全模型共享的 FlashInfer float workspace 大小 |
| `FLASHINFER64_DIRECT_MACRO_CSR` | `True` | FA3 上启用 compact macro CSR 和每层 plan cache；否则使用 VariableBlock 回退路径 |
| `FLASHINFER64_PARALLEL_CORE_RESIDUAL` | `False` | CUDA 上让 Core 与 Residual 分别提交到两个 stream；最终 LSE merge 仍在等待两者完成后执行 |
| `FLASHINFER64_PLAN_WORKSPACE_MB` | `8` | direct 路径每层的 integer scheduler workspace |
| `FLASHINFER64_CORE_ONLY` | `False` | 消融/测速：只运行 macro Core，跳过 Residual selection、promotion、kernel 和 merge |
| `FLASHINFER64_RESIDUAL_SPLIT_MIN_CAPACITY` | `128` | 从哪个 bucket 开始把长 Residual 行拆成多个 program；设为 `0` 恢复单 program/行基线 |
| `FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER` | C++ 为 `1`；DFS wrapper 为 `0` | direct CSR 展开 kernel 的 CTA 数；正整数 `N` 表示 `min(batch_size, N × SM数)`，`0` 表示一行一个 CTA。wrapper 会将实际值写入 FlashInfer output dir |

旧参数 `FLASHINFER64_TOKEN_TOP_RATIO` 和 `FLASHINFER64_TOKEN_TOP_K` 仅为命令兼容保留，当前论文主路径的 Residual 由 `FLASHINFER64_TOKEN_TOP_P` 控制。

## 5. 计时与实验观测

实现为主要阶段分别记录计时：

| 计时 phase | 内容 |
|---|---|
| `flashinfer_qkv_permute` | 视频 token 排序与 Q/K/V 重排 |
| `flashinfer_fine_score` | Q16×K16 fine score |
| `flashinfer_core_score` | 8×6 fine-to-macro 聚合 |
| `flashinfer_core_select` | macro Top-k/Top-p |
| `flashinfer_residual_select` | 非 Core microtile 的绝对质量补足 |
| `flashinfer_promotion` | residual occupancy promotion |
| `flashinfer_residual_compact` | Residual bool mask 压缩及 interaction 统计 |
| `flashinfer_plan` | FlashInfer mask 展开与 plan |
| `flashinfer_core_csr_expand` | direct 路径中 K96 macro bases 到 vector offsets 的 CUDA 展开；包含在 Core run 总阶段内，勿重复求和 |
| `flashinfer_core_run` | FlashInfer Core kernel |
| `flashinfer_residual_micro_run` | 全部 length-bucket Triton Residual kernels |
| `flashinfer_residual_bucket_leN` | Residual 长度不超过 N 的单桶 kernel；与上一总阶段嵌套，不应重复求和 |
| `flashinfer_lse_merge` | exact LSE merge |
| `flashinfer_output_unpermute` | 恢复原 token 顺序 |

同时记录 Core token interactions、Residual token interactions、Residual microtile 数、promotion macro 数、最终 density 和 sparsity。`timing.csv` 的 `residual_route_stats` 行以及逐层 `sparsity_records.csv` 还记录 `residual_count_mean/p50/p95/max/nonempty_ratio`，用于判断 Residual 长尾和空行比例。实验分析时应将“选择与 plan 开销”和“两类 kernel 的实际运行时间”分开，避免只用总稀疏率推断加速效果。

## 6. 方法定位

这套方法不是简单地把 Top-k 和 Top-p 串联，而是进行硬件感知的分工：

- macro Top-k/Top-p 提供规则性，匹配 FlashInfer 的高吞吐 block-sparse 路径；
- micro total Top-p 提供细粒度补偿，以较低额外质量恢复大块量化丢失的信息；
- occupancy promotion 根据局部结构动态选择更合适的 kernel；
- exact LSE merge 保证最终结果严格对应所选稀疏支持集。

因此其主要创新叙事可以概括为：**分层质量路由（hierarchical mass routing）与密度自适应异构 kernel 调度（density-adaptive heterogeneous kernel dispatch）相结合。**

`FLASHINFER64_CORE_ONLY=True` 是判断方法是否值得继续优化的关键消融：先把 Core density 调到与 DFSAttn 相同，比较 `flashinfer_core_run + 摊销后的 flashinfer_plan`；只有 Core 明显快于 DFSAttn，才有预算容纳 Residual 和 exact merge。该开关不是最终算法，也不用于掩盖质量差异。

## 7. RoDe center-token residual 实验（PK 路径）

为了单独测量 1×1 token residual 在 RoDe 上的性能，新增了一个与现有
Q16×K16 Triton residual 隔离的实验后端。该后端只允许用于本文前面定义的
`topk_topp`（PK）路径：Core 仍由 FlashInfer 执行，Residual 的每个已选
Q16×K16 tile 只取其中心 token，约定偶数长度 16 的中心为 0-based
`offset=8`。

当前实验开关为：

```bash
FLASHINFER64_ROUTE_MODE=topk_topp \
FLASHINFER64_RESIDUAL_BACKEND=rode_center \
FLASHINFER64_RODE_CACHE=True \
...
```

默认值 `FLASHINFER64_RESIDUAL_BACKEND=micro` 保持原有 residual 行为；
`FLASHINFER64_RODE_CACHE=False` 保持原有的低驻留显存行为。开启
`FLASHINFER64_RODE_CACHE=True` 时，不改变 `FLASHINFER64_ROUTE_CACHE=False`
的全局设置，而是只在 `rode_center` 路径保存当前层所需的 Core 支持、Residual
CSR/中心映射和 RoDe plan；按照 `CACHE_INTERVAL` 的 refresh step 更新，其余
step 复用。为保证 Core/Residual 支持集互斥，不能只缓存 Residual indices；
Core mask 也必须作为同一轻量 route 的一部分保存。缓存不保存 Q/K/V，也不保存
普通 micro residual 的 bucket 和完整 dense mask。
`fine_topk_occupancy` 和 tau 统计分支不会自动切换到 RoDe，若误用
`rode_center` 会直接报错。

RoDe 源码下载并保存在：

```text
third_party/RoDe/RoDe/
```

其中 `RoDe_SDDMM/RoDeSddmm.cu` 和 `RoDe_SpMM/RoDeSpmm.cu` 作为原始
RoDe 计算 kernel 直接编译；项目新增的 `dfsattn/rode_bridge.cu` 只负责
PyTorch tensor bridge、调用 RoDe 官方 row decomposition 所需的描述数组，
没有重写 RoDe 的 SDDMM/SpMM kernel。由于原始 RoDe API 只提供 float n128
版本，实验后端会将 Q/K/V 转为 FP32；转换单独记录为
`flashinfer_residual_rode_fp32`，不能把它从端到端时间中删除。

每个 residual microtile `(h, qb, kb)` 被映射为一个 CSR nonzero：

```text
row = h * sequence + qb * 16 + 8
col = h * sequence + kb * 16 + 8
```

随后执行：

```text
RoDe SDDMM(Q, K) → 独立 CSR row softmax → RoDe SpMM(P, V)
```

最后只把每个 Q16 行的中心 query token 与 Core 结果做 exact LSE merge，
Q16 中另外 15 个 token 不注入该 Residual 结果。这是一个性能实验路径，
不等价于当前完整 Q16×K16 residual 的质量覆盖范围。

RoDe 的原始接口要求 regular/residue row descriptors；本实验分别按官方
SDDMM 的 32-element segment 和 SpMM 的 512-element segment 组织 CSR 行。
针对 1×1 residual 可能出现 regular 或 residue 分区为空的情况，只在下载的
RoDe launcher 外加了 zero-grid guard，避免 CUDA 用 0-sized grid 启动；计算
kernel 仍来自 RoDe 源码。

首次运行 `rode_center` 时，`dfsattn/rode_backend.py` 会通过
`torch.utils.cpp_extension` 编译并缓存扩展；可用
`DFSATTN_RODE_BUILD_DIR` 指定缓存目录。实验日志新增：

| phase | 含义 |
|---|---|
| `flashinfer_residual_rode_plan` | RoDe CSR row descriptor 与扩展 plan 初始化 |
| `flashinfer_residual_rode_fp32` | Q/K/V BF16 → FP32 |
| `flashinfer_residual_rode_sddmm` | RoDe 原始 SDDMM |
| `flashinfer_residual_rode_softmax` | CSR row softmax |
| `flashinfer_residual_rode_spmm` | RoDe 原始 SpMM |
| `flashinfer_residual_rode_run` | 上述 residual 后端总阶段，包含嵌套 phase，不应与子项重复相加 |

建议比较 RoDe 时同时记录 `flashinfer_residual_rode_plan` 的调用次数；若
`FLASHINFER64_ROUTE_CACHE=False` 且 `FLASHINFER64_RODE_CACHE=False`，该阶段
会在每次 sparse layer call 中重新构建。开启独立 RoDe cache 后，plan 只在
route refresh 时重建，目标是摊薄 plan 初始化而不保留普通 residual 执行所需的
额外 bucket 数据。

当前 GPU 验证已通过：在 GH200、CUDA 12.4、head_dim=128 上，官方 RoDe
SDDMM/SpMM 的 3-edge 中心 token CSR smoke test 与 FP32 reference 的最大误差
分别约为 `3.8e-6` 和 `1.2e-7`；Python 侧 `rode_center` 集成路径也已验证，
中心 token 输出最大误差为 `0`，LSE 最大误差约为 `3.0e-8`。这只是后端和
数值路径验证，不代表已经完成整段 HyVideo 端到端性能结论；实际比较时应单独
报告 FP32 转换、CSR softmax、RoDe SDDMM/SpMM、center-only merge 以及首次
扩展编译/plan 的成本。

独立 RoDe cache 的 GH200 集成测试也已通过：在
`FLASHINFER64_ROUTE_CACHE=False`、`FLASHINFER64_RODE_CACHE=True` 时，首次
refresh 构建 route/plan，后续非 refresh 调用复用同一个轻量 route 和 direct
Core plan，plan 构建次数为 1；缓存 route 的 `residual_buckets` 已释放。

### 7.1 Core/Residual 双 CUDA stream 实验

默认 Core 和 Residual 仍按串行路径执行。设置：

```bash
FLASHINFER64_PARALLEL_CORE_RESIDUAL=True
```

后，代码先在调用方当前 stream 上提交 Core，再在持久化的第二个 CUDA
stream 上提交 Residual；第二个 stream 会等待 Q/K/V 重排、route 和 Core plan
准备完成，但不会等待 Core kernel 本身。两边完成后，调用方 stream 等待
Residual stream，再执行最终 merge。因此该开关只改变调度，不改变 Core 和
Residual 的支持集或 merge 数学定义。默认值为 `False`，便于和原来的串行
baseline 对比；输出目录名称会包含 `parallelTrue/False`。

该选项同时适用于原有 `micro` Residual 和 `rode_center` Residual，但仅在
CUDA、Residual 非空且未启用 `CORE_ONLY` 时生效；CPU 路径或空 Residual 会
自动回到普通执行路径。双 stream smoke test 已在 GH200 上通过，原有 Python
回归测试为 `26 passed`。是否实际加速仍取决于 Core 与 Residual kernel 的
SM/显存带宽占用，必须比较相同输入下的 `e2e_generation_gpu`、
`flashinfer_core_run`、Residual 总阶段和 `flashinfer_lse_merge`，不能把两个
可能重叠的 phase 时间直接相加。
