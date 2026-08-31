# 分层 Top-k/Top-p 与异构 Kernel 稀疏注意力：当前实现概述

## 1. 核心思路

当前新增的方法将一次注意力拆成两个互不重叠的计算分支：

- **Core**：以逻辑块 `Q128 × K96` 为单位选择规则、密集的大块，交给 FlashInfer block-sparse attention。
- **Residual**：以 `Q16 × K16` 为单位补充 Core 没覆盖但质量较高的细粒度区域，交给 grouped Triton/MMA kernel。
- **Merge**：两个分支分别计算输出和 LSE，最后用 exact LSE merge 得到与二者并集完全等价的 softmax 结果。

这样做的目的，是让规则的大块利用 FlashInfer 的高吞吐，同时用较小的 microtile 避免大块选择造成的精度损失。这里的 `Q128 × K96` 是传给 FlashInfer 的**逻辑稀疏块尺寸**；FlashInfer 内部实际采用多少 CTA、warp 以及如何切分，由其 SM90 kernel 自行决定，不能简单等同为一个物理 CTA。

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

## 3. 底层执行

### 3.1 Core：FlashInfer variable block-sparse attention

Core mask 的形状为：

```text
[head, ceil(S/128), ceil(S/96)]
```

每个 block 的 row size 为 128、column size 为 96，序列尾块使用实际长度。FlashInfer 的 `VariableBlockSparseAttentionWrapper.plan()` 将 macro mask 展开成 token-level sparse indices，随后 `run()` 计算：

```text
(O_core, LSE_core)
```

为解决此前的 OOM 和 `_vector_sparse_indices_buffer is not large enough`：

- 全模型只共享一个 FlashInfer wrapper 和 workspace，不再为每层永久保存 wrapper。
- 每次调用都重新 plan，并覆盖上一层的 expanded plan；expanded plan 不做 12 步缓存。
- vector sparse indices/indptr 在 plan 前按当前 Core mask 的实际展开规模精确分配。
- CUDA 路径不在每层保留完整 Residual bool matrix。
- `FLASHINFER64_ROUTE_CACHE=False` 为默认的低峰值显存模式；此时 compact route 也在每次调用后释放。

因此 `CACHE_INTERVAL=12` 不再意味着 FlashInfer expanded plan 被保留 12 步。若显式设置 `FLASHINFER64_ROUTE_CACHE=True`，只复用 Core mask 和 Residual compact indices，FlashInfer plan 仍会逐次重建。

### 3.2 Residual：Q16×K16 grouped Triton/MMA

Residual bool mask 会先压缩成真正的 CSR：

```text
indices : 所有被选 K16 block 编号的连续一维数组
indptr  : 每个 (head,Q16) 行在 indices 中的起止位置
```

CUDA 上不再保留原始三维 bool mask，也不再把所有行 padding 到全局最大 Residual 长度。非空 CSR 行按照实际 K16 数量进入 `≤4/8/16/32/64/...` 的 length buckets。

Triton kernel 的执行方式为：

- 一个 program 对应当前 bucket 中的一个非空 `(head, Q16 block)`；空行不启动 program。
- program 将该 Q16 block 的 query 载入一次。
- 循环其 CSR K16 列表，逐块执行 `16×d` 与 `d×16` 的 QK MMA；桶内 padding 有固定小上界，不受全局异常长行影响。
- 采用 online softmax，持续更新行最大值、指数和以及 value accumulator。
- 最终输出 `(O_res, LSE_res)`。

该 kernel 的优势是查询粒度小、选择灵活，并且同一 Q16 的多个 K16 tiles 在一个 program 内完成，减少逐 microtile launch 的开销。当前 SM90、BF16、head dimension 128 的编译路径已经验证。

### 3.3 Exact LSE merge

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
| `FLASHINFER64_ROUTE_CACHE` | `False` | 是否复用 compact route；不影响 FlashInfer 每次重新 plan |
| `FLASHINFER64_WORKSPACE_MB` | `128` | 全模型共享的 FlashInfer float workspace 大小 |

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
