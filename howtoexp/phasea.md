# Block Top-p + Token Top-k Oracle Feasibility Experiment

## 1. 实验目标

在 DFSAttn 的标准 benchmark 样本上直接测试：

> 将原来的 block Top-k 改为更灵活的 block Top-p，并在被 block Top-p 删除的区域中补充少量 token-level Top-k，是否可以在更高最终 sparsity 下保持或提高生成视频质量。

第一阶段测试：

```text
Full Attention
DFSAttn baseline
Block Top-p only
Block Top-p + Token Top-k
```

Token Top-k 使用 exact QK score 作为 oracle。

---

## 2. Block Top-p 的具体实现

DFSAttn 当前先计算 sub-block attention score，再将 sub-block score 聚合为 128×128 block importance score。其 hierarchical scoring 本质上是将更细粒度的 sub-block attention 信息聚合成 block ranking score。

设已有：

```python
block_score
# [B, H, num_q_blocks, num_k_blocks]
```

其中每个位置对应一个：

```text
128 query tokens × 128 key tokens
```

的 block。

### Step 1：转换成每个 query block 上的概率分布

对 key-block 维度 normalize：

```python
block_prob = block_score / (
    block_score.sum(dim=-1, keepdim=True) + 1e-8
)
```

因此对每个：

```text
batch × head × query_block
```

都有：

[
\sum_j p_{ij}^{block}=1.
]

### Step 2：按照 block probability 从大到小排序

```python
sorted_prob, sorted_idx = torch.sort(
    block_prob,
    dim=-1,
    descending=True,
)

cum_prob = torch.cumsum(
    sorted_prob,
    dim=-1,
)
```

### Step 3：Top-p 选择最小 block 集合

对于给定 `p_mass`，保留最小数量的 key blocks，使：

[
\sum_{j\in S_i}p_{ij}^{block}\ge p_{mass}.
]

实现：

```python
keep_sorted = (
    cum_prob - sorted_prob
) < p_mass
```

然后 scatter 回原来的 block index：

```python
block_mask = torch.zeros_like(
    block_prob,
    dtype=torch.bool,
)

block_mask.scatter_(
    dim=-1,
    index=sorted_idx,
    src=keep_sorted,
)
```

这与 SpargeAttention2 中 Top-p 的定义一致：对每一行保留累计概率达到给定阈值的最小集合。

需要注意：

```text
p_mass 不是 sparsity。
```

例如：

```text
p_mass = 0.8
```

表示保留能够覆盖约 80% block importance mass 的最小 block 集合。

每个设置最终都需要实际统计：

```text
realized_block_sparsity
```

---

## 3. Token Top-k residual

Block Top-p 得到：

[
M_{block}.
]

对于每个 query token，只在 **没有被 block Top-p 保留的 key tokens** 中寻找 Top-k。

首先计算 exact token score：

[
S_{ij}
======

Q_iK_j^T/\sqrt d.
]

然后将 block Top-p 已经覆盖的位置 mask 掉：

```python
residual_score = score.masked_fill(
    block_selected_token_mask,
    float("-inf"),
)
```

再选择：

```python
_, residual_idx = torch.topk(
    residual_score,
    k=k_token,
    dim=-1,
)
```

测试：

```text
k_token = 4
k_token = 8
k_token = 16
k_token = 32
```

最终 attention mask：

[
M_{hybrid}
==========

M_{block}
\cup
M_{token}.
]

Token Top-k 必须来自 block mask 的 complement，因此不会重复计算已经保留的 block token。

---

## 4. Hybrid attention

Oracle 实验可以直接根据最终 union mask 计算 attention。

对于每个 query：

```python
hybrid_score = score.masked_fill(
    ~hybrid_mask,
    float("-inf"),
)

hybrid_prob = torch.softmax(
    hybrid_score,
    dim=-1,
)

hybrid_output = hybrid_prob @ V
```

重点是 block tokens 和 residual tokens 必须放在 **同一个 softmax** 中。

---

## 5. 实验 sweep

先使用 DFSAttn 当前标准 benchmark pipeline，选固定的一批 prompt，例如 10～20 个，所有方法保持完全相同的：

```text
prompt
seed
resolution
sampling steps
scheduler
CFG
```

建议测试：

```text
p_mass:
0.3
0.4
0.5
0.6
0.7
0.8
0.9
```

因为不同 head/layer/timestep 的 attention distribution 不一样，不提前假设某个 `p_mass` 对应固定 sparsity。

每个 `p_mass` 再测试：

```text
k_token:
0
4
8
16
32
```

其中：

```text
k_token = 0
```

就是 Block Top-p only baseline。

---

## 6. Sparsity 统计

必须按照真实保留的 QK interactions 计算最终 sparsity。

对于 query token (i)：

[
C_i
===

128\times B_i+k_i,
]

其中：

* (B_i)：该 query 所属 query block 保留的 key blocks 数量；
* (k_i)：实际加入的 residual token 数量。

因此：

[
C_{total}
=========

\sum_i C_i,
]

最终 sparsity：

[
S
=

1-
\frac{C_{total}}{N^2}.
]

实验结果同时记录：

```text
p_mass
realized_block_sparsity
k_token
final_hybrid_sparsity
```

这样可以直接判断增加少量 token 后，是否仍显著高于原 DFSAttn 的 sparsity。

---

## 7. 视频质量评估

直接沿用 DFSAttn 的评估方式。

至少记录：

```text
PSNR ↑
SSIM ↑
LPIPS ↓
```

都以相同 prompt 和 seed 下的 Full Attention 视频作为 reference。DFSAttn 原实验也是用 PSNR、SSIM、LPIPS 来衡量 sparse generation 相对于 full attention 的 fidelity。

如果现有 evaluation pipeline 已支持 VBench，则同时保留已有 VBench 指标。

---

## 8. 最终结果表

最终主要需要一张：

| Method              | p_mass | Token k | Block Sparsity | Final Sparsity | PSNR | SSIM | LPIPS |
| ------------------- | -----: | ------: | -------------: | -------------: | ---: | ---: | ----: |
| Full Attention      |      - |       - |             0% |             0% |    - |    - |     - |
| DFSAttn baseline    |      - |       0 |       baseline |       baseline |      |      |       |
| Block Top-p         |    0.8 |       0 |                |                |      |      |       |
| Block Top-p + Token |    0.8 |       4 |                |                |      |      |       |
| Block Top-p + Token |    0.8 |       8 |                |                |      |      |       |
| Block Top-p + Token |    0.8 |      16 |                |                |      |      |       |
| Block Top-p         |    0.6 |       0 |                |                |      |      |       |
| Block Top-p + Token |    0.6 |       8 |                |                |      |      |       |
| Block Top-p + Token |    0.6 |      16 |                |                |      |      |       |

核心观察是：

> 当降低 `p_mass`、显著减少完整 128×128 blocks 后，加入 k=4/8/16 的 token residual，能否把 PSNR/SSIM/LPIPS 恢复到原 DFSAttn 水平，同时最终 sparsity 仍明显更高。

如果存在这样的点，就证明：

[
\boxed{
\text{少量 token residual}
\Rightarrow
\text{可以进一步 aggressive 地删除完整 blocks}
}
]

这就是后续设计低成本 token selector 和 Tensor Core / CUDA Core hybrid kernel 的直接依据。
