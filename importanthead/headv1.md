# 当前重要 `(Layer, Head)` 候选：v1

## 数据来源

本表来自：

```text
视频：prompt 0
Diffusion step：12
Core：Q128 × K96，Macro Top-k ratio = 0.16
Layer：0–59，共60层
Head：0–23，共24个 head
```

每个 Layer 从全部 `(Head, Q16)` 中选取：

- 64 个 `high_error` Q16；
- 64 个 `low_error_control` Q16。

对每个采样 Q16，逐个补回所有遗漏 K16，并定义：

```text
best_delta = 该 Q16 所有遗漏 K16 中最大的 DeltaE
```

下面的 Layer/Head 排名按 `high_error` 样本中的 `best_delta` 平均值排序。

## 当前候选排名

| Layer | Head | high-error Q16 数 | 平均 best ΔE | 最大 best ΔE | 平均 Eq | 可信度备注 |
|---:|---:|---:|---:|---:|---:|---|
| 43 | 17 | 40 | 0.871 | 1.378 | 1.376 | 样本较多，当前最稳定候选 |
| 30 | 13 | 30 | 0.837 | 1.058 | 1.585 | 样本较多，当前强候选 |
| 48 | 5 | 7 | 0.785 | 0.975 | 1.274 | 强度高，但样本较少 |
| 8 | 10 | 3 | 0.607 | 0.694 | 1.014 | 强度高，但样本很少 |
| 12 | 5 | 5 | 0.568 | 0.859 | 1.999 | 强度高，但样本很少 |

因此，当前优先级为：

```text
Layer 43 / Head 17
Layer 30 / Head 13
Layer 48 / Head 5
Layer 8  / Head 10
Layer 12 / Head 5
```

## 代表性高恢复案例

当前最大的单个 K16 恢复案例为：

```text
Layer 43 / Head 17
Q16 = 1993
遗漏 KV Macro = 332
best K16 = 1993
Eq: 1.866 -> 0.488
DeltaE(K16) = 1.378
恢复完整 Macro add-back 收益的约99.5%
```

其他代表性案例包括：

```text
Layer 43 / Head 17 / Q16=1996 / DeltaE=1.246
Layer 43 / Head 17 / Q16=2005 / DeltaE=1.183
Layer 43 / Head 17 / Q16=1289 / DeltaE=1.176
Layer 30 / Head 13 / Q16=2694 / DeltaE=1.058
Layer 48 / Head 5  / Q16=1932 / DeltaE=0.975
```

## 全局统计

```text
Layer 快照：60
采样 Q16：7680
独立 K16 add-back：17,971,200
正 DeltaE 比例：89.24%
```

对每个遗漏 Macro 内最优 K16 的恢复比例：

```text
mean   = 56.0%
median = 39.3%
P90    = 91.8%
```

这说明重要 Macro 的恢复收益通常可以由少量 K16 获得，但当前 mean-pooled proxy 选择的 K16 仍不是最优：

```text
Dense mass top：mean 38.0%，median 34.7%
当前 proxy top：mean 28.9%，median 25.9%
```

## 使用限制

这些候选是 **Step12、prompt0、Core ratio=0.16 条件下的高误差样本候选**，不能直接解释成全视频的发生概率。

原因是每层的 64 个 `high_error` Q16 是按误差筛出来的，并非均匀随机抽样；因此高恢复事件被过采样。`Layer 48/Head 5`、`Layer 8/Head 10` 和 `Layer 12/Head 5` 的样本数尤其少，只适合作为待验证候选。

本表适合用于：

1. 下一轮 scorer 的 Layer/Head 优先级；
2. 对这些 Head 分配更高的 Residual 检测预算；
3. 继续使用独立 prompt、seed 和 step 做验证。

本表不应被用作固定的 KV 位置先验。当前实验显示最佳 K16 坐标随视频内容和路由配置变化，Layer/Head 风险比固定 K16 ID 更稳定。
