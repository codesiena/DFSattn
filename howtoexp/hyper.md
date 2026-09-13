## FlashInfer64 路由参数说明

这些参数在 `hyvideo_t2v_720p_dfs.sh` 中通过 `FLASHINFER64_*` 环境变量设置，并传给 `--sparse_execution flashinfer64` 的 FlashInfer64 后端。当前脚本默认使用：

```bash
FLASHINFER64_ROUTE_MODE=fine_topk_occupancy
FLASHINFER64_FINE_TOP_RATIO=0.2
FLASHINFER64_TOKEN_TOP_P=0.9
FLASHINFER64_PROMOTION_THRESHOLD=24
```
FlashInfer64 的基本划分是：先在细粒度的 `Q16 x K16` micro-tile 上选择候选，再把每个 `Q128 x K96` macro-tile 内被选中的 micro-tile 数量统计为 occupancy；occupancy 达到阈值的 macro-tile 走 Core，其余候选走 Residual。

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `FLASHINFER64_TOP_P` | `0.16` | 旧版 `topp_topk` 路由的 macro Top-p 累积概率阈值。对每个 head 和每个 `Q128`，按 `Q128 x K96` macro 分数从高到低选择，直到累计分数达到该比例。当前默认的 `fine_topk_occupancy` 路由不使用此参数。 |
| `FLASHINFER64_TOKEN_TOP_RATIO` | `0.10` | 旧版 residual token Top-k 比例参数。当前实现中该参数仅为旧启动脚本保留，在 FlashInfer64 调用内部会被丢弃；当前默认路径不生效。不要用它控制当前 Residual 数量。 |
| `FLASHINFER64_ROUTE_MODE` | `fine_topk_occupancy` | 路由策略。可选值为 `topp_topk`、`topk_topp` 和 `fine_topk_occupancy`：`topp_topk` 先做 macro Top-p；`topk_topp` 先按 macro Top-k 比例选 Core，再做 residual；`fine_topk_occupancy` 直接在 `Q16 x K16` 级别做 Fine Top-k，再按 occupancy 划分 Core/Residual。 |
| `FLASHINFER64_TILE_TOP_RATIO` | `0.25` | `topk_topp` 路由使用的 macro Top-k 比例，即每个 head/`Q128` 选择约 `25%` 的 `K96` macro-tile。当前默认的 `fine_topk_occupancy` 路由不使用此参数。 |
| `FLASHINFER64_FINE_TOP_RATIO` | `0.2` | `fine_topk_occupancy` 路由中，每个 head/`Q16` 选择的 `K16` micro-tile 比例。`0.2` 表示选择约 `20%` 的 K-side micro-tile 作为 Fine Top-k 候选。若设置了 `FLASHINFER64_FINE_TOP_K`，则使用固定数量而不是该比例。 |
| `FLASHINFER64_FINE_TOP_K` | 空 | Fine Top-k 的固定数量覆盖值。设置为正整数后，每个 head/`Q16` 固定选择这么多个 `K16` micro-tile；不设置时使用 `FLASHINFER64_FINE_TOP_RATIO`。该变量为空时脚本不会向 Python 传入 `--flashinfer64_fine_top_k`。 |
| `FLASHINFER64_TOKEN_TOP_P` | `0.9` | 旧版 `topp_topk`/`topk_topp` 路由的 residual 总质量目标：在已经选入 Core 的 micro-tile 之外继续选择 Residual，使 Core+Residual 覆盖约 `90%` 的 proxy attention mass；剩余质量不会重新归一化。当前默认的 `fine_topk_occupancy` 路由不使用该总质量选择。 |
| `FLASHINFER64_PROMOTION_THRESHOLD` | `24` | macro occupancy 晋升阈值。一个 `Q128 x K96` macro-tile 最多包含 `8 x 6 = 48` 个 `Q16 x K16` micro-tile；当其中至少 `24` 个被 Fine Top-k 选中时，该 macro-tile 晋升为 Core，否则保留为 Residual 候选。阈值越低，Core 覆盖通常越多、Residual 越少；阈值越高则相反。有效范围是 `1` 到 `48`。 |

### 当前默认路径的实际生效关系

当不额外设置环境变量时，实际使用的是：

```text
Fine Top-k 候选：FINE_TOP_RATIO=0.2
Core/Residual 划分：occupancy >= PROMOTION_THRESHOLD=24
TOP_P：不使用
TILE_TOP_RATIO：不使用
TOKEN_TOP_P：不使用
TOKEN_TOP_RATIO：已废弃、不使用
```

因此，如果目的是调节当前默认的 `fine_topk_occupancy` 实验，主要应调 `FLASHINFER64_FINE_TOP_RATIO`、`FLASHINFER64_FINE_TOP_K` 和 `FLASHINFER64_PROMOTION_THRESHOLD`。如果改用 `topp_topk` 或 `topk_topp`，才需要重点调 `FLASHINFER64_TOP_P`、`FLASHINFER64_TILE_TOP_RATIO` 和 `FLASHINFER64_TOKEN_TOP_P`。
