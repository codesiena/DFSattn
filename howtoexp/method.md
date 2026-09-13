# HyMoR-Attn：硬件感知 Macro–Micro 粒度分配稀疏注意力

> 工作名称：**HyMoR-Attn (Hardware-aware Macro–Micro Routing Attention)**
>
> 文档用途：论文叙事、方法实现和实验设计的统一说明。
>
> 当前状态：机制实验已经完成；主体异构执行路径已有实现；Peak-aware scorer 和风险预算规则仍待实现与验证。
>
> 重要约定：本文严格区分“已有实验事实”“方法设计”和“预期结果”。预期结果不是已经取得的实验结论。

## 1. 一句话方法

HyMoR-Attn 将视频扩散模型中的稀疏注意力拆成两个互不重叠的部分：

1. 使用固定比例的 `Q128×K96` Macro Top-k 捕获主体注意力，并通过 FlashInfer FA3 执行规则 Core；
2. 在 Core 补集中，使用保留块内峰值的 `Q16×K16` proxy 按原始概率质量补足固定 total Top-p，并由离线得到的 Layer/Head 风险先验提供 Top-k 安全下界；
3. 当局部 Residual 足够稠密时，将其提升为完整 Macro；其余稀疏 Microtiles 由 grouped Triton/MMA kernel 执行；
4. Core 和 Residual 分别产生输出与 LSE，最后通过 exact online-softmax merge 得到与支持集并集严格等价的结果。

核心目标不是单纯降低 Core 比例，也不是在 DFSAttn 上附加一条补救分支，而是让 **attention support 的局部结构与 GPU 执行粒度匹配**：规则且局部稠密的区域使用 `Q128×K96` Macro kernel，分散但重要的交互使用 `Q16×K16` Micro kernel。更合适的粒度分配减少无效块内计算和不规则执行开销，使系统能够在更短端到端时间内保留更多高价值 attention interactions，从而同时获得更快推理和更高视频质量，形成对纯 DFSAttn 的 Pareto dominance。

---

## 2. 研究问题与论文主张

### 2.1 现有 DFSAttn 的结构性矛盾

DFSAttn 已经利用 `Q16×K16` 的细粒度信息改善大块选择，但最终执行仍量化为单一规则大块。这种设计有利于高 occupancy 区域的 GPU 吞吐，却无法同时适配两种结构不同的 workload：

- 局部稠密区域适合 Macro Tensor-Core kernel；若拆成大量 Microtiles，会增加索引、调度和循环开销；
- 局部稀疏区域只含少数高价值 interactions；若强制提升为完整 Macro，会执行大量低价值 QK；
- 为了避免遗漏这些分散交互而统一增大大块预算，又会进一步放大无效块内计算。

因此，纯 DFSAttn 的 Pareto frontier 受到“所有支持都采用同一执行粒度”的约束。HyMoR-Attn 的核心假设是：

> 如果先选择高价值 attention support，再根据每个局部区域的 occupancy 将其动态映射到 Macro 或 Micro kernel，就能降低单位有效 interaction 的执行成本；节省的硬件时间可以反过来用于保留更多高价值交互，最终同时改善端到端速度和视频质量。

### 2.2 计划形成的主要论文主张

论文最终应围绕以下三个主张组织，而不是仅叙述为“Top-k 加 Top-p”。

**主张一：Attention support 同时包含局部稠密主体和分散高价值残差。** 绝大多数遗漏 Macro 影响很小，但少数 Macro 内存在恢复收益极高的 K16；补回少量 K16 即可获得完整 Macro add-back 的大部分收益。这种非均匀结构说明单一大块执行粒度并非计算最优。

**主张二：风险位置在 Layer/Head 层面稳定，在具体 Token/Tile 坐标层面动态。** 因此先验适合决定“在哪里投入更强的检测和更多预算”，不适合直接指定固定 K 位置。

**主张三：质量选择与硬件粒度分配应当协同设计、职责分离。** Macro Top-k 与 Micro total Top-p 决定“哪些 interactions 值得保留”，occupancy-aware dispatcher 决定“这些 interactions 用哪种粒度执行”。这种联合优化使同一质量 support 获得更低执行时间，也允许在相同时间预算下选择质量更高的 support；exact LSE merge 保证两种粒度仍对应同一个稀疏 softmax。

贡献可概括为：

1. 揭示并量化视频 DiT 粗粒度 sparse attention 的 micro-recoverable long-tail error；
2. 提出风险门控、峰值保真的分层质量路由；
3. 提出与 H100 FA3 工作块对齐、由局部 occupancy 驱动的 Macro/Micro 粒度分配，以及严格 softmax 合并；
4. 在 matched latency 与 matched quality 协议下，同时获得更低端到端时间和更高视频质量，推进纯 DFSAttn 的 Pareto frontier。

---

## 3. 已有机制证据

### 3.1 证据一：Macro 漏选误差由少量 Microtile 主导

早期 `Q128×K128` 实验对遗漏 Macro 逐个 add-back，得到以下结果：

| Microtile 选择 | Top1 K16 | Top2 K16 | Top4 K16 |
|---|---:|---:|---:|
| Dense oracle 恢复完整 K128 收益 | 51.5% | 73.9% | 90.4% |
| 当前 proxy | 35.0% | 54.5% | 78.0% |
| 随机 | 13.2% | 26.2% | 51.5% |

这说明完整 Macro add-back 的收益通常集中在少数 K16 上。Residual 采用 `Q16×K16` 是有明确机制依据的，而不是任意选择的粒度。

该实验还发现遗漏 Macro 的恢复收益具有强长尾：一半遗漏 Macro 的收益低于 `0.00037`，但最严重单个 Macro 的 `ΔE` 可达到 `0.7603`。因此均匀增大所有 Query 的 Macro k 并不经济，应该做定点修复。

### 3.2 证据二：硬件对齐 K96 Core 后，严重遗漏仍然存在

正式 `Q128×K96` Core 实验覆盖三个视频、三个 diffusion steps、全部 60 个 attention layers 和 24 个 heads，并分别测试 Core ratio `0.25/0.20/0.16`。每个被采样 Q16 的所有 Core 补集 K16 均被独立 add-back。早期结果只覆盖 Layer `{9,12,15}`，不能用来确定全模型的 Layer/Head 风险先验。

| 指标 | Core 0.25 | Core 0.20 | Core 0.16 |
|---|---:|---:|---:|
| 严重遗漏最佳单 K16 平均 `ΔE` | 0.3469 | 0.3966 | 0.4754 |
| 当前 mean proxy Recall@20 | 9.2% | 12.1% | 9.8% |
| Dense K16 mass oracle Recall@20 | 89.0% | 95.4% | 99.4% |

随着 Core ratio 降低：

- 单个遗漏 K16 的潜在恢复收益增大；
- 真实 dense attention mass 对高收益 K16 的排序更稳定；
- 当前 mean-pooled proxy 并未随之改善。

因此 `0.16` 是一个有研究价值的压力测试和候选运行点：它能充分暴露 Macro-only 路由的粒度失配，也能检验 Macro/Micro dispatcher 是否可以用少量细粒度执行覆盖分散高价值交互。它不是论文叙事中必须追求的最低 Core 比例；最终 Core 比例应由质量—延迟 Pareto profiling 决定。

### 3.3 证据三：当前 total Top-p 的主要问题是预算投错 Query

在 Core ratio `0.16`、当前 mean-pooled proxy 下：

| `p_total` | 严重遗漏最佳 K16 召回 | Residual 为空 | Residual K16 均值/中位数 |
|---:|---:|---:|---:|
| 0.90 | 39.9% | 60.1% | 97.7 / 0 |
| 0.95 | 70.5% | 29.5% | 273.8 / 125 |

单纯提高 Top-p 虽然能提高召回，却会迅速产生数百个 Microtiles，并且大量最需要修复的 Query 仍被错误判断为“Core 已经覆盖足够质量”。这说明问题不是只有总预算不足，更关键的是：

1. mean pooling 抹掉了 tile 内少量大 QK；
2. proxy softmax 可能错误过度集中，导致估计的 Core mass 偏高；
3. Top-p 对概率校准误差比 Top-k 更敏感；
4. 没有 Query/Head 风险下界时，Residual 可能在真正严重的 Query 上为空。

### 3.4 证据四：Layer/Head 先验稳定，固定 K 位置先验不稳定

在早期只分析 Layer `{9,12,15}` 的 Core ratio `0.16` 数据中，173 个严重遗漏里：

- 173/173 全部发生在被分析的 Layer 12；
- Head 8 占 151/173；
- Head 8/5/18 合计占 167/173，即 96.5%；
- Step 12/24/36 分别占 83/62/28 个，后期风险下降但未消失。

但是，最佳 K16 坐标会随视频内容和 Core ratio 改变：

- 三个视频的高频 K16 集合没有稳定交集；
- ratio 从 `0.25` 调到 `0.20/0.16` 后，高频位置明显重排；
- ratio `0.16` 下，留一视频固定位置先验 Recall@20 仅为 12.7%。

这不能排除尚未被逐 K16 扫描的其他 layers 也存在严重遗漏。正式确定风险集合前，必须先对全部 layers 做一次较轻量的全覆盖风险扫描，再对候选 layers 做逐 K16 add-back。早期 `Q128×K128` 配置中高误差曾集中于其他 Layer/Head，也进一步说明风险先验会随模型结构、执行块形状和选择配置变化。

因此本文使用的先验必须定义为：

> **模型级风险预算先验，而不是跨视频固定 Token anchor。**

还需注意：现有 Q16 样本刻意包含高误差 Query。因此上述 96.5% 是“已分析 layers 中严重遗漏事件条件下”的 Layer/Head 集中度，不能直接解释为所有自然 Query 中的发生概率。最终论文必须在完整 Layer 和完整 Query 分布上复验风险覆盖率。

### 3.5 证据五：时间邻近可作软特征，空间邻近不能作硬规则

严重遗漏 K16 在时间维度上明显更接近当前 Q16，但 Hilbert 一维邻域、三维最近邻和纯空间最近邻的 Top20 召回均很低。故方法中可以加入轻量时间 bias，但不能硬删除空间远端候选，也不能只搜索 Q 周围的局部窗口。

---

## 4. 方法总览

对一层注意力，记排列后的 Query、Key、Value 为：

\[
Q,K,V\in\mathbb{R}^{H\times N\times d}.
\]

HyMoR-Attn 使用两级逻辑粒度：

- Microtile：`Q16×K16`；
- Macro tile：`Q128×K96`，包含 `8×6=48` 个 Microtiles。

完整数据流为：

```text
Q/K/V + Hilbert3D permutation
        │
        ├─ Cheap Q16/K16 proxy ──聚合──> Q128/K96 Macro score
        │                                  │
        │                                  └─ fixed Macro Top-k ──> Core
        │
        └─ High-risk Layer/Head:
             Peak-aware sampled-LSE proxy
                         │
                         └─ complement total Top-p
                            + risk-aware Top-k floor
                            + bounded Top-k cap
                                      │
                                      └─ Residual Q16/K16

Core ∪ occupancy-promoted tiles ──> FlashInfer FA3 ──> (O_c, LSE_c)
Remaining Residual CSR            ──> Triton MMA     ──> (O_r, LSE_r)
                                              exact LSE merge ──> O
```

---

## 5. 阶段一：固定 Macro Top-k Core

### 5.1 Macro score

普通路径继续使用低成本的 Q16/K16 mean-pooled proxy：

\[
\bar Q_i=\frac{1}{16}\sum_{a=1}^{16}Q_{i,a},\qquad
\bar K_j=\frac{1}{16}\sum_{b=1}^{16}K_{j,b},
\]

\[
S^{\mathrm{cheap}}_{ij}
=\operatorname{softmax}_j
\left(\bar Q_i\bar K_j^T/\sqrt d\right).
\]

将对应 `8×6` 个 Micro scores 聚合成一个 `Q128×K96` Macro score：

\[
M_{uv}
=\frac{1}{8}\sum_{i\in u}\sum_{j\in v}S^{\mathrm{cheap}}_{ij}.
\]

在每个 `(head,Q128)` 内重新归一化 Macro scores。

### 5.2 固定 Top-k 规则

设 K96 Macro 数为 `N_96`，Core ratio 为 `ρ_C`：

\[
k_C=\max\left(1,\left\lceil\rho_C N_{96}\right\rceil\right).
\]

论文主配置：

```text
rho_C = 0.16
Macro = Q128 × K96
```

每个 `(head,Q128)` 选 Macro score 最大的 `k_C` 个 K96 blocks。`rho_C=0.16` 是初始候选配置，不是方法目标本身；最终可以根据硬件 crossover 和 Pareto 曲线选择 `0.16/0.20/0.25`。使用固定 Top-k 而不是 Macro Top-p 的原因是：

- Core 提供稳定、可预测的规则计算骨架；
- Top-k 主要依赖 ranking，对 proxy 概率校准误差相对不敏感；
- `Q128×K96` 与当前 H100 FlashInfer SM90 FA3 工作块对齐；
- Residual 已负责针对 Query 分布自适应调整细粒度预算。

纯视频区域使用稀疏规则；文本 KV、文本 Query 和跨模态边界块保持 dense。

---

## 6. 阶段二：Peak-aware Q16×K16 scorer

### 6.1 设计目标

Residual scorer 必须比当前 mean pooling 更好地近似两类量：

1. 一个 K16 对当前 Q16 的总 attention mass；
2. Q16×K16 内是否存在少量极大的 token-pair logit。

直接计算完整 `16×16=256` 个 QK 会使 selector 接近 dense attention，因此主方案采用每块四个代表向量的 sampled log-mean-exp。

### 6.2 代表向量选择

对任意 16-token block `X={x_1,...,x_16}`：

\[
\mu_X=\frac{1}{16}\sum_t x_t,
\]

计算 token 相对均值的残差范数：

\[
r_t=\|x_t-\mu_X\|_2.
\]

选择残差范数最大的三个 token，组成：

\[
R(X)=\{\mu_X,x_{t_1},x_{t_2},x_{t_3}\}.
\]

均值向量保留整体趋势，三个离均值最远的 token 用于捕获被平均池化抵消的局部峰值。该选择不依赖具体视频坐标，且每个 block 只需一次 Top3 reduction。

### 6.3 Sampled log-mean-exp score

对 Q16 block `i` 和 K16 block `j`，计算 16 个代表向量 pair logits：

\[
L_{ij}^{ab}
=\frac{R(Q_i)_a R(K_j)_b^T}{\sqrt d},
\qquad a,b\in\{1,2,3,4\}.
\]

定义：

\[
Z_{ij}
=\log\left(\frac{1}{16}\sum_{a,b}\exp L_{ij}^{ab}\right).
\]

Log-mean-exp 在分布平缓时累积多个有效 pair，在存在极端大 logit 时又自然接近 max，因此无需手工在 mean score 和 max score 之间设置硬权重。

可选时间软先验写为：

\[
Z'_{ij}=Z_{ij}+\lambda_t
\exp\left(-d_t(i,j)/\sigma_t\right).
\]

主实验先令 `λ_t=0`，将时间项作为独立消融；若启用，`λ_t` 和 `σ_t` 必须只在校准集上确定。任何情况下均不按空间距离硬删除候选。

最后得到完整 K16 范围内的原始概率质量：

\[
\hat P_{ij}
=\operatorname{softmax}_j\left(Z'_{ij}/T_{lh}\right).
\]

`T_lh` 是 Layer/Head calibration temperature。第一版使用 `T_lh=1`；后续可用校准集令 proxy entropy 或累计质量与小规模 dense reference 更一致。Top-p 必须基于校准后的正概率质量，不能对 QK logit 取绝对值。

### 6.4 计算范围

本轮全 Layer calibration 先覆盖全部 60 个 attention layers 和 24 个 heads；每个 `(step, layer)` 快照先对全部 Q16 计算误差，再从该层采样高误差和低误差控制 Q16 做逐 K16 add-back。早期只扫描了 Layer `{9,12,15}`，得到的候选实例为：

```text
Layer 12, Heads {5, 8, 18}
```

Head 编号与代码一致，使用 0-based index。该集合在完成全 Layer calibration 前只能称为候选风险集合。

在全 Layer calibration 完成前，`Layer 12, Heads {5, 8, 18}` 只能作为早期候选，不能作为最终风险集合。全层结果用于重新确定哪些 Layer/Head 值得启用 sampled-LSE scorer；其余 Layer/Head 才继续使用 cheap proxy。

全 Layer calibration 阶段不预先固定三个 heads；若后续确认只对一个 layer 的三个 heads 使用 16/256 sampled QK，其理论乘加量相对所有 layer-head 的完整 dense QK 约为：

\[
\frac{3}{60\times24}\times\frac{16}{256}
\approx0.013\%.
\]

这只是算术量估计；实际开销仍可能由 score materialization、排序、访存和 kernel launch 主导，因此必须单独计时，并按 Q16 rows 分块计算，避免物化过大的全局中间张量。

---

## 7. 阶段三：Residual total Top-p + 风险 Top-k floor

### 7.1 原始质量补足

对每个 `(head,Q16)`，设 Core 覆盖的 K16 集合为 `C_i`。使用与当前 Residual 相同的概率分布计算：

\[
m_i^C=\sum_{j\in C_i}\hat P_{ij}.
\]

在 Core 补集中按 `P_hat` 降序选择最小集合 `R_i^p`，使：

\[
m_i^C+\sum_{j\in R_i^p}\hat P_{ij}\ge p_{total}.
\]

主配置：

```text
p_total = 0.90
```

这里不能对 Core 补集重新归一化。Residual 填补的是原始完整分布中尚未被 Core 覆盖的质量，而不是强制在剩余区域重新分配 100% 概率。

### 7.2 模型级风险先验

为了避免将 HunyuanVideo 的具体 head 编号写成无法泛化的方法常数，定义离线风险分数：

\[
r_{lh}
=\Pr\left(\max_{j\notin C}\Delta E_{lhqj}>\tau_E\right),
\]

或采用严重遗漏的平均/高分位恢复收益作为等价统计。所有风险统计只能来自与最终测试 prompt、seed 分离的 calibration set。

取风险最高且覆盖校准集绝大多数严重遗漏的 Layer/Heads 构成 `G_risk`。当前 HunyuanVideo 的已分析 Layer 子集给出候选集合：

\[
G_{risk}=\{(12,5),(12,8),(12,18)\}.
\]

论文中只有在完成全 Layer、独立 calibration set 验证后，才能写成“校准过程自动得到上述集合”；不能宣称这三个 head 对所有模型普遍成立。

### 7.3 Top-k 安全下界与上限

Top-p 结果数量记为 `k_i^p=|R_i^p|`。最终预算定义为：

\[
k_i
=\min\left(k_{max}^{lh},
\max\left(k_i^p,k_{min}^{lh}\right)\right).
\]

主配置：

| Layer/Head 类型 | `k_min` | `k_max` | scorer |
|---|---:|---:|---|
| 普通 | 0 | 16 | cheap mean proxy |
| `G_risk` | 20 | 32 | sampled-LSE proxy |

最终 Residual 是 Core 补集中按同一 score 排名前 `k_i` 的 K16：

\[
R_i=\operatorname{TopK}_{k_i}
\left(\hat P_{i,\overline C_i}\right).
\]

`k_min=20` 的依据是 Core ratio `0.16` 时 dense-mass oracle Recall@20 达到 `99.4%`。它并不保证在线 proxy 也达到该召回，而是给新 scorer 一个有数据依据、执行上可控的安全预算。`k_max` 防止误校准 Top-p 生成数百个 Residual tiles，并将 kernel row 长度约束在 `≤32` bucket。

由于存在 `k_max`，`p_total=0.90` 是一个受预算约束的质量目标，而不是每行都必然满足的硬保证。必须额外记录：

\[
\delta_i^{mass}
=\max\left(0,p_{total}-m_i^C-\sum_{j\in R_i}\hat P_{ij}\right),
\]

并报告 mass shortfall 的 mean/p95/max。这样可以区分“scorer 判断 Core 已足够”和“由于计算上限主动截断”两种失败原因。

Step 12/24 的风险更高，但 Step 36 仍有明显严重遗漏。因此主方法不在后期关闭风险 floor。Step-aware floor 可作为消融：例如早中期 `20`、后期 `8/12`，只有在质量无损时才进入最终配置。

### 7.4 不采用的固定 Token anchor

跨视频统计得到的固定 K16 IDs 不进入主方法。若部署场景能够从同一视频的前一个 refresh step 动态得到可靠的高恢复候选，可以把它们作为有时效的 content-specific anchors 与 `R_i` 取并集，并设置：

```text
anchor TTL = cache_interval
anchor count ≤ 8 per Q16
```

该功能属于可选扩展，必须与“无 anchor”及“跨视频固定 anchor”分别消融，不能与主方法混在一起归因。

---

## 8. 阶段四：Density-adaptive granularity allocation

每个 `Q128×K96` Macro 包含 48 个 Q16×K16。统计其中已选 Residual Microtiles 数量：

\[
o_{uv}=\sum_{i\in u,j\in v}\mathbf{1}[(i,j)\in R].
\]

当：

\[
o_{uv}\ge\tau_{promote}
\]

时，将整个 Macro 提升到 Core，并从 Residual 中删除其全部 Microtiles。否则保留为 Micro CSR。主配置：

```text
tau_promote = 24 / 48
```

从而始终满足：

\[
C\cap R=\varnothing.
\]

这里的核心不是机械地“把 Residual 补成更大的 Core”，而是做 **hardware-aware support rounding**。设质量路由得到的目标 Fine support 为 `F`：低 occupancy 区域按 Micro 精确执行 `F`；高 occupancy 区域向上取整为完整 Macro cover。因此 promotion 后的实际执行 support 是 `F` 的受控超集，而不是严格不变的 support：

\[
\operatorname{Dispatch}(u,v)=
\begin{cases}
\text{Macro/FA3}, & o_{uv}\ge\tau_{promote},\\
\text{Micro/Triton}, & o_{uv}<\tau_{promote}.
\end{cases}
\]

当局部 occupancy 较高时，规则 Macro kernel 的 Tensor-Core 利用率、访存连续性和调度效率更好；此时执行完整 Macro 可能比逐个执行已选 Microtiles 更快，同时额外纳入的中等分数 interactions 还可能提高质量。当 occupancy 较低时，Micro kernel 则避免整块取整造成的大量无效 QK。因此粒度分配本身同时影响物理成本和最终支持集质量，是实现 Pareto 改善的核心算法—系统协同点。

`tau_promote` 应由真实 kernel crossover 决定，而不是只按逻辑密度拍定。需要先测量：

\[
T_{micro}(o)\quad\text{与}\quad T_{macro}(48),
\]

再选择最接近交点的阈值，并对 `16/24/32` 做端到端 sweep。实验中必须同时报告 promotion 引入的 support expansion ratio，区分目标 Fine interactions、实际执行 interactions 和最终质量收益。

---

## 9. 阶段五：异构执行与 Exact LSE Merge

### 9.1 Core 路径

Core mask 被压缩为 Macro CSR：

```text
macro_indptr
macro_bases
kv_lens
qo_indptr
```

`Q128×K96` blocks 由 FlashInfer FA3 执行，返回：

\[
(O_C,L_C).
\]

Direct macro-CSR、每层 plan cache、共享 vector-offset workspace 和一行一个 CTA 的 CSR expand 调度沿用当前实现。

### 9.2 Residual 路径

未提升的 Q16×K16 被压缩为按 `(head,Q16)` 组织的 CSR：

```text
indices
indptr
active_rows
length buckets: ≤4/8/16/32
```

Grouped Triton/MMA kernel 每个 program 处理一个非空 Q16 row，在同一个 program 内循环对应 K16 tiles，通过 online softmax 得到：

\[
(O_R,L_R).
\]

论文质量主实验必须使用完整 `Q16×K16` micro backend。当前 `rode_center` 只执行每个 tile 的中心 token，不与完整 Residual 在质量语义上等价，只能放入独立性能分析或附录。

### 9.3 Exact merge

Core 和 Residual 各自归一化后不能直接相加。令：

\[
m=\max(L_C,L_R),
\]

\[
w_C=\exp(L_C-m),\qquad w_R=\exp(L_R-m),
\]

\[
O=\frac{w_C O_C+w_R O_R}{w_C+w_R}.
\]

该结果与在 `C∪R` 上一次执行 softmax 严格等价。若 FlashInfer 返回 log2 LSE，应先转换为自然对数域。空 Residual row 直接保留 Core 输出，不启动无效 kernel 或 merge。

### 9.4 Cache 与 stream 策略

主配置：

```text
cache_interval = 12
route cache = True
direct macro CSR = True
CSR expand CTA multiplier = 0  # one CTA per row
Core/Residual parallel streams = False
```

现有结果表明 route/plan cache 和 CSR expand 调度对端到端时间至关重要；当前双 stream 实现曾导致 FP32 转换和 SpMM 严重退化，因此在同步与 allocator 行为被重新验证前保持串行。

---

## 10. 完整算法

```text
Inputs:
    Q, K, V
    Core ratio rho_C = 0.16
    Total mass p_total = 0.90
    Risk set G_risk from calibration
    Normal budget [0, 16]
    Risk budget [20, 32]
    Promotion threshold tau = 24

1. Permute video Q/K/V with Hilbert3D; preserve dense text/boundary policy.

2. Compute cheap Q16/K16 mean-pooled score for all heads.

3. Aggregate cheap scores into Q128/K96 Macro scores.

4. For each (head, Q128), select top ceil(rho_C * N_K96) Macro blocks as Core.

5. For each Layer/Head:
       if (layer, head) in G_risk:
           compute sampled-LSE Q16/K16 probability P_hat
           k_min, k_max = 20, 32
       else:
           reuse cheap Q16/K16 probability P_hat
           k_min, k_max = 0, 16

6. For each (head, Q16):
       compute Core mass under P_hat
       find minimum complement prefix reaching total mass 0.90
       clamp selected count into [k_min, k_max]
       keep the highest-scoring complement K16 tiles

7. Count selected Microtiles inside every non-Core Q128/K96 Macro.
       occupancy >= 24: promote whole Macro to Core
       otherwise: retain selected Q16/K16 in Residual CSR

8. Run Core with FlashInfer FA3 -> (O_C, LSE_C).

9. Run non-empty Residual rows with grouped Triton/MMA -> (O_R, LSE_R).

10. Exact-LSE merge and inverse permutation.
```

---

## 11. 与纯 DFSAttn 的本质区别

| 维度 | 纯 DFSAttn | HyMoR-Attn |
|---|---|---|
| 细粒度信息用途 | 用于改善大块排名 | 同时用于真实 Microtile 执行 |
| 主执行粒度 | 规则大块 | Q128×K96 Core + Q16×K16 Residual |
| Core 预算 | 固定大块 Top-k | 硬件对齐 Macro Top-k，比例由 Pareto profiling 确定 |
| Query 自适应预算 | 较弱 | complement total Top-p |
| proxy 失败保护 | 无显式安全下界 | 校准得到的 Layer/Head Top-k floor |
| 固定 K anchor | 不适用 | 明确不作为主方法 |
| 局部过密处理 | 整体由大块定义 | occupancy-based promotion |
| softmax | 单一 block kernel | 两支状态 exact LSE merge |
| 预期优势 | kernel 规则、实现成熟 | 稠密支持走 Macro、分散支持走 Micro；相同时间保留更多高价值交互 |

因此，论文的核心比较不能只看“谁的 block 更小”或“谁的 density 更低”，而应比较：

1. **matched E2E latency** 下，谁能保留更多有效 attention mass 并获得更高视频质量；
2. **matched video quality** 下，谁具有更短的端到端时间；
3. 从相同的 pre-dispatch Fine candidate set 出发，occupancy-aware Macro/Micro dispatch 是否比统一 Macro 或统一 Micro 获得更好的质量—时间折中；
4. matched QK density 仅作为诊断协议，用于拆分“选得更好”和“执行得更好”，不作为最终主叙事。

---

## 12. 实验设计

### 12.1 数据划分

必须将风险先验和 scorer calibration 与最终评测分离。

建议：

- Calibration：至少 8–16 个 prompts，2 个 seeds；先覆盖全部 layers 做轻量风险扫描，再对候选 layers 做逐 K16 分析，用于确定 `G_risk`、temperature 和可选时间 bias；
- Validation：独立 8 个 prompts；用于选择 `rho_C/p_total/k_min/k_max/tau`；
- Test：VBench 33 prompts，至少 seed 0；论文主结果尽可能补充 3 seeds；
- 分辨率：先在 `480×720` 完成完整消融，再在 `720×1280` 验证可扩展性；
- 所有方法保持 prompt、seed、scheduler、CFG、steps 和 dense warmup 完全一致。

若计算预算不足，至少保证 calibration prompts 与最终 33 prompts 无重合，并对主要质量/E2E 指标报告 bootstrap 95% confidence interval。

### 12.2 Baselines

主表至少包括：

1. Full Attention；
2. 原生 DFSAttn；
3. `Q128×K96` Core-only；
4. Macro Top-k + 当前 mean-proxy total Top-p；
5. 当前 `fine_topk_occupancy` 路由；
6. HyMoR-Attn，无风险 floor；
7. 完整 HyMoR-Attn；
8. Dense-mass oracle，仅作为算法上界，不报告为可部署方法。

为了单独验证硬件粒度分配，必须对完整 HyMoR 生成的**同一份 pre-dispatch Fine candidate set**增加三种执行对照：

1. `All-Macro`：凡是包含被选 Microtile 的 Macro 均 densify 后执行；
2. `All-Micro`：所有被选 interactions 都按 Q16×K16 CSR 执行；
3. `Adaptive`：根据 occupancy/kernel crossover 分配到 Macro 或 Micro。

三者的选择分数、pre-dispatch Fine candidates 和 Q/K/V 输入必须一致，但执行 support 不会完全相同：All-Micro 精确执行 Fine support；All-Macro 执行其最小 Macro cover；Adaptive 只对高 occupancy 部分取 Macro cover。必须同时比较 kernel latency、E2E latency、实际执行 interactions、support expansion、输出质量和显存。纯 kernel crossover 另外使用合成等价 workload 测量，避免把由 support 扩张产生的质量变化误归因为 kernel 本身。

如论文对比外部方法，应严格匹配模型、分辨率、steps 和 realized density，不能直接抄不同论文中的 speedup。

### 12.3 机制实验

在现有逐 K16 add-back 数据上首先比较 scorer：

| Scorer | 用途 |
|---|---|
| Mean-pooled QK | 当前实现基线 |
| Max representative QK | 峰值消融 |
| Sampled log-mean-exp | 主方法 |
| Sampled-LSE + temperature | calibration 消融 |
| Sampled-LSE + temporal bias | 时间先验消融 |
| Dense K16 mass | oracle 上界 |

报告：

- Recall@1/5/20；
- `Recovered-ΔE@1/5/20`；
- NDCG 或 rank correlation；
- 严重遗漏上的 Residual-empty rate；
- 每 Q16 的 Residual mean/p50/p95/max；
- scorer latency 和峰值显存。

其中：

\[
\operatorname{Recovered\text{-}\Delta E@k}
=\frac{\max_{j\in\operatorname{Top}k}\Delta E_j}
{\max_{j\in\mathrm{all\ omitted}}\Delta E_j}.
\]

这个指标比“是否命中唯一 oracle-best”更稳定，因为多个候选可能具有非常接近的恢复收益。

### 12.4 方法消融

质量路由按以下顺序逐项加入：

```text
Macro Core-only
  + mean-proxy total Top-p
  + sampled-LSE scorer
  + Layer/Head risk floor
  + k_max cap
  + temperature calibration
  + optional temporal bias
```

硬件执行单独做正交消融：

```text
Fixed pre-dispatch Fine candidate set
  ├─ All-Macro
  ├─ All-Micro
  ├─ Static tau promotion
  └─ Profiled crossover-based adaptive dispatch
```

这样可以分别回答：

- scorer/risk prior 是否提高了每单位逻辑计算的质量；
- granularity allocator 是否用可控的 support rounding 获得更优的物理执行时间—质量折中；
- 两者联合后是否构成相对 DFSAttn 的严格质量—延迟 Pareto 优势。

另设两个负对照：

- 跨视频固定 K16 anchor；
- 只使用空间/Hilbert 局部窗口。

它们用于证明稳定先验存在于“风险位置”，而不是“固定 Key 坐标”。

### 12.5 超参数 sweep

建议使用分阶段小网格，避免全组合爆炸：

```text
Core ratio rho_C:       {0.12, 0.16, 0.20, 0.25}
Total p:                {0.80, 0.90, 0.95}
Risk k_min:             {8, 12, 20}
Normal/Risk k_max:      {(8,24), (16,32), (32,64)}
Promotion tau:          {16, 24, 32}
Representative count:  {2, 4, 8}
```

先用 snapshot selector 指标筛选，再运行视频，不能直接对所有组合生成完整视频。

### 12.6 视频质量指标

以相同 prompt/seed 的 Full Attention 视频为 reference，报告：

- PSNR ↑；
- SSIM ↑；
- LPIPS ↓；
- temporal LPIPS 或相邻帧一致性指标；
- VBench 语义和时序维度；
- 失败样例的可视化，特别关注网格、局部结构断裂、主体漂移和运动不连续。

PSNR/SSIM/LPIPS 衡量 sparse 对 Full 的 fidelity，VBench 衡量生成语义质量，两者不能互相替代。

### 12.7 系统指标

必须分开记录：

```text
QKV permutation
cheap score
peak-aware score
Macro aggregation/select
Residual select
occupancy/promotion
CSR compact/plan/expand
Core kernel
Residual kernel
LSE merge
output inverse permutation
E2E GPU time
wall time
peak memory
```

同时报告：

- Core/Residual/最终实际 QK interaction density；
- promotion 数量和 occupancy histogram；
- Residual row length histogram；
- Core 和 Residual 的有效 TFLOPS/带宽；
- route refresh 与 cache-hit step 的时间差；
- 不包含首次 Triton/CUDA 编译的 steady-state 时间。

---

## 13. 已有性能事实与当前缺口

现有 HunyuanVideo prompt-0 结果中：

- 原始 DFSAttn 平均最终 density 约 `19.67%`，E2E GPU 为 `200.957 s`；
- FlashInfer64 两支路径 density 约 `17.61%`；
- 未缓存的完整 Micro residual 路径 E2E 为 `288.659 s`，说明 route/plan/格式准备会抵消低 density；
- RoDe cache-only center-token 路径达到 `199.514 s`，与 DFSAttn 基本持平，但 center-token 与完整 Q16×K16 在质量语义上不等价；
- 修复 CSR expand 后的另一组 route-cache 实验仍比 DFSAttn 慢约 `4.2%`，说明速度优势尚未被完整证明。

因此目前可以声称：

> 算法上存在明确的细粒度恢复空间；Macro/Micro 异构执行已具备数值正确路径；但“完整质量语义下端到端快于 DFSAttn”仍是待验证目标。

目前不能声称：

- 现有 FlashInfer64 已经稳定快于 DFSAttn；
- RoDe center-token 结果证明完整 Residual 的质量或速度；
- 当前 mean proxy 的 total Top-p 已经是有效的严重遗漏发现器；
- layer 12/head 5/8/18 是跨模型普适规律。

---

## 14. 合理预期的最终实验结果

### 14.1 预期依据

预期建立在以下已知事实上：

1. Macro 漏选收益可由少量 K16 高比例恢复；
2. Dense mass oracle 在 Core ratio `0.16` 下 Recall@20 达到 `99.4%`；
3. 全 Layer calibration 完成后，再根据实际覆盖率决定风险 floor 的 Layer/Head 范围；早期“一个 layer 的三个 heads”仅是待验证候选；
4. 同一批 Fine candidates 可以依据局部 occupancy 选择精确 Micro 执行或受控 Macro rounding，因此目标 support、实际执行 support 与物理成本不再被单一粒度绑定；
5. 当前系统距离 DFSAttn 的速度差主要取决于 granularity dispatch、selector、plan、CSR 和 Residual 调度，说明端到端优势必须来自整条硬件执行路径，而不能只依赖更低 QK density。

最大不确定性是 sampled-LSE scorer 能否将当前 `9.8%` Recall@20 显著推近 dense oracle，以及其实际排序/访存开销。

### 14.2 Selector 结果预期

| 指标，Core ratio=0.16 | 当前 mean proxy | HyMoR 合理预期 | Dense oracle |
|---|---:|---:|---:|
| 严重遗漏 Recall@5 | 5.2% | 45%–70% | 97.7% |
| 严重遗漏 Recall@20 | 9.8% | 70%–90% | 99.4% |
| Recovered-ΔE@20 | 待统一统计 | 0.80–0.95 | 约 1.0 |
| 严重遗漏 Residual-empty rate | 60.1% | 0%–10% | 取决于预算规则 |
| Residual row p95 | 可能数百 | ≤32（由 cap 保证） | 不适用 |

如果 sampled-LSE 的 Recall@20 低于 `50%` 或 Recovered-ΔE@20 低于 `0.70`，说明代表 token 规则仍没有保住关键方向，应停止大规模视频实验，转向学习型 scorer 或增加代表数。

### 14.3 Pareto 主结果预期

以原始 DFSAttn 的约 `19.7%` density 和单 prompt `200.957 s` E2E 为参考，主结果应追求严格的 Pareto improvement，而不是预设必须降低 logical density：

| 指标 | 合理预期 |
|---|---:|
| 最终 logical interaction density | 18%–22%，允许与 DFSAttn 相近或略高 |
| E2E GPU time | 185–195 s |
| E2E 相对 DFSAttn | 快 3%–8% |
| PSNR 相对 DFSAttn | +0.2 至 +0.8 dB |
| SSIM 相对 DFSAttn | +0.003 至 +0.015 |
| LPIPS 相对 DFSAttn | -0.005 至 -0.020 |
| VBench | 持平或小幅提高；不以 fidelity 提升推断语义分数必然提升 |

这里允许 logical density 略高，是因为论文假设恰恰是：更合理的 Macro/Micro 映射可以用更低的物理执行成本承载更多高价值 interactions。若方法只靠降低 density 才获得速度，而视频质量仅持平，则不能充分支持本文的硬件粒度分配主张。

上述数字是合理目标区间，不是已有测量。最有说服力的最终结果应同时满足：相对 DFSAttn，E2E 显著下降，并且 PSNR/SSIM 提高、LPIPS 降低；即同一个运行点在速度和视频质量两个坐标上都严格占优。

### 14.4 端到端性能预期

以现有单 prompt DFSAttn `200.957 s` 为同机器参考，可以给出三档合理情景：

| 情景 | 方法状态 | 预期 E2E | 相对 DFSAttn |
|---|---|---:|---:|
| 保守 | scorer 有效，但 granularity dispatch/CSR 开销较高 | 198–205 s | 持平附近，尚未形成强 Pareto 优势 |
| 论文目标 | dispatcher 命中 kernel crossover，scorer 分块融合、route cache 和 Micro bucket 有效 | 185–195 s | 快 3%–8% |
| Stretch | Residual 极稀疏且 selector/CSR 高度融合 | 177–185 s | 快 8%–12% |

主论文不应预先承诺 Stretch 数字。更可信的目标是稳定获得 `3%–8%` E2E 加速，同时在同一个配置上展示明确的视频质量提升。720p 应报告实际测量，不从 480p 线性外推。

### 14.5 论文成败标准

进入最终论文主表前，建议同时满足：

```text
Mechanism:
    Recall@20 on severe omissions ≥ 70%
    Recovered-DeltaE@20 ≥ 0.80
    Severe-omission residual-empty rate ≤ 10%

Quality:
    PSNR ≥ DFSAttn + 0.2 dB
    SSIM > DFSAttn and LPIPS < DFSAttn
    No systematic grid/temporal artifacts

System:
    Residual p95 ≤ 32 K16 tiles
    Peak-aware scorer ≤ 10% of sparse attention time
    E2E speedup ≥ 3% on the same held-out outputs
    From matched pre-dispatch candidates, adaptive dispatch Pareto-dominates all-Macro/all-Micro
    No result relies on first-run compilation or center-token approximation
```

如果质量明显提高但 E2E 未加速，工作更接近“精度修复方法”，系统主张需要收缩；如果 E2E 加速但质量没有提高，则只能说明执行优化，不能支持 Pareto dominance；如果二者都改善但 adaptive dispatch 从 matched pre-dispatch candidates 出发没有优于统一粒度，则不能把加速归因于粒度分配。最终必须同时闭合 selector、quality、granularity dispatch 和 E2E 四层证据。

---

## 15. 论文叙事建议

### 15.1 推荐标题方向

可以从以下方向继续凝练：

1. **HyMoR-Attn: Hardware-Aware Macro–Micro Routing for Faster and Higher-Quality Video Diffusion**
2. **Recovering What Blocks Miss: Heterogeneous Macro-Micro Sparse Attention for Video Diffusion**
3. **Beyond Block Sparsity: Risk-Aware Fine-Grained Residual Attention for Video Generation**

第一种强调系统与完整方法；第二种更适合突出机制发现；第三种强调相对 DFSAttn 的算法改进。

### 15.2 摘要叙事骨架

可以按以下逻辑展开：

1. Block sparse attention 为视频扩散带来规则 GPU 执行，但把所有 selected support 统一映射到大块会在局部稀疏区域浪费计算；统一使用小块又会在局部稠密区域损失 Tensor-Core 效率；
2. 通过逐 Macro/K16 add-back，发现 attention support 同时具有局部稠密主体和分散高价值残差，且少量 K16 能获得完整 Macro 的大部分质量收益；
3. 进一步发现严重遗漏在 Layer/Head 层面集中，却在具体 K 坐标上随内容变化，同时现有 mean-pooled proxy 无法识别真实高质量 K16；
4. 因此提出 HyMoR-Attn：固定 Macro Top-k backbone、peak-aware Micro total Top-p、calibrated risk floor，以及由局部 occupancy 驱动的 Macro/Micro hardware granularity allocator；
5. 稠密区域交给 FA3 Macro kernel，分散区域交给 Triton Micro kernel，并通过 exact LSE merge 保持数学正确；route cache、CSR 和 kernel crossover profiling 控制系统开销；
6. 最终在同一运行点上同时取得低于 DFSAttn 的端到端时间和更高的视频质量，形成严格的 quality-latency Pareto dominance。

### 15.3 最重要的图表

论文至少需要以下四张核心图：

1. **机制图**：一个未选 Macro 内少量高 dense-mass K16 被 mean pooling 淹没，以及逐 K16 add-back 后误差恢复；
2. **三层方法图**：Macro Core、Micro Residual、promotion 与 exact LSE merge；
3. **Granularity crossover 图**：固定相同 Fine candidates，画出不同 occupancy 下 all-Micro、Macro cover 和 adaptive dispatch 的 kernel latency及 support expansion；
4. **Selector 曲线**：Recall/Recovered-ΔE 随 K16 budget 变化，比较 mean、sampled-LSE 和 oracle；
5. **主 Pareto 图**：PSNR/LPIPS 对 E2E latency，展示 Full、DFSAttn、Core-only、当前 hybrid 和 HyMoR；density 图作为辅助诊断。

附加图可以包括 Layer×Head 风险热力图、K16 位置跨视频不稳定图、Residual row-length 分布和 occupancy/kernel crossover。

---

## 16. 局限性与 reviewer 可能质疑的问题

### 16.1 风险先验是否过拟合 HunyuanVideo

回应方式：方法定义的是 calibration-derived risk set，而不是写死 layer/head ID；必须增加跨 prompt、seed 和至少一个额外模型的验证。若额外模型风险位置不同但同一校准流程仍有效，反而能强化方法主张。

### 16.2 当前机制数据是否有采样偏差

现有数据过采样了高误差 Q16。论文应明确这一点，并在完整 Query 分布上报告严重遗漏自然发生率、风险集合覆盖率及额外预算占比。

### 16.3 单块 `ΔE` 能否代表多块联合恢复

不能完全代表。多个 K16 同时加入会共享 softmax 分母，独立 `ΔE` 不可相加。现有 `ΔE` 适合训练/评价候选排序，最终必须通过联合 mask replay 和真实视频生成验证。

### 16.4 Sampled-LSE 是否真的比 mean proxy 便宜

算术量很低不等于实际 latency 低。需要 fused representative extraction、chunked scoring 和 GPU Top-k，并报告 selector 的真实时间与显存。若 scorer 开销过高，可退化为只在 risk heads 的 frontier candidates 上运行，但需要测量 frontier recall。

### 16.5 为什么不直接把风险 heads 设为 dense

将整个 head 设为 dense 会恢复质量，但丢失大部分 sparsity，且无法证明 Microtile 可恢复机制。应将“risk heads dense”作为质量上界和成本较高的 baseline，而不是主方法。

### 16.6 为什么不用固定 K anchor

现有留一视频实验已经显示固定位置泛化失败。稳定的是 Layer/Head 风险，不是 Key 坐标；主方法仍在线、随内容选择 K16。

---

## 17. 实现落地顺序

建议严格按以下顺序推进。由于论文的主叙事是硬件粒度分配带来 Pareto 改善，首先验证执行侧，再扩展质量侧：

1. 对固定 logical support 构造不同 occupancy 的 microbench，测出 `T_micro(o)` 与 `T_macro(48)` crossover；
2. 对同一真实 route 实现 All-Macro、All-Micro、Adaptive 三种 replay，确认 Adaptive 的 kernel/E2E 时间最低；
3. 在现有 snapshot/add-back 数据上实现 sampled-LSE 离线 scorer，不改推理 kernel；
4. 输出 Recall@k、Recovered-ΔE@k 和 row-length，确认达到 selector gate；
5. 在 `topk_topp` 中加入 per-layer/head `k_min/k_max`，先继续使用现有 score，验证预算语义和统计正确性；
6. 接入 risk-head sampled-LSE，并使用 chunked GPU scoring；
7. 验证 Core/Residual 不重叠、Macro rounding 的 support expansion 与构造出的 reference mask 一致、exact LSE merge 数值正确；
8. 在保存 QKV 上做 mask replay，比较 Core-only、旧 scorer、新 scorer 和 oracle 的联合输出误差；
9. 运行单 prompt 视频因果实验，要求同一配置同时优于 DFSAttn 的 E2E 与质量；
10. 完成 480p 33 prompts 后再优化 selector/CSR 并运行 720p；最后才考虑学习型 scorer、动态 anchors 或双 stream 等扩展。

当前仓库启动时需要特别注意：shell 默认 route 是 `fine_topk_occupancy`，且当前 `FLASHINFER64_CORE_ONLY` 默认值为 `True`。要运行本方法的现有近似基线，必须显式设置：

```bash
SPARSE_EXECUTION=flashinfer64 \
FLASHINFER64_ROUTE_MODE=topk_topp \
FLASHINFER64_TILE_TOP_RATIO=0.16 \
FLASHINFER64_TOKEN_TOP_P=0.90 \
FLASHINFER64_PROMOTION_THRESHOLD=24 \
FLASHINFER64_ROUTE_CACHE=True \
FLASHINFER64_CORE_ONLY=False \
FLASHINFER64_RESIDUAL_BACKEND=micro \
FLASHINFER64_PARALLEL_CORE_RESIDUAL=False \
FLASHINFER_CSR_EXPAND_CTA_MULTIPLIER=0 \
bash hyvideo_t2v_720p_dfs.sh
```

这条命令仍使用当前 mean-pooled scorer，也没有 risk floor，只能作为实现前基线，不能标记为完整 HyMoR-Attn。

---

## 18. 最终判断

现有证据足以支持继续推进这个研究方向，但支持的是以下更严格的版本：

> **以固定、硬件友好的 Macro Top-k 提供主体计算；以保留峰值的 Microtile mass proxy 在线发现内容相关遗漏；以校准得到的 Layer/Head 先验分配检测强度和 Top-k 安全下界；以 occupancy promotion 和 exact LSE merge 完成高效且数学正确的异构执行。**

它相对纯 DFSAttn 的潜在优势来自“更合理的物理执行粒度承载更高质量的逻辑 support”：局部稠密部分利用 Macro kernel 的规则吞吐，分散高价值部分利用 Micro kernel 避免块内浪费。论文能否成立最终取决于三个闭环：

1. sampled-LSE scorer 是否能在 `≤20/32` 的预算内显著接近 dense-mass oracle；
2. occupancy-aware dispatcher 是否从相同 Fine candidates 出发，通过受控 Macro rounding 同时优于 all-Macro 与 all-Micro 的质量—时间折中；
3. 执行侧节省是否足以覆盖 scorer、CSR 和 merge 开销，并允许保留更多高价值 interactions，使最终视频质量和 E2E 时间同时优于 DFSAttn。

若三者同时成立，HyMoR-Attn 有机会形成一篇由机制证据、质量路由、硬件粒度分配和端到端 Pareto 结果共同支撑的完整论文

---

## 19. 本文档的数据来源索引

- `finding.md`：K96 Core ratio `0.25/0.20/0.16` 的逐 K16 add-back、严重遗漏、Layer/Head、位置泛化和当前 proxy 召回结论；
- `writing-block(1).md`：早期 Q128×K128 Macro 漏选、长尾误差和 Macro 内 Top1/2/4 K16 恢复比例；
- `already2.md`：Q128×K96 Core、Q16×K16 Residual、promotion、CSR、FlashInfer/Triton 执行和 exact LSE merge 的当前实现；
- `timeres.md`：原始 DFSAttn、完整 Micro residual、RoDe center-token、cache 和端到端时间；
- `chat.md`：CSR expand 修复前后及 route-cache timing 分析；
- `hyper.md`：当前启动参数及不同 route mode 的实际生效关系；
- `phasea.md`：Block Top-p + exact Token Top-k 的早期 oracle 实验设计；
- `toppksurvey.md`：Top-k、Top-p、hybrid floor 和动态预算的相关工作线索。
