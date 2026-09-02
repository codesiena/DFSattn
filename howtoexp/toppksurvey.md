我按“**稀疏注意力里的 Top-k / Top-p token/block selection**”来查了，而不是 LLM 解码阶段的 top-k/top-p sampling。结合你上传的 **SpargeAttention2** 和 **SCOPE**，再补了 2025–2026 年公开论文。截止 **2026-08-24**，我目前能确认的脉络已经比较清楚了：

> **这个方向正在从“Top-k vs. Top-p”逐渐收敛到“Top-p 负责自适应预算 + Top-k 负责安全下界”，再进一步演化到 per-head / runtime adaptive Top-k。**

你给的两篇论文其实正好体现了这个演化：**SpargeAttention2 提出 hybrid Top-k+Top-p 的基本动机，SCOPE 则继续把固定 Top-k floor 改成了 online per-head floor。**

---

## 1. 先把 Top-k 和 Top-p 在 sparse attention 里的区别说清楚

假设一行 attention proxy 经 softmax 后为

$$
a_1,\ldots,a_N,\qquad \sum_j a_j=1
$$

按从大到小排序：

$$
a_{(1)}\geq a_{(2)}\geq\cdots\geq a_{(N)}.
$$

**Top-k** 是固定预算：

$$
S_k=\{(1),\ldots,(k)\}.
$$

也就是说，无论这一行 attention 是非常尖锐还是非常均匀，都保留固定数量/比例的 key。

而 **Top-p** 是固定 attention mass：

$$
k_p=\min\left\{m:\sum_{i=1}^{m}a_{(i)}\geq p\right\},
$$

最后保留前 \(k_p\) 个。因此真正的区别不是“两个不同排序算法”，而是：

|                             | Top-k           | Top-p             |
| --------------------------- | --------------- | ----------------- |
| 控制对象                        | token/block 数量  | 累积 attention mass |
| 预算                          | 固定              | 动态                |
| focused / peaked attention  | 容易 over-select  | 很省                |
| diffuse / uniform attention | 容易 under-select | 自动多保留             |
| 计算量                         | 可预测             | head/query 之间变化   |
| 只需要 ranking？                | 基本是             | 不够，还需要较准确的概率值     |
| softmax normalization       | 不一定             | 通常需要              |
| proxy score 误差敏感性           | 相对低             | 更高                |
| GPU/SP load balance         | 更容易             | 更困难               |

这个区别在 **Twilight** 里分析得最系统：作者指出 Top-k 的根本困难是一个固定 budget \(B\) 无法同时适合 focused 和 diffuse attention；Top-p 则直接针对累计 attention mass，使预算随分布变化。它还指出一个非常重要、经常被忽略的系统问题：**Top-k 主要要求顺序正确，而 Top-p 对概率数值精度也有要求**，所以低精度 proxy 对 Top-p 的伤害通常更大。

[Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](https://arxiv.org/abs/2502.02770?utm_source=chatgpt.com)

---

# 2. 目前明确把 Top-k 和 Top-p 结合起来的方法

我认为至少有 **5 条很明确的路线**值得看，其中 4 个是 2026 年视频 DiT sparse attention。

| 方法                             | 场景                  | Top-k + Top-p 怎么结合                          | 本质                                  |
| ------------------------------ | ------------------- | ------------------------------------------- | ----------------------------------- |
| **Twilight** (NeurIPS 2025)    | 长上下文 LLM            | Conservative Top-k candidate → Top-p prune  | Top-k 是候选集上界，Top-p 决定最终预算           |
| **SpargeAttention2** (2026.02) | Video DiT           | Top-k ∪ Top-p                               | 固定 minimum budget + adaptive mass   |
| **RAPID** (CVPR 2026)          | Video diffusion     | Top-k Anchor + Top-p Expansion              | 保底连接 + attention-mass 扩张            |
| **FVAttn** (2026.07)           | Multi-GPU Video DiT | Top-p routing + Top-k safety floor          | 算法 hybrid + runtime load balancing  |
| **SCOPE** (2026.08)            | Video DiT           | Top-p + fixed Top-k + online per-head Top-k | 从全局固定 floor 进化到 head-adaptive floor |

下面分别说。

---

## 3. SpargeAttention2：目前最直接的 Top-k + Top-p 理论动机

你上传的第一篇就是这个问题最直接的文章。

它首先发现，在**极高 sparsity（>90%）**下，Top-k 和 Top-p 有互补的 failure mode：

* attention 行比较 **uniform / diffuse** 时，概率散布在很多 token 上，固定 Top-k 很容易保留太少；
* attention 行非常 **skewed / concentrated** 时，Top-p 可能只靠几个 attention sink 就达到 \(p\)，从而漏掉后面的有效信息。

论文 Table 1 很有价值。在相同 sparsity 下：

$$
\text{uniform: Top-p}\approx\text{hybrid}>\text{Top-k}
$$

而

$$
\text{skewed: Top-k}\approx\text{hybrid}>\text{Top-p}.
$$

具体 L1 error 是：

| Attention distribution |      Top-k |  Top-p | Top-k + Top-p |
| ---------------------- | ---------: | -----: | ------------: |
| Uniform                |     0.4150 | 0.3726 |    **0.3707** |
| Skewed                 | **0.1664** | 0.2160 |    **0.1671** |

论文据此明确提出 Top-k/Top-p union。

形式是

$$
S=S_k\cup S_p.
$$

如果两者基于**同一条降序 ranking**，其实这个式子可以进一步简化理解成：

$$
|S|=\max(k,k_p).
$$

也就是说 SpargeAttention2 的 hybrid 本质上就是：

> **Top-p 决定自适应预算，但不允许预算低于 Top-k 给出的 minimum floor。**

论文原文也是这个定义。

它在视频 diffusion 上报告达到 **95% attention sparsity、16.2× attention speedup**。([arXiv][1])

[SpargeAttention2 论文](https://arxiv.org/abs/2602.13515?utm_source=chatgpt.com)

这是我认为你研究“Top-k + Top-p hybrid”时**第一篇应该精读的文章**。

---

# 4. RAPID：Top-k Anchor + Top-p Expansion

**RAPID: Reusing Attention Sparsity with Inter-step Adaptation for Efficient Video Diffusion** 是 **CVPR 2026** 正式论文。([CVPR Open Access][2])

它的 block selector 和 SpargeAttention2 非常相似，但写法更加直观：

### Step 1：Top-k Anchor

每个 query block 先至少选

$$
k_{\min}
$$

个最高分 key blocks。

目的就是：

> guarantee a baseline level of connectivity。

### Step 2：Top-p Expansion

然后继续按 score 从高到低加入 blocks，直到：

$$
\frac{\sum_{j\in S_i}s_{ij}}
{\sum_js_{ij}}\geq\tau.
$$

因此最终依然可以理解为：

$$
K_i=\max(k_{\min},K_i^{(p)}).
$$

论文把它直接称为 **Hybrid Block Selection Strategy**。([CVPR Open Access][3])

RAPID 还做了一个特别有用的 **Top-p / Top-k / Top-k+Top-p 消融实验**。

例如 density = 39.2%：

| Method            |     PSNR ↑ |    SSIM ↑ |   LPIPS ↓ |
| ----------------- | ---------: | --------: | --------: |
| Top-p             |     18.922 |     0.632 |     0.354 |
| Top-k             |     25.487 |     0.859 |     0.113 |
| **Top-k + Top-p** | **25.897** | **0.864** | **0.105** |

在 46.8%、54.4% density 下，hybrid 同样优于单独 Top-k，而 Top-p 在它这个场景明显更差。([CVPR Open Access][3])

这个结果非常值得注意，因为它说明：

> 在 video DiT 的 block-level / reused proxy 场景里，Top-p 并不像 LLM sparse attention 的一些论文描述得那么“一定优于 Top-k”。

这里有很强的**任务依赖性和 proxy accuracy 依赖性**。

[RAPID CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_RAPID_Reusing_Attention_Sparsity_with_Inter-step_Adaptation_for_Efficient_Video_CVPR_2026_paper.pdf?utm_source=chatgpt.com)

---

# 5. FVAttn：Top-p routing + Top-k Safety Floor

2026 年 7 月的 **FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation** 又沿用了这个结构。

它明确采用：

> **Top-p routing + Top-k safety floor**

作为 sparse-routing frontend。([arXiv][4])

因此仍然是：

$$
k_i=\max(k_p(i),k_{\min}).
$$

不过 FVAttn 的重点开始从“mask accuracy”转到了另一个 Top-p 的固有问题：

## Top-p 会造成 workload imbalance

因为不同：

* attention head
* query
* GPU rank

得到的 \(k_p\) 差别可能很大。

比如：

$$
k_p^{(h_1)}=100,\quad
k_p^{(h_2)}=500,\quad
k_p^{(h_3)}=2000.
$$

那么 sparse kernel 理论 FLOPs 是少了，但 sequence parallelism 下大家需要等最慢的 GPU，于是 Top-p 的动态性反而造成 **straggler**。

所以 FVAttn 后面又加了：

* Runtime Load Balancing；
* heavy-head migration；
* Slack-Aware Sparse Augmentation。

它报告平均 load imbalance 从 **1.34 → 1.08**，attention 相对 FlashAttention 达到 **4.41× speedup**，DiT inference 为 **2.02–2.11×**。([arXiv][4])

[FVAttn 论文](https://arxiv.org/abs/2607.16190?utm_source=chatgpt.com)

这篇对你很重要，因为它揭示了一个新的矛盾：

$$
\boxed{\text{Top-p 的算法自适应性}
\quad\leftrightarrow\quad
\text{硬件上的 workload regularity}}
$$

Top-k 天生比较规整，Top-p 天生不规整。

---

# 6. SCOPE：目前更进一步的版本——Online Per-Head Top-k

这就是你上传的第二篇，2026 年 8 月 13 日才放出来，基本是目前非常新的结果。([arXiv][5])

SCOPE 对 SpargeAttention2 式的 fixed Top-k floor 提出了进一步的问题：

> 固定 Top-k 能防止 Top-p under-selection，但**一个全局固定 k 无法适应不同 attention head 和不同 input**。

它首先仍然做：

$$
t_c^{(p)}
=
\min\left\{
t:\sum_{\ell=1}^{t}
\tilde a_{c,\pi_c(\ell)}\geq\rho
\right\}.
$$

然后设一个 fixed floor：

$$
k_{\text{fix}}=\lceil\alpha N\rceil
$$

得到

$$
b_c=
\max
\left(
t_c^{(p)},k_{\text{fix}}
\right).
$$

也就是标准的：

$$
\boxed{\text{Top-p + fixed Top-k}}.
$$



但 SCOPE 不停在这里。

它把一个 head 中各 query cluster 初步得到的 \(b_c\)，按照 cluster size \(n_c\) 加权平均：

$$
k_{\text{head}}
=
\left\lfloor
\frac{\sum_cn_cb_c}
{\sum_cn_c}
\right\rfloor.
$$

最终：

$$
r_c=
\max
\left(
b_c,k_{\text{head}}
\right)
$$

即：

$$
\boxed{
r_c
=
\max
\left(
k_p(c),
k_{\rm fix},
k_{\rm head}
\right)
}.
$$



这实际上把 hybrid Top-p+Top-k 推进了一步：

### SpargeAttention2 / RAPID

$$
K_i=\max(K_p(i),K_{\min})
$$

### SCOPE

$$
K_i=
\max
(
K_p(i),
K_{\text{fixed}},
K_{\text{head-adaptive}}
).
$$

而且 SCOPE 的 Figure 7 做了非常直接的三阶段消融：

> Top-p → Top-p + Top-k → Top-p + Top-k + Online

在 matched realized density 下，加入 fixed floor 后三个模型 PSNR 都明显提升，继续加入 online per-head estimation 后又进一步提升；论文特别强调这不是简单“多算了一点”，因为他们匹配了 realized attention density。

例如：

* Wan2.2：17.72 → 24.37 → **25.02**
* Wan2.1：14.91 → 23.34 → **23.83**
* HunyuanVideo：21.58 → 25.07 → **25.41**

所以你如果正在考虑下一步怎么改 hybrid 策略，我认为 **SCOPE 的思路比单纯 global \(k_{\min}\) 更值得继续研究**。

[SCOPE 论文](https://arxiv.org/abs/2608.12780?utm_source=chatgpt.com)

---

# 7. Twilight：另一种 Top-k + Top-p 结合，逻辑和前面不一样

这个很容易被忽略。

Twilight 虽然通常被归类成“Top-p sparse attention”，但实际系统是：

$$
\boxed{\text{Top-k-like Selector}\rightarrow\text{Top-p Pruner}}
$$

而不是：

$$
S_k\cup S_p.
$$

它先让 Quest / Double Sparsity 等已有 sparse selector 用一个**偏大的 conservative budget**选出候选集合 \(I_0\)，然后在 \(I_0\) 内重新估计 attention weights，再做 Top-p：

$$
I_1=\operatorname{TopP}(I_0,p).
$$

最终：

$$
I_1\subseteq I_0.
$$

论文把它称为：

> **Select-then-Prune architecture**。

所以目前其实存在两种完全不同的“Top-k + Top-p”范式：

### A. Floor / Union 型

SpargeAttention2、RAPID、FVAttn、SCOPE：

$$
K=\max(K_p,K_{\min})
$$

Top-k 是**下界**。

### B. Candidate → Prune 型

Twilight：

$$
I_p\subseteq I_k
$$

Top-k 是**候选集上界**，Top-p 决定最终预算。

这一点我认为是调研中非常值得明确分开的。

---

# 8. 哪些论文专门值得看“Top-k vs Top-p 的区别分析”

如果你的重点不是“有哪些方法”，而是准备写论文里的 **motivation / related work / design analysis**，下面几篇最有价值。

### 第一梯队：直接分析 Top-k vs Top-p

**1. Twilight, NeurIPS 2025**

这是从 **LLM sparse attention / theoretical budgeting** 角度最系统的一篇。

核心观点：

$$
\text{Top-k}: \max_{|I|=B}\sum_{i\in I}W_i
$$

固定 \(B\)。

Top-p 则是：

$$
\min_I|I|
\quad
\text{s.t.}\quad
\sum_{i\in I}W_i\geq p.
$$

论文给出的 top-p output error bound 是：

$$
(1-p)\|V\|_F.
$$

同时非常明确地分析了：

* Top-k fixed budget 的 over-selection；
* Top-k fixed budget 的 under-selection；
* Top-p adaptive budget；
* Top-p 比 Top-k 需要更高 proxy precision；
* Top-p 需要 normalization；
* GPU kernel 实现难度。



---

**2. SpargeAttention2, 2026**

这篇是从 **video diffusion + extreme sparsity** 的角度反驳“Top-p 总是更好”。

它真正重要的地方是指出：

$$
\boxed{\text{Top-p 也有 failure mode}}
$$

特别是 attention sink / skewed attention。

并且第一次非常直接地做：

$$
\text{Top-k vs Top-p vs Top-k+Top-p}
$$

在 uniform / skewed 分布上的 controlled comparison。([arXiv][6])

所以如果你的论文要论证 hybrid，这篇几乎是必须引用。

---

**3. RAPID, CVPR 2026**

它没有 SpargeAttention2 那么偏理论，但有很强的 empirical evidence：

$$
\text{Top-p},\quad
\text{Top-k},\quad
\text{Top-k+Top-p}
$$

在 matched density 下直接比较，而且 hybrid consistently best。([CVPR Open Access][3])

非常适合支撑一句：

> “Hybrid selection has been empirically shown to outperform either fixed-budget or mass-based selection alone under comparable sparse-attention density.”

---

**4. SCOPE, 2026**

它把问题进一步细化成：

> **Top-p 不是只有真实 attention distribution 的问题，还有 proxy-distribution calibration 的问题。**

SCOPE 指出 approximate block/cluster proxy 可能造成 softmax **over-concentration**，于是：

$$
\hat P \text{比真实 }P\text{更尖}
$$

导致

$$
K_p(\hat P)\ll K_p(P),
$$

最终严重 under-select。

因此 Top-k floor 除了抵抗 attention sink，还可以被理解成：

$$
\boxed{\text{对 proxy calibration error 的 robustness constraint}}
$$

这一点比 SpargeAttention2 又前进了一步。论文消融明确显示 fixed floor 可以显著改善纯 Top-p。

---

# 9. 另外几篇“纯 Top-p 阵营”的分析论文也值得看

它们没有直接采用 Top-k+Top-p union，但对你理解这个设计空间很重要。

**Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs**，ICLR 2026，也是反对 fixed Top-k budget，改为根据 cumulative attention score 动态确定数量。([OpenReview][7])

[Tactic 论文](https://arxiv.org/abs/2502.12216?utm_source=chatgpt.com)

**Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs**，2026 年，明确写道 fixed-budget Top-k 无法适应 head/layer 间异质 attention distribution，而 Top-p 可以 preservation attention mass；它随后重点解决 Top-p 的 estimation / selection / sparse computation overhead。([arXiv][8])

[Double-P 论文](https://arxiv.org/abs/2602.05191?utm_source=chatgpt.com)

**Training-free sparse attention based on cumulative energy filtering**，2026 年 6 月。这篇提出一个很有用的二维解释：

$$
\text{Top-k}\rightarrow \text{固定计算预算}
$$

而

$$
\text{Top-p}\rightarrow \text{固定 accuracy / recall constraint}.
$$

作者认为两者其实对应两个不同优化目标，并进一步提出 dynamic threshold。([arXiv][9])

[Cumulative Energy Filtering 论文](https://arxiv.org/abs/2606.16317?utm_source=chatgpt.com)

---

# 10. 一个看起来矛盾、但实际上很重要的研究结论

你读这些论文时会发现一个明显“冲突”。

Twilight / Tactic / Double-P 基本在说：

> **Top-p 比 fixed Top-k 更合理，因为预算应该随 attention distribution 自适应。**

而 SpargeAttention2 / RAPID / SCOPE 又在说：

> **纯 Top-p 不够稳，必须有 Top-k floor。**

实际上这两边并不矛盾。

它们研究的 assumption 不同。

对于理想、准确的 attention probability：

$$
P=\operatorname{softmax}(QK^\top)
$$

Top-p 确实拥有非常漂亮的 mass-preservation 性质。

但是实际 sparse attention 通常不是拿真实 \(P\) 做 selection，而是：

$$
Q,K
\rightarrow
\text{pool / cluster / quantize / proxy}
\rightarrow
\tilde P
\rightarrow
\text{sparse mask}.
$$

于是出现：

$$
\tilde P\neq P.
$$

而 Top-p 恰好**非常依赖 \(\tilde P\) 的 calibration**，因为它不只看 ranking，还看概率值。

Top-k 更多依赖：

$$
\operatorname{rank}(\tilde P).
$$

因此只要排名基本正确，数值尺度偏了未必特别严重。

所以现在这条研究线实际上正在变成：

$$
\boxed{
\text{Top-p adaptive budget}
+
\text{Top-k robustness floor}
}
$$

再从：

$$
\text{global fixed }k
$$

演化到

$$
\boxed{\text{head/input-adaptive }k}.
$$

**SCOPE 就是目前这个演化方向非常典型的最新一步。**

---

# 11. 我建议你重点沿这条文献链看

如果目标是研究 **“Top-p 与 Top-k hybrid sparse attention”**，我会把核心阅读顺序排成：

$$
\textbf{Twilight}
\rightarrow
\textbf{SpargeAttention2}
\rightarrow
\textbf{RAPID}
\rightarrow
\textbf{FVAttn}
\rightarrow
\textbf{SCOPE}.
$$

Twilight 回答：

> **为什么需要 Top-p？**

SpargeAttention2 回答：

> **为什么 Top-p 也不够，为什么需要 Top-k + Top-p？**

RAPID 回答：

> **这种 hybrid 在 video diffusion 中是否有强 empirical evidence？**

FVAttn 回答：

> **Top-p 的动态 budget 会带来什么硬件/并行问题？**

SCOPE 回答：

> **为什么 global fixed Top-k floor 仍然不够，能不能变成 online per-head floor？**

如果从研究空白看，我现在觉得最值得继续挖的不是再做一个简单的

$$
\max(k,k_p)
$$

而是进一步研究：

$$
\boxed{
k_{q,h,l,t}
=
f(
\text{attention entropy},
\text{proxy calibration},
\text{head},
\text{layer},
\text{denoising step},
\text{hardware budget}
)
}
$$

也就是让 **Top-k floor 自己成为动态变量**，同时保留 Top-p 的 mass constraint。SCOPE 已经向 \(k_h\) 迈了一步，但 query/head/layer/timestep 与 hardware-aware 联合预算还明显有继续做的空间。

如果你接下来是在做论文方向，我可以继续把这些论文整理成一份 **“Top-k / Top-p sparse attention 文献矩阵”**，逐篇列出 **selection granularity、proxy、Top-k/Top-p 规则、是否 trainable、kernel、实验模型、sparsity、speedup、核心缺点**，这样基本就可以直接拿来做 related work 和寻找创新点。

我也可以持续追踪这个方向后续新出的 Top-k/Top-p sparse-attention 论文。

[1]: https://arxiv.org/abs/2602.13515?utm_source=chatgpt.com "SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning"
[2]: https://openaccess.thecvf.com/CVPR2026?day=2026-06-07&utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[3]: https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_RAPID_Reusing_Attention_Sparsity_with_Inter-step_Adaptation_for_Efficient_Video_CVPR_2026_paper.pdf?utm_source=chatgpt.com "RAPID: Reusing Attention Sparsity with Inter-step Adaptation for Efficient Video Diffusion"
[4]: https://arxiv.org/abs/2607.16190?utm_source=chatgpt.com "FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation"
[5]: https://arxiv.org/abs/2608.12780?utm_source=chatgpt.com "SCOPE: Subspace Clustering with Online Per-Head Top-K Estimation for Sparse Video Attention"
[6]: https://arxiv.org/html/2602.13515v1 "SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning"
[7]: https://openreview.net/pdf?id=tJod11fK1A&utm_source=chatgpt.com "Published as a conference paper at ICLR 2026"
[8]: https://arxiv.org/abs/2602.05191?utm_source=chatgpt.com "Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs"
[9]: https://arxiv.org/abs/2606.16317?utm_source=chatgpt.com "Training-free sparse attention based on cumulative energy filtering"
