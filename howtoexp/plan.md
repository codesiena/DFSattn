我觉得这个方向**是成立的，而且比“简单把 SpargeAttention2 的 Top-k + Top-p 换个粒度”更有意思**。如果把它定义得准确一些，我会把核心 idea 概括成：

> **Block-level Top-p 负责规则的大块 attention，用 Tensor Core；token-level Top-k 只负责被 block Top-p 丢掉区域里的少量“重要残差 token”，用 CUDA Core；两条路径在同一个 attention kernel 内共享 Q/K/V tile、共同完成 softmax 和输出累加。**

这实际上把前两篇文章的“quality problem”与第三篇 FlashQuant 的“heterogeneous execution problem”接上了。我认为真正值得做的点，不只是 `Top-p(block) + Top-k(token)`，而是 **block–token heterogeneous sparse attention kernel**。

### 为什么这个思路是合理的

SpargeAttention2 现在的 Top-k 和 Top-p **本质上都还是 block-level 的**。它先 pool Q/K 得到 (\bar P)，然后 Top-k 和 Top-p 都作用在这个 block-pooled attention map 上，最后取两者的 union。论文公式就是
[
\bar M=\operatorname{TopK}(\bar P)\cup\operatorname{TopP}(\bar P),
]
并没有真正进入 token-level refinement。 它的 Algorithm 1 也是先构造 block mask，然后整块执行 (Q_iK_j^T) 和 (P_{ij}V_j)。

所以你这里其实可以做一个很自然的升级：

[
\underbrace{\text{Top-p Block}}*{\text{coarse / regular}}
+
\underbrace{\text{Top-k Token Residual}}*{\text{fine-grained / irregular}}.
]

Top-p 保证“大部分 attention mass”通过规则 block 被保留；Top-k token 则专门去**救回 block mask 不够精细而误删的关键 token**。

这正好对应 DFSAttn 发现的问题：视频 DiT 的 attention sparsity 是 dynamic + fine-grained，粗 block 很容易因为一个 block 内语义混杂而损失重要 interaction。DFSAttn 因此用 Hilbert3D + sub-block scoring 来改善 block selection。 而它的实验也非常说明问题：block size 固定 128 时，sub-block 从 64 降到 16，PSNR 从 28.894 提高到 29.378，latency 基本没变化。

但 DFSAttn 最后执行阶段仍然是 **block sparse FlashAttention**，并没有真的保留 token-level irregular computation。它用的是 block=128、sub-block=16，而且明确说 block 128 是为了匹配 GPU kernel execution。

所以你的切入点非常清楚：

**DFSAttn 用 fine-grained information 来“选 block”；你可以进一步让 fine-grained information 真的以 token computation 的形式保留下来。**

---

### 但有一个非常关键的地方：不能直接照搬 FlashQuant 做 “TC结果 + CUDA结果”

这是这个方向最重要的技术问题。

FlashQuant 能直接写成

[
Y = XW_Q + XW_O
]

是因为矩阵乘法是线性的。因此 regular dense path 在 Tensor Core 算，outlier sparse path 在 CUDA Core 算，最后直接把两个 partial output 加起来即可。FlashQuant 正是把两条路径放进同一个 CTA，activation tile 只 load 一次，两条路径共享 shared memory，最后 on-chip 累加。 它进一步做 warp specialization：regular path 用 Tensor Core MMA，irregular sparse path 用 CUDA cores。

**Attention 不一样。**

你不能简单做：

[
O =
O_{\text{block-TC}}
+
O_{\text{token-CUDA}}
]

因为中间有 softmax：

[
O=
\frac{\sum_{j\in\Omega}e^{s_j}V_j}
{\sum_{j\in\Omega}e^{s_j}}.
]

block path 和 token path 必须共享同一个 normalization。

不过这个问题是可以漂亮解决的，而且我反而觉得这可能成为你 kernel 最核心的技术点。

每条路径先独立产生一个 FlashAttention-style state：

[
(m,l,o),
]

其中

[
m=\max_j s_j,
\qquad
l=\sum_j e^{s_j-m},
\qquad
o=\sum_j e^{s_j-m}V_j.
]

假设 Tensor Core block path 得到

[
(m_b,l_b,o_b)
]

而 CUDA token path 得到

[
(m_t,l_t,o_t),
]

两者可以通过 online-softmax merge：

[
m=\max(m_b,m_t),
]

[
l=e^{m_b-m}l_b+e^{m_t-m}l_t,
]

[
o=e^{m_b-m}o_b+e^{m_t-m}o_t,
]

最后

[
O=o/l.
]

**这样 Tensor Core 和 CUDA Core 可以分别计算不同粒度的 attention interaction，但仍然得到数学上正确的统一 sparse softmax。**

这一点我觉得非常强，因为它不是泛泛的 heterogeneous kernel，而是一个真正针对 attention 非线性的 **heterogeneous online-softmax fusion**。

---

### 我会怎么具体设计

我不会让 token-level Top-k 在整个 (N\times N) attention space 上直接算，因为那样 selection 本身就可能把节省的 FLOPs 全吃回来。

我会把整个方法做成一个 **coarse-to-fine residual selection**：

1. **Block Top-p coarse path。** 例如 Q block / KV block 都是 64，先用 pooled Q/K 或 DFSAttn 的 hierarchical score 得到 block score。Top-p 选中完整的 (64\times64) tile。这些 tile 直接走 Tensor Core。这里最好表述成“64×64 Tensor-Core-friendly tile”，而不是“生成 64 个 Tensor Core”，因为实际硬件上 64×64 tile 会被进一步拆成 MMA/WGMMA instruction。

2. **Token Top-k residual path。** 只在 **Top-p 没选中的 block 中** 找 token。也就是说
   [
   \Omega_q =
   \Omega^{block}*{p,q}
   \cup
   \Omega^{token}*{k,q},
   ]
   并要求
   [
   \Omega^{token}*{k,q}
   \subseteq
   \overline{\Omega^{block}*{p,q}},
   ]
   这样绝不会重复计算。Top-k 的意义不是再选一遍最重要 token，而是专门做 **block pruning error correction**。

3. **不要全局 exact token Top-k。** 可以先像 DFSAttn 一样 block=64、sub-block=8/16，对被 Top-p 排除但 score 接近 threshold 的“frontier blocks”做细粒度 scoring，然后只在这些 candidate blocks 里选 token Top-k。DFSAttn 的 Algorithm 1 已经证明了 hierarchical scoring + mask caching 是一个合理框架。

4. **Single fused kernel。** 一个 CTA 对应一个 Q tile。Q tile 只 load 一次；block KV tile 和 residual sparse K/V metadata 分别进入 shared memory。部分 warps 负责 Tensor Core regular blocks，少量 sparse warps 负责 CUDA-core residual token。FlashQuant 页 3–5 的真正启发就在这里：不是简单“两个 kernel 同时跑”，而是**共同 tile hierarchy + shared data reuse + warp specialization**。 FlashQuant 甚至专门针对 irregular workload 做 bucket / reordering 来解决 load imbalance 和 bank conflict，这个思路也可以借给你的 residual-token metadata。

5. **Mask caching + distillation。** inference kernel 和 quality adaptation 最好分开看。DFSAttn 已经显示 mask 可以跨 diffusion steps cache，并且只定期更新，而 sparse attention output 每步仍重新算。 另一方面，SpargeAttention2 的结果说明在非常高 sparsity 下 training / velocity distillation 对 quality 很重要；它的 hybrid Top-k+Top-p 明显优于单独 Top-k/Top-p。 所以最后完全可以做成“kernel training-free 可运行 + distillation 后达到最佳 quality”。

---

### 我觉得你最应该强调的 novelty

如果论文写成：

> “Top-p 用 block，Top-k 用 token，然后 Tensor Core + CUDA Core。”

这个说法还是有点像工程组合，reviewer 很可能会问：“为什么不是一个简单 hierarchical sparse attention？”

但如果把它定义成：

**Heterogeneous Granularity Sparse Attention**

或者更具体一点：

**Block-Token Fused Sparse Attention**

核心 claim 变成：

> Existing block sparse attention trades fine-grained recall for GPU efficiency. We decouple attention sparsity into a regular high-mass block component and an irregular token-level residual component, and co-execute them on Tensor Cores and CUDA Cores within a unified online-softmax kernel.

这个故事就完整很多。

对应三个 contribution 可以非常自然：

**算法层面**：Top-p block captures bulk probability mass；Top-k token residual recovers critical interactions lost by block quantization。

**kernel 层面**：Tensor-Core block path + CUDA-Core token path，同 CTA / shared memory，并通过 unified online-softmax state merge 实现 mathematically correct fused attention。

**video 层面**：在相同 latency / FLOPs 下，比纯 block sparse 更高 attention recall 和 video quality；在相同 quality 下，允许 Top-p block 更 aggressive，从而 end-to-end 更快。

尤其第三点特别重要：**你的速度收益不能指望“多加了 CUDA token path 还天然更快”。**

真正应该论证的是：

> 有 token residual 兜底以后，可以把 block Top-p 砍得更狠。

例如原来需要保留 20% blocks 才不掉 quality，现在可能：

[
12%\text{ block tiles}
+
1%\text{ token residual}
]

就达到甚至超过原来 20% block 的 fidelity。

这时候才会同时得到：

[
\text{quality}\uparrow,\qquad
\text{computation}\downarrow.
]

这才是整个 paper 最关键的 hypothesis。

---

### 这里有三个风险，我会优先验证

第一个风险是 **token Top-k selection overhead**。如果为了找那 1–2% token 又算了一遍大量 (qk)，整个方法就没意义。因此第一次实验甚至不需要写 CUDA kernel，先用 oracle/full attention map 离线测：

[
\text{block Top-p}
\quad vs\quad
\text{block Top-p + residual token Top-k}.
]

看同样 retained pair 数量下，attention output error / recall 能不能明显提高。如果 oracle 都没明显提升，kernel 就不用写了。

第二个风险是 **CUDA Core residual path 太大**。FlashQuant 中 CUDA path 好用，是因为 outlier 极少，而且两条路径可以共享 activation。你的 token residual 同样必须非常 sparse。我的直觉是让 CUDA residual FLOPs 控制在 Tensor-Core path 的 **5–15% 左右**更合理；一旦残差占到 30–40%，warp imbalance 和 gather overhead 很可能开始反噬。

第三个风险是 **block=64 是否真的是最佳点**。64 很适合你的故事，也比 DFSAttn 的 128 更细，但不能先假定一定最快。DFSAttn 明确用了 block 128 来配合现有 sparse kernel。 你的 custom kernel 如果针对 Hopper/H100 做 64×64 tile，确实可能有优势，但最好 ablate 32 / 64 / 128，最终让“64”是 benchmark 得出的，而不是设计前提。

总体上，我会给这个方向很高的优先级。**算法上有明确的问题来源，kernel 上又恰好可以借 FlashQuant 的 heterogeneous fusion，但 attention 的 softmax 又带来了一个 FlashQuant 没有解决的新问题。** 这意味着它既不是单纯抄 DFSAttn，也不是把 FlashQuant 搬到 attention 上。

如果让我现在开始做，我第一阶段甚至完全不写 kernel：先做一个 oracle prototype，画出一条 **quality–cost Pareto curve**，验证“少量 token residual 是否真的能让大量 block 被额外删除”。只要这条 curve 明显优于 SpargeAttn2 block hybrid 和 DFSAttn block-only，后面的 fused Tensor Core/CUDA Core kernel 就非常值得投入。
ihe