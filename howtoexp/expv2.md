实验：Fine Top-k 驱动的多粒度执行

目标：
验证先在 \(16\times16\) 粒度完成统一 Top-k 选择，再根据局部 occupancy 自适应选择 \(16\times16\) Triton 或 \(128\times96\) FlashInfer，是否能降低真实 GPU 时间。

1. Fine Top-k

对每个 \((head,Q16)\) 在 K16 维做 Top-k，得到 compact fine indices：

$$ F=\{(h,q_{16},k_{16})\}. $$
2. 8-bit Macro Occupancy 统计

每个 fine tile 映射到：

$$ q_m=\lfloor q_{16}/8\rfloor,\qquad k_m=\lfloor k_{16}/6\rfloor. $$

每个 \(128\times96\) macro 最多包含：

$$ 8\times6=48 $$

个 microtiles，因此只需维护：

$$ \boxed{ C[h,q_m,k_m]\in \texttt{uint8} } $$

并对每个 selected microtile：

$$ C[h,q_m,k_m] \mathrel{+}=1. $$

因为：

$$ 0\le C\le48<256, $$

所以 uint8 足够，相比 int32 occupancy buffer 可降低到约 \(1/4\) 的存储量。

3. Granularity Scheduling

根据 occupancy：

$$ n_g=C[g]. $$

若：

$$ n_g\ge\tau, $$

则该区域 densify 为完整：

$$ 128\times96 $$

Macro，交给 FlashInfer。

否则保留原始 selected：

$$ 16\times16 $$

microtiles，交给 grouped Triton。

被 Macro 接管的 fine indices 必须从 Residual 中删除：

$$ Core\cap Residual=\varnothing. $$
4. Compact Route 构建

最终生成两套紧凑索引：

Fine Top-k indices
        ↓
uint8 macro occupancy
        ↓
   n >= τ ?
    /    \
  Yes     No
   ↓       ↓
Macro     Micro
route     CSR route
   ↓       ↓
FlashInfer Triton

整个过程在 CUDA 上完成，不保留完整三维 fine bool mask。

5. Threshold Sweep

测试：

$$ \tau\in\{8,16,24,32,40,48\}. $$

并比较：

All-Micro：全部走 Triton；
Adaptive：uint8 occupancy + adaptive dispatch；
必要时增加 All-Macro 作为参考。
6. 必须记录的时间

分别统计：

$$ T_{\text{TopK}}, \quad T_{\text{occupancy}}, \quad T_{\text{partition}}, \quad T_{\text{plan}}, \quad T_{\text{macro}}, \quad T_{\text{micro}}, \quad T_{\text{merge}}. $$

重点看：

$$ \boxed{ T_{\text{route}} = T_{\text{occupancy}}+ T_{\text{partition}} } $$

是否足够小，以及：

$$ \boxed{ T_{\text{adaptive}}<T_{\text{all-micro}}. } $$
7. 两个关键验证

第一，测 kernel crossover：

$$ T_{\text{micro}}(n) \quad\text{vs.}\quad T_{\text{macro}}(48), $$

找到：

$$ n^*. $$

第二，统计真实 workload 的：

$$ n_g\in[0,48] $$

occupancy histogram，确认大量区域是否落在 crossover 两侧。

最终希望证明：

Fine Top-k 决定选什么； 8-bit occupancy 以极低开销统计局部密度； 局部密度决定使用哪种 GPU 粒度。