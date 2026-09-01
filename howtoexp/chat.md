结论：就这两份 timing 而言，当前方法相对 DFSAttn 的最大问题几乎就是 `flashinfer_core_csr_expand`，而且这 32 ms 很可能主要来自本地 FlashInfer 展开 kernel 的低效实现，不是 CSR 展开天然必须这么慢。

### 1. timing 对比

| 指标 | Ours | DFSAttn | 差值 |
|---|---:|---:|---:|
| E2E GPU | 285.217 s | 211.328 s | +73.889 s |
| Attention | 164.461 s | 88.612 s | +75.849 s |
| CSR expand | 74.004 s | — | 32.458 ms/次 |
| Core run（含 expand） | 90.501 s | — | 39.694 ms/次 |

`flashinfer_core_run` 是外层计时，包含 `flashinfer_core_csr_expand`，见 [flashinfer64_attention.py](/cnic/work/liutt/mywork/attention_time/code/DFSAttn/dfsattn/flashinfer64_attention.py:902)。

所以真正 FA3 `paged_run` 的粗略时间是：

```text
39.694 - 32.458 ≈ 7.236 ms/call
```

如果只移除 CSR expand：

```text
Ours attention = 164.461 - 74.004 = 90.458 s
mean attention = 90.458 / 3000 = 30.153 ms
DFSAttn        = 29.537 ms
```

两者已经非常接近。CSR expand 解释了约：

```text
74.004 / 75.849 = 97.6%
```

的 attention 差距，也几乎完整解释了 73.889 s 的 E2E 差距。因此，在这次实验上答案是“是的，而且非常确定”。

### 2. 为什么要展开成 token-level offsets

不是为了“把 FA3 的 dense attention 计入统计”，而是为了适配 FA3 paged-attention 的输入 ABI。

当前路径是：

```text
macro CSR
每个元素表示一个 K96 block base
        ↓ expand
vector offsets
每个被选择的 KV token 一个 int32 offset
        ↓
FA3 paged_run(page_size=1)
```

你们把每个 head 展成一个独立 batch，并把 K/V reshape 成 `page_size=1` 的 paged KV：

```python
k_flat = k.reshape(heads * sequence, 1, 1, dim)
```

然后 FA3 通过 token-level `vector_indices` 找到这一 CSR row 允许访问的 KV tokens，见 [flashinfer64_attention.py](/cnic/work/liutt/mywork/attention_time/code/DFSAttn/dfsattn/flashinfer64_attention.py:611)。

因此更准确的描述是：

- FA3 在每个选中的 K96 block 内执行稠密 QK attention；
- 但不同 macro blocks 之间仍然是稀疏的；
- token-level offsets 是地址/页表元数据，不是为了恢复或统计全局 dense attention；
- `core_interactions` 才是统计 Core 实际计算量的东西，与这次展开不是同一目的。

FlashInfer 官方也把 block-sparse attention定义为用 BSR/CSR mask 得到与 dense masked reference 等价的结果，而不是执行完整 dense attention。[FlashInfer sparse API](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sparse.py)

### 3. 当前方案并没有保存每层 vector offsets

当前保存的是：

- 每层的 `macro_indptr`
- 每层的 `macro_bases`
- 每层的 `vector_indptr`
- 每层的 FA3 scheduler `plan_info`
- 一个全模型共享的 vector-offset workspace

见 [flashinfer64_attention.py](/cnic/work/liutt/mywork/attention_time/code/DFSAttn/dfsattn/flashinfer64_attention.py:472)。

但是每次 `run()` 都仍然调用：

```python
block_sparse_indices_to_vector_sparse_offsets(...)
```

见 [flashinfer64_attention.py](/cnic/work/liutt/mywork/attention_time/code/DFSAttn/dfsattn/flashinfer64_attention.py:586)。

所以当前优化解决的是：

- cache12 下不重复调用 scheduler `plan()`
- 不重复分配 512 MB 默认 workspace
- 不通过 Python VariableBlock 路径做展开

但没有解决 token offsets 的重复物化。

确实可以给每层保存完整 `vector_indices`，因为 route cache12 期间稀疏 pattern 不变。不过代价很大。根据这次 `sparsity_records.csv` 估算：

- 平均每层约 259 MiB
- 最大层约 591 MiB
- 60 层约 15.4 GiB

所以“每层缓存 offsets”能消除展开，但会把目前约 0.6 GiB 的共享最大 workspace 变成约 15 GiB 常驻显存。它是一个可行的显存换时间方案，但不是首选。

### 4. 现在最值得先修的是 FlashInfer 展开 kernel

你本地 FlashInfer 的 kernel 是：

```cpp
for (int b = blockIdx.x; b < batch_size; ++b)
```

见 [page.cuh](/cnic/work/liutt/mywork/attention/Sparse-VideoGen/svg/kernels/3rdparty/flashinfer/include/flashinfer/page.cuh:287)。

这意味着：

- CTA 0 处理 row 0、1、2、3……
- CTA 1 又处理 row 1、2、3……
- CTA 2 又处理 row 2、3……

很多后面的 CSR rows 会被多个 CTA 重复写入完全相同的数据。grid 又使用 `num_sms` 个 CTA，见同一文件的第 330 行，所以重复量可能接近 SM 数量级。这非常符合“物化几百 MB int32 offsets 却花 32 ms”的异常表现。

优先建议验证两种改法：

```cpp
for (int b = blockIdx.x; b < batch_size; b += gridDim.x)
```

或者直接：

```cpp
grid = batch_size;
b = blockIdx.x;
```

后一种是一行一个 CTA；考虑到每个 CSR row 的 `kv_lens` 较大，通常更加自然。修完重新编译 FlashInfer，再跑同一配置。如果展开降到亚毫秒或几毫秒，就没有必要付出 15 GiB 做 per-layer offset cache。

### 5. SVG2 怎么处理这个问题

需要区分原始 SVG2 和你本地仓库后续加入的 patch。

原始 SVG2 的动态路径并没有消除 token-level 展开。它使用：

```python
VariableBlockSparseAttentionWrapper
wrapper.plan(...)
wrapper.run(...)
```

而且原始代码每次 sparse forward 都新建 wrapper、workspace 并重新 plan。VariableBlock plan 会把 variable blocks 展开成 token-level `kv_indices`。也就是说，原版实际上是把这部分放在 `Planning` 里支付，并非通过保存每层 offsets 解决。

你本地 SVG2 后来加入了一个 Triton patch：

```python
_svg_kvidx_kernel
```

见 [flashinfer_patch.py](/cnic/work/liutt/mywork/attention/Sparse-VideoGen/svg/flashinfer_patch.py:16)。

它做的是：

- 一个 Triton program 对应一个 variable block；
- 直接从 `base + arange(length)` 写出 token indices；
- 替换 FlashInfer 原本的 `repeat_interleave + arange`；
- 跳过不必要的 `kv_indices` CPU copy。

见 [flashinfer_patch.py](/cnic/work/liutt/mywork/attention/Sparse-VideoGen/svg/flashinfer_patch.py:63)。

因此 SVG2 当前的思路是“更高效地生成完整 token indices”，不是“完全不生成”，也不是“给每层永久缓存 vector offsets”。SVG2 仓库里也有一个仍未关闭的问题专门讨论 planning-stage expansion 是否是瓶颈。[SVG2 issue #68](https://github.com/svg-project/Sparse-VideoGen/issues/68)

### 建议优先级

1. 修复/替换本地 FlashInfer page-op 的重复 row 写入。
2. 单独 microbenchmark `block_sparse_indices_to_vector_sparse_offsets`，确认输出量、有效带宽和正确性。
3. 重新跑同一 timing；理论上 attention 应从 54.82 ms 接近 30 ms。
4. 若修 kernel 后仍慢，再考虑：
   - 只缓存最慢/最密的少数层 offsets；
   - 使用 FA2 block CSR 路径作对照；
   - 将 K96 改成 FlashInfer 新 compact block-sparse kernel支持的 K64/K128，使 attention kernel直接消费 block IDs。当前官方 compact BSR API已支持 runtime block indices，但 K96 不是其直接支持的 block size，不能无修改替换。[FlashInfer block-sparse API](https://docs.flashinfer.ai/api/attention.html)有明显进步，但你给的路径还是修复前的旧结果，时间戳是 13:53。真正的新结果在：

[新 timing.csv](/cnic/work/liutt/mywork/attention_time/res/flashinfer64/vbench_11/HunyuanVideo/pageopfix_vbench11_seed0_tilekratio0.16_totaltopp0.50_promote12_hilbert3d_480_routecacheTrue_coreonlyFalse_directcsrTrue_cache12/timing.csv)

对比如下：

| 指标 | 修复前 | 修复后 | 改善 |
|---|---:|---:|---:|
| CSR expand | 32.458 ms | 5.187 ms | 6.26× |
| Core run（含 expand） | 39.694 ms | 12.486 ms | 3.18× |
| Attention | 54.820 ms | 33.035 ms | 1.66× |
| E2E | 285.217 s | 220.196 s | 减少 65.02 s |

相对 DFSAttn：

| 指标 | Ours 修复后 | DFSAttn | 差距 |
|---|---:|---:|---:|
| Attention | 33.035 ms | 29.537 ms | 慢 11.8% |
| E2E | 220.196 s | 211.328 s | 慢 8.87 s / 4.2% |

还有一个很有说服力的结果：

```text
修复前实际 FA3 = 39.694 - 32.458 = 7.236 ms
修复后实际 FA3 = 12.486 - 5.187  = 7.300 ms
```

FA3 本身几乎完全没变，说明这 65 秒加速确实来自修复 CSR expand，而不是路线或 attention 计算发生变化。Residual route statistics 也完全一致，说明稀疏模式没有改变。

不过实际推理中的 CSR expand 仍是 5.187 ms，比独立微基准的 0.253 ms 高很多。目前它仍然是剩余差距的第一嫌疑。如果能进一步降到约 0.3–1 ms，预计：

```text
Attention ≈ 28.1–28.8 ms
E2E ≈ 209–211 s
```

也就是有机会追平甚至略快于 DFSAttn。

下一步应针对实际非均匀 CSR 优化 CTA 调度。当前虽然消除了重复写入，但仍只有 `num_sms` 个 CTA，每个 CTA 串行处理很多 CSR rows；可以改成“一行一个 CTA”或使用更多 CTA，以减少不同 row 长度造成的负载不均衡。