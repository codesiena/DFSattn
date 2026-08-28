修改前：

- DFSAttn 用统一粒度执行：默认 `128×128` block mask，所有选中的块都交给同一种 block-sparse kernel。
- 代码只区分“选中 / 未选中”，不区分局部区域是稀疏还是稠密。

修改后（Hybrid 模式）：

- 必须使用 `--block_size 16`，所以逻辑 mask 的每个格子是一个 `16×16` microblock。
- 在 `partition_block_mask()` 中，把每个 `4×4` microblock 区域看作一个 `64×64` macro tile：
  - 计算其中选中的 `16×16` 块数 `n_B`；
  - `n_B >= hybrid_threshold`：归为稠密 Core 部分；
  - `n_B < hybrid_threshold`：归为稀疏 Residual 部分。
- 例如阈值为 8：一个 `64×64` 区域内 16 个 microblock 至少选中 8 个，就走 Core64；否则保留为若干 `16×16` Residual 块。

代码层面：

```python
promote = grouped.sum(dim=(-1, -2)) >= threshold
core = promote.nonzero()
residual_mask = grouped & ~promote[..., None, None]
residual = residual_mask.nonzero()
```

Core 部分是否“直接对接 64 Tensor Core”：

- 当前实现会把所有 Core tile 收集成紧凑 batch，并执行 `64×64` QK batched matmul：
  
  ```python
  scores = torch.bmm(q_tile, k_tile.transpose(1, 2))
  ```

- 当运行在 CUDA 且 Q/K 为 BF16 或 FP16 时，这种 `64×64` matmul 是 Tensor Core eligible，PyTorch/CUDA 后端通常会选择 Tensor Core 路径。
- 但它不是手写 WMMA/CUTLASS/Triton Tensor Core kernel，因此“是否实际发射 Tensor Core kernel”仍由 PyTorch 后端、GPU 架构和 shape 决定；需要在目标 GPU 上用 benchmark/profiler 最终确认。

关键点：Core64 虽然计算完整 `64×64` QK，但未被 DFSAttn 选中的 `16×16` 子块仍置为 `-inf`，所以逻辑 mask 和原 DFSAttn 完全一致；只改变执行粒度。