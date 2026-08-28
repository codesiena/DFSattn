# 已完成：Phase-A HyVideo Block Top-p + Token Top-k

## 1. 当前实现

已在 HyVideo 的 DFSAttn attention path 中加入 Block Top-p 和 residual Token
Top-k 实验支持。

当前 residual selector 使用 bounded frontier 方案，默认设置：

```text
R = 4
```

对每个 query block：

1. 使用 DFS hierarchical block score 完成 block Top-p；
2. 在未被 block mask 选中的 blocks 中，选 score 最高的 4 个 frontier blocks；
3. 仅在这些 frontier blocks（最多 `4 × 128` 个 key tokens）中计算 exact QK
   score，并执行 token Top-k；
4. 将 selected block tokens 和 residual tokens 放入同一个 softmax。

因此当前实现不再构造完整的 `N × N` token score，也不会扫描所有未选 token。

注意：当前 residual Top-k 是 frontier 范围内的 exact Top-k，不是所有未选
token 中的 global exact Top-k，属于 Phase-A 的 bounded-frontier 近似版本。

## 2. 新增参数

Inference driver：

```text
--block_top_p P
    启用 block Top-p，要求 `0 < P <= 1`；省略时保持原 DFS block Top-k。

--token_top_k K
    residual token Top-k，常用值为 `0/4/8/16/32`；`0` 表示不启用 residual。

--residual_candidate_blocks R
    每个 query block 搜索的未选 frontier block 数，默认 `4`。
```

当 `--token_top_k > 0` 时，`--residual_candidate_blocks` 必须大于 0。

Shell wrapper 也支持对应环境变量：

```text
BLOCK_TOP_P
TOKEN_TOP_K
RESIDUAL_CANDIDATE_BLOCKS
```

输出目录会将 `p_mass`、`token_top_k` 和 candidate block 数编码进目录名，
避免 sweep 时不同设置互相复用已有视频。

## 3. 推荐运行方式

例如运行 `p_mass=0.6`、`token_top_k=8`、`R=4`：

```bash
cd /cnic/work/liutt/mywork/attention_time/code/DFSAttn

BLOCK_TOP_P=0.6 \
TOKEN_TOP_K=8 \
RESIDUAL_CANDIDATE_BLOCKS=4 \
RECORD_DENSITY=True \
START_IDX=0 END_IDX=0 \
bash hyvideo_t2v_720p_dfs.sh
```

等价的 inference 参数为：

```bash
python hyvideo_t2v_inference.py \
  --block_top_p 0.6 \
  --token_top_k 8 \
  --residual_candidate_blocks 4
```

## 4. Sparsity 记录

设置 `RECORD_DENSITY=True`（或 `--record_density true`）后，每个视频输出：

```text
density_records.csv
density_summary.csv
sparsity_records.csv
```

其中 `sparsity_records.csv` 包含：

```text
p_mass
token_top_k
residual_candidate_blocks
block_density
realized_block_sparsity
residual_token_interactions
final_density
final_hybrid_sparsity
```

统计按照实际保留的 QK interactions 计算，并排除了最后一个不完整 block 的
padding tokens。

## 5. 代码位置

- `dfsattn/attention_hyvideo.py`
  - Block Top-p mask 和 frontier mask 生成；
  - packed selected-block/residual-token attention；
  - unified softmax；
  - mask cache 和 sparsity 统计。
- `dfsattn/replace_hyvideo.py`
  - 将新增参数传递到 DFS attention。
- `hyvideo_t2v_inference.py`
  - 新增命令行参数和参数校验。
- `hyvideo_t2v_720p_dfs.sh`
  - 新增环境变量和 sweep 输出目录标识。

## 6. 已完成验证

已完成以下检查：

- Python 编译检查；
- shell 脚本语法检查；
- block frontier 选择检查，确认 frontier 与 block mask 不重叠；
- 非整除 block size 的数值对照；
- packed unified-softmax 与 reference attention 数值对照；
- mask/frontier cache 路径检查；
- 参数帮助信息和非法参数检查。

尚未完成完整 HyVideo GPU sweep；下一步应比较不同 `p_mass`、`token_top_k` 和
`R` 设置下的实际生成时间、最终 sparsity 以及 PSNR/SSIM/LPIPS。

## 7. OOM 修复记录

使用：

```bash
BLOCK_TOP_P=0.6 TOKEN_TOP_K=8 RECORD_DENSITY=True \
RESIDUAL_CANDIDATE_BLOCKS=4 bash hyvideo_t2v_720p_dfs.sh
```

时如果在第 12 个 diffusion step 之后发生 OOM，需要注意默认
`skip_steps=12`，第 12 步正好是首次进入 sparse/frontier attention 的位置。

此前 packed attention 在合并 block value 和 residual value 时，会把 selected
values 扩展为：

```text
[heads, query_tokens, selected_tokens, head_dim]
```

这个临时张量会随 selected token 数量额外放大显存。现已改为分别计算：

```text
softmax(selected_score) @ selected_v
softmax(residual_score) @ residual_v
```

两部分共享同一个 concatenated softmax normalization，但不再 materialize
上述 4D selected-value tensor。非整除 block size 的数值对照已通过。

已在可访问 GPU 环境中复现：目标 GPU 为约 95 GiB 显存，旧实现于第 12 步
首次进入 sparse/frontier path 后，在 `selected_score = q_chunk @ selected_k.T`
处分配 526 MiB 时 OOM；当时仅剩约 409 MiB 可用显存。修复后重新运行到第
13 步未再 OOM，说明该 4D 临时张量是此次 OOM 的直接原因。

修复后的第 12 步可以通过；在下面第 8 节的 native/batched 优化加入前，frontier
attention 仍然较慢。

## 8. 运行时间优化（2026-08-24）

针对第 12 步之后 frontier attention 逐 query-block 运行很慢的问题，已完成第一版
不改变实验定义的加速实现：

1. selected block 仍完全使用当前 DFSAttn 的 block mask 和 block Top-p/Top-k 选择，
   仅在 CUDA、`block_size=128` 且环境安装 `block_sparse_attn` 时改用原生
   `block_sparse_attn` kernel。没有改变 block selector。
2. 原生 block kernel 的输出使用其返回的 row-wise `logsumexp`，与 frontier 的
   exact token Top-k 结果通过 online softmax state 合并，避免重新构造全局
   `[heads, query, key]` score 矩阵。
3. frontier token Top-k 不再逐 block Python 循环，而是按 8 个 query blocks
   一批计算；block/token index 在 GPU 上向量化生成，仍只搜索未选中的最多
   `RESIDUAL_CANDIDATE_BLOCKS` 个 block，R 的实验语义不变。
4. 如果 `block_sparse_attn` 不可用，或使用非 128 block size，仍自动回退到原有
   packed 实现，便于 CPU/兼容性测试。
5. `hyvideo_t2v_720p_dfs.sh` 已加入 PyTorch `libc10` 的 `LD_LIBRARY_PATH`，并且
   现在真正把 `BLOCK_TOP_P` 和 block-mask 参数传给 inference；此前这两行被注释，
   仅设置环境变量不会生效。另可用 `SPARSE_EXECUTION=native|hybrid` 显式选择
   执行后端（默认 `native`）。

已完成的 CUDA 校验：

- 随机 `H=2, L=256, D=32` 的 native block + frontier + merge 与当前 fp16
  Top-k 的 dense reference 对照，最大绝对误差约 `5.96e-4`；
- 非整除序列长度 `L=300` 的 partial-last-block 对照通过，最大 block-path
  误差约 `5.1e-4`；
- 代表性 720p 序列规模 `H=24, L=44928, D=128` 的单层测试：native block
  约 `0.061s`、frontier 约 `0.123s`、state merge 约 `0.003s`，合计约
  `0.187s`（这是单层随机 mask 的 kernel benchmark，不等同于完整视频生成时间）。

当前 GPU 上已有一个用户启动的 HyVideo 进程，因而没有再启动竞争性的完整 720p
生成来覆盖它。完整实验应在该进程结束后用新的 shell 运行；若直接执行 Python，
需要先设置同样的 `LD_LIBRARY_PATH`，否则会自动走兼容的 packed fallback，无法
体现上述 native 加速。

## 9. 质量异常排查（2026-08-24）

对 baseline 和已生成的 Top-p/Top-k 结果检查后，发现需要区分两个问题：

1. 旧的 `topp0.6_tokenk8` 输出目录中，`sparsity_records.csv` 的 `p_mass` 实际为
   `nan`，说明当时 shell 环境变量虽然写进了目录名，但 `--block_top_p` 没有真正
   传给 inference；该视频不能作为 Top-p 结果。脚本已修复，新的运行必须检查
   `sparsity_records.csv` 中的 `p_mass`。
2. 在真正的 `p_mass=0.9, token_top_k=8` 记录中，residual token 只带来约
   `0.0002` 的绝对 density 增量（例如 `0.301346 -> 0.301524`），因此如果视频
   内容发生明显变化，主因应先看 Top-p block pattern，而不是 K=8 本身。

此外，批量 frontier QK 在 fp16/bf16 量化分数并列时可能因 GEMM 形状不同选出不同
的 token。当前代码已加入基于全局 token index 的稳定 tie-break，并通过多 query-block、
非恒等 permutation 和 bf16 数值测试；该修复只影响并列分数的选择，不改变 score
或 softmax 权重。

根据实验定义，Top-p 已修正为只在原 DFSAttn 的 video-block 候选范围内归一化和
选择；text blocks 仍保持 dense，cache、hierarchical score、permutation 和其余
attention 路径不变。后续质量实验应使用相同 prompt/seed 做四组消融：DFS
block-only、Top-p-only、DFS block + residual Top-k、Top-p + residual Top-k，并
确认记录中的 `p_mass` 和 `final_density` 后再比较视频质量。
