# HunyuanVideo Hybrid 实验运行命令

## VBench 11 prompts，720p

使用 `examples/vbench_11_prompts.txt` 中的 11 条 prompt，运行 Hybrid-DFSAttn：

```bash
conda activate SVG

cd /cnic/work/liutt/mywork/attention_time/code/DFSAttn

export CUDA_VISIBLE_DEVICES=0

python idea2/benchmark_hybrid_attention.py \
  --run-vbench \
  --backend hyvideo \
  --model-id /work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5 \
  --prompt-file /cnic/work/liutt/mywork/attention_time/code/DFSAttn/examples/vbench_11_prompts.txt \
  --output-dir /cnic/work/liutt/mywork/attention_time/res/dual \
  --height 720 \
  --width 1280 \
  --num-frames 129 \
  --num-inference-steps 50 \
  --seed 0 \
  --hybrid-threshold 8 \
  --skip-existing
```

Hybrid 执行参数由脚本自动传入：

```text
tile_size=16
block_size=16
sparse_execution=hybrid
hybrid_threshold=8
```

其中 `block_size=16` 是当前 Hybrid 原型的必要条件，因为它需要将
`4×4` 个 `16×16` fine blocks 聚合成一个 `64×64` Core tile。它与原始
`block_size=128` DFSAttn 的 Top-k mask 不是同一粒度。

## 输出文件

视频输出：

```text
/cnic/work/liutt/mywork/attention_time/res/dual/0.mp4
...
/cnic/work/liutt/mywork/attention_time/res/dual/10.mp4
```

Timing 和 mask 输出：

```text
/cnic/work/liutt/mywork/attention_time/res/dual/0.timing.csv
...
/cnic/work/liutt/mywork/attention_time/res/dual/10.timing.csv
/cnic/work/liutt/mywork/attention_time/res/dual/masks/
```

每个 prompt 的 `*.timing.csv` 现在只保存整个视频的全局汇总，不再记录
step/layer 明细。字段为 `phase,total_ms`，其中包括：

```text
top_k_selection       # 所有实际 Top-k mask 计算的总时间
attention_execution   # 所有 attention execution 的总时间
e2e_generation_wall   # pipe(...) 生成阶段的端到端 wall-clock 时间
e2e_generation_gpu    # 生成阶段 CUDA stream 时间（可用时）
```

模型加载和视频编码不计入 `e2e_generation_*`。

`--skip-existing` 会跳过已经生成且文件大小大于 0 的视频。首次调试时可以只运行第一个 prompt：

```bash
python idea2/benchmark_hybrid_attention.py \
  --run-vbench \
  --backend hyvideo \
  --model-id /work/liutt/cache/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5 \
  --prompt-file /cnic/work/liutt/mywork/attention_time/code/DFSAttn/examples/vbench_11_prompts.txt \
  --output-dir /cnic/work/liutt/mywork/attention_time/res/dual \
  --height 480 --width 720 --num-frames 129 \
  --num-inference-steps 50 --seed 0 \
  --hybrid-threshold 8 \
  --start-idx 0 --end-idx 0
```

## 原始 DFSAttn baseline：hyvideo_t2v_720p_dfs.sh

脚本已直接设置为 HunyuanVideo 720p、VBench 11 prompts 和指定模型路径。
它运行原始 native DFSAttn，`block_size=128`，用于和 Hybrid 结果比较。
每个 prompt 的结果默认保存在：

```text
/cnic/work/liutt/mywork/attention_time/res/dfs/vbench_11/HunyuanVideo/16_128_0.3_0_hilbert3d_720/
```

目录字段依次为：

```text
数据集名/模型名/tile_size_block_size_sparsity_seed_order_height
```

直接运行：

```bash
conda activate SVG
cd /cnic/work/liutt/mywork/attention_time/code/DFSAttn
bash hyvideo_t2v_720p_dfs.sh
```

如需避免重复生成，可在脚本中保留默认的已有文件跳过逻辑；也可以通过
`START_IDX` 和 `END_IDX` 环境变量限制 prompt 范围，例如：

```bash
START_IDX=0 END_IDX=0 bash hyvideo_t2v_720p_dfs.sh
```

## Phase A：Block Top-p + exact Token Top-k oracle

`BLOCK_TOP_P` 启用 phase-A 的 block Top-p，`TOKEN_TOP_K` 从未被 block
保留的 token 中使用 exact QK 选择 residual Top-k。`TOKEN_TOP_K=0` 就是
Block Top-p only。该 oracle 路径每次只保存一个 query block 的 exact score，
不会构造全局 `N×N` attention mask；它用于质量实验，不是性能实现。

例如先跑一个 prompt 的 `p=0.6, k=8`：

```bash
cd /cnic/work/liutt/mywork/attention_time/code/DFSAttn

BLOCK_TOP_P=0.6 TOKEN_TOP_K=8 RECORD_DENSITY=True \
START_IDX=0 END_IDX=0 \
bash hyvideo_t2v_720p_dfs.sh
```

参数含义：

```text
--block_top_p P   # 0 < P <= 1；省略时保持原 DFS block Top-k
--token_top_k K   # 0, 4, 8, 16, 32；0 表示没有 residual token
--residual_candidate_blocks R  # 每个 query block 搜索的未选 frontier block 数，默认 4
```

Top-p 仍保持 DFSAttn 的文本语义：文本 key blocks 始终保留，文本 query
blocks 为全注意力。`RECORD_DENSITY=True` 会在每个视频目录中额外输出
`density_summary.csv` 和 `sparsity_records.csv`。其中
`density_summary.csv` 按 prompt 追加一行，保存 prompt index/text 及其稀疏
step/layer 范围内的平均 density；
`sparsity_records.csv` 包含 `p_mass`、`realized_block_sparsity`、
`residual_token_interactions` 和 `final_hybrid_sparsity`。

当前初步验证默认使用 `R=4`。Residual Top-k 只在这 4 个未选且
hierarchical score 最高的 frontier blocks 内进行；因此它是方案一的
bounded-frontier 近似，不再扫描全部未选 token。
