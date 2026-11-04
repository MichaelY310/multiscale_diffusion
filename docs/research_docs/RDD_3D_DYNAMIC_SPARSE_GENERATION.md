# RDD-3D：由生成过程自身产生 sparsity 的动态稀疏 SLat 方案

> 状态：2026-08-11 当前主方案。本文取代“先由 TRELLIS Structure Model 固定 fine support、再在 support 上做 RDD geometry flow”的旧路线。TRELLIS.2 在本方案中只提供 learned SLat representation、encoder/decoder 与可选 backbone 初始化，不提供生成时的 support。

## 1. 结论先行

核心生成过程是：

```text
完整 8^3 noise
  -> RDD 去噪并删除确定为空的 coarse blocks
  -> 每个保留 parent 展开完整 2x2x2 children
  -> 16^3 RDD 去噪并再次删除 empty children
  -> 32^3（以后可扩到 64^3）
  -> 只把最终 non-empty coords + TRELLIS 32D SLat 交给原版 decoder
```

没有独立 Structure Model。sparse topology 与 geometry feature 由**一个共享 RDD denoiser**共同生成。

当前 32 个真实 TRELLIS 512/32^3 SLat 的 oracle 统计已经验证：8/12/20 NFE 下，动态 candidate token-NFE 相比逐级 dense RDD 平均降低 **10.99x**，相比 40 steps 全程 dense 32^3 平均降低 **20.33x**；精确分解误差为 `1.91e-6`。这证明表示和潜在计算收益成立，但最终方法成败取决于 predicted pruning 的 coarse non-empty recall。

首个 2K-step、3.36M 参数 overfit pilot **没有通过自由生成 gate**：raw/EMA final support recall 分别为 50.9%/44.8%，precision 为 12.5%/10.5%。这不是表示反例，而是暴露了 teacher-forced parent 与 generated parent 的分布缺口；本文件明确记录失败，不把 oracle 加速当成完整方法结果。

## 2. 系统里有几个模型

Image-conditioned 版本只有三个网络模块：

| 模块 | 是否训练 | 输入 | 输出 |
|---|---|---|---|
| DINO image encoder | 冻结 | RGB render | 384D（smoke）image condition |
| Dynamic Sparse RDD Generator | **训练** | candidate coords、noisy mass-SLat、time、image condition | 33D velocity + non-empty logit |
| TRELLIS SC-VAE decoder | 冻结 | final sparse coords + 32D SLat | mesh |

不存在 TRELLIS Structure Model。CFG、EMA 和 auxiliary occupancy head 都不是额外生成模型。

## 3. Empty voxel representation

### 3.1 为什么不能只判断“32D feature 是否刚好等于零”

Diffusion/FM 是连续生成。即使规定 empty latent 为零，预测结果也几乎不会精确等于零；简单使用 L2 distance threshold 还会与真实 SLat 中接近零的 feature 混淆。coarse-stage 一个 false negative 会删除很大的 fine region，因此判空必须有显式 margin 和可校准置信度。

### 3.2 选定表示：occupancy mass + feature mass

对 fine voxel 定义 33D 内部表示：

```text
active voxel: h = [rho=1, q=z_slat]
empty voxel:  h = [rho=0, q=0]
```

其中 `rho` 是 occupancy mass，`q=rho*z` 是 feature mass。每次降分辨率都执行普通、固定除以 8 的 average：

```text
h_parent = (h_child_1 + ... + h_child_8) / 8
```

于是：

- 八个 children 全空时，parent **严格为 `[0,0]`**；
- 部分占用时，`rho_parent` 等于 occupied child fraction；
- `q_parent/rho_parent` 是 active children 的平均 SLat；
- 最终 active fine voxel 的 `rho=1`，因此 `q` 与 `q/rho` 都精确恢复原始 TRELLIS SLat；
- parent/detail decomposition 使用普通 average 和 nearest upsample，和 2D RDD 完全一致。

这比 learned empty embedding 更容易校准，也避免 learned embedding 在训练中漂移。若直接令 empty 为任意 learned vector `e`，all-empty average 虽仍是 `e`，但零吸收元、阈值和多级数值尺度都会随训练漂移；它本质上仍需先减去 `e` 才回到这里的 centered mass 表示。empty 的零向量是 RDD 内部约定，不会送进 TRELLIS decoder。

推理最终层直接把 `q` 作为 normalized SLat，而不默认除以预测 `rho`。原因是 active fine target 的 `rho` 严格等于 1；若一个 false-positive token 的生成质量接近 0，`q/rho` 会把很小的误差放大成任意大的 decoder feature。`q/rho` 只用于分析 coarse active-child semantic mean，并设置显式 denominator floor。

### 3.3 同一模型内的 auxiliary non-empty head

RDD denoiser除 33D velocity 外再输出一个标量 logit：

```text
P(block contains any non-empty descendant)
```

它只是共享 hidden feature 上的一个 linear head，不是 structure network。训练使用 weighted BCE，positive weight 默认 8，使早期模型偏向 false positive：多保留只增加计算，false negative 会永久删除几何。

推理时首版保留规则为：

```text
keep = (nonempty_probability >= calibrated_threshold)
       OR (generated_mass >= conservative_mass_threshold)
```

概率阈值不固定为 0.5，而是在 validation/pilot 上选择“满足目标 recall 的最大阈值”。当前目标为 8^3/16^3/32^3 = 99.9%/99.5%/99.0%。

正式规则是**保守删除而不是积极保留分类**：只有 occupancy head 与 mass 两者都以高置信度认为 empty 时才删除；不确定块保留。必须同时报告 support recall 和 candidate inflation，不能通过激进剪枝制造加速。可选的一格 parent halo 只作为 recall ablation，不进入默认结果。

## 4. 不构造 dense 32^3/64^3 的训练候选树

定义上 empty voxel 存在于完整 grid，但实现中只 materialize 会被 RDD 访问的 candidates：

1. 8^3 stage：完整 512 tokens；
2. teacher-forced 训练的 16^3 candidates：GT non-empty 8^3 parents 的全部八个 children；
3. 32^3 candidates：GT non-empty 16^3 parents 的全部八个 children；
4. 未来 64^3 candidates：GT non-empty 32^3 parents 的全部八个 children。

候选集合包含真实 non-empty children，以及需要模型学会删除的 hard-negative empty children。GT parent 为 empty 的整块区域不进入下一阶段。推理时只把 `GT non-empty parents` 换成 `predicted retained parents`，因此 sparsity 是 RDD 自身的输出。

v1 的关键漏洞是：refinement 训练只展开 GT-non-empty parents，但推理会展开上一级误保留的 empty parent。后一级从未见过这种全空分支。v2 每步另外采样默认 `1.0 x` 的 GT-empty parents，并把每个 parent 的完整八个 zero children 加入候选；这不会改变平均不变量，却显式训练后续 stage 清理 false-positive branch。false-negative parent 仍不可恢复，所以 coarse 阈值必须以 recall 为第一目标。

## 5. 训练目标

每个 asset 提供：

```text
render image
fine active coords [N,3]
fine TRELLIS SLat [N,32]
```

SLat 先按 pilot 数据集统计的 channel mean/std 归一化，再转成 33D mass representation。一个共享模型以均匀概率随机训练 8^3、16^3 或 32^3 stage。

### 5.1 Coarse stage

完整 8^3 target 中既有 non-empty coarse representations，也有 `[0,0]` empty targets。对它做普通 linear flow matching：clean representation 与完整 Gaussian noise 插值，模型输出 global-time velocity 和 non-empty logit。

### 5.2 Refinement stage

对每个 retained parent 展开完整八个 children。clean child target 写成：

```text
child = nearest(parent) + detail
```

因为 parent 是八个 children（含 empty）的普通平均，所以八个 detail 的平均严格为零。noise filling 也先减去 sibling mean；模型 velocity 在每一步再次投影为 zero-sibling-mean。这样 refinement 不改变已生成 parent 的平均值。

### 5.3 时间和 loss

| Stage | Global time | 训练抽样概率 | 推理 NFE |
|---|---:|---:|---:|
| 8^3 | 1.0 -> 0.8 | 1/3 | 8 |
| 16^3 | 0.8 -> 0.5 | 1/3 | 12 |
| 32^3 | 0.5 -> 0.0 | 1/3 | 20 |

velocity 除以 time-band width 变成 global-time derivative；MSE 再乘 width squared，消除单位变换造成的梯度放大。33D 中 mass velocity 默认权重 4，non-empty BCE 权重 0.25、positive weight 8。

v2 另以 20% 概率令 local time 精确等于 clean endpoint，保证 occupancy head 直接见过实际执行剪枝的位置。短 overfit gate 使用 raw weights；EMA 仍保存，但 2K pilot 中 raw 只略优于 EMA，不能解释主要失败。

## 6. 推理算法

1. 在完整 8^3 坐标上采样 512 个 33D Gaussian noise；
2. 同一 denoiser运行 8 steps；
3. 用 calibrated probability + generated mass 做 recall-first pruning；
4. 每个保留 parent 生成完整八个 child candidates；
5. child 初态为 nearest(parent) + zero-mean detail noise；
6. 16^3 运行 12 steps并再次剪枝；
7. 32^3 运行 20 steps并最终剪枝；
8. 对保留的 fine token 取 `z_normalized=q`；
9. 用 checkpoint 中保存的 mean/std 反归一化；
10. 把 `(coords,z_slat)` 交给未修改的 TRELLIS decoder。

当前第一版只在 stage endpoint 剪枝，不在 stage 内动态改变 token set，避免数值积分过程中 topology 跳变。

## 7. 最小模型结构

第一项实验故意不用 1.3B backbone，先验证“empty prediction + generated pruning”是否成立：

```text
Input per candidate:
  33D noisy mass-SLat
  logical xyz coordinate
  global time
  frozen DINOv2-S/14 384D CLS condition

State projection + Fourier xyz embedding
  -> 6 x SparseVoxelMixerBlock(width=192)
       - self projection
       - hashed 3x3x3 neighbor mean（只访问候选 token）
       - global token mean
       - time/image conditioning
       - 4x MLP
  -> LayerNorm
  -> 33D velocity head
  -> 1D non-empty head
```

`width=192, depth=6` 的实际参数量为 **3,356,578**。它不是论文最终 backbone，只是检查表示、branch cleanup 与剪枝闭环的廉价 gate。

邻域查找使用 integer coordinate hash/search，复杂度是 `O(27N)`；没有 dense 3D tensor，也没有 `O(N^2)` global attention。这个模型只用于 idea gate。若 gate 通过，再把相同输入/输出和 pruning protocol 移植到较大的 TRELLIS-style sparse transformer。

## 8. 已完成的真实 oracle 统计

数据：32 个相同 ObjaverseXL pilot assets，TRELLIS 512 output / 32^3 SLat。

| Stage | Mean candidates | Mean non-empty | Candidate empty fraction | Dense/candidate |
|---|---:|---:|---:|---:|
| 8^3 | 512.0 | 105.4 | 79.4% | 1.0x |
| 16^3 | 843.3 | 445.6 | 48.3% | 6.27x |
| 32^3 | 3564.8 | 1935.6 | 46.1% | 12.84x |

8/12/20 NFE 下：

- dynamic candidate token-NFE mean：`85,510`；
- 相对逐级 dense 8/16/32 RDD：mean `10.99x` reduction；
- 相对 40 steps dense 32^3：mean `20.33x` reduction；
- exact parent+detail reconstruction max error：`1.91e-6`；
- detail sibling-mean max error：`1.91e-6`。

这些是 oracle mask 计算量，不能替代 predicted pruning 结果。预测 false positives 会使 candidate 数上升，false negatives 会使形状缺失。

## 9. 已完成的 v1 自由生成结果（失败也保留）

32 个训练内 assets、2K steps；在相同四个训练 asset 上自由生成：

| Weights | Support precision | Support recall | Support IoU | 结论 |
|---|---:|---:|---:|---|
| EMA | 10.5% | 44.8% | 9.0% | fail |
| Raw | 12.5% | 50.9% | 10.7% | fail |

raw 生成的 8^3 平均保留约 402/512，而 oracle non-empty 只有 105/512；错误分支随后扩为数千 fine candidates。teacher-forced clean endpoint calibration 同时能得到很高 recall，说明该 calibration 严重高估了 generated-state 性能。旧版 `q/rho` 还造成 intersection feature MSE 达千万量级；这项数值病态已经修复为 fine-level 直接输出 `q`。

因此 v1 只通过：表示 exactness、有限 loss/gradient、teacher-forced classification。它没有证明可用的生成或剪枝。

## 10. 当前最小可决策实验配置（v2）

```text
Data: 32 ObjaverseXL pilot assets（训练内 overfit gate）
Output/SLat: 512 / 32^3
Stages: 8^3 -> 16^3 -> 32^3
Condition: frozen DINOv2 ViT-S/14 CLS, 384D
Generator: width 192, depth 6, 3.36M params, local 3x3x3 sparse mixer
State: 1 occupancy mass + 32 feature-mass channels
Optimizer: AdamW
LR: 2e-4
Weight decay: 0.01
Steps: 10,000
EMA: 0.999
Precision: BF16
Batch: one variable-length shape/step
Stage probability: 1/3 each
Occupancy BCE positive weight: 8
Extra empty-parent groups: 1.0x active-parent count
Exact clean-endpoint sampling: 20%
Gradient clip: 1.0
Sampling NFE: 8/12/20
Evaluation weights: raw（同时保留 EMA 对照）
Hardware: one GPU（当前 GPU 4）
```

这是最小的“继续/停止”实验，不是泛化或论文质量实验。若训练内 32 assets 都达不到 coarse/final recall gate，扩大到大模型和更多数据没有意义。

## 11. 当前服务器直接运行

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
RDD3D_DYNAMIC_GPU=4 \
RDD3D_DYNAMIC_RUN_NAME=dynamic_sparse_32asset_gate_v2 \
RDD3D_DYNAMIC_MAX_ASSETS=32 \
RDD3D_DYNAMIC_TRAIN_STEPS=10000 \
RDD3D_DYNAMIC_WEIGHTS=model \
bash scripts/rdd3d/run_dynamic_sparse_pilot_tmux.sh
```

两卡 25.62M 参数的推荐 gate 已固定为：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash scripts/rdd3d/run_dynamic_sparse_gate_2gpu.sh
tail -f logs/rdd3d_dynamic_sparse_gate_2gpu.log
```

它使用 physical GPU `0,1`、`torchrun` 两卡 DDP、`width=384`、`depth=12`、
`25,622,050` 参数和 10K optimizer steps。每个 rank 处理一个完整变长 shape，
global shapes/step 为 2；不会沿 token 维拆分一个 sparse shape。

监控：

```bash
tmux attach -t rdd3d_dynamic_sparse_pilot
tail -f /scr/yuheng/DevOpsBench/multiscale_diffusion/logs/rdd3d_dynamic_sparse_pilot.log
```

主要输出：

```text
results_rdd3d/dynamic_sparse_32asset_gate_v2/checkpoints/step0010000.pt
results_rdd3d/dynamic_sparse_32asset_gate_v2/samples_dynamic/*.npz
eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_gate_v2_oracle.json
eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_gate_v2_thresholds.json
eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_gate_v2_generated_support.json
```

## 12. 新服务器需要下载什么

### 只跑最小训练 gate

必须：

1. 32--128 个预计算 TRELLIS SLat `.npz`，每个包含 `coords [N,3]` 和 `feats [N,32]`；
2. 每个 asset 至少一张 render；
3. DINOv2 ViT-S/14 权重（脚本首次运行通过 `torch.hub` 自动下载）；
4. PyTorch、NumPy、Pillow。

只训练/测 support 时不需要 TRELLIS 4B generator，也不需要 Structure Model。

从新服务器准备 512/32^3 pilot 的顺序是：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion

# 固定 TRELLIS.2/FlexGEMM/CuMesh commit，建立 .venv_rdd3d。
bash scripts/rdd3d/setup_trellis2_pilot.sh

# 下载一个很小的 ObjaverseXL shard，mesh -> O-Voxel -> 32D SLat，
# 并为 image-conditioned overfit gate 渲染条件图。
RDD3D_OVOXEL_RESOLUTION=512 \
RDD3D_DOWNLOAD_WORLD_SIZE=8000 \
RDD3D_DOWNLOAD_RANK=0 \
RDD3D_RENDER_COND=1 \
RDD3D_GPU=4 \
bash scripts/rdd3d/prepare_trellis2_pilot_data.sh

# 启动 32-asset v2 gate。
RDD3D_DYNAMIC_GPU=4 \
bash scripts/rdd3d/run_dynamic_sparse_pilot_tmux.sh
```

`WORLD_SIZE=8000,RANK=0` 的目的只是取得一个约几十个 asset 的确定性小 shard，不表示分布式训练 world size。实际成功数受 mesh 下载/预处理成功率影响；训练脚本只取排序后的前 32 个。

模型权重分成三类：

1. `facebookresearch/dinov2: dinov2_vits14`，仅产生 384D 条件，`torch.hub` 首次运行自动下载；
2. `microsoft/TRELLIS.2-4B` 中官方 shape SLat encoder/SC-VAE decoder，数据编码或 mesh decode 时由 TRELLIS `models.from_pretrained` 自动下载；
3. Dynamic Sparse RDD Generator **从随机初始化训练**，最小 gate 不下载 TRELLIS 1.3B/4B flow checkpoint。

因此如果别人已经给出上述 `.npz + render`，只需第 1 类权重；若从 raw mesh 开始，才需要完整 TRELLIS 环境与第 2 类权重。

### 从 raw mesh 制作 SLat并解码 mesh

额外需要：

1. pinned TRELLIS.2 commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`；
2. TRELLIS O-Voxel/SC-VAE encoder 与 decoder 权重；
3. ObjaverseXL/Objaverse mesh 或 GLB；
4. TRELLIS.2 的 Blender/O-Voxel preprocessing 环境。

当前 surfi2 已有 32 个 512/32^3 和 1024/64^3 SLat、renders、TRELLIS decoder 与 DINOv2 cache，因此当前 pilot 不需要再下载数据。

## 13. 通过/停止门槛

10K overfit gate：

1. 所有 stage loss/gradient finite；
2. teacher-forced threshold calibration 达到设定 recall；
3. **真实 generated-state pruning** 的 8^3 recall >= 99%、16^3 recall >= 98%、final support recall >= 95%；
4. predicted candidates 不超过 oracle candidates 的 2x；
5. 至少 4/4 sparse outputs 能被 TRELLIS decoder 解码，且没有空 mesh/NaN。

若通过，再做 128 train / 32 held-out、20K step；若 8^3 generated recall 仍低，依次测试按 stage 单独 occupancy head、generated-parent curriculum，以及一格 conservative halo。若仍无法达到高 recall，应停止，而不是扩大模型掩盖 pruning 错误。

## 14. 代码地图

```text
RDD_DiT_3D/dynamic_sparse_representation.py
RDD_DiT_3D/dynamic_sparse_model.py
RDD_DiT_3D/dynamic_sparse_flow.py
RDD_DiT_3D/train_dynamic_sparse_rdd.py
RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py
RDD_DiT_3D/calibrate_dynamic_sparse_pruning_replay.py
RDD_DiT_3D/sample_dynamic_sparse_dataset.py
RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py
RDD_DiT_3D/evaluate_dynamic_sparse_decode.py
RDD_DiT_3D/summarize_dynamic_sparse_formal.py
RDD_DiT_3D/decode_dynamic_sparse_slat.py
scripts/rdd3d/build_dynamic_sparse_formal_split.py
scripts/rdd3d/run_dynamic_sparse_pilot{,_tmux}.sh
scripts/rdd3d/run_dynamic_sparse_gate_2gpu.sh
scripts/rdd3d/compare_dynamic_sparse_raw_ema.sh
scripts/rdd3d/run_dynamic_sparse_formal_heldout_{2gpu,tmux}.sh
tests/test_dynamic_sparse_rdd.py
```

## 15. 25.62M / 10K gate 的真实结果与诊断（2026-08-11）

当前有效 checkpoint：

```text
results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/checkpoints/step0010000.pt
```

四个训练内 asset 的自由生成（raw/model weights，Euler 8/12/20）：

| Variant | Precision | Recall | IoU | 8/16/32 stage recall | matched-dense reduction |
|---|---:|---:|---:|---|---:|
| baseline | 0.550 | 0.541 | **0.375** | 0.913 / 0.861 / 0.541 | **6.18x** |
| Euler 32/24/40 | 0.525 | 0.518 | 0.354 | 0.864 / 0.806 / 0.518 | 6.42x |
| Heun 8/12/20 | 0.415 | 0.595 | 0.324 | 0.879 / 0.852 / 0.595 | 3.65x |
| 8^3 halo radius 1 | 0.375 | 0.659 | 0.311 | 1.000 / 0.960 / 0.659 | 2.17x |
| keep all 8^3, Euler 32/24/40 | 0.319 | 0.667 | 0.272 | 1.000 / 0.964 / 0.667 | 1.55x |
| keep all 8^3, Heun 16/16/24 | 0.320 | 0.658 | 0.272 | 1.000 / 0.932 / 0.658 | 1.76x |

重要结论：

1. v2 相比 v1 的 IoU `0.107 -> 0.375`，hard-negative parent 与稳定 fine feature recovery 是实质改进；
2. 增加 Euler NFE 不改善结果，Heun 只用更多 token-NFE 换 recall，故当前瓶颈不是积分精度；
3. conservative halo 有效提高 recall，但 precision 与 sparse speed 同时下降；
4. 即使完全取消 8^3 剪枝，32^3 recall 仍只有约 `0.67`，错误不能归因于第一次剪枝；
5. teacher-forced clean endpoint 在 8/16/32 可达约 `0.999/0.995/0.990` recall，但自由生成远低于它，主瓶颈是 refinement 从模型生成 parent 出发时的 distribution shift；
6. RDD 的零均值 sibling detail 会有意保持 coarse parent mean，因此 coarse representation error 不能由后续 stage 修复。这是精确 average hierarchy 的性质，不应靠增加 solver NFE 掩盖。

因此当前 gate 仍未通过论文级动态剪枝门槛，但已经证明：在 support 不够准确时仍可得到约 `6.2x` matched-dense token-NFE reduction。下一步应优先解决 generated-state refinement，而不是扩大 backbone 或扫更多 solver。

## 16. Generated-parent replay 微调

当前实现保存每个 stage 实际 retained parent 的：

```text
coords_8/16/32
representation_8/16/32
```

微调 refinement 时，以一定概率直接展开模型生成的 parent，而不是 GT parent。监督 support 仍来自普通固定 2x2x2 average 构造的 ground truth pyramid。对于 generated parent 与 GT child mean 的偏差，detail target 投影到 sibling 零均值子空间；这保持 RDD 不变量，并诚实承认 refinement 不能修复 coarse mean error。occupancy head 则可学习在真实 inference candidate/state 分布下删除错误 branch、保留正确 child。

本轮配置：

```text
resume: 10K raw/model checkpoint
additional steps: 5K
GPUs: 0,1 (2-rank DDP)
LR: 5e-5, reset optimizer
stage probabilities coarse/middle/fine: 0.15/0.30/0.55
generated-parent probability: 0.60 on refinement stages
generated clean-endpoint probability: 0.70
generated velocity loss weight: 0.10
occupancy loss weight: 1.0
positive weight: 3.0
evaluation: every 1K continuation checkpoint, Euler 8/12/20, four assets
```

运行与监控：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash scripts/rdd3d/start_dynamic_sparse_replay_finetune_tmux.sh
tail -f logs/rdd3d_replay_finetune_2gpu.log
tmux attach -t rdd3d_replay_finetune
```

## 17. 第一轮 replay 微调结果

相同四个训练内 asset、相同采样 NFE、不同于 replay 的 seed `20260811`：

| Checkpoint | Precision | Recall | IoU | 8/16/32 recall | reduction |
|---|---:|---:|---:|---|---:|
| original 10K | 0.575 | 0.496 | 0.362 | 0.905 / 0.811 / 0.496 | 约 6x |
| replay 11K | 0.506 | 0.755 | 0.434 | 0.955 / 0.929 / 0.755 | 5.31x |
| replay 12K | 0.484 | 0.775 | 0.425 | 0.927 / 0.913 / 0.775 | 5.81x |
| replay 13K | 0.461 | **0.848** | 0.428 | 0.953 / 0.946 / **0.848** | 5.46x |
| replay 14K | **0.514** | 0.769 | 0.447 | **0.965** / 0.948 / 0.769 | 5.50x |
| replay 15K | 0.505 | 0.811 | **0.453** | 0.954 / 0.914 / 0.811 | **5.97x** |

严格同 seed 对照中，10K -> 15K 的 final recall 增加 `+0.315`，IoU 增加 `+0.091`（约 `25%` relative）。默认 replay seed 与独立 seed 的结论一致，故提升不是只记住一条噪声轨迹。

15K generated-state pruning 消融：

| Rule | Precision | Recall | IoU | reduction |
|---|---:|---:|---:|---:|
| teacher-calibrated probability OR mass | 0.505 | 0.811 | 0.453 | 5.97x |
| probability-only, relaxed | 0.531 | 0.771 | **0.460** | **6.44x** |
| lower fine threshold | 0.464 | **0.867** | 0.435 | 5.97x |
| mass-only | **0.553** | 0.614 | 0.411 | **7.50x** |
| 16^3 halo | 0.501 | 0.785 | 0.442 | 5.05x |
| keep all 16^3 candidates | 0.513 | 0.759 | 0.442 | 4.84x |

16^3 halo/全保留把该 stage recall 提到 `0.977`，final recall 却下降，说明错误 branch 会改变 candidate/global context；fine generator 对 candidate-set shift 仍不稳健。阈值只能提供 recall/precision/speed operating points，不能代替 generated-state 训练。

## 18. TRELLIS decoder paired mesh 验证

预测与 GT SLat 都通过同一个官方 `shape_dec_next_dc_f16c32_fp16` 解码；GT-SLat decode 作为 paired reference。四资产、每个 mesh 采样 50K surface points：

| Model | Chamfer L2 squared | P@0.01 | R@0.01 | F-score@0.01 | decode failures |
|---|---:|---:|---:|---:|---:|
| original 10K | 0.001993 | **0.296** | 0.204 | 0.242 | 0/4 |
| replay 15K, probability-only relaxed | **0.000886** | 0.281 | **0.320** | **0.297** | 0/4 |

replay 微调使 Chamfer 下降约 `55.5%`、surface recall 提高约 `56.7%`、F-score 提高约 `23.0%`。因此 support improvement 已传递到实际 decoded geometry，而非只改善 occupancy 表格。绝对 F-score 仍低，当前只是 32-asset overfit gate，不应写成论文级泛化结果。

服务器 Blackwell 上原始 FlexGEMM/Triton 3.x 会在 TRELLIS mixed fp32 activation / fp16 weight 路径报错。实验环境对以下 forward kernels 在 `tl.dot` 前加入了 activation-to-weight dtype cast，并保留 fp32 accumulator：

```text
scripts/rdd3d/flex_gemm_patch/sparse_submanifold_conv_fwd_implicit_gemm.py
scripts/rdd3d/flex_gemm_patch/sparse_submanifold_conv_fwd_masked_implicit_gemm.py
scripts/rdd3d/flex_gemm_patch/sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.py
```

虚拟环境原文件以 `.pre_rdd3d_dtype_fix` 结尾备份。

## 19. 第二轮 mixed on-policy replay

```text
resume: replay 15K
additional steps: 5K (to 20K)
replays: original 10K trajectory + 15K default trajectory + 15K probability-only trajectory
new replay seeds: 424242 / 777777
LR: 2e-5
stage probabilities: 0.25 / 0.30 / 0.45
generated-parent probability: 0.75
generated endpoint probability: 0.50
generated velocity weight: 0.75 (first round was 0.10)
occupancy weight: 0.75
evaluation seed: 314159 (not used by replay)
```

同 seed `314159` 的严格 checkpoint 结果：

| Checkpoint | Precision | Recall | IoU | 8/16/32 recall | reduction |
|---|---:|---:|---:|---|---:|
| original 10K | **0.594** | 0.503 | 0.372 | 0.915 / 0.831 / 0.503 | 约 6x |
| replay 15K | 0.514 | 0.810 | 0.460 | 0.944 / 0.921 / 0.810 | 约 5.8x |
| on-policy 16K | 0.519 | 0.871 | 0.483 | 0.954 / 0.956 / 0.871 | 5.69x |
| on-policy 18K | 0.538 | 0.868 | 0.499 | 0.958 / 0.958 / 0.868 | 5.73x |
| on-policy 19K | **0.572** | 0.843 | **0.518** | 0.919 / 0.943 / 0.843 | **5.87x** |
| on-policy 20K | 0.534 | **0.883** | 0.499 | 0.937 / 0.954 / **0.883** | 5.76x |

`19K` 是 balanced 主 checkpoint；`20K` 是默认 pruning 下的 recall-first checkpoint。相对原始 10K，19K 的 IoU 从 `0.372 -> 0.518`，约 `39%` relative improvement。

paired TRELLIS mesh decode：

| Checkpoint | Chamfer L2 squared | P@0.01 | R@0.01 | F-score@0.01 |
|---|---:|---:|---:|---:|
| original 10K | 0.001993 | 0.296 | 0.204 | 0.242 |
| replay 15K | 0.000886 | 0.281 | 0.320 | 0.297 |
| on-policy 19K | **0.000512** | **0.320** | 0.358 | **0.337** |
| on-policy 20K | 0.000593 | 0.279 | **0.366** | 0.314 |

相对 original 10K，19K Chamfer 下降约 `74.3%`、F-score 提高约 `39.5%`、surface recall 提高约 `75%`。20K 的额外 support/surface recall 伴随 precision 下降，所以不作为 balanced 主结果。

最后的 conservative pruning operating points：

| Checkpoint/rule | Precision | Recall | IoU | reduction |
|---|---:|---:|---:|---:|
| 19K default | **0.572** | 0.843 | **0.518** | 5.87x |
| 19K halo at 8^3 | 0.547 | 0.876 | 0.508 | **6.10x** |
| 19K force-keep 8^3 + relaxed refine | 0.430 | **0.954** | 0.421 | 4.76x |
| 20K force-keep 8^3 + relaxed refine | 0.434 | 0.951 | 0.424 | 4.84x |

因此 `>=0.95` final support recall 已可达到且仍保留约 `4.8x` matched-dense token-NFE reduction，但 precision 代价明显。阈值/halo 消融到此停止，下一瓶颈是更多数据上的 generated-state generalization。

运行与监控：

```bash
tail -f logs/rdd3d_onpolicy_finetune_2gpu.log
tmux attach -t rdd3d_onpolicy_finetune
```

## 20. 正式 held-out gate 数据（准备中）

当前 32 个 asset 全部参与了训练，只能证明 overfit feasibility。下一 gate 使用独立新数据根目录：

```text
candidate metadata: 256 GitHub ObjaverseXL objects
target successful SLats: >=128
resolution: 512 O-Voxel / 32^3 SLat
conditioning views: 1
planned split: 96 train / 32 held-out
GPU preprocessing: at most physical GPUs 0,1
```

监控：

```bash
tail -f logs/rdd3d_data_train_512.log
tmux attach -t rdd3d_data_train_512
```

本次实际 preprocessing 已得到 `239/239` mesh dump、`239/239` dual grid 和 `239/239` SLat。原 TRELLIS `render_cond.py` wrapper 一度没有设置 `CUDA_VISIBLE_DEVICES`，导致 Blender CYCLES 注册全部 8 张卡；在约 67 个 render 后已终止该子任务，2/3 卡上的训练进程未被终止，现有产物保留。永久修复将 render visibility 绑定到 `RDD3D_RENDER_GPUS`（默认继承 `RDD3D_GPU`）；当前剩余 render 通过 `rdd3d_render_resume_2gpu` 仅在 GPU0/1 上恢复，且自动跳过已有 `transforms.json`。这属于实验资源约束修复，不改变数据内容。

### 20.1 严格的数据隔离

正式 gate 不再使用“排序后前 96 个训练、后 32 个评测”的隐式约定，而由固定 seed `98317` 建立可审计 manifest。资产只有同时满足以下条件才有资格进入 split：

1. `32^3` TRELLIS SLat `.npz` 存在；
2. conditioning render 的 `transforms.json`、首帧图像和非空 alpha mask 均有效；
3. 不属于早期 32-asset pilot latent 目录。

选中的 96/32 资产分别通过只读软链接写入 `train_latents` 与 `heldout_latents`，manifest 显式列出 train、held-out、unused、rejected 与 rejection reason。held-out 资产不用于 feature normalization、训练、generated replay 或 pruning threshold calibration。

### 20.2 三段因果对照

同一组新 held-out 资产上依次比较：

| Method | 训练资产 | generated-parent training | 回答的问题 |
|---|---|---|---|
| 19K zero-shot | 早期 32 pilot | 是 | 原模型是否跨资产迁移 |
| teacher adaptation | 新 96 train | 否 | 更多数据/普通适配本身带来多少提升 |
| on-policy replay | 新 96 train | 是 | generated-state training 是否有独立收益 |

teacher adaptation 从 balanced 19K raw/model weights 初始化，reset optimizer，训练 5K steps；on-policy 阶段从 teacher checkpoint 出发，以两条不同训练 seed 的模型生成轨迹再训练 5K steps。两阶段均使用 2-rank DDP，且只暴露 physical GPU 0、1。

每个 checkpoint 的 held-out 评测使用 seed `314159` 与 `271828`，汇报：

- final support precision / recall / IoU；
- 8/16/32 每层 recall；
- candidate token-NFE 与 matched dense 8/12/20 NFE reduction；
- 同一个官方 TRELLIS decoder 下的 paired Chamfer²、P/R/F-score；
- decode failure 数量。

最终 on-policy checkpoint 同时保留两个 operating point：clean-calibrated threshold + 原始 probability-OR-mass 规则作为历史对照，generated-state IoU threshold + probability-only 规则作为严格匹配校准对象的 balanced point。二者使用完全相同的 checkpoint、NFE 与 held-out seeds，防止通过换模型或换随机数掩盖 precision/recall/compute trade-off；manifest 必须明确记录 pruning mode。

### 20.3 On-policy pruning calibration

clean calibration 把 GT stage representation 输入 occupancy head，只能描述 teacher-forced endpoint；实际采样阈值作用于模型自己生成的 state。正式 gate 因此新增 train-only on-policy calibration：最终模型先在 96 个训练资产上生成两条独立轨迹，再汇总每个 candidate 的 predicted probability 与 GT label 来选 threshold。

这两条最终 calibration trajectory 明确使用 probability-only，并以 clean threshold 作为第一次 rollout 的初始 operating point；IoU/F2 threshold 随后也只用于 probability-only。这里不能用 OR-mass trajectory 校准 probability-only sampler。训练阶段的 generated-parent replay 仍混合 OR 与 probability trajectory，因为它的目标是增强 denoiser 对 candidate-set shift 的鲁棒性，而不是估计某个固定 pruning rule 的阈值。

对于第 `r` 层，报告同时包含：

```text
conditional recall = retained GT candidates / GT candidates that reached this stage
candidate coverage = GT stage tokens that reached this stage / all GT stage tokens
global recall       = retained GT candidates / all GT stage tokens
```

只报告 conditional recall 会掩盖上游剪枝造成的不可恢复错误，因此论文表格/附录必须至少保留 candidate coverage 或 global recall。校准仍完全不看 held-out 数据。

早期 15K pilot 的纯 CPU smoke test 已显示该区别不能忽略：clean 阈值约为 `0.179/0.855/0.992`，generated-state 条件 recall 对应阈值约为 `2.7e-5/9.4e-8/4.3e-8`；generated candidate coverage 为 `1.000/0.970/0.941`。这说明 occupancy probability 存在严重 state-dependent calibration shift，而且 final global recall 的上限首先由前级 candidate coverage 决定。正式数据必须重新测量，不能直接把 pilot 数字外推。

把上述极低 target-recall threshold 直接用于 OR-mass 规则会退化成近乎 keep-all：4-asset pilot 得到 `R=1.000`，但 `P=IoU=0.095`。即使 generated-IoU probability threshold 本身约为 `0.998/0.998/0.993`，若仍保留 mass OR 分支，父 mass 会复制给整个 sibling group 并绕过 probability gate。因此 generated-IoU variant 必须使用 probability-only；OR-mass 只保留为历史 clean-calibrated 对照，而不能声称是 probability calibration 的闭环结果。

同 seed `314159` 的最终 pilot operating points：

| Calibration / rule | Precision | Recall | IoU | matched-dense reduction |
|---|---:|---:|---:|---:|
| clean / probability-OR-mass | 0.514 | 0.810 | **0.460** | 约 5.8x |
| generated IoU / probability-only | **0.553** | 0.683 | 0.440 | **7.66x** |
| generated F2 / probability-only | 0.433 | **0.942** | 0.422 | 5.29x |
| generated target-recall / probability-only | 0.095 | 1.000 | 0.095 | 近 dense |

前三者分别作为 historical-balanced、speed-first、recall-first 点；极端 target-recall 只保留为失败消融。阈值研究到此停止。正式 held-out 复用在 train replay 上定义的 IoU/F2 objective，不在 held-out 上调阈值；mesh decode 只覆盖 teacher、clean-balanced 与 generated-F2 三个更有解释价值的点，speed-first 仍汇报 support/token-NFE。

### 20.4 可恢复运行与统一产物

数据完成后启动：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash scripts/rdd3d/run_dynamic_sparse_formal_heldout_tmux.sh
tail -f logs/rdd3d_formal_heldout.log
```

最终统一产物为：

```text
eval_results/rdd3d_dynamic_sparse/<formal_run>/formal_summary.json
eval_results/rdd3d_dynamic_sparse/<formal_run>/formal_summary.md
```

`formal_summary.md` 是论文表格的直接候选，但只有当 32 held-out 的两个 seed 结论一致、且 mesh decode 无系统性失败时，才可把结果从“feasibility gate”升级为主要实验结论。

## 21. 正式 held-out gate 运行结果（进行中，2026-08-11）

最终数据成功数：

```text
metadata candidates: 256
download / mesh dump / dual-grid / SLat: 239 / 239 / 239 / 239
valid conditioning renders: 237
eligible after latent/render integrity audit: 237
split: 96 train / 32 held-out, seed 98317
rejected: 2 assets missing transforms.json
```

32 held-out 的纯 oracle hierarchy：

| Metric | Mean |
|---|---:|
| dynamic candidate token-NFE, 8/12/20 | 94,885 |
| progressive-dense / dynamic reduction | 9.10x |
| dense-32^3 / dynamic reduction | 16.83x |
| exact reconstruction max error | 1.91e-6 |
| sibling-detail parent-mean max error | 1.91e-6 |

这一步不使用 learned pruning，证明真实新资产上的 ordinary-average RDD hierarchy 与 sparse compute opportunity 成立。

split 的 sparsity 分布高度匹配：8^3 nonempty train/held-out 均值为 `117.3/116.2`；16^3 candidates/nonempty 为 `938.2/506.7` vs `929.5/497.7`；32^3 为 `4053.9/2195.3` vs `3981.8/2180.8`。因此 held-out 不是因为异常稀疏/稠密而变难。

相对旧 19K checkpoint normalization，formal train/held-out 的 feature mean z-shift RMS 仅 `0.0366/0.0354`，std ratio mean 为 `0.996/0.986`，per-channel std ratio 总范围约 `0.92–1.05`。旧 normalization 仍适用，zero-shot 失败不能解释为 TRELLIS SLat feature distribution mismatch。

19K pilot checkpoint 的完全 zero-shot held-out 结果：

| Seed | Precision | Recall | IoU |
|---:|---:|---:|---:|
| 314159 | 0.225 | 0.382 | 0.163 |
| 271828 | 0.230 | 0.386 | 0.167 |

seed `314159` 的 8/16/32 stage recall 为 `0.929/0.458/0.382`。两个 seed 高度一致，说明旧 32-asset checkpoint 有少量 coarse localization 泛化，但 refinement 从 generated parent 出发时明显崩溃；这与 pilot 的 generated-state distribution-shift 诊断一致，而不是偶然 seed。

zero-shot 平均 candidate token-NFE 为 `140,115±425`，progressive-dense reduction 只有 `5.06±0.02x`，明显低于 oracle `9.10x`。旧模型的 false-positive branches 同时损害几何质量和稀疏收益。

teacher-only adaptation 在 96 个新训练资产上从 19K checkpoint 继续训练 5K steps，但仍只看 ground-truth parent。它在完全相同的 held-out split 和两个 seed 上得到：

| Method | Precision | Recall | IoU | 8/16/32 recall | candidate token-NFE | reduction |
|---|---:|---:|---:|---:|---:|---:|
| 19K zero-shot | `0.228±0.002` | `0.384±0.002` | `0.165±0.002` | `0.929/0.460/0.384` | `140,115±425` | `5.06±0.02x` |
| teacher-only 24K | **`0.246±0.001`** | `0.217±0.006` | `0.131±0.002` | `0.687/0.403/0.217` | **`114,839±2,304`** | **`6.17±0.12x`** |

这个结果不是小 seed 波动：两个 teacher seed 的 final recall 分别为 `0.223/0.211`，IoU 为 `0.133/0.128`。teacher-only 稍微提高 precision 并减少预测 token，却把 final recall 相对降低约 `43.5%`，IoU 相对降低约 `20.6%`；退化还从 8^3 到 16^3、32^3 逐层累积。它说明仅增加新数据和训练步数会把 clean-parent 上学到的激进 occupancy boundary 带到 generated state，不能闭合 train/inference gap。该失败对照为 generated-parent replay 提供了更强的因果基线，但只有 on-policy 24K→29K 在同一 held-out 协议下恢复质量后，才能形成正面方法结论。

两条 96-asset train-only replay trajectory 已生成完成；24K→29K mixed on-policy replay 正在 GPU 0/1 上运行，训练过程截至 24.8K 未出现 NaN/Inf。

29K mixed on-policy 与完整 paired mesh 评测随后完成。正式两个 seed 的汇总为：

| Method / pruning rule | Precision | Recall | IoU | 8/16/32 recall | token-NFE reduction | Chamfer² | Mesh F-score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19K zero-shot | 0.228±0.002 | 0.384±0.002 | 0.165±0.002 | 0.929/0.460/0.384 | 5.06±0.02x | -- | -- |
| teacher-only 24K | 0.246±0.001 | 0.217±0.006 | 0.131±0.002 | 0.687/0.403/0.217 | 6.17±0.12x | 0.010891±0.000616 | 0.084±0.001 |
| on-policy 29K, clean/OR | 0.247±0.005 | 0.358±0.008 | 0.169±0.005 | 0.597/0.554/0.358 | 5.00±0.03x | 0.007035±0.000286 | 0.130±0.001 |
| on-policy 29K, generated IoU/probability | **0.251±0.001** | 0.294±0.001 | 0.155±0.002 | 0.568/0.487/0.294 | **6.11±0.06x** | -- | -- |
| on-policy 29K, generated F2/probability | 0.223±0.006 | **0.522±0.013** | **0.183±0.005** | **0.605/0.594/0.522** | 4.61±0.01x | **0.005998±0.000007** | **0.158±0.006** |

结论必须分成两层。正面部分是：on-policy replay 相对完全相同新数据上的 teacher-only 对照，把 F2 final recall 从 `0.217` 恢复到 `0.522`、IoU 从 `0.131` 提到 `0.183`；paired mesh Chamfer² 下降约 `44.9%`，F-score 从 `0.084` 提到 `0.158`（约 `88%` relative）。两个 seed 的标准差很小，generated-parent training 的收益不是 seed 偶然性。负面部分是：绝对 mesh 质量仍低，clean/OR 只勉强超过 19K zero-shot IoU，而且 8^3 recall 只有约 `0.60`，远低于 zero-shot 的 `0.929`。96-asset adaptation 的 coarse generator 仍是正式泛化瓶颈，当前结果只能作为小规模 feasibility / distribution-shift 实验，不能作为论文主 3D quality claim。

### 21.1 一次性 generated calibration 仍是 off-policy

正式脚本最初用 clean probability policy 生成两条轨迹，然后从同一批轨迹同时选 8/16/32 的 IoU 或 F2 threshold。对于层级动态剪枝，这并不严格闭环：改变 8^3 threshold 会改变 16^3 candidate set，改变 16^3 threshold 又会改变 32^3 candidate set 和模型内部的 candidate-set global context。后层 threshold 因而仍是在旧 policy 分布上选择的。

这个问题在 29K F2 中不可忽略。一次性 calibration 把 8^3 threshold 设为 `0.849`；clean-policy trajectory 上的 16/32 candidate coverage 分别为 `0.939/0.854`，F2 所选 threshold 的 global recall 分别为 `0.919/0.796`。这些数字不能直接当作最终 F2 policy 的闭环 train recall，因为应用新 8^3 threshold 后后两层分布已经变化。

修复采用逐层顺序 calibration，且不覆盖旧结果：

1. 用 root candidate 恒定的轨迹选择 8^3 threshold；
2. 固定 8^3 threshold 重新生成两条 train-only rollout；
3. 在新 16^3 candidate distribution 上选择 16^3 threshold；
4. 固定 8/16 threshold 再生成两条 rollout；
5. 在正确的 32^3 candidate distribution 上选择 final threshold；
6. 用最终闭环 policy 跑两个 held-out seed。

IoU 与 F2 objective 分开闭环，因为它们从第一层起就是不同 policy。运行 session 为 `rdd3d_closed_loop_calibration`，日志为 `logs/rdd3d_closed_loop_calibration.log`。旧表格明确标记为 one-shot/off-policy generated calibration；只有 closed-loop 结果可作为最终动态 pruning operating point。

闭环结果已经完成：

| Calibration | Precision | Recall | IoU | 8/16/32 recall | reduction |
|---|---:|---:|---:|---:|---:|
| one-shot IoU | 0.251±0.001 | 0.294±0.001 | 0.155±0.002 | 0.568/0.487/0.294 | **6.11±0.06x** |
| closed-loop IoU | 0.249±0.004 | **0.305±0.002** | **0.158±0.003** | 0.568/0.497/0.305 | 5.98±0.06x |
| one-shot F2 | 0.223±0.006 | **0.522±0.013** | 0.183±0.005 | 0.605/0.594/0.522 | 4.61±0.01x |
| closed-loop F2 | **0.232±0.007** | 0.476±0.013 | **0.183±0.006** | 0.605/0.586/0.476 | **4.70±0.01x** |

严格闭环没有掩盖或修复生成器瓶颈。IoU policy 的 final IoU 小幅提高；F2 policy 用更少 token 保持相同 IoU，但 recall 比一次性 policy 低。一次性 F2 的额外 recall 来自 policy distribution 偏移下更保守的实际行为，不能称为严格 calibrated F2。closed-loop F2 是可信 operating point；两者共同确认 8^3 recall 约 `0.605` 才是当前主要上限。

### 21.2 下一训练改进的预注册顺序

若 closed-loop 仍被约 `0.60` 的 coarse recall 限制，下一轮不先扩大模型或继续扫阈值，而按以下顺序做：

1. 用 29K 当前 policy 重新生成 replay，做第二轮 DAgger/data aggregation；
2. 训练 stage probability 从 `0.25/0.30/0.45` 改为 coarse-heavy `0.40/0.30/0.30`，因为 8^3 均值误差在后续 zero-mean sibling detail 中不可修复；
3. LR 降到 `1e-5`，固定追加 10K steps，避免用当前 held-out 选择停止点；
4. 单独测试 sampler endpoint occupancy-head replay，只训练 385 个 `nonempty_head` 参数，并逐参数断言 velocity/shared denoiser 完全不变；
5. 若多保留候选仍系统性改变正确 branch，才测试 candidate-set-stable sparse aggregation（固定零填充归一化或去掉全候选均值），而不把更多 halo/threshold sweep 当作模型改进。

当前 32 held-out 在上述方法开发后应降级为 validation/dev。数据审计中另有约 109 个有效未用资产，最终论文数字必须从中预先固定新 test split 并只评一次，防止反复观察当前 held-out 后产生 model-selection leakage。

### 21.3 Stage-specific hybrid 诊断

为了区分“29K coarse 被破坏”与“整个 29K refinement 无效”，做了一个零训练 hybrid：8^3 调用 zero-shot coarse recall 很高的 19K checkpoint，16^3/32^3 调用 29K on-policy checkpoint；feature normalization、NFE 和 clean/OR pruning 不变。

| Model | Precision | Recall | IoU | 8/16/32 recall | reduction |
|---|---:|---:|---:|---:|---:|
| 29K shared, clean/OR | 0.247±0.005 | **0.358±0.008** | **0.169±0.005** | 0.597/0.554/0.358 | **5.00±0.03x** |
| 19K coarse + 29K refine | 0.230±0.003 | 0.346±0.003 | 0.159±0.002 | **0.929**/0.464/0.346 | 4.75±0.03x |

hybrid 把 coarse recall 完整恢复到 `0.929`，但 16^3 recall 立即掉到 `0.464`，最终没有超过纯 29K。这排除了“只保留旧 coarse 模型即可”的简单修复：29K refinement 已适应 29K coarse 的 generated representation error 和 candidate distribution，19K/29K stage interface 不兼容。后续训练若采用分 stage 模型，必须在上游模型固定后用其真实 rollout 训练下游，而不能独立训练后直接拼接。

endpoint occupancy-head 实验随后启动：sampler 额外保存每层 pruning 前的完整 generated endpoint representation；只训练线性 `nonempty_head` 的 385 个参数，并断言 velocity/shared denoiser 所有参数最大变化严格为零。它将区分 pruning ranking 问题与 coarse representation/velocity 问题。

endpoint-head 已完成。train endpoint 上的 threshold-independent best IoU 从 `0.733/0.496/0.366` 提升到 `0.750/0.561/0.396`；checkpoint 审计确认除 `nonempty_head` 外所有参数最大变化严格为 `0.0`。正式 held-out sequential closed-loop 结果：

| Model / policy | Precision | Recall | IoU | 8/16/32 recall | reduction |
|---|---:|---:|---:|---:|---:|
| 29K closed-loop IoU | 0.249±0.004 | 0.305±0.002 | 0.158±0.003 | 0.568/0.497/0.305 | 5.98±0.06x |
| + endpoint head, closed-loop IoU | **0.270±0.002** | 0.298±0.004 | **0.165±0.002** | 0.547/0.386/0.298 | **8.40±0.04x** |
| 29K closed-loop F2 | 0.232±0.007 | **0.476±0.013** | 0.183±0.006 | 0.605/0.586/0.476 | 4.70±0.01x |
| + endpoint head, closed-loop F2 | **0.245±0.004** | 0.453±0.011 | **0.187±0.005** | 0.608/0.550/0.453 | **5.23±0.03x** |

因此 endpoint-state supervision 是有效的 Pareto 改进：IoU/speed 点在略低 recall 下把 reduction 从约 `6.0x` 提到 `8.4x`；F2 点同时提高 IoU 与 reduction。由于 velocity 完全未改，提升可归因于实际采样分布上的 pruning ranking，而不是生成模型变强。它仍没有提高 8^3 global recall 上限，故不能替代 coarse generator 或 stage-interface 训练。

下一实验固定 19K coarse checkpoint，不再直接拼接 29K refinement；先用该 hybrid 真实生成两条 train replay，再只以 `0/0.4/0.6` stage probability 和 `1e-5` LR 训练 16^3/32^3 5K steps。推理继续用固定 19K coarse + 新 refinement checkpoint。这是对 hybrid 所暴露 interface shift 的直接修复，且不会再次破坏 coarse 模型。

在查看 34K held-out 结果前，预先固定判断规则，避免从中间 checkpoint 或多个阈值里挑最好数字：只评固定 34K、两个既定 dev seed。最低成功条件是相对零训练 hybrid 同时提高 16^3 recall（`0.464`）、32^3 recall（`0.346`）和 final IoU（`0.159`）；更有意义的成功条件是在保留 19K coarse 的约 `0.929` recall 时超过 shared-29K clean/OR 的 IoU `0.169`。只有满足最低条件才继续运行该模型的 strict sequential F2 calibration；否则把结果记录为 stage-dispatch + downstream adaptation 的失败诊断，并转向第二轮同一 interface policy 的 DAgger，而不观察 30K--33K held-out 来选 checkpoint。

固定 34K 的 clean/OR 结果满足上述两个条件：

| Model | Precision | Recall | IoU | 8/16/32 recall | reduction |
|---|---:|---:|---:|---:|---:|
| 19K coarse + 29K refine, zero-train hybrid | 0.230±0.003 | 0.346±0.003 | 0.159±0.002 | 0.929/0.464/0.346 | 4.75±0.03x |
| 19K coarse + interface-adapted 34K refine | **0.238±0.001** | **0.472±0.003** | **0.186±0.000** | 0.929/**0.752**/**0.472** | 3.03±0.00x |

相对零训练 hybrid，16^3 global recall 提高约 `62%`、final recall 提高约 `36%`、IoU 提高约 `17%`；同时也超过 shared-29K clean/OR 的 IoU `0.169`，而 8^3 recall 保持 `0.929`。这验证了 stage-specific dispatch 本身并非无效，真正必要的是用固定上游模型的 generated endpoint 训练下游。当前 clean/OR policy 因保留大量 false-positive branch，compute reduction 仅 `3.03x`；因此下一步只运行预注册的 strict sequential F2 policy，检验能否在不丢失主要 recall/IoU 收益的前提下回收稀疏计算。运行 session 为 `rdd3d_stage_interface_closed_loop`，日志为 `logs/rdd3d_stage_interface_closed_loop.log`。

strict sequential F2 已完成：`P/R/IoU=0.238/0.470/0.181`，8/16/32 recall 为 `0.997/0.601/0.470`，reduction 为 `3.66x`。它相对 clean/OR 将 compute reduction 从 `3.03x` 提高到 `3.66x`，且 final recall 几乎不变，但 IoU 从 `0.186` 小降到 `0.181`。intersection feature MSE 为 `31.449`，比 shared-29K closed-loop F2 的 `31.878` 低约 `1.35%`。

这个结果要诚实拆开解释。正面上，它证明固定 coarse + generated-parent downstream adaptation 能在 probability-only 的严格闭环下保留约 `0.47` final recall，而且两个 seed 方差很小。负面上，它没有全面超过 shared-29K F2：后者 `P/R/IoU=0.232/0.476/0.183`、reduction `4.70x`。interface F2 为达到目标权重把 8^3 recall 推到近乎 keep-all 的 `0.997`，大量 false-positive coarse branch 使总体更慢。因此该训练目前是清楚的 stage-interface 因果验证，而不是最终 Pareto 最优模型。下一诊断保持 coarse/refine velocity 完全冻结，只在 strict policy 的真实 endpoint state 上训练 refine checkpoint 的 385 参数 occupancy head，并令 stage probability 为 `0/0.4/0.6`；目标是提高 16/32 pruning ranking，而不是再次改变生成轨迹。

同一正式协议下对 interface strict-F2 每个 seed 解码 4 个 paired TRELLIS mesh：Chamfer² 为 `0.005693±0.000263`，相对旧 on-policy generated-F2 的 `0.005998±0.000007` 低约 `5.1%`；Mesh F-score 为 `0.157±0.003`，与旧值 `0.158±0.006` 基本相同。这与 feature MSE 的小幅改善方向一致，但不支持“几何质量大幅提升”的表述；现阶段最强证据仍是 stage-interface 因果恢复和可学习动态剪枝，而非绝对 3D generation quality。

interface endpoint-head 的冻结审计确认除 `nonempty_head` 外参数最大变化为 `0.0`；train endpoint 的 best-IoU ranking 在 16/32 层从 `0.469/0.366` 提高到 `0.537/0.389`。但完全重新校准后的 dev 结果为 `P/R/IoU=0.250/0.400/0.176`、reduction `4.32x`，16/32 recall 为 `0.528/0.400`。它提高 precision 和速度，却丢失太多 recall，并被旧 shared-29K endpoint-head strict-F2（`0.245/0.453/0.187`、`5.23x`）在 IoU、recall 与速度上同时支配。因此停止继续在当前 dev 上做 head/threshold 搜索；stage-interface 保留为重要因果消融，不选作最终 operating point。

冻结的最终 balanced 候选为 shared-29K + endpoint-state occupancy head + strict sequential F2。冻结理由只使用当前 32-asset dev：它在所有已测 balanced 方法中给出最高 IoU `0.187`，同时保留 `0.453` recall 和 `5.23x` reduction，且 velocity/shared denoiser 与 29K 完全相同。下一步才建立 seed `20260812`、64 个资产的全新 test split；该 split 排除 pilot、96 train 和本 32 dev，只对冻结候选及预先指定基线各运行一次，不再据 test 结果改方法或阈值。

## 22. 冻结的 64-asset final test（已完成，2026-08-12）

final split 用 seed `20260812` 从剩余 109 个完整有效资产中固定选择 64 个；排除集合共 160 个资产，覆盖 pilot、96 train 与 32 dev。64/64 DINOv2 conditioning 已生成。learned evaluation 开始前固定的三行方法为：19K zero-shot clean/OR、29K on-policy clean/OR、29K endpoint-head strict sequential F2；另用最终 checkpoint 做 force-keep 8/16/32 runtime control。final test 不用于 checkpoint、threshold、pruning objective 或方法选择。

纯 oracle hierarchy（未运行 learned model）为：

| Metric | Final-test mean |
|---|---:|
| 8^3 candidates / nonempty | `512 / 123.3` |
| 16^3 candidates / nonempty | `986.5 / 542.3` |
| 32^3 candidates / nonempty | `4338.0 / 2429.8` |
| dynamic candidate token-NFE, 8/12/20 | `102,694` |
| progressive-dense / dynamic reduction | `9.45x` |
| dense-32^3 / dynamic reduction | `17.48x` |
| exact reconstruction max error | `1.91e-6` |
| sibling-detail mean max error | `1.91e-6` |

这比 32-dev oracle 的 `9.10x` progressive reduction 略高，说明 final test 没有因为特别稠密而人为削弱 sparse opportunity。learned sampling 同时记录 sampler-only wall-clock 与 peak allocated CUDA memory；正式速度结论以同 checkpoint、同 seed、同 8/12/20 NFE 的 force-keep control 为准，token-NFE 只作为硬件无关的补充。

### 22.1 一次性 learned final-test 结果

冻结评估已完成；运行后不再修改模型、head、threshold 或 pruning objective：

| Method | Precision | Recall | IoU | 8/16/32 recall | token-NFE reduction | wall/asset | peak allocated | Chamfer² | Mesh F-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19K zero-shot, clean/OR | 0.222±0.001 | 0.376±0.001 | 0.157±0.000 | 0.912/0.439/0.376 | **5.18±0.01x** | **0.507±0.001 s** | **911±53 MiB** | -- | -- |
| shared-29K on-policy, clean/OR | **0.246±0.001** | 0.406±0.000 | 0.175±0.000 | 0.616/0.593/0.406 | 4.38±0.01x | 0.572±0.003 s | 1440±60 MiB | -- | -- |
| shared-29K + endpoint head, strict F2 | 0.245±0.003 | **0.509±0.003** | **0.191±0.002** | **0.629/0.594/0.509** | 4.53±0.01x | 0.556±0.002 s | 1474±68 MiB | 0.008564±0.000050 | 0.159±0.000 |

最终方法相对相同 29K denoiser 的 clean/OR，将 micro support recall 从 `0.406` 提到 `0.509`（约 `25.4%` relative），IoU 从 `0.175` 提到 `0.191`（约 `9.1%` relative），precision 基本不变；相对 19K zero-shot，IoU 提高约 `21.7%`。final test 的结果不但没有消失，selected method 的 IoU/recall 还略高于 32-dev 的 `0.187/0.453`，说明结论不是在 dev 上过拟合出的单一数字。

按资产配对、先对两个 sampling seed 取均值、再对 64 个资产做 20,000 次 bootstrap：

| Comparison | Metric delta | 95% CI | asset win rate | P(delta>0) |
|---|---:|---:|---:|---:|
| endpoint head - 29K clean/OR | IoU `+0.0162` | `[+0.0122,+0.0203]` | 85.9% | 100.0% |
| endpoint head - 29K clean/OR | Recall `+0.1024` | `[+0.0927,+0.1124]` | 100.0% | 100.0% |
| endpoint head - 29K clean/OR | Precision `-0.0007` | `[-0.0050,+0.0036]` | 51.6% | 37.7% |
| endpoint head - 19K zero-shot | IoU `+0.0347` | `[+0.0175,+0.0534]` | 67.2% | 100.0% |
| endpoint head - 19K zero-shot | Recall `+0.1321` | `[+0.0879,+0.1784]` | 70.3% | 100.0% |

因此改进的统计机制很清楚：endpoint-state supervision 主要提高 recall，而不是用 precision 下降换一个不稳定的 aggregate IoU。它只修改 385 个 occupancy-head 参数，frozen-denoiser audit 的最大误差为零，所以 final 提升可归因于 RDD 动态剪枝决策，而非偷偷增加 generator capacity。

### 22.2 同 checkpoint 的实际稀疏速度

在 final test 的同一前 16 个资产、两个固定 seed/process 上，各丢弃首个 warm-up，得到 30 个 measured samples：

| Policy | wall/asset | candidate token-NFE | peak allocated CUDA memory |
|---|---:|---:|---:|
| selected strict-F2 sparse | `0.5056±0.2181 s` | `137,801` | `1542 MiB` |
| force-keep 8/16/32 | `1.9904±0.0027 s` | `708,608` | `2852 MiB` |

实际 sampler-only speedup 为 **`3.94x`**，measured token-NFE reduction 为 `5.14x`，peak allocated memory 减少约 **`46%`**。实际速度低于 token-NFE 比例是合理的：hash neighbor construction、kernel launch 和固定 8^3/conditioning 开销不会随 token 数同比下降。这个同模型 force-keep control 比单独报告 FLOPs proxy 更可信，也直接回答了 RDD progressive sparse generation 是否真的节省计算。

### 22.3 当前结论边界

这套结果足以支持三个严谨结论：ordinary-average RDD hierarchy 对 TRELLIS SLat 可精确构造；learned stagewise pruning 在未见资产上能保持约 `0.51` support recall；动态 sparse 路径带来 `3.94x` 实测 sampler 加速和约 `46%` peak allocated memory 降低。它还不足以支持“已经得到有竞争力的通用 3D generator”：模型仅 25.6M、训练资产 96 个，final paired mesh F-score 约 `0.159`，且没有与大型 3D generation backbone 做同预算质量比较。论文中应把它定位为 RDD sparse-generation mechanism 的 controlled proof-of-concept；下一阶段才是把同一层级/剪枝机制移入强 TRELLIS.2 denoiser，并在大规模数据上比较质量-速度 Pareto。
