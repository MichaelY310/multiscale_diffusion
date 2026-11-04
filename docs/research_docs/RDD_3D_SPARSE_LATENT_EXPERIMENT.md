# RDD-3D 稀疏 latent 路线：审计、理论与可执行实验方案

> 状态：历史 fixed-support 分支。2026-08-11 起，当前主方案改为 [RDD_3D_DYNAMIC_SPARSE_GENERATION.md](./RDD_3D_DYNAMIC_SPARSE_GENERATION.md)：empty voxel 是合法 mass-SLat，RDD 在每个阶段自行预测并剪除 all-empty blocks，不再依赖 TRELLIS Structure Model 预先固定 support。本文的真实 SLat 稀疏统计、decoder projection 和 fixed-support 对照仍作为历史证据保留，但训练路线不再是主方法。

## 1. 结论先行

RDD 在 3D 上仍然有研究价值，但必须把主张收窄并做严格：

1. **不要直接平均显式几何/PBR 参数。** 顶点位置、四元数、scale、opacity、face flag 的欧氏平均通常没有可解码的语义。GaussianCube 还是 100% dense，无法发挥稀疏优势。
2. **采用 TRELLIS.2 的 shape SC-VAE latent（SLat）。** 它把 O-Voxel 经 16 倍空间压缩变成 32 通道的稀疏、规则格点 latent；局部 latent 是由 VAE 学出来的连续表征，线性平均比直接平均物理参数合理得多。
3. **结构与几何分开。** 先沿用 TRELLIS.2 的 structure flow 得到最终 32³ active support；RDD 只替换 geometry flow。在固定 support 上，active-child average 与 masked nearest replication 构成严格的线性幂等投影。
4. **论文主张不是“第一个 sparse/multiscale 3D generator”。** 这一点已被 XCube、OctFusion、ShapeShifter、CADD、Grow3D、Fractal3DGen 等覆盖。可辩护的主张是：
   - topology-conditioned 的精确 latent Haar/RDD 分解；
   - detail-subspace noise filling；
   - 一个共享 sparse DiT、一个全局时间轴上的 piecewise rectified flow；
   - 在相同结构、初始化、训练数据与几何 decoder 下，实测降低 token-NFE、墙钟时间和显存，同时保持或改善几何质量。
5. **当前 pilot 已证明表示与稀疏计算可行，但尚未证明生成质量。** 32 个物体的 1024³ representation/decoder/latency 结果可以写成 preliminary ablation；论文主质量表仍需要成规模的 ObjaverseXL 训练子集、独立测试集、flat TRELLIS.2 匹配基线和多随机种子。

## 2. 仓库中已经做过什么

### 2.1 Vertex / Set Transformer 路线

旧路线把 mesh/point cloud 划入规则 cell，再用 Set Transformer 把每个 cell 内的变长点集压成固定 latent：

`mesh vertices -> sparse cells -> Set Transformer AE -> RDD-DiT -> Set Transformer decoder`

仓库中完成了 vertex 数据预处理、Set Transformer AE、多阶段 sparse dataset、3D DiT、DDPM/RDD drift、noise filling、采样和 GLB 导出代码。AE 做过大量结构消融：

- concat pooling 的最好代理 EMD 约 `3.55e-4`，但 flatten 依赖输入顺序，**不是 permutation invariant**；
- V2 concat 的 bottleneck 256、输入约 320 维，压缩率很弱，验证 EMD 约 `0.0024`；
- 真正 permutation-invariant 的 PMA1 最好约 `0.0098`，明显落后；
- KL/VAE 版本约 `0.15`，尚不足以支撑生成；
- 这些 EMD 是逐坐标 Wasserstein proxy，不是严格 3D matching EMD，不能当成高质量 mesh 重建证据。

因此，旧 AE 更像“记忆 cell 内小点集”的工程验证，而不是一个已经验证的、适合生成的 3D latent space。

### 2.2 Face / triangle-soup 路线

仓库实现了 face cell 数据、Face Set Transformer、存在性与 scale 头，并考虑了 Hungarian/确定性排序问题。但 triangle soup 在跨 cell 合并时没有稳定的共享顶点、边连接和拓扑恢复机制，尚未形成可量化的完整 mesh generation 结果。

### 2.3 MeshAnything 多阶段生成尝试

历史训练使用 4,216 个样本、六级 schedule：

`8:100, 16:150, 32:150, 64:200, 128:200, 256:200`

一次训练只进行到约 2.9K step；另一次在接近 95 GB 显存时 OOM。虽然数据是 sparse storage，模型仍在每一级对全部 active token 做全局 self-attention，复杂度近似 `O(N²)`，所以 128/256 级别并没有得到可扩展性。仓库中没有可信的生成评估、几何指标或与基线的比较。

### 2.4 GaussianCube 调研

旧调研已经实测/确认 GaussianCube 的 32³ 网格每个位置都有 Gaussian，训练数据本质上 100% dense。14 通道还混合 position offset、rotation、scale、opacity、SH 等参数；普通 average pooling 既没有物理不变性，也不能得到“同一物体的粗尺度 Gaussian”。它不适合作为 RDD sparse generation 的主表示。

## 3. 相关工作：哪些已经做过，RDD 还能剩下什么

| 方法 | 表示与层级 | 与 RDD-3D 的重叠 | 仍有区别 |
|---|---|---|---|
| Neural Wavelet-domain Diffusion (2022) | TSDF coarse + wavelet detail | 最接近“均值/低频 + detail”数学结构 | 分开的 coarse generator/detail predictor；不是固定稀疏 support 上的共享全局-time flow |
| XCube (CVPR 2024) | VDB sparse voxel hierarchy、级联 latent diffusion | sparse、coarse-to-fine、neural voxel feature | 多层级/多模块级联；没有本工作的精确 projector/detail-path 对照 |
| OctFusion (2024) | octree VAE + 共享 multiscale U-Net | 单模型共享层级、稀疏层级生成 | octree diffusion；不是一个固定 SLat support 上的 piecewise rectified flow |
| ShapeShifter (CVPR 2025) | 16→256 sparse point/normal/color/mask voxel | 2³ average、mask max、multiscale sparse diffusion，和旧 RDD3D 很接近 | 单物体 variation；各层独立 diffusion 与 learned upsampler；显式特征而非压缩 SLat |
| CADD (ICCV 2025) | coarse cube state 作为 fine discrete diffusion prior | coarse state、sparse local attention、分层生成 | discrete semantic scene voxel，不是连续 latent detail flow |
| TRELLIS / TRELLIS.2 | structure -> sparse geometry latent -> material | 提供本工作要采用的 structure/SLat/decoder | 原版 geometry flow 在全部 active latent 上全程 flat denoise |
| Direct3D-S2 (NeurIPS 2025) | sparse VAE + local/serialized sparse attention | 可扩展 sparse backbone | 单分辨率生成；没有 RDD 时间-分辨率路径 |
| Fractal3DGen (ICLR 2026 submission) | sparse SDF latent、hierarchical transformer | coarse-to-fine、层级自相似 | autoregressive/fractal hierarchy，不是 FM detail subspace |
| LATTICE (CVPR 2026) | coarse sparse anchor + VoxSet rectified flow | structure/geometry 两阶段、FM | semi-structured set latent，第二阶段不是逐分辨率 RDD flow |
| TRELLIS.2 (CVPR 2026) | O-Voxel -> 16× SC-VAE -> SLat，三阶段 FM | 直接 backbone/表示 | 本工作改变 geometry flow 的路径与计算分配，不声称新表示 |
| Grow3D (TOG 2026) | multiscale quantized sparse octree、next-scale AR | 层级、共享信息、快速生成 | 离散 next-scale AR，不是 continuous FM |
| LATO / LATO.2 (2026) | sparse topology latent，structure/vertex/connectivity flow | 拓扑与连续几何解耦 | 目标是显式 mesh topology；没有 RDD resolution-time decomposition |

**不能再写的 novelty：** sparse voxel、coarse-to-fine、nearest upsample、average pooling、共享 multiscale network、structure-first，单独任何一个都不是新的。

**可以尝试证明的 novelty（按重要性排序）：**

1. 固定生成拓扑条件下，SLat 的精确 active-child projector 与多级 detail decomposition；
2. 只在 projector 的 kernel/detail 子空间注入噪声，保持 coarse component 精确不变；
3. 把各尺度 path 嵌入同一 global-time rectified flow，模型预测对 global time 的速度，而不是训练多个独立超分模型；
4. 结构模型提供 fine support，geometry flow 用逐级 token 激活获得真实稀疏计算收益；
5. 同一框架跨 2D、video、3D 的实证统一性只能作为 breadth，不可代替 3D 的新颖性与实验。

## 4. 为什么选择 TRELLIS.2 SLat

### 4.1 表示选择矩阵

| 表示 | 稀疏 | 线性平均合理性 | 拓扑 | 现成 encoder/decoder | 决策 |
|---|---:|---:|---|---:|---|
| occupancy / TSDF | 是 | occupancy 较差；TSDF 有尺度语义 | 偏 watertight field | 有 | 可做简化 baseline，不做主结果 |
| point/vertex cell latent（旧 AE） | 是 | 取决于 AE，当前没验证好 | connectivity 丢失 | 自研、不成熟 | 放弃主线 |
| triangle soup | 是 | 差 | merge 困难 | 不完整 | 放弃 |
| GaussianCube | 否 | 物理参数平均不合理 | Gaussian set | 有 | 放弃 |
| O-Voxel 原始通道 | 是 | vertex/flags/PBR 平均不合理 | 任意拓扑 | 有 | 只作 SC-VAE 输入，不直接 RDD |
| TRELLIS.2 shape SLat | 是 | 学习的连续局部 latent，最合理 | decoder 支持 open/non-manifold/internal | 官方完整 | **主路线** |
| octree latent（OctFusion） | 是 | 合理 | 多为 field/manifold | 有 | 强 baseline/备选路线 |

TRELLIS.2 官方事实：O-Voxel 支持开放面、non-manifold 和内部结构；shape SC-VAE 16 倍空间压缩、每 active token 32 维；512³ 物体对应 32³ latent，论文平均约 2.2K token，1024³ 对应 64³ latent，平均约 9.6K token。它已经将“不适合直接平均的物理几何”变换成了学习 latent，正好符合本项目最初的判断。

### 4.2 必须诚实说明的限制

- 本工作复用而不是发明 O-Voxel、SC-VAE 和 sparse DiT。
- fine support 来自 TRELLIS.2 structure model，所以加速主要发生在 geometry stage；不能宣称整个 3D pipeline 都按同一倍数加速。
- 已知 fine support 是条件生成阶段的一部分，不等于训练时偷看 ground-truth。正式 end-to-end 推理必须使用 structure model 的预测 support。
- global self-attention 的计算更接近 token 数平方；token-NFE、实测延迟和 kernel 利用率都要报告，不能只报告理论 voxel 数。

## 5. 严格的 topology-conditioned RDD 分解

设 structure model 给出最细支持集 `M_R`，最细分辨率 `R=32`。由它确定各粗层支持：

`M_r = { floor(i / 2) : i in M_{2r} }`。

对固定 support，定义 active-child downsample：

`(D_M z)(p) = (1 / |C_M(p)|) * sum_{c in C_M(p)} z(c)`，

其中 `C_M(p)` 是父 voxel `p` 下真实 active 的 children。定义 masked nearest upsample：

`(U_M y)(c) = y(parent(c)), c in M_{2r}`。

于是 `P_M = U_M D_M` 是固定 support 上的线性幂等投影：

`P_M² = P_M`。

detail 为

`delta_{2r} = z_{2r} - U_M z_r`，

且

`D_M delta_{2r} = 0`。

这意味着 detail 在每个 active sibling group 内严格零均值。注意每一层必须从相邻的细一级**递归平均**；如果所有粗层都直接从最细 token 按数量加权平均，active child 数量不均时上述性质不成立。

### 5.1 与“dense-zero average pooling”的区别

若把不存在的 child 当作零，并对固定 8 个 child 平均，普通 `AvgPool3D` 的粗值是 `(1/8) sum child`。一个 active parent nearest-expand 后会产生 8 个候选 child；那些本应 inactive 的位置必须用负 residual 抵消 parent，不能免费删除。因此旧文档中“empty voxel drift=0，所以 parent 后所有空 child 都能直接 pruning”的说法只对整个 2³ block 全零时成立。

本方案使用 structure model 已知的 `M_R`，在 support 上定义 projector。这既保留严格数学性质，也避免生成 8 倍候选后再猜哪些 child 应被删除。为了透明，pilot 同时统计 dense-zero candidate expansion 的 waste，作为这种设计选择的依据。

## 6. RDD rectified flow

采用 clean-to-noise 时间 `t in [0,1]`，采样从 1 积分到 0。分辨率 8/16/32 的时间比例默认取 fine/middle/coarse = 0.5/0.3/0.2，因此全局边界为：

- fine 32³: `[0.0, 0.5]`
- middle 16³: `[0.5, 0.8]`
- coarse 8³: `[0.8, 1.0]`

最粗级：

`x(p) = (1-p) z_8 + [sigma_min + (1-sigma_min)p] epsilon`。

细级只在 detail 子空间注噪：

`eta = (I-P_M) epsilon`

`x(p) = U_M z_parent + (1-p) delta + [sigma_min + (1-sigma_min)p] eta`。

局部速度为：

`dx/dp = (1-sigma_min) eta - delta`。

网络输入是全局时间，所以训练 target 必须除以对应 band width `Delta t`：

`dx/dt = (dx/dp) / Delta t`。

采样进入新分辨率时，先 nearest-copy 已生成 parent，再加入 detail-subspace noise；随后只预测该尺度的 detail velocity。网络输出在细级也投影回 `ker(D_M)`，保证更新不会破坏已经生成的 coarse mean。

默认 40 NFE 分配为 coarse/middle/fine = **8/12/20**。这和 0.2/0.3/0.5 的 global-time band 宽度严格成比例，每个 Euler step 都有相同的全局 `Delta t=0.025`。旧的 8/8/24 只能作为“额外偏向 fine stage”的 schedule ablation，不能再作为公平默认。论文必须同时给出：

- 等 NFE：flat 40 vs RDD 40；
- 等 token-NFE：调节 flat/RDD step，令 `sum_s N_s * steps_s` 接近；
- 实际 latency/peak VRAM；
- 若使用全局 attention，再报告 `sum_s N_s² * steps_s` 作为 attention proxy。

## 7. 可执行实验计划

### Phase A：表示与稀疏性 pilot（已完成）

目标不是训练出会生成的模型，而是回答三个硬问题：

1. O-Voxel 和 32-channel SLat 到底有多稀疏？
2. 32→16→8 后 token 是否明显下降？
3. dense-zero 8-child candidate expansion 的 waste 是否足够大，证明 known-support projector 有必要？

当前脚本从 ObjaverseXL GitHub shard 得到 32 个成功资产，执行：mesh dump -> 512³/1024³ O-Voxel dual grid -> 官方 shape SC-VAE encode -> 三级层级统计。

#### 实测结果（32 个相同资产）

| 输出/latent 分辨率 | coarse / middle / fine 平均 token | 8/12/20 token-NFE reduction | attention proxy reduction |
|---|---:|---:|---:|
| 512³ / 32³ | 8³: 105.4 / 16³: 445.6 / 32³: 1935.6 | 1.719× | 1.933× |
| 1024³ / 64³ | 16³: 445.6 / 32³: 1935.7 / 64³: 8375.0 | 1.721× | 1.934× |

固定 support 分解的最大重建误差为 `1.91e-6`，最大 parent-mean detail 误差为 `3.05e-6`；都显著低于 `1e-5` 门槛。dense-zero 候选扩张的平均浪费约 46%--48%，说明“先扩 8 个 child 再 pruning”确实会制造大量无效 token。

1024³ decoder 的三物体 paired projection 结果（参考为原始 SLat 的 SC-VAE decode，而非 raw GT mesh）：

| 投影 | F-score@0.01 | squared CD | fine/coarse token reduction |
|---|---:|---:|---:|
| 64³ -> 32³ -> nearest | 0.9446 | `2.694e-5` | 4.50× |
| 64³ -> 16³ -> nearest | 0.8834 | `5.297e-5` | 20.22× |

最重要的效率结果来自 1024³：1.3B 官方 SLat denoiser、4 个真实 support、warm-cache、40 NFE 下，fine 64³ 全程 flat 与 RDD 8/12/20 相比，geometry-denoiser 前向平均加速 **1.402×**、中位 **1.392×**。512³ 旧 pilot 只有约 1.06×，说明固定模型/condition 开销在千级 token 时占主导；论文主效率表必须以 1024³ 为主，512³ 只做 scaling ablation。

这些结果支持继续做生成实验，但不能推出 RDD 的生成质量不降。特别是 decoder projection 只回答“平均后的 learned latent 是否仍有粗几何语义”。

通过门槛：

- 至少 20 个成功编码资产；
- 32³ SLat 中位 token 数与官方 512³ 量级相符（约千级，而非接近 32³ dense）；
- `tokens_8 + tokens_16 + tokens_32` 相比 `3*tokens_32` 至少下降 1.5 倍；
- 递归分解的重建误差和 sibling detail mean 小于 `1e-5`；
- 若 coarse token 极少（例如中位数 < 16），最粗层改为 16，不强行用 8。

运行与监控：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash scripts/rdd3d/run_setup_trellis2_tmux.sh
bash scripts/rdd3d/run_trellis2_pilot_tmux.sh
tmux attach -t rdd3d_pilot
tail -f logs/rdd3d_trellis2_pilot.log
```

输出：

```text
eval_results/rdd3d_pilot/ovoxel_512_sparsity.json
eval_results/rdd3d_pilot/ovoxel_1024_sparsity.json
eval_results/rdd3d_pilot/slat_32_pyramid.json
eval_results/rdd3d_pilot/slat_64_pyramid.json
eval_results/rdd3d_pilot/slat_decode_pyramid_1024.json
eval_results/rdd3d_pilot/forward_benchmark_1024.json
```

### Phase B：小规模 overfit / trainer smoke

在 32--128 个成功资产、每个 16 个随机 view 上，先验证训练目标能下降和采样能解码：

- backbone：主实验使用官方 `slat_flow_img2shape_dit_1_3B_1024_bf16`，512 只作低分辨率对照；
- 初始化：RDD 与 flat baseline 使用完全相同 pretrained geometry checkpoint；
- structure：pilot 中先用 ground-truth support 做受控验证；随后必须换成官方 structure model 预测 support；
- 优化器：AdamW，LR `1e-5`（fine-tune），weight decay `0.01`；
- CFG dropout：沿用官方 `0.1`；
- batch：2 GPUs，1/GPU；必要时 gradient accumulation；
- 训练：先 500-step feasibility gate，再 2K steps，保存 500/1K/2K；
- RDD stage sampling：1024 使用 16/32/64 等概率（512 对应 8/16/32），避免窄 coarse band 在训练中样本过少；
- flat baseline：相同数据、初始化、batch、LR、step 和随机种子。

最小命令：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion

# 生成 image conditions（初次 sparsity pilot 默认不渲染）
RDD3D_RENDER_COND=1 RDD3D_NUM_VIEWS=16 \
  bash scripts/rdd3d/prepare_trellis2_pilot_data.sh \
  2>&1 | tee logs/rdd3d_render_cond.log

# RDD 和匹配 flat baseline；默认 GPU 4,5，2K step
bash scripts/rdd3d/run_trellis2_finetune_tmux.sh rdd
bash scripts/rdd3d/run_trellis2_finetune_tmux.sh flat

tail -f logs/trellis2_rdd_1024_pilot_train.log
tail -f logs/trellis2_flat_1024_pilot_train.log
```

在正式 2K job 前必须各跑一次 `--tryrun`/单 batch smoke；小数据 overfit 只证明代码和目标可学，不作为论文质量表格。

#### 已完成的 1024 / 500-step feasibility run

32 个物体、每 GPU batch 1、2 GPUs/model、官方 1.3B 1024 checkpoint 初始化；由于 DINOv3 gated license 尚未被账号接受，本次使用 DINOv2-L/14 条件，仅验证 data/DDP/loss/sampler 路径。

| 指标 | RDD | flat | 解释 |
|---|---:|---:|---|
| 训练秒/step（10 个 50-step 窗口均值） | 1.264 | 2.215 | RDD 训练吞吐 1.75× |
| elapsed@500 | 600.2 s | 1058.2 s | 不含初始化/JIT 与最后 snapshot |
| 平均 tracked memory | 35.0 GiB | 37.7 GiB | 约少 2.7 GiB |
| 平均 input tokens | 4,524 | 11,136 | RDD 约少 2.46× |
| grad norm, step 50 -> 500 | 9.31 -> 0.97 | 1.27 -> 0.59 | RDD 初期 objective shift 明显但稳定收敛 |

RDD 的 band-normalized loss 从 `0.581` 降至 `0.212`；三个尺度分别：

- 16³ coarse：`0.875 -> 0.304`；
- 32³ middle：`0.463 -> 0.231`；
- 64³ fine：`0.313 -> 0.090`。

flat loss 从 `0.259` 降至 `0.175`。两种 loss 的 path、target 和权重不同，**数值不可横向比较**；只能说明两者各自优化稳定。RDD 的早期梯度明显更大，支持“从 flat pretrained objective 切换到 detail flow 需要适应”的担心，但 500 step 内已经降回 1 附近，不支持必须加 replay curriculum 的结论。是否需要 curriculum 必须由 held-out generation quality 决定。

本次保存的 EMA decay 为官方 `0.9999`。500 step 后 EMA 只有约 `1-0.9999^500=4.9%` 的新目标贡献，因此 RDD 的短程采样必须以 raw checkpoint 为主；EMA checkpoint只用于诊断。后续 5K decision run 两边匹配使用 `0.999`（5K step 后约 99.3% 适应），并同时报告 raw/EMA；这不是给 RDD 单独调参。

#### 500-step known-support 生成 gate

随后在 4 个训练内物体上进行了完全配对的 raw-checkpoint 评估：相同 image condition、ground-truth fine support、随机种子、官方 SC-VAE decoder、CFG=3 与 40 NFE。RDD 使用 coarse/middle/fine=`8/12/20`，flat 在 fine support 上运行 40 steps。该实验只检验小数据 overfit/采样可行性，**不是 held-out 泛化结果，也不能进入论文主质量表**。

| raw checkpoint，4 shapes | RDD | flat | RDD 相对结论 |
|---|---:|---:|---:|
| geometry sampling 秒/shape | 12.190 | 19.221 | **1.577x 加速** |
| peak allocated VRAM | 8.793 GiB | 8.872 GiB | 约少 0.08 GiB |
| normalized latent MSE | 1.196 | 0.418 | 更差；但不直接等于 mesh distance |
| F-score@0.01 | 0.9352 | 0.9964 | 绝对 -0.0612，相对约 -6.14% |
| squared Chamfer-L2 | `2.841e-5` | `9.096e-6` | RDD 约 3.12x 更高 |
| decode failures | 0/4 | 0/4 | 两边均数值稳定 |

因此 500-step gate 的结论是：**稀疏采样速度收益成立，但质量门槛暂未通过**。这不是可以被 EMA 掩盖的结论：旧 `0.9999` EMA 的单样本结果中 RDD F-score 曾略高于 flat，但 500 steps 后该 EMA 约 95.1% 仍是原始 flat vector field，而且样本数只有 1，故只保留作 staleness 诊断。raw 结果显示 objective switch 仍需更多适配。

下一步不是直接扩到正式 10K 数据，而是在同一 32-shape pilot 上从同一官方 checkpoint 重新进行匹配的 2K-step RDD/flat gate，统一使用 EMA=`0.999` 并同时评估 raw/EMA。只有 RDD 的 F-score 退化缩小至 3% 内，才下载/预处理 10K 训练集；若 2K 后仍明显落后，应优先测试短 flat-to-RDD replay curriculum 或放弃把 3D 作为主结果，而不是用更多算力隐藏方法的收敛代价。

可审计输出：

```text
eval_results/rdd3d_pilot/train_1024_500step_metrics.json
eval_results/rdd3d_known_support/rdd_1024_legacy500rawdecode.json
eval_results/rdd3d_known_support/flat_1024_legacy500rawdecode.json
results_rdd3d/trellis2_rdd_1024/ckpts/denoiser_ema0.9999_step0000500.pt
results_rdd3d/trellis2_flat_1024/ckpts/denoiser_ema0.9999_step0000500.pt
```

### Phase C：论文规模 fine-tune

建议先做 10K 资产、5K optimizer step 的决策实验；只有 RDD 在固定 compute 下不明显劣于 flat 才扩到 50K、20K step 主实验。预训练 1.3B 模型的 paired fine-tune 不需要一开始就烧 50K step；先比较 held-out 曲线再决定是否延长。

数据：

- 训练：ObjaverseXL Sketchfab/GitHub + ABO 的高质量 shape 子集；几何实验不要求 PBR 完整；过滤损坏、极端尺度和重复资产；
- 每资产 16 views，FoV 10°--70°，随机光照，遵循 TRELLIS.2 配置；
- 10K 决策阶段：5K optimizer steps；
- 50K 主阶段：20K optimizer steps起，根据 held-out curve 早停；
- 4--8 张 98GB GPU，global batch 尽量 32；不能达到时用 accumulation 保持有效 batch；
- RDD 与 flat 必须使用相同数据顺序、有效 batch、优化器步数和增强。

建议 curriculum（只在 pretrained flat -> RDD 转换不稳定时启用，并作为明确消融）：前 2K step 以 50% flat replay + 50% RDD；2K--5K 线性降 flat 到 0；之后纯 RDD。若不需要则使用纯 RDD，避免引入不必要复杂度。

### Phase D：评估与论文表格

#### D1. 受控 geometry 评估

对每个条件，RDD 和 flat 使用**同一个 predicted fine support**、同一个 SC-VAE decoder 和同一随机种子。这样差异只来自 geometry flow。

- Toys4K-PBR/shape 子集或严格去重的 ObjaverseXL holdout；
- Mesh Distance、CD、F-score、normal-map PSNR/LPIPS；
- TRELLIS.2 协议可采用 100 outer-shell views、1M points，但早期实验可先用 100K points；最终表使用完整协议；
- 每项报告均值、bootstrap 95% CI，至少 3 个训练 seed（资源不足时，主模型 1 seed + 小模型 3 seed）。

#### D2. End-to-end image-to-3D

- 相同官方 structure model；只替换 geometry flow；
- 100 个训练集外 image prompt；
- CLIP-N、ULIP-2、Uni3D，以及至少 30 人的 blinded pairwise user study（如果投稿篇幅允许）；
- 定性图专门选 thin/open/non-manifold/internal structure，而不是只选简单玩具。

#### D3. 效率主表

每个方法报告：

- structure / geometry / decode 分段 latency；
- geometry peak VRAM；
- 每尺度 active token 中位数与 P95；
- NFE、token-NFE、attention proxy；
- 输出失败率（空 mesh、断裂、NaN）；
- 512³ 为主，若资源允许再做 1024³/64-latent scaling。

#### D4. 必要消融（保持在 5 个以内）

1. flat FM；
2. RDD multires，但不用 detail projection；
3. RDD projector，但普通全空间 noise filling；
4. 完整 RDD detail-subspace noise filling；
5. 完整 RDD 的时间宽度匹配 8/12/20 与 fine-heavy 8/8/24。

不要把大量表示/solver sweep 塞进主线。最能支撑论文的是 projector、noise filling 和实际 sparse compute 三件事。

## 8. 数据集的 sparse 性质应如何判断

“3D 表面在 R³ 网格中只占 O(R²)”只是渐近直觉，不能直接当实验结论。实际收益受以下因素影响：

- SC-VAE downsampling 是否把邻近表面扩张成较厚 active band；
- 多部件、内部结构、毛发/树叶会增加 active token；
- 复杂资产可能填满许多 coarse cells，使 8³ 几乎不稀疏；
- 8-child candidate expansion 会在薄壳边界制造大量无效 child；
- 全局 attention 的 padding/packing 与 GPU kernel 利用率可能吞掉理论收益。

因此统计必须逐资产报告 median/P5/P95，而不是只给总体平均。至少分四类：solid/simple、thin/open、multi-part、high-genus/internal。若只有简单 ShapeNet chair/car，无法证明方法对现代 3D asset generation 的价值。

## 9. 继续或放弃的客观标准

满足以下条件才值得作为 RDD 论文的 3D 主实验：

1. 真实 1024 SLat 上 geometry token-NFE 至少降低 1.6 倍、geometry denoiser latency 至少降低 1.3 倍；end-to-end latency 必须单独报告，不能把 geometry speedup 当作整条 pipeline speedup；
2. 等 NFE 下几何质量不低于 flat 超过 3%，或等 token-NFE 下显著优于 flat；
3. projector/noise-filling ablation 能解释收益，而不是仅靠更多训练；
4. 预测 support 下仍成立，不只在 ground-truth support 下成立；
5. 至少在一个复杂拓扑测试集上展示细节/连通性优势或不退化。

若真实数据统计显示 8/16/32 token 几乎不降，或 20K-step 10K 资产实验仍比 flat 质量差很多，则应放弃把 RDD-3D 作为主结果。可以把 exact decomposition 作为分析/附录，而不要为了故事继续烧算力。

## 10. 代码地图

```text
RDD_DiT_3D/slat_pyramid.py
    dense-zero 与 fixed-support 两种稀疏金字塔；精确重建/候选浪费分析
RDD_DiT_3D/analyze_ovoxel_sparsity.py
    真实 .vxz O-Voxel 层级稀疏统计
RDD_DiT_3D/analyze_slat_pyramid.py
    真实 SLat .npz 的 8/16/32 token 与候选浪费统计
RDD_DiT_3D/trellis2_rdd/hierarchy.py
    递归 active-child projector、nearest expand、detail decomposition
RDD_DiT_3D/trellis2_rdd/rdd_sparse_flow_matching.py
    官方 TRELLIS.2 trainer 上的 RDD global-time flow objective
RDD_DiT_3D/trellis2_rdd/rdd_sampler.py
    known-support piecewise Euler + CFG sampler
RDD_DiT_3D/evaluate_trellis2_known_support.py
    相同 support/condition/decoder 的 RDD-vs-flat 受控生成评估
scripts/rdd3d/
    环境、确定性 train/val 子集、checkpoint、512/1024 config、tmux 训练/评估入口
tests/test_slat_pyramid.py
tests/test_trellis2_rdd_hierarchy.py
tests/test_trellis2_rdd_flow.py
tests/test_rdd3d_subset.py
    数学不变量与非均匀 child-count 回归测试
```

## 11. 主要外部依据

- TRELLIS.2: `https://arxiv.org/abs/2512.14692`, `https://github.com/microsoft/TRELLIS.2`
- XCube: `https://arxiv.org/abs/2312.03806`
- OctFusion: `https://arxiv.org/abs/2408.14732`
- ShapeShifter: `https://openaccess.thecvf.com/content/CVPR2025/html/Maruani_ShapeShifter_3D_Variations_Using_Multiscale_and_Sparse_Point-Voxel_Diffusion_CVPR_2025_paper.html`
- CADD: `https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Large_Scene_Generation_with_Cube-Absorb_Discrete_Diffusion_ICCV_2025_paper.html`
- Neural Wavelet-domain Diffusion: `https://arxiv.org/abs/2209.08725`
- Direct3D-S2: `https://arxiv.org/abs/2505.17412`
- LATTICE: `https://openaccess.thecvf.com/content/CVPR2026/html/Lai_LATTICE_Democratize_High-Fidelity_3D_Generation_at_Scale_CVPR_2026_paper.html`
- LATO: `https://arxiv.org/abs/2603.06357`
- Grow3D: DOI `10.1145/3820379`
