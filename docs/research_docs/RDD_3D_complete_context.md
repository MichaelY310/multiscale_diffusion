# RDD 3D 项目完整上下文文档

> **目的**：本文档包含 RDD 3D 项目所有调研、理论分析、代码实现、实验结果的完整记录。
> 新的 agent 只需阅读此文档即可完全了解项目全貌，无需任何先前对话上下文。
>
> **最后更新**: 2026-08-02

---

## 目录

1. [项目总览与目标](#1-项目总览与目标)
2. [RDD 理论基础与核心机制](#2-rdd-理论基础与核心机制)
3. [RDD 的 4 个硬性要求](#3-rdd-的-4-个硬性要求)
4. [8 种 3D Representation 详细调研](#4-8-种-3d-representation-详细调研)
5. [多分辨率 3D 生成方法调研（竞品分析）](#5-多分辨率-3d-生成方法调研竞品分析)
6. [GaussianCube 深入分析](#6-gaussiancube-深入分析)
7. [RDD 稀疏性理论加速分析](#7-rdd-稀疏性理论加速分析)
8. [核心问题与未解决矛盾](#8-核心问题与未解决矛盾)
9. [代码实现状态与文件路径](#9-代码实现状态与文件路径)
10. [RDD-Wan 视频生成系统](#10-rdd-wan-视频生成系统)
11. [RDD 2D 实验配置](#11-rdd-2d-实验配置)
12. [下一步建议](#12-下一步建议)
13. [Set Transformer AE 配置实验与结果](#13-set-transformer-ae-配置实验与结果)

---

## 1. 项目总览与目标

### 1.1 核心研究问题

**RDD (Resolution Drift Diffusion)** 是一种多分辨率扩散模型框架。项目目标是将 RDD 从 2D 图像扩展到 **3D 数据**，利用 3D 数据天然的稀疏性实现高效生成。

### 1.2 核心 Idea

RDD 的 **constant (nearest-neighbor) upsampling** 性质允许在 3D 中实现 **sparse diffusion**：
- 空 voxel 的 drift = 0（因为 UP(DOWN(0)) - 0 = 0）
- 空 block 在 forward/reverse 过程中始终保持零
- 因此可以跳过空区域的计算，实现加速

### 1.3 项目范围

- **RDD 2D**: 已完成，用 DiT 在 ImageNet 256×256 上训练，使用 `sd-vae-ft-ema` (32×32×4 latent)
- **RDD-Wan 视频**: 已完成代码实现，基于 Wan 2.1 架构，支持 Flow Matching + 时空 drift
- **RDD 3D**: 当前重点，寻找合适的 3D representation 来证明多分辨率+稀疏生成的有效性

### 1.4 用户背景

- 杨远 (yangyuan.335)，UCSD CS 学生
- 目标投稿 NeurIPS/ICLR 级别会议
- 资源路径：`/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/`
- 本地开发路径：`/home/tiger/multiscale_diffusion/`

---

## 2. RDD 理论基础与核心机制

### 2.1 RDD 核心公式（DDPM 版）

**Drift 定义**:
```
d^c = UP(DOWN(x_0)) - x_0
```
- `DOWN`: 2×2 average pooling (2D) 或 2×2×2 average pooling (3D)
- `UP`: nearest-neighbor upsampling (constant interpolation)
- drift 表示原信号与低通滤波版本的差异（即高频成分）

**Forward 过程**:
```
x_t = sqrt(α_t) * (x_0 + κ * drift) + sqrt(1-α_t) * ε
```

**Sparsity Theorem**: 若一个 2×2×2 block 全零，则：
- drift d = 0
- forward 过程中该区域仍为零（加上噪声除外）
- reverse 过程中可以跳过

### 2.2 RDD 核心公式（Flow Matching 版）

用于 RDD-Wan 视频和未来的 FM 3D 版本：

**时间轴**: t ∈ [0, 1]，t=0 是噪声，t=1 是数据

**FM 中的 RDD 扩展** (per-stage drift):
```
Forward: x_t = t*x_1 + (1-t)*ε + f(t)*d
where f(t) = t*(s_{k+1} - t)/L_k,  L_k = s_{k+1} - s_k
```

**Velocity target**:
```
v = (x_1 - ε) + f'(t)*d
where f'(t) = (s_{k+1} - 2t)/L_k
```

**关键性质**:
- f(s_{k+1}) = 0 → 保证 ODE 端点无 drift 残留
- 每个 stage 都运行 sigma ∈ [0, 1] 的完整 ODE（FM 版不切分时间轴）
- Stage 之间通过 noise_filling (ratio=0.7) 衔接

### 2.3 多阶段配置

**DDPM 版（2D）**: 切分 timestep 轴，如 [0-300, 300-600, 600-1000]

**FM 版（视频/未来 3D）**: 每个 stage 独立运行 t ∈ [0,1]，仅分辨率不同
- 配置格式: `空间下采样,时间下采样,推理步数`
- 示例: `4,2,20;2,1,20;1,1,10` → 粗/中/细三阶段，总 50 步

### 2.4 RDD 的核心优势（vs PyramidalWan 等竞品）

| 特性 | PyramidalWan | RDD |
|------|-------------|-----|
| 阶段过渡 | Corrective noise (renoising) | **连续 drift ODE** |
| ODE 连续性 | ❌ 被打断 | ✅ 全程连续 |
| 高阶 solver | ❌ 不兼容 | ✅ DPM-Solver-2/3 |
| 理论加速 | 4.5x (多分辨率) | **9-18x** (多分辨率 + 高阶 solver) |

---

## 3. RDD 的 4 个硬性要求

任何 3D representation 要与 RDD 兼容，必须满足：

1. **N³ grid 结构**: 数据必须映射到规则的 N×N×N 网格
2. **Per-voxel 显式属性**: 每个 voxel 存储固定维度的特征向量（无 MLP decoder 查询）
3. **天然稀疏信号**: 大部分 voxel 应为空（零向量）
4. **Constant upsample 兼容**: nearest-neighbor 上采样必须有物理意义

---

## 4. 8 种 3D Representation 详细调研

### 4.1 分类总览

| 方法 | 类型 | 通道数 | 需要预训练模型 | 天然稀疏 | Average Pool 合理 |
|------|------|--------|---------------|---------|-----------------|
| GaussianCube | Explicit | 14 | ❌ | ❌ (100% dense) | ⚠️ 物理意义不清 |
| Plenoxels | Explicit | 28 | ❌ | ✅ | ⚠️ SH 系数可以，密度不行 |
| DVGO | Explicit | 13 | ❌ | ✅ | ⚠️ 同上 |
| Direct3D-S2 | Latent | 8 | ✅ VAE | ✅ | ✅ latent average |
| SCube | Latent | 16 | ✅ VAE | ✅ | ✅ |
| LATO | Latent | 8 | ✅ VAE | ✅ | ✅ |
| DiT-3D (Point Cloud) | Explicit | 3 | ❌ | N/A (非grid) | N/A |
| NSVF | Explicit | ~32 | ❌ | ✅ | ⚠️ |

### 4.2 各方法详解

#### GaussianCube (Zhang et al., 2024)
- **数据**: 32×32×32 网格，每个 voxel 恰好 1 个 Gaussian
- **14 通道**: xyz_offset(3) + SH_DC(3) + opacity(1) + scale(3) + rotation(4)
- **构建流程**: 原始 3D mesh → Gaussian fitting (32768个) → Optimal Transport 排列到网格
- **问题**: OT 后是 **100% dense**，所有 voxel 都有 Gaussian → 稀疏性=0
- **归一化**: 用训练集的 `mean.pt` + `std.pt` (per-voxel, shape=(32,32,32,14))

#### Plenoxels (Yu et al., CVPR 2022)
- **数据**: 稀疏体素网格，每个 voxel 存储 SH 系数 + 密度
- **28 通道**: SH DC(3) + SH degree-1(9) + SH degree-2(15) + density(1)
- **稀疏性**: 好，只在有内容的区域有 voxel
- **问题**: SH 系数的 average pool 虽然是合法的低通滤波，但密度的平均物理意义需验证

#### DVGO (Sun et al., CVPR 2022)
- **数据**: 密度网格 + 颜色特征网格
- **13 通道**: density(1) + color features(12)
- **稀疏性**: 好，类似 Plenoxels

#### Direct3D-S2 (2024)
- **数据**: 稀疏体素 latent (由 3D sparse VAE 编码)
- **8 通道**: learned latent features
- **稀疏性**: 好
- **优势**: latent space 中 average pool 物理意义清晰
- **缺点**: 需要预训练 sparse VAE

#### SCube (NVIDIA, NeurIPS 2024)
- **数据**: 稀疏体素 + VoxSplats
- **16 通道**: latent features
- **优势**: NVIDIA 出品，质量高，配合 Gaussian Splatting 渲染
- **缺点**: 需要预训练模型，代码不完全开源

#### LATO (2024)
- **数据**: Mesh 顶点坐标体素化后由 VAE 编码
- **8 通道**: latent vectors
- **特点**: 两阶段解耦 — 几何(RDD负责) + 拓扑(TopologySiTFlow 独立模型负责)
- **面生成**: 对已生成顶点预测连接概率 → 阈值化 → 三角面

### 4.3 结论

**最适合 RDD 的方法**（按优先级排序）:
1. **TSDF/Occupancy Grid** — average pool 完美合理，天然稀疏，数据现成
2. **Direct3D-S2 / SCube** — latent 空间 average pool OK，但需要预训练 VAE
3. **Plenoxels/DVGO** — explicit + 稀疏，但 SH average 不完美

**不适合的**:
- GaussianCube — 100% dense，且 14ch 物理参数 average 无物理意义

---

## 5. 多分辨率 3D 生成方法调研（竞品分析）

### 5.1 完整列表

| 方法 | 会议 | Representation | 多分辨率策略 | 稀疏？ |
|------|------|---------------|-------------|--------|
| **XCube** | CVPR 2024 | Sparse voxel + TSDF/neural features | 层级 cascade latent diffusion | ✅ VDB |
| **SCube** | NeurIPS 2024 | Sparse voxel + latent features | 两层 coarse→fine | ✅ |
| **OctFusion** | CGF 2024 | Octree + SDF latent | **单模型多尺度 U-Net** | ✅ Octree |
| **Fractal3DGen** | ICLR 2025 | Sparse SDF grid | Fractal 分治 + 自适应剪枝 | ✅ |
| **Grow3D** | TOG 2025 | Octree + VQ-VAE latent | Next-scale autoregressive | ✅ Octree |
| **PDD** | ECCV 2024 Oral | Discrete semantic voxel | 3-level 金字塔 discrete diffusion | ✅ |
| **ShapeShifter** | 2025 | Point cloud + sparse voxel | 多尺度 SinDDM 3D | ✅ |
| **TRELLIS/TRELLIS-2** | CVPR 2025 | SLAT (sparse voxel + latent) / O-Voxel | 两阶段: 结构→特征 | ✅ |
| **UltraShape** | 2025 | SDF (VecSet + sparse voxel) | Coarse-to-fine 两阶段 | ✅ |
| **ArchComplete** | 2024 | Dense voxel (Minecraft) | VQ-VAE + 层级 upsampling | ❌ Dense |

### 5.2 关键竞品详解

#### XCube (NVIDIA, CVPR 2024) — 最直接竞争者
- **Representation**: Sparse voxel + TSDF + neural kernel features
- **多分辨率**: 2 层 hierarchy (64³→256³)，每层独立 sparse VAE + latent diffusion
- **稀疏实现**: VDB 数据结构
- **关键**: 不是 binary occupancy，是**连续 TSDF + learned latent**
- **分辨率**: 最高 1024³
- **和 RDD 的区别**: XCube 每层是独立模型，RDD 是统一连续 ODE

#### OctFusion (北大, 2024) — 最相似理念
- **Representation**: Octree + 每个节点存 SDF latent（由 octree-based VAE 编码）
- **关键创新**: **单个 diffusion 模型处理所有 octree 层级** — 和 RDD 单模型理念一致！
- **区别**: OctFusion 无 drift 概念，层间没有数学连续过渡，不能用高阶 solver
- **分辨率**: 256³~1024³

#### Fractal3DGen (ICLR 2025) — 稀疏剪枝最相似
- **Representation**: Sparse SDF grid
- **关键创新**: **自适应剪枝**（77.6% sub-block 被跳过）— 和 RDD 稀疏跳过非常类似！
- **区别**: 基于 transformer 分治，不是 ODE 框架

#### TRELLIS (Microsoft, CVPR 2025 Spotlight)
- **Representation**: SLAT = Sparse Structured Latent
- **关键**: 用 **Rectified Flow** transformer（不是 DDPM）
- **两阶段**: 先生成哪些 voxel 存在（结构），再生成每个 voxel 的 latent（特征）

#### PDD (ECCV 2024 Oral)
- **Representation**: **Discrete semantic voxel** (每个 voxel 是类别标签)
- **多分辨率**: 3-level 金字塔 (32²×4 → 64²×8 → 256²×16)
- **扩散类型**: Discrete diffusion (D3PM)，不是连续的
- **场景**: 户外大场景（LiDAR 点云）

### 5.3 RDD 3D 的差异化定位

| | XCube | OctFusion | Fractal3DGen | RDD 3D (ours) |
|---|---|---|---|---|
| 模型数量 | 多模型 cascade | 单模型 | 单模型 | **单模型** |
| 层间数学 | 独立生成 | 无连续保证 | 分治拼接 | **Drift ODE 连续** |
| 高阶 solver | ❌ | ❌ | ❌ | **✅ DPM-Solver** |
| 稀疏机制 | VDB 工程实现 | Octree | 剪枝 | **理论保证(drift=0)** |
| 训练 | 每层分开 | 共享权重 | 共享权重 | **统一 loss** |

**RDD 的核心 novelty 不在于"稀疏"（XCube/Fractal3DGen 都做了），而在于**：
1. ODE 连续性 → 兼容高阶 solver → 更少步数
2. Sparsity theorem 提供理论保证（不是启发式剪枝）
3. 单一统一模型 + 统一 loss（训练效率）

### 5.4 最主流的 Representation 选择

**不全是 occupancy grid！** 绝大多数方法用 **TSDF/SDF** 或 **learned latent**：
- TSDF/SDF: XCube, SCube, OctFusion, Fractal3DGen, UltraShape
- Learned Latent: TRELLIS, Grow3D, OctFusion (latent on octree nodes)
- Discrete categories: PDD
- Point cloud: ShapeShifter

**SDF 是最主流选择**，原因：
- 天然稀疏（只在表面 ± truncation 范围有值）
- Average pool 物理意义清晰（粗糙距离场）
- 可 Marching Cubes 提取 mesh
- 大量现成数据

---

## 6. GaussianCube 深入分析

### 6.1 数据结构

```
每个样本: (32, 32, 32, 14) float tensor
通道布局:
  [0:3]   xyz_offset — voxel 网格中心的偏移
  [3:6]   SH_DC — 球谐函数0阶系数(≈RGB颜色)
  [6:7]   opacity — 不透明度(logit 空间)
  [7:10]  scale — 各向异性缩放(log 空间)
  [10:14] rotation — 旋转(单位四元数 wxyz)
```

### 6.2 数据归一化

```python
# 原版 GaussianCube 用训练集统计量归一化（不是硬编码范围！）
volume = (volume - mean_volume_act) / std_volume_act  # mean/std shape = (32,32,32,14)
volume = dynamic_clip(volume, p=0.995)  # 按 99.5% 分位数 clip
```

### 6.3 代码关键文件

- **`/home/tiger/multiscale_diffusion/GaussianCube/`** — 原版代码
  - `dataset/dataset.py` — 数据加载 (.pt 文件)
  - `train.py` — 训练循环 (FP16 + DDP + EMA + render loss)
  - `model/unet.py` — 3D UNet
  - `gaussian_renderer/__init__.py` — Gaussian 渲染器 (需要 `diff_gaussian_rasterization` CUDA)
  - `configs/shapenet_uncond.yml` — 配置 (image_size=32, in_channels=14, out_channels=28)

### 6.4 关键发现

- `mean.pt` shape=(32,32,32,14)，dtype=float64
- **所有 32768 个 voxel 的 mean opacity 都 > 0.04**（range: 0.042~0.960）
- **GaussianCube 是 100% dense** → RDD 稀疏机制完全无法利用
- HuggingFace 仓库 (`BwZhang/GaussianCube-ShapeNetCar-v1.0`) **不提供 volume_act 训练数据**

### 6.5 14ch 物理参数 Average Pool 问题

**核心矛盾**: RDD 要求 2×2×2 average pool，但 Gaussian 物理参数：
- **Rotation (quaternion)**: 四元数不能线性平均！
- **Scale (log-space)**: log 空间的平均 ≠ 真实缩放的平均
- **Position offset**: 多个 Gaussian 位置的平均 ≠ 合理的粗 Gaussian

**解决思路**:
1. **直接用 average，接受不完美** — 2D RDD 中 pixel RGB 也不完美但 work
2. **学 small encoder: 14ch → latent → 14ch** — 增加复杂度
3. **只对 latent representation 用 RDD** — 放弃 GaussianCube 转向 Direct3D-S2/SCube

---

## 7. RDD 稀疏性理论加速分析

### 7.1 分析方法

计算 RDD sparse diffusion 对比 full-space diffusion 的理论加速比。

**计算公式**:
```
full_space_cost = sum(R_k³) for all stages k  (每个stage处理全部voxel)
rdd_cost = sum(exposed_voxels_k) for all stages k
speedup = full_space_cost / rdd_cost
```

**2×2×2 全暴露问题**: 只要 8 个 child 中有 1 个 occupied，parent 就必须展开全部 8 个 children。

### 7.2 几何模型结果

#### 紧凑实心球体 (radius=0.35)

| 分辨率 | 占据率 | 稀疏加速 | 多分辨率加速 | 总加速 |
|--------|--------|---------|-------------|--------|
| 32³ | 18% | 4.21x | 3.5x | **14.7x** |
| 64³ | 18% | 4.85x | 4.38x | **21.2x** |

#### 薄壳模型 (厚度=1.5 voxels)

| 分辨率 | 占据率 | 稀疏加速 | 多分辨率加速 | 总加速 | 2×2×2 浪费 |
|--------|--------|---------|-------------|--------|-----------|
| 32³ | 8.3% | 5.09x | 3.5x | **17.8x** | 37-66% |
| 64³ | 7.2% | 7.76x | 4.38x | **34.0x** | 37-55% |

#### 汽车形状薄壳

| 分辨率 | 占据率 | 稀疏加速 | 多分辨率加速 | 总加速 |
|--------|--------|---------|-------------|--------|
| 32³ | 4.3% | 10.83x | 3.5x | **37.9x** |
| 64³ | 2.5% | 19.11x | 4.38x | **83.6x** |

### 7.3 真实 ShapeNet Car 点云结果

从 GaussianCube 示例数据中的渲染图反推 3D 点云分布：

| 配置 | 占据率 | 稀疏加速 | 多分辨率加速 | 总加速 | + DPM-Solver |
|------|--------|---------|-------------|--------|-------------|
| 32³ | 36.9% | 1.02x | 3.5x | **3.56x** | 7.1x |
| 64³ | 6.0% | 2.14x | 4.38x | **9.36x** | **18.7x** |

### 7.4 2×2×2 Block 统计 (ShapeNet Car 64³, Stage 5)

| 占据数/8 | Block 数 | 占比 | 每 block 浪费 |
|----------|---------|------|-------------|
| 0/8 (空) | ~60% | — | 0 (跳过) |
| 1/8 | ~31% | — | 7 ← **最大浪费源** |
| 2/8 | ~7% | — | 6 |
| ≥3/8 | <2% | — | 较少 |

### 7.5 核心结论

1. **GaussianCube 32³ (dense) → RDD 稀疏加速 = 0**（数据100%填满）
2. 稀疏加速只有在 **高分辨率 (≥64³) + 真正稀疏表示** 下才有显著优势
3. **薄壳/表面物体**的 2×2×2 浪费在高分辨率下仍有 30-50%
4. 结合 DPM-Solver-2 后总加速可达 **18x+**

### 7.6 GaussianCube Opacity 分布分析

```json
{
  "threshold": 0.1, "empty": 1, "occupied": 32767, "occ_ratio": 0.99997
}
```
→ 即使用 opacity < 0.1 作为"空"的阈值，几乎所有 voxel 都是 occupied。

---

## 8. 核心问题与未解决矛盾

### 8.1 GaussianCube + RDD 的三个问题

1. **100% dense 数据** → RDD sparse 机制完全无法利用
2. **14ch 物理参数 average** 无明确物理意义（四元数、log scale）
3. **当前代码的归一化** 用硬编码范围而非真实统计量（已在 `gaussian_cube.py` 中标记待修复）

### 8.2 根本矛盾

- **天然稀疏的 explicit 表示** (Plenoxels, DVGO): 做不了有意义的 average pool
- **能做 average 的 latent 表示** (Direct3D-S2, SCube): 需要预训练 VAE，数据获取难
- **GaussianCube**: 既不稀疏，average 也没意义

### 8.3 推荐解决方案

**最佳选择: TSDF/Occupancy Grid**
- Average pool = 粗糙距离场/占据概率 → 物理意义完美
- 天然稀疏（只在表面附近有值）
- 数据现成（ShapeNet TSDF 大量存在）
- 可与 XCube, OctFusion, Fractal3DGen 直接对比

---

## 9. 代码实现状态与文件路径

### 9.1 RDD 3D 统一框架

路径: `/home/tiger/multiscale_diffusion/RDD_3D_unified/`

```
RDD_3D_unified/
├── models/
│   ├── dit3d.py          — DiT3D_RDD (全稀疏 3D DiT, 3D RoPE, SingleStreamBlock)
│   ├── unet3d.py         — 3D UNet
│   ├── sparse_unet3d.py  — SparseConv UNet (MinkowskiEngine)
│   └── __init__.py       — MODEL_REGISTRY
├── representations/
│   ├── base.py           — BaseRepresentation (ABC: encode/decode/per_voxel_channels)
│   ├── gaussian_cube.py  — 14ch, explicit (有归一化问题待修)
│   ├── plenoxels.py      — 28ch, explicit
│   ├── dvgo.py           — 13ch, explicit
│   ├── direct3d_s2.py    — 8ch, latent
│   ├── scube.py          — 16ch, latent
│   └── lato.py           — 8ch, latent
├── diffusion_3d.py       — 核心 sparse diffusion (drift计算, sparse coords操作)
├── dataset.py            — SparseVolumeDataset + sparse_collate_fn
├── train.py              — DDP + EMA 训练脚本
├── sample.py             — 采样脚本 (coarse-to-fine)
├── preprocess.py         — 原始数据→sparse .pt 预处理
├── test_smoke.py         — 冒烟测试 (已通过全部6项)
└── test_full_pipeline.py — 完整 pipeline 测试
```

**关键实现细节** (`diffusion_3d.py`):
- 操作在 `[B, N, C]` 稀疏 token 张量 + `[B, N, 3]` 坐标张量上
- `compute_drift_sparse()`: 通过 scatter_add 实现 sparse mean pooling → upsample → drift
- `upsample_coords()`: 每个 low-res coord → 8 个 high-res children
- 支持 padding mask 处理变长序列

**模型** (`dit3d.py`):
- 3D RoPE: Centralized Interpolative Positional Encoding
- SingleStreamBlock: 并行 Attention + MLP (PaLM 风格)
- QK Norm (per-head RMSNorm)
- adaLN-Zero 调制
- 支持 padding mask

### 9.2 GaussianCube 原版代码

路径: `/home/tiger/multiscale_diffusion/GaussianCube/`

主要用于参考数据格式和渲染逻辑，不直接用于 RDD 训练。

### 9.3 RDD-Wan 视频代码

路径: `/home/tiger/multiscale_diffusion/rdd_video/`

```
rdd_video/
├── flow_matching_video.py    — 核心 FM + drift 模块 (支持 pixel/latent 下采样模式)
├── train_rdd_wan.py          — 训练脚本
├── train_rdd_wan.sh          — 8卡启动脚本 (STAGE_CONFIGS="4,2,20;2,1,20;1,1,10")
├── sample_rdd_wan.py         — 采样 (支持 DPM-Solver-2)
├── generate_training_data.py — 合成数据生成
├── download_model.sh         — Wan 2.1 模型下载
└── test_rdd_video_cpu.py     — CPU 逻辑正确性测试 (已通过)
```

### 9.4 RDD 2D 代码

路径: `/home/tiger/multiscale_diffusion/2d/`

- `train_RDD_imagenet256.py` — DDPM 版训练
- `train_RDD_FM_imagenet256.py` — FM 版训练
- `RDD_sample.py` — 采样
- 实验配置: `exp/` 目录下的 `.jsonl` 文件

### 9.5 下载的数据

```
/home/tiger/multiscale_diffusion/.gc_download/
├── mean.pt           — (32,32,32,14) float64, 训练集均值
├── std.pt            — (32,32,32,14) float64, 训练集标准差
├── example_data/     — 渲染图片 (150 views/object)，不是 volume 数据
└── example_data.zip  — 28MB

/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/GaussianCube_ShapeNet/
├── mean.pt
├── std.pt
├── shapenet_car_ckpt.pt  — 315MB 预训练权重
└── example_data.zip
```

**注意**: 没有实际的 `volume_act/*.pt` 训练数据！需要从 GaussianCube_Construction 工具链构建。

### 9.6 分析脚本与结果

```
/home/tiger/multiscale_diffusion/rdd_sparsity_analysis.py  — 稀疏性分析主脚本
/home/tiger/multiscale_diffusion/rdd_detailed_analysis.py  — 详细分析脚本

/home/tiger/multiscale_diffusion/research_experiments/
├── rdd_sparsity_analysis/
│   ├── rdd_sparsity_analysis_results.json  — 几何模型分析结果
│   └── rdd_detailed_analysis.json          — ShapeNet Car 详细分析
├── deep_theory_results.json                — 深度理论分析
├── rdd_allocation_results.json             — 步数分配
└── rdd_optimal_allocation_results.json     — 最优分配
```

### 9.7 其他重要文件

```
/home/tiger/multiscale_diffusion/RDD_research_compendium.md — 视频生成调研+最优分配理论
/home/tiger/multiscale_diffusion/docs/theory/               — 理论推导脚本集合
/home/tiger/multiscale_diffusion/DiT/                       — DiT 原版 + RDD 改造
/home/tiger/multiscale_diffusion/Wan2.1/                    — Wan 2.1 原版代码
```

---

## 10. RDD-Wan 视频生成系统

### 10.1 系统概要

基于 Wan 2.1 (1.3B) 微调，不修改模型架构，仅修改 Loss 和采样逻辑。

### 10.2 关键设计

- **Sigma 约定**: σ=1 为噪声，σ=0 为数据（Wan 2.1 标准）
- **预测目标**: v = ε - x_0（Wan 2.1 标准）
- **下采样模式A (Pixel)**: 像素空间下采样 → VAE encode（PyramidalWan 风格）
- **下采样模式B (Latent)**: 直接在 latent 空间 avg-pool
- **阶段过渡**: noise_filling (ratio=0.7)
- **训练策略**: 随机均匀采样 stage (1/3 概率) 计算 Loss

### 10.3 对比 PyramidalWan

| 配置 | PyramidalWan (CVPR 2026) | RDD-Wan |
|------|--------------------------|---------|
| 基座 | Wan 2.1-1.3B | Wan 2.1-1.3B |
| 训练步数 | 5K | 5K |
| GPU | 2×H100 | 8×A100 |
| 每GPU batch | 6 | 6 |
| 采样步数 | 20-20-10 (50步) | 20-20-10 / **10-10-5 (25步)** |
| 阶段过渡 | Haar wavelet corrective noise | **Drift ODE** |
| DPM-Solver | ❌ | ✅ |
| 理论加速 | 4.5x vs vanilla | **9-12x vs vanilla** |

### 10.4 数据策略

- **PoC 阶段**: FastVideo 600K 预编码 Latent + Latent avg-pool 下采样
- **正式实验**: OpenVID-1M 子集 100K + 像素空间下采样后 VAE encode

---

## 11. RDD 2D 实验配置

### 11.1 已有配置文件

- `imagenet256_unifiedXL300300400_400000_908080step_ddpm_cfg115.jsonl`:
  - model_type: "unified", model: "DiT-XL/2"
  - stage_lengths: "300,300,400", resolutions: "32,16,8"
  - ckpt: `.../RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/0400000.pt`
  - num_steps: "80,80,90", sampler: "ddpm"
  - 两行: cfg_scale=1.0 和 cfg_scale=1.5

- `imagenet256_multistage300300400_400000_908080step_ddpm_cfg115.jsonl`:
  - model_type: "multistage" (非unified)
  - 400000 步 checkpoint

### 11.2 2D 功率谱参数

- VAE: `stabilityai/sd-vae-ft-ema` (32×32×4 latent)
- 拟合参数: A=13910, β=1.54
- 验证脚本: `docs/theory/exp1_fit_power_spectrum.py`

### 11.3 训练脚本

- `train_RDD_imagenet256_100_300_600_original.sh`:
  - RESULTS_DIR: `/mnt/bn/.../RDD/ImageNet_original/RDD-Multistage-100-300-600`

---

## 12. 下一步建议

### 12.1 推荐的 3D 实验路径

**首选: TSDF/Occupancy Grid on ShapeNet**

原因：
1. Average pool 物理意义完美
2. 天然稀疏 (5-15% 占据率 at 64³)
3. 数据现成 (ShapeNet voxelized data)
4. 可直接对比 XCube/OctFusion/Fractal3DGen
5. 无需预训练 VAE
6. 32³→64³→128³ scaling 实验

### 12.2 实验设计建议

1. **Baseline**: Full-resolution DiT3D on 64³ TSDF (同参数量，无多分辨率)
2. **RDD 3D**: 多分辨率 (8→16→32→64) sparse DiT3D
3. **Ablation**:
   - 有/无 drift
   - 有/无 sparse pruning
   - Euler vs DPM-Solver-2
   - 步数分配策略
4. **Metrics**: IoU, Chamfer Distance, F-Score, 生成 FID (Precision/Recall)
5. **对比方法**: 复现 XCube 的 cascade + 独立模型作为对照

### 12.3 关键差异化 Story

**不要说 "RDD 比 XCube 更稀疏"**（XCube 也做了稀疏）。
**要说**:
- "RDD 通过连续 drift ODE 实现层间数学连续 → 兼容高阶 solver → 同等质量少 2x 步数"
- "单一统一模型 vs 多个独立模型 → 训练效率更高"
- "Sparsity theorem 提供理论保证 vs 启发式剪枝"

### 12.4 待解决的工程问题

1. 获取 ShapeNet TSDF 64³ 数据集
2. 修改 `diffusion_3d.py` 支持 TSDF 的 truncation 特性
3. 评估 pipeline (Marching Cubes 提取 mesh → 质量指标)
4. 考虑是否需要 sparse conv (MinkowskiEngine) 还是标准 attention + mask 就够

---

## 附录 A: 项目环境

- 主机: tiger
- Python: 3.10
- PyTorch: 2.1+
- 工作目录: `/home/tiger/multiscale_diffusion/`
- 存储路径: `/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/`
- HTTP Proxy: `export HTTP_PROXY=http://bj-rd-proxy.byted.org:3128`（下载需设置）
- HuggingFace 镜像: `http://huggingface-proxy-sg.byted.org/`

## 附录 B: 2D 功率谱与最优分配理论

详见 `/home/tiger/multiscale_diffusion/RDD_research_compendium.md` Part II-V。

核心公式：
- 训练分配: $L_k \propto \sqrt{E_k \times W_k}$
- 推理分配 (Euler): $n_k \propto L_k \sqrt{R_k}$
- 推理分配 (Heun): $n_k \propto L_k R_k^{1/3}$

其中 $R_k$ 是模型曲率（drift norm 作为代理）。

## 附录 C: 视频生成论文完整对比

详见 `/home/tiger/multiscale_diffusion/RDD_research_compendium.md` Part I。

包含 Wan 2.1, PyramidalWan, TPDiff, Pyramid Flow, CogVideoX, Open-Sora 等 13 种方法的训练配置、评估结果和技术特点完整对比。

---

## 13. Set Transformer AE 配置实验与结果

本章记录了 Set Transformer Autoencoder 的全部配置实验。目标是找到最优的 3D 点集 → 固定维度 embedding 的编码方案，后续可作为 RDD 3D 系统中 voxel 内部细节的编码器。

### 13.1 实验概述

共进行了 **三大系列 + 一组早期探索** 实验：

| 系列 | 脚本 | K_MAX | 总实验数 | 目标 |
|------|------|-------|---------|------|
| V1 系列 (exp1-9) | `train_emd_ae_3d.py` | 10 | 9 | concat pooling 基线，消融 depth/width/bottleneck |
| K50 系列 (bn32-256) | `train_emd_ae_3d.py` | 50 | 4 | 验证 K=50 下的可扩展性 |
| V2 系列 (exp1-8) | `train_emd_ae_3d_v2.py` | 10 | 8 | 新架构：PMA pooling / AR vs Parallel / KL / 加宽 |
| 早期探索 (4组) | 多种脚本 | 10 | 4 | 直接EMD / mean pool / concat / embedding 对比 |

### 13.2 模型架构

#### AEBridgeReconModel (V1)

```
输入: [B, K_MAX, 4]  — 每个点 = [x, y, z, k/K_MAX]，最多K_MAX个点，带mask

  ┌─────────────────────────────────┐
  │  1. SetEncoder                  │
  │     Linear(4 → d) + ReLU       │
  │     N× MaskedSAB(d, d, H头)    │  ← 集合内自注意力 (masked)
  │     输出: [B, K_MAX, d]         │
  └──────────────┬──────────────────┘
                 ↓
  ┌─────────────────────────────────┐
  │  2. SetAutoencoder (AE Bridge)  │
  │                                 │
  │  Encode:                        │
  │    1× MaskedSAB(d, d, H头)     │
  │    pooling → [B, pooled_dim]    │  ← concat: K_MAX×d / mean: d
  │    MLP(pooled_dim→...→bn_dim)   │  ← 多层MLP压缩
  │    输出: [B, bn_dim] ← EMBEDDING│
  │                                 │
  │  Decode:                        │
  │    MLP(bn_dim→...→K_MAX×d)      │
  │    reshape → [B, K_MAX, d]      │
  │    1× MaskedSAB(d, d, H头)     │
  │    输出: [B, K_MAX, d] (memory) │
  └──────────────┬──────────────────┘
                 ↓
  ┌─────────────────────────────────┐
  │  3. Transformer Decoder (AR)    │
  │     BOS token + pos_embed       │
  │     N层 TransformerDecoderLayer │
  │     cross-attend to memory      │
  │     value_head → σ(Linear→3)    │
  │     end_head → σ(Linear→1)      │
  └─────────────────────────────────┘

输出: 按字典序逐个生成3D点，直到 end > 0.5 停止
Loss: MSE(预测点 vs 真值点) + BCE(end signal)
```

#### AEBridgeReconModelV2 (V2 新增特性)

- 支持 AR decoder 和 Parallel decoder 两种模式
- 支持 PMA pooling (pma1/pma4) — 置换不变
- 支持 max_mean pooling
- 支持 VAE 模式 (mu + logvar → reparameterization, KL loss)
- use_k_feature: 是否把 k/K_MAX 作为第4维输入
- sort_input: 是否对输入点做字典序排序

### 13.3 V1 系列实验结果 (K_MAX=10, 5M steps)

**共用参数**: batch_size=64, lr=1e-4, ae_pool=concat, ae_num_heads=4, ae_num_mlp_layers=3, ae_mlp_factor=4

| 实验 | dim_hidden | layers(enc/dec) | bottleneck | 参数量 | **最终 val_emd** | loss | 备注 |
|------|-----------|-----------------|-----------|--------|-----------------|------|------|
| v1_exp1 | 32 | 4/4 | 128 | 242K | **0.000672** | 1e-6 | 🥇 V1最优 |
| v1_exp2 | 32 | 4/4 | 64 | 226K | 0.000798 | 1e-6 | |
| v1_exp3 | 32 | 4/4 | 32 | 218K | 0.001083 | 1e-6 | |
| v1_exp4 | 32 | 2/2 | 256 | 233K | 0.000769 | 1e-6 | 浅层但大瓶颈 |
| v1_exp5 | 32 | 2/2 | 128 | 200K | 0.000821 | 1e-6 | |
| v1_exp6 | 32 | 2/2 | 64 | 184K | 0.000910 | 2e-6 | |
| v1_exp7 | **16** | 4/4 | 128 | 70K | 0.015413 | 47e-6 | ❌ d=16太窄 |
| v1_exp8 | **16** | 4/4 | 64 | 62K | ~0.041 | ~1e-3 | ❌ 提前终止(400K步) |
| v1_exp9 | 32 | 2/2 | 32 | 175K | 0.001181 | 2e-6 | |

**V1 关键结论**:
1. **dim_hidden=32 是最优选择** — d=16 的 exp7/exp8 效果差一个数量级
2. **4层 vs 2层差异很小** — exp1(4层/bn128) 0.000672 vs exp5(2层/bn128) 0.000821，2层已足够
3. **bottleneck 128 最优** — 比 256/64/32 都好（在 4 层 d=32 配置下）
4. **所有 d=32 实验都达到了 sub-0.001 的极低 EMD** — 任务本身不难

### 13.4 K50 系列实验结果 (K_MAX=50, ~1.3M steps, 被中止)

**共用参数**: batch_size=256, lr=1e-4, dim_hidden=32, layers=4/4, ae_num_mlp_layers=4, ae_mlp_factor=8, 4 GPU 并行

| 实验 | bottleneck | 训练步数 | **最终 val_emd** | loss | 备注 |
|------|-----------|---------|-----------------|------|------|
| K50_bn128 | 128 | ~1.29M | **~0.085** | ~0.008 | 🥇 K50最优 |
| K50_bn256 | 256 | ~1.30M | ~0.085 | ~0.016 | 与bn128持平 |
| K50_bn64 | 64 | ~1.30M | ~0.088 | ~0.009 | |
| K50_bn32 | 32 | ~1.29M | ~0.116 | ~0.017 | 明显更差 |

**K50 关键结论**:
1. K=50 难度远大于 K=10 — concat 展平后 1600 维 → bottleneck，信息压缩比高
2. ~1.3M 步仍未充分收敛（对比 V1 的 5M 步），被提前终止
3. bn128 和 bn256 效果持平，bn32 明显不足

### 13.5 V2 系列实验结果 (K_MAX=10, 新架构)

**共用参数**: batch_size=64, lr=3e-4, weight_decay=1e-4, dim_hidden=32, layers=4/4

| 实验 | Pool | Decoder | bn | kl_weight | d | 步数 | **val_emd** | 备注 |
|------|------|---------|-----|-----------|---|------|------------|------|
| v2_exp1 | concat | ar | 256 | 0 | 32 | 1.65M | **0.0024** | 🥇 V2最优，大瓶颈 |
| v2_exp7 | pma1 | ar | 64 | 0 | **40** | 906K | 0.0098 | 加宽模型，PMA最优 |
| v2_exp3 | pma4 | ar | 64 | 0 | 32 | 1.56M | 0.0156 | 4-seed PMA |
| v2_exp8 | max_mean | ar | 64 | 0 | 32 | 910K | 0.0168 | max+mean拼接 |
| v2_exp2 | pma1 | ar | 64 | 0 | 32 | 1.56M | 0.0171 | PMA基线 |
| v2_exp4 | pma1 | ar | 32 | 0 | 32 | 1.57M | 0.0302 | 最小瓶颈 |
| v2_exp6 | pma1 | ar+VAE | 64 | 0.001 | 32 | 875K | ~0.150 | ❌ KL正则有害 |
| v2_exp5 | pma1 | **parallel** | 64 | 0 | 32 | 1.26M | 0.186 | ❌ Parallel decoder极差 |

**V2 关键结论**:
1. **Concat pooling 仍然最优** — v2_exp1(concat) 0.0024 >> v2_exp2(pma1) 0.0171
2. **AR decoder 远优于 Parallel decoder** — v2_exp5(parallel) 0.186 vs v2_exp2(ar) 0.017
3. **KL 正则化有害** — v2_exp6 加入 kl_weight=0.001 后 val_emd 恶化至 0.15
4. **加宽模型(d=40)有帮助** — v2_exp7(d=40) 0.0098 vs v2_exp2(d=32) 0.0171
5. **PMA pooling 虽然效果稍差但具有置换不变性** — 这在实际3D应用中很重要

### 13.6 早期探索实验结果

这是最早期的 4 组对比实验，验证了不同编码方案的基本可行性：

| 方案 | 脚本 | 最佳 val_emd | 训练步数 | 配置 |
|------|------|-------------|---------|------|
| 🥇 AE + concat pool | `train_emd_ae_3d.py` | **0.000355** | ~8.5M | d=32, bn=256, concat |
| 🥈 Fused 3D Embedding | `train_3d_embedding.py` | 0.000642 (val_loss) | 100K | 冻结1D encoder + fused 256d |
| 🥉 AE + mean pool | `train_emd_ae_3d.py` | 0.130 | 57K | d=256, bn=256, mean |
| 4th DETR-style (Hungarian) | `train_emd_3d.py` | 0.288 | 36K | d=128, H=8, 并行decoder |

**早期实验关键结论**:
- Concat pooling 在充分训练后达到近乎完美的重建 (val_emd=0.000355)
- Mean pooling 丢失太多信息，即使 dim_hidden=256 也只能到 0.13
- 并行 decoder + Hungarian matching 收敛慢，但训练步数不足（仅 36K）
- Fused Embedding（两阶段方案）在少量步数内也能达到极低 loss

### 13.7 架构设计分析与问题

#### 已确认的设计问题

1. **Concat Pooling 破坏置换不变性** — 依赖字典序排序，不是真正的 set encoder
2. **Bottleneck 维度不对称** — d=32, K=10 时 flatten=320 → bn=256，几乎无压缩
3. **AR Decoder 推理效率低** — 生成 k 个点需要 k 次 forward
4. **k/K_MAX 作为第4维特征** — 全局标量混入 per-element 属性，不优雅

#### 推荐的改进方向

```
变长3D点集 → N×MaskedSAB → PMA(1 seed) → MLP压缩 → embedding
                                ↑
                         置换不变，无需排序
```

- 用 PMA (Pooling by Multihead Attention) 替代 concat — 天然置换不变
- 用 DETR-style 并行 decoder + Hungarian loss — 无需排序假设
- 加宽 dim_hidden (≥64) 配合真正的压缩瓶颈

### 13.8 与 RDD 3D 的关系

Set Transformer AE 在 RDD 3D 系统中的定位：

```
3D mesh / point cloud
       ↓ 预处理
  N×N×N voxel grid
       ↓ 每个 occupied voxel
  内部顶点集合 (变长, 最多 K_MAX 个点)
       ↓ Set Transformer AE Encode
  固定维度 embedding (32~256 维)
       ↓ 作为 voxel 特征
  RDD 3D Diffusion (DiT3D_RDD)
       ↓ 生成 embedding
  Set Transformer AE Decode
       ↓
  重建 3D 顶点 → mesh
```

**当前结论**: 对于 vertex-level 编码，AE + concat (bn=128, d=32, 4层) 已充分验证可行，val_emd=0.000672。但 concat pooling 的置换不变性问题仍需解决——PMA pooling (d=40) 可达到 val_emd=0.0098，精度比 concat 差 10 倍但仍在可接受范围。

**最终推荐**: 若后续采用 TSDF/Occupancy Grid 路径（见第12章），则 Set Transformer AE 可能不再需要，因为每个 voxel 的值就是标量 SDF/occupancy 而非变长点集。Set Transformer 仅在 mesh vertex 路径或 Gaussian Splatting 路径下才是必要组件。

### 13.9 实验文件路径

> **注意**: 代码已合并至统一 repo `/home/tiger/multiscale_diffusion/RDD_DiT_3D/`

```
/home/tiger/multiscale_diffusion/RDD_DiT_3D/
├── model.py                    # DiT3D_RDD 核心模型
├── diffusion_3d.py             # 稀疏 3D 扩散
├── dataset.py                  # SparseVolumeDataset
├── train.py                    # Vertex 版 DDP 训练
├── sample.py                   # 多阶段渐进采样
├── sample_to_glb.py            # 端到端采样→GLB
├── preprocess_mesh_to_sparse.py # 数据预处理
├── set_transformer_ae/         # Set Transformer AE 模块
│   ├── __init__.py
│   ├── masked_attention.py     # MaskedMAB/SAB/PMA
│   ├── ae_model.py             # V1 AEBridgeReconModel (原 train_emd_ae_3d.py)
│   ├── ae_model_v2.py          # V2 AEBridgeReconModelV2 (原 train_emd_ae_3d_v2.py)
│   └── infer.py                # 推理/加载工具
└── face/                       # Face 面片版
    ├── face_set_transformer.py # FaceSetTransformerAE
    ├── train_face_set_transformer.py # Face AE 训练
    ├── train_face_diffusion.py # Face 扩散模型训练 (复用 vertex train.py)
    ├── face_sample.py          # Face 采样
    ├── face_dataset.py         # FaceSparseDataset
    ├── generate_cell_data.py   # mesh → cell 预处理
    └── preprocess_meshanything.py # MeshAnything 预处理

日志和指标 (已整理):
/home/tiger/multiscale_diffusion/logs/
├── set_transformer_v1/         # V1 log_v1_exp[1-9].txt
├── set_transformer_v2/         # V2 log_exp[1-8].txt
├── set_transformer_K50/        # K50 log_K50_bn*.txt
├── set_transformer_csv/        # 所有 metrics CSV
└── rdd_dit_3d/                 # RDD 训练日志

实验脚本:
/home/tiger/multiscale_diffusion/scripts/set_transformer/
├── run_v1_exp[1-9].sh          # → ae_model.py
├── run_K50_bn[32|64|128|256].sh # → ae_model.py
└── run_exp[1-8].sh             # → ae_model_v2.py
```
