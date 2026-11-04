# RDD-Wan 项目完整文档

> **目的**：本文档为自包含的项目状态记录，使任何新接手的 AI agent 或研究者无需额外上下文即可完全理解项目全貌。

---

## 1. 项目概述

### 1.1 什么是 RDD

RDD（Rectified Drift Diffusion）是一种多分辨率扩散方法，其核心创新在于向 Flow Matching 的前向过程中注入"drift"（漂移项）。这使得模型在不同分辨率阶段之间的过渡保持 ODE 连续性，从而兼容高阶 ODE 求解器（如 DPM-Solver、Heun），在更少的采样步数下实现更好的生成质量。

### 1.2 RDD-Wan：从 2D 到视频

本项目将已有的 2D RDD 方法扩展到视频生成领域，使用 **Wan 2.1** 作为骨干模型（backbone）：

- **RDD-Wan**：基于 Wan 2.1-1.3B 进行微调，采用 RDD 的连续 ODE 多分辨率方案
- **对标方法**：PyramidalWan（arXiv:2601.04792），同样基于 Wan 2.1-1.3B，但使用 corrective noise 过渡（打断 ODE 连续性）

### 1.3 核心优势（RDD vs PyramidalWan）

| 特性 | PyramidalWan | RDD-Wan (Ours) |
|------|-------------|----------------|
| 分辨率过渡 | Corrective noise（打断 ODE） | 连续 ODE（无 renoising） |
| 兼容高阶求解器 | ❌ 只能用 Euler | ✅ DPM-Solver, Heun 等 |
| 少步采样质量 | 受限 | 更优（Pareto 曲线优势） |
| 训练数据/流程 | 80K 合成视频, 5K steps | 相同（公平对比） |

---

## 2. 数学基础（RDD Flow Matching）

### 2.1 时间轴定义

- 时间轴 `t ∈ [0, 1]`：**t=0 为纯噪声，t=1 为干净数据**
- K 个阶段将 [0,1] 分为 K 段：`[s_0, s_1], [s_1, s_2], ..., [s_{K-1}, s_K]`
- 边界条件：`s_0 = 0`, `s_K = 1`

### 2.2 阶段划分

对齐 PyramidalWan 的时间边界：

```
Stage 0: [0, 0.06]      长度 L_0 = 60/1000 = 0.06    分辨率 ÷4/÷4
Stage 1: [0.06, 0.41]   长度 L_1 = 350/1000 = 0.35   分辨率 ÷2/÷2
Stage 2: [0.41, 1.0]    长度 L_2 = 590/1000 = 0.59   分辨率 ÷1/÷1
```

### 2.3 Drift 信号

对于阶段 k，区间为 `[s_k, s_{k+1}]`，长度 `L_k = s_{k+1} - s_k`：

**Drift 定义：**
```
d = Upsample(Downsample(x_1)) - x_1
```
这是将干净数据下采样再上采样与原始数据的差异，编码了"被分辨率丢失的高频信息"。

**Drift 包络函数：**
```
f(t) = t · (s_{k+1} - t) / L_k
```

**关键性质：**
- `f(s_k) = s_k · (s_{k+1} - s_k) / L_k = s_k`（注意：当 s_k=0 时 f(0)=0）
- `f(s_{k+1}) = s_{k+1} · (s_{k+1} - s_{k+1}) / L_k = 0`
- 即在每个阶段的**终点** f 归零，保证阶段边界处 drift 消失

### 2.4 前向过程（Forward Process）

在阶段 k 内，时刻 t ∈ [s_k, s_{k+1}]：

```
x_t = t · x_1 + (1 - t) · ε + f(t) · d
```

其中：
- `x_1`：干净数据（VAE latent）
- `ε ~ N(0, I)`：标准高斯噪声
- `d = Upsample(Downsample(x_1)) - x_1`：drift 信号
- `f(t)`：drift 包络

### 2.5 速度目标（Velocity Target）

网络学习的条件速度场：

```
u_t = (x_1 - ε) + [(s_{k+1} - 2t) / L_k] · d
```

推导来源：对 x_t 关于 t 求导，即 `dx_t/dt`。

### 2.6 Tweedie 公式（预测干净数据）

从 x_t 和预测的速度 v_θ 恢复 x_1：

```
pred_x1 = x_t + (1 - t) · v_θ(x_t, t)
```

### 2.7 ODE 采样（Euler 方法）

在阶段 k 内，从 s_k 积分到 s_{k+1}：

```
dt = L_k / num_steps
for i in range(num_steps):
    t_cur = s_k + i * dt
    v = model(x_t, t_cur)
    x_t = x_t + dt * v
```

### 2.8 阶段过渡（Stage Transition）

当从低分辨率阶段 k 过渡到高分辨率阶段 k+1 时：

```
x_new = s_{k+1} · Upsample(pred_x1) + (1 - s_{k+1}) · ε_new
```

其中 `ε_new ~ N(0, I)` 是新采样的噪声，`s_{k+1}` 是新阶段的起始时间。

**这是 RDD 的核心设计**：这个过渡公式本身就是 flow matching 直线插值的形式 `x_t = t·x_1 + (1-t)·ε`，因此 ODE 路径保持连续。

---

## 3. PyramidalWan 配置（对标基线）

### 3.1 论文信息

- 论文：arXiv:2601.04792
- 方法：对 Wan 2.1-1.3B 进行多分辨率金字塔微调

### 3.2 分辨率阶段

```
Stage 0: ÷4/÷4（时空同比下采样4倍）
Stage 1: ÷2/÷2（时空同比下采样2倍）
Stage 2: ÷1/÷1（原始分辨率）
```

空间和时间的下采样因子始终同步。

### 3.3 时间边界

基于 σ（sigma）的划分，映射到 t 轴：
```
σ 空间: [0.94, 1.0] → [0.59, 0.94] → [0, 0.59]
t 空间: [0, 0.06] → [0.06, 0.41] → [0.41, 1.0]
```

阶段长度（1000步制）：`[60, 350, 590]`

### 3.4 下采样策略

**关键设计**：在像素空间（pixel space）进行三线性插值下采样，然后通过 VAE 编码为 latent。

**不是** latent 空间的 avg-pool！这一点已被 RepLDM（NeurIPS 2025）和 PyramidalWan 论文验证。

### 3.5 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Wan 2.1-1.3B（微调目标） |
| 数据生成模型 | Wan 2.1-14B |
| 训练数据 | 80K 合成视频（14B 生成） |
| 分辨率 | 448×832, 81 frames |
| 训练步数 | 5,000 |
| 硬件 | 2×H100 |
| Batch size | 6/GPU（每阶段2个样本） |
| 学习率 | 1e-5 |
| 优化器 | AdamW |
| 架构修改 | 无（3D RoPE 天然支持变长序列） |

### 3.6 过渡机制

PyramidalWan 使用 "corrective noise"（等价于 renoising），这会**打断 ODE 连续性**，导致：
- 无法使用高阶求解器
- 少步采样时质量下降明显

---

## 4. RDD-Wan 实现（当前状态）

所有代码位于 `/home/tiger/multiscale_diffusion/rdd_video/`

### 4.1 核心模块：`flow_matching_video.py`

完全重写以对齐 2D FM 版本，包含以下类：

#### `SpatioTemporalDownsample`
- 5D 平均池化：`nn.AvgPool3d(kernel_size=(t_factor, s_factor, s_factor))`
- 输入形状：`(B, C, T, H, W)`

#### `SpatioTemporalUpsample`
- 最近邻重复插值：`F.interpolate(..., mode='nearest')`
- 因子：`scale_factor=(t_factor, s_factor, s_factor)`

#### `VideoFlowMatchingWithDrift`
单阶段 RDD Flow Matching：
- `compute_drift(x_1)`: 计算 `d = Upsample(Downsample(x_1)) - x_1`
- `sample_xt(x_1, eps, t)`: 前向采样 `x_t = t·x_1 + (1-t)·ε + f(t)·d`
- `velocity_target(x_1, eps, t)`: 计算目标速度 `u_t`
- `predict_x1(x_t, v, t)`: Tweedie 公式恢复 x_1

#### `RDDVideoSchedule`
多阶段调度器：
- 默认配置：`stage_lengths=[60, 350, 590]`, `factors=[(4,4), (2,2), (1,1)]`
- `upsample_with_noise(pred_x1, t_boundary, noise)`: 阶段过渡
  - `t_boundary * upsampled + (1 - t_boundary) * noise`
- `sample_training_batch(x_1)`: 为训练随机采样阶段和时间步

### 4.2 训练脚本：`train_rdd_wan.py`

支持多种数据模式：
- `multires`：读取预计算的多分辨率 latent（推荐）
- `pixel`：在线 VAE 编码（慢但灵活）
- `single`：单分辨率（用于 ablation）
- `dummy`：随机数据（调试用）
- `distill`：蒸馏模式

**MultiResLatentDataset**：读取预计算的分阶段 latent 数据。

默认超参：
```python
global_batch_size = 24
learning_rate = 1e-5
max_steps = 5000
```

### 4.3 Latent 预计算：`precompute_latents.py`

两种模式：
1. **pixel 模式**（推荐）：pixel-space 三线性下采样 → VAE 编码
2. **latent 模式**：全分辨率 VAE 编码 → latent-space avg-pool

支持多 GPU 分片处理。

### 4.4 数据生成脚本

- `generate_data_machine1.sh`：从 index 0 开始，8-GPU Ulysses SP
- `generate_data_machine2.sh`：从 index 100000 开始，无重叠

特性：
- 无限循环生成，支持断点续跑
- Ctrl+C 安全停止
- 自动删除 .lock 文件以避免 HuggingFace 下载卡死

### 4.5 测试：`test_rdd_video_cpu.py`

9 个单元测试，全部通过：
- 下采样/上采样形状验证
- Drift 计算正确性
- 前向采样数值验证
- 速度目标计算
- 阶段过渡连续性
- 边界条件 f(t)=0

### 4.6 训练启动：`train_rdd_wan.sh`

封装训练启动命令，配置 DDP/多GPU 参数。

---

## 5. 关键设计决策

### 5.1 像素空间下采样 > Latent 空间 avg-pool

**结论**：在像素空间做三线性插值下采样，再 VAE 编码，效果显著优于直接在 latent 空间做平均池化。

**证据**：
- RepLDM（NeurIPS 2025）实验验证
- PyramidalWan 论文采用此策略

### 5.2 预计算 Latent

训练时的瓶颈是 VAE 编码，预计算后：
- 训练瓶颈变为 IO（可通过 NVMe/内存缓存解决）
- 显存节省巨大（无需 VAE 前向）

### 5.3 模型选择

- **微调模型**：Wan 2.1-1.3B（与 PyramidalWan 相同，公平对比）
- **数据生成模型**：Wan 2.1-14B（与 PyramidalWan 相同）

### 5.4 全参数微调

不使用 LoRA 或蒸馏，全参数微调：
- 1.3B 参数量可控
- 更好地学习多分辨率 drift 分布

### 5.5 RDD 核心优势

连续 ODE → 兼容高阶求解器 → 在相同采样步数下质量更好，或在相同质量下需要更少步数。

---

## 6. 2D RDD 参考实现

### 6.1 核心文件

- **Ground Truth 2D FM 版本**：`/home/tiger/multiscale_diffusion/diffusion/flow_matching_with_drift.py`
- **2D 统一训练脚本**：`/home/tiger/multiscale_diffusion/2d/train_RDD_FM_imagenet256_unified.py`

### 6.2 当前训练状态

- 模型：DiT-XL/2
- 数据集：ImageNet 256×256
- Checkpoint 路径：`/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Unified-XL-v2/`

### 6.3 Video 版本与 2D 版本的对应关系

| 2D 概念 | Video 对应 |
|---------|-----------|
| `Downsample2D` (avg-pool) | `SpatioTemporalDownsample` (3D avg-pool) |
| `Upsample2D` (nearest) | `SpatioTemporalUpsample` (3D nearest) |
| `FlowMatchingWithDrift` | `VideoFlowMatchingWithDrift` |
| `RDDSchedule` | `RDDVideoSchedule` |
| 空间因子 (2,4) | 时空因子 (t,s)=(4,4),(2,2),(1,1) |

---

## 7. 方法对比表

| 方法 | Backbone | 分辨率变化 | 时间划分 | 过渡机制 | 核心创新 |
|------|----------|-----------|---------|---------|---------|
| **PyramidalWan** | Wan 2.1-1.3B | ÷4→÷2→÷1 (时空同比) | σ [0.94,1],[0.59,0.94],[0,0.59] | Corrective noise（打断ODE） | 多分辨率微调 |
| **Pyramid Flow** | SD3/CogVideo | ÷4→÷2→÷1 | 均匀分段 | Gaussian noise reset | 联合图像-视频训练 |
| **TPDiff** | Custom U-Net | ÷4→÷2→÷1 (仅空间) | 等分 | 直接上采样 | 时间金字塔 |
| **RDD-Wan (Ours)** | Wan 2.1-1.3B | ÷4→÷2→÷1 (时空同比) | [0,0.06,0.41,1.0] | 连续 ODE（无中断） | Drift 注入 + ODE 连续性 |

**核心差异总结**：
- PyramidalWan 的 corrective noise 等价于在阶段边界"重置"噪声，导致 ODE 路径不连续
- Pyramid Flow 使用完全的高斯噪声重置，同样不连续
- TPDiff 仅做空间金字塔，不做时间金字塔
- **RDD-Wan** 通过 drift 包络的边界条件（f(boundary)=0）保证路径连续

---

## 8. 训练流水线（逐步）

### Step 1: 下载模型

```bash
# Wan 2.1-14B（数据生成用）
HF_ENDPOINT=https://hf-mirror.com hf download Wan-AI/Wan2.1-T2V-14B \
  --local-dir /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video2/Wan2.1-T2V-14B/

# Wan 2.1-1.3B（微调用）
HF_ENDPOINT=https://hf-mirror.com hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video/Wan2.1-T2V-1.3B/
```

**注意**：恢复下载前必须删除 .lock 文件：
```bash
rm -f <path>/.cache/huggingface/download/*.lock
```

### Step 2: 生成合成训练数据

使用 8-GPU Ulysses Sequence Parallelism：
```bash
# Machine 1
bash generate_data_machine1.sh  # 从 index 0 开始

# Machine 2
bash generate_data_machine2.sh  # 从 index 100000 开始
```

目标：生成 2000+ 合成视频（448×832, 81 frames）。

### Step 3: 预计算多分辨率 Latent

```bash
python precompute_latents.py \
  --mode pixel \
  --video-dir /path/to/generated_videos \
  --output-dir /path/to/latents \
  --model-path /path/to/Wan2.1-T2V-1.3B
```

pixel 模式流程：
1. 读取原始视频
2. 对每个阶段的分辨率因子，在像素空间做三线性下采样
3. 将下采样后的视频通过 VAE 编码为 latent
4. 保存每个阶段的 latent

### Step 4: 运行 RDD-Wan 微调

```bash
python train_rdd_wan.py \
  --data-mode multires \
  --latent-dir /path/to/latents \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --max-steps 5000 \
  --lr 1e-5 \
  --global-batch-size 24
```

### Step 5: VBench 评估

使用 946 个标准 prompt 评估生成质量，与基线对比。

---

## 9. 评估策略

### 9.1 主要指标

**VBench**：视频生成领域标准评估基准，包含 946 个 prompt，覆盖多维度质量评估。

### 9.2 对比方法

| 方法 | 评估来源 |
|------|---------|
| Wan 2.1 原始（全步基线） | 自行复现 |
| PyramidalWan | 自行复现（如有 checkpoint）或引用论文 |
| RDD-Wan (Ours) | 自行评估 |
| 其他方法 | 直接引用论文结果 |

### 9.3 公平对比原则

- 相同评估协议（同 prompt、同分辨率、同指标）
- 相同模型大小（1.3B）
- 相同训练数据规模和来源

### 9.4 核心论点

**RDD 在 Pareto 曲线上优于 PyramidalWan**：
- 在相同采样步数下，RDD 质量更高
- 在相同质量下，RDD 需要更少步数
- 原因：ODE 连续性允许高阶求解器

---

## 10. 当前状态与待办事项

### ✅ 已完成

- [x] RDD-Video 核心模块重写并验证（`flow_matching_video.py`）
- [x] 配置对齐 PyramidalWan（时空 ÷4/÷2/÷1，时间边界 [0,0.06,0.41,1.0]）
- [x] 9 个单元测试全部通过
- [x] Latent 预计算脚本（`precompute_latents.py`）
- [x] 训练脚本 + MultiResLatentDataset（`train_rdd_wan.py`）
- [x] 双机数据生成脚本（`generate_data_machine1/2.sh`）

### 🔄 进行中

- [ ] 下载 Wan 2.1-14B 模型（用于数据生成）
- [ ] 安装 Wan 依赖到独立 venv（`/home/tiger/multiscale_diffusion/rdd_video/venv` 或 `/home/tiger/multiscale_diffusion/2d/wan`）

### 📋 待完成

- [ ] 生成 2000+ 合成视频
- [ ] 预计算多分辨率 latent
- [ ] 运行 RDD-Wan 微调（5K steps）
- [ ] VBench 评估
- [ ] Pareto 曲线实验（RDD 少步 vs PyramidalWan 多步）

---

## 11. 环境配置

### 11.1 2D RDD 实验环境

- 路径：`/home/tiger/multiscale_diffusion/2d/2d/`（Python 3.12, uv venv）
- 关键依赖：
  - numpy 2.2.6
  - scipy 1.14.0
  - diffusers 0.38.0
  - torch（需检查版本）

### 11.2 Video RDD 环境

**核心冲突**：Wan 要求 `numpy<2`，但 2D 环境需要 `numpy>=2` → **必须使用独立 venv**

**方案 A**：`/home/tiger/multiscale_diffusion/rdd_video/venv`（Python 3.10）

**方案 B**：`/home/tiger/multiscale_diffusion/2d/wan`（uv venv）

**必需依赖**：
```
torch >= 2.4
flash_attn
easydict
xfuser          # 14B Ulysses SP 推理必需
wan             # pip install -e Wan2.1/
numpy < 2       # Wan 硬性要求
```

### 11.3 模型路径

| 模型 | 路径 | 状态 |
|------|------|------|
| Wan 2.1-1.3B | `/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video/Wan2.1-T2V-1.3B/` | 下载未完成 |
| Wan 2.1-14B | `/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video2/Wan2.1-T2V-14B/` | 下载未完成 |

### 11.4 下载命令

```bash
# 使用镜像站加速
HF_ENDPOINT=https://hf-mirror.com hf download Wan-AI/Wan2.1-T2V-14B \
  --local-dir /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video2/Wan2.1-T2V-14B/

# 恢复下载前必须删除 lock 文件
rm -f /path/to/.cache/huggingface/download/*.lock
```

---

## 12. 硬性约束（项目记忆）

### 12.1 时间轴配置（不可更改）

```
Stage 0: [0, 0.06]    长度 60    分辨率 ÷4/÷4（时空）
Stage 1: [0.06, 0.41]  长度 350   分辨率 ÷2/÷2（时空）
Stage 2: [0.41, 1.0]   长度 590   分辨率 ÷1/÷1（原始）
```

### 12.2 技术约束

- **时空下采样因子必须同步**：spatial 和 temporal 使用相同倍率
- **强烈偏好像素空间下采样**：不使用 latent avg-pool
- **不下载完整数据集**：磁盘空间有限
- **不修改 Wan 架构**：3D RoPE 天然处理变长序列

### 12.3 沟通约束

- 用户对逻辑错误和偏离 RDD 设计原则极为敏感
- 用户偏好中文沟通，直接且以结果为导向
- 需要精确到公式级别的正确性

---

## 13. Wan 2.1 架构笔记

### 13.1 模型架构

Wan 2.1 是基于 DiT（Diffusion Transformer）的视频生成模型：

- **3D RoPE**：支持可变长度 token 序列（关键！使得无需架构修改即可处理多分辨率）
- **Flow Matching**：`t∈[0,1]`，前向 `x_t = t·x_1 + (1-t)·ε`，速度目标 `v = x_1 - ε`

### 13.2 VAE 压缩率

```
temporal × height × width = 4 × 8 × 8
```

即：
- 时间维度压缩 4 倍（81 frames → ~20 latent frames）
- 空间维度各压缩 8 倍（448×832 → 56×104 latent）

### 13.3 模型规格

| 参数 | 1.3B | 14B |
|------|------|-----|
| 隐藏维度 | 1536 | 5120 |
| 注意力头数 | 12 | 40 |
| 层数 | 30 | 40 |
| 推理要求 | 单GPU | 8-GPU Ulysses SP |

### 13.4 Ulysses Sequence Parallelism

14B 模型推理必须使用 Ulysses SP：
- `ulysses_size` 必须整除 `num_heads`
- 14B 有 40 个头 → 合法值：1, 2, 4, 5, 8
- 推荐：`ulysses_size=8`（使用全部 8 GPU）

---

## 14. 为什么不用 Wan 2.2

### 14.1 Wan 2.2 概况

- 发布时间：2025年7月28日（极新，仅5天前）
- 架构：MoE（Mixture of Experts），2个专家（高噪声 + 低噪声）
- 版本：仅有 A14B（总参数27B，激活14B）——**没有小模型**

### 14.2 不采用的原因

1. **MoE 与 RDD 冲突**：Wan 2.2 的 MoE 按噪声级别切换专家，这与 RDD 的多分辨率阶段机制存在根本冲突
2. **无小模型**：只有 A14B，无法用于高效微调实验
3. **生态不成熟**：发布仅数天，缺乏社区验证和工具支持
4. **公平对比**：PyramidalWan 基于 Wan 2.1 → 使用相同模型才能公平比较
5. **复现风险**：新模型可能有未知 bug 或行为差异

### 14.3 结论

坚持使用 **Wan 2.1**（1.3B 微调 + 14B 生成数据），这是当前最合理的选择。

---

## 附录 A：文件目录结构

```
/home/tiger/multiscale_diffusion/
├── rdd_video/                          # 视频 RDD 主目录
│   ├── flow_matching_video.py          # 核心 RDD-Video 模块
│   ├── train_rdd_wan.py                # 训练脚本
│   ├── train_rdd_wan.sh                # 训练启动脚本
│   ├── precompute_latents.py           # Latent 预计算
│   ├── generate_data_machine1.sh       # 数据生成（机器1）
│   ├── generate_data_machine2.sh       # 数据生成（机器2）
│   ├── test_rdd_video_cpu.py           # 单元测试
│   └── venv/                           # Video 环境 venv
├── diffusion/
│   └── flow_matching_with_drift.py     # 2D RDD Ground Truth
├── 2d/
│   ├── train_RDD_FM_imagenet256_unified.py  # 2D 统一训练
│   ├── 2d/                             # 2D 实验 venv
│   └── wan/                            # Wan 环境（备选）
└── RDD_Wan_Project_Documentation.md    # 本文档
```

## 附录 B：关键公式速查

| 名称 | 公式 |
|------|------|
| 前向过程 | `x_t = t·x_1 + (1-t)·ε + f(t)·d` |
| Drift 包络 | `f(t) = t·(s_{k+1} - t) / L_k` |
| Drift 信号 | `d = Upsample(Downsample(x_1)) - x_1` |
| 速度目标 | `u_t = (x_1 - ε) + [(s_{k+1} - 2t)/L_k]·d` |
| Tweedie | `pred_x1 = x_t + (1-t)·v_θ(x_t, t)` |
| 阶段过渡 | `x_new = s_{k+1}·Upsample(pred_x1) + (1-s_{k+1})·ε_new` |
| Euler 步 | `x_{t+dt} = x_t + dt·v_θ(x_t, t)` |

## 附录 C：超参数总结

| 超参数 | 值 | 来源 |
|--------|-----|------|
| stage_lengths | [60, 350, 590] | PyramidalWan 对齐 |
| 时间边界 | [0, 0.06, 0.41, 1.0] | PyramidalWan σ 映射 |
| 时空因子 | (4,4), (2,2), (1,1) | PyramidalWan |
| 学习率 | 1e-5 | PyramidalWan |
| 训练步数 | 5000 | PyramidalWan |
| Batch size | 24 (global) | 配置 |
| 分辨率 | 448×832, 81 frames | PyramidalWan |
| VAE 压缩 | 4×8×8 (T×H×W) | Wan 2.1 |

---

## 15. 2D RDD 实验结果汇总

### 15.1 实验架构概览

所有 2D 实验基于 DiT（Diffusion Transformer）在 ImageNet 256×256 上训练，使用 `sd-vae-ft-ema` 将图片编码为 32×32×4 latent。

**训练入口**：
- RDD DDPM Multistage：`train_RDD_imagenet256.py`（每个分辨率阶段独立模型）
- RDD DDPM Unified：`train_RDD_imagenet256_unified.py`（单模型统一处理所有阶段）
- RDD FM Multistage：`train_RDD_FM_imagenet256.py`
- RDD FM Unified：`train_RDD_FM_imagenet256_unified.py`
- DiT Baseline：`DiT/train_imagenet256.py`

**采样与评估入口**：
- 批量 FID：`sample_fid_unified.py`（50K 图片，自动计算 FID/IS/sFID/Precision/Recall）
- 预览：`RDD_sample_unified.py`

**通用训练参数**：batch_size=256, ckpt_every=5000, ADM center_crop 预处理

### 15.2 Checkpoint 路径

所有 checkpoint 存储在远程挂载路径，前缀为：
```
/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/
```

| 模型 | 子路径 | 已有步数 |
|------|--------|----------|
| DiT-XL-2 Baseline | `DiT-XL-2/*/checkpoints/` | 100K - 400K+ |
| RDD Unified XL (300-300-400) | `RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/` | 100K - 765K |
| RDD Unified L (300-300-400) | `RDD-Unified-L-300-300-400/imagenet256-unified/checkpoints/` | 100K - 490K |
| RDD Multistage (300-300-400) | `RDD-Multistage-300-300-400/imagenet256-multiscale/checkpoints_{32x32,16x16,8x8}/` | 至 400K |
| RDD FM Multistage | `RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_{res}/` | ~60K（训练不足） |
| RDD FM Unified XL v2 | `RDD-FM-Unified-XL-v2/000-imagenet256-fm-unified/checkpoints/` | ~30K（训练中） |
| RDD Multistage (100-300-600) | `RDD-Multistage-100-300-600/imagenet256-multiscale/` | 训练中 |

**清理规则**：仅保留 50K 倍数 + 最新 checkpoint。

### 15.3 FID 评估结果

FID 评估结果存放在 `/home/tiger/multiscale_diffusion/2d/eval_results_unified/`，共 31 个 .txt 文件。

#### DiT-XL/2 Baseline（400K step）

| 配置 | FID | IS | sFID |
|------|-----|----|----|
| 250step DDPM cfg=0（无CFG） | 19.78 | 67.59 | 6.16 |
| 250step DDPM cfg=1.25 | **10.26** | 108.36 | 5.19 |

#### RDD Unified XL（400K step, 最优结果）

| 采样配置 | FID | IS | 说明 |
|---------|-----|-----|------|
| 45-40-40 DPM-Solver-2 cfg=1.5 | **2.63** | 237.54 | 最佳 FID |
| 25-20-20 DPM-Solver-2 cfg=1.5 | 3.33 | 218.91 | 仅 65 NFE |
| 150-50-50 DDPM cfg=1.5 | **3.40** | 274.93 | 最佳 DDPM |
| 130-60-60 DDPM cfg=1.5 | 3.50 | 274.10 | |
| 90-80-80 DDPM cfg=1.5 | 3.54 | 275.52 | 标准配置 |
| 70-80-100 DDPM cfg=1.5 | 3.62 | 275.27 | |
| 45-40-40 DDPM cfg=1.5 | 3.89 | 270.40 | |
| 25-20-20 DDPM cfg=1.5 | 4.43 | 258.67 | |
| 15-10-10 DPM-Solver-2 cfg=1.5 | 11.76 | 149.94 | NFE 过低 |
| 90-80-80 DDPM cfg=1.0 | 7.06 | 130.10 | 无CFG |
| 90-80-80 DDPM cfg=1 bilinear | 9.54 | 116.70 | bilinear 上采样（差） |

#### RDD Multistage（380K-400K step）

| 采样配置 | FID | IS | 说明 |
|---------|-----|-----|------|
| 200-200-200 DDPM cfg=1.5 | 3.63 | 218.58 | |
| 150-50-50 DDPM cfg=1.5 | 3.70 | 218.30 | |
| 90-80-80 DDPM cfg=1.5 | 3.81 | 218.86 | 标准配置 |
| 45-40-40 DPM-Solver-2 cfg=1.5 | 4.38 | 180.61 | |
| 25-20-20 DPM-Solver-2 cfg=1.5 | 5.86 | 165.38 | |
| 15-10-10 DPM-Solver-2 cfg=1.5 | 17.44 | 111.10 | NFE 过低 |
| 90-80-80 DDPM cfg=1 bilinear | 14.37 | 90.23 | bilinear（差） |

#### RDD Unified L（400K step）

| 采样配置 | FID | IS |
|---------|-----|-----|
| 90-80-80 DDPM cfg=1.0 | 8.46 | 120.10 |

### 15.4 核心实验结论

1. **RDD >> DiT baseline**：同 400K 步，RDD Unified XL (FID=3.54) vs DiT (FID=10.26)
2. **DPM-Solver-2 进一步提升 RDD**：最佳 FID=2.63（仅 125 NFE），证明 ODE 连续性的价值
3. **步数分配不敏感**：cfg=1.5 下各种分配方案 FID 在 3.4-3.9 范围内波动小
4. **Nearest-neighbor >> Bilinear**：bilinear 上采样导致 FID 从 3.5 恶化到 9.5-14.4
5. **Unified > Multistage**：单模型优于独立模型（3.54 vs 3.81）
6. **FM 版本训练不足**：仅 30-60K 步，低优先级

### 15.5 待产出的论文核心图表

- **FID vs Steps 训练曲线**：每 50K 一个点，对比 DiT / RDD Unified XL / RDD Multistage
- **NFE vs FID Pareto 曲线**：展示 RDD + DPM-Solver-2 在少步采样的优势
- **消融实验**（`new_ablation_400k.sh`）：Stage split 敏感性 / CFG sweep / DPM-Solver-2 NFE 曲线

---

## 16. HuggingFace 上传配置

### 16.1 仓库信息

- **仓库 ID**：`yang335/search_agent_hl_to_i18n_transport`
- **类型**：model

### 16.2 网络配置（字节跳动内部环境）

```bash
# 上传必须走官方域名 huggingface.co（不能用镜像）
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1   # 必须！否则走 AWS S3 会被代理拦截

# 下载使用镜像（国内直连 huggingface.co 超时）
export HF_ENDPOINT=https://hf-mirror.com
```

### 16.3 上传脚本

- **精简版**：`/home/tiger/multiscale_diffusion/upload_to_hf.sh`（~167 GB）
  - DiT: 100K/200K/300K/400K
  - Unified XL: 100K/200K/300K/400K/750K
  - Unified L: 100K/200K/300K/400K
  - Multistage: 仅 400K
  - 所有日志和 FID 结果

- **全量版**：`/home/tiger/multiscale_diffusion/upload_to_hf_full.sh`（~495 GB）
  - 所有 50K 倍数 checkpoint
  - 单个 DiT-XL ckpt ≈ 11 GB

### 16.4 上传注意事项

- `hf upload` 命令具有幂等性（已存在的文件自动跳过）
- 使用 `hf`（新版工具），不是 `huggingface-cli`（已废弃）
- 需要 `export HF_TOKEN=hf_xxxxxxx`（write token）

---

## 17. 3D RDD 调研总结

### 17.1 核心目标

将 RDD 从 2D 扩展到 3D 生成领域，利用 3D 数据的天然稀疏性实现高效生成。RDD 的 constant (nearest-neighbor) upsampling 保证：空 voxel 的 drift = 0，因此可以跳过空区域计算。

### 17.2 3D Representation 选择结论

**推荐使用 TSDF/Occupancy Grid**：
- 物理意义明确，average pool 天然合理
- 数据现成（ShapeNet TSDF）
- 可与 XCube/OctFusion/Fractal3DGen 直接对比

**主流方法并非单纯 Occupancy Grid**，而是 TSDF/SDF 连续距离场或 Learned Latent：
- TSDF/SDF: XCube, SCube, OctFusion, Fractal3DGen, UltraShape
- Learned Latent: TRELLIS, Grow3D, OctFusion (latent on octree nodes)

### 17.3 竞品分析

| 方法 | 会议 | Representation | 多尺度策略 | 稀疏支持 |
|------|------|---------------|-----------|---------|
| **XCube** | CVPR 2024 | Sparse voxel + TSDF + neural features | 层级 cascade（独立模型） | ✅ VDB |
| **OctFusion** | CGF 2024 | Octree + SDF latent | 单模型多尺度 U-Net | ✅ Octree |
| **SCube** | NeurIPS 2024 | TSDF + sparse voxel | 3 级 cascade | ✅ |
| **TRELLIS** | CVPR 2025 | SLAT (sparse voxel + latent) | 两阶段: 结构→特征 | ✅ |
| **Fractal3DGen** | 2024 | SDF + 稀疏 latent | 自相似级联 | ✅ |

### 17.4 RDD 3D vs 竞品的核心区别

**RDD 的 novelty 不在于"稀疏"**（XCube/Fractal3DGen 都做了），而在于：
1. **Drift ODE 的连续性过渡**：数学上保证阶段间无中断
2. **高阶采样器兼容**：DPM-Solver-2/Heun 等可直接使用
3. **单模型统一处理所有分辨率**：不是 cascade 独立模型

### 17.5 推荐实验路线

1. 获取 ShapeNet TSDF 64³ 数据集
2. Baseline: Full-resolution DiT3D on 64³ TSDF（同参数量，无多分辨率）
3. RDD 3D: 16³→32³→64³ 的多分辨率生成
4. 对比 XCube cascade 独立模型方案
5. 核心 metric: IoU, F-Score, Chamfer Distance

### 17.6 代码现状

- `/home/tiger/multiscale_diffusion/RDD_DiT_3D/` — 基本框架代码
- `/home/tiger/multiscale_diffusion/RDD_DiT_3D_face/` — Face 变体
- `/home/tiger/multiscale_diffusion/GaussianCube/` — GaussianCube 参考
- 完整设计文档：`/home/tiger/multiscale_diffusion/RDD_3D_complete_context.md`

---

## 18. 环境问题排错完整记录

### 18.1 2D 实验环境 (已修复)

**路径**：`/home/tiger/multiscale_diffusion/2d/2d/`（Python 3.12, uv venv）

**问题**：为安装 Wan 依赖（需 numpy<2），错误地在 2D venv 里降级了 numpy
- 表现：`module 'numpy' has no attribute 'long'`
- 根因：scipy 旧版 (1.18) 使用已废弃的 `np.long`
- 修复：
  ```bash
  /home/tiger/multiscale_diffusion/2d/2d/bin/python -m pip install --no-cache-dir "scipy==1.14.0"
  # 这同时把 numpy 升级到了 2.2.6
  ```

**最终稳定状态**：
- numpy == 2.2.6
- scipy == 1.14.0
- diffusers == 0.38.0

**重要教训**：
- uv 管理的环境必须用 `uv pip freeze` 检查，不能用系统 pip
- `which pip` 可能指向 `/usr/local/bin/pip`（系统），而非 venv 内部
- 正确做法：`python -m pip install` 或 `uv pip install`

### 18.2 Wan 视频环境 (配置中)

**路径**：`/home/tiger/multiscale_diffusion/2d/wan`（uv venv）

**核心冲突**：Wan 要求 numpy<2，2D 环境要求 numpy>=2 → 必须独立 venv

**flash-attn 构建失败**：
- 原因：缺少 torch 作为 build dependency
- 解决：先 `uv pip install torch`，再 `uv pip install flash-attn --no-build-isolation`
- 或：从 GitHub Release 下载预编译 wheel

**必需依赖**：
```
torch >= 2.4
flash_attn
easydict
xfuser          # 14B Ulysses SP 推理必需
wan             # pip install -e Wan2.1/
numpy < 2       # Wan 硬性要求
imageio / imageio-ffmpeg / opencv-python
```

### 18.3 HuggingFace 下载问题

**Lock 文件残留（下载卡住）**：
- 症状：`Still waiting to acquire lock on ...`
- 原因：Ctrl+C 中断下载，残留 `.lock` 文件
- 修复：
  ```bash
  rm -f <path>/.cache/huggingface/download/*.lock
  ```

**Connection timed out**：
- 原因：国内直连 huggingface.co 不稳定
- 修复：`export HF_ENDPOINT=https://hf-mirror.com`

**huggingface-cli 命令不可用**：
- 新版工具入口改为 `hf`，不是 `huggingface-cli`
- 备用方案：`python -c "from huggingface_hub import snapshot_download; ..."`

### 18.4 pip 与 venv 路径陷阱

**问题**：`which pip` 指向系统 pip，安装包到错误位置
**解决方案优先级**：
1. `uv pip install xxx`（uv 环境）
2. `python -m pip install xxx`（确保用 venv 的 python）
3. 绝不使用裸 `pip install`

---

## 19. 理论分析文档索引

所有理论分析集中在 `/home/tiger/multiscale_diffusion/docs/theory/`，共 41 个文件：

### Markdown 文档
| 文件 | 内容 |
|------|------|
| `RESEARCH_REPORT.md` | RDD 自主科研实验报告（13 组实验，40+ 图表） |
| `experiment_plan.md` | RDD 论文补充实验计划 |
| `SNR_VISUALIZATION_GUIDE.md` | 频域 SNR 可视化工具使用指南 |
| `posterior_upsample_derivation.md` | Posterior Upsample 数学推导 |
| `RDD_related_work_survey.md` | RDD 相关工作完整调研（Relay Diffusion, LapFlow, DiMR, DCTdiff, SPEED 等） |
| `RDD_research_compendium.md` | RDD-Wan 研究全书：理论推导、实验方案与数值结果 |

### Python 分析脚本（核心实验）
- `exp1_fit_power_spectrum.py` — VAE latent 功率谱拟合
- `exp2_snr_theory_analysis.py` — 频域 SNR 理论分析
- `exp3_drift_ablation.py` — Drift 包络消融
- `exp4_fm_vs_ddpm_training_signal.py` — FM vs DDPM 训练信号对比
- `exp7_dpm_solver_compatibility.py` — DPM-Solver 兼容性验证
- `exp10_linear_velocity_theorem.py` — 线性速度场定理证明
- `exp12_sampling_efficiency_showcase.py` — 采样效率展示
- `exp13_unified_theory_framework.py` — 统一理论框架分析
- `rdd_allocation_theory.py` / `rdd_optimal_allocation.py` — 最优 Stage 分配理论

---

## 20. 资源与时间约束

### 20.1 GPU 资源

- 当前机器：138723.master（字节跳动内部机器）
- GPU：8×（型号未明确，可能 A100/H100）
- **GPU 资源剩余约 3 天**（截至 2026-08-04 的信息）

### 20.2 磁盘路径

| 路径 | 用途 | 权限 |
|------|------|------|
| `/home/tiger/multiscale_diffusion/` | 本地代码 | 完全 |
| `/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/` | 远程存储（ckpt/数据） | 读写（有时需 sudo） |
| `/mnt/bn/.../RDD/ImageNet_original/` | 2D 实验 checkpoint | 读写 |
| `/mnt/bn/.../RDD/video/` | Wan 2.1-1.3B 模型 | 读写 |
| `/mnt/bn/.../RDD/video2/` | Wan 2.1-14B 模型 | 读写 |

### 20.3 实验优先级（GPU 资源有限时）

**高优先级（必做）**：
1. 产出 FID vs Steps 训练曲线（DiT / RDD Unified XL / RDD Multistage）
2. 运行 `new_ablation_400k.sh`（DPM-Solver-2、Stage split、CFG sweep）

**中优先级**：
3. RDD-Wan 视频生成流水线（需先完成模型下载 + 数据生成）

**低优先级**：
4. FM 版本（训练步数不足，仅维持后台训练）
5. RDD 3D（实验性，需更多 GPU 时间）

---

## 21. dLLM Speculative Decoding 调研方向（探索性）

用户曾探讨过将 RDD 与 Diffusion Language Model (dLLM) 的 Speculative Decoding 结合的可能性。核心文献方向：

- **离散扩散模型**：MDLM, SEDD（2024-2025）
- **推理加速路径**：并行解码、KV 缓存、早期退出
- **与 RDD 的潜在结合点**：多分辨率 token 生成（粗→细的 text sequence 生成）

该方向尚处于文献调研阶段，未有实际代码实现。

---

## 附录 D：完整文件目录结构（扩展版）

```
/home/tiger/multiscale_diffusion/
├── rdd_video/                          # 视频 RDD 主目录
│   ├── flow_matching_video.py          # 核心 RDD-Video 模块
│   ├── train_rdd_wan.py                # 训练脚本
│   ├── train_rdd_wan.sh                # 训练启动脚本
│   ├── precompute_latents.py           # Latent 预计算
│   ├── generate_data_machine1.sh       # 数据生成（机器1，index从0开始）
│   ├── generate_data_machine2.sh       # 数据生成（机器2，index从100000开始）
│   ├── test_rdd_video_cpu.py           # 单元测试（9个，全通过）
│   └── venv/                           # Video 环境 venv (Python 3.10)
├── diffusion/
│   └── flow_matching_with_drift.py     # 2D RDD Ground Truth 实现
├── 2d/
│   ├── train_RDD_imagenet256_unified.py    # 2D DDPM Unified 训练
│   ├── train_RDD_FM_imagenet256_unified.py # 2D FM Unified 训练
│   ├── train_RDD_imagenet256.py            # 2D DDPM Multistage 训练
│   ├── train_RDD_FM_imagenet256.py         # 2D FM Multistage 训练
│   ├── sample_fid_unified.py               # 批量 FID 评估（50K 图片）
│   ├── RDD_sample_unified.py               # 预览采样
│   ├── eval_results_unified/               # FID 结果（31个.txt）
│   ├── exp/                                # 实验脚本+JSONL配置
│   │   ├── new_ablation_400k.sh            # 核心消融（16个实验）
│   │   ├── ablation_batch_400k.sh          # 批量消融（15个实验）
│   │   └── *.jsonl                         # 实验配置文件
│   ├── speed_benchmark/                    # 速度基准测试
│   ├── 2d/                                 # 2D 实验 venv (numpy 2.2.6)
│   └── wan/                                # Wan 环境 venv (numpy < 2)
├── DiT/
│   ├── train_imagenet256.py                # DiT baseline 训练
│   └── train_imagenet256_original.sh       # DiT 启动脚本
├── docs/theory/                            # 理论分析（41个文件）
│   ├── RESEARCH_REPORT.md                  # 13组实验报告
│   ├── RDD_related_work_survey.md          # 完整相关工作调研
│   ├── RDD_research_compendium.md          # 研究全书
│   └── exp*.py                             # 分析脚本
├── RDD_DiT_3D/                             # 3D RDD 代码框架
├── RDD_DiT_3D_face/                        # 3D Face 变体
├── GaussianCube/                           # GaussianCube 参考
├── RDD_3D_complete_context.md              # 3D 项目完整上下文
├── RDD_Wan_Project_Documentation.md        # 本文档
├── upload_to_hf.sh                         # 精简版上传脚本
└── upload_to_hf_full.sh                    # 全量版上传脚本
```

## 附录 E：远程存储目录结构

```
/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/
├── ImageNet_original/                  # 2D 实验主目录
│   ├── DiT-XL-2/                      # DiT baseline (至400K+)
│   ├── RDD-Unified-XL-300-300-400/    # RDD Unified XL (至765K)
│   ├── RDD-Unified-L-300-300-400/     # RDD Unified L (至490K)
│   ├── RDD-Multistage-300-300-400/    # RDD Multistage (至400K)
│   ├── RDD-Multistage-100-300-600/    # Stage 配置变体（训练中）
│   ├── RDD-FM-Multistage-300-300-400/ # FM Multistage (~60K)
│   └── RDD-FM-Unified-XL-v2/         # FM Unified XL v2 (~30K, 训练中)
├── video/
│   └── Wan2.1-T2V-1.3B/              # 微调目标模型（下载未完成）
├── video2/
│   └── Wan2.1-T2V-14B/               # 数据生成模型（下载未完成）
├── saved/
│   ├── all_logs/                      # 600+评估日志
│   └── old/                           # 旧实验（已清理.pt，保留日志）
└── exp_samples_original/              # 采样图片输出目录
```

## 附录 F：用户关键偏好（给新 Agent 的提醒）

1. **用 `uv pip`**：所有 Python 包操作使用 `uv pip install` 或 `uv pip freeze`
2. **不要破坏环境**：修改任何环境前先确认当前状态
3. **中文沟通**：直接了当，结果导向
4. **下载用镜像**：`HF_ENDPOINT=https://hf-mirror.com`
5. **上传走代理**：`sys-proxy-rd-relay.byted.org:8118` + `HF_HUB_DISABLE_XET=1`
6. **日志续写不新建**：实验目录不用递增编号（001, 002...），直接续写
7. **磁盘空间敏感**：定期清理中间 checkpoint
8. **学术标准高**：目标 NeurIPS/ICLR，要求方法新颖性和理论深度
9. **对错误零容忍**：尤其是逻辑错误和环境损坏

---

*文档最后更新：2026-08-04*
*项目负责人：杨远 (yangyuan.335)，UCSD CS*
*环境：Linux, Python 3.10/3.12, PyTorch >= 2.4, 字节跳动内部机器 (138723.master)*
