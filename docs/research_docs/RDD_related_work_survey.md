# RDD 相关工作完整调研报告

> **Residual Drift Diffusion (RDD) vs. 12 种多尺度扩散方法**
> 
> 本文档覆盖每篇论文的完整方法理论、工程实现、主要实验结果，以及与 RDD 的深度对比分析。
> 
> 最后更新：2026-07-30

---

## 目录

1. [RDD 核心方法回顾](#1-rdd-核心方法回顾)
2. [Relay Diffusion (ICLR 2024)](#2-relay-diffusion-iclr-2024)
3. [LapFlow (ICLR 2026)](#3-lapflow-iclr-2026)
4. [DiMR (NeurIPS 2024)](#4-dimr-neurips-2024)
5. [DCTdiff (ICML 2025)](#5-dctdiff-icml-2025)
6. [SPEED (Stanford 2025)](#6-speed-stanford-2025)
7. [Pyramid-Flow (ICLR 2025)](#7-pyramid-flow-iclr-2025)
8. [SwD — Scale-wise Distillation (Yandex 2025)](#8-swd--scale-wise-distillation-yandex-2025)
9. [LowDiff (2025)](#9-lowdiff-2025)
10. [MrFlow (2025)](#10-mrflow-2025)
11. [XCube (CVPR 2024)](#11-xcube-cvpr-2024)
12. [OctFusion (SGP 2025)](#12-octfusion-sgp-2025)
13. [PDD — Pyramid Discrete Diffusion (ECCV 2024)](#13-pdd--pyramid-discrete-diffusion-eccv-2024)
14. [综合对比大表](#14-综合对比大表)
15. [RDD 行动计划](#15-rdd-行动计划)
16. [FlowAR (ICML 2025)](#16-flowar-icml-2025)
17. [PixelFlow (2025)](#17-pixelflow-2025)
18. [FreCaS (NeurIPS 2024)](#18-frecas-neurips-2024)
19. [f-DM (ICLR 2023)](#19-f-dm-iclr-2023)
20. [DFM — Diffusion Flow Matching (NeurIPS 2025)](#20-dfm--diffusion-flow-matching-neurips-2025)
21. [Matryoshka Diffusion Models (ICLR 2024)](#21-matryoshka-diffusion-models-iclr-2024)
22. [TPDiff — Temporal Pyramid Diffusion (2025)](#22-tpdiff--temporal-pyramid-diffusion-2025)
23. [Open-Sora Plan (2025)](#23-open-sora-plan-2025)
24. [Blockwise Parallel Flow Matching (2025)](#24-blockwise-parallel-flow-matching-2025)
25. [更新后的实验清单](#25-更新后的实验清单)
26. [Frequency Diffusion (ICLR 2025 Workshop)](#26-frequency-diffusion-iclr-2025-workshop)
27. [Hierarchical Rectified Flow (ICLR 2025)](#27-hierarchical-rectified-flow-iclr-2025)

---

## 1. RDD 核心方法回顾

### 1.1 核心思想

RDD 在标准 DDPM 的前向过程中引入一个与 $x_0$ 相关的**连续漂移项 (drift)**，使得扩散路径的均值在时间推进中从原始图像 $x_0$ 逐渐过渡到其低分辨率版本 $\text{blur}(x_0)$。这实现了在**单一 Markov 链**中编码多分辨率信息。

### 1.2 数学公式

**前向 SDE：**

$$dx = \left[-\frac{1}{2}\beta(t)x + \gamma \cdot f(x_0, t)\right] dt + \sqrt{\beta(t)}\, dW$$

其中 $f(x_0, t) = \sqrt{\bar\alpha_t} \cdot \kappa(t) \cdot d$。

**离散前向分布：**

$$q(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{\bar\alpha_t} \cdot \kappa(t) \cdot d,\; (1 - \bar\alpha_t)\mathbf{I}\right)$$

**Drift 定义：**

$$d = \text{UP}(\text{DOWN}(x_0)) - x_0$$

这等价于 **负的 Laplacian 高频残差**：$d = -L_0$。

**时间调度：**

$$\kappa(t) = \frac{t_{\text{global}} - T_c}{T_{c+1} - T_c} \in [0, 1]$$

在每个阶段内线性增长。阶段划分：$\text{stage\_lengths} = [300, 300, 400]$，对应全局时间边界 $[0, 300, 600, 1000]$。

**预测目标：**

$$\tilde\varepsilon = \varepsilon + \sqrt{\frac{\bar\alpha_t}{1 - \bar\alpha_t}} \cdot \kappa(t) \cdot d$$

网络只需预测这个**单一修正噪声** $\tilde\varepsilon$。

**后验分布：**

$$q(x_{t-1} | x_t, x_0)$$

仍为高斯分布，均值额外包含 drift 修正项：$\text{posterior\_mean\_coef\_drift} \cdot d$。

**DPM-Solver 修正步：**

$$x_{t-1} = \text{standard\_dpm\_step}(x_t) + \sqrt{\bar\alpha_{t-1}} \cdot (\kappa(t-1) - \kappa(t)) \cdot d$$

### 1.3 关键实现

| 模块 | 文件 | 行号 |
|------|------|------|
| q_sample (含 drift) | `gaussian_diffusion_with_drift.py` | L245-272 |
| 训练 target $\tilde\varepsilon$ 构造 | `gaussian_diffusion_with_drift.py` | L991-997 |
| 后验 drift 修正 | `gaussian_diffusion_with_drift.py` | L285-287 |
| DPM-Solver 实现 | `gaussian_diffusion_with_drift.py` | L1310-1553 |
| 阶段过渡 (noise_filling) | `RDD_sample.py` | L147-158 |
| stage_lengths 划分 | `train_RDD_imagenet256.py` | L192-206 |

### 1.4 RDD 的 4 个核心 Novelty

1. **单一 Markov forward process 中连续 drift**（非级联/非离散跳变）
2. **闭式 posterior + DPM-Solver 兼容**
3. **单一预测目标 $\tilde\varepsilon$**（非双重预测如 f-DM）
4. **Drift coefficient $\kappa$ 自动处理跨分辨率 SNR**

### 1.5 频域等效 SNR（关键理论推导）

设 $x_0$ 的 DCT 分解为 $\hat{x}_0(f)$，drift 在频域为：

$$\hat{d}(f) = (LP(f) - 1) \cdot \hat{x}_0(f)$$

RDD 前向均值在频域：

$$\mu_t(f) = \sqrt{\bar\alpha_t} \cdot [(1 - \kappa(t)) + \kappa(t) \cdot LP(f)] \cdot \hat{x}_0(f)$$

有效 SNR：

$$\text{SNR}_{\text{RDD}}(f, t) = \frac{\bar\alpha_t \cdot [(1 - \kappa(t)) + \kappa(t) \cdot LP(f)]^2}{1 - \bar\alpha_t}$$

- **低频** ($LP(f) \approx 1$)：$\text{SNR}_{\text{RDD}} = \bar\alpha_t / (1 - \bar\alpha_t)$（不变！被 drift 保护）
- **高频** ($LP(f) \approx 0$)：$\text{SNR}_{\text{RDD}} = \bar\alpha_t \cdot (1-\kappa(t))^2 / (1 - \bar\alpha_t)$（额外衰减 $(1-\kappa(t))^2$ 倍！）

**结论：RDD 的 drift 等效于频率自适应的 SNR 衰减——低频 SNR 不变，高频 SNR 被额外衰减。**

---

## 2. Relay Diffusion (ICLR 2024)

### 2.1 论文信息

**"Relay Diffusion: Unifying diffusion process across resolutions for image synthesis"**, ICLR 2024.

### 2.2 方法概述

两阶段级联：
- **Stage 1**：标准 EDM 在 64×64 生成低分辨率图像
- **Stage 2**：将 64×64 双线性上采样到 256×256，然后用"Blurring Diffusion"（DCT 域模糊扩散 + Block Noise）精炼高分辨率细节

### 2.3 理论/数学

**前向过程 (Stage 2, Blurring Diffusion)：**

$$x_t = \text{DCTBlur}(x_0, \sigma_{\text{blur}}(t)) + \sigma(t) \cdot (\varepsilon + s_{\text{block}} \cdot \varepsilon_{\text{block}})$$

其中 DCTBlur 定义为在 4×4 patch 的 DCT 域中按频率指数衰减：

$$\alpha(f, t) = \exp\left(-|f|^2 \cdot \sigma_{\text{blur}}^2(t) / 2\right) \cdot (1-\delta) + \delta$$

模糊强度随时间变化：

$$\sigma_{\text{blur}}(t) = \sigma_{\max} \cdot \sin(t\pi/2)^2$$

**Block Noise：** 空间 4×4 相关噪声，通过循环位移累加实现：

```python
for px in range(block_size):
    for py in range(block_size):
        g_noise += noise.roll(shifts=(px, py), dims=(-2, -1))
g_noise /= block_size  # 归一化
```

**预测目标：** 网络预测**未模糊的干净原图**（非 epsilon），通过 EDM 预条件化间接实现：

$$D(x) = c_{\text{skip}} \cdot x + c_{\text{out}} \cdot F_\theta(c_{\text{in}} \cdot x, c_{\text{noise}})$$

**损失函数：**

$$\mathcal{L} = w(\sigma) \cdot \|D(x_{\text{blurred}} + n, \sigma) - x_0\|^2$$

$$w(\sigma) = \frac{\sigma^2 + \sigma_d^2}{\sigma \cdot \sigma_d^2}, \quad \text{clamp} \leq 100000$$

**采样过程 (blur_sampler)：** 在 DCT 频域执行 Euler/Heun 步骤，同时处理噪声衰减和频率模糊衰减。Stage 2 从 `truncation_sigma=0.9` 开始（非纯噪声），因为起点是有意义的上采样图像。

### 2.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 框架 | EDM（σ 参数化，VE-SDE 风格） |
| 网络 | DhariwalUNet (ADM, conv-based)，2 个独立 UNet |
| 输入空间 | 像素空间（无 VAE） |
| Batch | `batch_gpu=8`，累积 64 轮 → 全局 4096 |
| 混合精度 | FP16 |
| EMA | 50K images |
| Stage 过渡 | 双线性上采样 + truncated σ=0.9 开始 |
| 预条件化 | $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}$（EDM 式） |
| 模型参数 | Stage1 ~200M, Stage2 ~400M, 总 ~600M |

**关键代码文件：**
- 模糊扩散核心：`blurring.py` L143-175
- 损失函数：`training/loss.py` L102-153 (BlurLoss)
- 采样器：`generate_imagenet.py` L33-127 (blur_sampler)

### 2.5 主要结果

| 指标 | 结果 |
|------|------|
| ImageNet-256 FID (CFG) | **2.64** |
| 总采样步数 | 456 步 (Stage1: 256步 + Stage2: 200步) |
| 训练步数 | ~305K 步 (1250M images / batch4096) |
| 训练设备 | 8 GPU |

### 2.6 与 RDD 相似之处

1. **都修改前向过程来编码分辨率信息**
2. 都实现了"随时间逐渐丢失高频"的效果
3. **数学联系**：
   - RDD 均值 = $\sqrt{\bar\alpha_t} \cdot [(1-\kappa(t)) \cdot x_0 + \kappa(t) \cdot \text{blur}(x_0)]$（线性插值式模糊）
   - Relay 均值 = DCTBlur（指数衰减式模糊）
   - 两者都是 $x_0 \to \text{blurred version}$ 的渐进过渡，区别在于线性 vs 指数

### 2.7 该方法优势（RDD 劣势）

- 频域操作有明确的物理含义（每个 DCT 系数独立控制）
- Block Noise 比纯高斯更好地模拟了"低分辨率→高分辨率"引入的结构化误差
- 大 batch（4096）训练更稳定/收敛更快
- 像素空间操作避免了 VAE 重建误差

### 2.8 该方法劣势（RDD 优势）

- 两个独立模型分别训练，总参数量是 RDD 的 2-3 倍
- DCT 域自定义采样器复杂度高，**无法用 DPM-Solver 加速**
- 推理需 Stage1(256步) + Stage2(200步) = 456 步；RDD 用 DPM-Solver **10-50 步**
- 无闭式后验，采样数学不够优雅
- 阶段间硬切换 (64→256)，信息传递断裂

### 2.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 低 | 在 `noise_filling` 中加 Block Noise 消融实验 |
| 中 | 频域分析：画 RDD 在不同 t 的 DCT 功率谱，证明 drift = progressive high-freq loss |
| 中 | 增大 batch size（梯度累积到 1024-2048） |
| 论文 | 对比：RDD 10步 DPM-Solver vs Relay 456步（计算量/质量 Pareto） |

---

## 3. LapFlow (ICLR 2026)

### 3.1 论文信息

**"LapFlow: Laplacian Pyramid Flow Matching for Multi-Scale Image Generation"**, ICLR 2026.

### 3.2 方法概述

将图像做 Laplacian 金字塔分解，各频带在对应的时间段内独立进行 Flow Matching，使用 Mixture of Transformers (MoT) 实现跨尺度注意力共享。

### 3.3 理论/数学

**Laplacian 金字塔分解：**

$$L_0 = x - \text{UP}(\text{DOWN}(x)) \quad \text{(高频残差, 32×32)}$$
$$L_1 = \text{DOWN}(x) - \text{UP}(\text{DOWN}^2(x)) \quad \text{(中频残差, 16×16)}$$
$$L_2 = \text{DOWN}^2(x) \quad \text{(低频基础, 8×8)}$$

**分段时间 Flow Matching（3级，time_segments=[0.33, 0.67]）：**

每个 Scale $i$ 在对应时间段 $[T_i, 1]$ 内做线性插值：

$$x_t^{(i)} = \frac{t - T_i}{1 - T_i} \cdot L_i + (1-t) \cdot \varepsilon_i$$

**预测目标（速度场）：**

$$u_i = \frac{dx_t^{(i)}}{dt} = \frac{L_i}{1 - T_i} - \varepsilon_i$$

**训练损失：**

$$\mathcal{L} = \sum_i w_i \cdot \text{MSE}(v_{\text{pred}}^{(i)}, u^{(i)})$$

**与 RDD 的关键数学联系：**

$$d = \text{UP}(\text{DOWN}(x_0)) - x_0 = -L_0 \quad \text{(RDD 的 drift = 负的高频残差！)}$$

因此 RDD 前向均值：

$$\sqrt{\bar\alpha_t} \cdot (x_0 + \kappa(t) \cdot d) = \sqrt{\bar\alpha_t} \cdot (x_0 - \kappa(t) \cdot L_0)$$

即 RDD 在前向过程中逐渐**减掉高频残差** → 使 $t \to T$ 时趋向低分辨率版本。

**关键区别：**
- LapFlow 在**不同维度空间**操作（$L_0$ 是 32×32，$L_2$ 是 8×8），需要不同大小的 tensor
- RDD 全程在**同一维度空间**（32×32），通过 drift 隐式实现频率分离
- LapFlow 各 scale 可并行（causal mask 允许），RDD 必须串行（Markov 依赖）

### 3.4 工程实现

| 维度 | LapFlow | RDD |
|------|---------|-----|
| 框架 | Flow Matching (ODE, $t \in [0,1]$) | DDPM (SDE, 离散 $t$) |
| 网络 | MultiScaleDiT (MoT) | DiT (plain) |
| 多尺度输入 | 各 scale 独立 patchify → 不同 token 数 | 统一 32×32 latent |
| 注意力 | 全局 attention + causal mask | 标准全局 attention |
| 参数共享 | MoT：共享 attention + 独立 MLP/LN/QKV | 完全共享 (unified) |
| ODE solver | dopri5 / euler (adaptive) | DPM-Solver 1-2 阶 |
| 梯度裁剪 | grad_clip_norm=1.0 | 无 |
| 学习率调度 | CosineAnnealing | 恒定 |

**MoT 架构核心：**
- 每个尺度独立的：QKV 投影、MLP、AdaLN
- 共享的：Attention 计算（跨尺度交互）
- **Causal mask**：scale $i$ 只能看到 scale $\leq i$ 的 token

**关键代码文件：**
- 主训练：`feb12_train_ours.py` L392-427
- Laplacian 分解：`sampler/edify_image_sample.py` L154-186
- MoT 架构：`models/MultiScaleDiT.py` L48-194
- Causal Mask：`models/MultiScaleDiT.py` L156-164

### 3.5 主要结果

| 配置 | ImageNet-256 FID |
|------|------------------|
| DiT-B | ~15 |
| DiT-L | ~8 |
| 训练 | ~1000 epochs, 8-16 GPU |

### 3.6 与 RDD 相似之处

1. **时间段划分几乎相同！**
   - LapFlow：$[0, 0.33, 0.67, 1.0]$
   - RDD 归一化：$[0, 0.3, 0.6, 1.0]$
2. 都是**先低频后高频**的渐进式生成
3. **代码级相似性：**
   - LapFlow `feb12_train_ours.py` L392-411 (scale_selector → 时间范围映射)
   - RDD `train_RDD_imagenet256.py` L192-206 (stage_lengths → $T_c$ 边界)
4. 可直接复用：causal attention mask、per-scale loss weight、梯度裁剪 + LR cosine decay

### 3.7 该方法优势（RDD 劣势）

- MoT 让各 scale 有专属的投影/MLP，但共享 attention，参数效率更高
- Causal mask 从理论上保证了因果依赖的正确性
- 部分并行采样加速推理
- 各 scale 独立 loss weight 可精细调优
- Flow Matching 训练通常比 DDPM 更稳定（方差更低）

### 3.8 该方法劣势（RDD 优势）

- ODE 采样无法使用 DPM-Solver（高阶 solver 效率不如 DPM-Solver）
- 需要显式 Laplacian 分解作为预处理
- 各 scale 的 tensor 维度不同，实现更复杂
- **无闭式后验，无法做精确的概率推断**

### 3.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 高 | 实验 RDD 的 Flow Matching 版本放 Appendix：$x_t = (1-t)(x_0 + \kappa(t) \cdot d) + t \cdot \varepsilon$ |
| 中 | 在 unified 模型中加入 per-stage loss weighting |
| 论文 | 明确说明 LapFlow time_segments ≡ RDD stage_lengths，但 RDD 保持闭式后验 + DPM-Solver |

---

## 4. DiMR (NeurIPS 2024)

### 4.1 论文信息

**"DiMR: Denoising in Multi-Resolution"**, NeurIPS 2024. 当前 ImageNet-256 **绝对 SOTA (FID=1.70)**。

### 4.2 方法概述

纯架构多分辨率设计——**完全不修改扩散前向过程**。单次标准扩散 + 多分辨率去噪网络（Transformer 处理低分辨率 + ConvNeXt 处理高分辨率）。

### 4.3 理论/数学

**前向过程（完全标准 VP-SDE，不做任何修改！）：**

$$x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \varepsilon$$

$$\beta_t = \left(\sqrt{\beta_0} + \frac{\sqrt{\beta_1} - \sqrt{\beta_0}}{T} \cdot t\right)^2, \quad \beta_0=0.00085, \; \beta_1=0.012$$

**多尺度目标构造：**

对 epsilon 做均值池化降采样：

$$\text{target}_{16\times16} = \frac{\text{AvgPool}_{2\times2}(\varepsilon)}{2} \quad \text{(scale correction)}$$
$$\text{target}_{32\times32} = \varepsilon$$

**损失函数：**

$$\mathcal{L} = 0.25 \cdot \text{MSE}(\text{pred}_{16}, \text{target}_{16}) + 1.0 \cdot \text{MSE}(\text{pred}_{32}, \text{target}_{32})$$

**TDP (Time-Dependent Parameter) 时间条件注入：**

$$\text{result} = \sigma(a) \cdot \text{param}_0 + (1-\sigma(a)) \cdot \text{param}_1$$
$$a = (\log\_\text{snr} / 4) \cdot w + b$$

其中 $\log\_\text{snr} = (n/1000) \times 8 - 4$。

**与 RDD 的数学联系：**
- DiMR 让网络在 16×16 预测 $\text{AvgPool}(\varepsilon)$ → 等效于"学习噪声的低频结构"
- RDD 让网络预测 $\varepsilon + \text{drift\_correction}$ → 等效于"学习噪声 + 分辨率先验"
- 两者都在告诉网络低分辨率信息，但方式不同：DiMR 通过架构显式分解，RDD 通过修改 SDE 隐式编码

### 4.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 网络架构 | MRModel: Transformer(16×16, 39 blocks, dim=960) + ConvNeXt(32×32, 20 blocks, dim=480) |
| Stage 连接 | PixelShuffle 上采样 + skip connection |
| 输出 | 多个 head: [eps_16×16, eps_32×32]；采样时只用 32×32 输出 |
| DPM-Solver | predict_x0=True, 250 步 |
| Optimizer | AdamW(β=(0.99, 0.99), wd=0.03)，warmup 5000 步 |
| Batch | 1024 |
| 训练设备 | 16 GPU, 2 nodes |
| 总步数 | 1,000,000 |
| 参数量 | XL=505M, G=1.06B |
| CFG | p_uncond=0.1, scale=1.5 |

**关键代码文件：**
- MRModel 架构：`libs/core/blocks.py` L296-345
- 多尺度目标：`sde.py` L313-334 (`multi_scale_targets`)
- TDP 模块：`libs/core/sigmoid/module.py` L28-60
- XL 配置：`configs/DiMR-XL-2R_imagenet256.py`

**训练命令：**
```bash
accelerate launch --multi_gpu --num_processes 16 --num_machines 2 --mixed_precision fp16 \
    train.py --config=configs/DiMR-XL-2R_imagenet256.py
```

### 4.5 主要结果

| 配置 | ImageNet-256 FID | 参数量 |
|------|------------------|--------|
| DiMR-XL/2R | **1.70** | 505M |
| DiMR-G/2R | **1.63** | 1.06B |

### 4.6 与 RDD 相似之处

1. 都使用 DPM-Solver 采样
2. 都是单一模型处理多分辨率
3. 都将多尺度信息编码到训练目标中（DiMR: 池化 ε; RDD: $\tilde\varepsilon$）
4. `multi_scale_targets`（对 ε 做 AvgPool）与 RDD 的 drift correction 概念上类似

### 4.7 该方法优势（RDD 劣势）

- FID=1.70，绝对最强；架构设计成熟
- Transformer(低分辨率) + ConvNeXt(高分辨率) 混合效率高
- TDP 时间条件注入比 MLP embedding 更精细
- PixelShuffle 特征传递无损
- Batch 1024 + 16GPU 训练充分

### 4.8 该方法劣势（RDD 优势）

- 完全不改前向过程 → 没有利用"分辨率结构"作为归纳偏置
- 在同等训练步数下，RDD 的 drift 提供额外结构先验，可能**收敛更快**
- 1M步 × batch1024 × 16GPU = 巨大计算量；RDD 如果在 200-700K 步达到类似效果则训练效率更高
- 多 head 输出增加参数量和复杂度

### 4.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 高 | 训练到 700K 步后对比：RDD-XL/2 @ 700K vs DiMR-XL @ 700K (same compute) |
| 中 | DiMR 架构 + RDD drift 融合实验：在 MRModel 中加 drift，target 从 ε 改为 $\tilde\varepsilon$ |
| 中 | 学习 DiMR 超参：AdamW(β2=0.99, wd=0.03)，batch 1024，warmup 5000 |
| 论文 | 定位 RDD 为"与 DiMR 正交的改进"：DiMR 改架构，RDD 改过程，二者可结合 |

---

## 5. DCTdiff (ICML 2025)

### 5.1 论文信息

**"Diffusion on the Discrete Cosine Transform Space"**, ICML 2025.

### 5.2 方法概述

将图像转到块状 DCT 系数空间后再做扩散，通过全局 SNR_scale 修正和频率加权损失处理不同频带的信号幅度差异。提出"扩散 = 频谱自回归"的理论视角。

### 5.3 理论/数学

**数据预处理流水线：**

$$\text{RGB} \to \text{YCbCr} \to \text{4:2:0 下采样} \to B \times B \text{ 分块} \to \text{2D DCT} \to \text{Zigzag 排序} \to \text{高频截断}$$

最终 token shape: $(H/B \times W/B, \; 6 \times \text{low\_freqs})$，对于 64×64 图像 = $(256, 24)$。

**修改版 VP-SDE：**

$$\beta_{\text{eff}}(t) = \beta(t) + \frac{d}{dt}\left[\log\left(1 + (S-1) \cdot \exp\left(-\int_0^t \beta(s)ds\right)\right)\right]$$

设 $A(t) = \int_0^t \beta(s)ds$，展开：

$$\beta_{\text{eff}}(t) = \beta(t) \cdot \frac{1}{1 + (S-1) \cdot \exp(-A(t))}$$

**效果：** 终端 SNR 被放大 $S$ 倍，$\text{SNR}_{\text{new}} = S \cdot \text{SNR}_{\text{old}}$。

**关键限制：所有频率使用相同的 $\beta_{\text{eff}}(t)$ schedule——全局 SNR_scale $S$ 是统一的，不区分频率！**

**频率加权损失：**

$$\mathcal{L} = \text{MSE}(\varepsilon_{\text{pred}}, \varepsilon) \cdot Y_{\text{reweight}}$$

其中 $Y_{\text{reweight}}$ 为各频率系数的统计标准差。

**与 RDD 的关键对比——频域 SNR：**

| | DCTdiff | RDD |
|--|---------|-----|
| SNR 修正方式 | 全局标量 $S$，所有频率统一放大 | 频率自适应，由 $LP(f)$ 决定 |
| 高频 SNR | $S \cdot \bar\alpha_t / (1-\bar\alpha_t)$ | $\bar\alpha_t \cdot (1-\kappa(t))^2 / (1-\bar\alpha_t)$（额外衰减） |
| 低频 SNR | $S \cdot \bar\alpha_t / (1-\bar\alpha_t)$ | $\bar\alpha_t / (1-\bar\alpha_t)$（不变） |
| 结论 | 不区分频率 | **频率自适应**，RDD 更精细 |

### 5.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 网络 | UViT (embed=768, depth=16, heads=12) |
| 操作空间 | DCT 系数空间 (block-wise 2D DCT) |
| Token 形式 | $(H/B \cdot W/B, \; 6 \cdot \text{low\_freqs})$ |
| DPM-Solver | 兼容（修改了 noise_schedule 中的 beta） |
| 验证分辨率 | **仅 64×64** |
| SNR_scale | 全局标量参数 $S$（grid search） |
| Batch | 1024, 4 GPU |
| 训练步数 | 500,000 |

**关键代码文件：**
- DCT 预处理：`datasets.py` L361-457 (DCT_4YCbCr 类)
- SDE + SNR_scale：`sde.py` L72-147
- 频率加权训练：`train.py` L78-97

### 5.5 主要结果

| 分辨率 | 结果 |
|--------|------|
| ImageNet 64×64 | 有结果（论文中） |
| ImageNet 256 | **未报告**（未 scale 到高分辨率） |

### 5.6 与 RDD 相似之处

1. 都修改了有效 SNR schedule
2. 都使用 DPM-Solver
3. 都处理频率相关的噪声动态
4. **理论联系：RDD 以更精细的方式实现了 DCTdiff 的目标（频率感知 SNR）**

### 5.7 该方法优势（RDD 劣势）

- "扩散 = 频谱自回归"理论优雅
- SNR_scale 有闭式最优解 $S^* = \arg\min \text{FID}$（可 grid search）
- DCT 系数物理含义明确
- 频率加权 loss 有统计学依据

### 5.8 该方法劣势（RDD 优势）

- **全局 SNR_scale 不区分频率，不如 RDD 的频率自适应精细**
- 仅在 64×64 验证，未 scale 到 256/512
- 数据预处理复杂（RGB→YCbCr→DCT→zigzag）
- 不能处理连续分辨率变化

### 5.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 高/理论 | 推导并展示频域等效 SNR 公式（已在 §1.5 给出），证明 RDD > DCTdiff |
| 中 | 频域可视化实验：对不同 t 的 $x_t$ 做 DCT，画标准 DDPM vs RDD vs DCTdiff 的 SNR 曲线 |
| 论文 | 引用 DCTdiff 的"扩散=频谱自回归"理论，说明 RDD drift 是该理论的优雅实现 |

---

## 6. SPEED (Stanford 2025)

### 6.1 论文信息

**"SPEED: Spectrally Progressive and Efficient Estimation of Diffusion"**, Stanford 2025.

### 6.2 方法概述

Training-free 推理加速方法：基于自然图像功率谱衰减理论，给出数学最优的分辨率转换时刻；在转换时用频谱噪声扩展 + kappa 时间步对齐实现无损分辨率提升。

### 6.3 理论/数学

**三个核心组件：**

**(a) 频谱噪声扩展 (Spectral Noise Expansion)：**

$$x_{\text{expanded}} = \text{IDCT}\left[\text{DCT}(x_{\text{src}}) \oplus t \cdot \varepsilon_{\text{high}}\right]$$

低频位置保持源信号，高频位置填入幅度为 $t$ 的噪声。

**(b) 最优转换时刻 (Optimal Resolution Schedule)：**

基于自然图像功率谱模型 $P(\omega) = A \cdot |\omega|^{-\beta}$：

$$t^*(\omega) = \frac{1}{1 + \sqrt{\frac{\delta}{P(\omega) \cdot (1 + P(\omega) - \delta)}}}$$

在 $\omega = s_i \cdot \omega_{\max}$（当前尺度的 Nyquist 频率）处取值作为转换时刻。

**(c) 时间步对齐 (Timestep Alignment)：**

$$\kappa(t, r) = \frac{r}{1 + (r-1) \cdot t}$$
$$\tilde{t} = t \cdot \kappa(t, r), \quad \tilde{x} = \kappa(t, r) \cdot x_{\text{expanded}}$$

**与 RDD 的关键联系：**

SPEED 的理论可以直接计算 RDD 的**最优 stage_lengths**！

对于 ImageNet latent (32×32, f=8)，设 Nyquist 频率 $\omega_1=0.5, \omega_2=0.25$：
- 用 $A=203.6, \beta=1.915$（FLUX VAE 参数），$\delta=0.01$
- 计算 $t^*(\omega_1)$ 和 $t^*(\omega_2)$ = 最优阶段边界

**RDD noise_filling vs SPEED spectral_expand：**
- RDD：空间域 nearest 上采样 + **全频段**加噪
- SPEED：频域低频保持 + **仅高频**加噪 → 更精细

### 6.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 核心代码 | `utils.py` (249 行总共) |
| activation_time | L33-47 |
| kappa + align_timestep | L80-103 |
| spectral_expand_and_align | L195-249 |
| 支持变换 | DCT, DWT, FFT |
| 应用模型 | FLUX.1-dev (图像), WAN2.1 (视频) |
| 配置 | `configs.yaml` 含每个模型的 $A, \beta$ 参数 |

**推理命令：**
```bash
python latent_image_gen.py \
    --prompts "a translucent jellyfish" \
    --transform dct --scales 0.5 1.0 --delta 0.01 --save_dir ./out_flux
```

### 6.5 主要结果

| 指标 | 结果 |
|------|------|
| FLUX 加速 | 2-3× 无质量损失 |
| WAN 视频加速 | 2-3× 无质量损失 |
| 训练需求 | **完全不需要训练** |

### 6.6 与 RDD 相似之处

1. 都在特定时间点定义分辨率转换
2. 都在分辨率扩展时填充高频噪声
3. SPEED 的理论**直接可用于优化 RDD 的 stage_lengths**
4. SPEED 的 $\kappa(t, r)$ 与 RDD 的 $\kappa(t)$ 命名相似、概念相关

### 6.7 该方法优势（RDD 劣势）

- 数学最优的分辨率调度（RDD 的 stage_lengths 目前是手动设定的）
- kappa 对齐保证分辨率转换后的状态与模型期望一致
- 频域扩展只在高频加噪，保持低频信号（比 RDD 的全频段加噪更合理）
- 不需要任何训练

### 6.8 该方法劣势（RDD 优势）

- Training-free = 受限于预训练模型的能力上限
- 无法提升模型本身的生成质量
- 需要准确的功率谱参数估计

### 6.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 高/低工作量 | 用 SPEED 公式计算最优 stage_lengths：测 RDD VAE latent 功率谱参数 $A, \beta$ → 计算 activation_time |
| 中 | 改进 noise_filling：全频段噪声 → 频谱扩展（仅高频噪声）：$x = \sqrt{\bar\alpha} \cdot \text{UP}(x_0) + \sqrt{1-\bar\alpha} \cdot \text{IDCT}[0_{\text{low}} \oplus \varepsilon_{\text{high}}]$ |
| 论文 | 引用 SPEED 理论作为 RDD 设计动机的理论支撑 |

---

## 7. Pyramid-Flow (ICLR 2025)

### 7.1 论文信息

**"Pyramid Flow Matching for Efficient Video Generative Modeling"**, ICLR 2025.

### 7.2 方法概述

将 Flow Matching 路径拆分为 3 个空间金字塔阶段，各阶段在不同分辨率工作，阶段间用 **renoising**（加块噪声）连接。面向视频生成。

### 7.3 理论/数学

**多阶段 Flow Matching：**

3 个阶段，`stage_range = [0, 1/3, 2/3, 1]`：
- 阶段 0 ($t \in [0, 1/3]$)：start=纯噪声(低分辨率), end=低分辨率 clean
- 阶段 1 ($t \in [1/3, 2/3]$)：start=UP(阶段 0 结果)+renoise, end=中分辨率 clean
- 阶段 2 ($t \in [2/3, 1]$)：start=UP(阶段 1 结果)+renoise, end=高分辨率 clean

每阶段内线性插值：

$$x_t = \text{ratio} \cdot \text{start\_point} + (1-\text{ratio}) \cdot \text{end\_point}$$

预测目标：$u = \text{start\_point} - \text{end\_point}$（rectified flow）

**Renoising 精确公式（推理时阶段过渡）：**

$$\alpha = \frac{1}{\sqrt{1 + 1/\gamma} \cdot (1-\sigma_{\text{ori}}) + \sigma_{\text{ori}}}$$
$$\beta = \frac{\alpha \cdot (1 - \sigma_{\text{ori}})}{\sqrt{\gamma}}$$
$$\text{latents}_{\text{new}} = \alpha \cdot \text{UP}(\text{latents}_{\text{old}}) + \beta \cdot \text{block\_noise}$$

其中 $\gamma$ 控制 renoising 强度：
- $\gamma \to \infty$：$\alpha \to 1, \beta \to 0$，不加噪声（直接上采样）
- $\gamma \to 0$：$\alpha \to 0$，噪声完全覆盖

**与 RDD 的数学联系：**

RDD 的 noise_filling：

$$x_{\text{next}} = \sqrt{\bar\alpha_{t_{\text{start}}}} \cdot \text{UP}(\hat{x}_0) + \sqrt{1-\bar\alpha_{t_{\text{start}}}} \cdot \varepsilon$$

即 $\alpha_{\text{RDD}} = \sqrt{\bar\alpha_{t_{\text{start}}}}, \; \beta_{\text{RDD}} = \sqrt{1-\bar\alpha_{t_{\text{start}}}}$，由 $t_{\text{start}}$ 唯一决定。

Pyramid-Flow 通过额外参数 $\gamma$ 控制加噪强度；RDD 通过 stage_lengths（决定 $t_{\text{start}}$）隐式控制。

### 7.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 网络 | DiT-based (flux_modules) |
| 空间金字塔 | stages=[1, 2, 4] (分辨率比) |
| 视频 | 时间金字塔 + temporal downsampling |
| Renoising 噪声 | Block noise (空间相关) |
| ODE solver | Euler |
| 应用 | 视频生成 (768p, 10s) |

**关键代码文件：**
- 金字塔 latent 获取：`pyramid_dit_for_video_gen_pipeline.py` L555-570
- 训练加噪：L309-347
- 推理 renoising：L725-787

### 7.5 主要结果

| 指标 | 结果 |
|------|------|
| 应用 | 视频生成 768p, ~10s |
| 质量 | 与商用模型竞争 |
| ImageNet 配置 | **无**（面向视频） |

### 7.6 与 RDD 相似之处

1. 都有多阶段分辨率递进
2. 阶段边界相似：$[0, 1/3, 2/3, 1]$ ↔ RDD $[0, 0.3, 0.6, 1.0]$
3. 都在阶段过渡时加噪
4. Pyramid-Flow 的 $\gamma$ ↔ RDD 的 $t_{\text{start}}$（都控制过渡噪声水平）

### 7.7 该方法优势（RDD 劣势）

- $\gamma$ 参数提供阶段过渡的额外自由度
- block_noise 比纯高斯更自然
- 支持时空联合金字塔（视频生成）

### 7.8 该方法劣势（RDD 优势）

- **需要显式 renoising**（信息丢失 → 重新注入）
- 各阶段独立 flow matching → 无全局数学统一性
- 不兼容 DPM-Solver
- RDD 无需 renoising：drift 在整个过程中**连续保持低分辨率信息**

### 7.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 低 | 在 noise_filling 中加入可调 $\gamma$ 参数作为消融 |
| 论文 | 明确论述：RDD 无需 renoising，drift 连续保持低分辨率信息 |

---

## 8. SwD — Scale-wise Distillation (Yandex 2025)

### 8.1 论文信息

**"Scale-wise Distillation of Diffusion Models"**, Yandex 2025.

### 8.2 方法概述

蒸馏框架：将 28 步采样划分为 4 个 boundary 区间，每区间对应不同分辨率。Teacher=原始预训练模型，Student=LoRA 微调版本。用 DMD+GAN+MMD 三重损失蒸馏。

### 8.3 理论/数学

**Boundary + Scale 配置：**

```
boundaries = [0, 7, 14, 18, 28]  (28步总共)
scales = [64, 80, 96, 128]       (latent空间尺寸)
```

| 步骤范围 | Latent 尺寸 | 像素分辨率 |
|----------|-------------|-----------|
| 0-6 | 64×64 | 512×512 |
| 7-13 | 80×80 | 640×640 |
| 14-17 | 96×96 | 768×768 |
| 18-27 | 128×128 | 1024×1024 |

**阶段间 Renoising：**

在 boundary 处，将上采样后的 latent 与噪声按 boundary sigma 混合。

**蒸馏损失组合：**

$$\mathcal{L}_{\text{total}} = \lambda_{\text{DMD}} \cdot \mathcal{L}_{\text{DMD}} + \lambda_{\text{GAN}} \cdot \mathcal{L}_{\text{GAN}} + \lambda_{\text{MMD}} \cdot \mathcal{L}_{\text{MMD}}$$

- **DMD Loss**：分数差异梯度 (Distribution Matching Distillation)
- **GAN Loss**：Transformer 判别头判断真假
- **MMD Loss**：中间特征分布距离（最大均值差异）

**与 RDD 的结构同构性：**

SwD 的 `boundaries = [0, 7, 14, 18, 28]` 与 RDD 的 `stage_lengths = [300, 300, 400]` → $T = [0, 300, 600, 1000]$ **完全同构**！都是定义"哪些时间步对应哪个分辨率"。

### 8.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 基础模型 | SD3.5-Medium |
| 微调方式 | LoRA rank=64 |
| 训练步数 | 仅 3000-4000 步 |
| 训练设备 | 8×A100 |
| 判别器 | Transformer with discriminator head |
| 多尺度采样器 | `src/flow_matching_sampler.py` |

**关键代码文件：**
- 多尺度采样器：`src/flow_matching_sampler.py` L8-230
- 蒸馏损失：`src/losses.py` L152-379
- 判别器：`src/transformer_with_discriminator.py`

### 8.5 主要结果

| 指标 | 结果 |
|------|------|
| 生成步数 | **4 步** |
| 分辨率 | 1024×1024 |
| 质量 | 接近 SD3.5-Medium 28步 |
| 训练成本 | 仅 3-4K 步 LoRA |

### 8.6 与 RDD 相似之处

1. **Boundary 结构 ≡ RDD stage_lengths（完全同构！）**
2. 都逐步增大分辨率
3. 都定义了时间步→分辨率的映射

### 8.7 该方法优势（RDD 劣势）

- 仅 3-4K 训练步（蒸馏预训练模型）
- 4 步生成（极快）
- 可应用于任何预训练 FM 模型

### 8.8 该方法劣势（RDD 优势）

- 需要预训练 teacher 模型
- Boundary 处需要 renoising（RDD 不需要）
- GAN+DMD+MMD 损失复杂度高
- 质量上限受限于 teacher

### 8.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| Future | RDD 收敛后，用 SwD 方案蒸馏到 4 步生成 |
| 论文 | 指出 SwD boundaries 与 RDD stage_lengths 的结构同构性 |

---

## 9. LowDiff (2025)

### 9.1 论文信息

**"LowDiff: Low-Resolution Diffusion for Efficient Image Generation"**, 2025.

### 9.2 方法概述

单一统一模型在多个分辨率上独立进行去噪，低分辨率结果作为高分辨率的条件（channel concatenation）。通过截断采样和层分配策略实现 50-83% 吞吐量提升。

### 9.3 理论/数学

**多分辨率训练逻辑 (Flow Matching)：**

低分辨率 (8×8)，无条件：
$$x_t^L = (1-t) \cdot \varepsilon_L + t \cdot x_L$$

高分辨率 (16×16)，条件于 $\hat{x}_L$：
$$x_t^H = (1-t) \cdot \varepsilon_H + t \cdot x_H$$

网络输入：$\text{input} = [x_t^H; \; \text{UP}(\hat{x}_L)]$（channel concat）

预测目标：$u = x - \varepsilon$（标准 velocity）

**与 RDD 的等效性分析：**

- RDD 的 drift 将 $x_L$ 信息**连续注入到 SDE 均值**中
- LowDiff 的低分辨率信息通过**网络输入的 concat channel** 注入

如果 RDD 在 $t > T_c$ 时 $\kappa(t) = 1$，则均值 = $\sqrt{\bar\alpha_t} \cdot \text{blur}(x_0)$，等效于"已生成好低分辨率，在此基础上加噪"——与 LowDiff 的 $x_t^H | \hat{x}_L$ 条件生成在直觉上类似。但 RDD 通过 SDE drift 实现，有数学连续性保证。

### 9.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 网络 | MSLightningDiT (28 层 DiT) |
| 层分配 | 8×8 用中间 14 层，16×16 用全部 28 层 |
| 分辨率标识 | ResEmbedder：加到 time + class embedding |
| 条件注入 | 低分辨率上采样后 channel concatenation |
| 截断采样 | trunc_point=0.3（低分辨率在 30% 时停止） |
| VAE | VA-VAE f16d32 (16× 下采样, 32 通道) |
| Batch | 512 |
| 训练步数 | 1,000,000 |
| 参数量 | ~675M (XL) |

**采样流程：**
1. 生成 8×8 noise，`trunc_point=0.3` 时截断
2. 以步骤 1 结果为条件，生成 16×16 完整采样
3. VAE 解码到 256×256

**关键代码文件：**
- 多尺度损失：`edm/training/loss.py` L84-170 (MultiscaleEDMLoss)
- DiT 模型：`LightningDiT/models/lightningdit.py` L457-699 (MSLightningDiT)
- 推理：`LightningDiT/inference.py` L310-360

### 9.5 主要结果

| 指标 | 结果 |
|------|------|
| ImageNet-256 | 吞吐量提升 50-83% |
| 质量 | 接近全分辨率基线 |

### 9.6 与 RDD 相似之处

1. 都用单一模型处理多分辨率
2. 都是 coarse-to-fine 生成顺序
3. **ResEmbedder** 概念对 RDD unified 模型有价值
4. 层分配策略（不同分辨率用不同数量的层）

### 9.7 该方法优势（RDD 劣势）

- 推理加速 50-83%：低分辨率步数少，高分辨率步数多
- 层分配策略计算效率高
- 架构简单（共享层 + concat）
- VA-VAE f16d32 重建质量更好

### 9.8 该方法劣势（RDD 优势）

- 各分辨率是独立扩散过程，**无数学统一性**
- Concat 条件是"事后补救"，不如 drift 的"内嵌先验"自然
- 低分辨率结果不好时，高分辨率**无法修正**（串行依赖，无修正机制）
- RDD 的 drift 在整个过程中持续修正

### 9.9 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 中 | 在 RDD unified 模型中加 ResEmbedder（告诉模型当前处理哪个阶段） |
| 低 | 考虑层分配：低分辨率阶段用较少层 |
| 论文 | 对比 RDD 隐式条件（drift）vs LowDiff 显式条件（concat） |

---

## 10. MrFlow (2025)

### 10.1 论文信息

**"MrFlow: Multi-Resolution Flow Matching for Image Generation"**, 2025.

### 10.2 方法概述

Training-free 的 4 阶段 pipeline 加速预训练 Flow Matching 模型。使用 Real-ESRGAN 做像素空间超分辨率桥接。

### 10.3 理论/数学

**4 阶段流程：**

1. 低分辨率生成：512×512, 12 步
2. Real-ESRGAN ×2 超分到 1024×1024
3. 重编码到 latent 空间 + 注入低强度噪声
4. 高分辨率精炼：从 $\sigma=0.12$，仅 1 步

**关键公式：** `direct_sigma_schedule` — 精炼时从特定 $\sigma$ 开始（非纯噪声）

### 10.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 核心代码 | `mrflow_utils.py` (28 行) |
| 适用模型 | FLUX, Qwen-Image |
| 训练需求 | **完全不需要** |
| 超分模型 | Real-ESRGAN |

### 10.5 主要结果

| 指标 | 结果 |
|------|------|
| 加速比 | ~2× (1024×1024 生成) |
| 质量 | 接近原始全步数 |

### 10.6 与 RDD 相似之处

- 都认识到高分辨率应从低分辨率 + 噪声开始，而非纯噪声

### 10.7 优势与劣势

- **优势**：不需训练，简单，适配任何模型
- **劣势**：质量受限于 ESRGAN + 预训练模型；无数学框架

### 10.8 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 低 | 借鉴"从特定 σ 开始精炼"的思路用于 RDD 最后阶段 |

---

## 11. XCube (CVPR 2024)

### 11.1 论文信息

**"XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies"**, CVPR 2024.

### 11.2 方法概述

两阶段 3D latent diffusion：Dense 16³ → Sparse 128³。两个独立模型分别处理粗粒度和细粒度。

### 11.3 理论/数学

**扩散公式：**
- 连续 DDPM，**v-prediction**：$v = \alpha'(t) \cdot x_0 + \sigma'(t) \cdot \varepsilon$
- $T=1000$，linear beta schedule (0.0001→0.02)
- 推理：DDIM 100 步

### 11.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 粗粒度 | Dense UNet, 16³ |
| 细粒度 | Sparse UNet, 128³ |
| 训练方式 | 两个模型独立训练 |
| 推理 | DDIM 100 步 |

### 11.5 主要结果

高质量 3D 形状生成，无 ImageNet 实验。

### 11.6 与 RDD 对比

- **相似**：多分辨率级联概念
- **RDD 优势**：单一 Markov 链 vs 独立两阶段 cascade
- **可借鉴**：v-prediction 可能比 ε-prediction 更稳定

### 11.7 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 低 | 考虑 v-prediction 消融实验 |

---

## 12. OctFusion (SGP 2025)

### 12.1 论文信息

**"OctFusion: Octree-based Diffusion Models for 3D Shape Generation"**, SGP 2025.

### 12.2 方法概述

单一模型内 LR + HR 两个 UNet 分支，通过八叉树结构连接。LR 用 x0-prediction，HR 用 epsilon-prediction。

### 12.3 理论/数学

**扩散公式：**
- 连续高斯扩散，log-SNR 参数化
- `noise_level = beta_linear_log_snr(times)`，$t \sim \text{Uniform}(0, 1)$
- $\text{noised\_data} = \alpha \cdot \text{data} + \sigma \cdot \text{noise}$
- 采样：DDIM 变体，LR 输出做 sign clipping ($t < 0.7$)

**训练流程：**
1. 先训练 LR 分支
2. 冻结 LR，训练 HR 分支（融合 LR 特征）

### 12.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 网络 | Graph UNet (图卷积) on octree |
| LR 分支 | x0-prediction |
| HR 分支 | epsilon-prediction, 融合 LR 特征 |
| HR 融合 | `graph_unet_hr.py` L257-262 |

### 12.5 主要结果

3D 形状生成，八叉树效率优化。

### 12.6 与 RDD 对比

- **相似**：单一模型处理多分辨率；特征融合
- **RDD 优势**：全程联合训练 vs 先 LR 再冻结训 HR
- **可借鉴**：self-conditioning（用上一步 pred 作为条件），文献报告 +5-15% FID

### 12.7 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 低 | Self-conditioning 实验（+5-15% FID 潜力） |

---

## 13. PDD — Pyramid Discrete Diffusion (ECCV 2024)

### 13.1 论文信息

**"Pyramid Discrete Diffusion for 3D Generation"**, ECCV 2024.

### 13.2 方法概述

离散扩散 (D3PM) + 空间金字塔分阶段超分：32³ → 64³ → 256³。3 个独立模型。

### 13.3 理论/数学

**离散扩散 (D3PM / Multinomial)：**

$$q(x_t | x_{t-1}) = \alpha_t \cdot x_{t-1} + (1 - \alpha_t) \cdot \frac{1}{K}$$

每步以概率 $1-\alpha_t$ 替换为均匀分布。

**仅 100 步扩散**，cosine beta schedule。

### 13.4 工程实现

| 维度 | 具体配置 |
|------|----------|
| 模型数 | 3 个独立模型 |
| 第 1 阶段 | 无条件生成 32³ |
| 第 2 阶段 | 条件于 32³，生成 64³ |
| 第 3 阶段 | 条件于 64³，生成 256³ |
| 扩散步数 | 仅 100 步 |

### 13.5 主要结果

3D 语义体素生成。

### 13.6 与 RDD 对比

- **相似**：多阶段渐进分辨率；极少步数（100 步）暗示强结构先验有助于减少步数
- **RDD 优势**：单一 Markov 链 vs 3 个独立模型级联；连续 vs 离散

### 13.7 RDD 应做的事情

| 优先级 | 行动项 |
|--------|--------|
| 低 | 如果 drift 提供了足够强的先验，尝试将 diffusion_steps 从 1000 减少到 500 |

---

## 14. 综合对比大表

### 14.1 方法论对比

| 方法 | 框架 | 前向修改? | 多尺度来源 | 采样器 | 模型数 | DPM-Solver |
|------|------|-----------|-----------|--------|--------|-----------|
| **RDD (ours)** | DDPM/SDE | ✓ drift | SDE drift | DPM-Solver | 1 | ✓ |
| Relay | EDM/VE-SDE | ✓ blur | 级联 | 自定义 DCT | 2 | ✗ |
| LapFlow | FM/ODE | ✓ 分段 | Laplacian 分解 | ODE solver | 1 (MoT) | ✗ |
| DiMR | VP-SDE | ✗ | 多尺度架构 | DPM-Solver | 1 | ✓ |
| DCTdiff | VP-SDE | ✓ SNR | DCT 空间 | DPM-Solver | 1 | ✓ |
| SPEED | FM | ✗ | 推理时频谱扩展 | 原始 solver | 0 | — |
| Pyramid-Flow | FM | ✓ 多阶段 | 金字塔+renoise | Euler ODE | 1 | ✗ |
| SwD | FM | ✗ | 蒸馏+推理策略 | Euler 4 步 | 1+LoRA | ✗ |
| LowDiff | FM/EDM | ✗ | 条件级联 | Euler ODE | 1 | ✗ |
| MrFlow | FM | ✗ | 推理+ESRGAN | 原始 solver | 0 | — |
| XCube (3D) | DDPM | ✗ | 独立 2 阶段 | DDIM | 2 | ✗ |
| OctFusion (3D) | Gaussian | ✗ | LR→HR 特征融合 | DDIM 变体 | 1 (2 分支) | ✗ |
| PDD (3D) | D3PM 离散 | ✗ | 独立 3 阶段 | 离散采样 | 3 | ✗ |

### 14.2 RDD 的独特定位

1. **唯一**在单一 Markov 链中通过连续 drift 实现多分辨率编码
2. **唯一**同时满足：修改前向过程 + 保持闭式后验 + 兼容 DPM-Solver
3. 不需要 renoising（Pyramid-Flow、SwD 都需要）
4. 不需要多模型级联（Relay、XCube、PDD 都需要）
5. 不需要显式频率分解（LapFlow、DCTdiff 都需要）
6. 单一简洁的预测目标 $\tilde\varepsilon$（不像 f-DM 需要双重预测）

### 14.3 ImageNet 实验配置对比

| 方法 | 分辨率 | Batch | GPU 数 | 总步数 | LR | 参数量 | FID |
|------|--------|-------|--------|--------|-------|--------|-----|
| DiMR-XL/2R | 256 | 1024 | 16 | 1,000,000 | 2e-4 | 505M | **1.70** |
| DiMR-G/2R | 256 | 1024 | 16 | 1,000,000 | 2e-4 | 1.06B | **1.63** |
| Relay-Stage2 | 256 | 4096 | 8 | ~305K | 1e-4 | ~400M | 2.64 |
| LapFlow (DiT-L) | 256 | 256×16 | 16 | ~1000ep | 1e-4 | ~460M | ~8 |
| DCTdiff | 64 | 1024 | 4 | 500,000 | 3e-4 | ~120M | — |
| LowDiff-XL | 256 | 512 | — | 1,000,000 | 2e-4 | ~675M | — |
| DiT-XL/2 (官方) | 256 | 256 | — | 7,000,000 | 1e-4 | 675M | 2.27 |
| **RDD-XL/2 (ours)** | 256 | 256 | 8 | 目标 700K | 1e-4 | ~675M | TBD |

---

## 15. RDD 行动计划

### P0：必做（低工作量，高回报）

| # | 行动项 | 依据 |
|---|--------|------|
| 1 | **频域 SNR 公式推导** | 证明 drift = 频率自适应 SNR 衰减，理论贡献 |
| 2 | **$\gamma=0$ 消融实验** | 证明 drift 必要性，核心消融 |
| 3 | **DPM-Solver 10/20/50 步 FID 表** | 核心差异化卖点 vs 所有竞品 |

### P1：应做（中工作量）

| # | 行动项 | 来源 |
|---|--------|------|
| 4 | 用 SPEED 理论确定最优 stage_lengths | SPEED §6 |
| 5 | 改进 noise_filling：频谱扩展（仅高频加噪）替代全频段加噪 | SPEED + Relay |
| 6 | 加入 per-stage loss weighting | LapFlow §3 |
| 7 | 加入分辨率嵌入 ResEmbedder | LowDiff §9 |

### P2：加分项（高工作量）

| # | 行动项 | 来源 |
|---|--------|------|
| 8 | DiMR 架构 + RDD drift 融合 | DiMR §4 |
| 9 | Flow Matching 版本 RDD（理论放 Appendix） | LapFlow §3 |
| 10 | 增大 batch size（梯度累积到 1024） | Relay/DiMR |
| 11 | Self-conditioning | OctFusion §12 |
| 12 | Block Noise 实验 | Relay §2 / Pyramid-Flow §7 |

---

> **核心论点总结：** RDD 是目前唯一在单一 Markov 链中通过连续 drift 编码多分辨率、保持闭式后验、且兼容 DPM-Solver 的简洁统一方案。

---

## 16. FlowAR (ICML 2025)

### 16.1 论文信息

**"FlowAR: Scale-wise Autoregressive Image Generation Meets Flow Matching"**, ICML 2025. FID=1.65 on ImageNet 256.

### 16.2 方法概述

FlowAR 将多尺度图像生成重新定义为自回归 (AR) 过程：从最低分辨率开始，逐步预测更高分辨率的图像，每个尺度使用 Flow Matching (FM) 在连续空间中建模。

- 构建 scale tokens 序列：$[z_1, z_2, ..., z_K]$，分辨率逐级翻倍
- 使用 GPT 风格因果 Transformer 建模跨尺度依赖
- 每个 scale 内使用 10 步 Flow Matching 从噪声生成对应分辨率的 latent
- VAE 用 MAR 的定制 VAE (16 通道)

### 16.3 关键结果

| 模型 | ImageNet 256 FID↓ | NFE | 参数量 |
|------|-------------------|-----|--------|
| FlowAR-B | 3.93 | — | — |
| FlowAR-L | 2.64 | — | — |
| FlowAR-XL | **1.65** | ~50 (5 scales × 10 steps) | ~1B |
| DiT-XL/2 | 2.27 | 250 | 675M |
| RDD-XL (ours) | TBD | 10-50 (DPM-Solver) | ~675M |

### 16.4 与 RDD 对比

| 维度 | FlowAR | RDD |
|------|--------|-----|
| 多尺度方式 | 显式 AR：每个 scale 独立 FM | 隐式：单一 Markov 链 + drift |
| 采样步数 | ~50 (5×10) | 10-50 (DPM-Solver 全程) |
| 跨尺度依赖 | GPT 因果注意力（显式建模） | drift 自然编码（隐式） |
| 模型复杂度 | AR + FM 双框架 | 单一 DDPM 框架 |
| 理论简洁性 | 组合式设计 | 统一 SDE |

### 16.5 可借鉴技术

- **16通道 VAE**：FlowAR 使用 MAR 的 16 通道 VAE（vs DiT 的 4 通道），latent 信息更丰富，FID 改善明显。RDD 可尝试替换 VAE。
- **尺度依赖的 CFG 策略**：不同 scale 使用不同 CFG 强度（低分辨率高 CFG，高分辨率低 CFG）。

---

## 17. PixelFlow (2025)

### 17.1 论文信息

**"PixelFlow: Pixel-Space Cascaded Flow Matching"**, 2025. FID=1.98 on ImageNet 256.

### 17.2 方法概述

PixelFlow 在像素空间实现级联 Flow Matching：
- 4 个分辨率级别（32→64→128→256）
- 每个级别用 Flow Matching 从上一级别的输出生成更高分辨率
- 使用 **Block Noise** 作为初始噪声（空间相关高斯噪声）
- **Renoising 机制**：低分辨率输出上采样后加噪，再去噪精细化

### 17.3 关键结果

| 模型 | ImageNet 256 FID↓ | 备注 |
|------|-------------------|------|
| PixelFlow-XL | **1.98** | 像素空间，无 VAE |
| DiT-XL/2 | 2.27 | latent 空间 |
| FlowAR-XL | 1.65 | latent + AR |

### 17.4 与 RDD 对比

| 维度 | PixelFlow | RDD |
|------|-----------|-----|
| 空间 | 像素空间（无 VAE） | latent 空间 |
| 过渡机制 | 离散 renoising（上采样+加噪+去噪） | 连续 drift（无显式过渡） |
| Block Noise | 核心组件（空间相关噪声） | 可选的 noise_filling 策略 |
| 分辨率跳变 | 硬切换（32→64→128→256） | 连续过渡 |
| DPM-Solver | 不兼容（级联打断 ODE） | 完全兼容 |

### 17.5 可借鉴技术

- **Block Noise**：PixelFlow 证明 block noise（空间相关噪声）在级联采样的过渡阶段优于白噪声。RDD 的 noise_filling 步骤可直接采用。
- **Renoising 系数设计**：信号系数 + 噪声系数的平衡公式可参考用于 RDD 的采样过渡。

---

## 18. FreCaS (NeurIPS 2024)

### 18.1 论文信息

**"FreCaS: Efficient Higher-Resolution Image Generation via Frequency-Aware Cascaded Sampling"**, NeurIPS 2024.

### 18.2 方法概述

FreCaS 是一个 **training-free** 的加速采样框架：
- 在低分辨率空间执行大部分去噪步骤
- 仅在最后若干步切换到全分辨率处理高频细节
- **FA-CFG**：频率自适应 Classifier-Free Guidance，对不同频率带使用不同 CFG 强度

### 18.3 FA-CFG 核心公式

$$\hat\epsilon = \epsilon_{\text{uncond}} + w(f) \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$$

其中 $w(f)$ 是频率依赖的引导权重：
- 低频分量：$w_{\text{low}} = w_{\text{base}} \times 1.5$（更强引导，确保语义正确）
- 高频分量：$w_{\text{high}} = w_{\text{base}} \times 0.5$（弱引导，避免过饱和/artifacts）

### 18.4 关键结果

- 在 SDXL/SD3 等预训练模型上 training-free 应用
- 1.5-2× 加速，FID 不降反升（FA-CFG 改善高频质量）
- 高分辨率生成（1024→2048）效果尤其显著

### 18.5 与 RDD 对比 / 可借鉴

**FA-CFG 是 RDD 的零成本正交改进：**
- RDD 本身已经按 stage 区分了低频/高频的生成时机
- 直接按 RDD 的 stage 差异化 CFG 强度：
  - Stage 1（低分辨率）：高 CFG（w=7-9），强化全局语义
  - Stage 3（高分辨率）：低 CFG（w=2-4），保护高频细节
- 实现零成本，只需修改采样代码中的 CFG scale

---

## 19. f-DM (ICLR 2023)

### 19.1 论文信息

**"f-DM: A Multi-stage Diffusion Model via Progressive Signal Transformation"**, Jiatao Gu et al. (Apple), ICLR 2023.

### 19.2 方法概述

f-DM 将一组变换函数整合进扩散过程，每个阶段内信号从精细过渡到粗糙：
- 时间区间 $[0,1]$ 划分为 $K+1$ 阶段
- 信号变换序列：$f = \{f_0, ..., f_K\}$，可以是下采样、模糊、或 VQVAE 编码
- 阶段内插值 + 阶段间跳变

三种实例化：
- **f-DM-DS**：双线性下采样（256→128→64→16）
- **f-DM-Blur**：高斯模糊 / UP(DOWN(x))
- **f-DM-VQVAE**：预训练 VAE 编码器

### 19.3 数学公式

**前向分布（阶段内）：**
$$q(x_t | x_0) = \mathcal{N}(x_t; \bar{x}_t, \sigma_t^2 I)$$

其中 $\bar{x}_t$ 是 $x_k$ 和 $\hat{x}_k$ 的时间依赖加权插值。

**训练目标（双重预测）：**
$$\mathcal{L} = \|\epsilon_\theta - \epsilon\|^2 + \lambda \|\delta_\theta - \delta_t\|^2$$

- $\epsilon$：注入噪声
- $\delta_t = x_k - \bar{x}_t$：信号变换导致的退化量
- 网络需要**同时预测噪声和退化量**

### 19.4 关键结果

| 数据集 | DDPM baseline | f-DM-DS | f-DM-VQVAE |
|--------|--------------|---------|------------|
| FFHQ 256 | 11.6 | 10.8 | — |
| ImageNet 256 | 10.9 | 8.2 | 6.8 |

### 19.5 与 RDD 的核心差异

| 维度 | f-DM | RDD |
|------|------|-----|
| 前向过程 | 多个离散阶段，每阶段独立变换+噪声调度 | 单一连续 Markov 链 |
| 阶段切换 | 硬边界，可能改变维度 | 无显式边界，drift 连续变化 |
| 噪声调度 | 多个独立 schedule + γ_k 重缩放 | 单一全局 schedule |
| 预测目标 | 双重预测（ε + δ） | 单一修正噪声 ε̃ |
| DPM-Solver | ❌ 不兼容（阶段边界打断 ODE） | ✅ 完全兼容 |
| 超参数 | 需为每个变换调节 γ_k | κ(t) 线性调度，自动处理 |

### 19.6 f-DM 的局限

1. 阶段边界不连续性 → 无法使用高阶 ODE solver
2. 双重预测增加训练复杂度和超参数
3. 维度变化需要 stage-specific adapters
4. 逆映射 $g_k$ 是近似的，有不可恢复的信息损失

**RDD 的优势**：连续 drift 在统一框架中解决了 f-DM 的所有局限性。

---

## 20. DFM — Diffusion Flow Matching (NeurIPS 2025)

### 20.1 论文信息

**"DFM: Interpolant-based Diffusion-Flow Matching via Laplacian Pyramid"**, NeurIPS 2025.

### 20.2 方法概述

DFM 对 Laplacian 金字塔的各层独立做 Flow Matching：
- 将图像分解为 Laplacian 金字塔残差 $\{L_0, L_1, ..., L_K\}$
- 每层独立学习一个 flow：$x_t^{(k)} = (1-t) \cdot \epsilon + t \cdot L_k$
- 各层无内建耦合——独立训练、独立采样
- 使用共享参数的 UNet（通过 scale embedding 区分层）

### 20.3 与 RDD 对比

| 维度 | DFM | RDD |
|------|-----|-----|
| 层间依赖 | **无**（各层独立 FM） | **有**（drift 自然编码因果依赖） |
| 采样 | 各层并行生成 | 串行（低频→高频） |
| 条件建模 | 无（各层独立） | 低频自动作为高频的条件 |
| 理论依据 | 简单拼接 | 单一 Markov 链 |

**RDD 优势**：DFM 各层独立意味着无法利用低频信息指导高频生成——RDD 的 drift 天然编码了这种因果依赖。

---

## 21. Matryoshka Diffusion Models (ICLR 2024)

### 21.1 论文信息

**"Matryoshka Diffusion Models"**, Jiatao Gu et al. (Meta), ICLR 2024.

### 21.2 方法概述

在扩展空间 (extended space) 中联合多分辨率扩散：
- 隐变量 $z_t = (z^1_t, ..., z^R_t)$，每个对应一个分辨率
- **NestedUNet**：低分辨率参数嵌套在高分辨率模型内部
- 所有分辨率**并行去噪**
- 渐进训练：从低分辨率开始，逐步加入更高分辨率

### 21.3 数学公式

**多分辨率前向：**
$$q(z^r_t | x) = \mathcal{N}(\alpha^r_t \cdot D_r(x), (\sigma^r_t)^2 I)$$

**联合训练损失：**
$$\mathcal{L} = \sum_{r=1}^{R} \omega^r_t \cdot \|x^r_\theta(z_t, t) - D_r(x)\|^2$$

权重默认 $\omega^r_t / \omega^R_t = N_R / N_r$（按维度比例）。

### 21.4 关键结果

| 数据集 | FID | 备注 |
|--------|-----|------|
| ImageNet 256 | 3.51 | CFG=1.5，class-cond |
| CC12M 1024² | 定性展示 | 像素空间，单模型 |
| WebVid 16帧视频 | 定性展示 | 文本到视频 |

### 21.5 与 RDD 对比

| 维度 | Matryoshka | RDD |
|------|-----------|-----|
| 多分辨率方式 | 显式扩展空间，R 条并行链 | 单一链，drift 隐式编码 |
| 推理 | 所有分辨率同时生成 | 低频→高频顺序生成 |
| 噪声调度 | 分辨率相关 shifted cosine | 统一调度 + κ(t) drift |
| 架构 | NestedUNet（嵌套权重共享） | 标准 DiT |
| 训练 | 渐进训练（低→高） | 端到端 |

### 21.6 可借鉴

- **渐进训练策略**：先训低分辨率部分，再"生长"到全分辨率，加速收敛
- **分辨率相关噪声调度 (shifted cosine)**：高分辨率使用更大 shift

---

## 22. TPDiff — Temporal Pyramid Diffusion (2025)

### 22.1 论文信息

**"TPDiff: Temporal Pyramid Video Diffusion Model"**, arXiv:2503.09566, 2025.

### 22.2 方法概述

时间维度的渐进分辨率扩散：
- 将扩散过程分为 K=3 个阶段
- 高噪声阶段用低帧率（原始/4），低噪声阶段用全帧率
- 单一模型处理所有帧率
- **核心洞察**：高噪声时帧间时序关系极弱，无需全帧率

### 22.3 数学公式

**阶段内 Flow Matching：**
$$x_t = (1-t')\hat{x}_{e_k} + t'\hat{x}_{s_k}$$

训练目标：$dx_t/dt' = \hat{x}_{s_k} - \hat{x}_{e_k}$

**阶段衔接（Renoising）：**
$$\hat{x}_{s_{k-1}} = \frac{\sqrt{2}\gamma_{s_k}}{\sigma_{s_k} + \sqrt{2}\gamma_{s_k}} \text{Up}(\hat{x}_{e_k}) + \frac{\sqrt{2}\sigma_t}{2} n'$$

其中 $n'$ 使用负相关噪声。

**计算量：** 3阶段金字塔注意力计算降至 $0.44T^2$（节省 56%）。

### 22.4 关键结果

- 训练加速 **2×**，推理加速 **1.5-1.7×**
- VBench 总分优于 vanilla diffusion
- 仅训练 5000 步即可生成时间稳定视频

### 22.5 与 RDD 结合

**TPDiff (时间金字塔) + RDD (空间金字塔) = 时空双金字塔：**
- 高噪声：低空间分辨率 + 低帧率
- 中噪声：中空间分辨率 + 中帧率
- 低噪声：全空间分辨率 + 全帧率

RDD 的 drift 可替代 TPDiff 的 renoising——连续过渡无需显式注噪。

### 22.6 可借鉴技术

- **数据-噪声对齐（Data-Noise Alignment）**：匈牙利算法最小化 batch 内数据-噪声距离，可直接应用于 RDD 训练
- **时间 drift 设计**：$d_{\text{temporal}} = \text{TemporalUp}(\text{TemporalDown}(x_0)) - x_0$
- **负相关噪声**：阶段衔接时确保新帧与已有帧有差异

---

## 23. Open-Sora Plan (2025)

### 23.1 论文信息

**"Open-Sora Plan: Open-Source Large Video Generation Model"**, 2025.

### 23.2 方法概述

完整开源视频生成框架，三大组件：
- **WF-VAE**：Wavelet-Flow VAE，压缩率 4×8×8 (时间×高×宽)
- **Skiparse Denoiser**：3D DiT + Skiparse Attention + 3D RoPE，2.7B 参数
- **Condition Controllers**：mT5-XXL + Inpainting 式图像注入

### 23.3 架构细节

**Skiparse Attention（核心创新）：**
- 在 Full 3D 和 2+1D Attention 间取平衡
- sparse ratio k=4，序列长度降为 1/k
- 首尾各2层保持 Full 3D Attention
- 交替执行 Single Skip 和 Group Skip 两种模式

**训练策略：**
- Min-Max Token 策略：多分辨率数据在同一 bucket 聚合
- 自适应梯度裁剪：EMA 跟踪 + 3-sigma 异常检测
- v-prediction + zero terminal SNR + min-snr 加权(γ=5.0)
- 渐进训练：Image → 低分辨率 Video → 高分辨率 Video

### 23.4 与 RDD 的关系

STDiT/Skiparse Denoiser 可直接作为 RDD 视频版的 backbone：
- 3D RoPE 天然支持可变序列长度
- Skiparse 的稀疏模式适合 RDD 高分辨率阶段（高频信息空间相关性局部）
- RDD 需推导 v-prediction 下的 drift 修正：$\tilde{v} = v + \text{drift 修正项}$

### 23.5 可借鉴技术

| 优先级 | 技术 | 用途 |
|--------|------|------|
| P0 | Skiparse Attention | 降低视频生成计算量 |
| P0 | WF-VAE + Causal Cache | 高效视频编解码 |
| P1 | Min-Max Token + Bucket Sampler | 多分辨率高效训练 |
| P1 | 自适应梯度裁剪 | 训练稳定性 |
| P2 | 数据 curation pipeline | 数据质量保证 |

---

## 24. Blockwise Parallel Flow Matching (2025)

### 24.1 论文信息

**"Blockwise Parallel Transformers for Efficient Flow Matching"**, 2025.

### 24.2 方法概述

将时间轴分为多个 block，每个 block 使用专用小模型并行生成：
- 全局步数不变，但各 block 可并行执行
- 每个 block 的模型较小（参数共享或独立）
- 本质是 Flow Matching 版的分段并行采样

### 24.3 与 RDD 对比

- Blockwise FM 的分段设计与 RDD 的 stage_lengths 类似
- 但 Blockwise FM 的各段独立建模（无跨段依赖），RDD 通过 drift 自然编码跨段依赖
- 并行采样的思路可用于 RDD 的未来加速

---

## 25. 更新后的实验清单

### 已完成

| # | 实验 | 状态 | 结果 |
|---|------|------|------|
| A1 | RDD-XL 300-300-400 训练 | ✅ 400K done | baseline |
| A2 | 频域 SNR 可视化 | ✅ | 4张图生成 |
| A3 | Ablation batch 15组 | 🔄 运行中 | `ablation_batch_400k.jsonl` |
| A4 | Unified L/XL 训练 | 🔄 运行中 | 目标 700K |

### P0：必做（低工作量，高回报）

| # | 实验 | 依据 | 详细 |
|---|------|------|------|
| 1 | **频域 SNR 对比图** | §1.5 + Relay + DCTdiff | 加入 Relay/f-DM/DCTdiff 的 SNR 曲线对比 → Figure 4 |
| 2 | **DPM-Solver 10/20/50 步 FID 表** | 核心差异化 | 证明 RDD 兼容高效 solver |
| 3 | **γ=0 消融**（去掉 drift） | 核心消融 | 证明 drift 必要性 |
| 4 | **FA-CFG 按 stage 差异化 CFG** | FreCaS §18 | Stage1 高CFG=7，Stage3 低CFG=3，零代码成本 |

### P1：应做（中工作量）

| # | 实验 | 依据 | 详细 |
|---|------|------|------|
| 5 | SPEED 最优 stage_lengths | SPEED §6 | 用功率谱公式计算理论最优阶段划分 |
| 6 | 频谱渐进 noise_filling | SPEED + PixelFlow §17 | DCT 域仅注入高频噪声 / Block Noise |
| 7 | Block Noise 采样 | PixelFlow §17 + Relay §2 | 空间相关噪声，无需重训练 |
| 8 | 数据-噪声对齐 | TPDiff §22 | 匈牙利算法，实现一行代码 |

### P2：加分项（高工作量，高回报）

| # | 实验 | 依据 | 详细 |
|---|------|------|------|
| 9 | Flow Matching 版 RDD | LapFlow §3 + DFM §20 | drift 转为速度场修正，理论 Appendix |
| 10 | 增大 batch (累积到 1024) | Relay / DiMR | 对 FID 提升明确 |
| 11 | SwD 蒸馏加速 | SwD §8 | MMD 损失 per-stage 蒸馏，4步采样 |
| 12 | DiMR 架构融合 | DiMR §4 | Multi-Resolution Network + RDD drift |

### P3：视频扩展（长期）

| # | 实验 | 依据 | 详细 |
|---|------|------|------|
| 13 | 时空双金字塔 RDD | TPDiff §22 + Pyramid-Flow §7 | 空间 drift + 时间 drift |
| 14 | STDiT backbone 集成 | Open-Sora §23 | Skiparse Attention + 3D RoPE |
| 15 | 渐进训练：Image→Video | Matryoshka §21 + Open-Sora §23 | 先训图像 RDD → 扩展视频 |

---

## 26. Frequency Diffusion (ICLR 2025 Workshop)

### 26.1 论文信息

**"Shaping Inductive Bias in Diffusion Models Through Frequency-Based Noise Control"**, Jiralerspong et al. (Mila, Yoshua Bengio), ICLR 2025 DeLTA Workshop.

### 26.2 方法概述

通过操控噪声的**频率成分**来引导模型学习特定频段信息：
- 在频域生成高斯噪声 → 频率加权 $w(f)$ → 逆变换回空间域
- 加权函数 $w(f)$ 决定不同频率的噪声强度
- 标准白噪声是 $w(f)=1$ 的特殊情况

### 26.3 数学公式

**频率加权噪声：**
$$\epsilon^{(w)} = \Re\left[\mathcal{F}^{-1}\left(N_{\text{freq}} \odot w(\mathbf{f})\right)\right]$$

**频率依赖 SNR：**
$$\text{SNR}_{\text{FD}}(f, t) = \frac{\bar{\alpha}_t}{(1-\bar{\alpha}_t) \cdot w(f)^2}$$

### 26.4 与 RDD 的殊途同归

**RDD 的频率自适应 SNR：**
$$\text{SNR}_{\text{RDD}}(f, t) = \frac{\bar{\alpha}_t \cdot [(1-\kappa(t)) + \kappa(t) \cdot LP(f)]^2}{1 - \bar{\alpha}_t}$$

**等价条件：** 若令 $w(f)^2 = \frac{1}{[(1-\kappa(t)) + \kappa(t) \cdot LP(f)]^2}$，两者 SNR 完全等价。

**关键差异：**
| 维度 | Frequency Diffusion | RDD |
|------|---------------------|-----|
| 机制 | 改变噪声功率谱（噪声侧） | 改变信号均值（信号侧） |
| 时间依赖 | 恒定 w(f)（论文未做时变） | κ(t) 随时间变化 |
| 实现 | 显式频域操控 | drift 隐式实现 |

**理论互证价值：** Frequency Diffusion（Bengio 组）独立验证了“不同频率需要不同 SNR 调度”的基本结论。RDD 可被视为 Frequency Diffusion 的**时变推广**——且 FD 论文 Discussion 中提到的“dynamic frequency noise strategies”恰好就是 RDD 已实现的。

### 26.5 论文引用方式

> "Jiralerspong et al. (2025) independently showed that frequency-dependent noise scheduling improves diffusion model performance, supporting our theoretical derivation that RDD’s drift implicitly implements a time-varying frequency-adaptive SNR schedule."

---

## 27. Hierarchical Rectified Flow (ICLR 2025)

### 27.1 论文信息

**"Towards Hierarchical Rectified Flow"**, Zhang et al. (UIUC), ICLR 2025. 后续扩展：Mini-Batch Coupling (2507.13350).

### 27.2 方法概述

在速度空间中再应用一次 Rectified Flow 来建模多模态速度分布：
- **D=1 (标准 RF)**：学习速度场 $v(x_t, t)$
- **D=2 (HRF2)**：学习加速度场 $a(x_t, t, v_\tau, \tau)$
- 通过嵌套 ODE 积分采样：先积速度（τ轴），再积位置（t轴）
- 路径可以交叉（建模完整多模态速度分布而非均值）

### 27.3 关键公式

**HRF2 训练目标：**
$$\mathcal{L} = \mathbb{E}\left[\|(x_1 - x_0 - v_0) - a_\theta(x_t, t, v_\tau, \tau)\|^2\right]$$

**Mini-Batch OT Coupling：**
用最优传输耦合 $(x_0, x_1)$ 对，减少速度分布多模态性 → 更直轨迹 → 更少 NFE。

### 27.4 与 RDD 的关系

| 维度 | RDD | HRF |
|------|-----|-----|
| “层次”含义 | 空间分辨率（粗→细） | 运动学阶次（位置→速度→加速度） |
| NFE 结构 | 各阶段之和 (J₁+J₂+J₃) | 各层之积 (J×L) |
| Markov 性 | 单一连续 Markov 链 | 耦合多 ODE |
| DPM-Solver | ✅ 兼容 | ❌ 不兼容 |

### 27.5 可借鉴技术

| 优先级 | 技术 | 实现难度 |
|--------|------|---------|
| P0 | Mini-Batch OT Data Coupling | 低（10行代码，`pip install pot`） |
| P1 | 速度分布分析（诊断 stage 划分） | 低 |
| P2 | 二阶动力学/动量采样器 | 中 |

---

> **更新后核心论点：** RDD 是目前唯一在单一 Markov 链中通过连续 drift 编码多分辨率、保持闭式后验、且兼容 DPM-Solver 的简洁统一方案。相比 FlowAR (FID=1.65) 和 PixelFlow (FID=1.98) 的组合式设计，RDD 的理论优雅性（单一 SDE + 闭式后验 + DPM-Solver 兼容）是核心差异化。通过整合 FA-CFG (FreCaS)、SPEED 最优调度、Block Noise (PixelFlow/Relay) 等正交技术，RDD 有望在保持理论简洁性的同时达到 SOTA FID。
