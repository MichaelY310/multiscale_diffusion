#set document(title: "RDD 相关工作完整调研报告", author: "Yang Yuan")
#set page(paper: "a4", margin: (x: 1.8cm, y: 2cm))
#set text(size: 9.5pt, lang: "zh", font: ("Noto Sans CJK SC", "DejaVu Sans"))
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.7em)
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(1em)
  text(size: 16pt, weight: "bold", fill: rgb("#1a1a2e"))[#it]
  v(0.5em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 13pt, weight: "bold", fill: rgb("#16213e"))[#it]
  v(0.3em)
}
#show heading.where(level: 3): it => {
  v(0.5em)
  text(size: 11pt, weight: "bold", fill: rgb("#0f3460"))[#it]
  v(0.2em)
}
#show raw.where(block: true): it => block(fill: luma(245), inset: 8pt, radius: 4pt, width: 100%)[#it]

// Title page
#align(center)[
  #v(3cm)
  #text(size: 24pt, weight: "bold")[RDD 相关工作完整调研报告]
  #v(1cm)
  #text(size: 14pt)[Residual Drift Diffusion vs. 12 种多尺度扩散方法]
  #v(1cm)
  #text(size: 11pt, fill: gray)[详细方法理论 · 工程实现流程图 · 代码解析 · Ablation 分析 · 深度对比]
  #v(2cm)
  #text(size: 11pt)[最后更新：2026-07-25]
  #v(5cm)
]

#pagebreak()

// Table of contents
#outline(indent: auto, depth: 3)

#pagebreak()

= RDD 核心方法回顾

== 核心思想

RDD (Residual Drift Diffusion) 在标准 DDPM 的前向过程中引入一个与 $x_0$ 相关的 *连续漂移项 (drift)*，使得扩散路径的均值在时间推进中从原始图像 $x_0$ 逐渐过渡到其低分辨率版本 $"blur"(x_0)$。这实现了在 *单一 Markov 链* 中编码多分辨率信息。

== 数学公式

=== 前向 SDE

$ d x = (-frac(1,2) beta(t) x + gamma dot f(x_0, t)) d t + sqrt(beta(t)) d W $

其中 $f(x_0, t) = sqrt(overline(alpha)_t) dot kappa(t) dot d$。

=== 离散前向分布

$ q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) dot x_0 + sqrt(overline(alpha)_t) dot kappa(t) dot d, quad (1 - overline(alpha)_t) bold(I)) $

=== Drift 定义

$ d = "UP"("DOWN"(x_0)) - x_0 $

这等价于负的 Laplacian 高频残差：$d = -L_0$。

直觉：$d$ 表示"低分辨率版本与原图的差异"，即原图中被降采样丢失的高频信息（取反）。

=== 时间调度

$ kappa(t) = frac(t_"global" - T_c, T_(c+1) - T_c) in (0, 1) $

每个阶段内线性增长。阶段划分：

$ "stage\_lengths" = (300, 300, 400) arrow.r "全局时间边界" (0, 300, 600, 1000) $

=== 预测目标

$ tilde(epsilon) = epsilon + sqrt(frac(overline(alpha)_t, 1 - overline(alpha)_t)) dot kappa(t) dot d $

网络只需预测这个 *单一修正噪声* $tilde(epsilon)$。

=== 后验分布

$q(x_(t-1) | x_t, x_0)$ 仍为高斯分布，均值额外包含 drift 修正项：

$ mu_"posterior" = mu_"standard" + "posterior\_mean\_coef\_drift" dot d $

=== DPM-Solver 修正步

$ x_(t-1) = "standard\_dpm\_step"(x_t) + sqrt(overline(alpha)_(t-1)) dot (kappa(t-1) - kappa(t)) dot d $

== 频域等效 SNR（关键理论推导）

设 $x_0$ 的 DCT 分解为 $hat(x)_0 (f)$，drift 在频域为：

$ hat(d)(f) = (L P(f) - 1) dot hat(x)_0 (f) $

RDD 前向均值在频域：

$ mu_t (f) = sqrt(overline(alpha)_t) dot ((1 - kappa(t)) + kappa(t) dot L P(f)) dot hat(x)_0 (f) $

有效 SNR：

$ "SNR"_"RDD" (f, t) = frac(overline(alpha)_t dot ((1 - kappa(t)) + kappa(t) dot L P(f))^2, 1 - overline(alpha)_t) $

- *低频* ($L P(f) approx 1$)：$"SNR"_"RDD" = overline(alpha)_t \/ (1 - overline(alpha)_t)$（不变！被 drift 保护）
- *高频* ($L P(f) approx 0$)：$"SNR"_"RDD" = overline(alpha)_t dot (1-kappa(t))^2 \/ (1 - overline(alpha)_t)$（额外衰减 $(1-kappa(t))^2$ 倍！）

#block(fill: rgb("#e8f4f8"), inset: 12pt, radius: 4pt)[
  *结论：RDD 的 drift 等效于频率自适应的 SNR 衰减——低频 SNR 不变，高频 SNR 被额外衰减。这比 DCTdiff 的全局统一 SNR_scale 更加精细。*
]

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌─────────────────── RDD 训练流程 ───────────────────┐
│                                                      │
│  Input: x_0 (32×32 latent from VAE)                 │
│     ↓                                                │
│  Compute drift: d = UP(DOWN(x_0)) - x_0             │
│     ↓                                                │
│  Sample t ~ Uniform[0, T_stage], compute κ(t)       │
│     ↓                                                │
│  Forward: x_t = √ᾱ_t·(x_0 + κ(t)·d) + √(1-ᾱ_t)·ε │
│     ↓                                                │
│  Construct target: ε̃ = ε + √(ᾱ_t/(1-ᾱ_t))·κ(t)·d  │
│     ↓                                                │
│  Network prediction: ε̃_pred = DiT(x_t, t, class)   │
│     ↓                                                │
│  Loss: MSE(ε̃_pred, ε̃)                              │
└──────────────────────────────────────────────────────┘
```
]

#v(0.5em)

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌─────────── RDD 采样流程（多阶段 + DPM-Solver）──────────┐
│                                                            │
│  Stage 3 (t: 1000→600): 8×8 latent                       │
│    ├─ DPM-Solver 反向采样 (加 drift 修正步)               │
│    └─ 输出: pred_x0_8x8                                   │
│         ↓                                                  │
│  noise_filling_upsample:                                   │
│    x_16 = √ᾱ_{t_start} · UP(pred_x0_8x8)                │
│          + √(1-ᾱ_{t_start}) · ε_new                      │
│         ↓                                                  │
│  Stage 2 (t: 600→300): 16×16 latent                      │
│    ├─ DPM-Solver 反向采样 (加 drift 修正步)               │
│    └─ 输出: pred_x0_16x16                                 │
│         ↓                                                  │
│  noise_filling_upsample → 32×32                            │
│         ↓                                                  │
│  Stage 1 (t: 300→0): 32×32 latent                        │
│    ├─ DPM-Solver 反向采样 (加 drift 修正步)               │
│    └─ 输出: final pred_x0_32x32                           │
│         ↓                                                  │
│  VAE decode → 256×256 image                               │
└────────────────────────────────────────────────────────────┘
```
]

== 关键代码片段

=== q_sample (含 drift) — `gaussian_diffusion_with_drift.py` L245-272

```python
def q_sample(self, x_start, t, noise=None, drift=None, kappa=None):
    """Forward process: add noise + drift to x_start."""
    if noise is None:
        noise = th.randn_like(x_start)
    sqrt_alpha_bar = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_bar = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    # Standard DDPM forward + drift
    sample = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
    if drift is not None and kappa is not None:
        sample = sample + sqrt_alpha_bar * kappa * drift  # 加入drift
    return sample
```

=== 训练 target 构造 — L991-997

```python
# Construct modified noise target: ε̃ = ε + correction
correction = sqrt_alpha_bar / sqrt_one_minus_alpha_bar * kappa * drift
target = noise + correction  # This is ε̃
```

=== DPM-Solver drift 修正步 — L1310-1553

```python
# Standard DPM-Solver step
x_prev = dpm_solver_step(x_t, t, t_prev, model_output)
# Add drift correction
if drift is not None:
    kappa_prev = self._get_kappa(t_prev, stage_info)
    kappa_curr = self._get_kappa(t, stage_info)
    sqrt_alpha_prev = _extract(self.sqrt_alphas_cumprod, t_prev, x_prev.shape)
    x_prev = x_prev + sqrt_alpha_prev * (kappa_prev - kappa_curr) * drift
```

== RDD 的 4 个核心 Novelty

+ *单一 Markov forward process 中连续 drift*（非级联/非离散跳变）
+ *闭式 posterior + DPM-Solver 兼容*
+ *单一预测目标 $tilde(epsilon)$*（非双重预测如 f-DM）
+ *Drift coefficient $kappa$ 自动处理跨分辨率 SNR*

#pagebreak()

= Relay Diffusion (ICLR 2024)

== 论文信息

*"Relay Diffusion: Unifying diffusion process across resolutions for image synthesis"*, ICLR 2024.

== 方法概述

两阶段级联：Stage 1 标准 EDM 在 64×64 生成，双线性上采样到 256×256 后，Stage 2 用 "Blurring Diffusion"（DCT 域模糊扩散 + Block Noise）精炼高分辨率细节。

== 数学理论

=== 前向过程 (Stage 2, Blurring Diffusion)

$ x_t = "DCTBlur"(x_0, sigma_"blur"(t)) + sigma(t) dot (epsilon + s_"block" dot epsilon_"block") $

DCTBlur 定义（4×4 patch 的 DCT 域频率衰减）：

$ alpha(f, t) = exp(-|f|^2 dot sigma_"blur"^2 (t) \/ 2) dot (1 - delta) + delta $

$ sigma_"blur" (t) = sigma_max dot sin(t pi \/ 2)^2 $

=== Block Noise

空间 4×4 相关噪声，通过循环位移累加实现：

```python
# blurring.py L143-158
g_noise = torch.zeros_like(noise)
for px in range(block_size):
    for py in range(block_size):
        g_noise += noise.roll(shifts=(px, py), dims=(-2, -1))
g_noise /= block_size  # normalize
combined_noise = epsilon + block_scale * g_noise
```

=== 预测目标

网络预测未模糊的干净原图（非 epsilon），通过 EDM 预条件化：

$ D(x) = c_"skip" dot x + c_"out" dot F_theta (c_"in" dot x, c_"noise") $

=== 损失函数

$ cal(L) = w(sigma) dot || D(x_"blurred" + n, sigma) - x_0 ||^2 $

$ w(sigma) = frac(sigma^2 + sigma_d^2, sigma dot sigma_d^2), quad "clamp" <= 100000 $

=== 采样

在 DCT 频域执行 Euler/Heun 步骤。Stage 2 从 truncation_sigma=0.9 开始（非纯噪声）。

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌──── Relay Diffusion 训练 & 采样流程 ────┐
│                                           │
│  ┌─── Stage 1: 标准 EDM (64×64) ───┐    │
│  │ 训练: 标准去噪目标                │    │
│  │ 网络: DhariwalUNet (ADM)         │    │
│  │ 采样: 256步 EDM sampler           │    │
│  │ 输出: 64×64 clean image          │    │
│  └───────────────────────────────────┘    │
│         ↓ Bilinear Upsample 4×           │
│  ┌─── Stage 2: Blurring Diffusion ──┐   │
│  │ 输入: 256×256 (from Stage 1)      │   │
│  │ 前向: DCTBlur + Block Noise       │   │
│  │ 训练: 预测未模糊干净图            │   │
│  │ 采样: 200步 DCT域Heun sampler     │   │
│  │       从 σ=0.9 开始(非纯噪声)     │   │
│  │ 输出: 256×256 refined image       │   │
│  └───────────────────────────────────┘    │
│                                           │
│  总采样步数: 256 + 200 = 456 步          │
└───────────────────────────────────────────┘
```
]

== 实验结果与 Ablation

#table(
  columns: 4,
  stroke: 0.5pt,
  inset: 6pt,
  [*配置*], [*FID↓*], [*步数*], [*备注*],
  [Stage1 only (64×64)], [2.13], [256], [低分辨率],
  [Stage1+Stage2 (w/o Block Noise)], [3.21], [456], [无空间相关噪声],
  [Stage1+Stage2 (w/ Block Noise)], [*2.64*], [456], [完整方法],
  [不同 block_scale], [—], [—], [block_scale=0.15 最优],
  [不同 blur_sigma_max], [—], [—], [sigma_max=3.0 最优],
  [truncation_sigma=0.5], [2.89], [456], [从更低噪声开始],
  [truncation_sigma=0.9], [*2.64*], [456], [最优截断点],
)

*关键 Ablation 发现：*
- Block Noise 对 FID 贡献约 0.6 个点（3.21→2.64）
- truncation_sigma 的选择影响显著：太低会丢失 Stage 1 信息
- blur_sigma_max 过大导致过度模糊，过小则无法有效引导

== 与 RDD 的深度对比

=== 数学本质差异

$ "RDD均值" = sqrt(overline(alpha)_t) dot ((1-kappa(t)) dot x_0 + kappa(t) dot "blur"(x_0)) quad "(线性插值式模糊)" $

$ "Relay均值" = sum_f (alpha(f,t) dot hat(X)_0 (f)) dot "DCT\_basis"(f) quad "(指数衰减式模糊)" $

两者都实现"随时间逐渐丢失高频"，但：
- RDD：线性过渡 ($kappa$ 线性增长)
- Relay：指数过渡 ($exp(-|f|^2 dot t)$ 高频指数衰减)

=== 工程对比表

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*维度*], [*Relay Diffusion*], [*RDD*],
  [框架], [EDM (σ参数化, VE-SDE)], [DDPM (离散β, VP-SDE)],
  [网络], [DhariwalUNet ×2], [DiT ×1-3],
  [输入空间], [像素空间(无VAE)], [VAE latent],
  [Batch], [4096 (累积64轮)], [256],
  [采样步数], [456 (256+200)], [10-50 (DPM-Solver)],
  [DPM-Solver], [✗ (自定义DCT采样器)], [✓],
  [阶段过渡], [硬切换(64→256)], [连续drift],
  [模型数], [2个独立UNet], [1-3个DiT],
)

=== RDD 优势
- 单模型，参数量 1/2-1/3
- DPM-Solver 10-50步 vs Relay 456步 (9-45× 更快)
- 连续过渡无信息断裂
- 闭式后验

=== Relay 优势
- 频域操作物理含义明确
- Block Noise 模拟结构化误差
- 大 batch 训练更稳定

=== RDD 弥补策略

+ 在 noise_filling 中加 Block Noise 消融 [低工作量]
+ 频域分析实验：画 RDD 的 DCT 功率谱 [中工作量]
+ 增大 batch (梯度累积) [中工作量]
+ 论文对比：RDD 10步 vs Relay 456步 Pareto 曲线 [写作]

#pagebreak()

= LapFlow (ICLR 2026)

== 论文信息

*"LapFlow: Laplacian Pyramid Flow Matching"*, ICLR 2026.

== 方法概述

将图像做 Laplacian 金字塔分解，各频带在对应时间段内独立进行 Flow Matching，使用 Mixture of Transformers (MoT) 实现跨尺度注意力共享。

== 数学理论

=== Laplacian 金字塔分解

$ L_0 = x - "UP"("DOWN"(x)) quad "(高频残差, 32×32)" $
$ L_1 = "DOWN"(x) - "UP"("DOWN"^2 (x)) quad "(中频残差, 16×16)" $
$ L_2 = "DOWN"^2 (x) quad "(低频基础, 8×8)" $

=== 分段时间 Flow Matching

3 级，time_segments=[0.33, 0.67]：

- Scale 2 ($t in (0, 0.33)$)：$x_t = t dot L_2 + (1-t) dot epsilon_2$
- Scale 1 ($t in (0.33, 0.67)$)：$x_t = frac(t-0.33, 1-0.33) dot L_1 + (1-t) dot epsilon_1$
- Scale 0 ($t in (0.67, 1.0)$)：$x_t = frac(t-0.67, 1-0.67) dot L_0 + (1-t) dot epsilon_0$

=== 预测目标（速度场）

$ u_i = frac(L_i, 1 - T_i) - epsilon_i $

=== 与 RDD 的关键数学联系

#block(fill: rgb("#fff3cd"), inset: 12pt, radius: 4pt)[
  *关键发现：* $d = "UP"("DOWN"(x_0)) - x_0 = -L_0$（RDD 的 drift = 负的高频残差！）

  RDD 前向均值 = $sqrt(overline(alpha)_t) dot (x_0 - kappa(t) dot L_0)$

  即 RDD 在前向过程中逐渐 *减掉高频残差* → 使 $t arrow T$ 时趋向低分辨率版本。
]

=== 时间段对应关系

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*方法*], [*时间段划分*], [*归一化*],
  [LapFlow], [[0, 0.33, 0.67, 1.0]], [[0, 0.33, 0.67, 1.0]],
  [RDD], [[0, 300, 600, 1000]], [[0, 0.3, 0.6, 1.0]],
)

*几乎完全相同！*

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌──────────── LapFlow 训练流程 ────────────┐
│                                            │
│  Input: x_0 (32×32 latent)                │
│     ↓                                      │
│  Laplacian Decomposition:                  │
│     L₂ = DOWN²(x)         [8×8]           │
│     L₁ = DOWN(x)-UP(L₂)   [16×16]        │
│     L₀ = x - UP(DOWN(x))  [32×32]        │
│     ↓                                      │
│  Random select scale i (per iter)          │
│     ↓                                      │
│  Sample t ~ Uniform[T_i, 1]               │
│     ↓                                      │
│  Construct x_t = α_i(t)·L_i + σ(t)·ε_i   │
│     ↓                                      │
│  ┌─── MultiScaleDiT (MoT) ───────────┐   │
│  │ Per-scale: QKV proj, MLP, AdaLN    │   │
│  │ Shared: Attention (cross-scale)     │   │
│  │ Causal mask: scale i sees ≤ i only │   │
│  └────────────────────────────────────┘   │
│     ↓                                      │
│  Loss: MSE(v_pred[i], u[i]) × w_i        │
│     + gradient clip (norm=1.0)            │
│     + cosine LR schedule                  │
└────────────────────────────────────────────┘
```
]

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌──────────── LapFlow 采样流程 ────────────┐
│                                            │
│  t=0: All scales start from noise          │
│     ↓                                      │
│  ODE Solver (dopri5 / euler):              │
│     ↓                                      │
│  t=0→0.33: Scale 2 (8×8) 生成 L₂         │
│  t=0.33→0.67: Scale 1 (16×16) 生成 L₁    │
│    (可看到 L₂ via causal mask)             │
│  t=0.67→1.0: Scale 0 (32×32) 生成 L₀     │
│    (可看到 L₁, L₂ via causal mask)        │
│     ↓                                      │
│  Reconstruct: x = L₀ + UP(L₁) + UP²(L₂) │
│     ↓                                      │
│  VAE decode → 256×256 image               │
└────────────────────────────────────────────┘
```
]

== 代码关键片段

=== Causal Mask (MultiScaleDiT.py L156-164)

```python
# Causal mask: scale i can only attend to scale <= i
def get_causal_mask(scale_ids, num_scales):
    mask = torch.zeros(len(scale_ids), len(scale_ids))
    for i in range(len(scale_ids)):
        for j in range(len(scale_ids)):
            if scale_ids[j] > scale_ids[i]:
                mask[i, j] = -float('inf')  # block future scales
    return mask
```

=== 时间段映射 (feb12_train_ours.py L392-411)

```python
# Scale selector → time range mapping
time_segments = [0.33, 0.67]  # boundaries
scale_id = random.randint(0, num_levels-1)
t_min = time_segments[scale_id] if scale_id > 0 else 0
t_max = 1.0
t = torch.rand(batch_size) * (t_max - t_min) + t_min
```

== 实验结果与 Ablation

#table(
  columns: 4,
  stroke: 0.5pt,
  inset: 6pt,
  [*配置*], [*FID↓*], [*训练*], [*备注*],
  [DiT-B (单尺度 baseline)], [~20], [~800ep], [无多尺度],
  [DiT-B + LapFlow (2级)], [~15], [~800ep], [2级金字塔],
  [DiT-B + LapFlow (3级)], [~14], [~1000ep], [3级金字塔],
  [DiT-L + LapFlow (3级)], [~8], [~1000ep], [更大模型],
  [w/o causal mask], [~18], [~800ep], [信息泄露],
  [w/o MoT (全共享)], [~16], [~800ep], [参数不足],
  [w/o per-scale loss weight], [~16], [~800ep], [各scale权重相同],
)

*关键 Ablation 发现：*
- Causal mask 贡献约 3 FID 点（防止信息泄露）
- MoT 比全共享好 ~1 FID（专属投影的重要性）
- Per-scale loss weight 贡献 ~1 FID

== 与 RDD 对比

=== RDD 优势
- 闭式后验 + DPM-Solver（更高效采样）
- 全程固定维度（无需处理不同大小 tensor）
- 实现更简洁

=== LapFlow 优势
- MoT 参数效率高
- Causal mask 保证因果正确性
- 部分并行采样
- FM 训练通常更稳定

=== RDD 弥补策略

+ 实验 FM 版本 RDD 作为 Appendix [高]
+ 加入 per-stage loss weighting [中]
+ 论文明确 LapFlow time_segments ≡ RDD stage_lengths [写作]

#pagebreak()

= DiMR (NeurIPS 2024) — 当前 SOTA

== 论文信息

*"DiMR: Denoising in Multi-Resolution"*, NeurIPS 2024. *当前 ImageNet-256 绝对 SOTA (FID=1.70)*。

== 方法概述

纯架构多分辨率设计——*完全不修改扩散前向过程*。单次标准扩散 + 多分辨率去噪网络（Transformer 处理低分辨率 + ConvNeXt 处理高分辨率）。

== 数学理论

=== 前向过程（完全标准，不做任何修改！）

$ x_t = sqrt(overline(alpha)_t) dot x_0 + sqrt(1-overline(alpha)_t) dot epsilon $

=== Beta Schedule

$ beta_t = (sqrt(beta_0) + frac(sqrt(beta_1) - sqrt(beta_0), T) dot t)^2, quad beta_0=0.00085, beta_1=0.012 $

=== 多尺度目标构造

$ "target"_(16 times 16) = frac("AvgPool"_(2 times 2)(epsilon), 2) quad "(scale correction)" $
$ "target"_(32 times 32) = epsilon $

=== 损失函数

$ cal(L) = 0.25 dot "MSE"("pred"_16, "target"_16) + 1.0 dot "MSE"("pred"_32, "target"_32) $

=== TDP (Time-Dependent Parameter)

$ "result" = sigma(a) dot "param"_0 + (1-sigma(a)) dot "param"_1 $
$ a = (log_"snr" \/ 4) dot w + b $

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌────────────── DiMR 网络架构 ──────────────┐
│                                              │
│  Input: x_t (4, 32, 32)                     │
│     ↓                                        │
│  ┌──── Stage 0: Transformer (16×16) ────┐   │
│  │ Stride=2 Conv 降到 16×16              │   │
│  │ 39 Transformer blocks (dim=960)       │   │
│  │ TDP 时间条件注入                      │   │
│  │ 输出: eps_pred_16x16 + features       │   │
│  └───────────────────────────────────────┘   │
│         ↓ PixelShuffle 2× (特征上采样)       │
│         + Skip Connection                    │
│  ┌──── Stage 1: ConvNeXt (32×32) ───────┐   │
│  │ 20 ConvNeXt blocks (dim=480)          │   │
│  │ TDP 时间条件注入                      │   │
│  │ 输出: eps_pred_32x32                  │   │
│  └───────────────────────────────────────┘   │
│     ↓                                        │
│  Loss = 0.25·MSE(pred_16, pool(ε)/2)        │
│       + 1.0·MSE(pred_32, ε)                 │
│                                              │
│  采样时只使用 Stage 1 (32×32) 的输出         │
└──────────────────────────────────────────────┘
```
]

== 代码关键片段

=== 多尺度目标 (sde.py L313-327)

```python
def even_multi_scale_targets(eps, levels=2, scale_correction=True):
    targets = [eps]  # 32x32 original
    for l in range(1, levels):
        kernel_size = 2 ** l
        # Average pooling via conv2d
        pooled = F.conv2d(
            eps, 
            torch.ones(C, 1, kernel_size, kernel_size) / (kernel_size**2),
            stride=kernel_size, groups=C
        )
        if scale_correction:
            pooled = pooled / kernel_size  # scale correction factor
        targets.append(pooled)
    return targets
```

=== TDP 时间条件 (sigmoid/module.py L28-60)

```python
class TDP(nn.Module):
    def __init__(self, dim):
        self.param0 = nn.Parameter(torch.randn(dim))
        self.param1 = nn.Parameter(torch.randn(dim))
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, log_snr):
        a = (log_snr / 4) * self.weight + self.bias
        mix = torch.sigmoid(a)
        return x * (mix * self.param0 + (1-mix) * self.param1)
```

== 实验结果与 Ablation

#table(
  columns: 4,
  stroke: 0.5pt,
  inset: 6pt,
  [*配置*], [*FID↓*], [*参数量*], [*备注*],
  [DiMR-XL/2R], [*1.70*], [505M], [当前 SOTA],
  [DiMR-G/2R], [*1.63*], [1.06B], [更大模型],
  [DiMR-XL/1R (单分辨率)], [2.35], [505M], [无多分辨率],
  [DiMR-XL/2R w/o TDP], [1.95], [505M], [标准时间嵌入],
  [DiMR-XL/2R w/o scale_correction], [1.89], [505M], [无缩放修正],
  [DiMR-XL/2R w/o PixelShuffle], [2.12], [505M], [无特征传递],
  [DiMR-XL/2R loss_weight=[1,1]], [1.82], [505M], [等权重],
)

*关键 Ablation 发现：*
- 多分辨率贡献巨大：1R→2R 提升 0.65 FID
- TDP 贡献 0.25 FID
- PixelShuffle skip connection 贡献 0.42 FID
- Loss weight [0.25, 1.0] 比 [1, 1] 好 0.12 FID

== 与 RDD 对比

=== RDD 优势
- Drift 作为归纳偏置可能加速收敛
- 在同等步数下 FID 可能更低（待验证）
- 数学上更优雅（修改 SDE 而非硬编码架构）

=== DiMR 优势
- FID=1.70 绝对最强
- Transformer+ConvNeXt 混合效率高
- 训练充分（1M步×batch1024）
- TDP 精细时间条件

=== RDD 弥补策略

+ 训练到 700K 步对比 RDD-XL vs DiMR-XL [高]
+ DiMR 架构 + RDD drift 融合实验 [中]
+ 学习 DiMR 超参 (AdamW β2=0.99, wd=0.03, batch1024) [中]
+ 论文定位：RDD 改过程, DiMR 改架构, 二者正交可结合 [写作]

#pagebreak()

= DCTdiff (ICML 2025)

== 论文信息

*"Diffusion on the Discrete Cosine Transform Space"*, ICML 2025.

== 方法概述

将图像转到块状 DCT 系数空间后做扩散。通过全局 SNR_scale 修正和频率加权损失处理频带差异。提出"扩散 = 频谱自回归"理论。

== 数学理论

=== 数据预处理

$ "RGB" arrow "YCbCr" arrow "4:2:0" arrow B times B "分块" arrow "2D DCT" arrow "Zigzag" arrow "截断" $

最终 token shape: $(H\/B times W\/B, 6 times "low\_freqs")$

=== 修改版 VP-SDE

$ beta_"eff" (t) = beta(t) + frac(d, d t)(log(1 + (S-1) dot exp(-integral_0^t beta(s) d s))) $

展开：

$ beta_"eff" (t) = beta(t) dot frac(1, 1 + (S-1) dot exp(-A(t))) $

*效果：终端 SNR 被放大 $S$ 倍。*

=== 关键限制

所有频率使用*相同*的 $beta_"eff"(t)$ schedule——全局 SNR_scale $S$ 是统一的，*不区分频率！*

=== 频率加权损失

$ cal(L) = "MSE"(epsilon_"pred", epsilon) dot Y_"reweight" $

$Y_"reweight"$ = 各频率系数的统计标准差。

=== 与 RDD 的频域 SNR 对比

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*频率*], [*DCTdiff SNR*], [*RDD SNR*],
  [低频 ($L P(f) approx 1$)], [$S dot overline(alpha)_t \/ (1-overline(alpha)_t)$], [$overline(alpha)_t \/ (1-overline(alpha)_t)$ (不变)],
  [高频 ($L P(f) approx 0$)], [$S dot overline(alpha)_t \/ (1-overline(alpha)_t)$], [$overline(alpha)_t dot (1-kappa)^2 \/ (1-overline(alpha)_t)$ (衰减)],
  [特点], [全局统一放大], [*频率自适应*],
)

#block(fill: rgb("#d4edda"), inset: 12pt, radius: 4pt)[
  *结论：RDD 比 DCTdiff 更精细！*
  - DCTdiff：所有频率 SNR 统一放大 $S$ 倍
  - RDD：低频 SNR 保护不变，高频 SNR 额外衰减 $(1-kappa)^2$ 倍
  - RDD 实现了真正的 *频率自适应* SNR 调整
]

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌────────────── DCTdiff 数据流水线 ──────────────┐
│                                                  │
│  RGB Image (H×W×3)                              │
│     ↓ Color transform                           │
│  YCbCr (Y: H×W, Cb: H/2×W/2, Cr: H/2×W/2)    │
│     ↓ Block partition (B=2)                     │
│  Blocks: (H/B×W/B) × (B×B) per channel         │
│     ↓ 2D DCT per block                          │
│  DCT coefficients: low_freq components          │
│     ↓ Zigzag ordering + truncation              │
│  Token: (H/B×W/B, 6×low_freqs) = (256, 24)    │
│     ↓ Normalize to [-1,1]                       │
│     ↓                                           │
│  ┌──── Modified VP-SDE Diffusion ─────────┐    │
│  │ β_eff(t) with SNR_scale S              │    │
│  │ x_t = √ᾱ_eff·x_0 + √(1-ᾱ_eff)·ε     │    │
│  │ Network: UViT (d=768, L=16, H=12)      │    │
│  │ Loss: freq-weighted MSE                 │    │
│  │ Sampler: DPM-Solver (modified beta)     │    │
│  └─────────────────────────────────────────┘    │
│     ↓ Inverse process                           │
│  IDCT → de-block → YCbCr→RGB                   │
└──────────────────────────────────────────────────┘
```
]

== 实验结果

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*配置*], [*FID↓*], [*备注*],
  [ImageNet 64×64 (标准DDPM)], [~5.0], [baseline],
  [ImageNet 64×64 (DCTdiff, S=1)], [~4.2], [无SNR修正],
  [ImageNet 64×64 (DCTdiff, S=optimal)], [~3.5], [最优S],
  [ImageNet 256], [*未报告*], [未扩展到高分辨率],
)

== RDD 弥补策略

+ 推导并展示频域 SNR 公式 [高/理论]
+ 频域可视化实验 [中]
+ 论文引用"扩散=频谱自回归"理论 [写作]

#pagebreak()

= SPEED (Stanford 2025) — Training-free

== 论文信息

*"SPEED: Spectrally Progressive and Efficient Estimation of Diffusion"*, Stanford 2025.

== 方法概述

Training-free 推理加速：基于自然图像功率谱衰减理论，给出数学最优的分辨率转换时刻；在转换时用频谱噪声扩展 + kappa 时间步对齐实现无损分辨率提升。

== 数学理论

=== 三个核心组件

*(a) 频谱噪声扩展 (Spectral Noise Expansion)：*

$ x_"expanded" = "IDCT"("DCT"(x_"src") plus.circle t dot epsilon_"high") $

低频保持源信号，高频填入幅度为 $t$ 的噪声。

*(b) 最优转换时刻：*

基于功率谱模型 $P(omega) = A dot |omega|^(-beta)$：

$ t^* (omega) = frac(1, 1 + sqrt(frac(delta, P(omega) dot (1 + P(omega) - delta)))) $

*(c) 时间步对齐 (kappa)：*

$ kappa(t, r) = frac(r, 1 + (r-1) dot t) $
$ tilde(t) = t dot kappa(t, r), quad tilde(x) = kappa(t, r) dot x_"expanded" $

=== 与 RDD 的直接关联

#block(fill: rgb("#fff3cd"), inset: 12pt, radius: 4pt)[
  *SPEED 的理论可以直接计算 RDD 的最优 stage_lengths！*

  对于 ImageNet latent (32×32, f=8)，设 Nyquist 频率 $omega_1=0.5, omega_2=0.25$：
  - 用 $A=203.6, beta=1.915$ (FLUX VAE 参数), $delta=0.01$
  - 计算 $t^*(omega_1)$ 和 $t^*(omega_2)$ = 最优阶段边界
]

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌──────── SPEED 推理加速流程 ────────┐
│                                      │
│  Pre-compute:                        │
│    A, β = fit power spectrum          │
│    t*(ω_i) = optimal transition times│
│                                      │
│  Stage 1: Low-res generation         │
│    ├─ Resolution: 0.5× (16×16 lat)  │
│    ├─ Steps: t=1.0 → t*(ω₁)        │
│    └─ Standard FM sampling           │
│         ↓                            │
│  Spectral Expand:                    │
│    ├─ DCT(x_src)                     │
│    ├─ Append t·ε at high-freq        │
│    ├─ IDCT → expanded latent         │
│    └─ Kappa align: x̃ = κ·x_exp     │
│         ↓                            │
│  Stage 2: High-res generation        │
│    ├─ Resolution: 1.0× (32×32 lat)  │
│    ├─ Steps: t̃ → 0                  │
│    └─ Standard FM sampling           │
│         ↓                            │
│  Output: final image                 │
│                                      │
│  Speedup: 2-3× (no quality loss)    │
└──────────────────────────────────────┘
```
]

== 代码关键片段

=== Activation Time (utils.py L33-47)

```python
def activation_time(omega, A, beta, delta=0.01):
    """Optimal transition time for frequency omega."""
    P = A * abs(omega) ** (-beta)
    t_star = 1.0 / (1.0 + math.sqrt(delta / (P * (1 + P - delta))))
    return t_star
```

=== Kappa Alignment (utils.py L80-103)

```python
def kappa(t, r):
    """Timestep alignment factor."""
    return r / (1 + (r - 1) * t)

def align_timestep(t, x_expanded, r):
    k = kappa(t, r)
    t_aligned = t * k
    x_aligned = k * x_expanded
    return t_aligned, x_aligned
```

== RDD 弥补策略

+ 用 SPEED 公式计算最优 stage_lengths [高/低工作量]
+ 改进 noise_filling 为频谱扩展 [中]
+ 引用 SPEED 理论作为 RDD 设计动机 [写作]

#pagebreak()

= Pyramid-Flow (ICLR 2025) — 视频

== 论文信息

*"Pyramid Flow Matching for Efficient Video Generative Modeling"*, ICLR 2025.

== 方法概述

将 Flow Matching 路径拆分为 3 个空间金字塔阶段，各阶段在不同分辨率工作，阶段间用 renoising（加块噪声）连接。面向视频生成。

== 数学理论

=== 多阶段 Flow Matching

3 阶段，stage_range = [0, 1/3, 2/3, 1]：

每阶段内线性插值：$x_t = "ratio" dot "start" + (1-"ratio") dot "end"$

目标：$u = "start" - "end"$ (rectified flow)

=== Renoising 精确公式

$ alpha = frac(1, sqrt(1 + 1\/gamma) dot (1-sigma_"ori") + sigma_"ori") $

$ beta = frac(alpha dot (1-sigma_"ori"), sqrt(gamma)) $

$ "latents"_"new" = alpha dot "UP"("latents"_"old") + beta dot "block\_noise" $

$gamma$ 控制 renoising 强度：
- $gamma arrow infinity$：$alpha arrow 1, beta arrow 0$（不加噪声）
- $gamma arrow 0$：$alpha arrow 0$（噪声覆盖）

=== 与 RDD 的数学联系

RDD noise_filling：$x_"next" = sqrt(overline(alpha)_(t_"start")) dot "UP"(hat(x)_0) + sqrt(1-overline(alpha)_(t_"start")) dot epsilon$

即 $alpha_"RDD" = sqrt(overline(alpha)_(t_"start"))$, $beta_"RDD" = sqrt(1-overline(alpha)_(t_"start"))$。

Pyramid-Flow 用额外参数 $gamma$ 控制；RDD 用 stage_lengths（决定 $t_"start"$）隐式控制。

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌───── Pyramid-Flow 视频生成流程 ─────┐
│                                        │
│  Stage 0 (t∈[0, 1/3]): 低分辨率       │
│    ├─ Resolution: 1/4× spatial         │
│    ├─ Temporal: full frames             │
│    ├─ FM: noise → low-res clean        │
│    └─ Output: coarse video             │
│         ↓ Spatial upsample 2×          │
│         ↓ Renoising: α·UP(x)+β·block  │
│                                        │
│  Stage 1 (t∈[1/3, 2/3]): 中分辨率     │
│    ├─ Resolution: 1/2× spatial         │
│    ├─ FM: renoised → mid-res clean     │
│    └─ Output: medium video             │
│         ↓ Spatial upsample 2×          │
│         ↓ Renoising: α·UP(x)+β·block  │
│                                        │
│  Stage 2 (t∈[2/3, 1]): 全分辨率       │
│    ├─ Resolution: 1× spatial           │
│    ├─ FM: renoised → full-res clean    │
│    └─ Output: final video              │
└────────────────────────────────────────┘
```
]

== RDD 弥补策略

+ 在 noise_filling 中加可调 $gamma$ 参数消融 [低]
+ 论文论述：RDD 无需 renoising，drift 连续保持低分辨率信息 [写作]

#pagebreak()

= SwD — Scale-wise Distillation (Yandex 2025)

== 论文信息

*"Scale-wise Distillation of Diffusion Models"*, Yandex 2025.

== 方法概述

蒸馏框架：将 28 步采样划分为 4 个 boundary 区间，每区间对应不同分辨率。Teacher=预训练模型，Student=LoRA。

== 数学理论

=== Boundary + Scale 配置

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*步骤范围*], [*Latent 尺寸*], [*像素分辨率*],
  [0-6], [64×64], [512×512],
  [7-13], [80×80], [640×640],
  [14-17], [96×96], [768×768],
  [18-27], [128×128], [1024×1024],
)

=== 蒸馏损失

$ cal(L)_"total" = lambda_"DMD" dot cal(L)_"DMD" + lambda_"GAN" dot cal(L)_"GAN" + lambda_"MMD" dot cal(L)_"MMD" $

- *DMD*：Distribution Matching Distillation（分数差异梯度）
- *GAN*：Transformer 判别头
- *MMD*：最大均值差异（中间特征）

=== 与 RDD 的结构同构性

#block(fill: rgb("#fff3cd"), inset: 12pt, radius: 4pt)[
  SwD `boundaries = [0, 7, 14, 18, 28]` ≡ RDD `stage_lengths = [300, 300, 400]`

  *完全同构！*都定义"哪些时间步对应哪个分辨率"。
]

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌────── SwD 蒸馏流程 ──────┐
│                            │
│  Teacher: SD3.5-Medium     │
│  Student: Teacher + LoRA   │
│                            │
│  Step 0-6 @ 64×64 latent  │
│     ↓ upsample + renoise  │
│  Step 7-13 @ 80×80        │
│     ↓ upsample + renoise  │
│  Step 14-17 @ 96×96       │
│     ↓ upsample + renoise  │
│  Step 18-27 @ 128×128     │
│     ↓                      │
│  Loss: DMD + GAN + MMD     │
│                            │
│  训练: 仅 3-4K 步 LoRA    │
│  推理: 4 步生成 1024×1024 │
└────────────────────────────┘
```
]

== 实验结果

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*配置*], [*步数*], [*质量*],
  [SD3.5 原始], [28步], [baseline],
  [SwD 蒸馏], [*4步*], [接近28步质量],
  [w/o GAN loss], [4步], [明显模糊],
  [w/o MMD loss], [4步], [细节损失],
  [w/o multi-scale], [4步], [退化严重],
)

== RDD 弥补策略

+ RDD 收敛后用 SwD 蒸馏到 4 步 [Future]
+ 论文指出结构同构性 [写作]

#pagebreak()

= LowDiff (2025)

== 论文信息

*"LowDiff: Low-Resolution Diffusion for Efficient Image Generation"*, 2025.

== 方法概述

单一统一模型在多个分辨率独立去噪，低分辨率结果通过 channel concat 条件高分辨率。实现 50-83% 吞吐量提升。

== 数学理论

=== 多分辨率训练 (Flow Matching)

低分辨率 (8×8)，无条件：
$ x_t^L = (1-t) dot epsilon_L + t dot x_L $

高分辨率 (16×16)，条件于 $hat(x)_L$：
$ x_t^H = (1-t) dot epsilon_H + t dot x_H $

网络输入：$"input" = (x_t^H; "UP"(hat(x)_L))$（channel concat）

=== 与 RDD 的等效性

- RDD drift：将 $x_L$ 信息连续注入到 SDE 均值
- LowDiff concat：将 $x_L$ 信息通过网络输入通道注入
- 当 RDD $kappa(t)=1$ 时，均值 = $sqrt(overline(alpha)_t) dot "blur"(x_0)$，等效于"已生成好低分辨率"

== 工程实现流程图

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌────── LowDiff 层分配策略 ──────┐
│                                  │
│  MSLightningDiT (28 层):         │
│                                  │
│  8×8 阶段:                       │
│    └─ 使用中间 14 层             │
│       (blocks 8-21)              │
│                                  │
│  16×16 阶段:                     │
│    └─ 使用全部 28 层             │
│       (blocks_16_L + blocks_8    │
│        + blocks_16_R)            │
│                                  │
│  + ResEmbedder: 分辨率嵌入       │
│    res_embed = MLP(one_hot(id))  │
│    condition = t_emb + cls_emb   │
│              + res_emb           │
│                                  │
│  采样: 8×8 在 t=0.3 截断        │
│       → 上采样 → 条件 16×16     │
└──────────────────────────────────┘
```
]

== 代码关键片段

=== 层分配 (lightningdit.py L457-699)

```python
class MSLightningDiT(nn.Module):
    def __init__(self, depth=28, ...):
        # Low-res uses middle layers only
        self.blocks_8 = nn.ModuleList([...])  # 14 blocks
        # High-res uses all layers
        self.blocks_16_l = nn.ModuleList([...])  # 7 blocks (front)
        self.blocks_16_r = nn.ModuleList([...])  # 7 blocks (back)
        # Resolution embedding
        self.res_embedder = ResEmbedder(num_res=2, hidden_dim=hidden)
    
    def forward(self, x, t, cls, res_id):
        cond = self.t_embed(t) + self.cls_embed(cls) + self.res_embedder(res_id)
        if res_id == 0:  # 8×8
            for blk in self.blocks_8:
                x = blk(x, cond)
        else:  # 16×16
            for blk in self.blocks_16_l:
                x = blk(x, cond)
            for blk in self.blocks_8:
                x = blk(x, cond)
            for blk in self.blocks_16_r:
                x = blk(x, cond)
        return x
```

== 实验结果

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*配置*], [*吞吐量提升*], [*FID 变化*],
  [标准全分辨率], [baseline], [baseline],
  [LowDiff (trunc=0.3)], [+50%], [+0.2 FID],
  [LowDiff (trunc=0.5)], [+83%], [+0.8 FID],
  [w/o ResEmbedder], [+50%], [+0.5 FID],
  [w/o 层分配 (all layers for 8×8)], [+20%], [+0.1 FID],
)

== RDD 弥补策略

+ 加 ResEmbedder 到 RDD unified 模型 [中]
+ 考虑层分配策略 [低]
+ 论文对比隐式(drift) vs 显式(concat)条件 [写作]

#pagebreak()

= MrFlow (2025) — Training-free

== 论文信息

*"MrFlow"*, 2025. Training-free 4 阶段 pipeline。

== 方法概述

$ "Low-res gen (12步)" arrow.r "ESRGAN ×2" arrow.r "Re-encode + noise" arrow.r "Refine (1步)" $

== 工程实现

核心代码仅 28 行 (`mrflow_utils.py`)：

```python
def direct_sigma_schedule(start_sigma=0.12, num_steps=1):
    """Refinement from specific sigma, not pure noise."""
    return torch.linspace(start_sigma, 0, num_steps)
```

== 主要结果

- ~2× 加速 (1024×1024)
- 质量接近原始全步数

== RDD 关联

- 借鉴"从特定 σ 开始精炼"用于最后阶段 [低]

#pagebreak()

= XCube (CVPR 2024) — 3D

== 论文信息

*"XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies"*, CVPR 2024.

== 方法概述

两阶段 3D latent diffusion：Dense $16^3$ → Sparse $128^3$。

== 数学理论

连续 DDPM，*v-prediction*：

$ v = alpha'(t) dot x_0 + sigma'(t) dot epsilon $

$T=1000$，linear beta schedule (0.0001→0.02)。推理 DDIM 100 步。

== 工程实现

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌──── XCube 两阶段 3D 生成 ────┐
│                                │
│  Stage 1: Dense UNet           │
│    Input: noise (16³)          │
│    Output: dense latent (16³)  │
│    DDIM 100 steps              │
│         ↓                      │
│  Stage 2: Sparse UNet          │
│    Input: upsample(Stage1)     │
│    Output: sparse voxels(128³) │
│    DDIM 100 steps              │
└────────────────────────────────┘
```
]

== RDD 关联

- v-prediction 可能比 ε-prediction 更稳定 [低优先级消融]

#pagebreak()

= OctFusion (SGP 2025) — 3D

== 论文信息

*"OctFusion"*, Siggraph / SGP 2025. 八叉树单模型双分支。

== 方法概述

单一模型内 LR + HR 两个 UNet 分支，通过八叉树连接。

== 数学理论

连续高斯扩散，log-SNR 参数化：
- LR: x0-prediction
- HR: epsilon-prediction

训练流程：先训 LR → 冻结 LR → 训 HR（HR 融合 LR 特征）

== 工程实现

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌──── OctFusion 双分支架构 ────┐
│                                │
│  ┌─── LR Branch ────┐         │
│  │ Graph UNet (octree)│         │
│  │ x0-prediction     │         │
│  │ sign clip (t<0.7) │         │
│  └────────┬───────────┘         │
│           │ feature fusion      │
│  ┌────────↓───────────┐         │
│  │ HR Branch           │         │
│  │ Graph UNet (octree) │         │
│  │ ε-prediction        │         │
│  │ uses LR features    │         │
│  └─────────────────────┘         │
└────────────────────────────────┘
```
]

== RDD 关联

- Self-conditioning 思想（+5-15% FID 潜力）[低优先级]

#pagebreak()

= PDD — Pyramid Discrete Diffusion (ECCV 2024) — 3D

== 论文信息

*"Pyramid Discrete Diffusion"*, ECCV 2024.

== 方法概述

离散扩散 (D3PM) + 空间金字塔 3 阶段：$32^3 arrow 64^3 arrow 256^3$。

== 数学理论

离散扩散 (D3PM / Multinomial)：

$ q(x_t | x_(t-1)) = alpha_t dot x_(t-1) + (1 - alpha_t) dot frac(1, K) $

每步以概率 $1-alpha_t$ 替换为均匀分布。*仅 100 步扩散*，cosine beta schedule。

== 工程实现

#block(fill: luma(245), inset: 12pt, radius: 4pt, width: 100%)[
```
┌── PDD 3阶段级联 ──┐
│                     │
│  Model 1: 32³      │
│    无条件 D3PM      │
│    100 steps        │
│       ↓             │
│  Model 2: 64³      │
│    条件于 UP(32³)   │
│    100 steps        │
│       ↓             │
│  Model 3: 256³     │
│    条件于 UP(64³)   │
│    100 steps        │
│       ↓             │
│  Output: 256³      │
└─────────────────────┘
```
]

== RDD 关联

- 100 步即可生成 → 如果 drift 先验足够强，RDD 可尝试减少到 500 步 [低优先级]

#pagebreak()

= 综合对比大表

== 方法论对比

#table(
  columns: 7,
  stroke: 0.5pt,
  inset: 5pt,
  [*方法*], [*框架*], [*前向修改*], [*多尺度来源*], [*采样器*], [*模型数*], [*DPM-Solver*],
  [*RDD (ours)*], [DDPM/SDE], [✓ drift], [SDE drift], [DPM-Solver], [1], [✓],
  [Relay], [EDM/VE], [✓ blur], [级联], [自定义DCT], [2], [✗],
  [LapFlow], [FM/ODE], [✓ 分段], [Laplacian], [ODE solver], [1(MoT)], [✗],
  [DiMR], [VP-SDE], [✗], [多尺度架构], [DPM-Solver], [1], [✓],
  [DCTdiff], [VP-SDE], [✓ SNR], [DCT空间], [DPM-Solver], [1], [✓],
  [SPEED], [FM], [✗], [推理频谱扩展], [原始solver], [0], [—],
  [Pyramid-Flow], [FM], [✓ 多阶段], [金字塔+renoise], [Euler], [1], [✗],
  [SwD], [FM], [✗], [蒸馏策略], [Euler 4步], [1+LoRA], [✗],
  [LowDiff], [FM/EDM], [✗], [条件级联], [Euler], [1], [✗],
  [MrFlow], [FM], [✗], [推理+ESRGAN], [原始solver], [0], [—],
  [XCube], [DDPM], [✗], [2阶段级联], [DDIM], [2], [✗],
  [OctFusion], [Gaussian], [✗], [LR→HR融合], [DDIM变体], [1(2分支)], [✗],
  [PDD], [D3PM], [✗], [3阶段级联], [离散采样], [3], [✗],
)

== RDD 的独特定位

#block(fill: rgb("#e8f4f8"), inset: 12pt, radius: 4pt)[
+ *唯一*在单一 Markov 链中通过连续 drift 实现多分辨率编码
+ *唯一*同时满足：修改前向过程 + 保持闭式后验 + 兼容 DPM-Solver
+ 不需要 renoising（Pyramid-Flow、SwD 需要）
+ 不需要多模型级联（Relay、XCube、PDD 需要）
+ 不需要显式频率分解（LapFlow、DCTdiff 需要）
+ 单一预测目标 $tilde(epsilon)$
]

== ImageNet 实验配置对比

#table(
  columns: 8,
  stroke: 0.5pt,
  inset: 4pt,
  [*方法*], [*分辨率*], [*Batch*], [*GPU*], [*总步数*], [*LR*], [*参数量*], [*FID*],
  [DiMR-XL/2R], [256], [1024], [16], [1M], [2e-4], [505M], [*1.70*],
  [DiMR-G/2R], [256], [1024], [16], [1M], [2e-4], [1.06B], [*1.63*],
  [Relay-S2], [256], [4096], [8], [~305K], [1e-4], [~400M], [2.64],
  [LapFlow-L], [256], [256×16], [16], [~1000ep], [1e-4], [~460M], [~8],
  [DCTdiff], [64], [1024], [4], [500K], [3e-4], [~120M], [—],
  [LowDiff-XL], [256], [512], [—], [1M], [2e-4], [~675M], [—],
  [DiT-XL/2], [256], [256], [—], [7M], [1e-4], [675M], [2.27],
  [*RDD-XL/2*], [256], [256], [8], [700K(目标)], [1e-4], [~675M], [TBD],
)

#pagebreak()

= RDD 行动计划

== P0：必做（低工作量，高回报）

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*No.*], [*行动项*], [*依据*],
  [1], [频域 SNR 公式推导与可视化], [DCTdiff §5：证明 drift = 频率自适应 SNR],
  [2], [$gamma=0$ 消融实验], [核心消融：证明 drift 必要性],
  [3], [DPM-Solver 10/20/50 步 FID 表], [核心差异化卖点 vs 所有竞品],
)

== P1：应做（中工作量）

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*No.*], [*行动项*], [*来源*],
  [4], [用 SPEED 理论确定最优 stage_lengths], [SPEED §6],
  [5], [改进 noise_filling 为频谱扩展（仅高频加噪）], [SPEED + Relay],
  [6], [加入 per-stage loss weighting], [LapFlow §3],
  [7], [加入分辨率嵌入 ResEmbedder], [LowDiff §9],
)

== P2：加分项（高工作量）

#table(
  columns: 3,
  stroke: 0.5pt,
  inset: 6pt,
  [*No.*], [*行动项*], [*来源*],
  [8], [DiMR 架构 + RDD drift 融合], [DiMR §4],
  [9], [Flow Matching 版本 RDD (Appendix)], [LapFlow §3],
  [10], [增大 batch (梯度累积到 1024)], [Relay/DiMR],
  [11], [Self-conditioning], [OctFusion §12],
  [12], [Block Noise 实验], [Relay §2 / Pyramid-Flow §7],
)

#v(2em)

#align(center)[
  #block(fill: rgb("#1a1a2e"), inset: 16pt, radius: 8pt)[
    #text(fill: white, size: 12pt, weight: "bold")[
      核心论点：RDD 是目前唯一在单一 Markov 链中通过连续 drift 编码多分辨率、#linebreak()
      保持闭式后验、且兼容 DPM-Solver 的简洁统一方案。
    ]
  ]
]
