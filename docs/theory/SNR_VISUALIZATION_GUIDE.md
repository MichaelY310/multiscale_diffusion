# 频域 SNR 可视化工具

## 概述

`visualize_snr.py` 用于可视化 RDD 的核心理论优势：频率自适应的 SNR 调度。

**核心公式：**

- 标准 DDPM：`SNR(f,t) = ᾱ_t / (1 - ᾱ_t)`（与频率无关）
- RDD：`SNR_RDD(f,t) = ᾱ_t · ((1-κ(t)) + κ(t)·LP(f))² / (1 - ᾱ_t)`
  - 低频 LP(f)≈1 → SNR 不变（被 drift 保护）
  - 高频 LP(f)≈0 → SNR 额外衰减 (1-κ(t))² 倍
- DCTdiff：`SNR_DCT(f,t) = S · ᾱ_t / (1 - ᾱ_t)`（全频段统一放大 S 倍，不区分频率）

**结论：RDD 的 drift 等效于频率自适应 SNR——低频保护，高频加速破坏。这比 DCTdiff 的全局均匀 SNR 更精细。**

## 依赖

```bash
pip install numpy matplotlib
```

## 基本用法

```bash
# 使用默认参数（stage_lengths=300,300,400），输出到当前目录
python visualize_snr.py

# 指定输出目录
python visualize_snr.py --output-dir ./snr_figures

# 使用自定义 stage 配置
python visualize_snr.py --stage-lengths 200,400,400 --output-dir ./snr_200_400_400

# 计算 SPEED 理论最优 stage lengths
python visualize_snr.py --compute-speed-optimal
```

## 生成的图表

### Figure 1: `snr_vs_t.png` — SNR vs 时间步

- X轴：全局时间步 t (0→1000)
- Y轴：log₁₀(SNR)
- 黑色虚线：DDPM（所有频率相同）
- 彩色实线：RDD 在不同频率 (f=0.1 低频, f=0.3 中频, f=0.5 Nyquist 高频)
- 灰色竖线：阶段边界

**论文用途：** 证明 RDD 的 drift 自动让高频 SNR 先于低频下降，使模型在早期时间步（粗分辨率阶段）专注于低频结构。

### Figure 2: `snr_vs_f.png` — SNR vs 频率

- X轴：归一化频率 (0→0.5)
- Y轴：log₁₀(SNR)
- 虚线：DDPM（水平直线，频率无关）
- 实线：RDD（高频衰减曲线）
- 多条线对应不同时间步 {100, 300, 500, 700, 900}

**论文用途：** 直观展示 RDD 的频率选择性——越高的频率在越早的时间步被破坏。

### Figure 3: `snr_ratio_heatmap.png` — SNR 比值热力图

- X轴：频率
- Y轴：时间步
- 颜色：log₁₀(SNR_RDD / SNR_DDPM)
  - 蓝色 = RDD 比 DDPM 更小（高频被加速破坏）
  - 白色 = 一样
  - 红色 = RDD 比 DDPM 更大（不太可能出现）

**论文用途：** 二维全景展示 drift 的作用区域。阶段边界处可观察到 κ 重置的效果。

### Figure 4: `snr_rdd_vs_dctdiff.png` — RDD vs DCTdiff 对比

- 在 t=500 处横截对比
- DDPM：水平基准线
- DCTdiff：水平线（统一放大 S=4 倍，仍然频率无关）
- RDD：自适应曲线（低频≈DDPM，高频下降）

**论文用途：** 核心差异化论据——DCTdiff 只做全局 SNR 缩放，RDD 做频率自适应。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `.` | 图片输出目录 |
| `--stage-lengths` | `300,300,400` | 逗号分隔的阶段长度（finest→coarsest），须加和为 T |
| `--beta-min` | `0.0001` | 线性 beta schedule 最小值 |
| `--beta-max` | `0.02` | 线性 beta schedule 最大值 |
| `--T` | `1000` | 总时间步数 |
| `--dpi` | `300` | 输出图片分辨率 |
| `--compute-speed-optimal` | (flag) | 计算并打印 SPEED 理论最优 stage lengths |

## SPEED 最优 Stage Lengths 计算

基于 SPEED (Stanford 2025) 论文的理论，给定自然图像功率谱 P(ω) = A·|ω|^{-β}：

```bash
python visualize_snr.py --compute-speed-optimal
```

会输出：
- 功率谱参数
- 各尺度 Nyquist 频率对应的 activation time
- 理论最优 stage_lengths

**进阶用法：** 要测量真实的 VAE latent 功率谱参数：

```python
import torch
import numpy as np

# 加载一批 VAE latent (N, C, 32, 32)
latents = torch.load('some_latents.pt')

# 计算 2D DCT 功率谱
from scipy.fft import dctn
power = np.zeros(16)  # 按频率半径分桶
for i in range(len(latents)):
    dct = dctn(latents[i].numpy(), axes=(-2, -1))
    # ... 按径向频率分桶求均值
    
# 拟合 P(f) = A * f^{-beta}
# 用 np.polyfit(log(f), log(P), 1) 得到 -beta 和 log(A)
```

## 论文写作建议

在论文的 "Theoretical Analysis" 或 "Method" 小节中：

1. 先推导公式（参考 `RDD_survey.typ` 中 §1.2 的推导）
2. 引用 Figure 4 说明 RDD vs DCTdiff 的差异化
3. 引用 Figure 1 说明"高频先破坏"的归纳偏置
4. 引用 Figure 3 展示 drift 的全局效果

建议配文：
> "Unlike DCTdiff which applies a uniform SNR scale factor S across all frequencies, our drift mechanism achieves frequency-adaptive SNR scheduling: low-frequency components are preserved throughout the diffusion process, while high-frequency details are progressively attenuated according to κ(t)·LP(f)."
