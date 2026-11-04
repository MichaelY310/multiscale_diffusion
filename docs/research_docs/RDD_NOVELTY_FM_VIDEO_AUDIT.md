# RDD novelty、FM推导与Video实现审计（2026-08-10）

> **2026-11-02 后续核查更正**：下面关于“残差化表达”和“projection clean inverse”辨识度的判断偏强。[Pyramid Flow 对照报告](../../../unified_multires/PYRAMID_FLOW_RDD_OVERLAP_20261102_ZH.md) 证明 PF 也可写成同一残差路径族，并在相同投影假设下有解析 clean 逆。RDD 的具体阶段曲线不同，但这些能力不是其独有贡献；默认 fresh 边界也不能凭 oracle clean 推导声称网络推理下分布精确匹配。旧文保留作历史，不将旧评分和效果推断当作本轮证据。

## 结论先行

RDD不是“第一个多分辨率/变换信号扩散模型”，也不能把“低分辨率处理高噪声、逐步升分辨率”本身作为novelty。f-DM已经给出广义transformed-signal diffusion；Pyramidal Flow、PyramidalWan、TPDiff、DFM、LapFlow、Scale Space Diffusion和Spectral Progressive Diffusion覆盖了相邻的大块设计空间。

RDD仍然有可防守、而且组合起来相当有分量的贡献：

1. **同状态残差化概率路径。** 对任意同形状lift `G: X -> X` 使用 `d(x)=G(x)-x`，把变换残差嵌入标准VP Gaussian path，并保持一个标准DiT prediction head。平均池化+重复上采样只是幂等投影实例，不是RDD定义。
2. **有效训练目标、Gaussian innovation和deterministic drift三者的显式分离。** 网络训练预测的effective target不能直接当作score/PF-ODE中的Gaussian noise；这一点决定posterior和solver是否正确。
3. **该残差路径下的完整reverse calculus。** 包括stage-local Markov transition、任意jump posterior、连续SDE、noise-filling marginal match以及drift-aware exponential-integrator identity。它们是RDD路径的严谨闭环，但不是“比f-DM更一般”的理论。
4. **RDD-FM的投影clean inverse。** RDD-FM不是把reverse SDE换个名字，而是一条独立conditional path。对线性幂等投影，可以解析恢复clean sample；这比“把金字塔用于FM”更具体，也更容易做因果实验。
5. **受控的ImageNet训练收敛结果。** 400K、DDPM250、CFG1.0下RDD FID 7.06对DiT 19.78是当前最强的实证资产。它证明“该parameterization很有效”，但在完成`d=0`和f-DM-style controls前不能证明原因必然是drift curriculum。

最稳妥的总定位是：

> RDD is a residualized same-state construction within the broader transformed-signal design space. Its contribution is not the first coarse-to-fine path, but a tractable VP/FM parameterization that exposes one transform residual, admits exact reverse corrections, and empirically changes the convergence and compute frontier of a standard DiT.

## Novelty强度排序

### 1. 最强：残差化同状态parameterization及其受控收敛效果

f-DM允许更一般、甚至dimension-changing的transform，因此RDD不能按“generality”赢。RDD真正的区别是主动选择`G(x)-x`并保持fixed state、isotropic covariance、single prediction target和standard backbone。这个限制不是“只为了DiT的缩水版”；它换来了可复现的工程和解析性质。

审稿人会问：收益来自residual parameterization、低分辨率compute allocation，还是普通transformed-signal training？必须用以下四路矩阵回答：

- fixed-resolution DiT；
- 相同stage/token allocation但`d=0`；
- matched f-DM-style transformed-signal parameterization；
- full RDD。

### 2. 很强但属于technical novelty：effective target与pure innovation分离

RDD训练时可写成

`x_t = alpha_t x_0 + sigma_t tilde_epsilon`，

但真实conditional mean是`alpha_t(x_0+r_t d)`。因此solver需要

`epsilon_pure = [x_t-alpha_t(x0_hat+r_t d_hat)]/sigma_t`，

不能直接使用`tilde_epsilon`。这项区分使posterior、PF-ODE和DPM update自洽，是很好的method section核心，但不要把它写成新的score-matching理论。

### 3. 很有辨识度：RDD-FM projection clean inverse

在data time `t=0`为noise、`t=1`为data，stage `[s_k,s_{k+1}]`使用

`x_t = t x_1 + (1-t) epsilon + b_k(t)[P_k x_1-x_1]`,

`b_k(t)=t(s_{k+1}-t)/(s_{k+1}-s_k)`。

velocity为

`dx/dt=x_1-epsilon+b'_k(t)(P_k-I)x_1`。

令`y=x_t+(1-t)dx/dt`、`c=b+(1-t)b'`，则

`y=[I+c(P-I)]x_1`，

所以对`P^2=P`且`1-c != 0`，

`x_1=P y+(I-P)y/(1-c)`。

Wan使用`σ=1-t`并预测`w=dx/dσ=-dx/dt`，因此`y=x_σ-σw`。FM并不从RDD reverse SDE“推导出来”；二者共享residual design principle，但定义不同的probability path。

### 4. 中等：posterior、SDE、noise filling和drift-aware solver

这些公式值得作为贡献，因为它们防止方法沦为heuristic，也能直接驱动实现测试。但它们很大程度上是选定Gaussian residual path之后的严谨推论。论文应写“we derive the exact reverse process for this parameterization”，不写“we introduce the first transformed-signal reverse process”。

连续SDE正确形式为

`dx=[f(t)x+a(t)]dt+g(t)dW`，

`f=d log(alpha)/dt`，`a=alpha dot(r)d`，

`g^2=d(sigma^2)/dt-2f sigma^2`。

仅写`g^2=d(sigma^2)/dt`会漏掉线性drift对covariance的传播。

### 5. 只有正结果后才强：RDD-Wan

“把pyramid用于video”不新。RDD-Wan能成为贡献的条件是，在相同Wan checkpoint、stage boundaries、data、solver和compute下至少证明一个轴：

- 更快fine-tuning convergence；
- 更好的VBench--NFE或VBench--latency frontier；
- analytic clean inverse带来稳定、可重复的改善。

如果只有“能生成video”，应放附录；如果四路因果矩阵和matched-NFE曲线都为正，才适合进入标题级贡献。

## 与最相关工作的诚实边界

| Work | 已覆盖的空间 | RDD不能声称 | RDD可比较的区别 |
|---|---|---|---|
| [f-DM (ICLR 2023)](https://openreview.net/pdf?id=iBdwKIsg4m) | stage-internal transformed signals、dimension changes、reverse conditionals、boundary noise | transformed-signal diffusion或noise resampling首创 | same-state residual VP target、isotropic posterior、standard DiT、controlled convergence |
| [Scale Space Diffusion (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Mukhopadhyay_Scale_Space_Diffusion_CVPR_2026_paper.pdf) | generalized linear degradations、anisotropic posterior、Flexi-UNet | generalized linear scale diffusion首创 | fixed-dimensional tractability与backbone simplicity |
| DFM / LapFlow | multiscale flow decomposition、time-segmented residual generation | progressive FM首创 | same-state transform residual与analytic projection reconstruction |
| [Pyramidal Flow (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ab228c4703c4459b1a600ebadc5732c-Abstract-Conference.html) | unified spatiotemporal video pyramid与aligned links | unified pyramidal video FM首创 | stage-interior residual path和clean inverse的受控作用 |
| [PyramidalWan (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Korzhenkov_PyramidalWan_On_Making_Pretrained_Video_Model_Pyramidal_for_Efficient_Inference_CVPR_2026_paper.pdf) | pretrained Wan低成本pyramidal fine-tuning、orthogonal transforms、step distillation | pretrained-Wan pyramid、generalized transition或pyramidal distillation首创 | 在完全相同设置下只替换interior path的比较 |
| [TPDiff (ICLR 2026)](https://openreview.net/forum?id=Eg3KqoI9tS) | temporal-resolution probability-flow partition | temporal pyramid首创 | joint spatial-temporal residual及其path-specific inverse |
| [Spectral Progressive Diffusion (2026)](https://arxiv.org/abs/2605.18736) | pretrained image/video的spectral schedule、training-free和fine-tuned版本 | spectral coarse-to-fine首创 | training-time residualized path和ImageNet convergence |

## 论文叙事建议

### 应该强调

1. 用一句话先承认f-DM的广义优先性，立即转向RDD选择的“residual same-state tractable corner”。主动承认比让审稿人指出更安全。
2. Contributions按证据排序：ImageNet convergence → residual path → reverse correctness → measured inference compute → FM/video falsifiable extension。
3. 把理论写成三层：path definition；exact reverse/solver correctness；mechanism hypothesis。不要把第三层的curriculum解释伪装成定理。
4. 用`G`定义RDD、用average pooling定义当前instance。一般化条件写清：forward marginal只需deterministic same-shaped residual；projection decomposition和FM inverse额外要求linear/idempotent；pixel VAE re-encode不满足这些条件。
5. Video只讲“interior path是否有额外价值”，不讲“第一个video pyramid”。

### 应该避免

- “first continuous resolution transition”——f-DM等工作会直接反驳。
- “PyramidalWan不能使用高阶solver”——错误；边界内可用，边界重采样后重置history即可。
- “RDD与REPA/REG/SiT自动正交”——只能说plausibly composable，必须实验。
- “deterministic residual必然降低gradient variance”——不成立，需测量。
- “早期stage一定drift dominated”——DNR由递增stage envelope和递减SNR共同决定，且stage入口残差系数为零。
- “pixel projection也能直接使用analytic inverse”——VAE re-encode非线性，不满足`P^2=P`。

## 当前video FM代码原先为什么不能直接蒸馏

### 概念问题

旧`distill`模式只是让冻结Wan生成terminal clean latent，再对这些样本做普通RDD conditional FM。这是**offline/online synthetic-data fine-tuning**，不是few-step distillation。它没有让student匹配teacher的两步轨迹、PF-ODE endpoint、score difference或DMD distribution objective，因此不会自动减少NFE。

真正蒸馏至少需要：固定teacher和student step schedule；耦合noise/trajectory；明确teacher target；student少步objective；以及RDD和PyramidalWan使用同一distillation recipe的公平对照。

### 原实现问题

1. Wan预训练使用noise time `σ=1-t`，旧代码曾混用data time，velocity符号也不统一。
2. 旧clean shortcut `x_1=x_σ-σw`在RDD residual path上不成立。
3. 被称为`dpm2`的实现实际是AB2，不是DPM-Solver-2。
4. 每个DDP rank同时驻留student、EMA、teacher和T5，显存开销远大于普通fine-tuning。
5. unconditional context使用全零tensor，而官方Wan使用negative-prompt T5 embedding。
6. T5调用缺少`device`参数，并错误地从batch context list中只取第一个样本。
7. 合成latent与prompt分开保存，precompute后`text_emb`丢失；训练会静默退化成零context。
8. 多GPU resume只扫描连续index，对interleaved shards无效。
9. 81帧对应21 latent frames，旧repeat transition产生`6->12->24`而不是`6->11->21`。
10. pixel downsample + VAE re-encode被当作latent幂等投影，理论和实现不一致。

### 已完成修复

- 明确paper data-time与Wan sigma-time，训练target改为`dx/dσ`。
- 实现projection residual的exact clean inverse。
- Euler、AB2、Heun正确命名并在stage boundary重置history。
- stage按区间长度proportional sampling，对应uniform global time。
- DDP调用wrapper而不是绕过reducer hooks；device使用`LOCAL_RANK`。
- 合成数据原子保存`latent + text_emb + prompt + teacher seed`；缺失text context默认fail fast。
- 使用官方negative prompt embedding；旧`distill`改为`online_synthetic`语义并保留deprecated alias。
- 修复interleaved multi-GPU resume和奇数latent-frame transition。
- 记录真实conditional/unconditional model forward calls、分stage calls、text/denoise/decode latency和peak memory。
- RDD-Wan主路径限制为latent idempotent projection；pixel VAE re-encode明确要求另一个clean mechanism。

本地已经通过Python compile、12个CPU数学/solver tests以及全部shell入口的`bash -n`。尚未在本机执行真实Wan 1.3B GPU forward，因为本机没有Wan源码和权重；`setup_wan21.sh`与`RUN_GPU_SMOKE=1`的一步真实checkpoint测试用于服务器最终验收。

## 最小、但足够有说服力的实验包

### P0：投稿前必须完成

1. **Image mechanism matrix**：DiT、staged `d=0`、f-DM-style、RDD；至少小模型2个training seeds，关键端点50K FID。
2. **Image matched-NFE**：DDPM / first-order / midpoint在64、124、244真实NFE，统一CFG和50K reference；同时报告wall-clock和peak memory。
3. **RDD-FM image validation**：在同一SiT backbone/protocol上至少一个可信50K FID checkpoint。负结果也要据实收缩“diffusion+FM统一性”主张。
4. **RDD-Wan four-way pilot**：Vanilla、PyramidalWan、staged `d=0`、RDD，0/1K/2K/5K；先小prompt set筛选，趋势为正再跑完整VBench。
5. **Video matched-NFE curve**：两种pyramidal方法都跑Euler/AB2/Heun，记录真实model calls和相同seed。

### P1：最值得增加、但不应扩散范围

1. 一个非average-pool的**线性幂等同状态transform**（例如正交wavelet/spectral projection）小规模验证。这比再加十种stage allocation更能支撑`G`级别的method claim。
2. 只选择一个强正交组合：RDD+REPA优先，或RDD+REG；不要同时铺开四个组合。
3. ImageNet-512用L级模型做pilot，只有256因果闭环完成后再投入。

### 暂缓

- Video few-step distillation：先证明undistilled RDD-Wan有独立收益，再对RDD和PyramidalWan用同一DMD/progressive recipe。
- 14B从头训练、80K以上数据扩张、solver zoo、更多CFG sweep。

## 当前论文诚实评分

| 方面 | 当前分数 | 做完P0且结果正常 | 主要短板 |
|---|---:|---:|---|
| Novelty定位 | 6.5/10 | 7.5/10 | 宽泛idea拥挤，必须靠residual path与因果control |
| 理论完整性/正确性 | 7.5/10 | 8.0/10 | 公式闭环已较强，但多为特定path推论；最好补独立proof review |
| Image实验 | 6.5/10 | 8.0/10 | 主结果强，缺`d=0`/f-DM-style和training-seed统计 |
| FM实验 | 2.5/10 | 7.0/10 | 推导和代码已有，尚无生成质量结果 |
| Video实验 | 1.5/10 | 7.0/10 | 当前只有实现和计划，没有真实Wan/VBench结果 |
| 写作与诚实边界 | 7.5/10 | 8.5/10 | 已承认prior overlap；仍需压缩篇幅与清理旧归档源文本 |
| 可复现性 | 6.5/10 | 8.0/10 | CPU tests/scripts已齐；真实Wan GPU smoke和baseline接线未完成 |
| NeurIPS readiness（整体） | **5.5/10** | **7.5/10** | 目前像“强结果+待完成因果闭环”，还不是稳收稿包 |

这不是对方法潜力的低估。RDD的400K结果确实值得认真推进；但NeurIPS审稿最可能攻击的是“是否只是普通pyramid/f-DM parameterization加低分辨率compute”。只要最小因果矩阵成立、RDD-FM有一个可信结果、video pilot至少显示一个独立正轴，论文会从“结果惊人但解释未闭环”升级为“定位清楚、理论自洽、跨范式有效”的强投稿。Spotlight还需要更强条件：机制control非常干净，且Image FM或Video中至少一个出现明显而可重复的Pareto改善。

## 一手参考

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Wan technical report](https://arxiv.org/abs/2503.20314)
- [Official Wan sigma schedule and solvers](https://github.com/Wan-Video/Wan2.1/blob/main/wan/text2video.py)
- [VBench (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html)
