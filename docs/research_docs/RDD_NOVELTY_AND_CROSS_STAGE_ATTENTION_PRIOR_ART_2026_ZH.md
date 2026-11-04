# RDD novelty 与跨分辨率 attention prior-art 审计

> 审计日期：2026-09-07  
> 目的：回答两件事：当前工程是否已经“只差训练”；RDD、RDD-Video 和 C2F-Route 到底还能安全声称什么 novelty。  
> 结论先说：**零微调原型与真实性能实验已经完成；投稿级工作不是只差按下训练按钮。** RDD 不能声称首次多分辨率 diffusion，也不能声称首次把 progressive resolution 与 sparse attention 结合。当前最有辨识度的系统贡献，是把**前一个真实生成分辨率 stage 已经计算过的 attention route**直接映射给下一 stage，从而不在高分辨率 stage 重新构造 coarse router。本文检索到概念上很接近的视觉自回归方法 SparVAR，但尚未在所检查的 diffusion/video-diffusion 工作中发现相同做法。

> **2026-09-09 更新：** 后续公平消融表明，单独跨 stage 复用 route 的端到端收益太小，已不再作为主要速度贡献。新的主系统方法是 [Cross-Scale Block Residual Reuse（CSBR）](RDD_CROSS_SCALE_BLOCK_REUSE_RESULTS_ZH.html)：保存已完成 coarse RDD stage 的整 block residual，并在选定 fine layers 直接放大、注入，从而跳过整个 fine Transformer block。相对完全相同的 Spatial-421 RDD + 在线 VSA-20，温和设置稳定降低 12–16% denoiser 延迟；激进设置降低 51–61%，且在包含当前 VAE 后仍降低 11–15% 总延迟。检索到的 Pyramidal Flow/PyramidalWan 有真实低到高轨迹，DraftAttention/VSA 有当前调用内部的 coarse proxy，generic cache 有跨 timestep/layer 复用；目前没有发现同时使用“已完成的上一生成分辨率 stage”与“跨尺度整 block residual 替代 fine block”的视频扩散方法。该结论应使用“在我们检索到的公开工作中尚未发现”的限定措辞，而不是无条件 first claim。

## 1. 完成度审计

### 1.1 如果“任务”指上一轮零微调加速原型：完成度 100%

已经完成：

- 在同一个 `FastWan2.1-T2V-1.3B-Diffusers` checkpoint、相同 DMD 3 NFE、相同 prompt/seed、832×448×29 帧下完成公平比较；
- baseline、纯 Temporal RDD、RDD+在线 VSA、RDD+C2F、RDD+C2F+StepReuse 均可运行；
- 所有分辨率 transition 统一使用 noise filling；
- C2F 与 VSA 使用同一个 FastVideo block-sparse attention executor，避免靠不同底层 kernel 制造不公平速度差；
- 已实现并实测 Triton route-lift kernel；
- 已做 50/25/20/12.5% keep 的同密度 sweep、前后夹测、20 次 warmup + 50 次重复稳定性复核；
- 已生成三个 prompt 的配对视频并检查视觉失败模式；
- 已记录原始 metrics、视频、contact sheet、命令和实现入口。

完整结果见 [RDD × FastWan 公平加速报告](RDD_FASTWAN_FAIR_ACCELERATION_SWEEP_ZH.html)。最保守主结果为：

| 方法 | denoiser | 相对固定全分辨率 VSA-20 | sample+VAE |
|---|---:|---:|---:|
| 固定全分辨率 VSA-20 | 0.869 s | 1.00× | 2.019 s |
| Temporal RDD + 在线 VSA-20 | 0.624 s | 1.39× | 1.769 s |
| Temporal RDD + C2F-RouteOnly-20 | 0.607 s | 1.43× | 1.748 s |
| Temporal RDD + C2F + Projected StepReuse(2) | 0.550 s | **1.58×** | **1.690 s / 1.19×** |

长预热复核得到 1.63× denoiser、1.21× sample+VAE。route builder 微基准中，C2F route lift 为 0.031 ms/layer，在线 VSA route builder 为 0.209 ms/layer，即 route 部分 6.66×；但 route 只是整网的一小部分，所以端到端增益远小于 6.66×。

### 1.2 如果“任务”指可投稿论文：约 70%，不能叫 100%

训练确实是当前最大的阻塞项，但训练完成后还剩正式评测。下面这些不是可选装饰：

1. **等预算微调**：固定 Wan/FastWan teacher、数据、update 数、LoRA rank、batch、精度和硬件，至少训练 `RDD+online-VSA`、`RDD+C2F`、`RDD+C2F+Reuse`；Jenga 作为 training-free baseline 单独报告。
2. **质量评测**：足量 prompt 上的 VBench 全维度、至少一种分布/语义指标、人工偏好和置信区间。当前 3 prompt 只能叫 sanity check。
3. **最邻近方法对照**：尤其是 Jenga 的 `ProRes+AttenCarve`。不能只对 VSA。
4. **因果消融**：同 stage schedule 下 `d=0`、完整 RDD；在线当前-stage router、跨-stage route；reuse 间隔 1/2/3；相同 keep 比较。
5. **非 3-step 场景**：在 20–50 NFE 的非蒸馏 checkpoint 上复测。现在 VAE 占比太高，且只有一次 reuse 命中，无法展示方法的完整适用区间。
6. **冷启动与稳态分表**：论文主表用编译后稳态延迟，同时补首次 JIT/新 shape 的冷启动成本。

所以更精确的说法是：**“训练前所需的工程、kernel 和实验脚手架已经 100% 准备好；科学结论仍需要训练后的大样本质量评测与消融。”**

## 2. 先把 RDD 分成三个可能的贡献

否则所有 prior work 会混在一起。

### 2.1 生成路径：RDD

RDD 在同一个 stage 状态空间中定义 lifted transform $G_c$，并写成 residual：

$$
d_c(x)=G_c(x)-x.
$$

VP 形式的边缘路径为：

$$
x_t=\sqrt{\bar\alpha_t}\bigl(x_0+r_c(t)d_c(x_0)\bigr)
    +\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

平均池化再上采样只是 $G_c$ 的一个实例。这里真正特殊的不是“从低分辨率到高分辨率”，而是：

- transform 被 lift 回当前状态空间；
- transform 以 residual 形式连续进入路径；
- covariance 仍保持各向同性；
- drift 可以折叠进一个 VP-style effective target，因此可继续使用标准 DiT 单输出头；
- 对投影型 $G_c$ 可以得到显式 clean reconstruction、任意步 posterior 和 drift-aware update。

RDD-FM 则从自己的 conditional path 推导，而不是套用 DDPM posterior：

$$
x_t=t x_1+(1-t)\epsilon+b_c(t)d_c(x_1),\qquad
u_t=\frac{d x_t}{dt}.
$$

### 2.2 生成系统：RDD-Video

RDD-Video 把空间/时间分辨率变化放进训练路径，让高噪声阶段使用较少 token，低噪声阶段恢复完整 token。仅这一点并不新；真正要验证的是 residualized path、matched noise filling 和可解析 clean reconstruction 是否在相同 Wan checkpoint 与训练预算下改善收敛或质量—延迟前沿。

### 2.3 硬件/attention：C2F-Route 与 StepReuse

- **C2F-Route**：低分辨率 generative stage 本来就必须计算 attention。保存其 block route，按父子 block 关系 lift 到下一高分辨率 stage，直接驱动同一个 sparse executor。
- **StepReuse**：同一 resolution stage 的邻近 NFE 复用已经投影后的 self-attention 输出；当前 residual、cross-attention 和 MLP 仍重新计算。

C2F 的关键价值是“利用已经付过的钱”，不是简单地把当前高分辨率 Q/K 再池化一次。

## 3. 与 RDD 生成路径最接近的方法

| 方法 | 它做了什么 | 与 RDD 的重合 | 仍可区分之处 |
|---|---|---|---|
| [f-DM](https://arxiv.org/abs/2210.04955) | 将手工或学习的 signal transform 作为各 diffusion step 的均值，覆盖 downsampling、blur、VAE transform，并推导训练与采样 | **最强理论先例**；“扩散中加入随时间变化的 transform”不是 RDD 首创 | RDD 是更窄的 same-state residual VP 子类；要把贡献落在单头闭合、各向同性 posterior、projection clean inverse 和具体优化性质上 |
| [Relay Diffusion](https://proceedings.iclr.cc/paper_files/paper/2024/hash/522ef98b1e52f5918e5abc868651175d-Abstract-Conference.html) | 用 blurring diffusion 与 block noise 把低分辨率状态转换成等价高分辨率状态并继续 denoise | 低→高、频域解释、boundary noise 都高度相关 | RDD 把 residual drift 分布在 stage 内，而非只把 resolution relay 当边界机制 |
| [Matryoshka Diffusion](https://proceedings.iclr.cc/paper_files/paper/2024/hash/81166fbd9cc5adf14031cdb69d3fd6a8-Abstract-Conference.html) | NestedUNet 同时联合 denoise 多个分辨率；覆盖图像和视频 | 单模型、多分辨率、渐进训练 | 它维护联合多尺度状态，并非 RDD 的单状态 sequential path |
| [Pyramidal Flow Matching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ab228c4703c4459b1a600ebadc5732c-Abstract-Conference.html) | 单一视频 DiT 在多个时空 pyramid stage 上做 flow matching，并对 stage flow 做连接 | 视频、单模型、低→高、FM 都重合 | RDD-FM 的区别必须是 residualized interior path 与 projection-specific inverse，而不是“首次 pyramidal FM” |
| [Decomposable Flow Matching](https://arxiv.org/abs/2506.19839) | 在用户指定的多尺度分解（如 Laplacian pyramid）各层独立做 FM；用于图像和视频 | progressive generation、单模型、FM、收敛改善 | DFM 直接在 decomposition levels 上学习；RDD 在 same-state lifted residual path 上学习。但两者不是天然正交，需要直接对照 |
| [PyramidalWan](https://openaccess.thecvf.com/content/CVPR2026/papers/Korzhenkov_PyramidalWan_On_Making_Pretrained_Video_Model_Pyramidal_for_Efficient_Inference_CVPR_2026_paper.pdf) | 低成本把 Wan2.1 微调成时空 pyramidal 模型，并研究 step distillation | RDD-Wan 最直接的应用竞争者 | RDD 必须在同 checkpoint/数据/预算下证明其路径或边界分布更好，不能只说“也能 pyramidize Wan” |
| [TPDiff](https://arxiv.org/abs/2503.09566) | 沿视频时间轴逐阶段提高帧率，并分别给出 DDIM 与 FM 的 stage-wise 概率路径、训练目标和边界噪声修正 | 对“视频渐进分辨率 + 分阶段 diffusion/FM 理论”构成非常直接的先例 | 它的 DDIM residual 系数由 noise-to-signal ratio 决定，FM 使用端点直线；均不等于当前 RDD 的固定线性 DDPM drift 或 curved FM envelope |
| [Scale Space Diffusion](https://openaccess.thecvf.com/content/CVPR2026/papers/Mukhopadhyay_Scale_Space_Diffusion_CVPR_2026_paper.pdf) | 建立 generalized linear degradation diffusion 理论；downsampling 形成 Scale Space Diffusion，并用 Flexi-UNet 反向升分辨率 | **对理论 novelty 威胁很强**：resolution-changing diffusion、线性退化、posterior 都已系统研究 | RDD 的剩余空间是受限 residual VP/FM 参数化、标准 DiT 兼容性及其特有闭合性质；不能再说首次为 resolution-changing diffusion 建数学理论 |
| [Spectral Progressive Diffusion](https://arxiv.org/abs/2605.18736) | 根据 power spectrum 推导 resolution schedule，支持图像/视频 training-free 或微调 | 低→高和谱解释重合 | 它强调预训练模型的谱日程；RDD 强调训练路径与 residual drift |
| [Fresco](https://arxiv.org/abs/2601.07462) | progressive upsampling，试图在跨 stage 保留同一全局结构并减少反复 re-noise | 动态分辨率与边界一致性重合 | 没有把上一生成 stage 的 attention route 当下一 stage sparse router |

结论：**RDD 不是一个从无人区出现的 broad paradigm。** 它位于 f-DM、Relay、Pyramidal Flow、DFM、PyramidalWan、SSD 形成的 transformed/progressive family 内。数学推导首先证明“这个具体参数化是正确且易实现”，本身不自动证明概念首创。

### 3.1 纯 RDD-Video 的严格包含关系审计

这里把视频下采样再升采样记为同一状态空间上的线性投影

$$
P_{st}=U_{st}D_{st},\qquad d(x)=(P_{st}-I)x.
$$

#### RDD-DDPM：是 f-DM 的精确特例

RDD 在一个 stage 内的条件边缘分布是

$$
q_{\mathrm{RDD}}(x_t\mid x_0)
=\mathcal N\!\left(
\alpha_t\big[(1-r_t)I+r_tP_{st}\big]x_0,
\sigma_t^2I
\right).
$$

f-DM 的 Eq. (4) 在第 $k$ 个 stage 使用

$$
q_f(z_t\mid x)=\mathcal N(\alpha_t\tilde x_t,\sigma_t^2I),
\qquad
\tilde x_t=(1-r_t)x^k+r_t\hat x^k.
$$

令 $x^k=x_0$、$\hat x^k=P_{st}x_0$，两式完全相同。f-DM Eq. (5) 的 clean-conditioned stage transition 也随同代换成 RDD 使用的加性 transform-residual transition。因此：

- RDD-DDPM 的 **probability path、stage-local forward kernel 和由其得到的 Gaussian posterior** 都是 f-DM transformed-signal diffusion 的特例；
- 把 $P$ 从二维空间投影换成视频时空投影 $P_{st}$ 不产生新的概率理论；f-DM 的定义对确定性 transform 没有限定数据必须是图像；
- RDD 的单一 effective-$\epsilon$ 输出头、固定 residual 参数化及 DiT 实现与 f-DM 的双重重建头不同，但这是 **训练参数化/工程 specialization**，不能反过来说概率模型不是其子类。

所以，若“纯 RDD-Video”指 DDPM 版本且不包含后续 attention routing，那么最诚实的结论是：**它在核心随机路径上确实是 f-DM 的严格子例，而不是独立的新 diffusion family。**

#### RDD-FM：与 Pyramidal Flow/PyramidalWan 同端点，但不是同一条内部路径

设 fine stage 的时间区间为 $[s,e]$，$L=e-s$。把 Pyramidal Flow Eq. (9)--(10) 的两个端点做直线插值，可改写成

$$
x_t^{\mathrm{PF}}
=t x_1+(1-t)\epsilon
+\frac{s(e-t)}{L}(P_{st}-I)x_1.
$$

当前仓库实际实现的 RDD-FM 是

$$
x_t^{\mathrm{RDD}}
=t x_1+(1-t)\epsilon
+\frac{t(e-t)}{L}(P_{st}-I)x_1.
$$

所以两者的精确差值为

$$
x_t^{\mathrm{RDD}}-x_t^{\mathrm{PF}}
=\frac{(t-s)(e-t)}{L}(P_{st}-I)x_1.
$$

差值在两个 stage 端点均为零，在内部非零。这说明：

- RDD-FM、Pyramidal Flow 和采用其损失的 PyramidalWan 拥有相同的 coarse/fine 条件端点；
- Pyramidal Flow/PyramidalWan 在端点间走直线，RDD-FM 在内部额外向低通投影弯曲；
- 因而当前 RDD-FM **不是** Pyramidal Flow/PyramidalWan 的逐公式复制，也不是它们固定直线路径的严格子集；
- 但 RDD-FM 当然仍是 general conditional Flow Matching 的一个特定 conditional path。仅“选择了一条新曲线”通常不足以构成强理论贡献，必须证明该曲线改善收敛、误差、低 NFE 质量或跨尺度稳定性。

PyramidalWan 的 Eq. (1)--(4) 明确复用了 Pyramidal Flow 的端点直线，并把该方法用于 Wan2.1 时空金字塔微调。因此“首个 pyramidal video FM”“首个把 Wan 从低时空分辨率生成到高分辨率”都不能作为 RDD-Wan claim。

#### TPDiff：与 RDD 属于同一个 residual-path 形状族，但 schedule 不同

TPDiff 对 temporal pyramid 同时推导了 DDIM 与 FM。将其 DDIM Eq. (10)--(11) 展开后，也能写成

$$
x_t=\gamma_t\big[(1-r_{\mathrm{TP}}(t))I+r_{\mathrm{TP}}(t)P_t\big]x_0
+\sigma_t\epsilon,
$$

但 $r_{\mathrm{TP}}(t)$ 由 $\lambda_t=\sigma_t/\gamma_t$ 在 stage 端点间的插值决定，而当前 RDD-DDPM 使用原始 diffusion timestep 上的线性 $r_t$。TPDiff 的 FM 版本则仍是 stage 端点直线。于是 TPDiff 不是当前 RDD 两个固定公式的精确复制，却已经覆盖了以下更宽的主张：视频 temporal resolution 随 denoising 提升、stage-wise diffusion/FM 训练目标、单模型多阶段采样和带噪边界连接。

#### Scale Space Diffusion：边缘均值族包含 RDD，但完整 joint process 不宜宣称完全等价

SSD 写成

$$
q(x_t\mid x_0)=\mathcal N(M_{1:t}x_0,\sigma_t^2I).
$$

取

$$
M_{1:t}=\alpha_t\big[(1-r_t)I+r_tP_{st}\big]
$$

即可得到 RDD 的单时刻条件边缘分布。因此 RDD 位于 SSD 所讨论的 generalized linear-Gaussian degradation **marginal family** 内。可是 SSD 从 state-only Markov transition $x_t=M_tx_{t-1}+\eta_t$ 出发，并处理由此产生的非各向同性一步/后验协方差；RDD 的 stage kernel 显式依赖 clean $x_0$。两者完整 joint process 并不应直接写成“完全同一个模型”。安全结论是：SSD 吞掉了“线性 resolution degradation + Gaussian marginal + posterior theory”这条 broad novelty，RDD 仍可主张其受限 VP/DiT 参数化的实现简洁性。

#### 最终分类

| 对照 | RDD-DDPM-Video | RDD-FM-Video | 判断 |
|---|---|---|---|
| f-DM | 代换 $\hat x=P_{st}x$ 后概率路径与 stage kernel 完全相同 | f-DM 不是 FM 方法 | **DDPM 核心理论是其精确子例** |
| Pyramidal Flow / PyramidalWan | 不属于其主要 FM 公式 | 同端点，RDD 多一个内部弯曲项 | **非常接近但不完全相同** |
| TPDiff | 同 residual 结构，drift schedule 不同 | TPDiff 使用端点直线 | **同族直接竞争者，不是逐公式相同** |
| Scale Space Diffusion | 条件 marginal 可精确嵌入；完整 Markov joint 不同 | 非 FM | **marginal 层面被更广线性族包含** |
| UDPM / Relay Diffusion | 都做低到高及噪声修正，但 forward/operator 构造不同 | 非当前 RDD-FM 曲线 | **相邻方法，不是精确子集关系** |
| General Conditional FM | 非 FM | 任意合法 RDD 曲线都属于其定义 | **形式上是子例，但这是所有 FM 方法共有的上位关系** |

截至本次逐式检索，没有找到在视频上同时使用 **当前 $t(e-t)/L$ curved residual envelope、projection-specific clean inverse、相同 noise-filling 约定** 的已发表方法。但这只留下一个很窄的路径设计区别；在没有 matched-control 实验前，不能把它包装成一个全新的多分辨率视频生成理论。

## 4. 视频：低分辨率是否已经被用于指导高分辨率 sparse attention

### 4.1 最危险的直接竞争者：Jenga

[Jenga](https://proceedings.neurips.cc/paper_files/paper/2025/file/75b2a6632ab7b9196e9cb93750986e45-Paper-Conference.pdf) 已经把两件事放在同一个 training-free 视频系统中：

1. Progressive Resolution：前段低分辨率、后段高分辨率；
2. Attention Carving：将当前 3D video tokens 分块，构造当前 block-wise attention mask，再运行定制 block-sparse kernel。

因此以下 claim 已经不能写：

- 首次把 progressive resolution 和 sparse attention 结合；
- 首次在多分辨率视频生成中使用 sparse attention；
- 首次用 coarse information 帮助 fine attention。

但 Jenga 的公开算法在每个 stage、每个 timestep 调用 `BuildMask(Q_v,K,...)`：它对**当前 stage 当前层的 Q/K**做 block pooling/scoring，再得到 mask。Progressive Resolution 与 AttenCarve 是并列的两项加速，前一 generative stage 的 attention route 没有被直接传给后一 stage。

这恰好留下 C2F 的窄区别：

$$
M_{s+1}=\operatorname{Lift}(M_s),
$$

其中 $M_s$ 是 stage $s$ 真正 denoise 时已经得到的 route，不是 stage $s+1$ 内为 routing 新算的 proxy。

### 4.2 DraftAttention、VSA 与 PSA：都是“当前高分辨率调用内造一个 coarse proxy”

- [DraftAttention](https://arxiv.org/abs/2505.14708) 把**当前完整分辨率** Q/K reshape 后做 2D average pooling，计算低分辨率 draft attention，再据此重排并执行 full-resolution sparse attention。它没有先完成一个低分辨率 generative stage，也没有跨 stage route cache。
- [VSA](https://arxiv.org/abs/2505.13389) 在当前 attention layer 内把 Q/K/V 聚合成 tiles，当前 coarse score 选出 critical tiles，fine sparse branch 再算被选 token；coarse/fine 输出由学习 gate 混合。它需要训练，并且每个调用仍产生当前 coarse branch。
- [Pyramid Sparse Attention](https://arxiv.org/abs/2512.04025) 在同一次 attention 中使用多层 pooled KV 与动态 pooling level。它是 attention 内部层级化，不是生成 trajectory 的跨 stage route reuse。

所以“low-resolution attention guides high-resolution sparse attention”这个宽句子已经属于 prior art；必须加上 **previous generative stage / no current-stage router** 两个限定词。

### 4.3 跨 timestep route reuse 也已有大量先例

- [RAPID](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_RAPID_Reusing_Attention_Sparsity_with_Inter-step_Adaptation_for_Efficient_Video_CVPR_2026_paper.pdf) 在 early step 做一次 dense scoring，缓存 block importance/mask，随后跨 denoising steps 复用并适应；
- [AdaSpa](https://arxiv.org/abs/2502.21079) 使用 cached search/LSE 等机制降低在线 sparse-index 搜索开销；
- [DiTFastAttn](https://arxiv.org/abs/2406.08552) 包含跨 timestep attention sharing；
- [TeaCache](https://arxiv.org/abs/2411.19108) 根据 timestep-aware 输入差异决定何时复用 block output；
- [LiteAttention](https://arxiv.org/abs/2511.11062) 利用相邻去噪步 sparse pattern 的时间一致性传播 skip decision。

因此 `StepReuse` 应被写成与 RDD/C2F 正交的实现组件和消融，不能单独包装成核心原创方法。

### 4.4 一个非常接近 C2F 的非 diffusion 先例：SparVAR

[SparVAR](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_SparVAR_Exploring_Sparsity_in_Visual_AutoRegressive_Modeling_for_Training-Free_Acceleration_CVPR_2026_paper.pdf) 在 visual autoregressive next-scale generator 中观察到 cross-scale attention pattern 相似，从一个“sparse decision scale”把 pattern 映射到后续高分辨率 scales，并配套 block-sparse kernel。

它与 C2F 的核心直觉非常接近，所以不能说：

> 我们首次在任何生成模型中用低 scale 的 attention pattern 决定高 scale sparse attention。

但仍有清晰差异：

- SparVAR 是离散 next-scale autoregressive generation；RDD 是 diffusion/FM denoising trajectory；
- SparVAR 的历史 scales 本来就是 VAR 上下文的一部分；RDD 跨 stage 后只保留当前 noisy state，route cache 是额外引入的计算 metadata；
- RDD 的 stage transition 需要与随机路径/noise filling 一致，并处理不同噪声水平下 route 的生命周期；
- 当前检索没有发现 SparVAR 用于 video diffusion。

它必须进 related work，否则 reviewer 很容易指出概念先例。

### 4.5 图像生成侧的结论

图像 diffusion 中，“先得到低分辨率结构，再指导高分辨率 refinement”也早已有多种形式：

- [DiffuseHigh](https://arxiv.org/abs/2406.18459) 利用已经生成的低分辨率图像指导更高分辨率生成；
- [AP-LDM](https://arxiv.org/abs/2410.06055) 通过 parameter-free self-attention 改善 progressive high-resolution generation 的结构一致性；
- [ASGDiffusion](https://arxiv.org/abs/2412.06163) 使用 attention-mask-weighted low-resolution noise 作为异步高分辨率结构指导；
- [FlexiDiT](https://arxiv.org/abs/2502.20126) 让同一个 DiT 在 denoising trajectory 中使用不同 patch size/compute budget，图像与视频都能减少 FLOPs；
- [LLSA](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Trainable_Log-linear_Sparse_Attention_for_Efficient_Diffusion_Transformers_CVPR_2026_paper.html) 在当前 attention 内做 hierarchical Top-K 与多粒度 KV enrichment，把超长图像 token attention 降到 log-linear complexity。

这些工作进一步说明“低分辨率帮助高分辨率”和“多尺度 sparse attention”都不能单独作为 novelty。就本文检查到的图像 **diffusion** 工作而言，仍未看到与 C2F 完全相同的 previous-generative-stage route lift；但 SparVAR 已在图像 **visual autoregression** 中建立了明确的跨 scale pattern propagation 先例，所以论文必须把 claim 限定到 diffusion/FM trajectory。

## 5. 目前没有找到的精确组合

截至 2026-09-07，在上面列出的论文、公开算法和代码说明中，未找到同时满足以下全部条件的方法：

1. 模型沿 diffusion 或 flow-matching trajectory 真正从低时空分辨率生成到高时空分辨率；
2. 低分辨率 stage 的 attention 是生成过程本身已经计算的，不是高分辨率 layer 内临时池化出来的 proxy；
3. 该 attention route 被显式 lift 到后一个 stage；
4. 后一个 stage 直接用它调用 block-sparse executor，从而避免当前-stage Q/K pooling、coarse $QK^\top$ 和 router；
5. route reuse 与训练路径一起优化，并有相同 checkpoint/数据/keep/kernel 的公平质量—速度验证。

这是一项文献检索结论，不是专利级“绝对不存在”证明。论文应使用 **“To our knowledge”** 并精确定义 scope。

## 6. RDD novelty 排名（强到弱）

### 1. 跨生成 stage 的零额外 router：最有辨识度的系统贡献

推荐名称：**Generative Cross-Stage Attention Routing (G-CSAR)**，保留 `C2F-Route` 作为实现名。

一句话：前一个低分辨率 denoising stage 同时产生图像/视频状态和下一 stage 的 sparse-compute plan；后一 stage 无须再从自己的 Q/K 重建 coarse route。

它区别于：

- Jenga：progressive resolution 与当前-stage AttenCarve 并列；
- DraftAttention/VSA：当前高分辨率调用内再造 coarse proxy；
- RAPID：同分辨率跨 timestep cache；
- SparVAR：相近的跨 scale 直觉，但生成范式是 visual autoregression。

当前 kernel 已证明这个 route 能真正省计算；但相对 `RDD+online VSA` 的 denoiser 增量只有约 2.7%。所以它是**清晰 novelty，不是已经证明的巨大端到端收益**。在长序列、更多 NFE 或 router 占比更高的模型上是否扩大，必须实测。

### 2. RDD 的 same-state residual VP/FM 参数化：中等强度理论贡献

最安全的叙事不是“首次让 resolution 随 diffusion 改变”，而是：

> 在 generalized transformed-signal family 内，RDD 识别出一个可 tractably residualize 的 same-state 子类；该子类保留标准 DiT 单头与各向同性 covariance，同时允许显式区分 effective prediction target、Gaussian innovation 与 deterministic transform drift，并在 projection lift 下给出闭式 clean inverse 和采样修正。

这一点具有技术价值，但 f-DM 与 SSD 使它不太可能被视为全新理论范式。

### 3. “生成路径同时产出计算计划”的算法—硬件协同：有潜力

RDD 不只是少算 coarse tokens；它让 coarse stage 的内部结构成为后续 kernel 的路由 metadata。这个叙事把生成理论和系统贡献连起来，比“RDD 后面再外挂一个 sparse attention”更完整。

要变强，训练 loss 应显式加入 coarse/fine route consistency、coverage 或稳定性约束，并证明：同质量下 C2F route 的构造成本低于 VSA/Jenga 当前-stage router。

### 4. projection-specific posterior、noise filling 与 FM clean inverse：正确性贡献

这些推导重要，但单独 novelty 较弱：

- arbitrary-jump posterior/SDE 很可能被 reviewer 视为选定线性 Gaussian path 后的代数结果；
- boundary noise resampling 在 f-DM、Relay、Pyramidal Flow 等已有近邻；
- “isotropic family 内唯一匹配”是一个窄而清楚的命题，适合当 proposition，不适合当摘要第一贡献。

### 5. convergence acceleration：可能是最有影响力的 empirical finding，但不是概念 novelty

若正确 ImageNet 数据、同 compute、同 architecture 下收敛明显更快，这会显著提高论文价值。必须用 `d=0` progressive-resolution、f-DM-style parameterization 和标准 diffusion/FM control 排除“只是便宜 curriculum”或实现差异。

### 6. StepReuse：有效，但不应作为主要 novelty

已有 RAPID、AdaSpa、DiTFastAttn、TeaCache 等先例。它适合写成与 G-CSAR 正交的 acceleration module，以及展示组合 Pareto frontier。

## 7. 论文应该怎样讲

### 7.1 推荐主叙事

> Existing transformed-signal and pyramidal methods show that visual generation can move from coarse to fine, while sparse-attention methods build a coarse proxy from the current high-resolution activations. RDD studies a tractable same-state residual path that preserves a standard VP/FM backbone. More importantly for video efficiency, its coarse generative stages already compute structured attention. We turn this already-paid computation into the sparse routing plan of later stages, eliminating repeated current-resolution routing rather than adding another coarse branch.

中文就是：

> 别人已经证明低→高可行，也已经证明 coarse attention 可以指导 sparse attention。我们的关键不是重复这两句话，而是让 **RDD 的真实低分辨率生成过程一物两用**：既生成 coarse content，又免费产出 fine stage 的计算路线。

### 7.2 推荐贡献顺序

1. same-state residual RDD path 与适用条件；
2. diffusion/FM 的 projection-specific reverse/clean reconstruction 和 matched transition；
3. G-CSAR：上一 generative stage route 直接驱动下一 stage sparse kernel；
4. 在图像与视频上的 convergence—quality—latency Pareto，以及同 stage schedule、同 kernel、同训练预算消融。

如果 video 训练结果很强，可以把 3 提到 1；如果只有弱微调结果，则仍以 RDD 图像路径为主，G-CSAR 放系统扩展。

### 7.3 禁止写的句子

- “RDD is the first multi-resolution diffusion model.”
- “RDD is the first continuous resolution-changing generative process.”
- “We are the first to combine progressive-resolution video generation with sparse attention.”
- “We are the first to use low-resolution attention to guide high-resolution attention.”
- “StepReuse is a novel cross-timestep caching method.”
- “The derivation of a posterior proves novelty.”

### 7.4 可以写的、经过限定的句子

> To our knowledge, RDD is the first diffusion/flow video framework to reuse the attention routing computed by an earlier **generative resolution stage** as the block-sparse routing plan of a later stage, avoiding a fresh current-resolution coarse router.

以及：

> RDD is a residualized same-state specialization of transformed-signal diffusion. Its value lies not in claiming the broad idea of progressive signal transformation, but in the resulting single-head VP/FM parameterization, projection-specific reverse calculus, and the ability to recycle already-computed coarse-stage structure for later sparse execution.

## 8. 训练后必须回答的 reviewer 问题

| Reviewer 会问 | 最小实验 |
|---|---|
| 提升来自 RDD drift，还是只来自低分辨率算得便宜？ | 相同 stage schedule 的 `d=0` 与完整 RDD |
| C2F 比 Jenga/VSA 好在哪里？ | 同 keep、同 sparse executor，对比 online router latency、VBench/FVD、route mass coverage |
| 上一 stage route 到后面会过期吗？ | 按 layer、head、timestep 报 Top-k overlap/coverage；adaptive refresh ablation |
| 去掉 VSA coarse branch 是否损害全局信息？ | online VSA、C2F route-only、C2F+coarse-summary、finetuned C2F 四组 |
| StepReuse 的收益是不是已有 caching 就能得到？ | C2F 单独、RAPID/TeaCache 类 baseline、C2F+reuse；分别报增量 |
| 三步 DMD 是否掩盖收益？ | 3/20/50 NFE 延迟—质量曲线 |
| 理论是否真的带来训练优势？ | 正确数据、等 update/seen samples/FLOPs 的收敛曲线和 gradient/subspace 统计 |
| 跨范式的 SparVAR 是否已经做了同一件事？ | related work 坦诚承认；强调 diffusion trajectory、noise-level route lifespan 与无当前-stage router，并用实验支持 |

## 9. 最终客观评分

| 方面 | 当前 | 训练和关键对照成功后 |
|---|---:|---:|
| broad multi-resolution idea | 3/10 | 3/10 |
| RDD 理论参数化 | 6/10 | 7/10（需真正的定理/机制验证） |
| 跨-stage attention routing | 7/10 | 8/10（需直对 Jenga/SparVAR-style prior） |
| 工程与 kernel 完成度 | 9/10 | 9/10 |
| 当前论文证据强度 | 5/10 | 8/10（取决于等预算质量结果） |
| 整体投稿潜力 | 6.5/10 | 8/10 左右 |

最诚实的总判断：**RDD 不是“没人想到过 coarse-to-fine”的基础概念突破，但它可以成为一篇有清晰技术身份的论文。** 身份不应是“又一个 pyramidal generator”，而应是：

> 一个受限但可解析的 residualized diffusion/FM path，以及由该 path 自然产生的、无需当前高分辨率 router 的 generative cross-stage sparse execution。

真正决定影响力的已经不是再写更多 novelty 形容词，而是三张硬表：

1. 正确数据上的收敛曲线与 `d=0/f-DM-style` 因果对照；
2. RDD-Wan、PyramidalWan、Jenga 的等预算 VBench/FVD—latency Pareto；
3. online VSA/Jenga router 与 G-CSAR 在同 sparse executor 下的质量、route coverage 和真实端到端耗时。
