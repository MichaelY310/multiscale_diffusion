# RDD 方法组合手册：从直觉、数学到实现

> 最后更新：2026-08-11  
> 目标：不读原论文，也能理解主要方法在做什么、为什么可能有效、怎样实现、怎样与 RDD 公平组合。  
> 公式说明：配套 HTML 中的公式在构建时由 KaTeX 预先渲染并内嵌字体，打开文件时不需要联网，也不依赖浏览器执行 JavaScript。

---

## 0. 先说结论：RDD 应该怎样吸收别的方法

RDD 的主方法不是“把所有 trick 堆起来”。RDD 的核心是：**生成路径本身随着时间改变空间分辨率，并显式加入分辨率变换引起的 drift**。其他方法能否与 RDD 正交，取决于它修改的是不是同一个东西。

可以把完整生成系统想成五层：

1. **路径层**：从噪声到数据究竟沿什么路径走。RDD、f-DM、Pyramidal Flow、SiT 都会碰这一层。
2. **训练分配层**：哪些时间更常被抽到、每个样本损失有多大权重。Min-SNR、P2、loss-second-moment 属于这一层。
3. **表示学习层**：模型内部特征应该学到什么。REPA、REG 属于这一层。
4. **采样层**：训练完以后怎样数值积分、怎样做 guidance。LIG、Autoguidance、DPM-Solver、AYS、Restart 属于这一层。
5. **表征与架构层**：latent 是否好学、网络数值尺度是否稳定。更强 VAE、EDM2 架构改造属于这一层。

最干净、最容易解释的一条组合路线是：

$$
\boxed{
\text{RDD path}
+\text{Min-SNR/P2}
+\text{REPA}
+\text{LIG}
+\text{better schedule/solver}
}
$$

这里每一项解决的问题不同：

- RDD：让高噪声阶段在便宜、低分辨率的空间中学习全局结构。
- Min-SNR/P2：减少不同噪声任务之间互相打架。
- REPA：让隐藏特征更快学到语义。
- LIG：只在 guidance 真正有帮助的区间放大条件信号。
- AYS/solver：用更少或更合理的函数调用走完已经学好的路径。

这条路线的最大优点不是“每项都可能降一点 FID”，而是论文中容易回答：**增益来自哪里，RDD 自己贡献了什么，组合方法又贡献了什么。**

---

## 1. 阅读这份手册前只需要知道的 RDD 数学

### 1.1 普通 diffusion 在做什么

把干净 latent 记为 $x_0$，高斯噪声记为 $\epsilon\sim\mathcal N(0,I)$。普通 diffusion 在时间 $t$ 构造：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon.
$$

直观上：

- $\alpha_t$ 决定还保留多少原图信号；
- $\sigma_t$ 决定加入多少噪声；
- $t=0$ 通常接近干净数据，$t=1$ 接近纯噪声；
- 模型从 $x_t$ 预测 $\epsilon$、$x_0$、velocity 或 score，再通过 DDPM/ODE solver 逐步返回 $x_0$。

信噪比为

$$
\operatorname{SNR}(t)=\frac{\alpha_t^2}{\sigma_t^2}.
$$

SNR 大表示输入比较干净，SNR 小表示输入几乎全是噪声。

### 1.2 RDD 多做了什么

设 $R_t$ 是随时间改变的分辨率算子。它可以是分段常数的 average pooling，也可以推广为更一般的 transformed signal。RDD 的扰动状态可写成：

$$
x_t=\alpha_t R_t(x_0)+\sigma_t\epsilon.
$$

如果 $R_t$ 不随时间变，这就是普通 diffusion。只要 $R_t$ 随时间变，对时间求导时就会多出一项：

$$
\frac{\mathrm d x_t}{\mathrm dt}
=
\dot\alpha_t R_t(x_0)
+\alpha_t\dot R_t(x_0)
+\dot\sigma_t\epsilon.
$$

中间的

$$
\alpha_t\dot R_t(x_0)
$$

就是最直观的 **resolution drift**：路径不仅在去噪，还在改变“信号应该长什么样、位于哪个分辨率空间”。在离散的 8×8、16×16、32×32 阶段中，$\dot R_t$ 不一定作为连续导数直接实现；它通常体现为阶段内模型目标和边界处的分辨率 transport/noise filling。

### 1.3 判断是否正交的简单问题

面对一个新方法，只问三句话：

1. 它是否改变 $R_t$ 或整条 $x_t$ 路径？如果是，与 RDD 高度重叠。
2. 它是否只改变损失权重、特征监督或采样算法？如果是，通常与 RDD 正交。
3. 它能否直接加载同一个 RDD checkpoint？能直接加载的是采样级组合；需要重训但不改路径的是训练级正交组合；连目标和路径都改了的是新模型族。

---

## 2. 一张图看懂所有方法放在哪里

~~~text
干净图像
  │
  ├─ VAE / latent 表征 ────────────── 更强 VAE、RAE、DC-AE
  │
  ▼
RDD 路径 x_t = α_t R_t(x_0) + σ_t ε ─ f-DM / SiT / Pyramidal Flow 会改这一层
  │
  ├─ 抽哪个 t、给多大权重 ───────── Min-SNR、P2、loss-second-moment
  │
  ├─ 隐藏特征学什么 ─────────────── REPA、REG
  │
  ▼
训练后的 RDD checkpoint
  │
  ├─ 条件怎样放大 ───────────────── LIG、scheduled CFG、Autoguidance
  │
  ├─ 时间点怎样选择 ─────────────── AYS
  │
  └─ 数值上怎样走完路径 ─────────── DDPM、DPM-Solver、Restart
  ▼
生成样本
~~~

### 2.1 分类总表

| 方法 | 它主要改什么 | 旧 RDD checkpoint 可用 | 与 RDD 的关系 | 第一优先级 |
|---|---|---:|---|---:|
| Limited-Interval Guidance | CFG 生效区间 | 是 | 强正交 | 高 |
| Scheduled CFG | CFG 随时间变化 | 是 | 强正交 | 高 |
| Autoguidance | 用弱模型作为负方向 | 是，但要第二个模型 | 强正交 | 中高 |
| AYS | 采样时间节点 | 是 | 强正交 | 中高 |
| DPM/ODE solver | 数值积分器 | 是 | 正交，但必须分 stage | 高 |
| Restart sampling | 局部加噪再去噪 | 是 | 正交，但必须分 stage | 中 |
| Min-SNR | diffusion 损失权重 | 否，需重训 | 强正交 | 高 |
| P2 weighting | diffusion 损失权重 | 否，需重训 | 强正交 | 中 |
| Loss-second-moment | timestep 抽样概率 | 否，需重训 | 强正交 | 中高 |
| FM logit-normal sampling | FM 时间抽样概率 | 否，需重训 | 强正交 | 高 |
| REPA | 中间特征监督 | 否，需重训 | 强正交 | 高 |
| Post-hoc EMA | 权重平滑配置 | 取决于保存方式 | 正交 | 中 |
| EDM2 magnitude preservation | 网络参数化与数值尺度 | 否，需重训 | 可组合但改动大 | 中 |
| 更强 tokenizer | latent 表征 | 否，通常重训 | 系统级正交 | 中 |
| SiT / stochastic interpolant | 整条概率路径 | 否 | 可做 RDD-SiT，但不是小开关 | 研究线 |
| REG | 联合语义变量与生成状态 | 否 | 可组合但不是简单正则 | 研究线 |
| f-DM | transformed-signal 路径 | 否 | 与 RDD 高度重叠 | 竞争/理论定位 |
| Pyramidal Flow/Wan | 分阶段多分辨率路径 | 否 | 高度重叠 | 竞争基线 |

---

# 第一部分：不需要重训的采样方法

## 3. Limited-Interval Guidance：不要从头到尾都用大 CFG

### 3.1 一句话直觉

CFG 像一个“把生成结果往类别方向推”的力。刚开始图像还是噪声时，模型根本没有可靠物体可推；最后细节已经确定时，再强推容易损坏纹理。**最有价值的是中间阶段：大体结构已出现，但语义仍可调整。**

### 3.2 普通 CFG 的公式

模型分别做一次无条件和有条件预测：

$$
\hat\epsilon_u=\epsilon_\theta(x_t,t,\varnothing),\qquad
\hat\epsilon_c=\epsilon_\theta(x_t,t,y).
$$

然后组合为

$$
\hat\epsilon_{\mathrm{cfg}}
=
\hat\epsilon_u+w\left(\hat\epsilon_c-\hat\epsilon_u\right).
$$

每一项的意思：

- $\hat\epsilon_u$：模型认为“不指定类别时”应该怎样去噪；
- $\hat\epsilon_c-\hat\epsilon_u$：类别 $y$ 带来的方向；
- $w$：把类别方向放大多少；
- $w=1$ 恰好得到条件预测；$w>1$ 是外推，不再是模型原本学到的条件分布。

Limited-Interval Guidance 引入开关 $g(t)$：

$$
\hat\epsilon
=
\hat\epsilon_u+
\left[1+(w-1)g(t)\right]
\left(\hat\epsilon_c-\hat\epsilon_u\right),
$$

其中

$$
g(t)=
\begin{cases}
1,&t_{\mathrm{low}}\le t\le t_{\mathrm{high}},\\
0,&\text{otherwise}.
\end{cases}
$$

注意：这里区间外不是“完全无条件”，而是 $w=1$ 的普通条件预测。

### 3.3 为什么它与 RDD 特别适合组合

普通单分辨率模型只有“高噪声、中噪声、低噪声”。RDD 还多了清楚的阶段语义：

- 8×8：主要决定布局和类别轮廓；
- 16×16：补充部件与中频结构；
- 32×32：完成纹理和局部细节。

因此 RDD 不应只搜索一个全局区间。更合理的是把 guidance 区间写成“stage × local time”的配置。例如：

$$
w_k(\tau)=
\begin{cases}
1,&\tau\notin[a_k,b_k],\\
w_k,&\tau\in[a_k,b_k],
\end{cases}
$$

这里 $k$ 是 8/16/32 阶段，$\tau\in[0,1]$ 是该阶段内部的局部时间。

### 3.4 实现步骤

1. 同一个 batch 拼接 conditional/unconditional 输入，只做一次 batched forward。
2. 将输出拆成 $\hat\epsilon_c,\hat\epsilon_u$。
3. 根据当前全局时间找到 RDD stage 和局部时间 $\tau$。
4. 计算该 stage 的 $w_k(\tau)$。
5. 只对预测目标通道做 CFG；旧 DiT 复现通常只 guide 前 3 个 epsilon 通道，variance 通道不要随意外推。
6. 在阶段边界两侧单独检查，避免浮点误差让同一步被算到两个区间。

伪代码：

~~~python
stage, tau = rdd_stage_and_local_time(t)
scale = cfg_schedule[stage](tau)
eps = eps_uncond + scale * (eps_cond - eps_uncond)
~~~

### 3.5 最小实验

固定 checkpoint、随机种子、DDPM 250 steps 和 80/80/90 分配，仅比较：

- CFG 1.0；
- constant CFG 1.5；
- 全局 limited interval CFG 1.5；
- stage-aware limited interval CFG 1.5。

同时报告 FID、sFID、precision、recall 和 wall-clock。这样可以区分“质量提升”与“只是 precision/recall 交换”。

### 3.6 常见错误

- 把区间外设成 $w=0$，导致变成无条件采样；
- diffusion 的 $t=1$ 是高噪声，却把 schedule 方向写反；
- 在 variance 通道也直接做同样外推；
- RDD 跨阶段使用同一个局部区间，却不说明它对应的物理噪声不同。

论文：[Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models](https://arxiv.org/abs/2404.07724)

---

## 4. Scheduled CFG：不是开或关，而是平滑改变 guidance

### 4.1 一句话直觉

Limited interval 是矩形开关；Scheduled CFG 是调光旋钮。开关简单、容易复现；平滑曲线更符合“模型置信度逐渐变化”的事实。

把固定 $w$ 改为 $w(t)$：

$$
\hat\epsilon
=
\hat\epsilon_u+w(t)(\hat\epsilon_c-\hat\epsilon_u).
$$

一个 stage 内的线性 schedule 可以写成

$$
w_k(\tau)
=
w_{\min}+
\left(w_{\max}-w_{\min}\right)r_k(\tau),
$$

其中 $r_k(\tau)\in[0,1]$。余弦版本更平滑：

$$
r_k(\tau)
=
\frac{1-\cos(\pi u_k(\tau))}{2}.
$$

$u_k$ 是把有效区间归一化到 $[0,1]$ 的局部位置。采样从高噪声走向低噪声，所以代码中“时间递减”和公式中“曲线递增”很容易写反；应直接打印几个关键时间点的 scale 验证。

### 4.2 怎样与 RDD 组合

推荐先限制自由度：

- 8×8：从 $1.0$ 缓慢升到 $w_{\max}$；
- 16×16：保持或轻微下降；
- 32×32 末段：回到 $1.0$。

不要一开始为三个阶段各搜索四五个超参数。先比较 constant、rectangular LIG、一个对称 cosine schedule。只有确定 schedule 有稳定收益，再做 stage-aware 搜索。

### 4.3 潜在优势

- 减少高噪声时无意义的语义外推；
- 减少低噪声时饱和、过锐和多样性下降；
- 不需重训，可直接测试所有旧 RDD checkpoint；
- 能帮助判断 RDD 的某个 stage 是否对条件信号过度或不足敏感。

---

## 5. Autoguidance：用“较弱的同类模型”定义坏方向

### 5.1 一句话直觉

CFG 的负方向是无条件模型。Autoguidance 的负方向换成一个“会生成同类图像、但质量较差”的模型。好比你不仅知道正确答案，还知道一个常见的错误答案，于是可以明确地从常见错误方向远离。

设好模型输出为 $f_{\mathrm{good}}$，弱模型输出为 $f_{\mathrm{weak}}$：

$$
f_{\mathrm{AG}}
=
f_{\mathrm{weak}}
+s\left(f_{\mathrm{good}}-f_{\mathrm{weak}}\right).
$$

当 $s=1$ 时就是好模型；当 $s>1$ 时，从弱模型穿过好模型继续外推。

它与 CFG 的几何形式相似，但含义不同：

- CFG：远离无条件分布，增强条件一致性；
- Autoguidance：远离同一任务中的低质量生成模式，提升样本质量。

### 5.2 弱模型应该怎样选

对于 RDD，弱模型必须尽量只在“能力”上弱，而不是在“任务定义”上不同。优先顺序：

1. 同一次正确 ImageNet 原生数据训练的早期 checkpoint；
2. 同架构、同 RDD stage 划分、同 prediction target 的小模型；
3. 同设置但训练步数更少的模型。

不能使用：

- 旧的错误 Hugging Face 数据集模型；
- 不同 class mapping 的模型；
- 不同 VAE、不同 latent scaling 的模型；
- resolution schedule 不同且无法一一对齐的模型。

否则 $f_{\mathrm{good}}-f_{\mathrm{weak}}$ 混入的是数据或坐标系差异，不再是“质量方向”。

### 5.3 与 RDD 组合时的关键实现

1. 两个模型必须接收完全相同的 $x_t,t,y$。
2. 在同一个 RDD stage 上计算两者输出。
3. prediction parameterization 必须一致；不能直接拿 epsilon 模型减 velocity 模型。
4. 若模型学习 variance，通常从好模型保留 variance，只对 latent prediction 做 guidance。
5. 两模型 forward 会接近翻倍 NFE 成本，论文必须报告真实耗时。

~~~python
good = good_model(x_t, t, y)
weak = weak_model(x_t, t, y)
pred = weak.latent + scale * (good.latent - weak.latent)
variance = good.variance
~~~

### 5.4 最有解释力的 RDD 实验

用 400K 作为 good，100K/200K/300K 作为 weak，固定采样设置搜索少量 $s$。如果 200K weak 最好而 100K 太弱，说明“负模型必须与好模型共享足够的分布结构”；这比只报最优 FID 更有研究价值。

### 5.5 潜在优势与代价

优势：零训练、可能显著提升 FID、与 RDD path 正交。  
代价：两次模型前向、依赖合适的弱模型、scale 过大会降低 recall 或产生过锐伪影。

论文入口：[Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/abs/2406.02507)

---

## 5A. Classifier Guidance：直接沿着“更像这个类别”的梯度走

### 5A.1 一句话直觉

CFG 让同一个生成模型比较“有条件预测”和“无条件预测”。Classifier guidance 另外训练一个能识别 noisy image 的分类器，然后在采样时问它：**把当前 latent 往哪个方向轻推，会让目标类别概率上升？**

分类器给出

$$
\nabla_{x_t}\log p_\phi(y\mid x_t,t).
$$

这是对当前 noisy latent 的梯度。梯度指向能最快提高类别 $y$ 对数概率的方向。对于 score model：

$$
s_{\mathrm{guided}}(x_t,t,y)
=
s_\theta(x_t,t)
+w\,\nabla_{x_t}\log p_\phi(y\mid x_t,t).
$$

若模型预测 epsilon，并使用

$$
s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sigma_t},
$$

则常见转换为

$$
\epsilon_{\mathrm{guided}}
=
\epsilon_\theta
-w\sigma_t\nabla_{x_t}\log p_\phi(y\mid x_t,t).
$$

负号来自 score 与 epsilon 的关系。这里最常见的 bug 就是漏掉 $\sigma_t$ 或符号写反。

### 5A.2 为什么它在 RDD 中不只是“接一个 ImageNet classifier”

普通干净 256×256 分类器没有见过 noisy latent，更没见过 8×8、16×16、32×32 三种 RDD 状态。直接使用时梯度很可能没有意义。合格的 RDD classifier 需要：

- 接受与生成模型完全相同的 noisy latent；
- 输入 time embedding；
- 能处理三个 token grid，或三个 stage 分别训练；
- 使用正确的 ImageNet 原生 class mapping；
- 在每个噪声区间都有可测的分类准确率和梯度范数。

### 5A.3 怎样实现和验证

~~~python
x_t = x_t.detach().requires_grad_(True)
log_prob = noisy_classifier(x_t, t, stage).log_softmax(-1)
grad = autograd.grad(log_prob[:, target].sum(), x_t)[0]
eps = eps_uncond - guidance_scale * sigma_t * grad
~~~

先做三个 sanity check：

1. 沿 $+\nabla\log p(y\mid x_t,t)$ 走一小步后，分类器的目标 log-prob 确实上升；
2. 每个 stage 的梯度范数处于合理量级，不在边界突然爆炸；
3. $w=0$ 与原 sampler bitwise 或数值近似一致。

### 5A.4 与 RDD 的关系和优先级

路径层面它与 RDD 正交，但需要额外训练多分辨率 noisy classifier，采样还增加 backward 成本。RDD 已有 class conditioning 和 CFG，因此 classifier guidance 的性价比低于 LIG、REPA 和 Min-SNR。它更适合作为“指导机制的完整性实验”，而不是主力降 FID 方法。

论文：[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

---

## 6. DPM-Solver 与 ODE solver：用更聪明的步法走同一条路

### 6.1 一句话直觉

训练好的模型给出“当前位置应该往哪里走”。DDPM 像每隔很短距离走一步并加入随机性；高阶 ODE solver 会利用局部曲线信息，一大步估计更远的位置。它不改变目的地或训练目标，只改变怎样数值地走完路径。

### 6.2 最少数学背景

很多 diffusion sampler 可以写成 ODE：

$$
\frac{\mathrm d x}{\mathrm dt}=F_\theta(x,t).
$$

最简单 Euler 步为

$$
x_{i+1}=x_i+\Delta t_i F_\theta(x_i,t_i).
$$

误差来自把弯曲轨迹当成直线。二阶方法会在中间再看一次方向：

$$
\begin{aligned}
k_1&=F_\theta(x_i,t_i),\\
k_2&=F_\theta(x_i+\tfrac12\Delta t_i k_1,t_i+\tfrac12\Delta t_i),\\
x_{i+1}&=x_i+\Delta t_i k_2.
\end{aligned}
$$

DPM-Solver 进一步利用 diffusion ODE 中已知的线性噪声结构，把容易解析的部分精确积分，只让神经网络近似剩余部分。因此同样 NFE 下通常比朴素 Euler 更准。

### 6.3 为什么 RDD 不能直接把全程交给一个 solver

RDD 在 stage 边界改变 token grid 和状态维度。普通 solver 默认 $x(t)$ 始终位于同一个向量空间，并且 $F_\theta$ 在积分区间内足够平滑。RDD 的 8→16、16→32 边界违反了这个假设。

正确方式是：

$$
\text{solve stage 8}
\rightarrow
\text{RDD boundary transport}
\rightarrow
\text{solve stage 16}
\rightarrow
\text{boundary transport}
\rightarrow
\text{solve stage 32}.
$$

每个 stage 内 solver 可以是一阶、二阶或三阶；**绝不能用一次多步更新跨过边界**。多步 solver 还会缓存过去的模型输出，进入新 stage 时必须清空历史。

### 6.4 公平比较

必须同时报告：

- 名义 steps；
- 真实 NFE；
- 两次边界 transport 是否额外调用模型；
- stochastic noise filling 的次数；
- wall-clock 和吞吐量。

例如“DPM 90 steps”如果二阶方法每步两次 forward，可能比“DDPM 250 steps”的成本差距小得多。论文中只写 steps 会误导。

### 6.5 与 RDD 组合的潜在优势

- 相同 NFE 下更低离散误差；
- 在 16×16/32×32 的光滑区段节省大量步数；
- 可以验证 RDD 的收益究竟来自训练路径还是仅来自长采样。

但你现有 400K diffusion 结果显示 DDPM 可能优于某些 DPM 配置，这并不反常：模型未充分收敛、solver 时间变量或 prediction conversion 不匹配、边界缓存未清空，都可能让高阶求解器更差。

论文：[DPM-Solver](https://arxiv.org/abs/2206.00927)；[DPM-Solver++](https://arxiv.org/abs/2211.01095)

---

## 7. Align Your Steps：优化“在哪些时间看模型”

### 7.1 一句话直觉

solver 决定每一步怎么算；AYS 决定每一步放在哪里。山路上十个等距离路标不一定合理，弯道密集处应该多放几个，直路少放几个。

给定 $N+1$ 个时间节点

$$
1=t_0>t_1>\cdots>t_N=0,
$$

普通 schedule 用均匀时间、均匀 log-SNR 或手工规则。AYS 针对特定模型、数据和 solver 优化这些 $t_i$，目标是减少离散采样分布与连续目标分布之间的误差。

### 7.2 在 RDD 中为什么必须 stage-aware

RDD 的全程并非一个光滑区间。每个 stage 都有独立的难度和边界：

$$
\mathcal T=
\mathcal T_{8}\cup\mathcal T_{16}\cup\mathcal T_{32}.
$$

合理做法：

1. 固定每个 stage 的 NFE 预算 $N_8,N_{16},N_{32}$；
2. 分别优化三个 stage 内部的时间节点；
3. 强制保留边界端点；
4. 优化时使用与最终评估相同的 solver；
5. 最后再小范围搜索 $N_8:N_{16}:N_{32}$。

不能把单分辨率 ImageNet 的 AYS 节点直接线性映射到 RDD 全局时间。那样没有考虑边界不连续，也没有考虑三个分辨率的误差尺度不同。

### 7.3 实现思路

最实用的第一版不需要复现完整理论优化器：

- 收集一小批固定噪声和标签；
- 以高步数 DDPM/ODE 结果作为 teacher trajectory；
- 为每个 stage 搜索少量候选节点；
- 最小化 student sampler 与 teacher 在边界/最终状态的误差；
- 用独立 seed 做 FID 验证，避免对小批样本过拟合。

### 7.4 潜在优势

AYS 尤其适合回答 RDD 的一个关键问题：**80/80/90 是合理的计算分配，还是历史上随手选的？** 如果优化后更多 NFE 自动落到 8×8，支持“全局结构阶段更难”；若更多落到 32×32，则说明当前高频恢复仍是瓶颈。

论文：[Align Your Steps: Optimizing Sampling Schedules in Diffusion Models](https://arxiv.org/abs/2404.14507)

---

## 8. Restart Sampling：走偏后局部“重来一次”

### 8.1 一句话直觉

确定性 ODE 采样的误差会逐步累积。Restart 在某个中间点主动加一点噪声，再重新去噪，相当于把已经僵硬、走偏的轨迹重新放回模型熟悉的噪声分布附近。

设当前在较干净时刻 $t_a$ 得到 $x_{t_a}$，先向更噪的 $t_b>t_a$ 做一次 forward jump：

$$
x_{t_b}
=
\frac{\alpha_{t_b}}{\alpha_{t_a}}x_{t_a}
+
\sqrt{
\sigma_{t_b}^2
-\frac{\alpha_{t_b}^2}{\alpha_{t_a}^2}\sigma_{t_a}^2
}\,z,
\qquad z\sim\mathcal N(0,I),
$$

然后再次从 $t_b$ 去噪到 $t_a$。直观上，新噪声给 sampler 机会跳出之前的数值误差。

### 8.2 RDD 中的硬约束

Restart 区间必须完全位于同一个 stage。原因不是工程洁癖，而是上式默认 $x_{t_a}$ 和 $x_{t_b}$ 位于同一个维度、共享同一信号定义。跨 8×8→16×16 restart 会同时混入 resolution transport，公式不再成立。

### 8.3 最小实现

1. 先只在 32×32 stage 放一个 restart，因为最终 FID/sFID 对细节误差敏感；
2. 固定总 NFE，将普通额外 steps 与 restart steps 公平比较；
3. 检查 stochasticity 是否只是提高 recall、牺牲 precision；
4. 若有收益，再试 16×16；8×8 restart 可能扰乱已经形成的全局布局。

### 8.4 潜在优势和风险

优势：不重训、可修正 ODE 累积误差、可能改善多样性。  
风险：计算增加；加噪过强会破坏类别结构；跨 stage 的实现从数学上不合法。

论文：[Restart Sampling for Improving Generative Processes](https://arxiv.org/abs/2306.14878)

---

# 第二部分：需要重训，但不改变 RDD 路径的方法

## 9. Min-SNR：别让某些时间任务霸占优化器

### 9.1 一句话直觉

训练 diffusion 不是一个任务，而是上千个噪声强度不同的任务共用一个模型。高噪声样本在学“这团噪声大概是什么类别”，低噪声样本在学“这一小块纹理应该怎么修”。它们给参数的梯度可能互相冲突。Min-SNR 通过限制极端任务的权重，让优化更均衡。

### 9.2 epsilon prediction 下的公式

对 timestep $t$ 的基础损失：

$$
\mathcal L_t
=
\left\|
\epsilon_\theta(x_t,t,y)-\epsilon
\right\|_2^2.
$$

Min-SNR 权重常写为

$$
w_{\mathrm{MinSNR}}(t)
=
\frac{\min\left(\operatorname{SNR}(t),\gamma\right)}
{\operatorname{SNR}(t)}.
$$

最终目标：

$$
\mathcal L
=
\mathbb E_t\left[
w_{\mathrm{MinSNR}}(t)\mathcal L_t
\right].
$$

当 SNR 很大时，权重约为 $\gamma/\operatorname{SNR}(t)$，防止非常干净的任务支配训练；当 SNR 不大时，权重为 1。

### 9.3 为什么不能盲抄公式

Min-SNR 权重与 prediction target 有关：

- epsilon prediction；
- $x_0$ prediction；
- velocity prediction；
- flow velocity；

这些目标的自然尺度不同。把 epsilon 版本原封不动乘到 velocity/FM loss 上，不叫公平复现。当前仓库实现应在非 epsilon prediction 时主动拒绝或走专门公式，而不是悄悄使用错误权重。

### 9.4 RDD 中怎样算 SNR

RDD 最安全的定义是使用**当前 stage 的局部噪声日程**：

$$
\operatorname{SNR}_k(\tau)
=
\frac{\alpha_k(\tau)^2}{\sigma_k(\tau)^2}.
$$

原因是 8×8、16×16、32×32 的同一全局 index 不一定代表相同物理难度。建议先共享一个 $\gamma$，不要同时为三个 stage 搜索三个 gamma，否则消融成本和叙事复杂度都会爆炸。

### 9.5 实现步骤

~~~python
per_sample_mse = mean_over_nonbatch_dims((pred_eps - true_eps) ** 2)
snr = alpha_bar / (1.0 - alpha_bar)
weight = minimum(snr, gamma) / snr
loss = mean(weight * per_sample_mse)
~~~

若模型同时学习 variance，通常只重加权 simple MSE，VB/variance 项保持原有权重。否则会改变 learned variance 的标定。

### 9.6 与 RDD 组合的潜在优势

- 更快 convergence；
- 减少 32×32 低噪声细节任务对共享参数的支配；
- 可能让 8×8 的全局结构得到更稳定梯度；
- 与 RDD path 完全不冲突。

最小消融：baseline、$\gamma=5$、$\gamma=10$，先在较短训练预算比较 FID 曲线，不要只看最终点。

论文：[Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)

---

## 10. P2 Weighting：把注意力放在“人眼真正关心”的难度上

### 10.1 一句话直觉

极高噪声时只剩粗语义；极低噪声时模型可能忙于逐像素修补。感知质量最关键的往往是中间区域：物体结构、形状和可辨认部件正在形成。P2 用一个平滑权重减弱高 SNR 区域。

典型权重：

$$
w_{\mathrm{P2}}(t)
=
\left(k+\operatorname{SNR}(t)\right)^{-\gamma}.
$$

其中：

- $k>0$ 防止 SNR 接近 0 时权重爆炸；
- $\gamma$ 控制抑制高 SNR 的强度；
- $\gamma=0$ 退化为不加权。

### 10.2 与 Min-SNR 的区别

Min-SNR 像“硬限幅器”：超过阈值后按比例压低。  
P2 像“平滑均衡器”：整个 SNR 轴都在连续改变权重。

它们解决高度相似的问题，因此第一轮实验**不要同时开启**。先做：

$$
\text{RDD baseline}
\quad\text{vs}\quad
\text{RDD+Min-SNR}
\quad\text{vs}\quad
\text{RDD+P2}.
$$

等确认哪一种更好，再考虑组合；同时开启后很难判断有效原因，而且有效权重可能过度压低某一段。

### 10.3 RDD 实现与潜在优势

像 Min-SNR 一样按 stage-local SNR 计算。P2 的平滑性可能更适合 RDD 边界附近，因为不会突然发生大权重变化；但它也可能让最细 stage 的低噪声训练不足。必须同时观察 FID 和 sFID，并看重建/样本细节。

论文：[Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227)

---

## 11. Loss-Second-Moment Sampling：把训练 batch 花在“最不稳定”的时间

### 11.1 一句话直觉

Min-SNR/P2 决定抽到某个时间后给它多大权重。Loss-aware sampler 决定**它多常被抽到**。如果某一段损失长期很大且波动强，均匀抽样会浪费 batch 在模型已经学会的时间上。

### 11.2 怎样保持无偏

原始目标是均匀时间平均：

$$
\mathcal L
=
\frac1T\sum_{t=1}^{T}\mathbb E[\mathcal L_t].
$$

改为按 $p(t)$ 抽样后，要乘 importance weight：

$$
\mathcal L
=
\mathbb E_{t\sim p(t)}
\left[
\frac{\mathcal L_t}{T\,p(t)}
\right].
$$

因为

$$
\sum_t p(t)\frac{\mathcal L_t}{T p(t)}
=
\frac1T\sum_t\mathcal L_t.
$$

所以改变的是梯度估计的方差，不是理想目标本身。

Loss-second-moment 用最近历史估计：

$$
p(t)\propto
\sqrt{\mathbb E[\mathcal L_t^2]}.
$$

平方损失大，说明该 timestep 既难又可能不稳定，于是更常抽到。

### 11.3 RDD 的正确做法

每个 stage 维护独立统计：

$$
p(k,\tau)=p(k)\,p(\tau\mid k).
$$

第一版建议固定 $p(k)$ 为原本的 300/300/400 训练预算，只在 stage 内学习 $p(\tau\mid k)$。这样不会无意中把“时间采样优化”变成“重新分配三个分辨率的训练量”。

实现要点：

1. warm-up 时均匀抽样，历史不足不能相信 second moment；
2. 分布加一个小 uniform mixture，保证任何时间都有非零概率；
3. DDP 下同步每个 timestep 的损失统计；
4. importance weight 乘到完整目标还是仅 MSE，必须与设计一致；
5. 记录实际 histogram，否则无法知道训练究竟偏向哪里。

### 11.4 与 RDD 的研究价值

这不仅可能提高 FID，还能直接回答当前的核心疑问：**8×8 是否训练不足？** 如果 sampler 自动把更多概率放到 8×8 某些局部时间，说明该段确实产生大而不稳定的损失；如果没有，100/300/600 的问题可能不是简单的 8×8 step 数不足。

论文背景：[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

---

## 12. FM 的 Logit-Normal Time Sampling：别假设每个 flow 时间一样难

### 12.1 FM 背景

最简单的直线路径：

$$
x_\tau=(1-\tau)x_{\mathrm{noise}}+\tau x_{\mathrm{data}},
\qquad \tau\in[0,1].
$$

对时间求导：

$$
u^\star_\tau
=
\frac{\mathrm d x_\tau}{\mathrm d\tau}
=
x_{\mathrm{data}}-x_{\mathrm{noise}}.
$$

模型学习 velocity：

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{\tau}
\left[
\left\|
v_\theta(x_\tau,\tau)-u^\star_\tau
\right\|_2^2
\right].
$$

均匀采样 $\tau$ 很简单，但不保证训练难度均匀。Logit-normal 先抽高斯：

$$
z\sim\mathcal N(\mu,s^2),
$$

再映射：

$$
\tau=\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}}.
$$

这样样本自然集中在中间或由 $\mu,s$ 控制的区域，极端端点较少。

### 12.2 RDD-FM 中最容易犯的方向错误

diffusion 通常按 $t=1\rightarrow0$ 从噪声走向数据；上面的 FM 约定按 $\tau=0\rightarrow1$ 从噪声走向数据。两者是反向坐标。

如果 diffusion 的 finest/middle/coarse 训练预算写作 300/300/400，含义是 coarse 8×8 最多，那么严格镜像到 FM 的 noise→data 顺序应是：

$$
\boxed{\text{coarse/middle/fine}=400/300/300}
$$

对应 FM 边界：

$$
\boxed{[0,\ 0.4,\ 0.7,\ 1]}.
$$

旧 FM-v2 的 coarse/middle/fine=300/300/400 实际让 32×32 最多，不是 diffusion 300/300/400 的公平镜像。这个错误与 logit-normal 是两件事：先修正 stage 边界，再比较时间采样。

### 12.3 怎样与 RDD 组合

在每个 stage 内抽局部变量 $\xi\in(0,1)$：

$$
\xi=\operatorname{sigmoid}(z),
\qquad
\tau=b_k+\xi(b_{k+1}-b_k).
$$

$[b_k,b_{k+1}]$ 是该 stage 的全局 FM 区间。这样 stage 总预算由区间长度决定，logit-normal 只重排 stage 内部时间。

不要直接在全局 $[0,1]$ 抽 logit-normal 再看落到哪个 stage。那会同时改变 stage 采样比例，导致与 diffusion 不公平。

### 12.4 潜在优势

- 避免大量 batch 落到过于容易或梯度信息少的端点；
- 对 flow/rectified-flow 训练通常比机械 uniform 更合理；
- 保持 RDD stage allocation 不变，因而能做干净消融；
- 与 FM velocity path 不冲突。

最小实验：正确 400/300/300 边界下，uniform 与一个固定 logit-normal 配置；不要一开始网格搜索很多 $\mu,s$。

相关背景：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)；[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)

---

## 13. REPA：让生成模型借用成熟视觉模型的“看图方式”

### 13.1 一句话直觉

生成模型训练早期既要学会识别“这是狗还是车”，又要学会怎样去噪。DINOv2 之类的视觉 encoder 已经很会提取语义。REPA 暂时请这个老师告诉 DiT：**你的中间特征至少应该和干净图像的语义表示对齐。**

老师只在训练时使用，采样时完全移除，所以不会增加生成 NFE。

### 13.2 数学

设：

- $h_\theta(x_t,t)$：RDD/DiT 某一层的 noisy hidden tokens；
- $z_\phi(x_0)$：冻结 teacher 从干净图像提取的 tokens；
- $P_\psi$：把学生 hidden dimension 投影到 teacher dimension 的小 MLP；
- $\operatorname{pool}_k$：把 teacher tokens 变为当前 RDD stage 的 token grid。

对齐损失：

$$
\mathcal L_{\mathrm{REPA}}^{(k)}
=
-
\frac1N\sum_{i=1}^{N}
\frac{
P_\psi(h_i)^\top z_i^{(k)}
}{
\|P_\psi(h_i)\|_2\,
\|z_i^{(k)}\|_2
}.
$$

这是负 cosine similarity。两个向量方向越一致，cosine 越接近 1，负值越小。

总目标：

$$
\mathcal L
=
\mathcal L_{\mathrm{RDD}}
+\lambda_{\mathrm{REPA}}
\mathcal L_{\mathrm{REPA}}.
$$

### 13.3 为什么 RDD 不能直接照搬单分辨率 REPA

teacher 看到的是干净的完整图像，而学生在不同时间看到 8×8、16×16、32×32 token。若直接把 teacher 的 16×16 patch tokens 和学生 8×8 tokens 一一比较，空间位置和语义粒度都不一致。

正确策略：

$$
z_\phi^{(k)}
=
\operatorname{AdaptivePool}_{H_k\times W_k}
\left(z_\phi(x_0)\right).
$$

然后再与当前 stage 的 $H_kW_k$ 个学生 tokens 对齐。

### 13.4 三种消融

1. **finest-only**：只在 32×32 阶段做 REPA。最稳、最接近原论文。
2. **all-stage/fixed-total**：三个阶段都对齐，但总 $\lambda$ 不变，每阶段约 $\lambda/3$。这是最公平的主实验。
3. **all-stage/full-each**：每个阶段都用完整 $\lambda$。总辅助损失变成三倍，只能作为强监督探索，不能与前两者直接归因。

### 13.5 实现步骤

~~~python
with no_grad():
    teacher_tokens = dino(clean_image)

student_tokens = model.return_intermediate(noisy_latent, t)
teacher_tokens = adaptive_pool_to_rdd_grid(teacher_tokens, stage)
student_tokens = projection_head(student_tokens)
repa_loss = -cosine_similarity(student_tokens, teacher_tokens).mean()
loss = rdd_loss + repa_lambda * repa_loss
~~~

工程注意：

- teacher 输入应是解码后的干净 RGB，使用 teacher 要求的 normalization；
- teacher 必须冻结并处于 eval mode；
- DDP 中 projection head 要正常同步；
- checkpoint loader 对缺失 REPA head 要允许非严格加载，保证旧模型仍能 sampling；
- 训练日志分别记录 base loss 和 REPA loss，避免辅助项掩盖主目标发散。

### 13.6 与 RDD 的潜在优势

RDD 声称低分辨率高噪声阶段更高效地学习全局语义；REPA 恰好直接监督语义表示。两者组合可能不是简单相加，而是让 8×8 stage 更快形成可用的类别结构，从而把后续 stage 的容量留给细节。

最有说服力的不只是最终 FID，而是固定 100K/200K/400K 训练预算的 convergence 曲线。若 RDD+REPA 在早期明显领先，能直接支持“训练效率”主张。

论文：[Representation Alignment for Generation](https://arxiv.org/abs/2410.06940)

---

## 14. Post-hoc EMA：训练后再选择“平滑了多久”

### 14.1 一句话直觉

训练权重每一步都在抖动。EMA 像拍长曝光照片，把近期多个 checkpoint 平均起来。短曝光跟得紧但噪声大；长曝光稳定但可能落后。普通训练提前选一个 EMA decay，选错后无法补救；post-hoc EMA 尽量把选择推迟到训练后。

普通 EMA：

$$
\theta_{\mathrm{EMA}}^{(n)}
=
\beta\theta_{\mathrm{EMA}}^{(n-1)}
+(1-\beta)\theta^{(n)}.
$$

$\beta$ 越接近 1，平均窗口越长。

Post-hoc 思路是训练时保存少量不同平滑 profile 的快照，训练后用线性组合近似目标 profile：

$$
\theta_{\mathrm{target}}
\approx
\sum_{j=1}^{m}c_j\theta_{\mathrm{EMA},j}.
$$

系数 $c_j$ 不是随便平均，而是根据各 profile 的核形状求解。

### 14.2 与 RDD 的组合

它完全不改变 $R_t$、loss 或 sampler，因此概念上正交。尤其适合 RDD，因为不同 stage 的学习速度可能不同：短 EMA 可能保留最新的 32×32 改进，长 EMA 可能保留更稳定的 8×8 结构。

### 14.3 实现和边界

- 如果旧训练只保存一个 EMA 和 raw weights，不能凭空重建任意 post-hoc profile；
- 新训练要按论文要求保存多个 profile/快照；
- 所有候选 EMA 必须用同一采样配置比较；
- 不要把“挑最优 EMA”与“RDD 方法增益”混为一谈，baseline 也要获得相同选择机会。

论文：[Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)

---

# 第三部分：可以组合，但已经不是小型插件

## 15. SiT / Stochastic Interpolants：把 RDD 放进更一般的路径语言

### 15.1 一句话直觉

普通 diffusion 规定一条特定的加噪路径；flow matching 常用直线；stochastic interpolant 允许你自己设计“信号系数”和“噪声系数”怎样变化。它像一套描述生成路径的通用语法。

一般路径：

$$
x_t=\alpha(t)x_{\mathrm{data}}+\sigma(t)\epsilon.
$$

速度为

$$
\dot x_t
=
\dot\alpha(t)x_{\mathrm{data}}
+\dot\sigma(t)\epsilon.
$$

RDD-SiT 可以写成：

$$
x_t
=
\alpha(t)R_t(x_{\mathrm{data}})
+\sigma(t)\epsilon,
$$

于是

$$
\dot x_t
=
\dot\alpha(t)R_t(x_{\mathrm{data}})
+\alpha(t)\dot R_t(x_{\mathrm{data}})
+\dot\sigma(t)\epsilon.
$$

这正好显式出现 RDD drift。因此 SiT 不是与 RDD 冲突，而是可作为 RDD 理论的更一般承载框架。

### 15.2 为什么它不是“在旧 checkpoint 上开 SiT”

路径、velocity/score 转换、时间分布、loss weighting 和 sampler 都可能改变。旧 DDPM RDD checkpoint 没有在该目标下训练，不能直接声称成为 RDD-SiT。

正确论文结构是：

- 主线：RDD-Diffusion，展示离散 posterior/drift 和现有强结果；
- 扩展：RDD-FM 或 RDD-SiT，证明同一 resolution-dependent signal 思想可跨生成范式；
- 不把 SiT 本身算作 RDD 的创新。

### 15.3 公平实验

RDD-SiT 与 SiT baseline 必须共享：

- tokenizer、DiT 架构和参数量；
- 数据和训练 images 数；
- time sampler；
- optimizer、EMA；
- guidance 和 NFE。

然后只改变 $R_t$ 及对应 drift/边界处理。否则无法说明增益来自 RDD。

论文：[Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)；[SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)

---

## 16. REG：给生成过程增加一个全局语义状态

### 16.1 一句话直觉

REPA 是老师在训练时检查学生的中间笔记，采样时老师离开。REG 更像让模型始终随身携带一个“整张图的语义摘要变量”，这个变量也参与加噪和生成。

把普通 latent 状态 $x$ 扩展为联合状态：

$$
z=
\begin{bmatrix}
x\\
r
\end{bmatrix},
$$

其中 $r$ 是由预训练 encoder 得到或定义的 representation latent。联合路径：

$$
z_t=
\alpha_t
\begin{bmatrix}
x_0\\
r_0
\end{bmatrix}
+
\sigma_t
\begin{bmatrix}
\epsilon_x\\
\epsilon_r
\end{bmatrix}.
$$

模型同时预测图像状态和 representation 状态的变化：

$$
\mathcal L
=
\mathcal L_x+\lambda_r\mathcal L_r.
$$

### 16.2 它与 REPA 的根本区别

| 问题 | REPA | REG |
|---|---|---|
| teacher/representation 是否进入生成状态 | 否 | 是 |
| 采样时是否保留额外语义变量 | 否 | 是 |
| 能否简单移除辅助头 | 是 | 否 |
| 是否改变模型输入输出维度 | 通常不改主输出 | 会改 |
| 旧 checkpoint 是否兼容 | sampling 兼容，训练头需新建 | 不兼容 |

因此不能把 REG 描述成“另一个 cosine auxiliary loss”。

### 16.3 怎样做 RDD-REG

一个合理设计是：

$$
z_t^{(k)}
=
\begin{bmatrix}
x_t^{(k)}\\
r_t
\end{bmatrix},
$$

其中图像分量 $x_t^{(k)}$ 随 RDD stage 改变分辨率，而全局 $r_t$ 维度保持不变。这样 $r_t$ 像贯穿 8→16→32 的语义记忆。

潜在优势：

- 在分辨率切换时保留类别和全局布局；
- 防止 32×32 细节阶段忘掉 8×8 决定的全局语义；
- 可能同时改善 convergence 和 class consistency。

风险：

- 模型、loss 和 sampler 都要改；
- 参数量和 token 数增加，公平 baseline 必须匹配计算；
- $r$ 太强可能降低多样性；
- 与 RDD 都声称改善全局语义，归因比 REPA 更困难。

所以 REG 是高价值研究线，但不应在主论文最小组合实验中排在 REPA 前面。

论文：[Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think](https://arxiv.org/abs/2507.01467)

---

## 17. EDM2 的 magnitude preservation：让深网络里的数值尺度别乱跑

### 17.1 一句话直觉

训练很深的生成网络时，每层激活和权重的幅度会慢慢漂移。有的层越来越大，有的越来越小，优化器要不断适应。EDM2 的许多设计是在网络内部建立“音量标准”，让每一层输出保持可预测的尺度。

一个简化的归一化线性层思想：

$$
\tilde W=\frac{W}{\|W\|+\varepsilon},
\qquad
y=\tilde W x.
$$

两个方差相近、近似独立的 residual 分支相加时：

$$
y=\frac{x+F(x)}{\sqrt2},
$$

可以避免方差每层近似翻倍。真实 EDM2 实现还包括更系统的 weight normalization、pixel normalization、embedding/residual gain 和训练配置，不能只加一个 $\sqrt2$ 就称为 EDM2。

### 17.2 与 RDD 的关系

RDD 改输入路径和分辨率；magnitude preservation 改网络内部数值条件，概念上正交。它可能特别有利于 RDD，因为不同 token grid 的激活统计不同，stage 切换容易引起幅度漂移。

但它不是 checkpoint-compatible：

- 参数化发生变化；
- 初始化和学习率最佳值可能变化；
- 必须从头训练；
- baseline 也应采用同样的架构改造，才能隔离 RDD 增益。

建议把它放在“最终强系统”而不是“RDD 必要组件”中。

论文：[Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)

---

## 18. 更强 VAE / DC-AE / RAE：先提高 latent 世界的天花板

### 18.1 一句话直觉

latent diffusion 不直接生成像素，而是在 VAE 压缩后的世界里生成。如果 VAE 把小字、纹理或物体边界弄丢了，diffusion 再强也无法恢复不存在的信息。更好的 tokenizer 像换一张更清楚、结构更合理的地图。

编码和解码：

$$
z=E(x),\qquad \hat x=D(z).
$$

最终误差至少来自两部分：

$$
\text{generation error}
\approx
\text{latent modeling error}
+\text{reconstruction error}.
$$

### 18.2 为什么它可组合却不能直接复用模型

换 VAE 后：

- latent channel 数可能不同；
- latent scaling 和方差不同；
- 空间压缩率不同；
- average pooling 在 latent 中代表的频率含义改变；
- 旧 RDD checkpoint 的输入分布不再匹配。

所以 tokenizer 改进属于系统级正交，不是旧 checkpoint 的插件。RDD 中的 $R_t$、noise filling 和 stage grid 都要重新检查。

### 18.3 最公平的实验

做一个 $2\times2$：

| | 原 VAE | 新 tokenizer |
|---|---:|---:|
| 普通 DiT | A | B |
| RDD | C | D |

RDD 的贡献看 C−A 和 D−B；tokenizer 的贡献看 B−A 和 D−C。只报告 D 很强，无法说明是 RDD 还是 tokenizer。

### 18.4 潜在优势

- 降低 FID 的 reconstruction floor；
- 让低分辨率 stage 保留更有意义的语义；
- 更高压缩率可能进一步降低 RDD 计算；
- RAE 的语义 latent 可能与 RDD 的 coarse-to-fine 路径天然互补。

风险是工作量很大，也容易让论文主线从 RDD 偏到 tokenizer。

---

## 19. Zero-terminal-SNR 与 velocity prediction：修正端点，但要重推目标

### 19.1 一句话直觉

某些 noise schedule 在最后一个训练时刻仍残留一点信号，而采样却从纯高斯开始，训练与推理端点不完全一致。Zero-terminal-SNR 强制终点真的没有数据可见。

velocity prediction 常定义为：

$$
v_t=\alpha_t\epsilon-\sigma_t x_0.
$$

结合

$$
x_t=\alpha_t x_0+\sigma_t\epsilon
$$

可反解：

$$
x_0=\alpha_t x_t-\sigma_t v_t,
\qquad
\epsilon=\sigma_t x_t+\alpha_t v_t,
$$

这里假设 $\alpha_t^2+\sigma_t^2=1$。

### 19.2 为什么 RDD 不能只换一个 target 名字

RDD 的信号是 $R_t(x_0)$，并且还有 resolution drift。把 $x_0$ 替换为 $R_t(x_0)$ 后，velocity、posterior mean、边界 transport、SNR weighting 和 sampler conversion 都要一致重推。否则训练预测的是一种量，采样器按另一种量解释。

所以它是可组合研究方向，但必须：

1. 明确每个 stage 的 $\alpha,\sigma$；
2. 推导 RDD-v target；
3. 写出 v↔epsilon↔clean transformed signal 的单元测试；
4. 检查 terminal state 与 noise filling；
5. 从头训练公平 baseline。

---

# 第四部分：与 RDD 高度重叠，应该作为竞争方法而不是组合 trick

## 20. f-DM：更一般的 transformed-signal family

### 20.1 一句话直觉

普通 diffusion 加噪的是原信号 $x_0$。f-DM 类方法允许“被加噪的信号本身也随时间变化”：

$$
x_t=\alpha_t f_t(x_0)+\sigma_t\epsilon.
$$

RDD 采用 $f_t=R_t$，其中 $R_t$ 具有明确的分辨率/低通结构。因此 RDD 可以看作 transformed-signal family 中一个有结构、面向 coarse-to-fine 生成的实例，但这不等于 RDD 没有自己的贡献。

### 20.2 RDD 应怎样讲区别

最稳妥的 novelty 不是声称“首次让 signal 随时间变”，而是依次强调：

1. **特定结构**：$R_t$ 是分辨率投影/transport，而非任意 transform；
2. **生成计算语义**：token 数随时间改变，直接减少高噪声阶段的计算；
3. **可实现 posterior/transition**：对离散 stage、noise filling 和反向过程给出明确推导；
4. **统一 diffusion 与 FM**：同一 resolution-dependent path 在两类训练目标中得到一致解释；
5. **系统消融**：分辨率预算、drift、边界、solver、training convergence 均被单独验证。

### 20.3 为什么不建议“RDD + f-DM”

两者都改 $f_t(x_0)$。若再叠加另一个任意 transform，需要重新定义复合算子：

$$
\tilde f_t = g_t\circ R_t
$$

并重新推导 drift：

$$
\frac{\mathrm d}{\mathrm dt}\tilde f_t(x_0).
$$

这可以成为未来工作，但不是一个干净的正交 baseline。

---

## 21. Pyramidal Flow：同一类 coarse-to-fine 路径竞争者

### 21.1 一句话直觉

Pyramidal Flow 也让生成从低分辨率逐步走向高分辨率，并通常使用 flow matching。它与 RDD 的共同动机都是：高噪声阶段没必要支付完整高分辨率计算。

### 21.2 真正应该比较的维度

不要只问“都用了多尺度，所以是否一样”。应拆成：

| 维度 | RDD 应明确说明 | 比较意义 |
|---|---|---|
| 路径定义 | $R_t(x_0)$ 与显式 drift | 理论对象是否相同 |
| 边界 | posterior/transport/noise filling | stage 如何连接 |
| 训练目标 | diffusion、FM 两版 | 是否依赖一种范式 |
| 网络 | 同一模型跨 stage 还是特殊模块 | 参数和复用方式 |
| 计算 | 各 stage token 数、训练/采样 FLOPs | 是否真的省算力 |
| 预训练兼容 | 从头训练或可微调 | 实用性 |
| 实验 | 同数据、同参数、同 NFE/FLOPs | 是否公平 |

### 21.3 论文叙事

Pyramidal Flow 不应藏在脚注里。正面承认共同的 coarse-to-fine 方向，然后把 RDD 的“drift-aware stochastic process、posterior 推导、diffusion/FM 双实现和系统实验”讲成清楚的技术差异。审稿人更容易接受准确边界，而不是过度宽泛的 first claim。

论文：[Pyramidal Flow Matching for Efficient Video Generative Modeling](https://arxiv.org/abs/2410.05954)

---

## 22. PyramidalWan：预训练视频模型上的多尺度微调竞争者

### 22.1 一句话直觉

PyramidalWan 把 pyramidal/多阶段思路迁移到现有 Wan 类视频模型，重点是用较低成本适配强预训练 checkpoint。它和 RDD-video 的最大重合点，是时间上按阶段改变空间分辨率。

### 22.2 为什么不能当作“正交增强”

如果直接把它的 pyramidal path 加到 RDD-video，实际上是在叠加两套多分辨率路径定义。更合理的用法有两种：

1. **竞争 baseline**：同一个 Wan checkpoint、同一微调数据和 compute，比较 PyramidalWan-style 与 RDD-video。
2. **工程参考**：借用它的数据管线、冻结策略、LoRA/全参微调方案，但路径和目标保持 RDD 自己的定义。

### 22.3 RDD-video 需要突出什么

- spatial resolution drift 是否被显式建模；
- 时间维和空间维是否采用不同 transform；
- stage boundary 如何保持运动连续性；
- 能否同时支持 diffusion/SDE 与 flow matching；
- 相同训练成本下，比原 Wan 和 pyramidal baseline 的收敛速度；
- 相同采样成本下的视频质量与运动质量。

这才是“方法差异”；仅仅把 8×8、16×16、32×32 换成视频 latent grid 不足以形成强 novelty。

---

# 第五部分：真正落到当前仓库的实现

## 23. 当前代码应该怎样接这些方法

### 23.1 已有公共组件

主要实现位于：

- **diffusion/rdd_improvements.py**
  - diffusion SNR；
  - Min-SNR/P2 权重；
  - FM logit-normal 时间采样；
  - scheduled CFG；
  - Autoguidance；
  - teacher token pooling；
  - REPA cosine loss。
- **diffusion/timestep_sampler.py**
  - uniform sampler；
  - loss-aware sampler；
  - loss-second-moment sampler。
- **diffusion/repa.py**
  - DINOv2 teacher；
  - teacher patch tokens；
  - 兼容旧 checkpoint 的 REPA head loader。
- **2d/sample_fid_unified.py**
  - scheduled CFG wrapper；
  - Autoguidance wrapper；
  - DDPM/DPM 采样入口。
- **2d/train_RDD_imagenet256_unified.py**
  - diffusion 训练入口。
- **2d/train_RDD_flow_matching_imagenet256_unified.py**
  - FM 训练入口。

### 23.2 推荐的统一训练目标接口

训练代码不要为每个方法复制一份。应把目标拆成：

$$
\mathcal L_{\mathrm{total}}
=
w_{\mathrm{task}}(t)
\mathcal L_{\mathrm{RDD}}
+\lambda_{\mathrm{REPA}}\mathcal L_{\mathrm{REPA}}.
$$

其中：

- $t$ 从 uniform 或 loss-aware sampler 得到；
- $w_{\mathrm{task}}$ 只能在 baseline、Min-SNR、P2 中选一种；
- importance weight $1/(Tp(t))$ 由 timestep sampler 独立乘入；
- REPA 是额外辅助项，不应被误乘为 Min-SNR，除非明确设计并消融。

推荐数据流：

~~~python
t, importance = schedule_sampler.sample(batch_size)
x_t, target, stage = rdd_forward_process(clean_latent, noise, t)
pred, hidden = model(x_t, t, label, return_hidden=use_repa)

base = prediction_loss(pred, target)
task_weight = diffusion_weight(method, t, stage)
loss = mean(importance * task_weight * base)

if use_repa:
    loss = loss + repa_lambda * repa_loss(hidden, clean_image, stage)
~~~

### 23.3 推荐的统一采样接口

~~~python
for stage in [coarse, middle, fine]:
    solver.reset_history()
    for t_next in stage_schedule[stage]:
        pred = model(x, t, condition)
        pred = apply_scheduled_cfg(pred, stage, t)
        pred = apply_autoguidance_if_enabled(pred, stage, t)
        x = solver.step(pred, x, t, t_next)

    if stage is not fine:
        x = rdd_boundary_transport_and_noise_filling(x, stage)
~~~

最关键的一行是 **solver.reset_history()**。多步 solver 的旧历史不能跨分辨率边界。

---

## 24. 一套不会把变量搅在一起的实验矩阵

### 24.1 第一阶段：零训练成本

固定一个正确数据集的 RDD checkpoint：

| ID | Guidance | Solver/schedule | 目的 |
|---|---|---|---|
| S0 | CFG 1.0 | DDPM 250, 80/80/90 | 基线 |
| S1 | CFG 1.5 constant | 同 S0 | 现有强基线 |
| S2 | LIG 1.5 | 同 S0 | 测 guidance 区间 |
| S3 | cosine scheduled CFG | 同 S0 | 测平滑 schedule |
| S4 | 最优 guidance | stage-safe DPM | 测 solver |
| S5 | 最优 guidance | stage-aware AYS | 测时间节点 |
| S6 | Autoguidance | 与 S0 同 NFE 另报耗时 | 测质量方向 |

每次只改一列。S4/S5 必须报告实际 NFE。

### 24.2 第二阶段：短训筛选训练方法

同初始化或至少同 seed family，训练到 100K/200K：

| ID | Loss weighting | Time sampler | REPA |
|---|---|---|---|
| T0 | none | uniform | off |
| T1 | Min-SNR | uniform | off |
| T2 | P2 | uniform | off |
| T3 | none | loss-second-moment | off |
| T4 | none | uniform | finest-only |
| T5 | none | uniform | all-stage fixed-total |

先挑出 loss weighting winner 和 REPA winner，再做 T6=winner+winner。不要直接把所有开关全开。

### 24.3 第三阶段：最终强系统

只把第二阶段稳定有效的方法训练到完整预算：

$$
\text{RDD}
+\text{best loss allocation}
+\text{best REPA}
+\text{best zero-train sampler}.
$$

论文表中仍需保留纯 RDD 行。否则审稿人无法知道“最终数值很强”到底有多少来自 RDD。

### 24.4 FM 独立矩阵

先修正 coarse/middle/fine=400/300/300，再比较：

1. FM baseline + uniform time；
2. RDD-FM + uniform time；
3. RDD-FM + stage-local logit-normal；
4. 上述最佳模型分别用 Euler/Heun 或论文一致的 ODE solver。

DDPM 不是 FM 的自然 sampler 名称。若代码把某个随机 Euler/ancestral 更新叫 DDPM，论文必须准确写更新公式，不能只沿用脚本参数名。

---

## 25. 论文中怎样报告组合结果

### 25.1 三张最重要的表

**表 A：RDD 自身**

- DiT baseline；
- RDD without drift/boundary correction；
- full RDD；
- 不同 stage allocation；
- 同训练 images、参数量、采样 NFE。

**表 B：正交组合**

- pure RDD；
- +Min-SNR 或 P2；
- +REPA；
- +LIG；
- +solver/schedule；
- final stack。

**表 C：竞争路径**

- DiT/SiT；
- f-DM 类方法；
- Pyramidal Flow/Wan；
- RDD。

表 C 不能把别人论文中的最佳大模型数字与自己的小训练预算直接横比后下结论。必须注明分辨率、模型规模、数据、训练 compute、guidance 和 NFE；最好有同代码库重跑的 parity baseline。

### 25.2 一条很重要的措辞边界

可以说：

> RDD is compatible with training reweighting, representation alignment, and inference-time guidance because these methods optimize complementary layers of the generative system.

不要说：

> Every improvement in the final stack validates the RDD path.

LIG 或 REPA 的增益只能证明组合系统更强；只有控制变量的 RDD vs non-RDD 才验证 RDD。

---

## 26. 当前最推荐的优先顺序

### 必做

1. LIG / scheduled CFG：零训练、最便宜，直接用旧 checkpoint。
2. stage-safe solver parity：查清 DDPM 与 DPM 差异是否来自实现/配置。
3. Min-SNR：低实现风险，直接测试 convergence。
4. REPA finest-only 与 all-stage fixed-total：最有希望强化训练效率主张。
5. 正确 400/300/300 的 RDD-FM：修复公平性基础。

### 有余力再做

6. Loss-second-moment：既可能提效，也能诊断哪个 stage 真难。
7. AYS：在模型质量稳定后优化少步采样。
8. Autoguidance：需要合格 weak model，且推理成本翻倍。
9. Post-hoc EMA：新训练开始前把保存方案设计好。

### 不应抢占主线

10. REG：潜力大，但已经是新联合生成模型。
11. EDM2 全架构迁移：工程和调参成本高。
12. 新 tokenizer：会改变 latent 空间，适合后续强系统。
13. 任意 transform 叠加 f-DM：理论可做，但会模糊 RDD 主贡献。

---

## 27. 最后的直观记忆法

- **RDD**：改变“路本身”，让路在早期更窄、更便宜。
- **Min-SNR/P2**：改变“每一段路练多少力”。
- **Loss-aware sampling**：改变“训练时更常去练哪一段路”。
- **REPA**：请老师教模型“沿路应该看懂什么”。
- **LIG/Scheduled CFG**：只在需要时打开方向盘助力。
- **DPM/AYS**：一个决定怎样迈步，一个决定脚落在哪里。
- **Restart**：走偏时退回一点重新走。
- **EMA**：把训练抖动平均掉。
- **SiT/f-DM/Pyramidal Flow**：它们也在重新定义路，因此是理论框架或竞争路线，不是普通外挂。
- **REG/新 VAE/EDM2**：能组合，但会换车的一部分，必须重新训练和做公平 baseline。

如果只记住一句话：

$$
\boxed{
\text{先证明 RDD path 独立有效，再用正交方法做强系统；不要用强系统替代 RDD 消融。}
}
$$

---

## 参考论文入口

- [Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models](https://arxiv.org/abs/2404.07724)
- [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/abs/2406.02507)
- [DPM-Solver](https://arxiv.org/abs/2206.00927)
- [DPM-Solver++](https://arxiv.org/abs/2211.01095)
- [Align Your Steps](https://arxiv.org/abs/2404.14507)
- [Restart Sampling](https://arxiv.org/abs/2306.14878)
- [Efficient Diffusion Training via Min-SNR](https://arxiv.org/abs/2303.09556)
- [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227)
- [Improved DDPM](https://arxiv.org/abs/2102.09672)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Representation Alignment for Generation](https://arxiv.org/abs/2410.06940)
- [Representation Entanglement for Generation](https://arxiv.org/abs/2507.01467)
- [Stochastic Interpolants](https://arxiv.org/abs/2303.08797)
- [SiT](https://arxiv.org/abs/2401.08740)
- [EDM2](https://arxiv.org/abs/2312.02696)
- [Pyramidal Flow Matching](https://arxiv.org/abs/2410.05954)
