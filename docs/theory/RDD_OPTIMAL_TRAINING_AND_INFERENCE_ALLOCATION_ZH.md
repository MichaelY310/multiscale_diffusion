# RDD 最优训练分段与 Equal-FLOPs 推理分配

> 目标：用尽量少的完整训练，回答两个可以被实验检验的问题。
>
> 1. 对 `32→16→8`（反向生成顺序为 `8→16→32`），最好的训练时间区间是什么？
> 2. 对 `32→8`（反向生成顺序为 `8→32`），最好的训练时间区间是什么？
> 3. 给定训练好的 checkpoint，在相同总 FLOPs 下，每个 stage 应该采多少步？

本文中的训练分段一律按 **finest-to-coarsest** 写。例如：

- `300/300/400` = 32×32 负责全局 `t∈[0,300)`，16×16 负责 `[300,600)`，8×8 负责 `[600,1000)`；
- `400/600` = 32×32 负责 `[0,400)`，8×8 负责 `[400,1000)`。

采样 allocation 一律按真实生成顺序 **coarse-to-fine** 写。例如：

- `80/80/90` = 8×8 用 80 步、16×16 用 80 步、32×32 用 90 步；
- `400/100`（两阶段）= 8×8 用 400 步、32×32 用 100 步。

---

## 1. 最简单的人话版本

RDD 的训练分段问题，不是“哪个模型应该多训练几步”。当前代码每个 optimizer step 都会分别更新所有启用的 stage；三个 stage 的更新次数相同。

真正的问题是：

> 在噪声非常大时，8×8 已经够不够？什么时候图像中 8×8 表示不了的结构开始从噪声里显现，值得切换到 16×16？又什么时候 16×16 表示不了的细节开始显现，值得切换到 32×32？

所以，训练边界应该放在“下一层细节开始变得可恢复”的位置附近，而不应该由静态频带能量直接分配。

推理分配则是另一个问题：

> 已经有一个训练好的模型后，哪一段的数值积分最难、误差增长最快，就在那里多走几步；但 32×32 的一步比 8×8 贵很多，所以必须用 FLOPs 而不是总步数约束。

完整方法可以概括成一句话：

> **训练边界看“信息什么时候出现”，推理步数看“真实模型在哪里积分最难”，最后都用少量真实 FID 校准。**

---

## 2. 先把三个不同变量分开

### 2.1 训练 stage boundary

三阶段训练边界为：

$$
0=b_0<b_1<b_2<b_3=1000.
$$

对应：

$$
\begin{aligned}
32\times32 &: [0,b_1),\\
16\times16 &: [b_1,b_2),\\
8\times8 &: [b_2,1000).
\end{aligned}
$$

因此训练 split 是：

$$
(L_{32},L_{16},L_8)=(b_1,b_2-b_1,1000-b_2).
$$

两阶段只有一个边界：

$$
(L_{32},L_8)=(b,1000-b).
$$

### 2.2 stage 内部训练 timestep sampling

固定边界后，每个 stage 内部仍然可以 uniform sampling，或者使用 loss-second-moment importance sampling。

这个变量改变的是某个 stage 内哪些 local timestep 更常被看见，不改变 stage 负责的全局时间范围。

为了让 stage-boundary ablation 干净，主搜索中保持：

```text
timestep sampler = uniform
loss weighting    = none
```

找到边界后，再单独验证 `loss-second-moment` 或 Min-SNR；不要把这几个变量一起调。

### 2.3 inference allocation

训练完成以后，采样步数记作：

$$
(n_8,n_{16},n_{32})
$$

或两阶段的：

$$
(n_8,n_{32}).
$$

它不需要重新训练，可以在同一组 checkpoint 上搜索。

---

## 3. 为什么旧的训练分配公式不能直接使用

旧文档使用：

$$
C_k=E_kW_k,\qquad L_k\propto\sqrt{C_k},
$$

并假设误差为：

$$
\mathcal E=\sum_k\frac{C_k}{L_k}.
$$

这个推导隐含了“区间长度越长，stage 得到的训练样本越多”。但当前代码在每个 batch 中为每个 stage 分别采一个 local timestep，并更新每一个 stage。因此 400K optimizer steps 时，每个 stage 都得到 400K 次更新。

区间长度为 $L_k$ 时，每个 local timestep 的平均曝光密度反而约为：

$$
\text{exposure per local timestep}\propto\frac{400000}{L_k}.
$$

所以 $L_k$ 是模型负责的噪声区间，不是训练预算。

此外，静态频带能量只描述数据的二阶统计量，无法包含模型容量、SNR、gradient conflict、边界误差以及 rollout 误差。因此旧公式最多只能作为一个很弱的频谱启发式，不能称作 FID 最优理论。

---

## 4. 训练边界：信息什么时候值得用更高分辨率处理

### 4.1 数据分解

令 $P_r$ 表示“先 average pool 到分辨率 $r$，再 nearest-neighbor 放回 32×32”的投影。

三阶段中可以把一个 clean latent 分成：

$$
x_0=P_8x_0+(P_{16}-P_8)x_0+(I-P_{16})x_0.
$$

三部分分别代表：

- $P_8x_0$：8×8 已经能表示的最粗结构；
- $(P_{16}-P_8)x_0$：16×16 相比 8×8 新增的结构；
- $(I-P_{16})x_0$：只有 32×32 才能表示的细节。

两阶段则是：

$$
x_0=P_8x_0+(I-P_8)x_0.
$$

这比把 FFT 半径粗暴地切成三个理想频带更贴近真实代码，因为它使用了实际的 average-pool/repeat 算子。

### 4.2 每个细节带在时间 $t$ 的可恢复程度

忽略通道间相关性时，**不带 drift 的参考 forward process** 中某个细节分量可以近似写成：

$$
x_t^{(j)}=a_j(t)x_0^{(j)}+\sigma(t)\epsilon^{(j)}.
$$

其中 $j$ 可以是 `8→16` 新增的细节带，或 `16→32` 新增的细节带。先用这个 no-drift SNR 判断“如果不主动删除，该细节何时也会自然变得不可恢复”。随后再用 RDD 的真实 forward 公式检查整个 stage 内是否过早删除信息。

在当前线性 drift stage $I_k=[T_k,T_{k+1})$ 内，令

$$
u_k(t)=\frac{t-T_k}{T_{k+1}-T_k}.
$$

对即将被 projection 删除的 residual band，真实 RDD clean coefficient 是：

$$
a_j^{\rm RDD}(t)=\sqrt{\bar\alpha_t}\,[1-u_k(t)],
$$

而噪声系数仍为 $\sqrt{1-\bar\alpha_t}$。这个式子用于计算 candidate 的 path penalty；但由于它本身依赖候选边界，不能假装在搜索前就给出唯一解析最优解。

定义该细节带的有效信噪比：

$$
\operatorname{SNR}_j(t)
=
\frac{a_j(t)^2\operatorname{Var}[x_0^{(j)}]}
{\sigma_j(t)^2}.
$$

直觉非常简单：

- $\operatorname{SNR}_j(t)\ll1$：这部分基本仍是噪声，使用高分辨率很浪费；
- $\operatorname{SNR}_j(t)\approx1$：信号开始和噪声一样强，是合理的切换候选；
- $\operatorname{SNR}_j(t)\gg1$：细节已经明显，继续用低分辨率很可能损失质量。

在高斯二阶近似下，这部分 clean signal 的条件 MMSE 为：

$$
\operatorname{MMSE}_j(t)
=
\frac{\operatorname{Var}[x_0^{(j)}]}
{1+\operatorname{SNR}_j(t)}.
$$

定义“已经可恢复、但当前低分辨率表示不了的能量”：

$$
R_j(t)
=
\operatorname{Var}[x_0^{(j)}]
\frac{\operatorname{SNR}_j(t)}{1+\operatorname{SNR}_j(t)}.
$$

当 $R_j(t)$ 超过阈值时，就应该考虑切换分辨率。

### 4.3 数据先验给出的候选边界

沿 forward diffusion 的方向，SNR 随 $t$ 增大而下降。对 `16→32` 新增细节，先找它第一次低于阈值的位置：

$$
b_{16\to32}^{\rm data}
=
\min\{t:\operatorname{SNR}_{32\setminus16}(t)\le\tau\}.
$$

对 `8→16` 新增细节同理：

$$
b_{8\to16}^{\rm data}
=
\min\{t:\operatorname{SNR}_{16\setminus8}(t)\le\tau\}.
$$

通常高频细节能量更小，会更早变得不可恢复，因此应有 $b_{16\to32}<b_{8\to16}$。训练文件使用 finest-to-coarsest 边界，所以 $(b_1,b_2)=(b_{16\to32},b_{8\to16})$。生成时则反向经过 `8→16→32`。

阈值 $\delta$ 不是假装从天上掉下来的“普适常数”。推荐使用以下顺序：

1. 先报告无调参的 $\tau=1$ 预测；
2. 再用一个已有实验，例如 `300/300/400`，校准一个全局容忍度；
3. 用同一个容忍度预测两阶段边界和未见过的三阶段候选；
4. 不允许针对每个 split 单独调阈值。

这样能够区分“零样本理论预测”和“单点校准后的数据驱动预测”。

### 4.4 模型感知修正

数据频谱只告诉我们信息什么时候出现，并不知道真实 DiT 是否学得动。因此使用短训练 checkpoint 测量三个量。

#### A. 分 timestep 的验证 denoising loss

$$
\ell_r(t)=
\mathbb E\left[\|\epsilon_\theta(x_t,t)-\epsilon_{\rm target}\|^2\right].
$$

应按全局 timestep bin 记录，而不是只报告整个 stage 的一个平均值。

#### B. clean latent 重建误差

不同 timestep 的 epsilon MSE 不一定可直接比较，因此还要换算为 predicted-$x_0$ 误差：

$$
e_r(t)=
\mathbb E\left[\|\hat x_{0,r}(x_t,t)-x_{0,r}\|^2\right].
$$

论文主预测评分优先使用 $e_r(t)$。

#### C. stage transition 误差

在候选边界 $b$，用 ground-truth coarse latent 做 teacher-forced transition，再比较 noise filling 后的状态和真实高分辨率 forward state：

$$
h_{r\leftarrow r'}(b)
=
\mathbb E\left[
\|\tilde x_b^{r}-x_b^{r}\|^2
\right].
$$

这个量会直接惩罚过于激进的 `8→32` 跳跃。

### 4.5 最终训练边界评分

三阶段评分定义为：

$$
\begin{aligned}
J_3(b_1,b_2)=
&\int_0^{b_1}w(t)e_{32}(t)dt\\
&+\int_{b_1}^{b_2}w(t)e_{16}(t)dt\\
&+\int_{b_2}^{1000}w(t)e_8(t)dt\\
&+\eta h_{32\leftarrow16}(b_1)
+\eta h_{16\leftarrow8}(b_2)\\
&+\gamma J_{\rm spectral}(b_1,b_2).
\end{aligned}
$$

两阶段评分为：

$$
\begin{aligned}
J_2(b)=
&\int_0^bw(t)e_{32}(t)dt
+\int_b^{1000}w(t)e_8(t)dt\\
&+\eta h_{32\leftarrow8}(b)
+\gamma J_{\rm spectral}(b).
\end{aligned}
$$

其中：

- $w(t)$ 是固定评价权重；主实验使用 uniform，Min-SNR 只做敏感性分析；
- $J_{\rm spectral}$ 是上一节的数据先验；
- $\eta,\gamma$ 只能用已有训练配置校准一次；
- 两阶段和三阶段共用校准规则。

在 50-step 网格上，三阶段只有约 171 个合法边界组合。可以直接枚举；若以后增加更多 stage，可以用 dynamic programming。

### 4.6 从零样本边界到最终边界：逐轮 refine 的完整算法

这一节把前面的量连成一条真正可以执行的搜索流程。先明确：最终要优化的变量只有训练边界。

三阶段变量为：

$$
\mathbf b=(b_1,b_2),\qquad 0<b_1<b_2<1000,
$$

并转换为：

$$
(L_{32},L_{16},L_8)=(b_1,b_2-b_1,1000-b_2).
$$

两阶段变量只有：

$$
b,qquad (L_{32},L_8)=(b,1000-b).
$$

以下所有 `left/right` 都是移动边界，不是改变 optimizer update 数，也不是改变 stage 内的 timestep sampler。

#### Round 0：只看数据，得到一个不训练模型的初始边界

对固定的 10K 张 native ImageNet 图片执行：

1. 使用论文训练时相同的 SD-VAE，把图片编码为 $z_{32}$；
2. 用代码真实的 average-pool/repeat 投影得到 $P_{16}z$ 和 $P_8z$；
3. 构造三个 residual band：

$$
\begin{aligned}
d_{32\setminus16}&=z_{32}-P_{16}z_{32},\\
d_{16\setminus8}&=P_{16}z_{32}-P_8z_{32},\\
d_{32\setminus8}&=z_{32}-P_8z_{32};
\end{aligned}
$$

4. 在 image、channel 和 spatial location 上计算每个 band 的每元素方差：

$$
v_j=\mathbb E[d_j^2];
$$

5. 使用训练代码中真实的 1000-step beta schedule，计算：

$$
\bar\alpha_t=\prod_{s=0}^{t}(1-\beta_s),
$$

以及 no-drift reference SNR：

$$
\operatorname{SNR}_j^{\rm ref}(t)
=
\frac{\bar\alpha_t v_j}{1-\bar\alpha_t};
$$

6. 使用无调参起点 $\tau=1$，寻找 forward diffusion 中第一次满足 SNR 低于阈值的 timestep：

$$
\begin{aligned}
b_1^{(0)}&=\min\{t:\operatorname{SNR}_{32\setminus16}^{\rm ref}(t)\le1\},\\
b_2^{(0)}&=\min\{t:\operatorname{SNR}_{16\setminus8}^{\rm ref}(t)\le1\},\\
b^{(0)}&=\min\{t:\operatorname{SNR}_{32\setminus8}^{\rm ref}(t)\le1\}.
\end{aligned}
$$

如果出现 $b_1^{(0)}\ge b_2^{(0)}$，说明简单的独立高斯/SNR近似不能给出合法的三阶段顺序。此时不能偷偷交换两个值，而应把它报告为 data-only predictor failure，并把相邻合法 50-step grid 点交给 pilot search。

这个 Round 0 的含义是：

> 当 32×32 独有的细节在 forward diffusion 中已经比噪声弱时，继续使用昂贵的 32×32 表示可能没有必要；16×16 与 8×8 的边界同理。

它还没有考虑模型能力、训练曝光、真实 rollout 和 RDD transition，所以只是中心点，不是最终答案。

#### Round 0.5：给初始边界加上不确定区间

不能默认 `predicted±100`。边界移动量应由数据不确定性决定。

具体做两种 perturbation：

1. **Dataset bootstrap**：从 10K latent 中有放回抽样 200 次，每次重新计算 $v_j$ 和 crossing boundary；
2. **Threshold sensitivity**：除 $\tau=1$ 外，再计算 $\tau\in\{0.5,2\}$ 的 crossing。

将 bootstrap 的 5%–95% 分位点和三个阈值的结果合并，得到：

$$
I_1=[b_{1,\min},b_{1,\max}],\qquad
I_2=[b_{2,\min},b_{2,\max}],
$$

或两阶段的 $I=[b_{\min},b_{\max}]$。

定义某个边界第一轮的搜索半径：

$$
\Delta_j=operatorname{clip}
\left(
50\left\lceil\frac{|I_j|/2}{50}\right\rceil,
50,
150
\right).
$$

因此边界不稳定时搜索得远一些，数据预测稳定时只搜索相邻 50 timestep。文档里写的 `±100` 只是 $\Delta=100$ 的一个例子。

#### Round 1：理论中心附近的第一批候选

三阶段一次改变一个边界，避免无法判断究竟是哪一个边界产生了变化：

$$
\begin{aligned}
C_0&=(b_1^{(0)},b_2^{(0)}),\\
C_{1-}&=(b_1^{(0)}-\Delta_1,b_2^{(0)}),\\
C_{1+}&=(b_1^{(0)}+\Delta_1,b_2^{(0)}),\\
C_{2-}&=(b_1^{(0)},b_2^{(0)}-\Delta_2),\\
C_{2+}&=(b_1^{(0)},b_2^{(0)}+\Delta_2).
\end{aligned}
$$

通常删除不满足 $100\le b_1<b_2\le900$ 或任一 stage 少于 100 timestep 的点；但有一个例外：如果 data-only predictor 的**原始中心点**落在 50-step 网格上，且某个 stage 恰好只有 50 timestep，就保留这个中心点作为机制检验，同时再加入一个每阶段至少 100 timestep 的稳健对照。不能在 predictor 给出短 stage 后，悄悄把它投影成 100 再声称“理论预测就是投影后的点”。已有的 `300/300/400` 和 `100/300/600` 作为免费观测加入候选表。

为了控制成本，推荐使用**顺序坐标搜索**而不是一次训练五个新三阶段配置：

1. 比较 $I_1$ 和 $I_2$ 的宽度，以及已有结果主要沿哪个方向变化；
2. 第一轮只训练中心点和更不确定边界的两个邻居，共最多 3 个新三阶段配置；
3. 在 Round 2 固定第一轮找到的较好边界，再搜索另一个边界。

例如预测中心为 `(350,250,400)`，即 $(b_1,b_2)=(350,600)$，且第一个边界更不确定、$\Delta_1=100$：

| 候选 | $(b_1,b_2)$ | 训练 split $(L_{32},L_{16},L_8)$ |
|---|---:|---:|
| center | `(350,600)` | `350/250/400` |
| $b_1$ left | `(250,600)` | `250/350/400` |
| $b_1$ right | `(450,600)` | `450/150/400` |

注意 `left` 并不是把三项都减 100。它移动一个 boundary，因此相邻两个 stage 的长度一个减少、一个增加，总和仍为 1000。

两阶段没有第二个坐标，直接训练：

$$
C_0=b^{(0)},\qquad C_-=b^{(0)}-\Delta,\qquad C_+=b^{(0)}+\Delta,
$$

再加入 `500/500` 作为不依赖理论的中心对照。例如 $b^{(0)}=450,\Delta=100$ 时得到 `450/550`、`350/650`、`550/450` 和 `500/500`。

#### Round 1a：50K 只做健康检查，不选最优边界

每个 candidate 都要独立训练一套模型，因为边界改变了每个模型看到的 $(x_t,t)$ 分布以及 drift 区间。所有候选必须固定：

```text
native ImageNet split/crop
global batch size = 256
optimizer、learning rate、EMA
stage 内 timestep sampler = uniform
loss weighting = none
seed 和 checkpoint step
```

50K 的淘汰只包含：

- loss 或 latent 出现 NaN/Inf，且重跑后仍复现；
- loss 持续发散，没有下降趋势；
- checkpoint 无法正常加载或 sample；
- 生成结果出现明确 mode collapse；
- 在相同 2K sanity sample 中，FID 比所有健康候选差一个数量级，并且 reconstruction/transition 指标也同时崩溃。

普通的早期 FID 差异不算失败。比如 45、48、53 都应继续到 100K，因为不同 split 的收敛速度可能不同。

实现 bug、磁盘损坏或 OOM 也不算科学意义上的 candidate failure；应修复后重跑。

#### Round 1b：100K 测量模型感知的三个分量

所有健康候选继续到 100K。固定一份 validation protocol：

```text
10K held-out native ImageNet images
固定 crop、固定 label 顺序、固定 noise seed
使用 EMA checkpoint
每个 stage 划分 20 个等宽 global-t bins
每个 bin 使用相同数量样本
CFG = 1.0；不做生成 guidance
```

对每个候选 $C$ 计算以下量。

**A. epsilon loss 曲线（诊断项）**

$$
\ell_{r,C}(t_m)
=\frac{1}{N_mD_r}
\sum_{i=1}^{N_m}
\|\hat\epsilon_{r,C}(x_{t_m}^{(i)},t_m)-\epsilon^{(i)}\|_2^2.
$$

它检查某个 timestep bin 是否没有学会，画成曲线并报告。它不再单独加入总分，因为下面的 predicted-$x_0$ error 已经把同一预测误差转换到了更可比较的 clean-signal 空间；两者同时加入会重复计数。

**B. predicted-$x_0$ reconstruction error（模型主项）**

$$
e_{r,C}(t_m)
=\frac{1}{N_mD_r}
\sum_{i=1}^{N_m}
\|\hat x_{0,r,C}^{(i)}-x_{0,r}^{(i)}\|_2^2.
$$

对一个 candidate 的区间做数值积分：

$$
E_C=
\frac{1}{1000}
\left[
\int_0^{b_1}e_{32,C}(t)dt
+\int_{b_1}^{b_2}e_{16,C}(t)dt
+\int_{b_2}^{1000}e_{8,C}(t)dt
\right].
$$

实际实现使用 20-bin trapezoidal sum。每个误差先除以 latent 元素数，防止 32×32 仅因元素更多而天然更大。

**C. transition error（边界主项）**

每个边界测两个版本。

Teacher-forced 版本使用真实 coarse clean latent，只检查 topology、boundary 和 noise filling：

$$
H_C^{\rm teacher}
=h_{32\leftarrow16}^{\rm teacher}(b_1)
+h_{16\leftarrow8}^{\rm teacher}(b_2).
$$

Rollout 版本使用 coarse model 真正预测的 $\hat x_0$，然后执行 noise filling，与同 image、同 noise 的真实 higher-resolution boundary state 比较：

$$
H_C^{\rm rollout}
=h_{32\leftarrow16}^{\rm rollout}(b_1)
+h_{16\leftarrow8}^{\rm rollout}(b_2).
$$

选边界时使用 rollout 版本；teacher-forced 只解释误差来自 transition 本身，还是来自 coarse model 已经预测错了。两阶段把上式替换为单个 $h_{32\leftarrow8}(b)$。

**D. spectral recoverability penalty（数据先验项）**

定义归一化 spectral penalty：

$$
S_C=J_{\rm spectral}(b_1,b_2)
=\frac{
\int_{b_1}^{1000}R_{32\setminus16}(t)dt
+\int_{b_2}^{1000}R_{16\setminus8}(t)dt
}{1000\,\operatorname{Var}(z_{32})}.
$$

它累计“已经交给低分辨率处理、但高分辨率独有信息仍理论可恢复”的能量。边界越早，通常这个 penalty 越大。两阶段使用：

$$
S_C^{(2)}
=\frac{\int_b^{1000}R_{32\setminus8}(t)dt}
{1000\,\operatorname{Var}(z_{32})}.
$$

#### 如何把 B、C、D 合成可比较的 predictor score

$E_C$、$H_C$、$S_C$ 的数值单位不同，不能直接相加。对同一 topology、同一 checkpoint step 的所有 candidates，分别使用 median 和 IQR 做稳健标准化：

$$
\tilde X_C=
\frac{X_C-\operatorname{median}_{C'}X_{C'}}
{\operatorname{IQR}_{C'}X_{C'}+10^{-8}},
\qquad X\in\{E,H,S\}.
$$

预注册的无拟合 score 使用等权：

$$
J_C^{\rm equal}=\tilde E_C+\tilde H_C+\tilde S_C.
$$

这样不需要事后手调 $\eta$ 和 $\gamma$。如果候选不少于 5 个，还可以额外拟合一个**只用于下一轮预测**的 non-negative ridge regression：

$$
\widehat{\operatorname{FID}}_{10K,C}
=\beta_0+\beta_E\tilde E_C+\beta_H\tilde H_C+\beta_S\tilde S_C,
\qquad \beta_E,\beta_H,\beta_S\ge0.
$$

必须用 leave-one-candidate-out prediction 报告误差，不能在同一批候选上拟合后又把训练拟合优度当作理论成功。

100K 时真实 `FID-10K` 仍然是主要筛选量；$J_C$ 的作用是：

- 在 FID 很接近时选择更合理的候选；
- 判断向哪个方向继续移动边界；
- 检验提出的 boundary predictor 是否真的与 FID 排名相关。

#### Round 2：用局部结果移动中心并搜索另一个边界

假设第一轮沿 $b_1$ 测了三个点 $b_1^-<b_1^0<b_1^+$，固定 $b_2$。用 100K FID 或 cross-validated predicted FID 记作 $y^-,y^0,y^+$。

- 若中间最好，缩小半径：$\Delta_1\leftarrow\max(50,\Delta_1/2)$；
- 若左边最好，中心移到左边，并在更左侧补一个点；
- 若右边最好，中心移到右边，并在更右侧补一个点；
- 若三个点差异小于预注册的 10K-FID noise tolerance，例如 0.5，则认为该方向形成 plateau，不再精调。

然后固定得到的 $b_1^{(1)}$，对另一个边界训练：

$$
(b_1^{(1)},b_2^{(0)}-\Delta_2),\quad
(b_1^{(1)},b_2^{(0)}),\quad
(b_1^{(1)},b_2^{(0)}+\Delta_2).
$$

已有 checkpoint 与新中心完全相同时直接复用；不能通过修改 checkpoint metadata 假装它属于另一个边界。

这个过程是离散 coordinate descent。推荐最多两轮 refine，避免把 ImageNet validation FID 调成一个超大的 hyperparameter search。

#### Round 3：决定哪些配置跑到 400K

每种 topology 最多保留两个 100K candidates：

1. 100K FID 最好者；
2. 以下候选之一：FID 与最好者接近但 $J_C$ 更低，或 50K→100K 改善斜率最大者。

两个候选独立续训到 400K。最终训练边界只由相同 protocol 下的 50K-sample FID 决定。若 FID 差小于 0.10，报告为统计上近似持平，并优先选择：

1. transition error 更小；
2. 实测 latency 更低；
3. split 更简单、对 bootstrap/seed 更稳定。

不允许因为 400K 赢家与理论预测不同而回头重新调 $\tau$，再声称 data-only predictor 精确命中。正确报告应同时给出 initial prediction、每一轮实际移动和最后经验最优。

### 4.7 完整训练边界搜索伪代码

```python
# ---------- Round 0: data-only prediction ----------
z32 = encode_fixed_imagenet_10k_with_sd_vae()
z16 = project_avgpool_repeat(z32, target=16)
z8  = project_avgpool_repeat(z32, target=8)

bands = {
    "32-16": z32 - upsample(z16),
    "16-8":  upsample(z16) - upsample(z8),
    "32-8":  z32 - upsample(z8),
}
variance = {name: mean(square(band)) for name, band in bands.items()}

alpha_bar = cumulative_product(1 - real_training_betas)
snr[name, t] = alpha_bar[t] * variance[name] / (1 - alpha_bar[t])
recoverable[name, t] = variance[name] * snr[name, t] / (1 + snr[name, t])

b1_0 = first_t(snr["32-16", t] <= 1)  # fine -> middle
b2_0 = first_t(snr["16-8",  t] <= 1)  # middle -> coarse
b_0  = first_t(snr["32-8",   t] <= 1) # direct fine -> coarse

# Bootstrap images and vary tau in {0.5, 1, 2}.
I1, I2, I_direct = boundary_uncertainty_intervals()
delta1 = grid50(clamp(width(I1) / 2, 50, 150))
delta2 = grid50(clamp(width(I2) / 2, 50, 150))
delta  = grid50(clamp(width(I_direct) / 2, 50, 150))

# ---------- Round 1: train local candidates ----------
uncertain_axis = argmax(width(I1), width(I2))
three_stage_candidates = center_and_two_neighbors_on_one_axis(
    center=(b1_0, b2_0),
    axis=uncertain_axis,
    radius=(delta1, delta2),
)
two_stage_candidates = unique([b_0 - delta, b_0, b_0 + delta, 500])

# Free observations: only checkpoints trained on native ImageNet with exact boundaries.
add_if_exact_checkpoint_exists(three_stage_candidates, split="300/300/400")
add_if_exact_checkpoint_exists(three_stage_candidates, split="100/300/600")

for candidate in new_candidates:
    train_independent_models(
        boundary=candidate,
        max_optimizer_steps=50_000,
        timestep_sampler="uniform",
        loss_weighting="none",
        global_batch_size=256,
    )
    if implementation_or_numeric_failure(candidate):
        fix_and_rerun(candidate)
    if not_reproducible_catastrophic_model_failure(candidate):
        continue_training(candidate, max_optimizer_steps=100_000)

# ---------- Round 1b: score healthy 100K candidates ----------
for candidate in healthy_candidates:
    epsilon_curve[candidate] = evaluate_epsilon_mse_in_20_t_bins(candidate)
    E[candidate] = integrate_pred_x0_mse_in_20_t_bins(candidate)
    H_teacher[candidate] = evaluate_teacher_forced_transitions(candidate)
    H[candidate] = evaluate_rollout_transitions(candidate)
    S[candidate] = integrate_recoverable_energy_after_each_boundary(candidate)
    fid10k[candidate] = sample_and_compute_fid(candidate, samples=10_000)

E_z, H_z, S_z = robust_standardize_across_candidates(E, H, S)
J_equal = E_z + H_z + S_z
check_spearman_correlation(J_equal, fid10k)

# ---------- Round 2: discrete coordinate refinement ----------
best = rank_primarily_by_fid10k_then_J(fid10k, J_equal)
move_or_shrink_searched_boundary(best, tolerance_fid=0.5, minimum_grid=50)
fix_first_boundary_and_search_other_boundary()
train_only_new_points_to_100k_and_repeat_scoring()

# ---------- Round 3: final training ----------
top2_three_stage = choose_best_and_best_alternative(topology="32-16-8")
top2_two_stage   = choose_best_and_best_alternative(topology="32-8")
continue_all_to_400k(top2_three_stage + top2_two_stage)

final_boundary = select_by_equal_protocol_fid50k()
report_full_search_trace_without_retuning_tau_after_seeing_final_fid()
```

### 4.8 这套训练方法真正声称什么

正确的论文表述是：

> 我们提出一个低成本、RDD-specific 的 boundary predictor。它结合 projection-residual recoverability、短训练的 timestep-wise reconstruction error 和 transition error，用少量 pilot checkpoint 预测接近 FID 最优的分段。

不要声称：

> 仅凭自然图像功率谱，我们解析证明了 ImageNet FID 的全局最优 split。

---

## 5. Equal-FLOPs inference allocation

### 5.1 每个 stage 一步到底有多贵

不能直接使用 $2\times\text{参数量}\times\text{token 数}$：AdaLN 等参数每次只作用在 conditioning vector 上，不能再乘 token 数。本文按当前 `DiT/models.py` 逐项计算主要 dense operation：patch/time embedding、QKV 与 attention、MLP、每层 AdaLN、final projection。一个 multiply-add 按 2 FLOPs；忽略 layer norm、激活、位置编码相加和 VAE。

对 hidden size $d$、depth $D$、token 数 $N$，每个 DiT block 的主要 MACs 为：

$$
\operatorname{MACs}_{\rm block}
=12Nd^2+2N^2d+6d^2.
$$

前三项依次对应 QKV/output 与 MLP、attention 矩阵乘、AdaLN modulation。加上输入/时间 embedding 和 final layer 后得到下表。

当前 multistage 模型为：

| Stage | 模型 | Token 数 | 单次 forward FLOPs |
|---|---:|---:|---:|
| 32×32 | DiT-XL/2 | 256 | 237.243 G |
| 16×16 | DiT-L/2 | 64 | 39.372 G |
| 8×8 | DiT-B/2 | 16 | 2.817 G |

以 8×8 为 1，一步成本比例约为：

$$
c_{32}:c_{16}:c_8=84.21:13.97:1.
$$

这说明“总步数相同”远远不等于“计算量相同”。

### 5.2 CFG 与 solver 的真实模型调用数

DDPM 中，每个 update 调用一次模型：

$$
m_r^{\rm DDPM}=n_r.
$$

当前 DPM2 实现的第一步是一阶；中间 update 调用模型两次；最后一个 internal `t=0→0` trivial update 仍调用模型一次。因此直接按源码计数：

$$
m_r^{\rm DPM2}=\max(1,2n_r-2).
$$

CFG=1.5 时，代码把 conditional 和 unconditional batch 拼在一起，因此按单样本理论 FLOPs 计，需要再乘 2：

$$
q_{\rm CFG}=
\begin{cases}
1,&\mathrm{CFG}=1,\\
2,&\mathrm{CFG}>1.
\end{cases}
$$

总 FLOPs 为：

$$
B=q_{\rm CFG}\sum_r c_rm_r.
$$

### 5.3 主预算

已有 `80/80/90`（8/16/32）的非 CFG 基础预算是：

$$
B_0=24.727043768320\ \text{TFLOPs/image}.
$$

CFG=1.5 时真实条件+无条件总预算约为：

$$
B_{\rm CFG}=49.454087536640\ \text{TFLOPs/image}.
$$

新 allocation search 固定这个预算，而不固定 250 updates。

为了说明结论是否依赖预算，最终赢家再增加两个预算：

$$
0.5B_0,\quad B_0,\quad 2B_0.
$$

主论文以 $B_0$ 为主，另外两个只画 quality-compute 曲线。

### 5.4 如何预测最优步数

对固定 checkpoint，在参考轨迹上测量局部积分误差密度 $a_r(t)$。

最实用的方法是 step doubling：

1. 从同一个状态走一个大步，得到 $x_{\rm one}$；
2. 用两个半步走到同一终点，得到 $x_{\rm two}$；
3. 用二者之差估计局部离散化误差：

$$
a_r(t)\propto
\frac{\|x_{\rm one}-x_{\rm two}\|^2}{h^{2(p+1)}}.
$$

对 $p$ 阶 solver，在固定 FLOPs 下的连续最优步密度满足：

$$
\rho_r^*(t)
\propto
\left(\frac{a_r(t)}{c_r}\right)^{1/(p+1)}.
$$

因此：

- 一阶 solver：$\rho\propto\sqrt{a/c}$；
- 二阶 solver：$\rho\propto(a/c)^{1/3}$。

stage 的预测步数是：

$$
n_r\propto
\int_{I_r}\rho_r^*(t)dt.
$$

然后把连续解取整，并在严格 FLOPs 预算附近搜索整数邻域。

### 5.5 为什么仍然需要 5K FID screen

局部 latent MSE 不是 FID，误差还会经过：

- 后续 stage 放大或修正；
- noise filling transition；
- CFG；
- VAE decode；
- 生成分布的非线性变化。

因此理论 allocation 只负责把巨大整数空间缩小成 5–7 个候选。最终用相同 seed、5K samples 做 screen，再对前两名做 50K FID。

这不是暴力搜索：三阶段在 1000-step 上有几十万种 allocation，而我们只完整评价理论解附近的少数点。

### 5.6 Equal-FLOPs 候选必须包含哪些对照

每个 checkpoint 至少比较：

1. **Predicted**：局部误差公式预测的 allocation；
2. **Uniform-time transformed**：从 80/80/90 出发，投影到相同 FLOPs；
3. **Fine-heavy**：在相同 FLOPs 下尽量增加 32×32；
4. **Middle-heavy**：增加 16×16；
5. **Coarse-heavy**：增加 8×8；
6. **Predicted − one grid step**；
7. **Predicted + one grid step**。

两阶段删除 middle-heavy，其余相同。

所有步数还必须满足：

$$
1\le n_r\le L_r,
$$

避免向一个只有 100 个原始 timestep 的 stage 请求 300 个不同 respaced timestep。

---

## 6. 实验计划表

### 6.1 Phase A：不训练模型的数据先验

| 编号 | 实验 | 数据量 | 输出 | 是否训练 |
|---|---|---:|---|---:|
| A1 | ImageNet 原图经 SD-VAE 得到 latent | 10K images | 32×32 latent | 否 |
| A2 | 测量 $P_8x_0$、$(P_{16}-P_8)x_0$、$(I-P_{16})x_0$ 方差 | 同 A1 | projection-residual spectrum | 否 |
| A3 | 真实 beta schedule 的 no-drift SNR crossing | 同 A1 | 两阶段和三阶段的数据先验边界 | 否 |
| A4 | 对 image bootstrap 与 $\tau\in\{0.5,1,2\}$ 做敏感性分析 | 同 A1 | 每个边界的 5%–95% 区间和搜索半径 $\Delta$ | 否 |

### 6.2 Phase B：训练边界 successive halving

#### 三阶段候选

复用已有正确数据：

- `100/300/600 @ 400K`；
- `300/300/400 @ 400K`。

第一轮最多新训练三个配置：

1. 理论预测 split，取整到 50；
2. 沿 bootstrap/threshold 区间更宽的边界向左移动 $\Delta$；
3. 沿同一边界向右移动 $\Delta$。

其中 $\Delta$ 由 4.6 节的不确定区间得到，并限制在 50–150；不是固定为 100。如果预测 split 与已有配置相同则直接复用，不为凑够三个配置随意加入一个不相关 stress test。第一轮 100K 结束后固定较好的第一个边界，再沿第二个边界做最多三个点的 Round 2 coordinate refinement。

#### 两阶段候选

两阶段没有可复用的严格 checkpoint。训练四个配置：

1. 理论预测 `b/(1000-b)`；
2. `(b-\Delta)/(1000-b+\Delta)`；
3. `(b+\Delta)/(1000-b-\Delta)`；
4. `500/500` 平衡对照。

边界统一裁剪到 `[200,800]` 并取 50 的倍数。两阶段只有一个边界，因此不需要第二轮坐标切换；如果最好点落在搜索区间端点，再向该方向外扩一个 50-step 点。

#### 训练停止规则

| 阶段 | 新三阶段配置 | 新两阶段配置 | samples/config | 目的 |
|---|---:|---:|---:|---|
| 50K checkpoint | Round 1 最多 3 | 4 | 2K | 只做健康检查，不以普通 FID 差异淘汰 |
| 100K checkpoint | Round 1 健康项；必要时加 Round 2 | 最多 4；端点最优时加 1 | 10K | FID 主排序，并检查 predictor 相关性 |
| 400K checkpoint | top 2 | top 2 | 后续 equal-FLOPs search | 正式比较 |

不建议只凭 50K 排名直接判最终最优；50K 只排除可复现的数值崩溃或灾难性 mode collapse。OOM、文件损坏和代码错误要修复重跑，不能记成边界失败。100K 完成 coordinate refinement 后，每种 topology 的 top 2 才继续到 400K。

### 6.3 Phase C：固定 FLOPs 的 inference search

对每个进入 400K 的 checkpoint：

| 步骤 | allocation 数 | samples/allocation | CFG | 说明 |
|---|---:|---:|---:|---|
| C1 local-error profile | 0 | 256–1024 latent trajectories | 1.0 | 不算 FID，只估计 $a_r(t)$ |
| C2 equal-FLOPs screen | 5–7 | 5K | 1.5 | 相同 seed 与 $B_0$ |
| C3 confirmation | top 2 | 10K | 1.5 | 排除 5K FID 噪声 |
| C4 final | winner | 50K | 1.0 与 1.5 | 正式表格 |

主 sampler 先用 DDPM，因为已有结果最多。DPM2 在最终最佳训练 split 上单独重复，不与 DDPM 共用 allocation；DPM2 的模型调用数是 `max(1,2n-2)`，必须重新做 equal-FLOPs 搜索。

### 6.4 Phase D：验证预测方法，而不只报告赢家

论文至少报告：

1. 预测训练评分与 100K FID 的 Spearman correlation；
2. 预测边界距离经验最好边界多少 timestep；
3. spectral-only、model-only、spectral+transition 三种 predictor ablation；
4. predicted inference allocation 与 equal-FLOPs 经验最优的距离；
5. fixed-NFE 和 fixed-FLOPs 排名是否发生变化；
6. 两阶段与三阶段都成立，避免方法只适配一个结构；
7. 两阶段 transition error 是否显著高于三阶段，从而解释 16×16 的价值。

---

## 7. 到底需要训练多少个模型

这里区分“训练配置”和“stage 网络”。

### 推荐论文方案

已有可复用：

- 2 个三阶段训练配置；
- 每个包含 32/16/8 三个 checkpoint。

第一轮新增：

- 3 个三阶段配置，每个包含 3 个 stage 网络：共 9 个新 stage 网络；
- 4 个两阶段配置，每个包含 2 个 stage 网络：共 8 个新 stage 网络。

如果 Round 1 已经显示清楚的局部最优，就不再增加模型，此时新增总量为：

```text
7 个训练配置
17 个新 stage 网络
其中只有 4 个配置继续训练到 400K
```

若三阶段还需要搜索第二个边界，中心 checkpoint 直接复用，只新增左右两个点。因此推荐方案的严格上限为：

```text
9 个新训练配置
23 个新 stage 网络
仍然只有 4 个配置继续训练到 400K
```

也就是说 `7 configs / 17 networks` 是基础计划，`9 configs / 23 networks` 是完成一次三阶段 coordinate refinement 的上限。没有必要为了填满上限而训练额外模型。

### 最省资源方案

如果 GPU 时间非常紧：

- 三阶段只训练理论预测点和一个邻居：2 个配置；
- 两阶段只训练理论预测、500/500、一个邻居：3 个配置；
- 共 5 个新配置、12 个新 stage 网络；
- 每种 topology 只让最好的一个跑到 400K。

这个版本足以找到一个强配置，但论证“预测方法确实有效”会弱一些。

---

## 8. 需要生成多少 samples

按推荐方案估算新增生成量：

| 用途 | 计算 |
|---|---:|
| 50K checkpoint sanity | 最多 9 configs × 2K = 18K |
| 100K boundary ranking | 最多 9 configs × 10K = 90K |
| 4 个最终配置的 allocation screen | 4 × 6 allocations × 5K = 120K |
| screen top-2 复核 | 4 × 2 × 10K = 80K |
| 每个 topology top-2 正式 50K | 4 × 50K = 200K |
| CFG1.0 最终赢家 | 2 × 50K = 100K |

总新增约：

$$
18K+90K+120K+80K+200K+100K=608K\ \text{images}.
$$

这是执行完整 Round 2 的保守上限；若 Round 1 已经收敛，则回到约 584K。以下结果可以复用，从而减少实际生成量：

- `300/300/400` 的 50K DDPM CFG1.0/1.5 主结果；
- `100/300/600` 及 8×8 continued 的 50K 主结果；
- 现有 `80/80/90` 点恰好定义了主 FLOPs 预算 $B_0$；
- 旧 fixed-250 allocation sweep 可作为“fixed-NFE 排名”对照，但不能冒充 equal-FLOPs。

如果某个已有 50K 结果的 sampler、CFG、checkpoint 和 equal-FLOPs allocation 完全一致，就直接复用，不重新 sample。

---

## 9. 决策规则

为了避免看到结果后随意改变标准，提前固定：

1. 主指标：ImageNet-256 50K FID，CFG=1.5；
2. 主计算约束：$B_0=24.727043768320$ TFLOPs/image，不含 CFG duplication；
3. tie：FID 差小于 0.10 时，选择实测 latency 更低者；
4. 5K screen 只淘汰明显差的配置，差小于 0.25 必须进入 10K 复核；
5. 最终结论只使用 native ImageNet 数据训练的模型；
6. 所有 screen 使用相同 class-label 序列和 seed；
7. 同一 checkpoint 的 allocation search 固定 upsample=`noise_filling`；
8. DDPM 与 DPM2 分开优化，不跨 solver 宣称同一个 allocation 最优；
9. 训练 split 的比较必须使用各自 equal-FLOPs 优化后的 inference allocation，同时额外报告一个共同 allocation 作为控制；
10. 所有失败或缺失点保留为空，不插值。

---

## 10. 论文最终应出现的图表

### 图 1：训练边界预测

- 横轴：两阶段边界 $b$；
- 左纵轴：predictor score；
- 右纵轴：100K/400K FID；
- 标注 predicted boundary 与 empirical best。

### 图 2：三阶段 boundary heatmap

- 横轴：$b_1$；
- 纵轴：$b_2$；
- 背景：预测评分；
- 点：真正训练过的配置，颜色表示 FID。

### 图 3：Equal-FLOPs allocation

- 每个点 FLOPs 相同；
- 横轴：32×32 模型调用数；
- 颜色：16×16 调用数；
- 纵轴：FID；
- 对比 predicted 和 empirical best。

### 表 1：最终最优配置

| Topology | Train split | Sampler | Calls 8/16/32 | Updates 8/16/32 | FLOPs | Latency | FID |
|---|---|---|---|---|---:|---:|---:|

注意同时报告 `calls` 和 `updates`。DDPM 二者相同，DPM2 不相同。

---

## 11. 参考工作与本方法的关系

- Improved DDPM：说明 timestep importance sampling 应由 loss second moment 动态估计，而不是静态频谱决定；本项目已经实现该 sampler。
- Min-SNR：说明不同 timestep 存在 gradient conflict，因此训练难度不能只由数据能量解释。
- EDM 与 DPM-Solver：说明推理 schedule 应考虑数值参数化和 solver，而不是简单 uniform global timestep。
- Align Your Steps：说明最佳采样 schedule 是 dataset/model/solver-specific；真实分布上需要 Monte Carlo 或模型查询。
- Spectral Progressive Diffusion：提供“某个频率什么时候从噪声中变得可恢复”的二阶统计近似。本文借用这个思想，但使用 RDD 的真实 projection residual 和 drift，不把它错误地解释成 SGD 更新预算。

本文方案的关键区别是：

> 将 RDD 的分辨率边界选择和 solver 的步数选择拆成两个可测量的优化问题，再在相同 FLOPs 下联合评价。

---

## 12. 从理论到实验：直接运行顺序

### 12.1 用原生 ImageNet latent 产生边界先验

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion

python 2d/exp/estimate_rdd_boundaries.py \
  --data-path /scr/yuheng/DevOpsBench/multiscale_diffusion/imagenet/ILSVRC/Data/CLS-LOC/train \
  --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
  --num-images 10000 \
  --posterior mean \
  --snr-threshold 1.0 \
  --thresholds 0.5,1.0,2.0 \
  --bootstrap-replicates 200 \
  --output eval_results/rdd_boundary_prior.json
```

这个程序输出：

- `predicted_three_stage_finest_to_coarsest`；
- `predicted_two_stage_finest_to_coarsest`；
- `boundary_uncertainty_grid`；
- `recommended_uncertain_boundary`；
- `recommended_delta`；
- 一条已经填好上述数值的 `next_command`。

它只是数据先验；不要直接把它写成最终最优。如果 `predictor_order_valid_before_legal_projection=false`，必须在论文中说明 data-only predictor 没给出合法顺序。若预测合法但某一 stage 只有 50 timestep，应同时报告原始中心和“每阶段至少 100 timestep”的稳健对照，不能把二者混写。

#### 12.1.1 2026-08-13 实际初始三阶段预测

已经在原生 ImageNet training split 上实际运行：10,000 张图片、SD VAE posterior mean、真实 1,000-step linear beta schedule、$\tau\in\{0.5,1,2\}$、200 次 image bootstrap、50-step 搜索网格。运行只做 VAE 编码和统计，不训练模型。

这里有一个容易犯错、但非常关键的归一化：`16−8` 和 `32−16` 两个残差都铺回 32×32 网格后，前者只有 $16^2-8^2$ 个有效自由度，后者有 $32^2-16^2$ 个有效自由度。若直接对全部 32×32 像素取均方，窄频带会被人为稀释。因此用于 SNR crossing 的量是

$$
\widehat\sigma_j^2
=
\frac{\mathbb E\lVert B_jx_0\rVert_2^2}
{C\,d_j},
$$

其中 $d_{16\setminus8}=16^2-8^2$，$d_{32\setminus16}=32^2-16^2$。像素均方仍保留用于审计，但不用于两个不同秩的频带之间直接比较。

实际测得：

| 频带 | 每个有效系数的均方 | $\operatorname{SNR}=1$ 原始交点 | 50-step 网格区间 |
|---|---:|---:|---:|
| $32\setminus16$ | 0.29392 | 156 | 100–200，中心 150 |
| $16\setminus8$ | 0.44394 | 187 | 150–250，中心 200 |
| $32\setminus8$（两阶段） | 0.32392 | 163 | 100–200，中心 150 |

所以三阶段的原始数据先验为

$$
(b_1,b_2)=(150,200),
\qquad
(L_{32},L_{16},L_8)=(150,50,800).
$$

这不是说 `150/50/800` 已经最优，而是说：按这个简单的频带可恢复性标准，16×16 专属区间可能很短。由于 50 timestep 可能使中间模型训练样本不足，第一轮必须同时包含每阶段至少 100 timestep 的稳健近邻，例如 `100/100/800` 和 `150/100/750`。现有 `100/300/600`、`300/300/400` 则提供“较长 16×16 区间”的免费远距离对照。

两阶段的原始先验为 `150/850`（32×32/8×8）。它只负责确定 pilot search 的中心；最终边界仍由相同训练预算、相同采样协议下的 FID 和诊断指标决定。

### 12.2 构造七个短训练候选

把上一步输出代入，例如：

```bash
python 2d/exp/build_rdd_boundary_plan.py \
  --pred-three 150,50,800 \
  --pred-two 150,850 \
  --uncertain-boundary both \
  --delta 50 \
  --output eval_results/rdd_boundary_plan.json
```

当前实测结果下，脚本会打印 3 个新三阶段候选（`150/50/800`、`100/100/800`、`150/100/750`）和 4 个新两阶段候选（`150/850`、`100/900`、`200/800`、`500/500`）。每个命令最终调用：

```bash
STAGE_RESOLUTIONS=32,8 \
STAGE_LENGTHS=450,550 \
GPUS=0,1,2,3 \
MAX_TRAIN_STEPS=50000 \
bash 2d/exp/train_rdd_boundary_candidate.sh \
  2>&1 | tee logs/train_boundary_32_8_450_550_50k.log
```

50K 只排除明显失败；所有未失败候选续到 100K，之后每种 topology 只保留 top 2 到 400K。续训时设置 `RESUME_DIR`、`RESUME_STEP=0050000` 和新的 `MAX_TRAIN_STEPS=100000`。

### 12.3 在某个 checkpoint 上生成 equal-FLOPs allocations

先用 256 张图估计真实 checkpoint 的 coarse-to-fine local error density：

```bash
python 2d/exp/profile_rdd_local_error.py \
  --data-path /scr/yuheng/DevOpsBench/multiscale_diffusion/imagenet/ILSVRC/Data/CLS-LOC/train \
  --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
  --ckpt-dir checkpoints/RDD-Multistage-300-300-400 \
  --ckpt-step 0400000 \
  --resolutions 32,16,8 \
  --stage-lengths 300,300,400 \
  --solver ddpm \
  --num-images 256 \
  --output eval_results/profile_3stage_ddpm.json
```

读取输出的 `difficulty_cli`。它用 teacher-forced noisy states 做 step doubling；对 stochastic DDPM 使用确定性 posterior mean 来隔离离散化误差，所以它是候选生成 proxy，不是 FID 替代品。

三阶段 DDPM 主预算：

```bash
python 2d/exp/rdd_equal_flops_allocation.py \
  --resolutions 32,16,8 \
  --stage-lengths 300,300,400 \
  --architectures DiT-XL/2,DiT-L/2,DiT-B/2 \
  --baseline-steps 80,80,90 \
  --solver ddpm \
  --difficulty <上一步输出的三个逗号分隔数> \
  --output eval_results/allocation_3stage_ddpm.json
```

如果只想 smoke test，可以暂时使用 `--difficulty 1,1,1`；正式 predicted allocation 必须使用 profiler 的输出。

两阶段必须固定与主实验相同的绝对预算，而不是沿用一个任意总步数：

```bash
python 2d/exp/rdd_equal_flops_allocation.py \
  --resolutions 32,8 \
  --stage-lengths 450,550 \
  --architectures DiT-XL/2,DiT-B/2 \
  --baseline-steps 90,80 \
  --budget-tflops 24.727043768320 \
  --solver ddpm \
  --difficulty 1,1 \
  --output eval_results/allocation_2stage_ddpm.json
```

DPM2 要设置 `--solver dpm2 --budget-solver ddpm`；工具会按源码使用 `max(1,2n-2)` 个 model calls，而不是把 updates 当作 NFE。

`24.727T` 是不含 CFG duplication 的 canonical budget。CFG=1.5 会把每个候选的 conditional/unconditional batch 都乘 2，因此真实总计算是 `49.454T`；由于所有 stage 同乘 2，最优整数 allocation 不变。

### 12.4 生成 JSONL 并在四张卡上采样

```bash
python 2d/exp/build_rdd_allocation_configs.py \
  --allocation-json eval_results/allocation_3stage_ddpm.json \
  --ckpt-dir checkpoints/RDD-Multistage-300-300-400 \
  --ckpt-step 0400000 \
  --cfg-scales 1.5 \
  --output-root eval_results/equal_flops_samples \
  --config-out eval_results/equal_flops_3stage.jsonl

CONFIG=eval_results/equal_flops_3stage.jsonl \
TOTAL_SAMPLES=5000 \
GPUS=0,1,2,3 \
BATCH_SIZE=64 \
bash 2d/exp/run_rdd_allocation_search.sh \
  2>&1 | tee eval_results/logs/equal_flops_3stage_launcher.log
```

生成器采用和 sampler 一致的方向约定：`stage_lengths` 是 finest-to-coarsest，`num_steps` 是 coarse-to-fine。`sample_fid_unified.py` 现已同时支持 `32,16,8` 和直接 `32,8`，后者在 32×32 drift 中使用真实 `sample_scale=4`。
