# Posterior Upsample 推导：问题分析与正确公式

## 1. 问题定义

**场景**：多尺度扩散模型中，stage $c$ 完成去噪后得到干净预测 $\hat{x}_0^c$（coarse），
需要构造 stage $c-1$（fine）的起始噪声 $x_{T_c}^{c-1}$。

**算子定义**：
- $D \in \mathbb{R}^{m \times n}$：average pooling 下采样，$m = n/s^2$
- $U \in \mathbb{R}^{n \times m}$：nearest-neighbor 上采样（每个像素复制 $s^2$ 次）
- 性质：$DU = I_m$，$UD$ 是投影（将高分辨率投影到行空间）
- $P = I_n - UD$：零空间投影矩阵（$P^2 = P$, $PU = 0$, $DP = 0$）

**前向扩散**：
$$q(x_t \mid x_0) = \mathcal{N}(x_t \mid \sqrt{\bar\alpha_t}\, x_0, (1 - \bar\alpha_t) I)$$

记 $a = \sqrt{\bar\alpha_{T_c}}$, $b = 1 - \bar\alpha_{T_c}$，则 $a^2 + b = 1$。

**目标**：给定 $\hat{x}_0^c$，采样 $x_{T_c}^{c-1}$ 使得：
1. $D\, x_{T_c}^{c-1}$ 与 coarse 尺度一致
2. $x_{T_c}^{c-1}$ 的边缘分布接近 $q(x_{T_c}^{c-1} \mid x_0^{c-1})$ 的正确形式

---

## 2. Constant Upsample（当前有效的 baseline）

Constant upsample 的做法：
$$x_{T_c}^{c-1} = a \cdot U\hat{x}_0^c + \sqrt{b} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_n)$$

这等价于 $q(x_{T_c}^{c-1} \mid x_0^{c-1} = U\hat{x}_0^c)$，即假设高分辨率干净图就是
nearest-neighbor 上采样的版本，然后对其做标准前向加噪。

**性质**：
- 行空间：$D\, x_{T_c}^{c-1} = a \cdot DU\hat{x}_0^c + \sqrt{b} \cdot D\epsilon = a \cdot \hat{x}_0^c + \sqrt{b} \cdot D\epsilon$
- 零空间：$P\, x_{T_c}^{c-1} = a \cdot PU\hat{x}_0^c + \sqrt{b} \cdot P\epsilon = \sqrt{b} \cdot P\epsilon$

**问题**：行空间中的噪声 $D\epsilon$ 是独立采样的，和 coarse stage 的实际演化无关。

---

## 3. 论文中推导的错误

论文 Appendix A.4 的推导有以下关键错误：

### 错误 1：$I_n - D^T D$ 不是投影矩阵

对于 average pooling，$D$ 每行是 $\frac{1}{s^2}$ 乘以对应 patch 的求和。因此：
$$(D^T D)_{ij} = \begin{cases} \frac{1}{s^2} & \text{if } i,j \text{ in same patch} \\ 0 & \text{otherwise}\end{cases}$$

验证幂等性：$(D^T D)^2 = \frac{1}{s^2} D^T D \neq D^T D$（当 $s > 1$ 时）。

所以 $I_n - D^T D$ **不是投影矩阵**，论文中 $(I_n - D^T D)^+ = I_n - D^T D$ 是**错的**。

### 错误 2：先验协方差 $B = b(I_n - D^T D)$ 的问题

论文声称 $B$ 在行空间方向方差为零。检验：对任意 $v = D^T w$（行空间向量），
$$v^T B v = b \cdot w^T D (I_n - D^T D) D^T w = b \cdot w^T (DD^T - DD^TD D^T) w$$
$$= b \cdot w^T \left(\frac{1}{s^2}I_m - \frac{1}{s^4}I_m\right) w = \frac{b(s^2 - 1)}{s^4} \|w\|^2 \neq 0$$

所以 $B$ 在行空间方向**有非零方差**，违背了"低频完全确定"的物理直觉。

### 错误 3：$D^T B^{-1} D = 0$ 的推导

论文 Eq.(A.33) 声称 $(D^T D)^2 = D^T D$（幂等），这在 $s > 1$ 时是**错的**。

---

## 4. 正确的投影算子

正确的行空间投影是 $UD$（不是 $D^T D$），零空间投影是 $P = I_n - UD$。

性质验证：
- $(UD)^2 = U(DU)D = UID = UD$ ✓ 幂等
- $P^2 = (I - UD)^2 = I - 2UD + (UD)^2 = I - 2UD + UD = I - UD = P$ ✓
- $P \cdot UD = 0$ ✓ 正交分解

---

## 5. 正确的 Posterior Upsample 推导

### 5.1 问题重新表述

假设真实的高分辨率 $x_0^{c-1}$ 满足约束 $D x_0^{c-1} = \hat{x}_0^c$。

我们需要采样 $x_{T_c}^{c-1} \sim q(x_{T_c}^{c-1} \mid D x_0^{c-1} = \hat{x}_0^c)$。

### 5.2 分解

任意高分辨率向量可分解为行空间和零空间：
$$x = UD(x) + P(x)$$

对 $x_{T_c}^{c-1} = a x_0^{c-1} + \sqrt{b}\, \epsilon$, $\epsilon \sim \mathcal{N}(0, I_n)$：

**行空间分量**：
$$D\, x_{T_c}^{c-1} = a \cdot D x_0^{c-1} + \sqrt{b} \cdot D\epsilon = a\, \hat{x}_0^c + \sqrt{b} \cdot D\epsilon$$

所以行空间由 coarse 尺度的前向过程完全确定：
$$UD\, x_{T_c}^{c-1} = U(a\, \hat{x}_0^c + \sqrt{b} \cdot D\epsilon) = a \cdot U\hat{x}_0^c + \sqrt{b} \cdot UD\epsilon$$

**零空间分量**：
$$P\, x_{T_c}^{c-1} = a \cdot P x_0^{c-1} + \sqrt{b} \cdot P\epsilon$$

这里 $P x_0^{c-1}$ 是**未知的高频信息**（我们只知道低频 $\hat{x}_0^c$），
$P\epsilon$ 是独立噪声在零空间的投影。

### 5.3 关键问题：$P x_0^{c-1}$ 的先验是什么？

RDD 训练时的约束 $D x_0^{c-1} = x_0^c$ **一直成立**（训练数据就是 $x_0^c = D(x_0^{c-1})$）。
但这只约束了行空间，**不意味着** $P x_0^{c-1} = 0$。

真实图片的 VAE latent 在零空间有非零高频能量。设训练数据中：
$$P x_0^{c-1} \sim \mathcal{N}(0, \sigma_h^2 \cdot P)$$

其中 $\sigma_h^2$ 是 latent 高频分量的方差。

**$\sigma_h$ 的自然取值**：VAE latent 经 scaling factor 0.18215 归一化后近似 $\mathcal{N}(0, I)$，
所以零空间分量的方差也近似为 1，即 **$\sigma_h \approx 1$**。

### 5.4 两种情况

**(A) $\sigma_h = 0$（假设 $P x_0^{c-1} = 0$，即无高频）**：

则 $P\, x_{T_c}^{c-1} = \sqrt{b} \cdot P\epsilon$。

合在一起：
$$x_{T_c}^{c-1} = a \cdot U\hat{x}_0^c + \sqrt{b} \cdot UD\epsilon + \sqrt{b} \cdot P\epsilon$$
$$= a \cdot U\hat{x}_0^c + \sqrt{b}\, \epsilon$$

**这和 constant upsample 完全一样！** 因为 $UD\epsilon + P\epsilon = (UD + P)\epsilon = I\epsilon = \epsilon$。

但这个假设是**错的**——真实训练数据中高频分量非零。

**(B) $\sigma_h = 1$（正确的先验：latent 各方向方差为 1）**：

零空间分量为：
$$P\, x_{T_c}^{c-1} = a \cdot P x_0^{c-1} + \sqrt{b} \cdot P\epsilon$$

由 $P x_0^{c-1}$ 和 $P\epsilon$ 独立：
$$\text{Var}(P x_{T_c}^{c-1}) = a^2 \cdot \sigma_h^2 \cdot P + b \cdot P = (a^2 + b) \cdot P = 1 \cdot P = P$$

（因为 $a^2 + b = 1$。）

所以零空间分量：$P(x_{T_c}^{c-1}) \sim \mathcal{N}(0, P)$。

采样：$P(x_{T_c}^{c-1}) = P z$, $z \sim \mathcal{N}(0, I_n)$。

**合起来**：
$$x_{T_c}^{c-1} = \underbrace{U(a\, \hat{x}_0^c + \sqrt{b}\, D\epsilon)}_{\text{行空间}} + \underbrace{Pz}_{\text{零空间}}$$

其中 $\epsilon, z \sim \mathcal{N}(0, I_n)$ 独立。

**这和 constant upsample 不同！** Constant 的零空间方差是 $b \cdot P$（约 0.03），
而 posterior（$\sigma_h=1$）的零空间方差是 $P$（约 1.0）。Posterior 在零空间注入了更多噪声。

---

## 6. 核心结论

### 结论 1：$P x_0^{c-1} = 0$ 是错误的假设

RDD 的约束 $D x_0^{c-1} = x_0^c$ **不等于** $P x_0^{c-1} = 0$。
约束只固定了行空间（均值），零空间（patch 内细节）仍然自由。
真实 latent 有高频能量，$\sigma_h \approx 1$。

### 结论 2：正确的 Posterior Upsample 确实存在且不同于 Constant

当 $\sigma_h = 1$ 时，posterior upsample 的公式为：
$$x_{T_c}^{c-1} = U(a\, \hat{x}_0^c + \sqrt{b}\, D\epsilon) + Pz$$

其中 $\epsilon, z \sim \mathcal{N}(0, I_n)$ 独立。

与 constant upsample $a\, U\hat{x}_0^c + \sqrt{b}\, \epsilon$ 的区别：
- **行空间方差相同**：都是 $b \cdot UD$
- **零空间方差不同**：constant 是 $b \cdot P$，posterior 是 $P$（大了 $1/b \approx 33$ 倍）

### 结论 3：直觉解释

Constant upsample 假设"高分辨率干净图就是低分辨率的 NN 上采样（无高频）"，
所以零空间噪声量只有前向过程的 $\sqrt{b}$。

Posterior upsample 假设"高分辨率干净图有独立的高频信息（$\sigma_h = 1$）"，
经过前向加噪后，零空间的总方差 = $a^2 \cdot 1 + b = 1$，
所以零空间噪声量是 $1$（方差保持）。

### 结论 4：为什么之前效果差？

之前代码的零空间是 $Pz$（方差 = $P$），行空间是 $U(a\hat{x}_0^c + \sqrt{b}\epsilon_c)$。
**行空间方差 = $b \cdot UD$，零空间方差 = $P$**。

问题在于：fine-level 模型训练时在 $t = T_c - 1$（local 最大步）时看到的输入是
$\sqrt{\bar\alpha_{T_c}} x_0^{c-1} + \sqrt{1-\bar\alpha_{T_c}} \epsilon$，其**总方差是 $I$**。

但 posterior upsample 给出的总方差是：$b \cdot UD + P = b \cdot UD + (I - UD) = I - (1-b)UD = I - a^2 \cdot UD$。

这**不等于** $I$！（差了 $a^2 \cdot UD$ 这一项）

而 constant upsample 的总方差是 $b \cdot I$，加上信号项 $a^2 \cdot UU^T$... 

等等，让我重新仔细算：

**Constant upsample 的总方差**：
$$x = a\, U\hat{x}_0^c + \sqrt{b}\, \epsilon$$
信号项 $a\, U\hat{x}_0^c$ 是确定性的（给定 $\hat{x}_0^c$），噪声方差 = $b \cdot I$。

**Posterior upsample 的总方差**：
$$x = U(a\, \hat{x}_0^c + \sqrt{b}\, D\epsilon) + Pz$$
信号项 $a\, U\hat{x}_0^c$ 是确定性的。
噪声项 = $\sqrt{b}\, UD\epsilon + Pz$。
方差 = $b \cdot UD\, \text{Cov}(\epsilon)\, (UD)^T + P\, \text{Cov}(z)\, P^T$
     = $b \cdot UD + P$（利用 $(UD)^2 = UD$, $P^2 = P$）
     = $b\, UD + I - UD = I - (1-b)UD = I - a^2\, UD$

**所以 posterior 的噪声方差 = $I - a^2\, UD$，不是 $b\, I$！**

训练时模型在 $t=T_c-1$ 时期望的总输入方差（条件于 $x_0$）= $b \cdot I = (1-a^2) \cdot I$。

Posterior 给出的方差 = $I - a^2\, UD$。

差异 = $(I - a^2 UD) - (1-a^2)I = a^2(I - UD) = a^2 P$。

**Posterior 在零空间上方差多了 $a^2$！** 这就是问题所在：模型训练时在 $t = T_c - 1$
时看到的零空间噪声方差只有 $b$，但 posterior 给了方差 $1 = b + a^2$，多了 $a^2 \approx 0.97$。

---

## 7. 为什么 Posterior 和 Constant 在实践中等价

### 7.1 核心论点

虽然 posterior upsample 从贝叶斯角度给出了"正确的"条件分布 $q(x_T^{c-1} \mid \hat{x}_0^c)$，
但在 RDD 的训练框架下，**constant upsample 也是正确的**。原因如下：

### 7.2 fine-level 模型的去噪能力

Fine-level 模型在训练时对**所有** $t \in [0, T_{local})$ 都做过去噪。
特别是在 $t = T_{local}-1$ 时，模型看到的输入是：
$$x_t = a\, x_0^{c-1} + \sqrt{b}\, \epsilon$$

其中 $x_0^{c-1}$ 包含高频分量 $P x_0^{c-1} \neq 0$。

模型的任务是从 $x_t$ 预测 $x_0^{c-1}$（包括高频）。它学会了：
- 从行空间 $UD x_t$ 恢复低频信号
- 从零空间 $P x_t = a P x_0^{c-1} + \sqrt{b} P\epsilon$ 恢复高频信号

### 7.3 Constant upsample 的正确性

Constant upsample 给出：
$$x_T^{c-1} = a\, U\hat{x}_0^c + \sqrt{b}\, \epsilon$$

这等价于：$x_T^{c-1} = a\, x_0^{c-1} + \sqrt{b}\, \epsilon$，其中 $x_0^{c-1} := U\hat{x}_0^c$（$P x_0^{c-1} = 0$）。

模型会怎么处理？它在去噪的第一步把这个当作 $q(x_t \mid x_0 = U\hat{x}_0^c)$，
预测 $\hat{x}_0 \approx U\hat{x}_0^c$（正确的低频 + 零高频）。
然后在后续步骤中逐渐生成高频细节——**这就是扩散模型本来的工作方式**。

模型**不需要**在 $t = T_{local}-1$ 时就看到高频信息。它会在 $t$ 减小的过程中逐步补充。

### 7.4 Posterior upsample 的问题

Posterior 给出的零空间方差 = $1$（方差保持），但模型在固定 $t = T_{local}-1$ 时
期望零空间噪声标准差 = $\sqrt{b}$（因为 $t$ 告诉模型"当前噪声水平是 $\sqrt{b}$"）。

如果实际零空间噪声是 $1 \gg \sqrt{b}$，模型会**低估零空间的噪声量**，
错误地认为零空间中的大幅波动是"信号"而非噪声，导致第一步预测的 $\hat{x}_0$ 在高频上出现伪影。

### 7.5 论文中的表述

**Posterior upsample 和 constant upsample 的关系可以这样写**：

> The Bayesian posterior $q(x_T^{c-1} \mid D x_0^{c-1} = \hat{x}_0^c)$ marginalizes over
> the unknown high-frequency component $Px_0^{c-1}$. However, since the fine-level
> denoising model is trained with the standard forward process $q(x_t \mid x_0)$
> which conditions on the **full** $x_0^{c-1}$ (including high-frequency), the model
> expects noise variance $b\,I$ at timestep $T_c$.
>
> The constant upsample $x_T^{c-1} = a\,U\hat{x}_0^c + \sqrt{b}\,\epsilon$ respects this
> expectation by treating $U\hat{x}_0^c$ as the "clean signal" and adding isotropic noise
> of the correct magnitude. The denoising model then progressively generates
> high-frequency details during the reverse process at finer resolution,
> which is exactly its trained capability.
>
> In contrast, the marginalized posterior places variance $1$ (instead of $b \approx 0.03$)
> in the null space, creating an out-of-distribution input for the denoising model.
> Empirically, this leads to significantly worse generation quality.
>
> Therefore, in the multistage framework where each stage is independently trained
> with standard $q(x_t \mid x_0)$, the constant upsample is both simpler and optimal.

### 7.6 等价性的精确陈述

**定理**：在以下条件下，constant upsample 是最优的跨尺度初始化策略：
1. Fine-level 模型按标准前向过程 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t)I)$ 训练
2. 各 stage 独立训练（无跨尺度噪声耦合）
3. 采样时从 $t = T_{local}-1$ 开始去噪

**证明**：模型的 score function $s_\theta(x_t, t)$ 在 $t = T_{local}-1$ 时学到了
$\nabla_{x_t} \log q(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{b}}$。
要让模型正确工作，输入 $x_t$ 必须满足 $x_t = ax_0 + \sqrt{b}\epsilon$（某个 $x_0$ 加方差为 $bI$ 的噪声）。
Constant upsample 取 $x_0 = U\hat{x}_0^c$ 满足此条件。$\square$

---

## 8. 最终结论

| 方法 | 公式 | 零空间方差 | 匹配模型期望 | 效果 |
|------|------|-----------|-------------|------|
| Constant | $a U\hat{x}_0^c + \sqrt{b}\epsilon$ | $b$ | ✓ | 好 |
| Posterior | $U(a\hat{x}_0^c + \sqrt{b}D\epsilon) + Pz$ | $1$ | ✗ | 差 |

**Posterior upsample 数学上是 $q(x_T \mid D x_0 = \hat{x}_0^c)$ 的正确采样**，
但它不适合当前训练框架。Constant upsample 虽然基于 "$P x_0 = 0$" 的近似，
但它保证了给模型的输入在方差结构上匹配训练分布 $q(x_t \mid x_0)$，
模型可以在后续去噪步骤中自己恢复高频信息。

**两者本质上是对"未知高频"的不同处理策略**：
- Posterior：把高频的不确定性**前置**到初始噪声中（方差增大）
- Constant：把高频的生成任务**后置**到去噪过程中（标准方差，让模型逐步生成）

在独立训练的 stage 框架下，后者与模型的训练目标一致，是正确选择。

---

## 9. 如果未来想让 Posterior 起作用

需要修改**训练过程**，让 fine-level 模型在训练时也看到 posterior 分布的输入：
1. 训练时不是对 $x_0^{c-1}$ 做标准 $q$-sample，
   而是先对 $x_0^{c-1}$ 做 $D$ 得到 $x_0^c$，再用 posterior upsample 构造输入
2. 这样模型训练时就习惯了零空间方差为 $1$ 的输入
3. 采样时使用 posterior upsample 就不会 OOD

但这需要大改训练代码，且未必比 constant 有优势（因为模型最终还是要自己生成高频）。
