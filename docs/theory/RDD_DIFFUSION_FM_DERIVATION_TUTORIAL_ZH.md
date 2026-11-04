# RDD Diffusion 与 RDD Flow Matching：从零开始的逐步推导与 Novelty 路线图

> 目标读者：知道神经网络和基本微积分，但不默认熟悉扩散模型、随机过程、Flow Matching 或数值 ODE。
>
> 本文对应当前仓库中的实现与论文，而不是一份脱离代码的泛化介绍。主要参考文件：
>
> - `paper/main.tex`
> - `diffusion/gaussian_diffusion_with_drift.py`
> - `diffusion/flow_matching_with_drift.py`
> - `2d/train_RDD_imagenet256_unified.py`
> - `2d/train_RDD_FM_imagenet256_unified.py`
>
> 最后核对日期：2026-08-10。

## 导航

- [Part I：需要的数学基础](#part-i需要的数学基础)
- [Part II：从标准 DDPM 推导 RDD Diffusion](#part-ii从标准-ddpm-推导-rdd-diffusion)
- [Part III：Flow Matching 背景](#part-iiiflow-matching-背景)
- [Part IV：RDD Flow Matching 的完整推导](#part-ivrdd-flow-matching-的完整推导)
- [Part V：RDD-DDPM 与 RDD-FM 怎样公平比较](#part-vrdd-ddpm-与-rdd-fm-怎样公平比较)
- [Part VI：怎样把数学变成可防守的 novelty](#part-vi怎样把-rdd-的数学变成真正可防守的-novelty)
- [Part VII：实验与证明清单](#part-vii建议执行的实验与证明清单)
- [Part VIII：常见问题](#part-viii常见问题与自检)
- [Part IX：符号表](#part-ix符号表)
- [Part X：建议阅读顺序](#part-x建议阅读顺序)

当前应以 `paper/main.tex`、实际代码和本文档为准。仓库中部分较早的 survey/proposal 仍包含“RDD 是唯一连续 Markov chain”“PyramidalWan 不兼容高阶 solver”等已经被后续审计否定的表述，不能继续当作论文事实引用。

---

## 0. 先读结论：RDD 到底做了什么？

RDD 的核心不是“把低分辨率图像直接当作另一个训练样本”，而是改变从数据到噪声、或者从噪声到数据的**概率路径**。

在一个分辨率阶段中，令

\[
G(x)=\operatorname{UP}(\operatorname{DOWN}(x))
\]

表示“先降分辨率，再升回原形状”。定义同形状残差

\[
d(x)=G(x)-x.
\]

如果 \(G\) 是低通投影，那么 \(d(x)\) 是被低通操作移除的高频部分的负数。

RDD-DDPM 把这个残差放进 VP diffusion 的条件均值：

\[
x_t
=
\sqrt{\bar\alpha_t}\bigl[x_0+r_t d(x_0)\bigr]
+\sqrt{1-\bar\alpha_t}\,\epsilon.
\]

RDD-FM 则选择一条从噪声到数据的条件路径：

\[
x_t=t x_1+(1-t)\epsilon+b_k(t)d_k(x_1).
\]

二者共享的思想是：

1. 用同状态空间残差表示分辨率变换；
2. 让该残差随时间逐渐加入或移除；
3. 让模型先处理粗结构，再补充细节；
4. 在不同分辨率阶段使用同一个 DiT；
5. 在边界处通过与训练边缘分布一致的噪声补全连接阶段。

但是需要立即记住三点：

1. RDD-DDPM 和 RDD-FM 的训练目标不同：前者预测 effective noise，后者预测 velocity。
2. 当前多阶段 FM 的包络不是所有阶段都简单等于全局 \(t(1-t)\)。实际是
   \[
   b_k(t)=\frac{t(s_{k+1}-t)}{s_{k+1}-s_k}.
   \]
3. “加入 drift”“推导 Gaussian posterior”“写出 SDE”主要保证方法正确，不自动构成强 novelty。真正有潜力的理论问题在第 10 节。

---

# Part I：需要的数学基础

## 1. 向量、图像和随机变量

数学上可以把一张 latent image 看成一个长向量：

\[
x\in\mathbb R^n.
\]

例如一个 \(4\times32\times32\) latent 可以展平成 \(n=4096\) 维向量。卷积、pooling 和 upsampling 虽然在代码里保留张量形状，但推导时可以把它们当作矩阵作用在向量上。

大写 \(X\) 通常表示随机变量，小写 \(x\) 表示它的一次具体取值。本文为了与代码和论文一致，经常使用小写 \(x_t\)，但要记住训练时它是随机的。

### 1.1 期望

随机变量 \(X\) 的期望写作

\[
\mathbb E[X].
\]

给定另一个随机变量 \(Y=y\) 后的条件期望是

\[
\mathbb E[X\mid Y=y].
\]

条件期望可以理解为：已知观测 \(Y=y\) 后，对未知 \(X\) 的均方误差最优预测。

也就是说，若要寻找函数 \(f\) 最小化

\[
\mathbb E\|f(Y)-X\|^2,
\]

理论上的最优解为

\[
f^*(Y)=\mathbb E[X\mid Y].
\]

这个事实是 diffusion noise prediction 和 Flow Matching velocity regression 的共同基础。

### 1.2 协方差

向量随机变量的协方差为

\[
\operatorname{Cov}(X)
=
\mathbb E[(X-\mathbb E X)(X-\mathbb E X)^\top].
\]

若

\[
X\sim\mathcal N(\mu,\Sigma),
\]

表示 \(X\) 是均值为 \(\mu\)、协方差为 \(\Sigma\) 的多元 Gaussian。

特别地，

\[
\epsilon\sim\mathcal N(0,I)
\]

表示每一维都是独立标准 Gaussian。

## 2. 线性算子、仿射算子和投影

### 2.1 线性算子

若算子 \(P\) 满足

\[
P(ax+by)=aP(x)+bP(y),
\]

则称 \(P\) 为线性算子。

average pooling、nearest-neighbor repeat、bilinear interpolation 都是线性操作。神经网络 encoder、量化和带激活函数的 transform 通常不是线性的。

仿射函数是

\[
G(x)=Px+c.
\]

它比线性函数多了一个常量偏置 \(c\)。

### 2.2 RDD 使用的 block projection

令 \(D\) 表示 block average pooling，\(U\) 表示把每个低分辨率值重复回对应 block。则

\[
DU=I,
\qquad
P=UD.
\]

进一步有

\[
P^2=UDUD=U(DU)D=UD=P.
\]

满足 \(P^2=P\) 的算子称为幂等投影。对当前的 average/repeat 配对，\(P\) 还是对称的，因此可以看成 Euclidean 空间中的正交投影。

任何 \(x\) 都可以分解为

\[
x=Px+(I-P)x.
\]

记

\[
x_\ell=Px,
\qquad
x_h=(I-P)x.
\]

这里：

- \(x_\ell\) 是每个 block 内的常量部分，可解释为粗分辨率/低频子空间；
- \(x_h\) 是 block 内变化，可解释为被降采样丢掉的细节子空间。

RDD residual 为

\[
d(x)=Px-x=-(I-P)x=-x_h.
\]

因此

\[
x+d(x)=Px=x_\ell.
\]

这就是“把图像推向较粗分辨率”的精确含义。

### 2.3 为什么 linear 与 nonlinear 的区别重要？

对线性 \(d\)，条件期望可以交换：

\[
d(\mathbb E[X\mid Y])
=
\mathbb E[d(X)\mid Y].
\]

对一般非线性 \(d\)，通常不成立：

\[
d(\mathbb E[X\mid Y])
\neq
\mathbb E[d(X)\mid Y].
\]

这会直接决定 RDD 是否能只靠一个输出头恢复 reverse dynamics，后文会完整推导。

## 3. 导数、ODE、SDE 和 Markov 性

### 3.1 ODE

常微分方程写成

\[
\frac{dx_t}{dt}=v(x_t,t).
\]

给定起点 \(x_0\)，ODE 在适当条件下确定一条轨迹。Flow Matching 学习的就是速度场 \(v_\theta(x,t)\)。

### 3.2 SDE

随机微分方程写成

\[
dX_t=f(X_t,t)dt+g(t)dW_t.
\]

其中 \(W_t\) 是 Brownian motion，\(dW_t\) 表示无限小 Gaussian 随机增量。与 ODE 不同，同一起点的 SDE 可以产生多条随机轨迹。

### 3.3 Markov 性

Markov 过程满足：知道当前状态后，未来不再需要更早的历史。

离散时间写成

\[
q(x_t\mid x_{t-1},x_{t-2},\ldots,x_0)
=q(x_t\mid x_{t-1}).
\]

需要特别小心：RDD 的 residual \(d(x_0)\) 依赖未知 clean sample。因此严格写法通常是

\[
q(x_t\mid x_{t-1},x_0),
\]

它是**给定 \(x_0\) 的条件过程**，或者是扩充状态 \((x_t,x_0)\) 上的 Markov 过程。仅在 \(x_t\) 上，不应未经证明就声称是显式 autonomous Markov chain。

---

# Part II：从标准 DDPM 推导 RDD Diffusion

## 4. 标准 DDPM 从零开始

### 4.1 一步加噪

定义小的 noise schedule \(\beta_t\in(0,1)\)，并令

\[
\alpha_t=1-\beta_t.
\]

标准 DDPM forward transition 是

\[
q(x_t\mid x_{t-1})
=
\mathcal N(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I).
\]

等价采样形式为

\[
x_t
=
\sqrt{\alpha_t}x_{t-1}
+\sqrt{1-\alpha_t}\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
\]

### 4.2 从 \(x_0\) 直接采样 \(x_t\)

定义累计乘积

\[
\bar\alpha_t=\prod_{i=1}^t\alpha_i.
\]

递推展开可得

\[
x_t
=
\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

因此

\[
q(x_t\mid x_0)
=
\mathcal N
\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
\]

训练时不必真的迭代 \(t\) 次；随机采一个 \(t\) 和一个 \(\epsilon\)，即可直接构造 \(x_t\)。

### 4.3 SNR

信号标准差系数为 \(\sqrt{\bar\alpha_t}\)，噪声标准差系数为 \(\sqrt{1-\bar\alpha_t}\)。因此定义

\[
\operatorname{SNR}(t)
=
\frac{\bar\alpha_t}{1-\bar\alpha_t}.
\]

当 \(t\) 很小时，通常 SNR 较大；当 \(t\) 接近终点时，SNR 较小。

### 4.4 为什么预测 noise？

训练模型预测构造 \(x_t\) 使用的 \(\epsilon\)：

\[
\mathcal L_\epsilon
=
\mathbb E\|\epsilon_\theta(x_t,t)-\epsilon\|^2.
\]

由

\[
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon
\]

可恢复 clean estimate：

\[
\hat x_0
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
\]

## 5. RDD-DDPM 的阶段与残差

把 forward noise time 划分成

\[
0=T_0<T_1<\cdots<T_C=T.
\]

阶段 \(c\) 的范围为

\[
t\in[T_c,T_{c+1}],
\qquad
\Delta T^c=T_{c+1}-T_c.
\]

定义阶段内部归一化进度

\[
r_t^c
=
\frac{t-T_c}{\Delta T^c}.
\]

所以

\[
r_{T_c}^c=0,
\qquad
r_{T_{c+1}}^c=1.
\]

阶段 clean target 为 \(x_0^c\)，下一个更粗目标 lift 回相同形状后为 \(G_c(x_0^c)\)。定义

\[
d^c
=
G_c(x_0^c)-x_0^c.
\]

阶段开始时，signal target 是 \(x_0^c\)；阶段结束时，signal target 变成

\[
x_0^c+d^c=G_c(x_0^c).
\]

## 6. RDD 一步 transition 为什么长这样？

我们希望第 \(t\) 步的条件均值满足

\[
m_t
=
\sqrt{\bar\alpha_t}
\left(x_0^c+r_t^c d^c\right).
\]

假设一步 transition 写成

\[
x_t
=
\sqrt{\alpha_t}x_{t-1}
+a_t d^c
+\sqrt{1-\alpha_t}\epsilon_t.
\]

均值必须满足

\[
m_t=\sqrt{\alpha_t}m_{t-1}+a_t d^c.
\]

代入期望的 \(m_t\)：

\[
\sqrt{\bar\alpha_t}(x_0+r_t d)
=
\sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}
(x_0+r_{t-1}d)+a_t d.
\]

因为

\[
\sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}
=
\sqrt{\bar\alpha_t},
\]

所以

\[
a_t
=
\sqrt{\bar\alpha_t}(r_t-r_{t-1}).
\]

对线性 stage envelope，

\[
r_t-r_{t-1}=\frac1{\Delta T^c},
\]

得到

\[
a_t=\frac{\sqrt{\bar\alpha_t}}{\Delta T^c}.
\]

因此当前 RDD transition 是

\[
q(x_t\mid x_{t-1},x_0)
=
\mathcal N\left(
\sqrt{\alpha_t}x_{t-1}
+\frac{\sqrt{\bar\alpha_t}}{\Delta T^c}d^c(x_0),
(1-\alpha_t)I
\right).
\]

这里刻意写成 \(q(x_t\mid x_{t-1},x_0)\)，因为 \(d^c\) 是 clean-conditioned residual。

### 6.1 一般 envelope

如果不使用线性 \(r_t\)，正确的一步系数必须改成

\[
a_t=\sqrt{\bar\alpha_t}(r_t-r_{t-1}).
\]

不能只改变训练 marginal 中的 \(r_t\)，却继续使用原来 \(1/\Delta T\) 的 transition，否则 marginal 不再一致。

## 7. RDD forward marginal 的逐步推导

先忽略 stage 下标。第一步：

\[
x_1
=
\sqrt{\alpha_1}x_0
+\sqrt{\bar\alpha_1}(r_1-r_0)d
+\sqrt{1-\alpha_1}\epsilon_1.
\]

若 \(r_0=0\)，则

\[
x_1
=
\sqrt{\bar\alpha_1}(x_0+r_1d)
+\sqrt{1-\bar\alpha_1}\bar\epsilon_1.
\]

假设第 \(t-1\) 步有

\[
x_{t-1}
=
\sqrt{\bar\alpha_{t-1}}(x_0+r_{t-1}d)
+\sqrt{1-\bar\alpha_{t-1}}\bar\epsilon_{t-1}.
\]

代入一步 transition：

\[
\begin{aligned}
x_t
&=\sqrt{\alpha_t}x_{t-1}
+\sqrt{\bar\alpha_t}(r_t-r_{t-1})d
+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\bar\alpha_t}(x_0+r_{t-1}d)
+\sqrt{\bar\alpha_t}(r_t-r_{t-1})d
+\text{Gaussian noise}\\
&=\sqrt{\bar\alpha_t}(x_0+r_td)
+\text{Gaussian noise}.
\end{aligned}
\]

噪声方差递推为

\[
\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)
=1-\bar\alpha_t.
\]

因此

\[
\boxed{
q(x_t\mid x_0)
=
\mathcal N\left(
\sqrt{\bar\alpha_t}(x_0+r_t^c d^c),
(1-\bar\alpha_t)I
\right).
}
\]

阶段出口 \(r=1\) 时，均值为

\[
\sqrt{\bar\alpha_{T_{c+1}}}
G_c(x_0^c).
\]

## 8. Effective noise：为什么一个标准 DiT head 能训练 RDD？

从 marginal 采样式出发：

\[
x_t
=
\sqrt{\bar\alpha_t}(x_0+r_td)
+\sqrt{1-\bar\alpha_t}\epsilon.
\]

把它整理成标准 DDPM 外观：

\[
x_t
=
\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\tilde\epsilon_t,
\]

只要定义

\[
\boxed{
\tilde\epsilon_t
=
\epsilon
+\sqrt{\frac{\bar\alpha_t}{1-\bar\alpha_t}}
r_t d(x_0).
}
\]

这里 \(\tilde\epsilon_t\) 不是纯 Gaussian noise。它是

\[
\text{纯 Gaussian innovation}
+
\text{SNR 加权的 transform residual}.
\]

训练 loss 为

\[
\mathcal L_{mathrm{RDD}}
=
\mathbb E
\left[
\|\epsilon_\theta(x_t,t,c)-\tilde\epsilon_t\|^2
\right].
\]

### 8.1 为什么 standard-looking clean estimate 仍然成立？

定义

\[
\hat x_0
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t,c)}
{\sqrt{\bar\alpha_t}}.
\]

若模型达到 MSE 的 Bayes optimum，

\[
\epsilon_\theta^*(x_t,t)
=
\mathbb E[\tilde\epsilon_t\mid x_t].
\]

又因为逐样本恒等式

\[
x_t
=
\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\tilde\epsilon_t,
\]

对 \(x_t\) 取条件期望：

\[
x_t
=
\sqrt{\bar\alpha_t}\mathbb E[x_0\mid x_t]
+\sqrt{1-\bar\alpha_t}
\mathbb E[\tilde\epsilon_t\mid x_t].
\]

因此

\[
\boxed{
\hat x_0^*(x_t)
=
\mathbb E[x_0\mid x_t].
}
\]

这是 RDD 单头参数化最重要的数学事实之一。

### 8.2 Effective noise 与 pure noise 绝不能混淆

模型训练预测的是

\[
\tilde\epsilon
=
\epsilon+sqrt{\operatorname{SNR}}r_td.
\]

但是 score、probability-flow ODE 和 DPM 类 solver 需要围绕 transformed mean 的纯 Gaussian innovation：

\[
\boxed{
\hat\epsilon_{\mathrm{pure}}
=
\frac{x_t-sqrt{\bar\alpha_t}
(\hat x_0+r_t\hat d)}
{\sqrt{1-\bar\alpha_t}}.
}
\]

采样时正确顺序为：

1. 网络预测 \(\hat{\tilde\epsilon}\)；
2. 用标准外观公式恢复 \(\hat x_0\)；
3. 计算 \(\hat d=G(\hat x_0)-\hat x_0\)；
4. 从 \(x_t,\hat x_0,\hat d\) 中分离 \(\hat\epsilon_{\mathrm{pure}}\)；
5. 将 pure innovation 与显式 residual correction 分别交给 solver。

如果把 effective noise 直接当 pure noise 使用，会把 deterministic residual 计算两次或漏算一次。

## 9. RDD posterior：为什么多出一个 \(\gamma d\)？

考虑同一 stage 内 \(s<t\)。定义中心化变量

\[
y_u
=
x_u-\sqrt{\bar\alpha_u}r_u d.
\]

根据 RDD marginal，

\[
y_u
=
\sqrt{\bar\alpha_u}x_0
+\sqrt{1-\bar\alpha_u}\epsilon_u.
\]

也就是说，给定 \(x_0,d\) 后，\(y_u\) 完全服从标准 DDPM marginal。于是可以先对 \(y_s,y_t\) 使用标准 arbitrary-jump posterior，再把 deterministic shift 加回来。

标准 posterior 协方差为

\[
\boxed{
\Sigma_{s\mid t}
=
\frac{(\bar\alpha_s-\bar\alpha_t)(1-\bar\alpha_s)}
{\bar\alpha_s(1-\bar\alpha_t)}I.
}
\]

posterior mean 可整理为

\[
\boxed{
\mu_{s\mid t}
=A_{s,t}x_t+B_{s,t}x_0+\gamma_{s,t}d,
}
\]

其中

\[
A_{s,t}
=
\sqrt{\frac{\bar\alpha_t}{\bar\alpha_s}}
\frac{1-\bar\alpha_s}{1-\bar\alpha_t},
\]

\[
B_{s,t}
=
\frac{\bar\alpha_s-\bar\alpha_t}
{\sqrt{\bar\alpha_s}(1-\bar\alpha_t)},
\]

\[
\boxed{
\gamma_{s,t}
=
\sqrt{\bar\alpha_s}r_s
-
\frac{\bar\alpha_t(1-\bar\alpha_s)}
{\sqrt{\bar\alpha_s}(1-\bar\alpha_t)}r_t.
}
\]

当 \(d=0\) 时，直接退化成标准 DDPM posterior。

### 9.1 这是 conditional posterior，不是已经边缘化的 reverse kernel

上述精确结果是

\[
q(x_s\mid x_t,x_0).
\]

真正的边缘 reverse distribution 是

\[
q(x_s\mid x_t)
=
\int q(x_s\mid x_t,x_0)
q(x_0\mid x_t)dx_0.
\]

实践中用模型的 \(\hat x_0\) 和 \(\hat d\) 近似这个积分。不要把“给定 clean sample 的 Gaussian posterior”与“已经对数据 posterior 精确积分”混为一谈。

## 10. 单头何时足够？线性与非线性的真正边界

posterior mean 中需要

\[
\mathbb E[d(X_0)\mid X_t].
\]

单 effective-noise head 给出的是

\[
\hat x_0=\mathbb E[X_0\mid X_t].
\]

当前采样器使用

\[
d(\hat x_0)
=
d(\mathbb E[X_0\mid X_t]).
\]

若 \(d(x)=(P-I)x+c\) 是仿射函数，则

\[
d(\mathbb E[X_0\mid X_t])
=
\mathbb E[d(X_0)\mid X_t].
\]

因此 projection、linear blur、linear spectral mask 等情况下，单头 recovery 在 Bayes 均值意义下是闭合的。

若 \(d\) 非线性，二者一般不同。可以把误差定义为

\[
\Delta_d(x_t)
=
\mathbb E[d(X_0)\mid X_t=x_t]
-d(\mathbb E[X_0\mid X_t=x_t]).
\]

若 \(d\) 二阶可微，在 posterior 集中时可用二阶 Taylor 展开理解：

\[
\Delta_d
\approx
\frac12
\operatorname{Tr}
\left(
H_d(\mu_t)
\operatorname{Cov}(X_0\mid X_t)
\right),
\]

其中 \(\mu_t=\mathbb E[X_0\mid X_t]\)，\(H_d\) 表示 Hessian。曲率越大、clean posterior 方差越大，nonlinear plug-in bias 越严重。

因此“把 average pooling 轻松换成任意 transform”只能算代码层面的说法，不能自动继承单头理论。非线性 transform 需要：

- 第二个 residual head，直接预测 \(\mathbb E[d(X_0)\mid X_t]\)；或
- 预测 \(x\) 与 degradation 的联合充分统计量；或
- 使用 transform-specific inverse；或
- 给出 plug-in approximation 的可控误差界。

## 11. Stage transition 与 noise filling

在 coarse stage 完成后，需要进入 finer stage。 finer stage 训练边界 marginal 是

\[
x_{T_c}
\sim
\mathcal N
\left(
\sqrt{\bar\alpha_{T_c}}\,U(\hat x_0^{c+1}),
(1-\bar\alpha_{T_c})I
\right).
\]

因此 matched initialization 是

\[
\boxed{
x_{T_c}
=
\sqrt{\bar\alpha_{T_c}}U(\hat x_0^{c+1})
+\sqrt{1-\bar\alpha_{T_c}}\epsilon_{\mathrm{new}}.
}
\]

它保留 coarse information 于条件均值，并按训练时协方差重新采样随机部分。

“unique”必须带限定语：它是在**论文选择的 isotropic Gaussian initialization family** 中唯一匹配训练 marginal 的方案，不是所有可能 forward process 和 data prior 下的普适最优解。

## 12. RDD 的连续时间 SDE

令

\[
\alpha(t)=\sqrt{\bar\alpha_t},
\qquad
\sigma^2(t)=1-\bar\alpha_t.
\]

RDD conditional mean 为

\[
m(t)=\alpha(t)[x_0+r(t)d(x_0)].
\]

定义

\[
f(t)=\frac{d\log\alpha(t)}{dt}.
\]

对均值求导：

\[
\begin{aligned}
\dot m(t)
&=\dot\alpha(t)[x_0+r(t)d]
+\alpha(t)\dot r(t)d\\
&=f(t)m(t)+\alpha(t)\dot r(t)d.
\end{aligned}
\]

因此一个匹配该条件均值和方差的 conditional linear SDE 为

\[
\boxed{
dX_t
=
\left[f(t)X_t+\alpha(t)\dot r(t)d(X_0)\right]dt
+g(t)dW_t.
}
\]

协方差 \(\sigma^2(t)I\) 满足

\[
\frac{d\sigma^2}{dt}=2f(t)\sigma^2(t)+g^2(t),
\]

所以

\[
\boxed{
g^2(t)
=
\frac{d\sigma^2(t)}{dt}-2f(t)\sigma^2(t).
}
\]

对 VP SDE，\(f=-\beta/2\)，最终得到 \(g^2=\beta\)。

### 12.1 为什么这里仍有 Markov 注意事项？

drift 中仍有 \(d(X_0)\)。因此它首先是 clean-conditioned SDE。若要写只依赖当前状态的 marginal drift，需要条件期望

\[
\bar a(x,t)
=
\alpha(t)\dot r(t)
\mathbb E[d(X_0)\mid X_t=x].
\]

state-only probability-flow ODE 应使用这个 marginal drift，而不是某个不可见真实样本的 \(d(x_0)\)。线性 \(d\) 下，模型 clean posterior mean 可以闭合这个量；非线性情况下需要前述额外机制。

## 13. Probability-flow ODE 与 DPM correction

给定合适的 marginal drift，probability-flow ODE 的通用形式是

\[
\frac{dx}{dt}
=
f(t)x+\bar a(x,t)
-\frac12g^2(t)\nabla_x\log p_t(x).
\]

score 对应围绕 transformed mean 的 Gaussian innovation，因此需要 pure noise：

\[
\nabla_x\log p_t(x)
\approx
-\frac{\hat\epsilon_{\mathrm{pure}}}{\sigma(t)}.
\]

RDD DPM-Solver-1 当前使用更新

\[
\boxed{
x_s
=
\frac{\alpha_s}{\alpha_t}x_t
-\sigma_s(e^{\lambda_s-\lambda_t}-1)
\hat\epsilon_{\mathrm{pure}}
+\alpha_s(r_s-r_t)\hat d,
}
\]

其中

\[
\lambda_t=\log\frac{\alpha_t}{\sigma_t}.
\]

最后一项就是 residual-path correction。

若 \(\hat x_0=x_0\)、\(\hat d=d\)、\(\hat\epsilon_{\mathrm{pure}}=\epsilon\)，代入可验证更新恰好得到

\[
x_s=\alpha_s(x_0+r_sd)+\sigma_s\epsilon.
\]

当前 midpoint 版本会在中点重新估计 \(\hat x_0,\hat d\)。但“重新估计一次”不自动等于已经证明的二阶 DPM-Solver；需要 local/global truncation error 分析和 matched-NFE 实验。

---

# Part III：Flow Matching 背景

## 14. Flow Matching 想学习什么？

Flow Matching 的目标是学习 ODE

\[
\frac{dX_t}{dt}=v_\theta(X_t,t),
\]

使得：

\[
X_0\sim p_{\mathrm{noise}},
\qquad
X_1\sim p_{\mathrm{data}}.
\]

密度随 ODE 运动满足 continuity equation：

\[
\frac{\partial p_t(x)}{\partial t}
+\nabla\cdot[p_t(x)v_t(x)]=0.
\]

直觉上：一个位置附近的概率质量如何流入和流出，由速度场的散度决定。

## 15. Conditional Flow Matching

直接求 marginal velocity \(v_t(x)\) 很难。Conditional Flow Matching 先采样：

- 数据 \(x_1\sim p_{\mathrm{data}}\)；
- 噪声 \(\epsilon\sim\mathcal N(0,I)\)；
- 时间 \(t\)。

然后人为定义一条条件路径

\[
x_t=\phi_t(x_1,\epsilon).
\]

其条件速度是

\[
u_t(x_1,\epsilon)
=
\frac{d}{dt}\phi_t(x_1,\epsilon).
\]

训练

\[
\mathcal L_{\mathrm{CFM}}
=
\mathbb E
\|v_\theta(x_t,t)-u_t(x_1,\epsilon)\|^2.
\]

MSE 最优解为

\[
v^*(x,t)
=
\mathbb E[u_t(X_1,\epsilon)\mid X_t=x].
\]

在常见正则条件下，这个 conditional velocity 的条件平均给出正确的 marginal velocity，从而可用 ODE 运输整体分布。

## 16. 标准 rectified path

最简单路径为

\[
x_t=t x_1+(1-t)\epsilon,
\qquad t\in[0,1].
\]

它满足

\[
x_{t=0}=\epsilon,
\qquad
x_{t=1}=x_1.
\]

对时间求导：

\[
u_t=x_1-\epsilon.
\]

对固定的 \((x_1,\epsilon)\)，条件速度不随 \(t\) 变化，因此条件轨迹是直线。

训练完成后，从 \(x(0)\sim\mathcal N(0,I)\) 出发，数值积分

\[
x_{t+\Delta t}
\approx
x_t+\Delta t\,v_\theta(x_t,t)
\]

即可生成数据。上式是 Euler 方法。

## 17. Flow Matching 与 diffusion 的关系

二者不是“一个有噪声、一个没有噪声”这么简单。

- diffusion 常从 SDE 或 Gaussian perturbation kernel 出发；
- FM 从选择 probability path 并回归 velocity 出发；
- diffusion 的 probability-flow ODE 也可以生成相同边缘分布；
- 对固定 conditional path，noise、score、clean、velocity 往往可以互相转换；
- 实际训练差异还包括 path、time sampling、loss weighting 和 parameterization。

因此比较 RDD-DDPM 与 RDD-FM 时，必须明确是在比较完整系统，还是只比较 prediction target。

---

# Part IV：RDD Flow Matching 的完整推导

## 18. 单区间 \(t(1-t)\) RDD-FM

先看最简单的单区间版本：

\[
x_t
=
t x_1+(1-t)\epsilon
+t(1-t)d(x_1).
\]

端点为

\[
x_0=\epsilon,
\qquad
x_1=x_1.
\]

因为 residual coefficient 在两端都是零。

速度为

\[
\boxed{
u_t
=
x_1-\epsilon+(1-2t)d(x_1).
}
\]

### 18.1 为什么会出现 \(t(1-t)\)？

不要把它与 diffusion 的 \(r_t\) 直接比较。统一写成

\[
x_t
=
\alpha(t)[x_1+r(t)d]
+\sigma(t)\epsilon.
\]

rectified FM 有

\[
\alpha(t)=t,
\qquad
\sigma(t)=1-t.
\]

如果在数据生成方向中，transform fraction 从 coarse 逐渐回到 clean：

\[
r(t)=1-t,
\]

那么实际 residual coefficient 是

\[
b(t)=\alpha(t)r(t)=t(1-t).
\]

所以 \(t(1-t)\) 是“clean signal 系数 \(t\)”与“线性 degradation fraction \(1-t\)”的乘积，不是凭空选择的另一种 drift。

### 18.2 频率解释

若 \(P\) 是 projection，写

\[
x_1=x_\ell+x_h,
\qquad
d=-x_h.
\]

则

\[
\begin{aligned}
x_t
&=t(x_\ell+x_h)+(1-t)\epsilon-t(1-t)x_h\\
&=(1-t)\epsilon+t x_\ell+t^2x_h.
\end{aligned}
\]

低频 clean coefficient 是 \(t\)，高频 clean coefficient 是 \(t^2\)。当 \(t\) 很小时，\(t^2\ll t\)，因此形成显式 coarse-to-fine path。

速度为

\[
u_t=-\epsilon+x_\ell+2t x_h.
\]

早期 velocity 中高频目标较弱，后期逐渐增强。

**注意：这个 \(t\) 与 \(t^2\) 的漂亮分解严格对应单区间 \([0,1]\) 版本，不能未经处理直接套到当前每个多阶段区间。**

## 19. 当前代码实际使用的多阶段 FM envelope

生成方向使用 data time：

\[
t=0\text{ 是噪声},
\qquad
t=1\text{ 是数据}.
\]

阶段 \(k\) 的区间为

\[
t\in[s_k,s_{k+1}],
\qquad
L_k=s_{k+1}-s_k.
\]

定义 reverse transform fraction

\[
r_k(t)
=
\frac{s_{k+1}-t}{L_k}.
\]

它满足

\[
r_k(s_k)=1,
\qquad
r_k(s_{k+1})=0.
\]

当前 RDD-FM path 可以写成

\[
\boxed{
x_t
=
t[x_1+r_k(t)d_k]
+(1-t)\epsilon.
}
\]

展开后

\[
x_t=t x_1+(1-t)\epsilon+b_k(t)d_k,
\]

其中

\[
\boxed{
b_k(t)
=
t r_k(t)
=
\frac{t(s_{k+1}-t)}{L_k}.
}
\]

这正是 `diffusion/flow_matching_with_drift.py` 中的实现。

### 19.1 阶段端点

阶段入口：

\[
\begin{aligned}
x_{s_k}
&=s_kx_1+(1-s_k)\epsilon+s_kd_k\\
&=s_kG_k(x_1)+(1-s_k)\epsilon.
\end{aligned}
\]

阶段出口：

\[
x_{s_{k+1}}
=
s_{k+1}x_1+(1-s_{k+1})\epsilon.
\]

因此 current multi-stage envelope 的含义是：

- fine stage 的起点包含完整 coarse transform；
- 在该 stage 内逐渐移除 transform residual；
- stage 终点回到标准 rectified marginal；
- 然后进入下一个更细阶段。

它不是所有 stage 两端都满足 \(b=0\)。事实上

\[
b_k(s_k)=s_k,
\qquad
b_k(s_{k+1})=0.
\]

全局 \(t(1-t)\) 只是 \(s_k=0,s_{k+1}=1\) 时的特例。

### 19.2 Velocity target

因为

\[
b'_k(t)
=
\frac{s_{k+1}-2t}{L_k},
\]

所以

\[
\boxed{
u_t
=
x_1-\epsilon
+\frac{s_{k+1}-2t}{L_k}d_k.
}
\]

训练 loss 为

\[
\mathcal L_{mathrm{RDD-FM}}
=
\mathbb E
\|v_\theta(x_t,t,k)-u_t\|^2.
\]

不要随口说 velocity drift 一定在“区间中点”变号。其零点是

\[
t=\frac{s_{k+1}}2,
\]

而区间中点是

\[
\frac{s_k+s_{k+1}}2.
\]

二者一般不同，甚至 \(s_{k+1}/2\) 可能不在该 stage 内。

## 20. RDD-FM 的 clean reconstruction

标准 rectified FM 有

\[
x_t=t x_1+(1-t)\epsilon,
\qquad
u_t=x_1-\epsilon.
\]

因此

\[
x_t+(1-t)u_t=x_1.
\]

但 RDD-FM 多了 \(b(t)d(x_1)\)，标准公式不再成立。

令

\[
y_t=x_t+(1-t)u_t.
\]

代入 RDD path 与 velocity：

\[
\begin{aligned}
y_t
&=t x_1+(1-t)\epsilon+b(t)d\\
&\quad +(1-t)[x_1-\epsilon+b'(t)d]\\
&=x_1+[b(t)+(1-t)b'(t)]d.
\end{aligned}
\]

定义

\[
c(t)=b(t)+(1-t)b'(t).
\]

若 \(d=(P-I)x_1\)，则

\[
y_t=[I+c(t)(P-I)]x_1.
\]

对 projection \(P^2=P\)，分别作用 \(P\) 和 \(I-P\)：

\[
Py_t=Px_1,
\]

\[
(I-P)y_t=(1-c(t))(I-P)x_1.
\]

只要

\[
1-c(t)\neq0,
\]

就得到

\[
\boxed{
x_1
=
Py_t
+\frac1{1-c(t)}(I-P)y_t.
}
\]

这就是当前实现里的 `predict_x1_from_velocity`。

### 20.1 一般线性算子

若 \(G\) 是一般线性算子，不一定幂等，则

\[
y_t=[I+c(t)(G-I)]x_1.
\]

形式解为

\[
x_1=[I+c(t)(G-I)]^{-1}y_t.
\]

若 \(G\) 的特征值是 \(\lambda_i\)，可逆条件是

\[
1+c(t)(\lambda_i-1)\neq0
\quad\text{对所有 }i.
\]

该逆的 condition number 决定 velocity error 会被放大多少。projection 公式只是这个一般线性算子结果的一个特别简单的闭式特例。

## 21. FM stage transition

阶段出口没有 residual：

\[
x_s=sx_1+(1-s)\epsilon.
\]

进入更细 stage 时，需要构造

\[
x_s^{\mathrm{fine}}
=
sU(\hat x_1^{\mathrm{coarse}})
+(1-s)\epsilon_{\mathrm{fine}}.
\]

当前推荐实现会：

1. 从 coarse endpoint 和 clean estimate 恢复 coarse innovation；
2. 用归一化 repeat 把 coarse innovation lift 到 fine low-frequency subspace；
3. 在新出现的 null space 中采样独立 Gaussian；
4. 组合成 fine stage 训练时使用的 identity covariance。

这比简单把所有 noise 全部重抽更接近跨阶段 sample coupling，但论文中应准确区分：它匹配 marginal/covariance，不一定保证与训练时某个具体 noise realization samplewise 连续。

## 22. Wan 的时间方向

Wan 使用 noise time

\[
\sigma=1-t.
\]

因此

\[
\frac{dx}{d\sigma}
=
-\frac{dx}{dt}.
\]

若 RDD data-time velocity 是 \(u_t\)，Wan target 应为

\[
w_\sigma=-u_{1-\sigma}.
\]

clean reconstruction 中

\[
y=x_t+(1-t)u_t
\]

会变成

\[
y=x_\sigma-\sigma w_\sigma.
\]

fine-tune pretrained Wan 时，时间输入、velocity 符号和积分方向必须同时变换；只把 \(t\) 替换成 \(1-t\) 而不翻转 velocity 符号是错误的。

---

# Part V：RDD-DDPM 与 RDD-FM 怎样公平比较？

## 23. 统一路径表示

把两者都写成

\[
\boxed{
x_t
=
\alpha(t)[x_0+r(t)d(x_0)]
+\sigma(t)\epsilon.
}
\]

实际 residual displacement 是

\[
b(t)d=\alpha(t)r(t)d.
\]

因此不能直接比较 diffusion 的 \(r(t)\) 与 FM 的 \(b(t)\)。正确比较是

\[
r(t)
\quad\text{对比}\quad
\frac{b(t)}{\alpha(t)}.
\]

RDD-DDPM 使用 VP 系数：

\[
\alpha_{\mathrm{VP}}(t)=\sqrt{\bar\alpha_t},
\qquad
\sigma_{\mathrm{VP}}(t)=\sqrt{1-\bar\alpha_t}.
\]

RDD-FM 使用 rectified 系数：

\[
\alpha_{\mathrm{FM}}(t)=t,
\qquad
\sigma_{\mathrm{FM}}(t)=1-t.
\]

当前 FM 中

\[
\frac{b_k(t)}{\alpha_{\mathrm{FM}}(t)}
=
\frac{s_{k+1}-t}{L_k},
\]

正是从 coarse 到 clean 线性下降的 transform fraction。

所以 DDPM 的 linear \(r\) 与 FM 的 quadratic-looking \(b\) 在结构上可以是同一设计；二次项来自额外乘了 rectified clean coefficient \(t\)。

## 24. 三个层次的公平性

### 24.1 系统级公平

比较各自 canonical path 下的完整模型：

| Framework | Baseline | RDD |
|---|---|---|
| VP diffusion | Vanilla DiT/DDPM | RDD-DDPM |
| Rectified FM | Vanilla FM/SiT-style | RDD-FM |

主要比较各自 improvement：

\[
\Delta_{\mathrm{DDPM}}
=
\operatorname{FID}(\mathrm{RDD-DDPM})
-\operatorname{FID}(\mathrm{Vanilla-DDPM}),
\]

\[
\Delta_{\mathrm{FM}}
=
\operatorname{FID}(\mathrm{RDD-FM})
-\operatorname{FID}(\mathrm{Vanilla-FM}).
\]

这能说明 RDD 是否对两种 framework 都有效，但不能把 RDD-FM 与 RDD-DDPM 的差异完全归因于 velocity 与 noise target。

### 24.2 推理预算公平

同时报告：

- 相同 NFE 的 FID；
- 相同 wall-clock 的 FID；
- 相同 CFG；
- 相同 VAE 和 FID reference；
- Euler、midpoint、DDPM、DPM 各自真实的网络调用次数。

250 个 midpoint step 大约需要接近 500 NFE，不能与 250 NFE 的一阶方法直接标成“同样 250 steps”。

### 24.3 完全 path-matched 的 target 对照

固定完全相同的

\[
\alpha(t),\sigma(t),r(t),P,p(t),
\]

以及同一个 \(x_t\) distribution，然后分别训练：

1. effective-noise head；
2. velocity head。

对统一路径

\[
x_t=\alpha x_0+\sigma\epsilon+b d,
\]

effective-noise target 是

\[
\tilde\epsilon
=
\epsilon+\frac{b}{\sigma}d,
\]

velocity target 是

\[
u_t
=
\dot\alpha x_0+\dot\sigma\epsilon+\dot b d.
\]

这才严格隔离 parameterization 与 loss 的区别。还需要匹配 loss weighting，因为两个普通 MSE 会对时间区间产生不同隐含权重。

## 25. 当前实现中必须记录的两个事实

### 25.1 现有 FM-v2 与 DDPM 的 `300,300,400` 并不是镜像 stage boundary

现有 DDPM 脚本把 `--stage-lengths 300,300,400` 按 finest-to-coarsest 解释：

- 32×32：forward noise time \([0,0.3]\)；
- 16×16：\([0.3,0.6]\)；
- 8×8：\([0.6,1]\)。

在生成方向 \(t_{\mathrm{FM}}=1-t_{\mathrm{DDPM}}\) 下，严格镜像边界应该是：

- 8×8：FM data time \([0,0.4]\)；
- 16×16：\([0.4,0.7]\)；
- 32×32：\([0.7,1]\)。

也就是 FM 生成顺序长度应为

\[
400,300,300.
\]

但是当前 `RDD-FM-Unified-XL-v2` 脚本传入 `300,300,400`，当前 parser 先生成边界 \([0,0.3,0.6,1]\)，再把 resolution 反向分配，因此实际是：

- 8×8：\([0,0.3]\)，长度 300；
- 16×16：\([0.3,0.6]\)，长度 300；
- 32×32：\([0.6,1]\)，长度 400。

所以现有 FM-v2 与 DDPM-300/300/400 不是严格 stage-matched。已有 checkpoint 的结果仍然有效，但只能按其**真实 schedule** 汇报，不能称为与 DDPM 300/300/400 完全相同的路径分配。

如果未来要做镜像对照，应修改 parser 语义，或者在当前 parser 下传入能得到 \([0,0.4,0.7,1]\) 的长度顺序。

### 25.2 Unified 脚本是 stage-balanced loss，不是 global-uniform time

当前 DDPM unified 和 FM unified 训练脚本都在每个 optimizer step 中遍历所有 stage，并累加每个 stage 的 loss。每个 stage 内部再均匀采样 local time。

因此实际训练权重更接近：

\[
\mathcal L_{\mathrm{total}}
=
\sum_k\mathbb E_{t\sim\mathrm{Unif}(I_k)}
\mathcal L_k(t),
\]

即每个 stage 大致等权，而不是先从整个 \([0,1]\) 均匀采一个全局 \(t\)。后者会令 stage 被采样概率正比于区间长度 \(L_k\)。

论文、实验表和未来 theorem 必须明确使用的是哪一种 weighting。

---

# Part VI：怎样把 RDD 的数学变成真正可防守的 Novelty？

## 26. Correctness 与 novelty 的区别

下面内容主要证明 correctness：

- forward marginal 可以闭式采样；
- conditional posterior 是 Gaussian；
- drift 为零时退化到 DDPM；
- projection residual 可以闭式 clean inversion；
- noise filling 匹配选择的 boundary marginal；
- SDE/ODE 可以写出。

这些都重要，因为它们说明方法不是 ad-hoc hack。但 transformed-signal diffusion 已有 f-DM，通用 conditional path 已有 Flow Matching/Stochastic Interpolants，一般 diffusion solver 已有 DPM-Solver、DEIS、gDDIM。因此 correctness derivation 不能单独承担主要 novelty。

真正值得建立的新理论应回答下面几个问题。

## 27. Novelty 方向一：Single-head sufficiency / identifiability theorem

### 27.1 要解决的问题

f-DM 允许一般 transformed signal，并使用 signal/degradation 的更一般预测机制。RDD 希望只使用一个标准 DiT head。

需要正式刻画：

> 什么样的 transform residual 可以仅靠一个 effective-noise head，精确恢复 reverse dynamics 需要的 clean mean 和 residual mean？

### 27.2 建议定理

可以把定理分成四部分。

**命题 A：clean posterior mean recovery。** 对任意 deterministic same-shaped \(d\)，effective-noise MSE 的 Bayes predictor 都满足

\[
\hat x_0^*(x_t)=\mathbb E[X_0\mid X_t=x_t].
\]

这部分已经由第 8 节恒等式得到。

**命题 B：affine sufficiency。** 若

\[
d(x)=Ax+c,
\]

则

\[
d(\hat x_0^*)
=
\mathbb E[d(X_0)\mid X_t].
\]

因此单头对 posterior mean、marginal drift 和 projection clean inverse 足够。

**命题 C：必要性边界。** 如果要求

\[
d(\mathbb E[X])=\mathbb E[d(X)]
\]

对足够丰富的一类分布都成立，那么在适当连续性条件下，\(d\) 必须是仿射函数。这本质上是 Jensen equality 的函数刻画。

**命题 D：nonlinear bias bound。** 对非线性 \(d\)，用 Hessian 和 posterior covariance 给出

\[
\|\Delta_d(x_t)\|
\le
C_d\operatorname{Tr}\operatorname{Cov}(X_0\mid X_t)
\]

一类上界，并分析该偏差如何进入 posterior/ODE update。

### 27.3 为什么这比“有 drift”新？

叙事不再是“RDD 首次改变 signal”，而是：

> RDD 识别并刻画了一类可以被标准单头 DiT 精确参数化的 transformed-signal process；我们给出仿射充分性、广义必要性和非线性 plug-in bias。

这会直接解释为什么 RDD 的 restriction 有价值，而不是看起来只是 f-DM 的随意特例。

### 27.4 实验

至少比较：

- linear projection；
- linear blur；
- linear spectral mask；
- nonlinear encoder-decoder 或 nonlinear filter；
- single effective-noise head；
- clean + residual double head。

记录：

- \(\|d(\hat x_0)-d(x_0)\|\)；
- 对 residual posterior mean 的 calibration error；
- FID；
- 收敛曲线；
- nonlinear transform 下 single/double head 的差距。

## 28. Novelty 方向二：频谱 curriculum 与训练收敛定理

### 28.1 当前直觉还不等于证明

RDD 结果最有意思的地方是训练收敛显著加速。仅仅说“drift 是 deterministic，所以更容易学”是不够的；给定 \(x_0\) deterministic 不代表对训练分布边缘化后的 gradient variance 一定更小。

### 28.2 Two-band Gaussian toy model

设

\[
X_0=L+H,
\qquad
L\perp H,
\]

并假设

\[
L\sim\mathcal N(0,\Sigma_L),
\qquad
H\sim\mathcal N(0,\Sigma_H).
\]

projection 满足

\[
PX_0=L,
\qquad
d(X_0)=-H.
\]

RDD-DDPM 条件状态为

\[
X_t
=
\alpha_tL
+\alpha_t(1-r_t)H
+\sigma_t\epsilon.
\]

所以两部分有效 SNR 为

\[
\operatorname{SNR}_L(t)
=
\frac{\alpha_t^2\operatorname{Var}(L)}{\sigma_t^2},
\]

\[
\operatorname{SNR}_H(t)
=
\frac{\alpha_t^2(1-r_t)^2\operatorname{Var}(H)}{\sigma_t^2}.
\]

RDD 主动重排高频 SNR，而低频 SNR 不受 projection residual 直接缩放。

在单区间 RDD-FM 中：

\[
X_t=(1-t)\epsilon+tL+t^2H.
\]

于是高频信息按 \(t^2\) 而不是 \(t\) 出现。

### 28.3 可解析 posterior 的背景

对一维 Gaussian latent

\[
Z\sim\mathcal N(0,\lambda),
\qquad
Y=aZ+\sigma\epsilon,
\]

有

\[
\mathbb E[Z\mid Y]
=
\frac{a\lambda}{a^2\lambda+\sigma^2}Y,
\]

\[
\operatorname{Var}(Z\mid Y)
=
\frac{\lambda\sigma^2}{a^2\lambda+\sigma^2}.
\]

把 \(a\) 分别替换成 low/high band coefficient，就可以解析比较 standard 与 RDD 中各频段的 posterior difficulty。

### 28.4 真正要证明什么？

建议研究一个可解析 denoiser，例如线性模型 \(f_W(x)=Wx\)。平方损失的 Hessian 与输入 covariance 有关：

\[
\nabla_W^2\mathcal L
\propto
\operatorname{Cov}(X_t).
\]

目标是分析 RDD 是否在某些数据谱和 time weighting 下改善：

- Hessian condition number；
- 各频段 gradient scale mismatch；
- stochastic gradient variance；
- 达到给定 Bayes risk 所需的迭代数。

希望得到类似条件化结论：

\[
r^*(t)
=
\arg\min_r
\kappa(H_{\mathrm{train}}[r]),
\]

以及明确的适用条件，而不是声称任何数据上都必然加速。

### 28.5 实验验证

在 ImageNet latent 上记录：

- low/mid/high-frequency target loss；
- 每个频带的 gradient norm 和 variance；
- approximate Fisher/Hessian spectrum；
- 每个 checkpoint 的频带 reconstruction error；
- drift-free multiresolution control；
- 不同 \(r(t)\) 与 stage boundary。

如果理论预测哪个 schedule 更快，并且真实训练曲线服从预测，这会比单纯 FID ablation 强得多。

## 29. Novelty 方向三：从优化准则推导 transform schedule

### 29.1 首先优化 \(r(t)\)，不要把 \(b(t)\) 当作独立物理量

统一路径是

\[
x_t=\alpha(t)[x_0+r(t)d]+\sigma(t)\epsilon.
\]

实际 displacement 是

\[
b(t)=\alpha(t)r(t).
\]

不同 framework 的 \(\alpha(t)\) 不同，所以一个 framework 中看起来 linear 的 \(r\)，在另一个 framework 中可能产生 quadratic 的 \(b\)。

当前 FM 的

\[
b_k(t)=\frac{t(s_{k+1}-t)}{L_k}
\]

来自

\[
\alpha(t)=t,
\qquad
r_k(t)=\frac{s_{k+1}-t}{L_k}.
\]

因此当前 schedule 首先是“linear transform fraction”，不是任意 quadratic bump。

### 29.2 最简单的 minimum-energy 结论

若只要求

\[
r(s_k)=1,
\qquad
r(s_{k+1})=0
\]

并最小化

\[
J[r]=\int_{s_k}^{s_{k+1}}[r'(t)]^2dt,
\]

Euler-Lagrange 方程给出

\[
r''(t)=0,
\]

所以最优解正是线性 \(r(t)\)。这能给 current envelope 一个基础 smoothness 解释，但数学上较初等，不能单独成为强贡献。

### 29.3 为什么有人会从变分法得到 \(t(1-t)\)？

另一个不同的问题是：直接优化 endpoint-zero displacement \(b\)，要求

\[
b(0)=b(1)=0,
\qquad
\int_0^1b(t)dt=A,
\]

并最小化

\[
\int_0^1[b'(t)]^2dt.
\]

带积分约束的 Euler-Lagrange 方程给出常数二阶导数，最终

\[
b^*(t)=6A\,t(1-t).
\]

这说明 \(t(1-t)\) 也是固定 residual exposure 下的 minimum derivative-energy bump。

但是要注意：这个问题与当前多阶段 FM 的 boundary condition 不完全相同，因为 current fine-stage 有 \(b(s_k)=s_k\neq0\)。论文不能把单区间变分结论未经说明直接当作所有多阶段包络的证明。

### 29.4 更有价值的 data-aware objective

可以设计

\[
J[r]
=
\int
\left[
w_1(t)\,\mathcal V_{\mathrm{target}}(r,t)
+w_2(t)|r'(t)|^2\|d\|^2
+w_3(t)|r''(t)|^2\|d\|^2
\right]dt.
\]

其中：

- \(\mathcal V_{\mathrm{target}}\)：effective-noise 或 velocity target variance；
- \(|r'|^2\)：额外 transform velocity energy；
- \(|r''|^2\)：路径 stiffness，影响 solver error；
- \(w_i(t)\)：time weighting、SNR 和模型误差权重。

再加入 compute/resolution 约束，推导 \(r^*(t)\) 和 stage boundaries。真正的 novelty 应是：理论根据数据 spectrum 预测 schedule，并被实验验证，而不只是发现一个二次函数。

## 30. Novelty 方向四：drift-aware solver 的阶数与误差定理

### 30.1 数值方法中的“阶”是什么意思？

步长为 \(h\) 时：

- local truncation error 为 \(O(h^{p+1})\)；
- 积分固定总区间后的 global error 为 \(O(h^p)\)；

则称方法具有 \(p\) 阶收敛。

仅仅在中点多评估一次网络不自动证明二阶；需要展开 exact solution 与 numerical update 的 Taylor series。

### 30.2 RDD 的结构机会

RDD ODE 中有一个已知时间系数乘 residual：

\[
\dot x=F_\theta(x,t)+a(t)d_\theta(x,t).
\]

若一步内把 residual 固定，variation of constants 可以精确积分

\[
\int_t^s\Phi(s,\tau)a(\tau)d\tau.
\]

这就是 current first-order correction 能闭式出现的原因。

### 30.3 建议证明

1. exact residual 下的一阶 exponential-integrator identity；
2. learned residual 下的误差分解；
3. midpoint residual recomputation 的 local error；
4. 二阶/三阶 scheme 的充分条件；
5. residual prediction error 对 global order 的影响。

目标形式可以是

\[
\|e_{\mathrm{global}}\|
\le
C_1h^p
+C_2\sup_t\|\hat d_t-d_t^*\|.
\]

需要与 gDDIM、DEIS、DPM-Solver 等一般 solver 进行 prior-art 对照。RDD 的贡献必须是“利用这个 residual path 的可积结构并给出误差保证”，而不是泛泛宣称只有 RDD 能使用高阶 solver。

## 31. Novelty 方向五：一般线性算子的 FM invertibility

第 20 节已经得到

\[
y=[I+c(t)(G-I)]x_1.
\]

可以系统研究：

- 可逆的必要充分条件；
- operator spectrum 与 condition number；
- velocity error 到 clean error 的放大界；
- blur、wavelet、spectral truncation 的闭式/快速逆；
- envelope 如何影响最坏方向的数值稳定性。

例如若 \(G\) 可对角化，特征值为 \(\lambda_i\)，则 clean error 在第 \(i\) 个方向被放大为

\[
\frac1{|1+c(t)(\lambda_i-1)|}.
\]

可以选择 envelope 或 loss weighting，避免训练/采样集中在接近奇异的时间区域。这是一个干净但更适合作为辅助理论的方向。

## 32. Video 能增加什么，不能增加什么？

仅把 \(P\) 从 2D spatial projection 换成 3D spatiotemporal projection，不足以形成理论 novelty，因为 Pyramidal Flow/PyramidalWan 等已经研究 progressive video resolution。

更有意义的理论扩展是定义

\[
P_s\text{：空间投影},
\qquad
P_\tau\text{：时间投影},
\]

分析：

- 何时 \(P_sP_\tau=P_\tau P_s\)；
- 空间/时间 residual 是否能分离；
- clean inverse 是否可以按两个子空间分解；
- motion frequency 与 spatial frequency 是否需要不同 \(r_s(t),r_\tau(t)\)；
- 哪种 schedule 最小化 temporal flicker 与计算量。

Video 最主要的价值仍是 breadth 和 effectiveness 证据，而不是替代上述核心理论。

---

# Part VII：建议执行的实验与证明清单

## 33. P0：先修正理论和实现表述

- [ ] 把依赖 \(x_0\) 的 forward transition 写成 \(q(x_t\mid x_{t-1},x_0)\)，或明确扩充状态。
- [ ] marginal SDE/ODE 使用 \(\mathbb E[d(X_0)\mid X_t]\)，而不是不可见的真实 \(d(x_0)\)。
- [ ] 主定理限制在线性/仿射 residual；非线性部分明确为 plug-in approximation 或双头扩展。
- [ ] paper 与代码统一 stage weighting：global-uniform time 还是 stage-balanced。
- [ ] FM stage-length parser 和脚本说明统一语义。
- [ ] 所有 midpoint 结果报告真实 NFE。
- [ ] 单元测试验证 \(d=0\) 时完整退化为 baseline。
- [ ] 单元测试验证 effective noise 与 pure innovation 的分离。
- [ ] 每个 FM stage 数值检查 \(1-c(t)\) 不接近零。

## 34. P1：最小而有说服力的核心实验

### 34.1 四组 framework 内对照

1. Vanilla VP diffusion；
2. RDD VP diffusion；
3. Vanilla rectified FM；
4. RDD rectified FM。

控制模型、数据、batch、optimizer、EMA、训练样本数、训练 FLOPs、CFG、FID reference。

### 34.2 一个 path-matched 对照

固定同一 \(\alpha,\sigma,r,P,t\) distribution，分别训练 effective-noise 与 velocity target。这组专门回答“FM 与 diffusion 是否公平比较”。

### 34.3 关键 drift-free control

保留相同：

- 多分辨率输入；
- stage boundaries；
- token/FLOP 分配；
- 模型结构；

只令

\[
d=0.
\]

否则无法区分提升来自 drift path，还是仅来自 cheap multiresolution curriculum。

## 35. P2：最值得的 novelty 验证

### 35.1 Single-head theorem 实验

| Transform | Linear? | Single head | Double head | 要测什么 |
|---|---:|---:|---:|---|
| avg/repeat projection | 是 | ✓ | ✓ | 两者应接近 |
| Gaussian blur | 是 | ✓ | ✓ | 验证一般线性情况 |
| spectral mask | 是 | ✓ | ✓ | 频域控制 |
| nonlinear transform | 否 | ✓ | ✓ | 测 plug-in bias |

### 35.2 Mechanism measurements

- low/mid/high frequency loss；
- target norm 与 variance；
- per-band gradient norm/variance；
- approximate Hessian/Fisher condition number；
- 50K、100K、200K、400K checkpoint sweep；
- 至少第二个 seed。

### 35.3 Envelope ablation

比较时必须匹配某种公平约束：相同 peak、相同 \(\int r\)、相同 displacement energy 或相同 compute。候选包括：

- linear \(r\)；
- cosine \(r\)；
- polynomial \(r\)；
- data-aware optimized \(r^*\)；
- drift strength \(\lambda\in\{0,0.25,0.5,1,1.5\}\)。

### 35.4 Solver

固定真实 NFE，例如 20/50/100/250，比较：

- Euler/DDIM；
- standard DPM/ODE solver；
- RDD first-order correction；
- midpoint residual recomputation；
- 新的有阶数证明的 solver。

## 36. 推荐论文主线

不要把主线写成：

> 我们第一次在 diffusion 中加入 resolution drift。

更可防守的主线是：

> Existing transformed-signal and pyramidal methods establish a broad design space for time-dependent signal transformations. RDD identifies a tractable same-state residual subclass that can be folded into a single standard DiT target. We characterize when this single-head reduction is exact, explain its frequency-dependent optimization curriculum, and derive path-consistent reverse and flow formulations with explicit operator-dependent reconstruction and solver corrections.

对应中文：

> 已有工作已经建立了随时间变换信号的广义框架。RDD 研究其中一个受限但可解析的同状态残差子类：它能够折叠进标准单头 DiT。我们刻画单头归约精确成立的条件，解释其频率相关训练课程，并为 diffusion 与 Flow Matching 给出路径一致的反向恢复、算子反演和数值修正。

贡献优先级建议：

1. 单头充分性、必要性边界和 nonlinear bias；
2. 频谱 curriculum 与收敛机制；
3. data-aware transform schedule；
4. 有误差阶保证的 solver；
5. 一般线性算子 clean inverse；
6. image/video 的实证 breadth。

---

# Part VIII：常见问题与自检

## 37. “Drift”到底是哪个量？

至少有三个容易混淆的量：

1. transform residual：
   \[
   d(x)=G(x)-x;
   \]
2. path displacement：
   \[
   b(t)d(x)=\alpha(t)r(t)d(x);
   \]
3. velocity/SDE 中的 drift contribution：
   \[
   \dot b(t)d(x)
   \quad\text{或}\quad
   \alpha(t)\dot r(t)d(x).
   \]

论文和讨论中说“drift 更大”时，必须说明指的是哪一个。

## 38. “线性 diffusion”是什么意思？

可能指：

- \(G\) 是线性算子；
- Gaussian corruption 对 \(x_0,\epsilon\) 是线性的；
- SDE drift 对状态 \(x_t\) 是线性的；
- stage envelope \(r(t)\) 是线性的。

这四件事不同。论文中最好避免无修饰地写 “linear diffusion”。

## 39. \(t(1-t)\) 是概率吗？

不是。这里的 \(t\) 是归一化时间，\(t(1-t)\) 是 residual displacement 的时间包络，不是概率密度，也不是 Bernoulli variance 的统计解释。

## 40. FM 为什么不用 DDPM posterior？

FM 直接学习 ODE velocity 并数值积分，不需要复用 DDPM 的 reverse Gaussian posterior。RDD-FM 应从自己选择的 conditional path 求导，而不是把 RDD-DDPM posterior 硬套过来。

## 41. FM 路径弯了以后还叫 Flow Matching 吗？

可以。Conditional Flow Matching 允许一般可微 conditional path。只是它不再是直线 conditional OT path，不能自动继承“条件轨迹最直”或某些 transport-cost 最优保证。

## 42. 为什么 projection clean inverse 不能直接用于 VAE transform？

因为 closed form 使用了：

\[
d=(P-I)x,
\qquad
P^2=P.
\]

VAE encoder-decoder 通常非线性、非幂等，也可能改变维度。需要额外 head、implicit solve 或 transform-specific inverse。

## 43. 当前结果能否证明 RDD 理论新？

不能。强 FID 和快速收敛证明方法有效，但不能单独证明 drift 概念新。Novelty 需要依靠精确的 prior-art 定位，以及第 27–31 节中至少一个真正的新定理与对应实验。

---

# Part IX：符号表

| 符号 | 含义 |
|---|---|
| \(x_0\) / \(x_1\) | diffusion 中的 clean data / FM 中的数据端 |
| \(x_t\) | 时间 \(t\) 的随机状态 |
| \(\epsilon\) | 标准 Gaussian noise |
| \(D\) | downsampling / average pooling |
| \(U\) | upsampling / repeat lift |
| \(P=UD\) | 同状态低分辨率 projection |
| \(G\) | 一般 same-state transform |
| \(d(x)=G(x)-x\) | transform residual |
| \(\beta_t\) | DDPM 单步噪声强度 |
| \(\alpha_t=1-\beta_t\) | DDPM 单步信号保留系数平方 |
| \(\bar\alpha_t\) | \(\prod_{i=1}^t\alpha_i\) |
| \(r_t\) | transformed-signal fraction |
| \(b(t)=\alpha(t)r(t)\) | 状态中的实际 residual coefficient |
| \(\tilde\epsilon\) | effective training target，不是 pure Gaussian noise |
| \(\epsilon_{\mathrm{pure}}\) | 围绕 transformed mean 的 Gaussian innovation |
| \(u_t\) | conditional FM velocity target |
| \(v_\theta\) | 神经网络预测的 marginal velocity |
| \(s_k,s_{k+1}\) | FM stage 的 data-time 边界 |
| \(L_k\) | FM stage 长度 |
| \(c(t)\) | FM clean inverse 中的 operator coefficient |
| NFE | number of function evaluations，神经网络调用次数 |

---

# Part X：建议阅读顺序

第一次阅读：

1. 第 0–3 节：符号与基础；
2. 第 4、7、8 节：标准 DDPM、RDD marginal、effective noise；
3. 第 14–19 节：FM 背景和 current multi-stage path；
4. 第 23–25 节：DDPM/FM 对齐与当前实现事实；
5. 第 26–36 节：novelty 与实验。

第二次阅读再看：

- 第 9 节 posterior；
- 第 10、12、13 节 conditional expectation、SDE 和 solver；
- 第 20 节 clean inverse；
- 第 27–31 节 proposed theorems。

外部背景材料：

- f-DM: <https://arxiv.org/abs/2210.04955>
- Flow Matching: <https://arxiv.org/abs/2210.02747>
- Stochastic Interpolants: <https://arxiv.org/abs/2209.15571>
- DPM-Solver: <https://arxiv.org/abs/2206.00927>
- gDDIM: <https://arxiv.org/abs/2206.05564>
- Pyramidal Flow Matching: <https://arxiv.org/abs/2410.05954>
- PyramidalWan: <https://arxiv.org/abs/2601.04792>

最后记住一句话：

> RDD 最值得建立的数学，不是“我也能写出一个 drift”，而是“为什么这个受限 residual family 能被单头精确学习、为什么它改变训练条件与收敛、以及怎样从数据和数值误差原则推导最优路径”。
