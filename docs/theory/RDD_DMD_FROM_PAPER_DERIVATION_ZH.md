# 从 DMD 原论文逐行推导 RDD-DMD

本文只使用代码仓库里真实存在的量：

- `sqrt_alphas_cumprod[t]`；
- `1 - alphas_cumprod[t]`；
- `upsample(downsample(x)) - x`；
- `(timestep_map[t] - T_c) / (T_c_plus_1 - T_c)`。

不引入可选择的 path correction，也不使用在线单步教师输出代替 DMD 的固定配对数据。

## 1. DMD 原论文到底计算什么

DMD 希望最小化学生生成分布与教师目标分布之间的 KL：

\[
D_{\mathrm{KL}}(p_{\mathrm{fake}}\|p_{\mathrm{real}}).
\]

论文 Eq.7 给出的生成器梯度是：

\[
\mathbb E\left[
w_t
\left(
s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{real}}(x_t,t)
\right)
\frac{\partial x_t}{\partial x}
\frac{\partial G_\theta}{\partial\theta}
\right].
\]

其中 \(x=G_\theta(z)\) 是学生生成的 clean sample。必须先知道三个东西：

1. 学生 sample 怎样被扰动成 \(x_t\)；
2. 这种扰动下 real/fake score 怎样由 clean prediction 表示；
3. \(x_t\) 对学生 clean output 的导数。

## 2. 仓库中真实的 RDD forward

`gaussian_diffusion_with_drift.py::q_sample` 实际执行：

\[
\begin{aligned}
x_t={}&
\operatorname{signal\_scale}(t)x\\
&+\operatorname{signal\_scale}(t)r(t)
\left[\operatorname{down\_up}(x)-x\right]\\
&+\operatorname{noise\_scale}(t)\epsilon,
\end{aligned}
\]

其中完全对应代码：

\[
\operatorname{signal\_scale}(t)
=\texttt{sqrt\_alphas\_cumprod[t]},
\]

\[
\operatorname{noise\_scale}(t)
=\texttt{sqrt\_one\_minus\_alphas\_cumprod[t]},
\]

\[
r(t)=
\texttt{drift\_scale\_multiplier}
\frac{
\texttt{timestep\_map[t]}-\texttt{T\_c}
}{
\texttt{T\_c\_plus\_1}-\texttt{T\_c}
}.
\]

这里：

\[
\operatorname{down\_up}(x)
=
\operatorname{nearest\_repeat}
\left(\operatorname{average\_pool}(x)\right).
\]

将 forward mean 写完整：

\[
\operatorname{mean}_t(x)
=
\operatorname{signal\_scale}(t)
\left[
x+r(t)(\operatorname{down\_up}(x)-x)
\right].
\]

条件方差是：

\[
\operatorname{variance}(t)
=1-\texttt{alphas\_cumprod[t]}.
\]

## 3. RDD 下 score 怎样由 predicted clean 表示

对任意 clean 分布 \(p(x)\)，加上上述 RDD 扰动后：

\[
p_t(x_t)=\int p(x)q_t(x_t\mid x)dx.
\]

高斯条件核的 score identity 给出：

\[
s_p(x_t,t)
=-
\frac{
x_t-
\mathbb E[\operatorname{mean}_t(x)\mid x_t]
}{
\operatorname{variance}(t)
}.
\]

由于 `average_pool`、`nearest_repeat` 和加法都是线性的，令网络的
predicted clean 为：

\[
\mu_p(x_t,t)=\mathbb E[x\mid x_t],
\]

则：

\[
\mathbb E[\operatorname{mean}_t(x)\mid x_t]
=
\operatorname{signal\_scale}(t)
\left[
\mu_p+r(t)(\operatorname{down\_up}(\mu_p)-\mu_p)
\right].
\]

因此 fake score 减 real score 为：

\[
\begin{aligned}
s_{\mathrm{fake}}-s_{\mathrm{real}}
=
\frac{\operatorname{signal\_scale}(t)}{
\operatorname{variance}(t)}
\Big[&\Delta\\
&+r(t)(\operatorname{down\_up}(\Delta)-\Delta)
\Big],
\end{aligned}
\]

其中只有：

\[
\Delta=
\mu_{\mathrm{fake}}(x_t,t)-
\mu_{\mathrm{real}}(x_t,t).
\]

仓库的模型虽然训练为 combined-noise prediction，但
`p_mean_variance(...)["pred_xstart"]` 正是这里需要的 posterior clean mean。
原因是训练 target 满足逐样本恒等式：

\[
x=
\frac{x_t-
\operatorname{noise\_scale}(t)\,
\text{combined-noise}}
{\operatorname{signal\_scale}(t)}.
\]

## 4. 通过 RDD forward 对学生反传

对任意扰动 \(v\)，forward mean 对 clean input 的导数作用为：

\[
\frac{\partial x_t}{\partial x}[v]
=
\operatorname{signal\_scale}(t)
\left[
v+r(t)(\operatorname{down\_up}(v)-v)
\right].
\]

本仓库使用的 average-pool 后 nearest-repeat 满足：

\[
\langle\operatorname{down\_up}(u),v\rangle
=
\langle u,\operatorname{down\_up}(v)\rangle.
\]

所以转置导数使用完全相同的表达式。这个结论已经同时用矩阵性质和
PyTorch autograd 数值检查验证。

## 5. 代入 DMD Eq.8 的权重

DMD Eq.8：

\[
w_t=
\frac{\operatorname{variance}(t)}
{\operatorname{signal\_scale}(t)}
\frac{1}
{\operatorname{mean}|\mu_{\mathrm{real}}-x|}.
\]

论文写作 \(CS/\|\cdot\|_1\)。代码使用 `mean(abs(...))`，两者完全相同。

把第 3、4、5 节代入 DMD Eq.7：

\[
\boxed{
g_{\mathrm{RDD-DMD}}
=
\frac{\operatorname{signal\_scale}(t)}
{\operatorname{mean}|\mu_{\mathrm{real}}-x|}
\;\operatorname{drift\_expression}
\left(
\operatorname{drift\_expression}(\Delta)
\right)
}
\]

这里不是新定义一个可调算子；`drift_expression(v)` 只是逐字代入
`q_sample` 中已有的代码：

\[
\operatorname{drift\_expression}(v)
=v+r(t)(\operatorname{down\_up}(v)-v).
\]

因此最终展开式为：

\[
\boxed{
g_{\mathrm{RDD-DMD}}
=
\frac{\operatorname{signal\_scale}(t)}
{\operatorname{mean}|\mu_{\mathrm{real}}-x|}
\left\{
u+r(t)[\operatorname{down\_up}(u)-u]
\right\}
}
\]

其中：

\[
u=\Delta+r(t)[\operatorname{down\_up}(\Delta)-\Delta].
\]

没有其它 correction switch 或梯度裁剪。

## 6. 与 DMD 的两个严格退化检查

### 6.1 关闭 resolution drift

当 `c=None` 时，仓库 `q_sample` 没有 drift：

\[
g=
\frac{
\operatorname{signal\_scale}(t)
(\mu_{\mathrm{fake}}-\mu_{\mathrm{real}})
}{
\operatorname{mean}|\mu_{\mathrm{real}}-x|
}.
\]

这正是 VP forward 下 DMD Eq.7 加 Eq.8 的结果。

### 6.2 DMD Algorithm 2 使用的 unit-signal-scale 情形

当 forward 是 `x + sigma(t)*noise`，即 signal scale 为 1 时：

\[
g=
\frac{
\mu_{\mathrm{fake}}-\mu_{\mathrm{real}}
}{
\operatorname{mean}|\mu_{\mathrm{real}}-x|
},
\]

与 DMD Appendix Algorithm 2 第 420–428 行完全相同。

## 7. 旧实现具体错在哪里

旧实现对包含 signal scale 的 forward expression 连续调用两次，得到：

\[
\frac{
\operatorname{signal\_scale}(t)^2
\operatorname{drift\_expression}
(\operatorname{drift\_expression}(\Delta))
}{
\operatorname{mean}|\mu_{\mathrm{real}}-x|
}.
\]

但 DMD Eq.8 已经消掉一份 signal scale，因此正确结果只剩一份。旧实现
在高噪声位置额外压小梯度。

旧 `preservation_loss` 也不是 DMD Eq.9。DMD Eq.9 要求固定配对：

\[
(z,y),\qquad y=\text{教师完整确定性采样}(z),
\]

然后：

\[
L_{\mathrm{reg}}=\operatorname{LPIPS}(G_\theta(z),y).
\]

旧实现使用“最高噪声输入上的一次教师 predicted clean”，既没有完整教师
轨迹，也没有固定配对，并且使用 latent MSE 而不是论文的 LPIPS。

## 8. 最终训练目标与数据流

生成器：

\[
L_G=L_{\mathrm{distribution}}+0.25L_{\mathrm{reg}}.
\]

其中 `L_distribution` 使用 surrogate MSE，使其对学生输出的导数恰好是
第 5 节的 \(g_{\mathrm{RDD-DMD}}\)。

fake-score 网络每个 outer step 更新一次，使用同一轮刚生成的 detached
学生样本和仓库原本的 RDD denoising loss。该更新顺序对应 DMD Algorithm 1，
不使用 DMD2 的 5:1 TTUR，也不使用 GAN。

固定回归数据使用 25,000 对，与 DMD ImageNet 配置一致。每个 RDD pair
必须存：

1. 8×8 初始噪声；
2. 8→16 noise-filling 使用的 16×16 噪声；
3. 16→32 noise-filling 使用的 32×32 噪声；
4. class label；
5. 教师完整确定性 RDD250 输出。

## 9. 十二项代码检查

`2d/test_rdd_phased_dmd_math.py` 自动验证：

1. stage 顺序；
2. down-up 幂等性；
3. down-up 自伴性；
4. drift fraction 与 `q_sample` 一致；
5. RDD re-noise 与 `q_sample` 一致；
6. no-drift 恢复普通 VP；
7. forward Jacobian 与 autograd 一致；
8. RDD score difference identity；
9. 最终实现与 DMD Eq.7/8 一致；
10. no-drift 恢复 VP-DMD；
11. unit signal scale 恢复 DMD Algorithm 2；
12. surrogate loss 的反向梯度正确。

另外验证了显式存储 boundary noise 能完全复现 noise-filling 输入。

## 10. 原始资料

- DMD 论文：<https://arxiv.org/abs/2311.18828>
- DMD HTML Eq.1–9：<https://arxiv.org/html/2311.18828v4#S3>
- DMD Algorithm 2/3：<https://arxiv.org/html/2311.18828v4#A2>
- DMD Appendix F：<https://arxiv.org/html/2311.18828v4#A6>
# 重要更新（2026-08-31）

本文对应的旧 stagewise DMD 与 deterministic posterior-mean pair sampler 已被实验证明无效。请以 [RDD-DMD v2 经过验证的推导](RDD_DMD_V2_VERIFIED_DERIVATION_ZH.md) 为准。旧文保留仅用于追踪推导历史，不应再用于启动实验。
