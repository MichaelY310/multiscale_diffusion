# RDD-DMD v2：经过数值验证的推导与实现

## 结论

旧实验无效，原因有两个：

1. regression pairs 使用“DDPM posterior mean 去掉随机项”作为确定性采样器。它不是 DDIM/ODE，会持续压低方差，生成纯色或模糊 target；
2. distribution matching 只轮流训练一个中间 stage，并在前面的 stage 截断梯度。它不是最终三步生成分布的严格 DMD。

v2 修复为：

- regression target 使用保持同一个 forward noise 的 RDD marginal-DDIM；
- DMD 作用于最终 32×32 输出，梯度穿过完整的 8→16→32 三次 DiT 调用；
- 查询 16×16 或 8×8 score 时，对最终输出做与 Unified RDD 训练一致的 average-pool；
- 在全局 (t\in[20,980)) 上均匀采样，而不是让三个 stage 各占三分之一概率。

## 1. RDD forward marginal

令 (P=UD) 表示 average-pool 后 nearest-repeat。第 (j) 个 stage 的 RDD marginal 写成

\[
x_t=\alpha_t A_t x_0+\sigma_t\epsilon,
\qquad
A_t=(1-r_t)I+r_tP.
\]

其中 (r_t) 是 stage 内的线性 resolution-drift 比例。由于 (P=P^\top\)，所以

\[
A_t^\top=A_t.
\]

## 2. RDD score

对任意 clean distribution，设 denoiser 给出的 posterior clean mean 为

\[
\mu(x_t,t)=\mathbb E[x_0\mid x_t].
\]

广义 Tweedie identity 给出

\[
s(x_t,t)
=
-\frac{x_t-\alpha_tA_t\mu(x_t,t)}{\sigma_t^2}.
\]

因此 fake 与 real score 的差是

\[
s_{\rm fake}-s_{\rm real}
=
\frac{\alpha_tA_t
(\mu_{\rm fake}-\mu_{\rm real})}{\sigma_t^2}.
\]

## 3. 为什么 DMD gradient 中出现两次 (A_t)

RDD re-noising 为

\[
x_t=\alpha_tA_tx+\sigma_t\epsilon,
\]

所以从 noisy latent 拉回 clean generator output 的 Jacobian 是

\[
\frac{\partial x_t}{\partial x}=\alpha_tA_t.
\]

原始 DMD 的权重为

\[
w_t=
\frac{\sigma_t^2}{\alpha_t}
\frac{1}{\operatorname{mean}|\mu_{\rm real}-x|}.
\]

将 score difference、Jacobian 和权重相乘：

\[
g_{\rm RDD-DMD}
=
\frac{
\alpha_tA_t^2
(\mu_{\rm fake}-\mu_{\rm real})
}{
\operatorname{mean}|\mu_{\rm real}-x|
}.
\]

第一个 (A_t) 来自 Gaussian score；第二个 (A_t) 来自 forward kernel 的 Jacobian transpose。它不是重复计算。

## 4. 正确的确定性 RDD teacher path

已知当前状态和 clean prediction，先反推出当前 forward noise：

\[
\hat\epsilon_t
=
\frac{x_t-\alpha_tA_t\hat x_0}{\sigma_t}.
\]

移动到更干净的 (s<t) 时保持同一个 noise：

\[
\boxed{
x_s
=
\alpha_sA_s\hat x_0
+\sigma_s\hat\epsilon_t
}.
\]

如果 (hat x_0=x_0)，那么该更新严格得到

\[
x_s=\alpha_sA_sx_0+\sigma_s\epsilon.
\]

当 drift 关闭时，(A_t=A_s=I)，公式严格退化为普通 η=0 DDIM。

旧实现使用

\[
x_s=\mathbb E[x_s\mid x_t,\hat x_0]
\]

并直接删除 posterior variance。反复取条件均值会丢掉

\[
\mathbb E[\operatorname{Var}(x_s\mid x_t,\hat x_0)],
\]

因此方差不断收缩。该路径不是 DDIM，也不是 probability-flow ODE。

## 5. 最终三步生成器的严格目标

三步 student 是复合函数

\[
G_\theta(z)=
G_{32,\theta}\circ T_{16\to32}\circ
G_{16,\theta}\circ T_{8\to16}\circ
G_{8,\theta}(z).
\]

令最终输出为 (x^{32}=G_\theta(z))。三个 stage 的 clean score target 分别是

\[
x_0^{32}=x^{32},\qquad
x_0^{16}=D_{32\to16}x^{32},\qquad
x_0^{8}=D_{32\to8}x^{32}.
\]

对选中的 stage 使用其 RDD kernel re-noise，然后计算该 stage 的 RDD-DMD gradient。因为 downsample 和三次 generator 调用都保留在 autograd graph 中，梯度自动经过

\[
D_j^\top
\]

并继续传回全部三次共享参数调用。

## 6. timestep 分布

原始 DMD 在全局

\[
t\sim\mathcal U\{20,\ldots,979\}
\]

采样。对于 300/300/400，候选数是：

- 32×32：280；
- 16×16：300；
- 8×8：380。

因此 score batch 选择三个 resolution 的概率必须是

\[
0.2917,
\quad 0.3125,
\quad 0.3958,
\]

而不是各 (1/3)。v2 先按上述权重选择一个 batch resolution，再在该 stage 的合法 local timestep 中均匀采样。

## 7. 已完成的验证

`2d/test_rdd_phased_dmd_math.py` 包含 17 项测试：

1. inference stage 顺序；
2. (P^2=P)；
3. (P=P^\top)；
4. drift fraction；
5. RDD q-sample 等式；
6. 无 drift 的 VP 极限；
7. Jacobian 与 autograd；
8. Gaussian score identity；
9. DMD Eq. 7/8；
10. VP-DMD 极限；
11. unit-signal DMD Algorithm 2 极限；
12. surrogate loss 梯度；
13. RDD marginal-DDIM 的 exact-noise coupling；
14. 普通 DDIM 极限；
15. bitwise determinism；
16. average-pool transpose；
17. 全局 timestep 的 280/300/380 划分。

surfi2 实测为 17/17 PASS。修复后的 16-pair smoke 还通过：

- latent std：0.8145；
- decoded spatial std median：0.1989；
- decoded dynamic range median：0.5811；
- bitwise deterministic：true；
- global-batch-256 完整 forward/backward：通过，无 OOM、NaN 或 Inf。

预览与指标位于：

`eval_results/diagnostics/rdd_dmd_pair_smoke_v2/`
