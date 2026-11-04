# 协方差处理：原论文主张与 RDD 的可比范围

日期：2026-11-02（洛杉矶）。本轮为原文、附录和已有作者源码的只读核查；没有运行训练、采样或质量实验，没有使用 GPU。没有修改生产采样器。以下“作者结论”不等于本项目复现了这些实验。

需要区分：反向 Gaussian 转移的协方差；上采样已知 Gaussian 噪声产生的协方差；真实 clean 在给定当前带噪状态后的后验协方差。它们相关，但不是同一个量。RDD 本轮讨论的缺项只针对 fresh 边界及理想条件均值替换，不覆盖所有 RDD-DDPM/coupled 变体。

## 普通扩散与后验统计

| 原文与定位 | 做法及近似范围 | 作者实际结论 |
|---|---|---|
| [DDPM](https://arxiv.org/html/2006.11239v2)，§3.2 | 反向 Gaussian 使用固定时间相关的标量方差乘 I；比较 beta 和 posterior-beta | 两种固定选择实验表现相近；没有证明完整后验 covariance 无关紧要。 |
| [Improved DDPM](https://arxiv.org/html/2102.09672v1)，§3.1、§4 | 学习逐维对角方差，以 beta/posterior-beta 的 log 插值参数化；不是完整矩阵，插值权重没有硬限制 | 多步时两个固定值大部分时间很接近；学习方差改善 likelihood，并使少步采样更稳健。不能把多步观察推广到任意大跳步。 |
| [Analytic-DPM](https://arxiv.org/html/2201.06503v3)，§2–3、Theorem 1–2 | 在时间相关各向同性 Gaussian 反向族中推导 KL 最优方差；用预训练 score、Monte Carlo 和截断估计 | 某些手工方差低估该族内最优值；改善所测 likelihood/效率。所谓最优受模型族与目标约束，不是精确完整后验。 |
| [Estimating the Optimal Covariance with Imperfect Mean](https://arxiv.org/html/2206.07309v1)，§3–4、附录E.2 | 理论覆盖 full/diagonal covariance 和均值误差修正；实际主要用对角形式、共享网络与两阶段训练 | 对角设计有实测收益，但均值修正版本 NPR 并不保证优于 SN 的 FID；likelihood 与 sample quality 的优劣不必一致。 |
| [Score-SDE](https://arxiv.org/html/2011.13456)，§4.3、附录D | 构造相同边缘密度的 probability-flow ODE；改变概率运输方式 | 理想 score/精确积分下的等价，不是将离散 DDPM 的噪声项直接置零；没有任意边界转换自动正确的结论。 |
| [On the Posterior Distribution in Denoising](https://arxiv.org/html/2309.13598v2)，§3–4 | 理想 denoiser 的导数确定后验矩；实际通过方向探测估计主方向等统计，无需完整存矩阵 | 支持推理时从已有 denoiser 提取不确定性；主要验证 uncertainty quantification，不能据此声称我们的边界补偿已改善 FID。 |

## 多分辨率方法

| 原文与定位 | 边界如何处理 | 应如何解读结论 |
|---|---|---|
| [Pyramid Flow](https://arxiv.org/html/2410.05954v2)，§3.2.2、式12–15、附录A/C.2 | 直接传递带噪 endpoint，配合信号缩放、跨尺度时间映射和块相关补噪；nearest 2x 时 gamma=-1/3 | 在声明的线性上采样及端点关系下解析匹配 Gaussian 分布。去掉纠正噪声的可视化出现模糊/块状伪影。不是把预测 clean 的后验协方差设零；该消融也不等于 RDD fresh 补偿消融。 |
| [PyramidalWan](https://arxiv.org/html/2601.04792v1)，§3.2、附录A1.2–A1.3 | 将带噪状态作为低频带，正交变换中填充 Gaussian 高频，配套 natural/global noise clock | 所定义端点的噪声变换可解析匹配；关于原高频已被噪声掩盖的假设是近似，作者明确平均频谱选边界不能保证每个样本都满足，故训练匹配转换后的输入。 |
| [PixelFlow](https://arxiv.org/html/2504.07963v1)，§3.2–3.4 | 引用 renoising 处理跳点；作者源码先 nearest 上采样整个 noisy state，再缩放并加入块相关噪声 | 没有在所查部分给出 RDD fresh 后验缺项可忽略的结论。不能仅凭其未估计 C(z)，就说它忽略了同一个问题。 |
| [f-DM](https://arxiv.org/html/2210.04955v1)，§3.1、附录A.4 | 设计 dropping/averaging 噪声变换的逆，补充缺失 Gaussian 分量并重标定 SNR | 明确给出边界噪声方案；报告 averaging 逆经验上 FID 较好，同时指出相关噪声可能带来 DDIM 偏差。并非笼统断言边界协方差不重要，也没有在所查推导中直接估计 clean 后验 C(z)。 |

PixelFlow 本地源码核查点：`sources/pixelflow/pixelflow/pipeline_pixelflow.py` 的 `sample_block_noise` 与 stage transition；`sources/pixelflow/pixelflow/scheduling_pixelflow.py` 的 gamma/rectify ratio。生产共同实现的对应路径为 `umdiff/multires_process.py` 的 PixelFlow stage transition。本轮只读，没有重跑 parity。

## 与 RDD fresh 的关键区别（本轮分析）

我们的已知反例研究的是：

\[
Z_c=\alpha X_c+\sigma\epsilon,\quad
Y_{fresh}=\alpha U E[X_c\mid Z_c]+\sigma\eta.
\]

相对目标 alpha U Xc + sigma eta，理想条件均值下的协方差缺项为 alpha² U E[Cov(Xc|Zc)] U^T。

Pyramid Flow 路线则可抽象为（以下为对原公式的统一记号重写）：

\[
Z_c=eX_c+(1-e)\epsilon,\qquad
Y=\frac{s}{e}UZ_c+\xi,
\]

\[
\operatorname{Cov}(\xi)=(1-s)^2I-
\left(\frac{s}{e}\right)^2(1-e)^2UU^T\succeq0.
\]

若该矩阵半正定，且 xi 是独立 Gaussian，该核可将正确 coarse 边缘分布推到所设计的 fine 入口边缘分布，不需要求 clean 后验；实际网络和积分误差另计。这个证明不要求 ODE 保留训练时某个特定 clean/noise 配对。

对于重复上采样一个值到4个值，PSD 条件要求 s<=e/(2-e)，论文选择等号。它正是配套端点关系，而不是随意在固定同一时间上加噪。

因此不能直接将该核塞到当前固定 alpha/sigma 的 RDD checkpoint 并声称数学不变。若保持 s=e 且 U 为重复上采样，所需补噪矩阵含 I-UU^T，通常不半正定。可探索调整时钟或 coarse 归一化，但会影响训练路径/兼容性。

## 论文可写与不可写

- 可写：现有 fresh 是后验均值近似；指出准确的协方差缺项，报告当前简化的真实实测效果。
- 可写：固定/对角协方差是已有做法；反向方差研究显示收益依赖步数、模型误差和评价指标。
- 不可据此写：别人都忽略所以本方法影响很小；协方差推理时不可获得；补准协方差必然改善 FID；我们的 fresh 已严格匹配全分布。
- 估计后验方差并补偿，与重新设计 noisy-state 边界，是两条不同路线。前者可以先固定 checkpoint 消融；后者需重新审查时钟、表示和训练端点。

这些结论没有新增生成质量证据。此前解析 Gaussian 方差反例不代表真实图像/视频质量损失比例。
