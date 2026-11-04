# 多分辨率生成的等价性、训练瓶颈与 RDD 后续设计

日期：2026-10-11。问题：各方法在哪些层面等价、差异如何影响训练/生成、RDD 为什么值得选择，以及是否存在性质更好的配置。本次是数学与设计分析，附 CPU 合成代数检查；不宣布新算法已取得更好 FID，不修改生产实现，不启动 GPU 或正式训练。

## 1. 主要判断

多分辨率方法可以由共同对象描述，但一般不是同一个算法换名字。最关键的区别是：它们改变了概率过程、训练度量、跨尺度联合随机过程，还是只改变模型处理同一状态的方式。

RDD 有清晰的实用理由：阶段内路径简单，目标仍监督被移除的细节，effective-noise 恢复完整 clean 不需要除以消失的高频信号系数，容易沿用单状态骨干。另一方面，其路径属于已有广义族、目标可以写成加权 clean 重建，默认 fresh-noise 边界也不是唯一选择；这些都不能推出 RDD 普适最优。

更值得优先比较的配置是：保留带噪粗状态、只给新增正交维度补噪，并用各频带归一化的目标和误差/成本指导时钟。它具有更明确的状态连续性和目标坐标条件数，但这些优势不等于全网络优化或最终质量一定更好。相关构件已有 f-DM、DFM/LapFlow、Fresco、谱扩张、v-prediction 与预条件等先例，不能把组合直接宣称新算法。

## 2. 五种等价性，必须分别判断

| 等价层级 | 需要满足什么 | 不能从中推出什么 |
|---|---|---|
| 表示等价 | 存在明确可逆坐标映射，并正确推送信号、噪声、输出与度量 | 相同网络架构仍有相同容量或优化轨迹；有损 VAE/截频不属于可逆换坐标 |
| 条件边缘等价 | 在共同表示/时间下，F(t,x) 与 C(t) 相同，即 q(z_t\|x) 相同 | 多时刻或跨尺度的联合分布、反向条件核相同 |
| 训练泛函等价 | 相同输入分布；目标变换、输出读出、度量、采样权重都匹配 | 随意更换独立网络的 raw head 后梯度仍相同 |
| 生成过程等价 | 边界联合核、ODE/SDE/离散转移、条件信息、时间与 solver 对应 | 任意不同离散步数仍逐像素一样 |
| 数值执行等价 | 以上之外，架构、初值、数据、RNG、精度、归约、优化器、EMA、CLI 都匹配 | 未测分支、其他平台或完整最佳成绩也相同 |

对于线性 Gaussian 路径 z_t=A_t x+B_t ε，条件边缘只依赖 A_t 与 C_t=B_tB_tᵀ。若 B_t′=B_tR_t 且 R_t 为正交矩阵，每个时刻边缘相同；若 R_t 随时间变化且使用固定同一 ε，跨时刻协方差 B_sR_sR_tᵀB_tᵀ 可以改变。因此连 B Bᵀ 都相同，也不能省去联合耦合 K。

### 2.1 目标与损失

若给定状态 z，T′=L_tT+c_t(z)，且预测也按同一方式变换，要保留二次损失 W，应取 W′=L_t⁻ᵀW L_t⁻¹。如此同一参数化的梯度才对应。不变换度量时，目标名字的改变可能是在改变频带权重。

时间也一样。若 τ=φ(t)，u_t=φ′(t)u_τ；直接保持未加权 velocity MSE 会产生 φ′² 的权重变化，同时采样分布还涉及时间密度的 Jacobian。模型条件时间、路径时间和 solver 时间应分开记录。[PixelFlow 的端点构造](https://arxiv.org/html/2504.07963)是需要这样核对的例子。

### 2.2 多尺度表示的噪声反例

考虑一个四像素 block，D 是平均、U 是 repeat、P=UD、N=I−P。把完整图像写成 coarse c=Dx 与 detail h=Nx，重建 x=Uc+h。由同一白噪声分解出来的 noise 满足

\[
\operatorname{Var}(D\epsilon)=\tfrac14,
\quad \operatorname{Cov}(N\epsilon)=N,
\quad D(N\epsilon)=0.
\]

detail 有四个坐标，但只含三个自由度。若改为独立 c_noise∼N(0,1)、h_noise∼N(0,I₄)，重建噪声协方差是 UUᵀ+I₄，特征值为 {5,1,1,1}，而非白噪声 I₄。两者都是可定义的生成设计；后者不是无效模型，但不是同一 pixel noise law。

用 Q=U/2 和三个正交 detail 坐标则可保持独立标准 Gaussian 与单位重建协方差。因此，Laplacian product-state、非冗余正交 band state 和原图状态不能不检查噪声就宣称等价。当前源码的 `LapFlowNative` 两层与三层噪声分支也不同；本分析不能替代逐分支作者认证。

## 3. 各论文主要区别在哪里

| 方法家族/论文 | 主要改变的对象 | 可能影响训练与生成的原因 |
|---|---|---|
| f-DM、Edify、RDD；SSD 的广义线性退化子族 | 信号衰减/变换，配套噪声、目标、反向过程 | 改变各频带何时可辨识及网络的学习任务；条件边缘匹配后仍需核对联合过程。SSD 的 Markov 构造与 f-DM 的 clean-conditioned 转移不能自动视为相同 |
| [Relay](https://arxiv.org/abs/2309.03350)、[PixelFlow](https://arxiv.org/html/2504.07963)、Pyramid Flow | 带噪端点、跨尺度 lift、重加噪、时间对齐等 | 边界误差与噪声耦合会影响下一阶段输入，少步时校正预算更少；具体公式各自不同 |
| [DFM](https://arxiv.org/html/2506.19839)、[LapFlow](https://arxiv.org/html/2602.19461) | product-state、分解、每尺度时钟/激活、共享或因果网络 | 更灵活地安排粗细任务；代价是多输出/多时钟/联合噪声分布和参数共享的选择。它们不是 RDD 单状态轨迹的无条件等价写法 |
| [MDM](https://arxiv.org/html/2310.15111) | 多分辨率联合状态、NestedUNet、联合监督与 progressive training | 共享特征和多尺度损失可改变优化；其状态含多个尺度，不能只替换 upsample 就变成 RDD |
| [DiMR](https://arxiv.org/html/2406.09416) | 多分辨率 backbone、时间条件与辅助预测 | 多尺度主要发生在网络/监督；不能据此推断概率路径也阶段式降低分辨率 |
| [PPFlow](https://arxiv.org/html/2506.23543) | patch 大小、输入/输出投影和时段分配 | full-resolution diffusion/flow 状态仍保留，靠减少 token 节省计算；是隔离“少 token”收益的重要对照。其 learned patch 投影不保证可逆，不能与状态的可逆 packing 混淆 |
| [DCTdiff](https://arxiv.org/abs/2412.15032) | DCT 表示、坐标截取/量化、统计缩放与频率损失 | 完整正交变换可以换坐标；截频与 codec 会改变任务/可重建信息，pixel 与 coefficient MSE 也须匹配归一化 |
| [Diffusability / scale-equivariant AE](https://arxiv.org/html/2502.14831) | 表示 E 与 decoder 的尺度/频谱性质 | latent 低频不一定对应 pixel 粗语义。表示不对齐时，漂亮的投影代数也不能保障语义被保留 |
| [RMD](https://arxiv.org/html/2603.06136) | 蒸馏目标、teacher/fake/generator 与跨分辨率分布匹配 | 训练目的不再只是同一个 denoising MSE；teacher 知识与多网络成本不能忽略 |
| [Fresco](https://arxiv.org/html/2601.07462)、[SPD](https://arxiv.org/html/2605.18736)、[AViTS](https://arxiv.org/html/2608.17995) | 跨尺度噪声一致性、谱扩张/时钟和按需 token 计算 | 强调预训练模型与推理预算；已有保留噪声、正交扩张和自适应尺度先例。author-reported 收益不是本次独立质量认证 |

这张表表示主要差异的分组，不是逐对完整等价性证明。任何精确等价主张都需要给出所选分支、表示/归一化、时间、噪声与目标/损失的映射。

## 4. 为什么 RDD 是一个合理选择

### 4.1 延迟高频信号，同时仍监督完整内容

本节 effective-noise 的推导针对 VP/drift 版本的主头。RDD flow 训练的是 velocity，不能直接套用该主头的 loss 恒等式；它的高带系数为 a_h=t(t−s_k)/(s_{k+1}−s_k)，velocity 为 [(2t−s_k)/(s_{k+1}−s_k)]x_h−ε_h，同样需要检查时钟与频带权重。

设正交 P，低频系数 α、高频系数 a_h=αδ、δ=1−λ，噪声系数 σ>0。高频状态为 z_h=a_h x_h+σ ε_h。当 δ=0 时，ε_h=z_h/σ 已经是输入的确定函数；pure-epsilon loss 无须学习 x_h 的内容。

RDD 的 effective target 是

\[
\epsilon_{\rm eff}=\epsilon-\frac{\alpha\lambda}{\sigma}(I-P)x
=\frac{z-\alpha x}{\sigma}.
\]

即使 a_h=0，只要 α、σ 非零，它仍包含被移除的 detail。模型可从保留下来的低频/条件信息学习高频的条件均值；无法凭空确定不可预测的随机细节。其 clean 读出

\[
\hat x=(z-\sigma\hat\epsilon_{\rm eff})/\alpha
\]

不需要除以 a_h 或 δ。相比在相同 RDD 路径上用 pure epsilon 再反演，这一语义与读出更合适。这个理由不是“RDD 优于所有 flow”：velocity、full-clean 或合适的 band-v 目标也可以保持细节监督。

### 4.2 effective-noise 正好是加权 full-clean 重建

给定同一 z，若从 clean 预测 \(\hat x_\theta\) 构造 \(\hat\epsilon_\theta=(z-\alpha\hat x_\theta)/\sigma\)，则

\[
\|\hat\epsilon_\theta-\epsilon_{\rm eff}\|^2
=\frac{\alpha^2}{\sigma^2}\|\hat x_\theta-x\|^2.
\]

这个关系在 δ=0 也成立，与“pure-epsilon ↔ effective-epsilon”的奇异逆是两件事。它给出更准确的解释：RDD 的 effective-noise head 是一种带 skip/readout 的、按 SNR 加权的完整重建参数化。

这里仅比较该主头的二次项；extra drift head、learned-variance 项、其他辅助 loss 和它们的梯度归约不自动等价。

同一底层参数化和读出时，可把损失与参数梯度严格匹配；两个独立初始化的 raw clean/epsilon 网络则不必有相同优化轨迹。RDD 的新颖性或收益不能只归功于发明了这个目标标签。

### 4.3 相对权重变大不等于目标必然发散

在 δ>0 时，effective/pure 目标变换的高频系数有 1/δ。它说明未加权损失度量不同，不证明 effective 目标本身随 δ→0 发散。

若 x_h、ε_h 独立、x_h 方差 s_h²，effective 目标方差为 1+(αλ/σ)²s_h²。需要检查的是 αλ/σ 和真实数据分布；当 σ>0 固定，δ=0 仍有限。现有 schedule 是否受控必须具体核对，不能从一般最坏情况断言当前 RDD 不稳定。

Gaussian toy 中

\[
\operatorname{Var}(\epsilon_h|z_h)
=\frac{a_h^2s_h^2}{a_h^2s_h^2+\sigma^2},\quad
\operatorname{Var}(\epsilon_{{\rm eff},h}|z_h)
=\frac{\alpha^2s_h^2}{a_h^2s_h^2+\sigma^2}.
\]

前者随 a_h→0 消失，后者保留监督；但非奇异点上两种 Bayes-optimal 预测经正确 clean 读出后具有相同 clean Bayes risk。有限网络、目标度量与计算预算才是判断训练收益的关键。

### 4.4 简洁性与目前边界

RDD 的单状态路径、解析系数和现有共享骨干，降低了多头/多时钟维护负担，也便于分析。低分辨率早期计算可以省 token，但这不是独有性质；必须对 PPFlow 等 token-only 配置作公平对照。

当前 `rdd_fresh` 边界重新合成 αU x̂_c+σ ε_new；`isometric`、legacy posterior 是不同配置，不应都叫同一个 RDD。fresh 边界对 clean 估计和新噪声依赖更直接，跨尺度轨迹连续性不是它的自动保证。简洁与现有训练信号使 RDD 值得保留为基线，并不要求它成为框架里最终最好的配置。

## 5. 这个领域更实际的痛点

以下归纳来自所查论文的方法设计、讨论与失败试验，以及框架的数学合约；不是对研究者群体的统计调查。

1. **省计算与保留细粒度信息冲突。** 高频平均能量小，不代表文字、小物体、数量或快运动不重要。全局功率谱时钟与 token 方差只是代理指标；SPD 的 Gaussian/power-spectrum 假设、AViTS 的语义重要性设计分别说明不同侧面。[SPD](https://arxiv.org/html/2605.18736)、[AViTS](https://arxiv.org/html/2608.17995)
2. **oracle 边界正确，有限模型输入仍有偏移。** 实际 x̂、noise estimate 带误差；teacher-forced 粗状态与生成粗状态不同。新噪声还改变参考轨迹耦合；Fresco 已针对一致性提出方法，不能把该问题首次发现归于本项目。[Fresco](https://arxiv.org/html/2601.07462)
3. **粗阶段错了，细阶段未必修得回来。** 冻结或单向依赖保持结构，也可能锁死错误构图/语义。允许粗细联合修正更灵活，但需计算预算和训练分布支持。联合建模与跨尺度独立专家已有相关比较。[MDM](https://arxiv.org/html/2310.15111)、[DFM](https://arxiv.org/html/2506.19839)
4. **target、clock、归约和采样比例混在一起。** 相同 batch/steps 并不表示相同训练任务或计算。只换 logit-normal、head 或按 token 平均就可能改变结果；需要拆开测，而非把所有收益归于路径。
5. **latent 尺度并不天然具有 pixel 语义。** 可以测 decoder(D_latent E(x)) 与 D_pixel(x) 的误差；正交 latent transform 只保证 latent 代数，不能保证语义。codec 是共同基线的一部分。[Diffusability](https://arxiv.org/html/2502.14831)
6. **数学更整齐的坐标可能更难被网络利用。** 全局 DCT/独立 band 输入可能削弱局部结构或 attention 的相似性归纳偏置。DFM 附录 F 报告了直接 DCT-space 模型效果较差的失败试验。可以在正交坐标管理过程，但把 active state 还原空间布局送给骨干；不能从“正交”直接推断效果。[DFM 附录 F](https://arxiv.org/html/2506.19839)
7. **少 NFE 不等于低延迟。** token 数、CFG、transform、VAE、SR、controller、不规则 attention 和 kernel 利用率都要计入；跨尺度跳变还会削弱高阶 solver 的收益。
8. **视频存在额外的时间与因果限制。** 空间噪声一致不等于运动正确；直接所有帧复用同一噪声会改变时间协方差。时间降采样可能删除快运动，非因果 3D wavelet 不适合直接套入严格因果生成。这些是设计风险，不是本次已证明某篇论文必然存在的错误。

## 6. 框架需要产生什么新的分析能力

最有用的数学核心应给出过程/表示、目标/度量和边界的兼容条件，而不只是 14 个可以填任意函数的字段。建议强化四种合约：

- **表示/噪声合约**：有损还是可逆？冗余还是独立坐标？noise covariance 是否由相应表示正确推送？
- **目标/时钟合约**：哪里可逆、哪里只能预测条件期望？解析读出与 loss/time Jacobian 是否匹配？
- **边界合约**：匹配的是边缘分布、联合耦合还是 coarse-state 连续性？结论用了 oracle clean 还是有限网络预测？
- **预算/误差合约**：成本包括什么？哪个 stage 的误差会传播到最终结果？比较的是算法变化还是 loss/容量/codec 变化？

这些不是宣称当前所有 runtime 检查已实现；当前框架一些边界和目标已经记录部分条件，完整自动兼容性判定仍需实现。任意组合可表达也不等于任意组合有效。

在共同嵌入和匹配参考随机耦合下，Lipschitz ODE 的误差可按常见稳定性估计分为初值误差、向量场预测误差、数值积分误差和阶段跳变误差：后续 Lipschitz 放大系数乘上每项误差，再累加。它不是新定理，但可以指导实验：测哪些部分主导最终误差，避免只证明 oracle 的边缘匹配。

路径距离上界也不能直接变成 FID 改善保证；重新耦合可能改变与参考样本的距离而不改变边缘分布。需要同时观察分布指标、语义、recall 与预算。

## 7. 一个值得优先实现比较的候选配置

暂称为“保留粗状态的正交多尺度 flow”，不作新颖性主张。核心目标是让边界不经过有误差的 clean 重建，并让监督在各 band 端点都易于定义。

### 7.1 边界直接 lift 带噪状态

设 QᵀQ=I，P=QQᵀ。coarse 使用正交坐标 c=Qᵀx，而不是未归一化的 block average。粗细阶段低带信号、噪声系数已经对齐，新增 detail 的信号系数在激活时为零，则取

\[
z_f=Qz_c+\sigma(I-P)\eta,\qquad\eta\sim\mathcal N(0,I).
\]

立刻有

\[
Q^\top z_f=z_c.
\]

若 oracle coarse noise 为独立标准 Gaussian，fine noise 的协方差为 σ²[P+(I−P)]=σ²I。这个 lift 不需要先预测 x_c 或 ε_c，故不额外把这些估计误差写入边界。

fresh 重建在相同正交坐标中为 z_f′=αQ x̂_c+σ ε_f，低带跳变量是 α(x̂_c−x_c)+σ(Qᵀε_f−ε_c)。即使 oracle x̂_c=x_c，独立噪声下其期望平方为 2σ²d_c。这里比较的是与同一 coarse state 的路径扰动，不是一个 FID 或最优分布证明。

**三个限制**：真实 coarse state 已经有错时，直接保留也会保留错误；普通图像平均坐标必须先归一化并对齐时钟；若 fine 高带端点已有非零信号，这条公式不是完整目标边缘，不能仍声称精确。

这不是把当前 `isometric` 的 rho 改成 1 就完成：它的 mean 使用 repeat(clean)，本设计要求匹配坐标和整个带噪状态的 lift。f-DM 的 Gaussian noise completion、Fresco 的 Hadamard expansion 和 SPD 的谱补噪均有先例。新增价值只能来自更完整的误差处理或可靠收益。[Fresco](https://arxiv.org/html/2601.07462)、[SPD](https://arxiv.org/html/2605.18736)

### 7.2 各 band 用归一化角度目标

设一条 band 的信号标准差 s_j>0，w_j=x_j/s_j，原路径 z_j=a_jx_j+σ_jε_j。令

\[
r_j=\sqrt{a_j^2s_j^2+\sigma_j^2},\quad
c_j=a_js_j/r_j,\quad d_j=\sigma_j/r_j,
\]
\[
y_j=z_j/r_j=c_jw_j+d_j\epsilon_j,\qquad
v_j=c_j\epsilon_j-d_jw_j.
\]

当 r_j>0，(w_j,ε_j)→(y_j,v_j) 是二维正交旋转，坐标变换条件数为 1；逆是 w_j=c_jy_j−d_jv_j，ε_j=d_jy_j+c_jv_j。高带 a_j=0 时 v_j=−w_j，仍监督 detail，不需要除以 a_j。独立、中心化、单位方差的 w_j/ε_j 下 target 方差为 1。

这个 v_j 与当前 generic `Joint('v')` 在 mixed band 路径上用单一 alpha/effective_clean 的定义不同；不能静默复用名字或 decoder。它在每个普通 leaf path 上更容易实现，需要明确 band-specific statistics、归一化和输出结构。统计只能来自训练集或已授权的校准数据，测试真图不能成为生产输入。

写 c_j=sinθ_j、d_j=cosθ_j，noise→data 方向的 physical velocity 为

\[
u_j=\dot r_jy_j-r_j\dot\theta_jv_j.
\]

网络预测 v_j，sampler 用完整 r_j、θ_j 和导数恢复速度。可以选择平滑的 band 时钟，早期少计算高带，后期让粗细一起继续修正。

这保证的是**坐标变换条件数**，不是神经网络 Hessian 条件数或收敛率。改变 target 而沿用无权 MSE，也会改变原 RDD 的 clean-loss 权重；需要 matched-metric 对照。角度路径可能更弯，少步 Euler 也可能不如直线路径，所以普通 linear FM + PPFlow 应保留为强对照。

### 7.3 网络布局优先保留空间归纳偏置

正交坐标用于定义过程、噪声和目标，active state 可转换回空间布局进入共享 DiT/UNet。逐尺度 clocks、head 或 patch 可作为插件，但成本和容量须对齐。完整正交表示不能默认直接改善 attention；必须与空间 Laplacian 和 token-only 方法比较。

无需先加入全部 adaptive controller。第一版固定两阶段，只比较状态 lift 和目标/度量即可检验主要假设。视频第一版只改变空间尺度，保持时间网格及既有 causal contract。

## 8. 更可能形成研究新意的方向：依据实际误差分配计算和修正粗状态

### 8.1 每阶段步数由误差与真实成本决定

若采用明确的 surrogate E(n)=Σ_j A_j/n_j^p，预算 Σ_j c_j n_j=B，其中 A_j 包含局部曲率及后续误差放大，c_j 是一次模型调用的实际成本，则连续最优解为

\[
n_j\propto(A_j/c_j)^{1/(p+1)}.
\]

这只是常见资源分配计算，不是任意生成器的最优定理。值得研究的是怎样低成本估计实际 A_j：用嵌套 solver 的局部误差、校准后的预测不确定性和低频语义变化，而非只用数据平均功率。它必须胜过固定时钟、SPD 及 AViTS，并计入 controller 开销。真实误差若被模型偏差主导，增加 n_j 可能无效。

训练的 stage 采样也应区分目标改变和估计方差改善。给定每任务梯度 second moment m_j、成本 c_j，重要性权重正确补偿时，p_j∝√(m_j/c_j) 最小化“每步成本 × 梯度 second moment”的上界代理。这不是减去均值后的真实总方差的普适最优解；若不补 1/p_j，便改变训练目标本身。

### 8.2 允许经过校准的粗状态修正

只保证 Qᵀz_f=z_c 可能锁住粗错误。更有价值的扩展是：保留连续初始化，但允许 fine 模型继续估计低带，并按可靠性决定相信 coarse cache 还是 fine 预测。

两个 clean 估计的误差 e₁、e₂，若已知真实条件 second moments R₁=E[e₁²]、R₂=E[e₂²]、C=E[e₁e₂]，合并 λ x̂₁+(1−λ)x̂₂ 的平方误差为

\[
R(\lambda)=\lambda^2R_1+(1-\lambda)^2R_2+2\lambda(1-\lambda)C,
\]
\[
\lambda^*=\operatorname{clip}_{[0,1]}
\frac{R_2-C}{R_1+R_2-2C}.
\]

非退化分母下，这个凸组合族内的最优风险不大于单用任意一项；退化时两个误差相同，选一个即可。这里 second moments 可包含偏差，不假设两个模型误差独立。

这只是正确校准下的重建 MSE 性质，不是 FID、分布或细粒度语义的保证。实际 coarse/fine 误差高度相关；只估计各自 variance 而忽略 C，或根据不可靠的 uncertainty 设 λ，可能更坏。将混合 clean estimate 接入 sampler 时，必须重新定义和训练相应 conditional process/readout；不能测试时任意 blend 后仍宣称原 ODE 的精确采样。

这个方向的实质是有限模型误差与跨尺度修正，而非再次包装 Gaussian conditioning。若框架能定义相应 cache/条件过程，并验证可靠性估计能指导预算或修正，它比单独替换 upsample 更有研究价值。校准与 rollout 训练的 effort 较大，应该先用固定设计确认边界确实是瓶颈。

## 9. 怎样公平地决定是否替换 RDD

建议在同一数据、codec、骨干容量和真实计算预算下，先做下列小矩阵，而非同时训练全部复杂组合：

| 配置 | 要隔离的问题 |
|---|---|
| 普通 linear FM / diffusion + 固定 token | 基础模型 |
| 普通路径 + PPFlow 式 token schedule | 节省早期 token 本身能解释多少收益 |
| RDD native 目标/边界 | 当前有效配置 |
| 相同 RDD 路径 + full-clean 的匹配读出和 SNR metric | 检验 effective head 的等价解释，不改变目标泛函 |
| 相同 RDD 路径 + band-normalized target；分别配原等价 metric / 自然 metric | 区分坐标预条件和损失改变 |
| 匹配坐标/时钟的 no-reset lift + 原目标 | 隔离边界扰动 |
| no-reset + band-normalized target + 固定两阶段 | 检验候选组合是否超过单项收益 |
| 仅在前项有效后，加入 calibrated correction / error-aware budget | 判断新机制是否值得复杂度和额外成本 |

数学等价项应先在小 CPU 配置核对。质量项随后需要已授权真实资产和训练预算，本次没有执行。先看各 band 的 clean 误差、目标方差/梯度、边界低带跳变与自由运行偏移，再看到达预定质量所需成本、FID/recall、语义和少步曲线。视频还需运动与时间一致性，不能仅比较 frame-FID。

如果 PPFlow 消除 RDD 优势，应把贡献归于计算布局；如果 matched clean/metric 对照解释全部目标收益，应把贡献归于参数化/度量；如果更简单的连续 lift 配置稳定更好，就让框架推荐它。一个真正有用的统一框架不需要预先规定 RDD 必须获胜。

## 10. 本次证据与限制

已执行 [CPU 代数脚本](evidence/design_analysis_20261011/check_design_algebra.py)，[14 组检查结果](evidence/design_analysis_20261011/ALGEBRA_CHECKS.json)全部通过：正交 covariance、冗余表示噪声反例、端点监督/Bayes risk、effective target 的 full-clean 等价、目标/metric toy 参数梯度、跨尺度 coupling、连续 lift、角度坐标、时钟缩放、surrogate budget、importance gradient 与相关误差融合。

这些是浮点容差下的小矩阵/解析例检查，不是作者网络 activation/backward parity 或新颖性证明。匹配 metric 的 toy 参数梯度误差约 1e−14；目标旋转条件数约 1。第一次 importance expectation 检查因零值使用 atol=0 而在 4.44e−16 舍入误差处失败，已改为明确 atol=1e−14，再运行通过；未改生产代码或算法来掩盖不一致。

复跑：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python unified_multires/evidence/design_analysis_20261011/check_design_algebra.py`。

来源与源码哈希见 [PROVENANCE.json](evidence/design_analysis_20261011/PROVENANCE.json)。本次正文主要阅读方法/失败试验相关段落，并核对本地 paths/targets/boundaries/noise/representation；没有逐行重新审完所有作者代码，没有下载模型/数据，也没有开展新配置真实训练或 GPU 测速。实际主结果/原生认证状态仍见 [未完成事项清单](ORIGINAL_UNIFIED_GAPS_TODO_ZH.md)。
