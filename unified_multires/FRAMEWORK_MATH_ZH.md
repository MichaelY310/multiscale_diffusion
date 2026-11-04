# 多分辨率 diffusion / flow 的统一组件框架

日期：2026-10-08。实现与逐篇证据见 [README_ZH.md](README_ZH.md) 和 [REPRODUCTION_REPORT_ZH.html](REPRODUCTION_REPORT_ZH.html)。

## 1. 一个方法应当由哪些东西定义

仅列「加噪 / target / upsample」不足以确定一个方法。这里把方法写成

\[
\mathcal M=(E,\Pi,\{F,B,K,\phi,\rho,H,T,W\}_k,\{\mathcal Q_k\},S,\mathcal O,\mathcal D).
\]

|符号|含义|必须记录的细节|
|---|---|---|
|\(E\)|图像、latent、DCT、Laplacian 或 product-state 表示|颜色空间、codec、量化、缩放、频率截断、各叶节点顺序|
|\(\Pi\)|模型看到的 token/layout|有损 resize、可逆 channel↔space、patch、位置编码、attention mask、split/merge|
|\(F_k(t,x)\)|阶段内干净信号路径|下/上采样算子、滤波、band activation、端点|
|\(B_k(t)\)|噪声线性变换|协方差、每尺度 SNR、归一化、抖动项|
|\(K\)|跨尺度、跨时刻噪声联合耦合|同一噪声分解、独立噪声、保留旧噪声、额外 fresh noise|
|\(\phi_k\)|本地时间到模型条件时间的映射|noise→data 或 data→noise、离散索引、shift、单位转换|
|\(\rho_k\)|训练采样分布与 active mask|阶段比例、uniform/logit-normal、异形 batch packing|
|\(H_k\)|模型和参数共享|U-Net/DiT/MoT、shared core、分辨率头、progressive depth、条件/CFG|
|\(T_k\)|预测参数化|clean、pure/effective epsilon、velocity、v、多个 decoupled heads、variance|
|\(W_k\)|损失坐标、权重和 reduction|按图像还是 token 平均、是否按 band 归一化、SNR/EDM 权重、辅助损失|
|\(\mathcal Q_k\)|阶段边界随机核|状态、时间、噪声、额外缓存、codec 是否一起变化|
|\(S\)|ODE/SDE/discrete 求解器|步数含义、步长、corrector、posterior variance、CFG 次数|
|\(\mathcal O\)|训练过程|初始化、optimizer、LR、clip、EMA、精度、RNG、checkpoint|
|\(\mathcal D\)|数据与预处理|相同原始数据不能替代相同 crop/resize/VAE/latent scaling|

这个定义允许相同路径使用不同网络，也允许同一个网络处理不同路径。**换骨干属于 controlled transplant；使用作者骨干、原版算子和原版采样规则的数值对照属于 source-native conformance。** 两者在配置和报告中分别标记。

## 2. 信号、噪声和速度场

最一般的条件构造可以是

\[
z_{k,t}=F_k(t,E(x),\xi),\qquad
u_{k,t}=\partial_t F_k(t,E(x),\xi).
\]

这里 \(\xi\) 可以包含多次随机抽样、跨尺度共享随机变量和非高斯随机变量。当前内置连续路径主要实现线性高斯子族：

\[
z_t=A_t x+B_t\epsilon,\quad \epsilon\sim\mathcal N(0,I),
\quad C_t=B_tB_t^\top,
\quad u_t=\dot A_t x+\dot B_t\epsilon.
\]

多个尺度可用 product state \(z=(z^{(0)},\ldots,z^{(L-1)})\)，每个叶节点有自己的时间和 active mask。模型输出可为 tuple/list/dict；dict 的 key、尺度顺序及模型输出结构都必须匹配。

这里的 \(\partial_tF\) 用于可微的条件 flow 路径，不能当作 Brownian 轨迹的普通导数。连续 diffusion 若写为 \(dz=f(z,t)dt+G(t)dW_t\)，且扩散矩阵不依赖状态，其 probability-flow ODE 为 \(dz=[f-\tfrac12GG^\top\nabla_z\log p_t]dt\)；反向 SDE 在反向时间积分下使用 \(f-GG^\top\nabla_z\log p_t\) 与反向 Brownian 项。边缘构造中的 \(B_t\) 是累积噪声变换，一般不等于瞬时扩散系数 \(G(t)\)。离散 DDPM 与 SSD 则直接定义条件转移及 posterior，不依赖对随机轨迹求普通导数；当前内置 sampler 覆盖各自发布的离散/ODE规则，通用连续 SDE 求解器需作为插件增加。

设 \(P=UD\) 是正交投影，\(P^2=P=P^\top\)，可写为

\[
A_t=a_\ell(t)P+a_h(t)(I-P),\quad B_t=\sigma(t)I.
\]

代码对应 `BandPath.coefficients` 返回六个量：
`a_low, a_high, sigma, da_low, da_high, dsigma`。

`average D + nearest U` 在非重叠整数 block 上满足上述投影条件；`bilinear D + nearest U` 一般不满足。**不能把后者套进正交投影的解析逆。** SSD 使用真实 adjoint 和算子形式 posterior；PixelFlow 使用端点速度定义，避免把 bilinear 算子冒充正交投影。

### RDD 与相关路径

RDD 离散 VP 的阶段内边缘为

\[
z_t=\alpha_t\{x+\lambda_t(Px-x)\}+\sigma_t\epsilon,
\quad a_\ell=\alpha_t,\quad a_h=\alpha_t(1-\lambda_t).
\]

当前 RDD flow 为

\[
z_t=t x+(1-t)\epsilon+b_k(t)(P-I)x,
\quad b_k(t)=\frac{t(s_{k+1}-t)}{s_{k+1}-s_k},
\]
\[
u_t=x-\epsilon+b'_k(t)(P-I)x,
\quad b'_k(t)=\frac{s_{k+1}-2t}{s_{k+1}-s_k}.
\]

若 f-DM 的变换插值取 \(F_t=(1-\lambda_t)I+\lambda_tP\)，并匹配 \(\alpha,\sigma\)，其阶段内条件边缘可以与上面的 RDD VP 路径重合。这个等式不意味着 target、损失、边界联合过程或采样器重合。闭源 f-DM 的组件示例标成 paper-component transplant，不标成 native 复现。

PixelFlow 用本地 \(\tau\) 在两个带噪端点间插值：

\[
z_\tau=(1-\tau)[aPx+(1-a)\epsilon]+\tau[b x+(1-b)\epsilon],
\]
\[
u_\tau=b x-aPx+(a-b)\epsilon.
\]

若改成全局时间 \(t=a+\tau(b-a)\)，速度必须乘 \(d\tau/dt=1/(b-a)\)。PixelFlow 的 model timestep 又是另一张映射表；三个时间不能混用。

LapFlow 的延迟 band 路径默认

\[
z_t^{(j)}=\frac{t-a_j}{1-a_j}x^{(j)}+(1-t)\epsilon^{(j)},\quad t\ge a_j.
\]

DFM 则使用 data→noise 方向，多个 band 独立 noising、分阶段 drop mask、早期 band 的 logit-normal 时间。MDM 使用联合多尺度状态、尺度相关 SNR；DiMR 的输入仍是普通 full-resolution VP，multiresolution 主要在 backbone 和辅助 target，不能误称为阶段式低分辨率采样。

## 3. target 不是一个没有代价的标签

令 \(\bar x_t=A_tx/\alpha_t\)，当可定义时：

\[
\tilde\epsilon_t=\epsilon+\frac{A_tx-\alpha_tx}{\sigma_t},
\quad z_t=\alpha_tx+\sigma_t\tilde\epsilon_t.
\]

内置 `joint.quantity` 有 `velocity / epsilon / effective_epsilon / clean / effective_clean / v`，其中

\[
v_t=\alpha_t\epsilon-\sigma_t\bar x_t.
\]

\(v\) 是这个明确约定下的旋转参数化；一般不等于 \(\partial_tz_t\)。普通 VP 的 \(v\) 关系也不能不经检查地搬到 band 路径。

`decoupled_clean` 输出 `signal=bar_x`、`residual=x-bar_x`，两者相加得到干净样本。`rdd_decoupled` 保留 RDD effective-epsilon 主头，另加 raw drift 头；它与前者的语义不同。learned variance 是额外输出，不属于把 signal/drift 解耦。

在正交 band 路径上，给定 state 与 velocity：

\[
y=\dot\sigma z-\sigma u,
\quad x=\frac{Py}{\dot\sigma a_\ell-\sigma\dot a_\ell}
+\frac{(I-P)y}{\dot\sigma a_h-\sigma\dot a_h}.
\]

分母为零时信息不可辨识，实现报错；不使用任意 epsilon clamp 改变目标。共同零点的 effective-clean 使用可定义的导数极限。测试用有限差分检验速度，并检验不同 target 的解析逆。

对非各向同性高斯噪声，条件 score 是

\[
\nabla_z\log q(z|x)=-C_t^{-1}(z-A_tx).
\]

因此「预测 epsilon」与「预测 score」的转换需要完整 \(B,C\)，只除一个标量 sigma 一般不够。非线性 codec 的 pixel-space path 也不应通过线性公式偷偷替换。

若 \(T'=MT\)，无权 MSE 会变成误差上的 \(M^\top M\) 度量。只有同步改变 \(W\) 才能保持同一个目标泛函。decoupled heads 的容量、head 权重和 loss reduction 都要作为独立变量。

## 4. 边缘分布相同，边界仍可不同

通用边界是状态和缓存的随机核：

\[
(z_{k+1},c_{k+1},t_{k+1})\sim
\mathcal Q_k(\cdot\mid z_k,\hat x_k,\hat\epsilon_k,c_k,t_k).
\]

|实现|行为|
|---|---|
|`resize`|确定性改变空间网格；噪声通常因此相关|
|`rdd_fresh`|\(z_f=\alpha U\hat x_c+\sigma\epsilon_{new}\)，完全 fresh IID；当前 RDD 图像入口的默认 noise filling|
|`isometric`|精确保留指定强度的 coarse noise 相关性，同时得到 fine IID 边缘|
|`rdd_legacy_posterior`|历史 dense RDD posterior；保留原版 block-vector 顺序和 factor-two 限制；与 fresh variant 分开|
|`gaussian_conditional`|任意显式线性观测和协方差下的 Gaussian conditional lift|
|`pixelflow`|nearest lift + 负相关 2×2 block 噪声 + 状态/时间 rescale|
|`spectral`|SPEED 的 DCT/Haar/FFT 高频补噪和时间对齐|
|`permutation`|PPFlow 可逆 channel↔space；不丢维度、不补噪|
|`learned_affine`|NAMI 的 nearest lift × 可学习 scalar + 逐像素 shift|
|`codec_refine` / MrFlow workflow|VAE decode→PIL/uint8→SR→resize→VAE posterior encode→小 sigma refine|

### IID 边缘与保留 coarse noise 的连续族

对二维 factor \(f\) 的 nearest repeat \(U\)，有 \(U^\top U=f^2I\)。令 \(Q=U/f\)，\(Q^\top Q=I\)：

\[
\epsilon_f=Q[\rho\epsilon_c+\sqrt{1-\rho^2}\eta_c]
+(I-QQ^\top)\eta_f.
\]

当输入随机变量独立标准高斯时，\(\operatorname{Cov}(\epsilon_f)=I\)，但
\(\operatorname{Cov}(\epsilon_f,\epsilon_c)=\rho Q\)。这说明「fine 是 IID」不能确定跨阶段耦合。\(\rho=0\) 与 \(\rho=1\) 有相同 fine 边缘而不同轨迹。测试直接检查线性变换的协方差，避免仅靠 Monte Carlo 方差看起来接近。

### 一般 Gaussian conditional

若 \(X\sim\mathcal N(m,C)\)，观测 \(Y=DX+\eta\)、\(\eta\sim\mathcal N(0,R)\)：

\[
G=CD^\top(DCD^\top+R)^{-1},
\quad m_{X|Y}=m+G(y-Dm),
\quad C_{X|Y}=C-GDC.
\]

实现采用显式矩阵的审计版；零观测噪声时验证 \(D X=y\)。大网格应扩展为 matrix-free solver，不应直接实例化 dense 像素协方差。

### PixelFlow 与 SPEED

PixelFlow 的 2×2 block 协方差为对角 1、非对角 \(\gamma=-1/3\)，原代码另加 \(10^{-6}I\)。不能删掉 jitter 后仍声称 seeded parity。其边界系数和时间变换为

\[
a=[\sqrt{1-1/\gamma}(1-t)+t]^{-1},\quad
b=a(1-t)/\sqrt{-\gamma},\quad t'=at.
\]

SPEED 的谱扩张对应

\[
\kappa=\frac r{1+(r-1)t},\quad t'=\kappa t,
\]

但不同频谱归一化、FFT 实部截取及噪声抽样顺序属于算法的一部分，不能只匹配这个 scalar 公式。原版是在 NumPy fp32 中乘 \(\kappa\)，再转回 bf16；反过来做会产生可见数值差异。

## 5. representation、layout 与 adaptive 方法

有损多尺度表示、完整可逆布局和低频谱截断是三种不同操作。PPFlow 没有因为空间网格变小而丢失状态自由度；DCTdiff 选择固定低频坐标；CAT/DDiT/Fresco/AViTS/LoT 的可变 token 方案可以主要作用在 \(\Pi,H,S\)，不必改变 full-image 的概率路径。

一般的 adaptive layout 还需要维护：cell 坐标/面积/层级、split/merge 历史、attention mask、reconstruction map、选择策略和真实 compute cost。如果布局依赖样本，采样器应把布局作为状态缓存；如果改变实际随机变量维度，还要定义新增噪声的条件分布。若仅为可逆置换，无需补噪；若使用非线性可逆坐标变换并改变密度/score/SDE，则需处理 Jacobian 或 Itô 项。

这些内容可以由自定义 `model`、`batch`、`sampler` 和 `boundary` 插件表达。八篇尚无可执行官方源码的论文只给 paper-level 映射，**不把接口可表达性写成这些论文已经实现或复现**。本轮逐篇数值承诺覆盖十四篇有可执行源码的方法（RMD 来源归属限制见报告）及 RDD。

## 6. 工程接口与兼容性

通用引擎直接组合 `representation/noise/path/target/metric/clock/model/solver/stages`。多尺度 product 使用多个 path 和 leaf weights。阶段推理支持 joint 或 decoupled target，经明确 path 转回 velocity；边界上下文必须提供所需 clean/noise/cache，不隐式猜测。

为保持原版 packing、RNG 次序和 reduction，另外保留 `batch`、`prediction_adapter`、`sampler` 的 source-native 策略。它们把作者训练任务映射到 `Prepared(state,target,time,weights,metadata)`。native batch 已经定义其 target/noising，不接受会被忽略的通用 `path/noise/metric/target` 覆盖；此类配置明确报错。魔改应切换通用引擎或新注册一个 batch/sampler，并标为 controlled transplant。

`register(group,name)` + 配置的 `module` 可加载新组件；示例见 `examples/custom_components.py`。可学习参数需要归属于训练的 `model`；单独 inference boundary 中的 learned 参数由显式本地 checkpoint 提供。当前 `Trainer` 不会隐式把任意 Python 组件内部参数加入 optimizer。

## 7. 如何判定公平与复现成功

本轮按以下顺序验证：原版算子 → 带非零输出头的实际作者网络 → loss/每个参数梯度 → optimizer/EMA → 128 次更新 → 训练后的完整采样/可用 decoder → CLI 保存/恢复 → 本地数据/条件/权重加载。

精确对照共享同一小型作者骨干和初始参数，固定输入、时间、条件、随机抽样和 optimizer。源代码 oracle 与独立路径/loss/sampler 实现分开；明确标出复用的作者 backbone、DPM-Solver/torchdiffeq/backend。没有把「同一个函数跑两遍」当作独立算法验证。

真实论文比较还需固定数据/codec/训练成本，分别报告 matched architecture 的 controlled comparison 和作者架构的 native results。NFE 之外应报告每阶段 token、实际模型调用、CFG、teacher、SR/VAE、FLOPs、峰值显存和端到端时间。LoT/PPFlow 这类布局收益尤其不能只用 NFE 比较。

**本轮证明的是固定环境和所测配置下的数值/训练轨迹一致性。它不证明全规模训练稳定性、论文 FID，也不证明 RDD 比其他方法更强。** 用户禁止下载模型/数据，因此真实数据质量结论保持未验证；所有随机初始化与合成输入均明确标记。

## 8. 蒸馏也是框架的一部分：RMD

训练过程 \(\mathcal O\) 可以含多个网络与交替更新，而非只有一个 denoising loss。RMD 的 RF 坐标为 \(z_\sigma=(1-\sigma)x+\sigma\epsilon\)，网络预测 \(v=\epsilon-x\)，故 \(\hat x=z-\sigma v\)、\(\hat\epsilon=z+(1-\sigma)v\)。分辨率转换对 clean/noise 先做 spatial resize，再按源代码注入噪声：

\[
z'=(1-\sigma')U\hat x+\sigma'[\eta U\hat\epsilon+\sqrt{1-\eta^2}\xi].
\]

在真正切换 step，源代码直接用 fresh noise 替换方括号，独立于 eta；shift mode 决定转换后的 sigma table。代码将 resize、reinjection、clock、score weighting、revision loss 和三模型两优化器训练分开。严格复现还保留冗余 RNG draw、BCTHW normalization 的 width 维，以及 phase1 的 higher-shift 规则。teacher/fake/generator 的本地权重需要显式提供；本轮随机网络不能替代预训练 teacher 的分布知识。


## 9. 2026-10-09：训练程序和正式配方的边界

多尺度训练可以由一组明确的stage objective组成，而不必在模型类中硬编码：

\[
\mathcal J(\theta)=\sum_s w_s\,\mathbb E_{t_s,\epsilon_s}\,
\ell_s\!\left(f_\theta(\mathcal A_s(t_s)\,E_sx+\mathcal S_s(t_s)\epsilon_s,\kappa_s(t_s),c),\mathcal T_s(E_sx,\epsilon_s,t_s)\right).
\]

`training_stages`对应stage数据投影、time sampler、完整Method和loss权重。RDD原版的三个分辨率loss可在同一次optimizer更新内相加；mapped模型时钟与局部posterior时钟独立。它不是把同一条采样路径反向当作全部训练目标。mixed batch的有效样本mask、逆采样概率和normalized weighting由batch/reduction组件明确负责。

inference guidance是条件预测组合的策略，可以由各band自己的时间窗口控制。阈值、variance head、solver及解码空间也需要匹配。Native backend仍可复用作者网络和solver，但不能默默忽略用户配置；严格native分支对不兼容clock/path/head/variance参数拒绝执行。希望更换预测目标或算子时使用通用路径引擎，并将结果标为controlled transplant。

optimizer、初始化、数据预处理、precision、EMA和评估器是决定最终训练结果的recipe，而不是扩散路径方程中的新自由度。图像数或epoch也可作为优化器的时钟，分别用于Relay EMA/LR和LapFlow epoch cosine。这些组件加入不会改变上面的统一数学描述；它们同时说明：只对齐路径/target，不能推导出一定达到作者最佳FID。

新增分支的逐层认证及仍未接入的发布范围见 [2026-10-09最终复核](evidence/recipe_extensions/REPORT_ZH.md)。真实数据最佳训练指标仍未验证。

## 10. 2026-10-13：网格维数、联合状态与可选模型的实现

新增`task=multires`把网格写为显式的\(G_s=(H_s,W_s)\)或\((T_s,H_s,W_s)\)。数据表示\(E\)、扩散过程\(\mathcal P\)、预测网络\(f_\theta\)及网络时钟\(\kappa\)独立；配置编译器不根据论文最初是图像还是视频来限制网格维数。单尺度通用方法仍用原`Method`；作者native路径与新共同架构配置分别保留。

对嵌套网格，`GridMap`提供平均分析\(D\)与repeat合成\(U\)，满足

\[
DU=I,\qquad P=UD=P^*=P^2,\qquad D D^*=\Omega^{-2}.
\]

\(\Omega\)是每个粗网格单元的子单元数的平方根。保护视频首帧时，首帧只在空间做平均，而尾帧在时空做平均，所以\(\Omega\)可以是逐帧对角算子。与粗噪声相容的细网格创新为

\[
\epsilon_f=U\Omega^{-1}\epsilon_c+(I-P)\xi,
\quad \Omega D\epsilon_f=\epsilon_c,
\quad\operatorname{Cov}(\epsilon_f)=I.
\]

这个式子要求正交块分析/合成；一般bilinear插值不自动满足这些恒等式。实现因此区分`resize_signal`与正交噪声lift，不用信号插值替换噪声协方差计算。CPU验证直接构造线性算子的协方差矩阵，不仅估计样本方差。

Pyramid Flow的空间段写为\(z_\rho=\rho z_n+(1-\rho)z_c\)，监督\(z_n-z_c\)。作者网络时钟由stage flow长度计算，与\(\rho\)不同，训练还含1000点量化与FP64插值后cast。新实现保留这些细节；不能用裸\(\rho\)并声称作者逐元素一致。对应边界为

\[
z_f=\frac{Uz_c+\Omega\sigma_c(I-P)\xi}{1+(\Omega-1)\sigma_c}.
\]

PyramidalWan的natural clock \(\zeta\)映射为\(\sigma=\zeta/[\Omega_s-(\Omega_s-1)\zeta]\)。监督目标对global \(\sigma\)求导，积分步长也使用\(\Delta\sigma\)，而不是\(\Delta\zeta\)。首帧和尾帧可使用不同\(\sigma\)。改到图像时\(\Omega_s\)取二维网格值，这是一条可定义的路径移植，不能称完整视频论文复现。

f-DM的信号\(m_s(t)\)在阶段的两个clean表示之间变化，状态\(z_t=\alpha_s(t)m_s(t)+\sigma_s(t)\epsilon\)。默认双头输出epsilon与\(x_s-m_s(t)\)，由前者重建signal，优化signal与residual两项加权误差；不是简单把RDD drift头改个名字。SP保留alpha并缩放sigma，VP进一步归一化\(\alpha^2+\sigma^2\)。实现明确配置P2/SNR/uniform权重及边界噪声耦合。

Edify移植取\(A_s(\sigma)=P+a_s(\sigma)(I-P)\)，则路径速度为

\[
v=A'_s(\sigma)\hat x+[z-A_s(\sigma)\hat x]/\sigma.
\]

当前选择\(a_s\)在每个stage的sigma区间由0线性增至1，使下一stage的低频接入连续；训练sigma分布与sigma区间均是显式受控选择。论文定义的更一般衰减族不意味着某个未公布的最优函数已被恢复。

Non-Uniform使用Haar正交分量\(h_j\)，各自的VP SDE为\(dh_j=-\beta_j h_jdt/2+\sqrt{\beta_j}dW\)。不同截止时间控制何时加入新band。epsilon参数化对应的likelihood加权score目标为

\[
\tfrac12\sum_{j\in\mathcal A(t)}\beta_j(t)\sigma_j(t)^{-2}
\|\hat\epsilon_j-\epsilon_j\|^2.
\]

代码保留活动集合的联合条件与标量数归一化，不能把各band独立训练当作等价替代。原DiT/Wan的单张量API并不原生具备这类product-state接口；因此配置编译器在这种组合上明确拒绝，而`grid_dit`将各band token一起进入同一个Transformer，并按独立读写端口还原。

`grid_dit`将\(T,H,W\)三轴的位置编码放在不同通道子空间，或使用逐轴RoPE。图像是\(T=1\)；视频所有帧共同attention。原DiT adapter沿用作者blocks、2D PE并显式增加时间坐标；Wan adapter使用原生3D位置机制。模型结构与参数共享也分开：`stage_models`保留同结构而不共享参数。改变共享方式仍然会改变可表达函数族、参数预算与优化，不能以“统一架构”为由忽略。

新验证包含单帧等价、跨帧非零导数、轴置换不混淆、联合band mask、原DiT/Wan模块F/B、作者Pyramid Flow目标、独立paper oracle、训练/采样及精确恢复；[机器报告](evidence/cross_modal_20261013/audit.json)区分bitwise、数值容差与仅公式检查。真实视频质量、多机/GPU数值和论文最佳分数仍需正式实验。
