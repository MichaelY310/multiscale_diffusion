# Pyramid Flow 与当前 RDD：重叠、差异及贡献边界

日期：2026-11-02。本轮核查原论文 §3.1–3.3、附录 A，以及本地固定作者代码和 RDD 源码；新增 CPU FP64 代数实验。没有训练、模型 forward/backward、生成质量实验或 GPU 使用，没有改默认算法、训练配置、历史论文源码或共享 FastVideo。

结论：高度重叠但不是同一个具体过程。Pyramid Flow 已覆盖单 DiT、多分辨率连续生成、联合训练和有分布依据的阶段衔接。当前 RDD-FM 的阶段内路径不同，但“残差化表达”“解析 clean 逆”“可用高阶 solver”并非它独有。是否有算法优势，需要直接对照而不能由公式不同推出。原 RDD-DDPM 应另行区分，不能将其结果当成 RDD-FM 已优于 Pyramid Flow 的证据。

## 原文与源码依据

- [论文](https://arxiv.org/html/2410.05954v2)：式 9–11 为带噪端点插值及训练目标，式 12–15 为 noisy-state 边界；首版 2024-10-08，ICLR 2025。
- [作者 pipeline](https://github.com/jy0205/Pyramid-Flow/blob/main/pyramid_dit/pyramid_dit_for_video_gen_pipeline.py)：`add_pyramid_noise` 和 `add_pyramid_noise_with_temporal_pyramid` 构造 noisy endpoints；`generate_one_unit` 上采样整个状态，再缩放和补噪。
- 本地固定作者目录：`/scr/yuheng/bugswarm/official_video_baselines_20261004/reference/Pyramid-Flow`。pipeline 第 303–348 行为核心训练表达式，第 697–742 行为边界；`diffusion_schedulers/scheduling_flow_matching.py` 第 99–121 行修正阶段入口时钟。
- 当前 RDD-FM：`umdiff/paths.py::RDDFlow`；统一的 endpoint 插值：`EndpointFlow`；实际 PF/RDD 阶段程序：`umdiff/multires_process.py::PyramidFlowProcess/RDDProcess`。旧 RDD 路径也见 `multiscale_diffusion/diffusion/flow_matching_with_drift.py`。
- 原 RDD-DDPM：`multiscale_diffusion/diffusion/gaussian_diffusion_with_drift.py::q_sample/q_posterior_mean_variance`。

## 对齐端点后的阶段内比较（本轮推导）

只比较空间核心；固定相同 fine clean x、lifted coarse clean c、噪声 ε 及阶段端点 a<b。写 d=x-c、L=b-a、τ=(t-a)/L，时间方向为 noise→data。此处 t 是有效信号系数坐标，不等于作者直接输入网络的离散 timestep。完整 native 配置还包含不同的跨阶段时钟、条件、样本分配等；此对齐是隔离路径的分析，不声称这些配置原本相同。

两个条件端点均为 z_a=ac+(1-a)ε、z_b=bx+(1-b)ε。

Pyramid Flow 插值 noisy endpoints：

\[
z_t^{PF}=(1-\tau)z_a+\tau z_b
=tc+b\tau d+(1-t)\epsilon.
\]

当前 RDD-FM 对 clean 的分辨率成分插值后，再使用 FM 信噪系数：

\[
z_t^{RDD}=t[(1-\tau)c+\tau x]+(1-t)\epsilon
=tc+t\tau d+(1-t)\epsilon.
\]

因此

\[
z_t^{PF}-z_t^{RDD}=L\tau(1-\tau)d.
\]

端点相同，非零细节下内部通常不同；RDD 的细节系数更小，即高频进入更晚。a=.4、b=.7、t=.55 时：两者 coarse 系数均 .55，noise 系数均 .45，PF detail 系数 .35，RDD 为 .275。不能从“更晚”推出更好或更差；它同时影响细节信噪比与后期要完成的变化。

两者其实属于同一种残差路径族：

\[
z_t=tx+(1-t)\epsilon+\rho(t)(c-x),\qquad
\rho_{PF}=\frac{a(b-t)}L,\quad
\rho_{RDD}=\frac{t(b-t)}L.
\]

所以用 G(x)-x 写残差本身不能区别 PF。固定上述归一化信噪系数后，也不能只换一个标量时间名字就让两条路径相同，因为 noise 系数已经固定 t；这不排除更一般的状态变换或重新加权关系。

## 速度与生成过程

换成同一个 t 坐标后，条件速度为

\[
u^{PF}=c+\frac bL d-\epsilon,\qquad
u^{RDD}=c+\frac{2t-a}L d-\epsilon.
\]

作者预测的是局部噪声方向 rho=1-τ 的 endpoint difference，转到此处须乘 -1/L；该符号和缩放差别不是创新。两者速度在 τ=.5 恰好相同，但该处状态不同，不能只检查中点目标就判定路径一致。PF 的条件轨迹为直线，RDD 的条件曲率是 2d/L；学习到的边缘 ODE 轨迹不必因此直线/常曲率。

两者理想训练解均为 v*(z,t)=E[u|z_t=z]。输入分布与条件目标不同，通常产生不同边缘速度场，采样 ODE 因此通常不同；但共同使用 dx/dt=vθ 的 ODE 形式本身相同。两者都能使用常规高阶 ODE 求解器。不能主张只有 RDD 能用 DPM 类方法；具体 forcing、时钟与边界 history 仍需正确处理。

## 阶段边界：不是同一种转移

当前 RDDProcess 默认 fresh 边界为

\[
z^+=aU\widehat x_c+(1-a)\eta.
\]

这里 a 是相邻阶段共用边界，网络先预测 coarse clean，再上采样并重抽独立噪声。理想网络的 clean 是条件均值，也存在此前记录的后验不确定性缺项。这句话仅针对 fresh，不将原 RDD-DDPM、coupled 等边界混为一谈。

PF 则传递完整 noisy state。用 e 表示 coarse 出口、s 表示 fine 入口的信号系数，其转移可写成

\[
z_s^+=\frac{s}{e}Uz_e^-+\xi,
\quad \operatorname{Cov}(\xi)=(1-s)^2I-
\left(\frac{s}{e}\right)^2(1-e)^2UU^T\succeq0.
\]

这是对作者公式的统一记号重写。在线性 U、独立 Gaussian ξ 和配套端点下，可将正确 coarse 边缘推到设计的 fine 入口边缘；不要求 ODE 保留训练时某一组 clean/noise 配对，也不经过 clean 条件均值替换。模型/求解误差不在该精确声明内。

nearest 2× 空间上采样时，UUᵀ 每块为 4×4 全 1 矩阵。论文选 s=e/(2-e)，补噪有负的块内相关。当前作者代码 gamma=+1/3 是协方差表达式中的参数，论文矩阵的 off-diagonal gamma=-1/3，不应因记号误判实现矛盾。

PF 在这里调整噪声水平，既不是每阶段去噪到 clean 再跑完整生成，也不是不做边界修正的简单连续整数时间切片。若硬保持 s=e，所需补噪协方差为 (1-e)²(I-UUᵀ)，含负特征值；因此不能不改变端点或归一化就替换当前 RDD fresh。

PF 在这一项上有明确优势：其边界推导不依赖把未知 clean 替换成后验均值。不能把该优势外推成实际 FID 一定更好，也不能说 PF 已解决所有可能的后验估计问题。

## 解析 clean 逆并非 RDD 独有

取正交投影 P、Q=I-P，c=Px，写 A_t=tP+g(t)Q、σ=1-t。条件路径 z=A_tx+σε、速度 u=A'_tx-ε 给出

\[
z+\sigma u=[A_t+\sigma A'_t]x.
\]

RDD 的 g=t(t-a)/L，故 B_RDD=P+[(2t-t²-a)/L]Q；PF 的 g=b(t-a)/L，故

\[
B_{PF}=P+\frac{b(1-a)}LQ.
\]

系数可逆时，两者都有 clean 恢复公式 x=B^{-1}(z+σu)。将 u 换成理想 learned marginal velocity 后，恢复的是 E[x|z]，不是某个未知真样本。PF 的比例甚至在本阶段内不随 t 变化。当前 `EndpointFlow.clean_from_velocity` 已可实现这个逆。

因此应撤回“解析 clean 恢复能力是 RDD 对 PF 独有优势”的暗示；RDD 的具体逆系数仍然正确。`multiscale_diffusion/paper/main.tex` 当前正文 clean-inverse ablation 的 “unique to the residualized projection path” 表述应收缩。此轮记录修改依据，不直接改论文或历史结果。

## 原 RDD-DDPM 与论文定位

旧 DDPM 实现的阶段边缘是 z_j=√αbar_j [x+λ_j(c-x)]+√(1-αbar_j)ε，λ_j 来自阶段归一化离散时间和 drift multiplier。它的 VP 日程、effective-noise 训练坐标、反向 posterior/score 参数化不能直接等同本论文 FM。也不能仅凭选择 SDE/VP 而宣称首创多分辨率扩散；更早 transformed-signal diffusion 等先例仍需讨论。

以下广义贡献已被 PF 覆盖：单共享 DiT；多分辨率阶段联合训练；各阶段只负责部分生成过程；跨分辨率有噪端点；显式补噪衔接；高效视频实现。其 temporal pyramid 还压缩自回归历史条件，不能将整篇简化成空间路径而冒充完整系统复现。

本项目仍可区分：跨多篇方法的共同数学表示与组件代码、严格划分 native 与 controlled transplant 的公平实验，以及一个经实证证明有用的 RDD 路径配置。PF 所称 unified 主要是在一个模型中联合金字塔阶段，不等于已实现本项目跨论文的组件框架。但组件数或代数改写本身也不保证研究价值。

最小有价值对照是固定模型/数据/VAE/训练预算/目标归约后，比较 RDD 曲线路径与 endpoint 直线路径，并分别审计边界。可以做 path×boundary 实验，但每种边界必须使用数学兼容的训练端点，不能将不兼容核简单塞进同一 checkpoint。固定 NFE 与固定计算量是不同协议，应都说明。此前 RDD 对普通 DiT 的成绩不能替代 RDD 对 PF 的直接证据。

## 补充：RDD 较重的数学是否也已被 PF 覆盖？

2026-11-02 后续逐项复核 `paper/main.tex` 的 posterior、noise filling、SDE/DPM 正文及附录，以及 `docs/theory/RDD_DIFFUSION_FM_DERIVATION_TUTORIAL_ZH.md` 第 6–13、20 节；PF 原文第 3 节及附录 A。这里比较的是论文的明确推导范围，不是仅凭某关键词搜索不到就宣称不存在。此补充没有新增数值测试。

答案是否定的。上一轮主要比较空间 FM 路径，不能将原 RDD-DDPM 的整套 reverse calculus 缩减成一个 envelope 差异。另一方面，原文未给出的推导也不必然是只有 RDD 才具有的性质。

| 数学内容 | RDD 的具体内容 | PF 原论文及关系 |
|---|---|---|
| 离散前向转移与任意跨度后验 | 对带 clean-conditioned residual 的 VP 链推导一步核、边缘、q(x_s\|x_t,x_0) 及 gamma*d 修正 | PF 以条件 FM 路径和速度场训练，没有这套 VP 链及其后验；两者并非同一个随机过程 |
| 有效噪声与 Gaussian innovation | 训练 epsilon_tilde=epsilon+(alpha*r/sigma)*d；计算 score/ODE 时分离 deterministic residual | PF 直接训练 endpoint velocity，没有使用该 effective-noise 训练坐标；类似参数化转换可在更一般 Gaussian 路径上推导 |
| 单头恢复和闭合条件 | 理想 effective-noise head 恢复 E[x_0\|x_t]；仿射 d 时可得到 E[d(x_0)\|x_t]，非线性时一般不能 | 这是一类路径的条件期望/可辨识性分析，可能服务于 unified 框架，不是排斥 PF 的专有能力 |
| 条件 SDE、边缘 drift、probability-flow ODE | 线性项之外有 alpha*r'*d；边缘漂移应使用 E[d\|x_t]，pure noise 才能给正确 score | PF 直接以 FM 定义生成 ODE，未给同一 reverse-SDE 推导；这不代表其概率路径不能关联 SDE |
| 漂移修正的指数积分 | 一阶更新含 alpha_s*(r_s-r_t)*d_hat，冻结预测时有解析条件轨迹恒等式 | PF 未给此 RDD-specific 更新；其自身路径也可利用解析系数，故高阶求解能力不独有 |
| 投影频带与误差传播 | d=(P-I)x，可分解低/高频 SNR、残差预测误差；具体系数随 RDD 路径变化 | PF 也属可进行频带分析的路径；推导需区分特定系数与共有的线性代数性质 |
| 阶段边界的 covariance | fresh 的条件 Gaussian/noise-filling 论证；网络均值替换后另有后验协方差缺项 | PF 附录 A 明确推导 noisy-state 的块相关补噪和时钟约束；这一部分本身有实质数学，并非仅凭经验 upsample |
| FM clean inverse | 有具体的投影逆和可逆性条件 | PF 原文未给相同逆公式，但上文已证明相同投影假设下它也有解析逆 |

**后验推导的意义和范围。** 令 alpha_t=sqrt(alpha_bar_t)，RDD 阶段路径为 x_t=alpha_t[x_0+r_t d(x_0)]+sigma_t epsilon。给定 x_0 后，中心化 y_t=x_t-alpha_t*r_t*d 在所定义的条件链中是标准 VP/DDPM。若标准 y 的后验均值为 A*y_t+B*x_0，则 RDD 的后验均值为

\[
\mu_{s|t,x_0}=A x_t+B x_0+
(\alpha_s r_s-A\alpha_t r_t)d(x_0).
\]

这是 RDD 路径确实需要的修正，PF 没有相同的 VP 后验。但其推导工具是已知 Gaussian conditioning 加确定平移；公式正确、有工程价值，不自动意味着一般反向扩散理论的新定理。

**三个仍需约束的数学主张。**

1. 闭式 Gaussian 后验是给定 clean 的 q(x_s|x_t,x_0)。对未知 clean 积分后的 q(x_s|x_t) 通常是混合分布；均值闭合不意味着完整 reverse covariance 或分布闭合。
2. 含 d(X_0) 的 SDE/离散链首先是 clean-conditioned 过程。将其 drift 替成条件期望，在通常正则条件下可构造匹配一时刻边缘的 state-only Markov 过程；不应据此宣称原隐藏 clean 过程本身是 Markov，或二者所有多时刻联合分布相同。
3. 一阶冻结预测的解析恒等式不等于 learned model 大步采样精确；midpoint 重算也不自动证明二阶。历史主文对此已有部分限定，但附录仍有较强的 “For second-order accuracy” 措辞，不能作为已完成误差阶证明。fresh 的边界精确匹配也仍受前述 oracle/后验均值区别限制。

上述 SDE/ODE 联系依赖既有 [Score-SDE](https://arxiv.org/abs/2011.13456) 理论；[Flow Matching](https://arxiv.org/abs/2210.02747) 本就覆盖一般 Gaussian conditional paths。PF 没写 RDD 的那组公式，与全球范围内证明了独有理论贡献，是两个不同判断。

适当的研究定位可以是：在统一算子 Gaussian 路径中明确各训练目标、可辨识性、单头闭合条件、边界限制与采样转换，再将 RDD 作为其中一个有实测价值的 VP/FM 实例。当前可以确认数学内容不同且 RDD 覆盖了额外的 VP/DDPM 推导；尚不能由此确认 RDD 更优或这些数学工具为其首创。

## 补充：路径专用 FM 积分器与 PF 的适用性

本次进一步核查既有 `RDD_FM_SOLVER_AUDIT_20261101_ZH.md`、两维研究原型 `research/rdd_fm_integrator.py` 和原 CPU JSON。PF 第 3.2.2 节允许阶段内使用标准采样算法；所查固定官方 `diffusion_schedulers/scheduling_flow_matching.py` 第 283 行为 Euler 更新。正文及附录没有提出以下基于 clean 历史、解析频带系数的专用高阶方法。这是对该论文/代码范围的判断，不是整个文献不存在类似方法的声明。

令 P=UD 为正交投影、Q=I-P、L=b-a、sigma_t=1-t，RDD 的 A_t=tP+t(t-a)Q/L，B_t=A_t+sigma_t A'_t。对任意网络速度 v_theta，用 m_theta=B_t^{-1}(z+sigma_t v_theta) 定义 clean 读出，可以精确重写同一个 ODE：

\[
z_s=\frac{\sigma_s}{\sigma_t}z_t+
\sigma_s\int_t^s\frac{B_u}{\sigma_u^2}m_\theta(z_u,u)\,du.
\]

积分中的 B/sigma² 是已知的；只需近似未知的 m_theta 沿轨迹的变化，而不必把确定的路径系数也交给数值插值。

冻结当前 m_t 后有

\[
z_s\approx\frac{\sigma_s}{\sigma_t}z_t+
\left(A_s-\frac{\sigma_s}{\sigma_t}A_t\right)m_t
=z_t+h v_t+\frac{h^2}{L}Qm_t,\quad h=s-t.
\]

这是一次模型调用的曲率修正；对固定真实 clean/noise 的条件轨迹精确，对一般 learned ODE 仍然是一阶。二阶候选用同阶段历史估计 m(u)≈m_t+(u-t)(m_t-m_previous)/(t-t_previous)，再解析积分系数，具体 W0/W1 已在 11 月 1 日报告和原型中实现。在光滑、步长比/稳定性条件下有阶段内部二阶依据；不自动覆盖 t=1 的一致误差界或边界跳变。跨尺度/重噪后清空历史。

PF 在相同投影和有效时间记号下有 A_t=tP+b(t-a)Q/L，因此

\[
A''_t=0,\qquad B_t=P+\frac{b(1-a)}{L}Q
\]

为阶段内常数。相同的冻结-clean积分恰好等于 Euler；它没有 RDD 的 h²Qm/L 项，因为该路径没有这项已知条件曲率，而非其不能使用高阶方法。采用 clean 历史的二/高阶积分依然成立。由于 B 常数，在相同插值变量、时间网格及求积规则下，对 m 插值后乘 B，与直接对标准 forcing y=Bm 插值等价；不声称所有命名为 DPM++ 的 scheduler 在不同时间变量/末端特例下逐位相同。

对 RDD，B_t 随时间变化，结构化方法解析处理 B_t，只插值 m_t；普通 Flow-DPM++ 可以直接近似 y_t=B_t m_t。两者都是对原 ODE 的合法数值方法，但误差常数不同。不能只用真实 m 替换标准 forcing、却不改对应积分系数。

潜在研究价值是：RDD 延迟细节是否改善训练，同时路径积分能否减少它自身增加的条件曲率误差。PF 的对应修正为零，不代表 RDD 获得了 PF 无法享有的通用求解器能力。一般原则已有 [DEIS](https://arxiv.org/abs/2204.13902)、[gDDIM](https://arxiv.org/abs/2206.05564)、[Flow-Solver](https://arxiv.org/abs/2411.07627) 等先例；[AREX](https://arxiv.org/abs/2610.03483) 也研究 FM 的矩阵解析积分，但其分解基于目标矩而不是这里已知的路径。尚未确认具体 RDD 公式的文献新颖性。

新增 `tools/validate_pyramid_rdd_integrators_20261102.py`：两阶段各三个内部位置，CPU NumPy FP64 检查 RDD 曲率更新、PF 退化为 Euler、PF B 常数，以及双方线性 clean 外推的解析权重对独立 SciPy 求积。6 组检查最大残差 2.55352e-15；证据 `evidence/pyramid_flow_rdd_overlap_20261102/INTEGRATOR_IDENTITIES.json` 含源 hash。该工具未运行模型、作者完整 scheduler 或生成质量测试。

研究原型仍只有两维频带实现，未注册生产采样器、未接入 Wan。11 月 1 日解析分布结果有优于 Euler 的例子，也有输给普通二阶方法的例子；本轮没有新增质量或全流程结果。标准 solver、path solver 与 PF 的实际优劣仍需 matched-NFE 对照。

## 本轮可复跑证据

`tools/validate_pyramid_rdd_overlap_20261102.py`，结果 `evidence/pyramid_flow_rdd_overlap_20261102/ALGEBRA.json`；含源文件 SHA256。

- CPU FP64，图像与视频形状，两个阶段、五个阶段位置，共 20 组。
- 现有 RDDFlow/EndpointFlow 与推导的状态差、速度差及两种 clean 逆吻合，最大残差 2.44249e-15。
- 非零细节下中间路径不同；在示例随机张量与 [.4,.7] 中点，最大状态差 .1836119，而速度差为舍入量。它是非等价证据，不是质量分数。
- e=.7、s=.5384615 时补噪协方差特征值为 [0,.21301775,.21301775,.21301775]，目标噪声协方差代数误差 0；同钟 s=e 时最小特征值 -.27。
- 未执行 PF 完整 native pipeline；作者表达式按已检查源码转写来做代数比较。不是新增全网 parity、最优权重测试、训练结果或 FID/VBench 认证。CUDA 未初始化。

```bash
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" unified_multires/tools/validate_pyramid_rdd_overlap_20261102.py \
  --output unified_multires/evidence/pyramid_flow_rdd_overlap_20261102/ALGEBRA.json
```

## 2026-11-03：哪些差异具有可比较的性质？

以下为本项目的新推导与分析，不是 PF 论文声称的结论。只比较当前 RDD-FM 与 PF 空间路径，在相同端点 a,b、同一物理时间 t、正交 P=UD 和各向同性独立 Gaussian noise 下隔离路径因素；完整 native PF 的阶段时钟、temporal pyramid 与 RDD fresh 边界另算。Q=I-P、c=Px、d=Qx、L=b-a、τ=(t-a)/L。这里的“细节/高频”指投影残差，尤其在 VAE latent 上，不应等同于纯纹理或严格 Fourier 高频。

### 1. 理想的阶段生成上限相同，不意味着有限模型效果相同

两条条件路径的阶段起点与终点分布相同。若从正确入口分布出发，学到精确边缘速度并精确积分，在通常连续性方程/ODE 正则条件下，两者均到达设计的同一阶段出口分布。路径不同本身不意味着其中一个具有更高的理想生成上限。实际区别来自有限容量的速度估计、训练时间分配与 loss、有限步积分及边界误差。当前 RDD fresh 的入口匹配不能直接用这一理想结论担保。

### 2. 更晚加细节意味着更少观测信息，不是天然更容易预测 clean

写 z=tPx+gQx+σε，其中 σ=1-t，g_PF=bτ、g_RDD=tτ。单个细节方向 clean 方差为 λ 时，SNR_H=g²λ/σ²，故阶段内部

\[
\frac{\mathrm{SNR}^{RDD}_H}{\mathrm{SNR}^{PF}_H}=(t/b)^2\leq1.
\]

更强的结论不需要 clean 是 Gaussian。令 r=t/b、η 为额外独立标准 Gaussian，则对任意 clean 分布有条件分布恒等式

\[
\widetilde z_R=Pz_{PF}+rQz_{PF}+\sigma\sqrt{1-r^2}\,Q\eta,
\qquad (x,\widetilde z_R)\overset d=(x,z_{RDD}).
\]

理由是信号变成 tPx+rg_PF Qx，而噪声协方差为 σ²[P+r²Q+(1-r²)Q]=σ²I。于是可构造 Markov 链 x→z_PF→z_R；由条件期望和全协方差公式，平均 clean 后验协方差满足

\[
\mathbb E\operatorname{Cov}(x\mid z_R)
\succeq\mathbb E\operatorname{Cov}(x\mid z_{PF}).
\]

这是平均意义的 Bayes clean MMSE 排序，不是相同数值 z 下逐点后验矩阵的排序，也不是生成 FID、训练梯度方差或不同 velocity 目标难度的排序。存在共同文本/类别条件时，可以逐条件应用同一论证。非正交算子、非各向同性噪声或不同端点下不能直接套用该式。

独立标量 Gaussian clean H~N(0,λ) 的显式例子为 Var(H|Z)=λσ²/(g²λ+σ²)。a=.4,b=.7,t=.55,λ=1 时，PF/RDD 的 g 为 .35/.275，后验方差分别 .623077/.728090，RDD 的细节 SNR 为 PF 的 .617347 倍。这不是生成质量下降比例。

### 3. RDD 同时改变了任务的时间分配和隐含的频带 loss 权重

两者的条件 detail velocity 系数是 g'_PF=b/L 与 g'_RDD=(2t-a)/L。RDD 在前半阶段小于 PF，后半阶段大于 PF；因此是先少做、后面加速追上，不是消除了这部分生成任务。

令 B=A+σA'，则 B_PF=P+[b(1-a)/L]Q，B_RDD=P+[(2t-t²-a)/L]Q。在 σ>0、B 可逆处，将 velocity 预测读成 mθ=B^{-1}(z+σvθ)，有精确损失恒等式

\[
\|v_\theta-u\|^2=\sigma^{-2}\left(
\|P(m_\theta-x)\|^2+B_H(t)^2\|Q(m_\theta-x)\|^2\right).
\]

因此 PF 的相对 clean-detail 权重 B_H² 在阶段内固定，RDD 的该权重随时间上升。这给出可检验的优化假设：在有限容量的共享网络中，RDD 的时间安排可能减少前段细节拟合与结构拟合的竞争。然而，PF 可以显式修改频带 loss 权重；“能做课程式加权”并非 RDD 独有，且加权相同也不等于输入路径相同。

需要注意：RDD 的 B_H 在 t*=1-sqrt((1-a)(1-b)) 与 PF 相交；b<1 时早期小、晚期大，最后阶段 b=1 则直到终点前都小于 PF。不能把“RDD 内部随时间增加”错误写成“RDD 在所有阶段末期都比 PF 更重视 clean 细节”。同样，clean 读出的速度误差放大系数 σ/B_H 不具有 RDD 全程优于 PF 的排序。

对独立标量 Gaussian 例子，直接条件化 joint Gaussian 可得最优 velocity 的不可约 MSE：

\[
\operatorname{Var}(U\mid Z)=\frac{\lambda(g+\sigma g')^2}{g^2\lambda+\sigma^2}.
\]

a=.4,b=.7,λ=1 时，t=.43 的 RDD/PF 为 2.57350/5.94300，t=.67 为 5.67136/3.87505。即便在这个简单例子中也不存在 RDD 在整个阶段更易回归 velocity 的排序。不同目标下的原始 loss 大小不能直接拿来排名生成能力；这里仅说明早晚任务分配的权衡。

### 4. PF 有更简单的条件路径几何；RDD 可以补偿自身已知曲率

固定一对 x,ε，PF 条件路径为直线，RDD 有 z''=2d/L。由 u_RDD=u_PF+(2τ-1)d，积分交叉项为零，得到

\[
\int_a^b\|u_{RDD}\|^2dt-\int_a^b\|u_{PF}\|^2dt
=\frac{L}{3}\|d\|^2\geq0.
\]

这是固定端点、固定持续时间下的条件动能差；PF 达到直线所给的最小条件动能。不能将其直接改写为 learned 边缘速度的动能排序、Lipschitz 排序或 FID/NFE 定理，因为条件平均和轨迹相交会改变边缘场。

冻结 clean 时 RDD 的单步补偿 h²Qm/L 可以消除这部分已知条件曲率；PF 不需要这项且同一推导退化为 Euler。双方仍可使用高阶方法。合理研究问题是“RDD 的训练安排是否带来收益，且结构化 solver 能否抵消它新增的采样负担”，而不是“RDD 比 PF 多一项修正所以理论更强”。

### 5. 其他差异与判断

| 比较项 | 可确定的差异 | 性质及限制 |
|---|---|---|
| 默认阶段边界 | PF 传递 noisy state 并按兼容时钟补相关噪声；当前 RDD fresh 使用 clean 条件均值再加独立噪声 | PF 在理想入口边缘匹配上有更强保证；RDD fresh 有已记录的后验协方差缺项。该结论不覆盖所有 RDD-DDPM/coupled 变体，不等于实际 PF 质量一定更好 |
| 延迟细节的优化作用 | RDD 前段较弱 detail velocity、随时间增长的相对 detail clean 权重 | 可能使有限容量网络先形成结构，再细化；也可能因细节信息不足而使小物体、文字、细线结构更难恢复。两者都是待训练检验的假设 |
| 视频 | 空间细节的时间安排不同；PF 原系统另有自回归 temporal pyramid | 延迟空间细节不自动保证时间一致性，也可能使移动小物体出现闪烁；完整 PF 的历史压缩收益不能归因于空间路径，公平路径比较应固定视频架构与上下文 |
| 计算量 | 相同网络、阶段网格及每阶段调用数时，单次高分辨率输入形状相同 | 推迟信号系数不会自动少算 token；节省主要来自共同的低分辨率阶段。实际 NFE 还须计入 boundary clean 调用、CFG 等 |
| 理论工具 | RDD 另推导了 VP posterior、SDE、目标转换与结构化积分；PF 原文主要使用 FM 与匹配边界 | 这些内容支持统一分析和实现正确性，但不建立 RDD-FM 对 PF 的生成效果优势；通用高阶 solver 也非 RDD 专有 |

最直接的路径消融是 gη=τ[(1-η)t+ηb]，η=0 为当前 RDD、η=1 为 PF，加入 η=.5。先做相同端点/正确入口的单阶段隔离，再在数学兼容的边界时钟下训练完整系统；不要把不兼容 boundary 硬换进旧 checkpoint。固定 backbone/VAE/数据/训练预算，并对双方都给 Euler 与同等条件高阶 solver，报告质量—实际计算曲线及各频带预测误差、模型误差和积分误差。边界 fresh 的问题应单列，否则无法把质量变化归因于延迟细节。

### 本轮证据范围

新增 `tools/validate_pyramid_rdd_tradeoffs_20261103.py`，使用既有两维 RDD research 系数和 NumPy/SciPy，在 3 个阶段×3 个位置检查信息退化的联合协方差恒等式、相关 Gaussian clean 的平均后验矩阵差 PSD、SNR 比例、标量 velocity 条件方差、B 交点以及独立数值积分的条件动能差。9 个配置通过，最大恒等式残差 1.77636e-15；机器记录 `evidence/pyramid_flow_rdd_overlap_20261102/PATH_TRADEOFFS_20261103.json` 含源码 hash。一般分布的信息排序由上面的 Markov/条件期望证明支撑，不是由有限 Gaussian 数值例子推断。

本轮只有 CPU 解析例子，没有运行训练、作者完整 pipeline、真实网络或质量 benchmark；没有改训练/采样默认实现。复跑时需选择新输出路径，以保留已有证据：

```bash
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" unified_multires/tools/validate_pyramid_rdd_tradeoffs_20261103.py \
  --output /tmp/pyramid_rdd_tradeoffs_recheck.json
```

## 2026-11-03（洛杉矶；UTC 11-04）：RDD 路径能否使用 PF 式边界？

**可以。调整下一阶段的训练入口后，RDD-FM 可以保留阶段内部的分辨率漂移路径，同时使用与 PF 相同分布的边界转移。它解决的是理想输入下的边界分布匹配，不是已经证明生成质量与 PF 相同。** 本节是本项目推导；其中 noisy-state 重缩放及相关补噪的基本构造来自 [PF §3.2.2、公式 13–15 与附录 A](https://arxiv.org/html/2410.05954v2)，不能作为新的边界算法主张。

### 1. 输入、目标与一般可行条件

为避免将网络 timestep 与信号水平混淆，α_old、α_new 分别表示切换前、后的干净信号系数，σ_old、σ_new 是对应噪声标准差。X_c 是真实低分辨率 clean；下式是分布的表示，采样时不需要知道 X_c 或分解出 ε_c。

\[
Z_{\rm old}=\alpha_{\rm old}X_c+\sigma_{\rm old}\epsilon_c,
\qquad
Z_{\rm target}=\alpha_{\rm new}UX_c+\sigma_{\rm new}\epsilon_f.
\]

这里 clean 可以服从任意分布；ε_c、ε_f 在各自网格上都是与 clean 独立的标准 Gaussian。U 为 nearest 上采样，每个粗网格值复制 m 次；空间宽高各 2× 时 m=4。D=Uᵀ/m 为配对 block average，P=UD=UUᵀ/m 为细网格的正交分块常数投影，Q=I−P。

考虑不估计 clean 的转移

\[
Z_{\rm new}=rUZ_{\rm old}+\xi,
\qquad \xi\sim\mathcal N(0,\Sigma_\xi),
\]

其中 ξ 与旧状态独立。匹配条件均值和噪声协方差要求

\[
r=\frac{\alpha_{\rm new}}{\alpha_{\rm old}},\qquad
\Sigma_\xi=\sigma_{\rm new}^2I-r^2\sigma_{\rm old}^2UU^T
=(\sigma_{\rm new}^2-mr^2\sigma_{\rm old}^2)P+\sigma_{\rm new}^2Q.
\]

因此在 0<α_old<1、m>1 下，这种转移可行的充要条件是

\[
\boxed{\sigma_{\rm new}\geq\sqrt m\,r\sigma_{\rm old}.}
\]

这不仅匹配总方差：给定 clean 后，双方都是均值及完整协方差相同的 Gaussian，因此条件分布相同；再对任意 clean 分布积分，入口边缘分布也相同。生成 ODE 不需要保留某个训练样本的隐藏 clean/noise 配对：只要传入该核的边缘分布正确，就得到正确入口边缘。它不保证真实网络或有限步积分已经产生正确的旧边缘。

### 2. RDD-FM 的显式解：空间 2× 时与 PF 相同

在线性 FM 噪声日程 σ=1−α 下，取上面的可行条件等号，可得到该类核中最大的入口信号系数：

\[
\alpha_{\rm new}=\frac{\alpha_{\rm old}}{\sqrt m-(\sqrt m-1)\alpha_{\rm old}},
\qquad
r=\frac1{\sqrt m-(\sqrt m-1)\alpha_{\rm old}}.
\]

此时 Σ_ξ=σ_new²Q。无需大协方差矩阵或分解：采样一张独立白噪声 η，令 ξ=σ_new(η−UDη) 即可。空间 2×、m=4 时为

\[
\boxed{
Z_{\rm new}=\frac{UZ_{\rm old}}{2-\alpha_{\rm old}}
+\frac{2(1-\alpha_{\rm old})}{2-\alpha_{\rm old}}(\eta-UD\eta),
\qquad
\alpha_{\rm new}=\frac{\alpha_{\rm old}}{2-\alpha_{\rm old}}.
}
\]

η−UDη 就是在每个 2×2 块内，减掉该块四个噪声的均值。新噪声块内有负相关，恰好抵消复制旧噪声造成的正相关。总噪声协方差为

\[
r^2\sigma_{\rm old}^2UU^T+\sigma_{\rm new}^2Q
=\sigma_{\rm new}^2P+\sigma_{\rm new}^2Q
=\sigma_{\rm new}^2I.
\]

等价的逐像素检查：旧噪声对块内每个方差、每对协方差均贡献 σ_new²/4；新噪声对方差贡献 3σ_new²/4、对不同像素的协方差贡献 −σ_new²/4。因此总方差 σ_new²、不同像素协方差 0。Gaussian 性保证这些噪声分量独立。

PF 论文的单位方差相关噪声 n′ 可写成 n′=(2/√3)Qη，块内非对角相关系数 −1/3；代入其公式 15，与上式是同一 Gaussian 核。不同随机矩阵分解或随机数顺序不会保证相同 seed 下逐像素相等，这里主张分布等价。现有 controlled orthogonal 实现使用相同 η 时的逐元素结果另见本节 CPU 检查。

### 3. 阶段内部依然可以使用 RDD

令下一阶段结束信号系数为 b，X_f 是该阶段真实 fine clean，X_c=DX_f。把阶段入口改为刚推导的 α_new，并设

\[
\tau=\frac{t-\alpha_{\rm new}}{b-\alpha_{\rm new}},\qquad
Z_t^{RDD}=t\big[(1-\tau)UX_c+\tau X_f\big]+(1-t)\epsilon_f.
\]

在 t=α_new 时恰好为目标入口，在 t=b 时为 bX_f+(1−b)ε_f；中间仍是 RDD 的细节系数 tτ，而同端点 PF 为 bτ。对应训练速度仍由该 RDD 路径求导。**因此能组合 RDD 的内部路径与 PF 的边界保证，不需要把整个算法换成 PF。** 多阶段逐段匹配时，同样的论证可迭代；它需要每阶段模型及积分理想正确。

这里 α_new<α_old 是信号水平回退，不是完全去噪后从纯噪声重启；不能把回退幅度直接当作实际多出的 NFE。实际每阶段步数需要单独分配。无需为了边界额外调用模型；只有 Gaussian 采样、块均值和线性运算。

### 4. 为什么不能保持原入口系数再随便补噪？

若坚持 α_new=α_old、σ_new=σ_old，以及独立白噪声训练目标，则 r=1，所需补噪协方差为

\[
\Sigma_\xi=\sigma_{\rm old}^2(I-UU^T).
\]

其 P 子空间特征值为 (1−m)σ_old²<0，不是合法协方差。所以**在“缩放上采样旧 noisy 状态＋独立额外 Gaussian 噪声”这一类转移中，原入口系数不能全保持不变**。这不是任何非线性、先验相关或后验采样方法都不可能的证明；若改变训练噪声的协方差，也属于另一个设计问题。

当前 RDD fresh 用 clean 条件均值替换真实 clean，产生的 α² U E[Cov(X_c|Z_c)] Uᵀ 缺项，在这个方案里不再由均值替换引入，因为转移直接作用于完整 noisy 状态。它并没有估计或恢复某个样本的后验协方差，也不修复前面已有的模型/积分误差。

当前训练好的 RDD checkpoint 使用旧阶段入口和内部系数，不能仅凭上述推导认证直接换边界后的训练／推理一致性。旧模型上可以做采样消融，但具有该匹配保证的完整新配置需要同步训练端点、时间编码及 RDD 速度目标，或另行验证适配效果。

### 5. 原 RDD-VP 也有对应构造

若 σ²=1−α²，同一 PSD 等号给出

\[
\alpha_{\rm new}=\frac{\alpha_{\rm old}}
{\sqrt{m-(m-1)\alpha_{\rm old}^2}},
\qquad
\bar\alpha_{\rm new}=\frac{\bar\alpha_{\rm old}}
{m-(m-1)\bar\alpha_{\rm old}}.
\]

这里 bar α=α² 是通常 DDPM 的累计信号方差系数；转移依然是 (α_new/α_old)UZ_old+σ_new Qη。这是相同边界原则在 VP 日程下的表达，不是 PF 的 FM 时钟，也不是已接入旧 DDPM checkpoint 的采样器。适用前提仍包括阶段出口具有上述 clean+独立白噪声边缘、训练入口配套。

### 6. 本轮验证与实现边界

新增 [CPU 检查脚本](tools/validate_rdd_pf_boundary_20261103.py)，记录 [RDD_PF_BOUNDARY_20261103.json](evidence/pyramid_flow_rdd_overlap_20261102/RDD_PF_BOUNDARY_20261103.json)。24 组代数检查覆盖 m=2/4/8、四个信号水平、FM/VP 两种日程，检查总协方差、合法补噪、固定入口的负特征值及离散非 Gaussian clean 的特征函数例子；8 组组件检查覆盖图像／视频张量及四个信号水平，使用现有 PyramidFlowProcess 的 orthogonal 边界与 RDDFlow：

- 相同噪声实现下，推导边界与现有 PF controlled 边界逐元素吻合。
- 修改起点后的真实 RDDFlow 入口与该边界吻合，终点正确；中间路径仍区别于 PF。
- 全部恒等式最大绝对残差 **1.7763568394002505e−15**；CUDA 未初始化。

这不是作者完整 native pipeline、离散 scheduler、学习网络 forward/backward、FID/VBench 或旧 checkpoint 兼容性认证。本轮只增加研究检查与推导，**没有注册可直接用于正式训练的混合配置**；现有 RDDProcess 生产入口仍只接受 fresh/coupled。使用 orthogonal PF 分支且测试夹具显式设置出口噪声系数，没有测试原生时钟量化及 native_psd 随机数顺序。源码 hash 在 JSON 中。

复跑请用新输出路径，保留已有证据：

```bash
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" unified_multires/tools/validate_rdd_pf_boundary_20261103.py \
  --output /tmp/rdd_pf_boundary_recheck.json
```
