# RDD Flow Matching 推导与求解器审计

日期：2026-11-01，洛杉矶。范围：现有 RDD-FM 数学和源码、相关求解器文献、CPU 解析分布实验。没有 GPU、模型或数据下载，没有正式训练或生成质量测量。

RDD 的残差路径在 flow matching 中仍有解析结构，可以用一个 velocity head 恢复 clean 条件均值与 score，并构造利用已知路径系数的积分器。但普通 flow matching 已经能够使用 DPM-Solver++、UniPC；“只有 RDD/SDE 才能用高阶求解器”不能作为贡献。本文给出的积分器是研究候选，不能据此宣称新颖性或优于已有模型的生成质量。

已有默认路径的代数检查通过。另发现三类问题：旧注释与部分论文表述过强；unified 的非投影算子漏了逆公式适用性检查；以预测 clean 替代真实 clean 的边界拼接存在分布近似，即使模型是精确的 Bayes velocity 也不能自动消除。

## 已有实现与文档核查

| 文件或入口 | 本轮结论 |
|---|---|
| `multiscale_diffusion/docs/theory/RDD_DIFFUSION_FM_DERIVATION_TUTORIAL_ZH.md` 第19–22节 | 阶段路径、速度求导、投影逆和 Wan 时间反转基本正确；第21节的分布匹配结论需区分 oracle clean 与网络条件均值 |
| `multiscale_diffusion/docs/research_docs/RDD_NOVELTY_FM_VIDEO_AUDIT.md` | 已经否定“PyramidalWan不能用高阶solver”等说法；其中8月环境和执行状态不是当前状态 |
| `multiscale_diffusion/diffusion/flow_matching_with_drift.py` | 实际 q_sample、velocity、clean inverse 与默认路径相符；文件头及 q_sample 注释仍错写阶段出口为 clean、或 residual=0 后可用普通 clean shortcut |
| 同文件 `sample_ode` | 真实实现为 Euler / midpoint RK2；`dpm2` 只是 midpoint 的兼容别名，不能当作 DPM-Solver-2 的新证据 |
| `multiscale_diffusion/rdd_video/flow_matching_video.py` | Wan 目标是 dx/dsigma，符号与 sigma=1-t 正确；默认投影逆正确；文件头仍有“f(b)=0保证出口是clean”的错误描述 |
| `FastVideo-RDD-minimal/rdd/flow.py` | 默认边界0/.4/.7/1，空间 average/repeat，解析逆正确；transition 使用预测 clean + 全新独立噪声 |
| `FastVideo-RDD-minimal/rdd/comparison/sampling.py` | Euler/AB2/Heun，跨边界重置历史并计入额外端点网络调用；并没有本文的新积分器 |
| `unified_multires/umdiff/paths.py::RDDFlow` | 默认路径与旧2D实现相符；原 override 漏查非幂等算子，已增加保守的受支持正交投影检查 |
| `multiscale_diffusion/2d/train_RDD_FM_imagenet256_unified.py` | 当前默认已是生成顺序400/300/300，并拒绝不匹配的恢复；旧FM-v2 checkpoint可能仍是300/300/400，不能仅凭名字合并 |
| `multiscale_diffusion/paper/main.tex` 视频solver段落 | “标准clean转换直接用于RDD无效，换成解析clean即可”的描述不够准确，见后文：ODE forcing和真实clean必须区分，积分系数也必须匹配 |

旧论文中“PyramidalWan只兼容Euler”的一些草稿在 `\iffalse ... \fi` 内，不能把它们误认为当前编译正文。另一方面，当前正文仍需修订上述 forcing/clean 区分，以及将预测clean下的边界精确性限制写清楚。本轮不改历史论文源码或默认采样轨迹；修正依据集中记录于此。

## 当前 RDD-FM 的准确数学形式

采用生成时间 t：0为噪声，1为数据。在一个固定分辨率阶段 [a,b]，令 L=b-a，P 为 average-pool/repeat 正交投影，Q=I-P。以下 clean X 指该阶段的干净数据，epsilon 独立服从 N(0,I)。

\[
Z_t=tX+(1-t)\epsilon+\beta(t)(P-I)X,
\qquad \beta(t)=\frac{t(b-t)}{L}.
\]

等价地，

\[
Z_t=A_tX+\sigma_t\epsilon,
\quad A_t=tP+g(t)Q,
\quad g(t)=\frac{t(t-a)}L,
\quad\sigma_t=1-t.
\]

最粗阶段不启用漂移，A_t=tI。较细阶段的条件训练速度为

\[
U_t=PX+\frac{2t-a}{L}QX-\epsilon.
\]

MSE 最优网络为 v*(z,t)=E[U_t|Z_t=z]。这个条件平均后的生成 ODE 一般不是单个训练样本插值轨迹；不能用“条件轨迹很简单”替代对生成向量场的分析。

入口与出口为

\[
Z_a=aPX+(1-a)\epsilon,
\qquad Z_b=bX+(1-b)\epsilon.
\]

因此 b<1 时出口仍带噪声。beta(b)=0 也不代表 beta'(b)=0，所以不能仅据 beta=0 使用普通 FM 的 clean 读出。

Wan 使用 sigma=1-t，预测 w=dx/dsigma=-v，并输入 sigma*1000。改变采样网格的 shift 不意味着额外给 w 乘时间 Jacobian；只有真正把 ODE 独立变量换成新的连续时钟时，才需相应链式法则。

## 一个 velocity head 足以给出 clean 与 score

定义

\[
B_t=A_t+\sigma_t A'_t=P+k(t)Q,
\qquad k(t)=\frac{2t-t^2-a}{L}.
\]

则有

\[
Z_t+\sigma_t U_t=B_tX.
\]

对最优网络取条件平均，得到

\[
m(z,t)=E[X|Z_t=z]
=B_t^{-1}[z+\sigma_t v^*(z,t)]
=Py+\frac{Qy}{k(t)},\quad y=z+\sigma_t v^*(z,t).
\]

实际网络用 v_theta 代替 v*，得到估计 m_theta。它不是每个样本的真实 clean，也不自带后验不确定性。Wan 写法为 y=z-sigma*w。

这里有一个有用的结构性质：A_a 的高频系数为0，信号算子降秩，但

\[
k(a)=\frac{a(1-a)}L>0\quad (0<a<b\le1).
\]

所以在新频带刚出现时，velocity 仍允许有限条件数的 clean 读出。默认 [.4,.7] 的 k 从 .8 增至1.7，[.7,1] 从 .7 增至1。这个性质只在相应阶段和投影条件下成立；不能把它外推到带漂移的 t=0、任意上采样、或非线性 VAE 重编码。

在 sigma>0、Gaussian noise 独立于数据时，

\[
s^*(z,t)=\nabla_z\log p_t(z)
=-\frac{z-A_tm(z,t)}{\sigma_t^2}.
\]

因此阶段内同一边缘密度还允许

\[
dZ_t=\left[v^*(Z_t,t)+\tfrac12\gamma(t)^2s^*(Z_t,t)\right]dt
+\gamma(t)dW_t.
\]

这里 gamma 是另行选择的瞬时扩散强度，不是路径的累计噪声 sigma。该式使用生成方向递增时间；其 Fokker–Planck 扩散项和新增 score 漂移项相消。需要通常的正则性与端点条件。有限网络估计不提供精确密度保证，跨分辨率边界还需独立处理。这是 stochastic interpolants 的既有原则在 RDD 路径上的具体化，不是新的一般 SDE 理论。

## 利用路径系数的积分器

由 clean 读出恒等式，可将任意给定网络的 ODE 精确重写为

\[
\dot z=-\frac z{\sigma_t}+\frac{B_tm_\theta(z,t)}{\sigma_t}.
\]

积分因子给出阶段内的恒等式：

\[
z_s=\frac{\sigma_s}{\sigma_t}z_t+
\sigma_s\int_t^s\frac{B_u}{\sigma_u^2}m_\theta(z_u,u)du.
\]

这个恒等式本身不需要重新引入 SDE。它与 DPM-Solver/DEIS 的已知系数解析积分思路相通。

### 一次网络调用的路径修正

在一步中冻结 m_theta=m_t，得到

\[
z_s=\frac{\sigma_s}{\sigma_t}z_t+
\left[A_s-\frac{\sigma_s}{\sigma_t}A_t\right]m_t.
\]

对于当前二次包络，令 h=s-t，可进一步化简为

\[
\boxed{z_{t+h}=z_t+h v_\theta(z_t,t)+\frac{h^2}{L}Qm_t.}
\]

最粗阶段无 Q 修正，退化为 Euler。固定真实 clean/noise 的条件轨迹上，这一步精确积分已知二次曲率；但一般网络下 m_theta 沿轨迹变化，因此整体仍是一阶方法。不能因为含 h² 就宣称二阶，也不能将条件 oracle 的精确性写成真实生成过程一步精确。

该方案只需现有 velocity head 加 pool/repeat，无第二次模型前向；是否更快、更好仍需按实际延迟和质量验证。

### 用历史 clean 预测构造阶段内二阶方法

将 m(u) 近似为 m_t+(u-t)D_t，其中

\[
D_t=\frac{m_t-m_{t_-}}{t-t_-}.
\]

记 q=1-t、r=1-s、h=s-t，定义

\[
W_0=A_s-\frac rq A_t,
\quad j=h+r\log(r/q),
\quad W_1=jP+\frac{(1-a)j-rh^2/2}{L}Q.
\]

则

\[
\boxed{z_s=\frac rqz_t+W_0m_t+W_1D_t.}
\]

最粗阶段 W1=jI。在 r=0 处使用连续极限 j=h。实现使用 log1p；第一步采用上述一阶启动。一步只新增一次模型调用，历史在每次阶段切换或重新加噪时丢弃。

在远离奇异端点、m_theta足够光滑、步长比有界且满足稳定性条件的阶段内部，m 的线性外推误差为 O(h²)，积分后的局部误差为 O(h³)，得到二阶收敛。一次一阶启动的 O(h²) 初始误差与二阶全局精度相容。t=1 的一致误差界、跨阶段强收敛与有限精度稳定性需要单独论证；一个数值端点通过不等于完成这些证明。

生产代码未注册这两个新求解器。研究实现位于 `research/rdd_fm_integrator.py`，目前只在两维正交频带坐标中测试，不是已经完成 Wan/FastVideo 的可运行集成。

## 普通 Flow-DPM++ 不能被误判为数学上不可用

Wan 官方已有 `FlowDPMSolverMultistepScheduler` 与 `FlowUniPCMultistepScheduler`。标准 Flow-DPM++ 使用 y=z+sigma*v（Wan方向为z-sigma*w）。RDD 上 y=B_t m，并不是真实 clean 条件均值；但

\[
v=(-z+y)/\sigma
\]

仍然是精确代数恒等式。因而在阶段内部，把 y 当数值积分中的 forcing，保持匹配的时间、系数并重置跨阶段历史，普通 deterministic Flow-DPM++ 并不因此自动解错 ODE。它近似的是随时间变化的 y，而本文方案将已知的 B 解析积分，仅近似 m。

反而，把 scheduler 的 y 直接替换成真实 m，却保留标准 Flow 的标量积分系数，会得到 (-z+m)/sigma，通常改变原来的向量场。解析 clean 读出与 operator-aware 积分系数必须配套。

该解释不自动认证标准 scheduler 的 stochastic/SDE 模式、clean clipping、末端特例或跨分辨率跳变；这些语义需要分别匹配。

## 比积分阶数更重要的边界问题

当前 fresh transition 使用

\[
Z^+_a=aU\widehat X_c+(1-a)\epsilon_{new}.
\]

给定真实 clean Xc，这个公式可匹配下一阶段对应的条件 Gaussian。把 Xc 换成 E[Xc|Zc] 后，结论不再成立。

具体反例：Xc~N(0,tau²)，Zc=aXc+sigma*epsilon，完美网络给出 m=E[Xc|Zc]。则

\[
\operatorname{Var}(m)=\frac{a^2\tau^4}{a^2\tau^2+\sigma^2}<\tau^2.
\]

fresh lift 与目标 fine 分布的协方差差为

\[
\Sigma_{target}-\Sigma_{fresh}
=a^2\operatorname{Var}(X_c|Z_c)UU^T.
\]

在 a=.4、tau²=1、一个值重复到4个位置的例子中：目标每维方差 .52，fresh 后约 .40923，总方差少21.30%。这是精确 Gaussian 反例，不是预测真实视频会损失21%的质量。

旧2D代码的 isometric innovation completion 改善了 oracle 噪声耦合，但 epsilon_hat=E[epsilon|Zc] 也不是独立标准Gaussian；不能继承 oracle 假设。在同一反例中，其每维方差为 .49231，仍非 .52。

可以继续研究两条路线：

1. **保持当前 checkpoint 的近似修正。** 估计 coarse posterior uncertainty，再抽取 coarse clean，而非只使用条件均值。Gaussian 情况可精确补偿；一般数据仅匹配均值和协方差并不保证全分布精确。Tweedie/Jacobian 等估计已有先例，不能声称新原则。
2. **改变训练表示的精确边界设计。** 令 E=U/sqrt(m) 为等距嵌入，coarse clean 定义为 E^T X；从 zc=aE^T X+sigma*epsilon_c 构造 Ezc+sigma*(I-EE^T)eta，均值是aPX，协方差是sigma²I，且E^T zfine=zc。这个方案需要改变原来平均池化 coarse clean 的归一化或配套时钟，不是旧checkpoint的无损替换；正交补噪思路也已有相关工作。

这些误差在阶段内积分无限精确时仍可能保留。因此阶段内 solver 与边界 kernel 应分开消融；不能把新 solver 的收益全部归因于“严格连续的完整SDE”。现有系统是阶段内ODE/SDE加边界转移的组合。

## 一个有助于统一框架定位的推论

固定当前阶段的两个条件端点，要求条件路径为直线，则高频系数必须为

\[
g_{linear}(t)=\frac{b(t-a)}L.
\]

它对应框架已有的 endpoint flow，也与 PixelFlow 的端点线性插值机制相关；不能重新命名成新算法。这是过程组件层面的对应，不等于整个 PixelFlow 的训练/packing系统一致。

RDD 的 g_RDD=t(t-a)/L 在阶段内部小于 g_linear，意味着它比端点直线更晚引入高频信号，同时付出了额外的已知条件曲率。对照可以使用

\[
g_\eta(t)=\frac{t-a}{L}[(1-\eta)t+\eta b],\quad0\le\eta\le1.
\]

eta=0恢复RDD，eta=1恢复端点直线；对应解析曲率修正是 (1-eta)h² Qm/L。这个家族提供可解释消融，但插值本身不构成强新颖性。

更有价值的问题是：去掉已知条件曲率的离散化误差之后，RDD 延迟高频的训练收益是否仍然存在？实际决定积分难度的还包括条件均值沿生成轨迹的变化，而不是仅有 beta''。这可以把“RDD曲线更复杂但solver修好了”与“RDD改变了有益的学习任务”分开。

## CPU 验证结果

主证据：`evidence/rdd_fm_solver_20261101/audit_complete.json`。`initial_audit.json` 保留修补前非投影逆的反例；其他 audit 文件是补充检查过程中的小型日志，不是额外生成数据。

- 默认2D路径、unified路径、FastVideo路径的state/velocity/clean inverse，以及带真实时间维的空间时间投影通过float64代数检查；不是逐层神经网络parity。
- 有限差分验证 velocity 是路径导数。
- 非幂等 average-down/bilinear-up 之前返回错误 clean，最大误差约 .01239；现在明确抛错。默认 average/repeat 数值未改。
- 积分权重与独立SciPy数值积分的最大误差约1.11e-16。
- 固定clean/noise条件轨迹的一步解析更新误差约2.22e-16；相同例子Euler最大误差约.30427。
- Gaussian/混合Gaussian的Bayes clean恢复最大误差约1.55e-15，score恢复约9.05e-15；Gaussian SDE协方差演化恒等式约1.11e-16。
- 旧FM sampler的3项CPU检查通过；`test_component_contracts.py` 28项通过。
- 没有初始化CUDA，没有生成模型checkpoint、图片或视频。

下面比较相同NFE=16、相同64个初始状态，误差为相对高精度DOP853轨迹的RMSE。Heun/midpoint使用8步、每步2调用；其他方法使用16步、每步1调用。这里只测试单个阶段，无边界、CFG、预训练网络和解码器。

| 解析分布与阶段 | Euler | Heun | Midpoint | AB2 | 路径一阶 | 路径二阶历史 |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian [.4,.7] | .010784 | .000820 | .000271 | .000979 | .009301 | .000279 |
| Gaussian [.7,1] | .032456 | .001660 | .001112 | .001504 | .027263 | .005044 |
| 三分量混合Gaussian [.4,.7] | .024544 | .001649 | .001707 | .002029 | .019258 | .001066 |
| 三分量混合Gaussian [.7,1] | .059789 | .016186 | .009979 | .009140 | .054033 | .006233 |

路径一阶在这些例子中比Euler误差小；二阶历史在混合Gaussian例子中表现较好，但在末阶段单Gaussian中明显输给普通二阶方法。Gaussian中间阶段从64到128 NFE的观测阶分别约.998和2.050，符合一阶/二阶预期。其余分布、步数和负结果均在JSON中。

这些结果支持“推导可执行、阶数合理、值得小规模验证”，不支持“新solver最好”。没有对官方Wan DPM++/UniPC、gDDIM、AREX做真实网络或完整toy实现的对照，也没有FID/VBench证据。

## 相关工作与新颖性边界

| 工作 | 已有内容 | 对本研究的约束 |
|---|---|---|
| [DPM-Solver](https://arxiv.org/abs/2206.00927) | diffusion ODE已知线性部分解析积分，高阶加权求积 | 使用高阶solver不要求推理走SDE |
| [DEIS](https://arxiv.org/abs/2204.13902) | diffusion exponential integrator及历史插值 | “解析系数＋历史预测”不是新的一般原则 |
| [gDDIM](https://arxiv.org/abs/2206.05564) | 向非各向同性diffusion推广DDIM，包括blurring模型 | 投影/算子形式的求解器必须与其比较，不能只与Euler比较 |
| [Stochastic Interpolants](https://arxiv.org/abs/2303.08797) | flow与可调噪声SDE的共同边缘描述 | RDD-FM的SDE化是具体实例 |
| [Flow-Solver](https://arxiv.org/abs/2411.07627) | 使用历史步骤与多项式插值降低FM采样NFE | 缓存、多步、高阶FM本身已有工作 |
| [Wan官方Flow-DPM++](https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/fm_solvers.py) | flow_prediction和DPM++ | 不能宣称视频FM只用Euler |
| [AREX](https://arxiv.org/abs/2610.03483)，2026-10-02 | 利用目标矩构造矩阵传播子，解析推进affine部分并求积残差 | “首次在FM中使用矩阵解析传播子”也不成立；其分解依据数据统计，本文依据已知RDD路径，仍需进一步对照 |

本次没有确认某个完全相同的RDD阶段公式已经发表，也没有证明未发表。积分器可以推广到其他满足相同线性Gaussian/可逆条件的路径，所以“只有RDD formulation才能推导”不成立。可争取的具体贡献是：新频带出现时仍可稳定读出的条件、无需额外头的路径积分、边界不确定性审计，以及在真实预算下的实测收益。

## 下一组有判别力的实验

1. 现有RDD-FM checkpoint：Euler、midpoint、AB2、标准Flow-DPM++/UniPC阶段适配、本文路径一阶/二阶，统一真实NFE和额外边界调用，先小样本再正式FID。
2. 固定同一阶段内solver，比较fresh、现有noise completion及经过明确假设的边界改进；不得将训练表示变化伪装为纯推理优化。
3. 小DiT的RDD与端点直线过程，在同骨干/数据/预算下交叉使用通用solver和各自路径solver，检验优势来自路径还是只来自solver适配。
4. 最后再把有效组合接到Wan，固定实际sigma时钟、视频长度、提示词与CFG，报告质量/延迟曲线及边界误差。

在真实checkpoint结果出来前，最可信的结论是：RDD-FM数学结构并未失去价值，但“能够用DPM-Solver”不能独立承担贡献；更值得验证的是结构化积分是否改善真实生成，以及修正边界不确定性是否比增加阶段内积分阶数更有效。

复跑命令：

```bash
source unified_multires/activate_cpu.sh
PYTHONDONTWRITEBYTECODE=1 "$UMDIFF_PYTHON" -B \
  unified_multires/tools/validate_rdd_fm_solver_20261101.py \
  --output /tmp/rdd_fm_audit_new.json
```

输出必须使用不存在的路径。只有JSON证据，无大型测试产物。
