# 已有统一生成框架与本项目的比较

**2026-10-11 补充**：用户明确定位为“统一数学框架 + 对应训练实现 + 从中推导 RDD”后，新增[三项贡献结构的直接先例与理论复核](PAPER_POSITIONING_REVIEW_20261011_ZH.md)。EDM、Score-SDE、OT-CFM、SSD、DFM、统一 U-Net 和 EDA 均须比较；其中 SSD/DFM 已有多分辨率框架及有效实例，不能只作为普通算法/工程库看待。下文保留为此前检索，不是完整定位的最终近邻清单。

检索日期：2026-10-10。问题：是否已有工作统一多分辨率 diffusion/flow，并提供可组合的训练和推理工程；其完成度与本项目如何比较？

依据是作者论文、官方项目页、仓库说明和官方文档，以及本地[严格审计报告](evidence/strict_original_unified_audit_20261010/REPORT_ZH.md)。此次没有安装或运行下列外部框架，没有下载模型/数据，没有使用GPU。外部项目的实验成绩与功能状态属于作者公开报告，不是本次独立复验。

## 判断

已有相近的数学框架，也有成熟的多算法工程框架。“统一理论—组件化实现—实验比较”的研究路线有明确先例。不能把通用算子插值、可切换预测目标、模块注册、可组合pipeline或者统一trainer单独宣称为首次贡献。

在本次检索并查看的项目中，尚未找到明确同时完成以下组合的公开证据：系统映射我们关注的十余种空间多分辨率论文、可组合跨尺度路径/噪声/边界/目标、支持图像和视频训练推理、逐论文报告与作者完整入口的activation/gradient/最终像素一致性。这是有范围的检索结论，不是证明这样的项目不存在；本项目目前也未完整达到这个标准。

## 最接近的公开工作

| 工作与一手来源 | 已公开的统一与工程内容 | 与我们的关系和完成度判断 |
|---|---|---|
| [Multitask Learning with Stochastic Interpolants](https://arxiv.org/html/2508.04605)，[官方代码](https://github.com/multitaskstochasticinterpolant/Code-Multitask-Stochastic-Interpolants) | 将标量插值推广到向量、矩阵和线性算子，推导噪声/数据估计器与ODE/SDE；讨论多尺度和多任务生成。代码公开MNIST、φ4、maze的部分实验notebook。 | 与我们的算子路径核心直接重叠。公开代码定位是本论文部分实验，不是当前可见的跨十余篇空间多分辨率方法的配置库。理论先例必须重点引用。 |
| [f-DM: A Multi-stage Diffusion Model via Progressive Signal Transformation](https://jiataogu.me/fdm/) | 将渐进信号变换与多阶段diffusion统一，含分辨率变化和阶段衔接。 | 是空间多阶段数学叙事的重要先例。此次查看的官方项目页仍标示代码/数据待发布；不能把论文方法或第三方实现算成已发布的作者通用工程库。 |
| [Scale Space Diffusion](https://arxiv.org/abs/2603.08709)，[官方仓库](https://github.com/prateksha/ScaleSpaceDiffusion) | 广义线性退化与Gaussian噪声的数学框架；Flexi-UNet；公开训练、预训练采样脚本和主模型checkpoint链接。 | 数学与空间多分辨率任务非常接近。公开仓库主要实现SSD，而非当前可见的Relay/DFM/LapFlow等跨论文组件目录。发布主结果资产的程度超过我们尚未复现主成绩的状态。 |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching) | 共同接口实现基础CFM、OT-CFM、Target-CFM、Schrödinger Bridge CFM、VP-CFM；多个论文方法、训练runner、图像例子和CIFAR10质量结果。 | “统一理论并做多方法可用库”的强先例。主要统一耦合与概率路径，公开目录不是我们这组空间多分辨率论文的专门复现矩阵。 |
| [Flow Matching Guide and Code](https://github.com/facebookresearch/flow_matching) | 拆分path/scheduler/loss/solver，支持连续、离散flow；提供完整CIFAR10、face-blurred ImageNet训练例子和CI。 | 我们的路径、时间、目标/损失、求解器抽象已有直接工程先例。其公开标准训练例子和维护设施较完善；多分辨率论文的特定阶段边界仍需另做映射与验证。 |
| [DiffuseNNX](https://github.com/willisma/diffuse_nnx) | JAX/NNX统一SiT、EDM、MeanFlow等接口，含sampler、ImageNet训练、FID、分布式checkpoint；README列出模块比较测试和真实训练/推理FID表。 | 很接近我们希望达到的多方法研究工程。作者同时标注WIP、REPA质量差异、RAE仅推理、GPU测试不足。它已有正式质量证据，但不能因此替它认证所有作者入口或全网逐位一致。 |
| [Modular Diffusers](https://huggingface.co/docs/diffusers/modular_diffusers/overview)，[设计说明](https://huggingface.co/docs/diffusers/en/conceptual/philosophy) | 可复用、替换、组合的pipeline blocks；顺序、循环和条件容器，共享state及components管理。Diffusers也有[IF多级像素生成](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)和[Stable Cascade](https://huggingface.co/docs/diffusers/api/pipelines/stable_cascade)。 | 工程模块化与多级生成已有成熟生态。官方明确standard/modular pipeline面向推理；库另外有训练支持，不能将其误写成整个库只支持推理。它不是当前可见的跨上述论文统一训练数学及作者parity合同。 |
| [CleanDiffuser论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/9e08a1db869a9646418e3371b24c6ae6-Paper-Datasets_and_Benchmarks_Track.pdf)，[官方仓库](https://github.com/CleanDiffuserTeam/CleanDiffuser) | 统一diffusion model、network、guidance及数据/环境接口；论文比较9种算法、37个RL/IL环境，并做组件分析。 | 研究叙事和工程目标非常接近，领域是决策而非空间多分辨率图像。正式基准和消融已公开，是框架研究如何完成实验闭环的具体先例。 |
| [FastVideo](https://github.com/hao-ai-lab/FastVideo) | 视频模型推理、全参/LoRA微调、DMD2与因果蒸馏、预处理、FSDP2/序列并行等统一工程。 | 视频训练基础设施与工程集成本身已有先例；多分辨率信号/噪声/边界的数学合同仍是不同问题。此次只读线上说明，不修改本地dirty FastVideo。 |

补充：[DiffusionLab](https://github.com/DruvPai/DiffusionLab)也提供JAX的过程、预测参数化、训练loss和采样接口。它进一步说明通用diffusion/flow组件拆分已有多种实现；不能凭本次说明页阅读判定其全部实现质量。

表中“主要实现自身方法”“未看到跨论文目录”等是依据公开定位与所查看资料作出的范围判断，不代表代码不能扩展。没有逐项执行这些外部项目，也没有验证它们能复现全部最佳成绩。

## 数学上最需要处理的重叠

我们的基本路径写成

\[
z_t=A_t x+B_t\epsilon.
\]

Multitask Stochastic Interpolants的Definition 2.1写成

\[
I(\alpha,\beta)=\alpha x_0+\beta x_1.
\]

令其噪声\(x_0=\epsilon\)、数据\(x_1=x\)、\(\alpha=B_t\)、\(\beta=A_t\)，可识别相同形式的路径核心。该论文还允许一般端点联合分布\(\mu\)，用MSE学习噪声/数据条件期望，推导沿算子路径的流和扩散。因此，“算子化加噪”“两个端点的估计目标”“一般端点耦合”本身不能当作我们的新概念。[论文§2](https://arxiv.org/html/2508.04605)

这不是完整框架或RDD等价性证明。不同维度的表示与嵌入、非连续阶段边界kernel、跨阶段随机数联合分布、离散求解、损失归约以及训练生命周期，仍须分别比较。论文主张的同一个模型在算子空间中支持多任务，与我们通过配置切换和适配多篇作者实现，也不是同一个工程合同。

## 我们的现有状态与可成立的区别

本项目的数学目录明确记录表示、信号算子、噪声协方差、跨尺度/时间耦合、时间坐标、采样分布、backbone、预测目标、loss metric、阶段边界、solver、优化和数据资产。这个细粒度比较目录有研究价值，但列出更多轴并不自动证明全部可自由组合，也不自动构成数学创新。具体结构见[FRAMEWORK_MATH_ZH.md](FRAMEWORK_MATH_ZH.md)。

当前登记14种外部发布方法加RDD，另有仅数学映射的文献。**登记数、source-core对照通过数与最佳训练结果复现数是不同指标。** 当前事实是：

- 33份配方的native完整CLI配对均未认证。
- 近期30份paper、11份fair短测是CPU缩小作者网络、合成输入、每份两次更新；共享backbone、数据与Trainer，实际捕获模块边界的activation/gradient相等。
- 13份既存小sampling tensor的字节对照相等，范围包含latent、uint8和小随机网络；不是全部真实最终浮点像素认证。
- Relay增强/EMA/梯度缩放、CLI参数、数据尾部/标签、MDM额外RNG等仍有已确认偏离；通用组件还有容差内非零误差。
- 未完成全部论文最佳配置的正式训练、FID/其他主结果及真实资产全过程复验。此前PF/PW视频工程有独立证据，不能并入上述33份配置的认证数。

依据：[严格审计](evidence/strict_original_unified_audit_20261010/REPORT_ZH.md)、[全部缺口](ORIGINAL_UNIFIED_GAPS_TODO_ZH.md)。此次研究没有改变这些认证状态。

所以，按通用工程、真实训练与质量验证衡量，不能说我们完成度超过Diffusers、Meta Flow Matching等项目。按“空间多分辨率论文的跨方法覆盖与来源审计”衡量，我们有较专门的建设方向，但没有足够公开比较和自身验收来声称第一或最高完成度。

可继续建立的区别是：用明确合同比较各篇方法的跨尺度噪声、边界、目标与loss，区分native和controlled transplant，给出可追溯的配置、源版本和逐分支一致性证据；然后通过可运行入口和真实主结果证明兼容性。是否独有及是否为新理论，仍须逐项论证。

建议对外定位为“面向多分辨率图像与视频生成的组件化比较与可审计复现框架”。待原生入口、真实数据/资产、主结果和组件消融完成后，再增加“完整复现”一类描述。
