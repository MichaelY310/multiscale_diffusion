# Unified 的支持范围与原版对照约定

2026-10-15增量：当前共同模型38项完整采样对照与21项原生采样复核通过，详见[采样逐方法范围](evidence/sampling_parity_20261015/REPORT_ZH.md)。Pyramid Flow为明确PSD兼容后原代码对照；f-DM/PyramidalWan/Non-Uniform为独立公式对照，不能继承原仓库parity结论。随机CPU计算一致性不等于最佳FID保证。

2026-10-12本轮范围更正：SPEED论文第4.3节包含频谱变换微调目标，固定官方代码仅发布免训练推理；现有认证只覆盖推理，不能表述为论文没有训练方法。用户将Diffusability/CAT移出当前主研究范围，保留历史证据。详见[当前范围澄清](EXPERIMENT_SCOPE_CLARIFICATIONS_20261012_ZH.md)。这是文档更正，没有新增或重跑数值认证。

2026-10-12增量：DiMR已补随机原版VAE的RGB全链路对照，包含完整83.65M参数VAE、256×256合成RGB、缩小DiMR的独立训练F/B与三种CFG采样后浮点解码，全部逐字节一致；VAE按作者冻结，输入导数另作诊断。见[补验报告](evidence/dimr_rgb_20261012/REPORT_ZH.md)。用户当前接受共同随机权重作为核心准入、不以真实资产或GPU验证作为本轮阻塞；共同架构实验的界限见[澄清文档](EXPERIMENT_SCOPE_CLARIFICATIONS_20261012_ZH.md)。原76项报告保留其执行时哈希和范围。

本项目统一多分辨率生成方法的数学组件、训练目标和采样程序，并提供共同训练工程。主张是：在明确选定的公开算法配置、相同输入与权重下，框架保留原实现的计算；不是把所有作者仓库的训练服务、依赖版本、论文全部变体或最佳成绩重新复制一遍。

逐方法支持合同位于 [SUPPORTED_REPRODUCTIONS.json](SUPPORTED_REPRODUCTIONS.json)。本次76项固定代码CPU任务及补充验收全部通过，见[最终逐方法表](evidence/unified_scope_20261012/REPORT_ZH.md)和[机器证据](evidence/unified_scope_20261012/AUDIT.json)。完整重跑、回读与清理入口为 [run_supported_reproduction_checks.sh](run_supported_reproduction_checks.sh)；逐方法表由 `tools/build_scoped_reproduction_report.py` 从实际报告生成。

## 统一什么

统一配置与扩展接口覆盖：表示/投影，阶段内信号与噪声路径，跨尺度噪声耦合，时间抽样及网络时间映射，预测头与目标，损失权重和归约，尺度边界状态转移，以及求解器/CFG。网络、codec 和条件编码器通过明确接口接入。

保留作者 backbone 是有意的工程选择。对照两侧可以共享其网络类、原始数值 solver 或 reader，但训练目标与路径的组装要分别执行作者源码与 unified 实现。报告明确哪些组件复用，不能声称所有网络和 solver 均为独立重写。

部分方法需要整体的 batch/target 组件，例如 PPFlow 的 packing 和 DiMR 的多输出目标。这是有约束的组合接口；不会为满足“任意拼装”而允许数学不兼容的 target、schedule 或 sampler。通用组件的改组属于新的 controlled experiment，不自动继承 native 配置的证书。

## 可以不同，以及不能静默改变的内容

可以使用本地离线数据代替作者云端数据服务，使用共同日志、检查点格式和启动器；CPU 测试可以对两侧作相同的设备适配。固定输入的数学对照可以从已编码 latent、text embedding 或 teacher feature 开始。这些选择必须记录测试输入所在的边界。

目标、噪声相关性、时间分布、loss weighting、阶段归约、CFG、必要的 codec/teacher 不能静默替换。如果更换的是这些内容，应明确是方法变体或缩小支持范围。原版不同 backbone、私有训练数据、未发布 trainer 不属于可以凭接口名字补造出来的复现。

共同 optimizer/batch/预算用于方法公平比较时，两侧都按相同受控设置执行；原版 optimizer/LR/EMA 配方的对照另列。共同设置的成绩应写成“统一设置下的复现实验”，不直接冒称论文最佳主表复现。

## 数值验收

1. 固定源 commit、实际配置、测试代码及实现哈希。验收过程中不修改被测文件；发生变更的任务必须重新执行。
2. 同一非零初始化、输入和 Torch/NumPy/Python 随机状态，分别运行作者与 unified 图。合成和缩小夹具明确记录；另有完整发布 backbone/网格的检查。31份完整image配置每份两次forward/backward，不在这一项更新optimizer；128步权重更新使用缩小网络。完整codec和Wan的执行范围另见各自报告。
3. 对已执行模块的输入、输出、activation gradient 和处理前 parameter gradient 比较 shape、dtype 和原始字节。None gradient、未执行层、functional 操作与真正的零梯度分开报告。
4. 对关键时间抽样、目标、损失、跨尺度边界及采样状态单独验证；比较最终 latent 和量化前浮点像素。不能仅靠最终 PNG 相同或零初始化通过。
5. 多步检查更新后的模型/optimizer/EMA/RNG，检查新进程保存恢复。CPU Gloo、worker、FP16 GradScaler 是明确的 CPU 生命周期验证，不能当作 CUDA/NCCL/FSDP 认证。
6. intentional deviation、原源码缺陷修补、尚未完成的实现必须分别记录。限定范围内若存在未解释数值差异，不能给该范围标记 exact。

以上是实现一致性证据。没有执行的正式训练、FID/IS/VBench、多 seed 及 GPU 性能实验，不会随实现对照通过而自动成为已完成。

## 显式兼容选项和已知差异

| 情况 | 选择与依据 | 如何证明对应关系 |
|---|---|---|
| RDD decoupled + 非默认 loss weighting | 本地原辅助函数漏加 `mse_d`。基础 RDD 不受影响。默认保留 drift loss 是有意修正；原样行为用 `omit_drift_for_native_weighting=true` | 原辅助函数、兼容模式逐字节相同；修正版等于兼容模式加 `mse_d`，负对照证明两者确实不同 |
| LapFlow square sampler | 源码交换 noise/time 实参。默认拒绝不明确的 square 行为；`square_policy=source_bug` 保留原行为，`corrected` 显式修正 | 分别对照原调用和最小修补后的作者调用；linear/cosine 仍按原始算法检查 |
| PPFlow `learn_sigma=False` | 原模型有两处 channel/unpatchify 缺陷，必须明确选择 `native_ppflow_no_sigma_compat`；发布的默认 sigma 分支不用此补丁 | 先确认原分支失败，再比较仅修这两处的模型与 unified loss/采样；源文件不改 |
| PPFlow 在线 VAE | 原入口保留参数 requires_grad 标志并在 no_grad 下编码。CPU 冻结参数可能选择不同 Linear kernel，产生微小舍入差 | 原样模式 `freeze_parameters=false` 与原入口一致；编码按 microbatch 执行以保持 posterior/time/noise RNG 顺序 |
| LapFlow 分解再重构 | 浮点加减不能保证数学恒等式原样还原；两个 FP32 诊断的最大差为约 2.38e-7 | 单列恒等式诊断；它不是原版与 unified 的训练/采样差异，不计作两侧 exact 失败 |
| SSD stop-only / checkpoint | CPU 上限只截断原 while，不改变原 LR 总预算；原最后一次保存缺少完整恢复状态 | 实际作者 TrainLoop 对照；unified 保留完整恢复状态是工程改进，文件格式不要求与缺状态的原文件相同 |
| CPU attention / 导入依赖 | 原 CUDA kernel 通过明确 CPU 实现或导入适配运行 | 两侧在相同 CPU 实现上对照；不声称验证原 GPU kernel |
| PixelFlow 推理类别索引 | 作者 pipeline 显式构造 `int32`；公共标签接口通常给 `int64`。两者像素/RNG已相同，但原始索引字节不能算相同 | Dopri5额外检查给未修改的 `Method.sample` 传入作者同类型 `int32` 标签，比较四阶段原始输入/ODE轨迹、参数和浮点像素；原先类型不匹配的记录保留 |

## 可复跑的原版对照

以下不直观但会影响原计算的细节已经保留，不会因为共同训练工程而省略：

- Relay 的增强维度、像素求和/loss multiplier、按图像数计算的 LR/EMA；单 CPU 夹具没有因此被宣称为原8卡全局batch轨迹。
- MDM256 的 FP32 累积次序：原代码先 backward，再处理报告用的累积除法。原样配置使用对应的累积归一化策略；不能直接替换为共同 Trainer 默认 mean 后仍声称轨迹不变。
- DFM 的阶段抽样、REPA 独立标量归约以及源码额外消耗的 NumPy 随机数；真实 teacher 编码器在特征输入边界之外。
- DiMR 的 Python CFG dropout 和其 guidance scale/power 语义，不能以普通 CFG 乘法代替。
- DCTdiff 的 `*_std` 字段实际来自作者频率统计规则，不能按字段名字换成通常的标准差；量化前 decode 也单独比较。
- Diffusability 所测作者重建/KL 张量形状、广播和正则相加次序，保留原计算图；只认证这份明确的 codec loss 组合。
- RDD 每个阶段先求 batch mean、再求阶段和，不能任意换成先加向量再归约；自适应时间抽样还必须保存 loss history。
- PPFlow 的 packing、端点插值运算树、VAE posterior 抽样位置与微批次边界；这些都会影响严格数值与 RNG 对照。

DiMR 有两种显式配置位置：公共训练器中的 label dropout，或作者 `CFGDataset` 所在的 reader dropout。对齐作者 reader 时，设置 `training.data.kind="moments_npy"`、`training.data.cfg_dropout=0.1`、`training.data.null_label=1000`，同时将 `training.conditioning.y` 设为 `{"kind":"labels","classes":1000,"allow_null":true,"dropout":0.0}`。不能在两个位置重复丢弃。已直接执行作者 `FeatureDataset/CFGDataset`，对 0、0.1、1 三种概率分别检查44次读取：moments、标签、Python RNG 完全相同；null label 1000 能穿过统一条件接口。测试只读取本地生成的11条记录，不把作者硬编码的2,562,334条数据规模伪称为已经检查。证据：[author_reader_contracts.json](evidence/unified_scope_20261012/author_reader_contracts.json)。

以下命令均只用 CPU，无模型/数据下载。Gloo 和真实 worker 检查需要本地进程 IPC 权限；没有 GPU 许可要求。

```bash
# 完整重跑、独立回读、生成报告并清理临时产物；--plan只查看命令。
bash unified_multires/run_supported_reproduction_checks.sh --workers 2

# 完整支持范围：逐配置网络、128步、采样、恢复和额外分支。
# 这个底层入口保留payload，便于独立诊断；完整入口负责最终清理。
bash unified_multires/run_scoped_cpu_validation.sh --full --workers 2

# 一个直接例子：运行作者 PPFlow main（显式 CPU/随机小模型适配）
# 与统一 Method/Trainer 比较，而不是让两侧共用训练循环。
bash unified_multires/run_scoped_cpu_validation.sh --only native_ppflow2 native_ppflow3

# 原缺陷与显式修正的关系，以及 RDD 可选目标组合。
bash unified_multires/run_scoped_cpu_validation.sh --only fix_lapflow fix_ppflow rdd_optional
```

每次任务保存启动参数、日志和被测文件快照。正式资产不属于临时清理范围；生成的测试 checkpoint/像素/差异 Tensor 在比对、恢复检查和哈希归档完成后清理，保留证据文档。

新增 RDD 示例 [rdd_optional_objectives.json](configs/native_fixtures/rdd_optional_objectives.json) 是随机小模型训练夹具。它演示 `p2 + 两阶段 REPA + loss_second_moment`，从本地数据的 `conditions.teacher_features` 接收特征；不下载或替代 DINO。27组新增目标组合使用 joint、learned variance、两阶段真实缩小 DiTMultiRes，交叉检查3种loss weighting、3种REPA位置和3种时间抽样。基础 joint/decoupled 在基础变体矩阵中另验；decoupled加权的漏drift兼容选项通过原辅助函数对照，不能把这27组描述成所有joint/decoupled选项的笛卡尔积。单进程目标矩阵之外，另用真实两rank Gloo执行作者 `update_with_local_losses` 和 unified 聚合：12次更新、每rank本地batch4，history/counts/概率/抽样时间/重要性权重/NumPy RNG逐字节相同，见 [分布式采样器报告](evidence/unified_scope_20261012/rdd_distributed_sampler.json)。这个证明只覆盖采样器collective及均衡batch，不扩展为完整GPU多卡RDD训练认证。真实研究使用时必须绑定明确的 teacher/codec 及数据处理。

早先 Pyramid Flow/PyramidalWan 与 FastVideo 的视频工程独立列在项目交接，不属于这份16方法合同，不能继承本矩阵的 image/core-method 证书。
