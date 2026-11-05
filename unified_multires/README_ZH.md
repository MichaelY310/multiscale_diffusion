> **当前入口（2026-11-04）**：unified现为RDD仓库下的一个目录。安装、CPU示例、正式实验和视频配置请先看[README.md](README.md)。本文件保留比较协议与历史范围说明；历史大型证据未全部随Git发布。

<!-- SCOPE_DECISION_20261019 -->
> **2026-10-19 用户范围确认**：Relay Diffusion 后续不再作为研究/实验候选；当前主实验 manifest 已排除，原源码与历史证据仅存档。LapFlow 今后默认选论文 ImageNet-256 XL/2 配方 `configs/paper_recipes/lapflow_imagenet_xl_2_paper.json`（T=.5、LR1e-4、EMA.9999、EQVAE），不选 `lapflow_mot_xl_2` 的作者 CLI 默认。共同 DiT 主实验已是 T=.5/LR1e-4/EMA.9999，仍遵守已定版本3共同 VAE/训练预算协议；本次未改其数学/配置。历史复现工具的显式 CLI profile 留作旧证据复跑，不是默认实验推荐。SSD 所选配置、RMD 非GAN分支按用户标准接受，不扩大既有认证范围。PF 的 PSD 噪声兼容保留理论协方差但改变 RNG 映射，原未修补 Cholesky 路径在当前环境失败；已通过的是双方应用相同显式兼容后的对照。完整 native 视频入口与当前空间 process 分开。此次只复核源码/既有证据并核验配置，未重跑数值 parity、未用 GPU。
<!-- /SCOPE_DECISION_20261019 -->

# Unified Multiresolution Diffusion / Flow

**2026-11-04 RDD 原生复现补齐**：旧图像FM现有独立入口`task=rdd_fm_native`；共享DiTMultiRes与独立三模型DDPM/FM均有训练/完整采样对照。`fresh=refresh`表示丢弃旧残差，`noise_filling`表示由旧noisy和预测clean恢复残差；历史DDPM同名选项实际fresh，勿给新变体套用旧FID。18份可运行原生配方、严格旧权重导入与范围见[配置说明](configs/rdd_reproduction/README_ZH.md)、[本轮完整验收](evidence/rdd_reproduction_20261104/FINAL_SUMMARY_ZH.md)。通用`RDDProcess`自由组合不自动等于历史FM配方；网站[全部模型](https://rdd-math-reference.aa3101132006.chatgpt.site/models.html)与[采样档案](https://rdd-math-reference.aa3101132006.chatgpt.site/sampling.html)保存实际记录和缺失项。

**2026-10-16 最终规则：按用户选择，版本3统一采用RDD式全阶段求和。** 当前18份main/controlled配置均已接入：同一图片覆盖所有训练阶段，每阶段batch mean以系数1直接相加，再统一更新一次参数；不除以阶段数、不乘原阶段概率。阶段内部数学目标、条件时间与噪声结构保留。原生版本2配方、RDD两份配置与作者源码保持不变。[本轮检查与边界](evidence/all_stage_sum_20261016/REPORT_ZH.md)。

此前同日的位置编码修改证据仍在[comparison_protocol_20261016](evidence/comparison_protocol_20261016/AUDIT.json)，其“全阶段迁移待完成”已由本轮覆盖。**计数更正**：此前preflight实际每配置执行2次optimizer更新，每次累积2个microbatch；此前文字将其写成一次更新不准确。原JSON中的两次loss及测试代码不变，旧证据未重写。

## 版本2→3的比较协议（2026-10-16）

最终规则（用户已选择“和RDD一样”）：**固定数据、预处理、冻结VAE潜空间、DiT主体和优化设置；每张图片覆盖所有训练阶段，各阶段先求 batch mean，再以系数1直接相加，累积后只更新一次参数。保留各方法阶段内部的扩散过程、预测目标及数学权重、条件时间分布、噪声结构、分辨率切换、位置编码和采样算法。主实验保留各自尺度与日程，受控实验另固定可比尺度与名义阶段划分；同时报告质量和实际计算成本。** 这是共同条件下的比较，不等于作者完整系统或最优FID复现。

|统一项|具体约定|
|---|---|
|数据与任务|同训练/验证划分、类别条件、数据暴露量、crop/flip/归一化；同输出RGB尺寸。图像与视频分别成组|
|表示与VAE|同一冻结VAE及权重hash、编码精度、posterior取样策略、latent缩放和解码；同一图片访问得到的latent供所有阶段使用|
|DiT主体|同depth/width/heads/patch/MLP/常规attention和归一化、参数共享与初始化；方法PE及必要多band读写/预测头为显式例外，额外参数和计算单列|
|训练估计器|同一global batch的每张图覆盖所有**训练阶段/活动模式**；每阶段batch mean直接求和，不除以阶段数、不乘原阶段抽样概率；microbatch累积完再optimizer/EMA更新一次|
|优化与数值|同optimizer、学习率配方、有效batch、EMA、精度、TF32、clip和训练预算；改变microbatch需披露RNG/浮点归约变化；共同超参是实验约定，不冒称每方法最优|
|评估与调参|同评估样本数/类别分布/reference/指标实现，预先固定checkpoint选择；CFG采用共同主工作点并允许同预算验证集搜索的附表；不得用测试FID选超参|
|预算记录|固定数据暴露量不等于固定FLOPs；记录每尺度实际模型调用、CFG双分支、自适应solver NFE、训练计算/时间和解码成本|

|保留项|边界|
|---|---|
|扩散/flow过程|信号与噪声日程、时间映射、预条件化、噪声协方差和必要跨band/跨阶段耦合|
|目标与损失|epsilon/x0/v/drift/joint/decoupled、方差头、阶段内部SNR/时间/频带权重、目标单位与内部归约；外层阶段系数统一为1，不能把内部目标一律换成plain MSE|
|训练时间分布|保留uniform/lognormal等阶段内条件分布和必要联合时钟；原阶段抽样概率不再作为外层loss权重|
|分辨率切换|各方法的投影、上下采样、状态重标定、再加噪与噪声相关性；latent化要求的形状适配逐项记录|
|位置编码|保留选定作者配置的空间PE规则，重建当前token网格上的编码；无法直接移植的轴宽/坐标映射必须显式列出，不只是保留RoPE这个名字|
|采样算法|保留solver、更新式、边界及容差；采样步数/容差作为同预算调参项，实际NFE单列；可另做同solver控制，不能默默替换主表solver|
|主实验尺度与阶段日程|保留相对尺度、阶段数及方法clock；受控表另固定可比的网格与名义区间，不能声称同标称t就是同SNR|

**阶段loss权重的最终规则**：与RDD原版相同，写成

\[
\widehat L_m=\sum_{s=1}^{K_m}\left[\frac1B\sum_{i=1}^B
\ell_{m,s}(\theta;z_i,\xi_{i,s})\right],\qquad
\xi_{i,s}\sim P_m(\cdot\mid s).
\]

这里所有外层系数均为1。RDD原三个stage mean相加保持不变；PixelFlow四阶段从平均改为求和；DFM从原随机阶段混合的0.9/0.1改为每次两模式1/1。f-DM也不再用区间长度作外层权重。**这是明确且有意改变作者外层训练目标，不是等期望重写，也不是仅仅取消packing。** SNR、时间、频带、辅助预测头等定义单阶段目标的内部权重仍保留。多阶段数会增加loss总量和计算量，同学习率下不保证等优化强度；须披露阶段数与成本，不能宣称完全消除这些差异。

这里的s是**训练阶段/活动模式**，不总等于一个分辨率band。DFM每个活动模式含自己的drop mask、联合时钟和多个有效band loss；LapFlow是活动后缀；同一个band可以在多个模式中获得监督，不能把未激活band强行加入loss，也不能简单把每个band只算一次。Packing只是执行布局：同一批图片、时间、noise和权重下不得改变样本间隔离或loss归约，也不得抹掉联合band方法必需的注意力。

**噪声政策与表示边界**：本次只统一外层loss，不额外改变现有噪声耦合。RDD保留原阶段独立抽样；当前StageProgram方法（如PixelFlow/Pyramid Flow）保留已声明的共享noise pyramid；SSD/DFM/LapFlow每次活动模式重新抽其原生噪声，模式内部相关性不变。这是版本3的明确选择，不冒称原版的一图一阶段具有相同跨阶段联合分布。采样clip/threshold依赖RGB范围时不能机械移到latent；学习到的upsampler、必需teacher/GAN/专用网络若无法容纳共同主体，应单列或排除，不删后仍称完整原方法。

**实施范围**：当前`configs/comparison_imagenet256`的12份main与6份controlled均已接入上述规则（普通diffusion/flow为单阶段）。14份非RDD多阶段配置改变，RDD两份及普通两份配置不变。SSD按相同分辨率的时间集合分层、保留全部1000个时间索引及切换点；DFM/LapFlow固定枚举原活动模式，保留条件时间和mask。原生版本2配置没有批量迁移；生产默认行为仍兼容旧配方。证据见[本轮检查](evidence/all_stage_sum_20261016/REPORT_ZH.md)。

### 已接入的空间位置编码

通过`method.model.args.position`配置（`comparison_dit`）；原版复现配置和原作者源码不改。

|方法|版本3当前PE|明确的映射/限制|
|---|---|---|
|RDD|原RDD centralized 2D RoPE|模型及main/controlled JSON保持原字节|
|PixelFlow|`pixelflow_rope`，作者Diffusers0.32 NumPy网格2D RoPE|整数token坐标；使用当前latent patch2网格，不把它映射到原RGB patch4像素坐标|
|DFM|`dfm_sincos`，公开SiT固定2D sin-cos|应用于各个共同token网格；作者将各band投影到同网格再融合，当前联合tokens结构不同，不能声称原token对应关系不变|
|LapFlow|`lapflow_sincos`，每band固定2D sin-cos|对应所选公开配置use_rope=False；未把可选RoPE消融当作默认|
|Pyramid Flow|`pyramid_flow_rope`，作者FLUX三轴RoPE的T=1形式|无AR历史时每阶段取本地整数坐标；原head64的轴宽16/24/24改为共同head72的16/28/28，是明确的维度适配|
|PyramidalWan path|`wan_rope`，官方Wan三轴RoPE的T=1形式|轴宽按Wan规则随head_dim分配；无PW作者代码认证，只能称Wan骨干规则移植|
|f-DM、SSD、Non-Uniform|共同RDD RoPE，显式fallback|原U-Net的卷积空间归纳偏置不能独立搬成DiT PE；不能宣称保留了作者DiT位置编码|
|普通DDPM/flow基线|共同RDD RoPE|没有要复现的特定作者PE；作为同主体基线|

所以准确名称是“**共同DiT主体，允许方法特定位置编码和必要接口**”，不是“所有方法网络计算图完全一致”。如需进一步隔离位置编码效应，可另加共同PE消融。

**PixelFlow packing说明**：官方读入的图片先统一crop为256×256，然后每张图只随机分配一个阶段。例如同一batch的A/B/C/D分别下采样为32/64/128/256；patch4后各有64/256/1024/4096个tokens，拼接并记录各序列边界。不同图片不互相attention，每张图仍先计算自己的mean loss。因此“每图一个阶段”与“一个batch含多种分辨率”并不矛盾；不要求原数据集预先提供这些不同size。全阶段方案则让A/B/C/D各自都产生四个训练阶段样本。

**2026-10-15 最新：完整采样已作CPU原仓库对照。** 当前18份共同配置、两组随机种子/batch及额外PixelFlow Euler共38项通过，64,096对张量逐字节相同；另21项原生采样任务通过。检查未裁剪输出、模型调用、阶段轨迹、随机流和随机VAE浮点像素，详细范围逐方法列在[采样报告](evidence/sampling_parity_20261015/REPORT_ZH.md)、[机器证据](evidence/sampling_parity_20261015/AUDIT.json)。Pyramid Flow需明确PSD兼容修复；f-DM/PyramidalWan/Non-Uniform只有独立公式对照，不能算作者代码parity。原生RMD来源归属限定保留。已修PF时间表舍入、SSD随机流、非连续视频tensor噪声布局，补共同PixelFlow Dopri5/CFG；178项回归通过。RDD配置和原源码未改，无GPU/正式训练/模型数据下载。

**2026-10-14 历史：DFM 主配置恢复官方 lognormal 时间策略**：当时使用作者阶段抽样、较早 band 的 shifted-logit-normal 时间与 debug RNG 消耗，REPA 保持官方默认关闭。作者原 loss 与同一个缩小共同 DiT 的64次连续CPU更新逐字节相同。[历史审计](evidence/training_policy_audit_20261014/REPORT_ZH.md)、[证据](evidence/training_policy_audit_20261014/AUDIT.json)。现在阶段抽样已改为全阶段求和，条件lognormal仍保留；作者单阶段组件对照与全阶段外层对照分开认证。

**表示空间与公平性**：18份同DiT配置声明同一个SD-VAE-ft-ema，但尚未绑定/核验真实权重hash。它们比较固定表示下的数学过程，不保证保留原方法质量排名。当前外层估计已按本节统一为RDD式全阶段求和；内部loss、条件时钟与噪声仍有方法差异。pixel→latent、LapFlow的EQVAE→SD-VAE也不是无影响的替换。[此前表示复核](EXPERIMENT_SCOPE_CLARIFICATIONS_20261012_ZH.md#2026-10-14-表示空间codec与公平性复核)的旧阶段抽样状态以本节最终规则为准。未启动正式训练/GPU，也未下载模型或数据。

**2026-10-13 当前实验范围**：按用户要求移出PDDPM、Relay、MrFlow、Bottleneck；完整Edify级联也不在单过程主表。源码/历史parity仍保留，不等于当前实验推荐。阶段配置和原论文／本地默认差异见[最新范围与时钟核查](EXPERIMENT_SCOPE_CLARIFICATIONS_20261012_ZH.md)。旧全部方法复现脚本不能当作筛选后的主实验清单。

统一多分辨率生成的表示、阶段内路径、噪声耦合、预测目标、损失、尺度边界与采样接口，并提供公共训练工程。保留作者网络和明确列出的 solver backend；对照中分别执行作者目标程序与 unified 目标程序。支持普通 diffusion/flow、公开多分辨率方法，以及本地 RDD。

<!-- SUPPORTED_SCOPE_20261012 -->
**2026-10-12作者架构范围的验收**：76项固定代码CPU任务及补充回读全部通过，16项方法/组件逐一列在[历史固定代码报告](evidence/unified_scope_20261012/REPORT_ZH.md)。12项训练核心的已测forward/backward逐字节一致；3项仅推理；CAT仅codec及诊断梯度。这份证据不自动覆盖10月13日新增的共同骨干与视频移植，新增范围和证据见本文末尾“可选模型与图像／视频互换”。
<!-- /SUPPORTED_SCOPE_20261012 -->

| 想了解什么 | 入口 |
|---|---|
| 统一什么、保留哪些原版细节、哪些差异是有意的 | [支持范围与对照约定](SUPPORTED_REPRODUCTIONS_ZH.md) |
| 每个方法测了什么、实际 forward/backward 结果 | [当前结果表](evidence/unified_scope_20261012/REPORT_ZH.md)、[机器报告](evidence/unified_scope_20261012/AUDIT.json) |
| 数学与可扩展组件 | [数学框架](FRAMEWORK_MATH_ZH.md)、[组件映射](PAPER_COVERAGE.json) |
| 来源、版本与复用边界 | [源码和许可证](SOURCES_AND_LICENSES.md)、[机器合同](SUPPORTED_REPRODUCTIONS.json) |
| 公共实验设置和原版配方的区别 | [成对实验入口说明](PAIRED_EXPERIMENTS_ZH.md) |
| 历史发现和当时的未完成项 | [历史 README](README_HISTORY_20261012_ZH.md)、[旧严格审计](ORIGINAL_UNIFIED_GAPS_TODO_ZH.md) |

当前只用 CPU，禁止隐式下载模型或数据，不修改共享 FastVideo 环境。小模型/合成输入的逐字节一致是实现证据，完整发布模型也另有检查；这些证据不代替正式训练的 FID/IS/VBench。所固定的SPEED/NAMI/MrFlow公开代码提供推理；SPEED论文另有微调目标，本项目不能将其称为没有训练方法。CAT只提供已发布codec的数值及诊断梯度对照。

组件改组是新的受控实验，不继承原版配置的数值证书。共同 batch、optimizer 或预算也不能被叫作论文最佳训练配方。早先 Pyramid Flow/PyramidalWan 的 FastVideo 工程另行验收，不继承本矩阵结果。

测试结束后，临时 checkpoint、生成媒体和差异 Tensor 在回读核验、保存哈希后删除；报告、日志和配置保留。正式训练资产及用户数据不按临时产物规则删除。

相关分析：[方法等价与 RDD 设计](METHOD_EQUIVALENCE_AND_DESIGN_20261011_ZH.md)、[论文定位](PAPER_POSITIONING_REVIEW_20261011_ZH.md)、[已有统一框架](RELATED_UNIFIED_FRAMEWORKS_20261010_ZH.md)。

## ImageNet-256 同 DiT 主实验与受控实验（2026-10-13）

本轮当前实验清单是 [configs/comparison_imagenet256/manifest.json](configs/comparison_imagenet256/manifest.json)，共 **12 份 main、6 份 controlled**。主实验涵盖9种多尺度方法、f-DM的一个额外日程候选，以及普通DDPM/flow两条基线。每份JSON完整展开，训练不依赖隐含的父配置。作者架构下的历史parity清单不等同此处共同DiT实验；本轮保留数学过程，主动替换非RDD方法的网络、表示和公共训练配方。

### 配置设计与统一项

|字段|控制内容|
|---|---|
|`method.model`|骨干、位置编码、参数共享、必要的多band接口与预测头|
|`method`其余部分|表示/尺度、阶段路径与时钟、噪声与预测目标、loss、阶段边界、solver；使用原有组件registry|
|`method.training`|数据预处理、VAE、优化器、EMA、条件与时间采样|
|`run`|总更新次数、global/micro batch、随机种子、保存频率；新入口resume以总更新数为终点|
|`assets`|必须由用户提供的本地ImageNet与SD-VAE路径；不下载、不默认换codec|
|`sampling`|采样预算和CFG；`experiment sample`读取generic solver的步数，其余sampler读取`method`内的配置|
|`comparison`|原配方来源、主动改动、比较协议、不能声称的等价性|

共同主体来自**旧RDD的DiTMultiRes-XL/2**：28层、hidden1152、16 heads、patch2，1000类，class dropout0.1，主体674,834,720参数。RDD保留centralized 2D RoPE/base_size16及原网络；2026-10-16起其他方法按上表采用各自PE或明确fallback，不再声称全部位置编码相同。保留相同8通道末层及初始化，其中RDD为effective epsilon+variance，f-DM为epsilon+residual，普通flow/score只读前4通道。各实验重新从相同种子初始化，不是不同方法之间共同训练一份权重。

共同数据/训练：ImageNet原始256 RGB，ADM中心裁剪与随机翻转；本地`stabilityai/sd-vae-ft-ema`，每次访问重新posterior sample并乘0.18215，得到4×32×32 latent；FP32/允许TF32，AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd0、无LR衰减/梯度裁剪，EMA0.9999，global batch256，400,000 updates，seed0，每5000步覆盖保存当前checkpoint。默认micro32、累积8，单进程；改变micro会改变随机数消费与求和顺序。没有缓存固定的一次posterior sample来冒充旧RDD配方。

**RDD保留项**：原模型、1000步linear beta、fine→coarse `stage_lengths=[300,300,400]`、三个阶段各自取t/noise、`sum_stage_means`、joint effective-epsilon/learned-variance语义、optimizer/EMA数学超参数。采样以实际旧400K实验为锚，coarse→fine DDPM/jump `[90,80,80]`、fresh boundary、CFG1（另可报告旧CFG1.5）。旧`paired_experiments`模板中的DPM2/100-75-75不是这一锚点。原始证据是`multiscale_diffusion/2d/exp/imagenet256_unifiedXLtrain_r8-400_r16-300_r32-300_400000_sample_r8-80_r16-80_r32-90_ddpm_cfg115.jsonl`。

**RDD执行边界**：历史启动是8 ranks×32，本入口是单进程32×累积8，reader也不同。因此算法/模型/有效batch不变，但不是历史8卡随机轨迹和浮点all-reduce的逐位重放。本轮没有改原RDD训练器、DiT或采样器源码。若需要重放历史进程拓扑，应使用保留的原训练器，不能把本入口称作原8卡CLI。

### 主实验：保留各方法的日程与必要接口

所有网格均为latent边长，按coarse→fine书写。路径位于`configs/comparison_imagenet256/main/`。

|JSON文件|尺度／采样预算|与原版的有意差异|
|---|---|---|
|`rdd_ddpm_xl_legacy__train_r8-400_r16-300_r32-300__sample_r8-90_r16-80_r32-80.json`|8/16/32；90+80+80 DDPM|保留旧RDD算法与网络；上述执行拓扑差异|
|`ddpm_xl_fixed__train_r32-1000__sample_r32-250.json`|32；250 ancestral|普通DDPM基线，使用RDD的RoPE DiT和learned variance|
|`flow_xl_fixed.json`|32；250 Euler|普通线性velocity flow基线，同DiT|
|`fdm_xl_5_cosine_assumption.json`|2/4/8/16/32；250按区间分配|ADM U-Net→DiT、pixel DS→latent DS；论文cosine打印公式端点不一致，本文件明确假设补π/2，不称原作者准确数字|
|`fdm_xl_5_linear.json`|同上；每段50|使用论文允许且无歧义的linear划分；两份f-DM不是两个独立方法；epsilon/residual、VP rescaling、P2和随机边界保留|
|`pyramid_flow_xl_3.json`|8/16/32；20/20/20|视频网络→图像DiT；保留空间flow、相关噪声和作者clock；显式PSD平方根修复奇异协方差；不带时序/AR历史；统一为全尺寸SD latent后pool，而非逐尺度RGB各自VAE编码|
|`pyramidal_wan_path_xl_3.json`|8/16/32；20/20/10，ζ边界1/.9412/.5858/0|Wan预训练视频微调→scratch图像DiT；保留自然时间/噪声变换；不带首帧/时间压缩/DMD；Wan阈值是待调候选，非SD latent最优结论；本来就是paper实现|
|`pixelflow_xl_4.json`|4/8/16/32；Dopri5每阶段30保存点，atol1e-6/rtol.001|RGB patch4 transformer→latent patch2/RoPE DiT；多尺度packing→分阶段minibatch；使用作者Dopri5与跨边界双份CFG状态；共同CFG1；保存点数不是NFE；controlled仍为Euler90/80/80|
|`ssd_xl_4_convex.json`|4/8/16/32；1000 operator-DDPM|Flexi-UNet→DiT+必要输出resize；RGB→latent；关闭[-1,1] clean clipping；保留ConvexDecay.5、linear1000 beta、min-SNR5、算子posterior/Lanczos|
|`dfm_xl_2.json`|16/32；30/10，threshold.1|SiT→共同DiT联合tokens/逐band时间；保留官方条件lognormal、active loss；阶段混合.9/.1→枚举两模式以1/1求和；REPA维持官方默认关；ft-mse→共同ft-ema codec|
|`lapflow_xl_2.json`|16/32；T=.5，Dopri5 atol=rtol=1e-5|MoT experts→共享DiT；因果mask及延迟flow保留；EQVAE→SD-VAE；原两层采样器的多余coarse尾段仍计算且计费|
|`nonuniform_haar_xl_4.json`|4/8/16/32；64×4 EM|4个UNet→共享DiT；pixel128→latent256；保留Haar/逐band VP与unweighted DSM；需12通道Haar读写投影，额外2,768,304参数|

除RDD外还统一了训练数据、class conditioning、optimizer和预算；不能把这张表的结果直接命名为“复现原论文最佳FID”。所有方法外层均采用RDD式阶段mean直接求和；噪声耦合按本文开头的逐方法政策保留，与作者每次只抽单阶段的梯度估计不同。DFM/LapFlow/Haar有联合token、分量时钟或mask，属于**共同主体+必要接口**组，不能称所有网络forward计算图完全相同。

### 受控实验与公平性

`controlled/`保留RDD、f-DM、Pyramid Flow、PyramidalWan path、PixelFlow、SSD共6份：固定latent **8/16/32**及高噪声→低噪声名义区间 **400/300/300**；前五项固定采样 **90/80/80**。边界噪声校正、时间映射、预测目标和loss保持方法各自定义，没有强迫物理SNR相等。RDD与main是同一method，无需训练两遍。

SSD同尺度/名义分区，但仍采样1000步，单列成本；本轮没有实现其严格250步算子respacing，不能改beta表冒充。DFM/LapFlow/Haar不同band时钟不能被这一标量分区表达，未列伪受控配置。

main固定400K×256的**数据暴露预算**，不是已经找到各方法最优超参，也不是等训练FLOPs。未来调参要等搜索预算，在validation选择，最后统一50K RGB、CFG1、同ADM reference与预处理，额外CFG单列。采样步数与分辨率各异，成绩必须同时列实际NFE/分辨率、训练和推理成本，不能只看FID。只数主体已注册矩阵算子，RDD每例三阶段F/B约为单32基线的1.307倍，详见[CPU/meta计数](evidence/main_comparison_20261013/SCALAR_CORE_FLOPS.json)；不是GPU耗时或完整端到端FLOPs。等计算预算最优搜索与完整质量曲线尚未执行。

### 按实验文件运行

```bash
cd /scr/yuheng/bugswarm/rdd_workspace_20261005
source unified_multires/activate_cpu.sh
EXP=unified_multires/configs/comparison_imagenet256/main/rdd_ddpm_xl_legacy__train_r8-400_r16-300_r32-300__sample_r8-90_r16-80_r32-80.json

# 仅检查配置，不加载数据或XL权重。
"$UMDIFF_PYTHON" -m umdiff.experiment plan --config "$EXP"

# 正式训练入口示例；需自行指向已有真实资产，本轮没有执行。
"$UMDIFF_PYTHON" -m umdiff.experiment train --config "$EXP" \
  --data /path/to/imagenet/train --vae /path/to/sd-vae-ft-ema \
  --output unified_multires/runs/rdd_main --device cpu

# 使用同一实验配置的EMA和采样预算，输出latent。
"$UMDIFF_PYTHON" -m umdiff.experiment sample --config "$EXP" \
  --checkpoint unified_multires/runs/rdd_main/checkpoint.pt \
  --count 2 --labels 0 1 --sample-seed 0 \
  --output unified_multires/runs/rdd_main/latents.pt

"$UMDIFF_PYTHON" -m umdiff decode \
  --latents unified_multires/runs/rdd_main/latents.pt \
  --vae /path/to/sd-vae-ft-ema --scale 0.18215 \
  --output unified_multires/runs/rdd_main/rgb.pt
```

也可通过`IMAGENET_TRAIN_DIR`/`SD_VAE_EMA_DIR`绑定资产。`train --resume ...`补足到`run.updates`，不是追加400K。旧裸`umdiff train --steps`仍表示追加步数，二者不要混用。`sample`核对checkpoint中的method，防止用另一方法/日程误采样；普通flow的250步也从文件读取。输出目录不自动覆盖已有训练或采样成果。FID/IS批量评分沿用独立评估流程，本轮没有新建或认证完整50K评估入口。

当前只授权CPU，旧GPU deadline guard未移除；将命令直接改成`--device cuda`仍会被旧到期限制拒绝。本轮没有启动正式训练，也没有下载ImageNet/VAE。CPU入口已测通，完整XL的实际设备显存与吞吐未测；CPU上的400K示例不是运行时长建议。

### 本轮验证与可重跑证据

- [PREFLIGHT.json](evidence/main_comparison_20261013/PREFLIGHT.json)：18份配置在完整32/16/8等网格及原1000步表上，用width32/depth2网络各两次累积更新，再走各自完整采样预算，全部有限且形状正确。它们不是与各作者旧架构逐层相等的声明。
- 同报告中RDD独立原训练目标与unified对照：共同非零随机权重、相同输入/RNG，144项模块输出、144项activation gradient、31项parameter gradient逐字节一致。仅缩小网络，不能外推完整XL所有浮点执行环境。
- [ARCHITECTURE.json](evidence/main_comparison_20261013/ARCHITECTURE.json)：完整XL的18份配置在meta设备构建，共同主体state_dict键/shape与旧RDD相同；不是完整XL数值F/B。
- [pytest.log](evidence/main_comparison_20261013/pytest.log)：公共回归、实际子进程RGB→本地随机VAE→短训→新进程resume→EMA采样；连续与恢复后的模型/optimizer/EMA/RNG及采样相同。真实预训练资产质量未测。
- 初次RDD测试夹具误用`avg_pool2d`而非原`.view(...).mean((3,5))`，出现5.96e-8差异；已改用AST提取原函数。保留`PREFLIGHT_attempt1.json`与`attempt1.log`，没有改生产RDD来掩盖误差。

重跑：`tools/build_main_comparison.py`生成配置；`tools/validate_main_comparison.py`做CPU短测与RDD对照；`tools/profile_comparison_core.py`做完整结构和限定FLOPs计数；`python -m pytest -q tests`做公共回归。均需上述离线环境。

## 直接运行

在当前 surfi1 工作区使用已有 Python，不修改共享 FastVideo 环境：

```bash
cd /scr/yuheng/bugswarm/rdd_workspace_20261005
source unified_multires/activate_cpu.sh

"$UMDIFF_PYTHON" -m umdiff.cli components

"$UMDIFF_PYTHON" -m umdiff.cli train \
  --config unified_multires/configs/native_fixtures/rdd_joint.json \
  --synthetic --steps 20 --batch-size 8 \
  --output unified_multires/runs/my_rdd

"$UMDIFF_PYTHON" -m umdiff.cli sample \
  --checkpoint unified_multires/runs/my_rdd/checkpoint.pt \
  --labels 0 1 --count 2 \
  --output unified_multires/runs/my_rdd/sample.pt
```

四通道 RDD/DiT 输入是已按目标约定缩放的 latent；上述输出也是 latent。CLI 不会偷偷用另一个 VAE 解码，三通道图像输出才额外保存 PNG。普通像素 flow 可将配置换为 `configs/controlled/ordinary_flow.json`，普通 DDPM 用 `ordinary_ddpm.json`。

断点继续时 `--steps` 表示**额外更新次数**：

```bash
"$UMDIFF_PYTHON" -m umdiff.cli train \
  --config unified_multires/configs/native_fixtures/rdd_joint.json \
  --synthetic --steps 20 --batch-size 8 \
  --resume unified_multires/runs/my_rdd/checkpoint.pt \
  --output unified_multires/runs/my_rdd
```

恢复需使用同一个配置、数据、batch size 和 seed。checkpoint 保存模型、EMA、optimizer、步数、已见图像数、scaler、generator及Python/NumPy/Torch RNG。按checkpoint step决定数据遍历位置；恢复不会重跑第一个batch。新版CLI核对数据path/文件size/mtime、manifest、batch size和seed；不是整个数据集内容hash。历史无data_binding的checkpoint仍可用Trainer/native后端读取，但不能直接通过新版绑定CLI恢复。

## 已验证的配置

`native_fixtures` 是作者实际网络类的缩小实例。它们验证算法与代码互操作，参数量、训练预算、数据及 teacher 不等同论文实验。

|方法|配置 / 入口|本轮 native 数值覆盖|
|---|---|---|
|RDD / 普通 VP|`rdd_joint*`、`rdd_decoupled*`|drift 开/关，fixed/learned-range variance，joint/decoupled；loss、梯度、AdamW、EMA、完整 DDPM|
|Relay|`relay.json`|EDM/heat path、block noise、原版 DhariwalU-Net、stochastic blur sampler|
|Scale Space Diffusion|`ssd.json`|FlexiUNet、双线性算子/adjoint、anisotropic posterior/Lanczos、完整采样|
|PixelFlow|`pixelflow.json`|异形 packed training、原版 DiT、局部/模型时间、相关补噪、CFG 0/4 完整采样|
|DFM|`dfm.json`、`dfm_repa.json`|实际 released SiT、Laplacian/product state、active loss、REPA projector、两尺度采样|
|LapFlow|`lapflow_2.json`、`lapflow_3.json`|2/3 尺度组件；2 尺度实际 MoT 完整训练/采样；3 尺度 CLI 训练/采样|
|PPFlow|`ppflow_2.json`、`ppflow_3.json`|实际 DiT、2/3 stage packing、可逆布局、全轨迹采样|
|DCTdiff|`dctdiff.json`|作者 U-ViT、VP/SNR、DCT/YCbCr/chroma codec、DPM-Solver、decoded uint8 pixels|
|MDM|`mdm.json`|实际 NestedUNet、多尺度 v→epsilon loss、SNR shift、posterior、每步输出|
|DiMR|`dimr.json`|实际 Transformer/ConvNeXt/TDRMSN、多分辨率 target/loss、DPM 最终 latent|
|Diffusability|`diffusability.json`|实际 AutoencoderKL wrapper、随机 latent/KL、pixel MSE、mean/bilinear scale-equivariance、训练后 decoder|
|RMD|`rmd.json`|实际 WanTransformer、flow shift 0/1/2、噪声混合、fake score→generator 顺序更新；两个 transition 各 128 步、训练后 Wan VAE pixels；原版 train() 另测 2 步|
|SPEED|`configs/inference_only/speed.json`|原版完整 Flux generation；DCT/Haar/FFT 三种谱扩张，bf16、decoder pixels|
|NAMI|`configs/inference_only/nami.json`|作者 vendored Flux、progressive depth、learned bridge、scheduler、CFG 1/3、decoder pixels|
|MrFlow|`configs/inference_only/mrflow.json`|原版 main；Flux→VAE→PIL→真正 RRDB SR→VAE→sigma .12 refine；多 tile、非方形输入|

SPEED/NAMI/MrFlow 的公开仓库提供推理方法；不伪造作者没有发布的训练 objective。它们支持相应预训练 backbone 的本地权重，以及在通用框架中独立训练 backbone。

f-DM、Edify、PDDPM、DDiT、LoT 尚未确认可执行官方实现；Fresco、AViTS 所下载官方仓库只有 README/许可证或代码待发布。这七篇保留文献与接口映射，按用户要求跳过 native 对照。CAT 已找到官方补充中的 codec 源码，单列组件对照；没有将未发布的训练目标补造为作者实现。`fdm_decoupled_clean.json` 是明确标注的组件移植，不能用来宣称复现了闭源 f-DM。

## 使用已有本地数据

像素方法可直接使用 `class_name/*.png` 或普通图片目录：

```bash
"$UMDIFF_PYTHON" -m umdiff.cli train \
  --config unified_multires/configs/controlled/ordinary_flow.json \
  --data /absolute/path/to/local_images --steps 1000 --batch-size 8 \
  --output unified_multires/runs/local_pixels
```

图片经 RGB、中心正方形裁剪、BICUBIC resize，归一化到 `[-1,1]`。类别按父目录名排序映射；平铺目录得到单一类别。它不是每篇论文原版 augmentation。做论文质量复现时要匹配作者预处理，不能只复用这个方便的 loader。

本地 `.pt` 支持一个 float BCHW tensor，或以下字典，文件需能装入主存：

```python
torch.save({
    "latents": latents,             # 或 images，二者只能选一个
    "labels": labels,               # 可选；[N]
    "conditions": {                # 可选；每个字段首维必须为 N
        "text": text_embeddings,
        "mask": attention_mask,
        "teacher_features": teacher_features,
    },
}, "/absolute/path/to/local_latents.pt")
```

latent 的 codec、scale/shift 和尺寸必须与方法一致，CLI 会检查 shape，不隐式 padding 或重新编码。class labels 通过 `training.conditioning` 配置传给模型。MDM 默认的零文本只是 fixture；实际条件训练改为：

```json
{
  "conditioning": {"kind": "field", "field": "text"},
  "cond_mask": {"kind": "field", "field": "mask"},
  "micros": {"kind": "empty_dict"}
}
```

把这个对象放入 `training.conditioning`。DFM REPA 的 teacher 可用 `teacher_features` 字段，配置 `destination: "context", phase: "train"`；它必须与本地 teacher 的 token/grid 顺序一致。框架不下载 DINO/LPIPS。

条件采样使用单独的本地 tensor 字典，字段首维等于 `--count`：

```bash
"$UMDIFF_PYTHON" -m umdiff.cli sample \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --conditioning /absolute/path/to/conditions.pt --count 2 \
  --output unified_multires/runs/conditional.pt
```

## RMD 的多模型蒸馏入口

RMD 使用 BCTHW latent，三个网络分别是 generator、fake score 和冻结 real teacher；不是普通单网络 MSE。配置 `task: "rmd"` 运行两套 AdamW，先更新 fake score，再更新 generator。原版默认 constant LR；本入口不伪装支持其他 scheduler 或未验证的 GAN 分支。RMD checkpoint 保存三个网络和两套优化器，无 EMA。

```bash
"$UMDIFF_PYTHON" -m umdiff.cli train \
  --config unified_multires/configs/native_fixtures/rmd.json \
  --synthetic --steps 4 --batch-size 2 \
  --output unified_multires/runs/my_rmd
"$UMDIFF_PYTHON" -m umdiff.cli sample \
  --checkpoint unified_multires/runs/my_rmd/checkpoint.pt \
  --count 2 --output unified_multires/runs/my_rmd/sample.pt
```

正式蒸馏需要本地预训练模型。复制配置并匹配完整 Wan 架构，设置 `checkpoints: {"model": "/path/generator.pt", "fake": "/path/fake.pt", "real": "/path/teacher.pt"}`；三项必须同时给出，均为严格加载的 raw state dict，支持 `.pt` / `.safetensors`。没有 checkpoints 的配置明确是随机小模型 fixture。`--data` 输入为本地 `.pt` 字典，且只含 `prompt_embeds: [N,L,D]` 与 `negative_prompt_embeds: [1,L,D]`。采样的 `--conditioning` 字典使用 `prompt_embeds`；省略时复用 checkpoint 最后一个训练 batch 的 embedding。输出为 latent，不隐式下载或构建 Wan VAE。

RMD 仓库自称 official，metadata 作者名与论文一致，但论文页未给出仓库链接，因此来源身份尚未独立确认。数值结论严格限定固定 commit 的公开实现。原版 BCTHW loss 的 normalization 只 reduce `[1,2,3]`、保留 width；phase1 总是使用 higher-shift 的细节均保留，没有悄悄修正。

## inference-only 方法

先跑明确标记的随机小模型 fixture：

```bash
"$UMDIFF_PYTHON" -m umdiff.cli infer \
  --config unified_multires/configs/inference_only/speed.json \
  --synthetic-embeddings --seed 42 \
  --output unified_multires/runs/speed.pt
```

将 `speed` 换成 `nami` 或 `mrflow` 可运行另外两个。输出同时保存中间/最终 tensor 和 PNG。NAMI 保留作者 32→64→128 的 latent geometry，尽管 Transformer width/depth 缩小，最终 decode 仍为 1024²。

实际本地资产运行需复制配置、将模型/codec 架构改成权重对应的完整参数，设 `synthetic_fixture: false`，并填写 `checkpoints.model / vae / sr_model / bridge` 所需本地路径。模块逐项 `strict=True` 加载；缺一个必需权重就报错，不会用随机模型补位。state dict 支持 `.pt`，普通模型支持 `.safetensors`。NAMI bridge 当前使用原版 `.pt` state dict。仓库不同版本的 key/architecture 不会自动猜测转换。

```bash
"$UMDIFF_PYTHON" -m umdiff.cli infer \
  --config /absolute/path/to/local_workflow.json \
  --embeddings /absolute/path/to/prompt_embeddings.pt \
  --seed 42 --output unified_multires/runs/local_inference.pt
```

embedding 字典至少含 `prompt_embeds`、`pooled_prompt_embeds`；NAMI CFG>1 另需 negative 两项。文本编码器不在这个入口内隐式构造。本地权重/embedding/config 加载通过随机夹具验证；临时权重会在验收后清理，不能把历史 `runs/offline_entrypoints/` 当成可下载的正式模型。新增分支的当前CLI证据在 `evidence/unified_scope_20261012/workflow_cli/`。

NAMI vendored Diffusers 与其他版本有同名 Python package。**每个 native 方法用独立进程运行。** namespace 冲突会报错。

## 组合或添加组件

通用引擎的核心配置示例：

```json
{
  "name": "my_component_mix",
  "image_shape": [3, 32, 32],
  "model": {"name": "tiny_conv", "args": {"channels": 3, "width": 32}},
  "path": {"name": "rdd_flow", "args": {"start": 0.0, "end": 1.0}},
  "noise": {"name": "iid"},
  "target": {"name": "joint", "args": {"quantity": "velocity"}},
  "clock": {"name": "linear"},
  "solver": {"name": "euler"}
}
```

切换为 `decoupled_clean` 时，模型也应配置 `heads: ["signal", "residual"]`；可在 `metric.args.head_weights` 指定不同 head 权重。不同预测目标的输出通道、转换和奇点需要一起匹配。没有通用规则保证任意不相容组件组合可训练。

新插件无需改 trainer：

```json
"path": {
  "name": "smooth_detail",
  "module": "examples.custom_components"
}
```

`examples/custom_components.py` 展示了自定义 band signal path。`register` 可以扩展 representation/path/noise/target/metric/clock/boundary/model/batch/sampler/regularizer。需优化的 learned 参数归属 `model`；inference boundary 的 learned 参数使用显式 checkpoint。模型工厂可用 `python_factory` 从用户模块创建。

`Method.sample_stages` 接受各阶段 clock/path/boundary 和显式边界上下文，可以换 fresh、isometric、spectral、posterior 等 kernel。原版特殊方法使用独立 `batch/sampler`，保证 packing、RNG 和 reduction；对这些 native 策略随意覆盖通用 target/path 不会生效，因此实现会明确拒绝此类配置。做魔改时应使用通用引擎或注册一对明确的新策略。

固定作者 sampler 的步数在 `sampler.args` 中设置；普通通用 solver 才使用 CLI `sample --steps`。改变训练路径或模型后应重新训练，不能靠加载不兼容 checkpoint 比较效果。

## 重新执行验证

```bash
# --plan只查看任务；执行时需要本地worker/Gloo进程IPC权限。
bash unified_multires/run_supported_reproduction_checks.sh --plan
bash unified_multires/run_supported_reproduction_checks.sh --workers 2

# 单项诊断保留payload；不要在同一证据目录有任务运行时改被测源码。
bash unified_multires/run_scoped_cpu_validation.sh --only native_ssd native_ppflow2 native_ppflow3
```

完整入口运行76个固定代码任务及显式的Dopri5、RDD采样器Gloo补验，再独立回读checkpoint状态、模块字节统计和源码哈希，生成逐方法报告，最后清理临时payload。不同运行保留各自日志和代码快照；报告只接受与当前被测源码匹配的结果。全部使用CPU、随机非零网络和明确的合成/本地夹具，不调用正式长训练。

常规训练对照执行作者loss、模块方法或原始训练语句体，共用作者backbone；SSD `TrainLoop`、PPFlow两/三层 `main` 另外有实际入口检查。CPU适配和缩小模型清楚列在报告中，不称作所有官方CLI原样执行。

连续随机序列也要对齐：DFM 默认兼容配置保留作者额外的 NumPy debug 随机数消耗，MDM保留原累积归约次序，PPFlow保留VAE微批次posterior抽样顺序。改变这些配置后应重新验证。

PixelFlow 两侧使用同一CPU SDPA；未认证原CUDA FlashAttention。DCTdiff/DiMR复用作者DPM-Solver backend，PPFlow原solver和LapFlow使用的torchdiffeq也明确复用。这不削弱目标/阶段组装的对照，但不能称为独立重写这些solver。

## 依赖、授权和论文实验范围

当前可执行环境和依赖版本见 `evidence/unified_scope_20261012/environment.json`；安装信息见 `requirements-authors.txt`。基础框架依赖 torch/numpy/scipy/Pillow，作者网络额外依赖 diffusers/transformers/timm/einops 等。`_deps/` 只放本次隔离安装的少量 Python 依赖，没有改共享环境。`sources/` 保留各作者许可证，详见 [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md)。

全部验证强制 CPU/offline。用户最新授权为只用CPU；历史GPU窗口已结束，不能自动恢复卡号、deadline guard或旧监控。本轮无GPU或正式长训练，未修改共享dirty FastVideo。

本轮交付基础训练/采样框架与数值复现。仍需用户已有的真实数据、匹配的完整模型规模/训练预算/codec/teacher/评测资产，才能检验论文 FID 和“至少与论文一样好”。`PAPER_COVERAGE.json`、报告及 checkpoint metadata 都保留这一范围，不把小模型或合成像素的相同说成真实生成质量已经认证。

## 2026-10-13：可选模型与图像／视频互换

入口是 `umdiff configure`，实现位于 [cross_modal.py](umdiff/cross_modal.py)。数学过程、模型、数据网格现在可以分别配置。41份可执行小配置见 [configs/cross_modal](configs/cross_modal/manifest.json)，是共同模型实验的起点，**不是作者最佳配方**；默认宽度96、深度2。

|配置方法|保留的数学过程|这条共同模型路线的明确边界|
|---|---|---|
|`fdm`|f-DM-DS信号变换、SP/VP尺度归一化、signal/residual重建目标、双头、DDIM/DDPM与条件噪声lift|论文重实现，无可执行作者训练库；默认bilinear信号上采样、average条件噪声；t=1截至.9999；边界用完成最后一个阶段步后的clean/noise重组，是明确的离散化选择|
|`pyramid_flow`|作者空间端点、1000步分段网络时钟、共享噪声、Euler阶段与PSD边界|空间路径可用于图像或整段视频；不包含自回归历史帧条件；默认latent pooling是受控选择，也支持分别VAE编码的stage tensors；PSD与原Cholesky不要求同一随机基|
|`pyramidal_wan_path`|natural/global非线性时钟、三阶段端点与速度、首帧/后续帧不同omega、Haar边界|独立paper实现；此配置是path，不包含teacher/DMD/步数蒸馏，也不声称作者源码parity；相关旧teacher路线仍保留|
|`edify_laplacian`|dimension-varying EDM、频带衰减、EDM预条件、正交高频噪声填充|论文数学族中的明确实例；alpha在每阶段sigma区间线性由0到1，训练sigma按rho网格分位取样；不冒称作者未披露衰减函数或完整Edify级联|
|`nonuniform_haar`|Haar联合状态、不同频带扩散截止时间、VP log-SNR、likelihood加权DSM、reverse SDE/ODE|必须联合读写活动频带；共享权重是新选择，独立stage模型也支持；有限SNR末端按论文思路近似纯噪声，无额外Tweedie去噪|
|`pddpm`|完整低分辨率DDPM、resize、前向再加噪、短高分辨率逆过程|论文过程重实现；训练beta表与推理预算分开，少步使用明确的respaced祖先后验；配置中restart_steps表示原训练表的重启索引长度|
|`rdd`|原有RDD-FM path、joint目标或effective-epsilon/raw-drift双头、fresh/coupled边界|新共同网格执行程序；图像路径推广至视频；不替换历史RDD-DDPM native实验|
|`pixelflow`|作者端点、时钟及带jitter的block-noise边界|保留数学；去掉packed作者模型，默认每stage完整minibatch；跨stage训练噪声来自同一个fine draw，其边缘分布相同、跨loss相关性与独立draw不同|
|`relay`|低阶段EDM训练/生成→空间repeat→高阶段blur diffusion训练/采样|完整两阶段接力；低分辨率训练输入用同batch平均池化，模型可共享/分开；未接原增强条件与独立数据训练预算|
|`ssd`|原退化路径、loss、算子后验与Lanczos采样|Flexi-UNet替换为显式输出grid适配；只保留空间缩放，T不变|
|`dfm`|Laplacian多分量、分量时钟、stage loss与采样|共同联合DiT，REPA关闭；不是多个独立网络逐band调用|
|`lapflow`|Laplacian分量、延迟激活、loss/采样、因果scale mask|shared DiT替换MoT；fine-first活动后缀的stage身份保持一致|
|`mdm`|联合extended-space多尺度状态、尺度SNR及目标/后验|保留联合状态，去掉NestedUNet；不称完整MDM架构复现；默认不启mixed-batch或私有课程|
|`dctdiff`|原RGB→YCbCr/DCT→epsilon目标→原DPM-Solver→RGB|模型由UViT改成选定骨干；视频codec逐帧，生成器跨帧联合；RGB DCT并未被暗换成latent DCT|
|`flow` / `ddpm`|普通flow / 单尺度DDPM|共同骨干基线；DDPM默认1000步训练表，采样可独立respacing|

Pyramid Flow、PixelFlow和PDDPM等默认对每个stage完整训练minibatch，stage loss取平均；可配置`stage_weights`。这与作者packed/interleaved batch的每次更新成本不同。比较训练效果时需报告处理样本数、每stage token数及实际FLOPs，不能只固定optimizer更新次数就叫等计算预算。

`flow`可另选`--inference-policy speed`或`bottleneck`，训练目标仍是普通flow。SPEED复用已验证的DCT/DWT/FFT扩张与时间对齐，但新默认边界日程不是作者VAE拟合谱的最优日程，也不是论文微调。Bottleneck实现高→低→高、再加噪和scheduler shift，默认浮点Lanczos-3；论文式(7)附近文字与算法第12行对保留步数写法不一致，本实现以显式`start_indices`选择前者，不声称恢复了未公开代码的约定。详见 [论文式(7)–(9)](https://arxiv.org/html/2503.18940)。两者都需要实测已训练模型的质量收益。

### 模型选择和数据布局

|`--backbone`|实现与可配项|限制|
|---|---|---|
|`grid_dit`|联合时空attention；width/depth/heads/patch；轴分离sin-cos或3D RoPE；stage/多band时钟；label/text条件；梯度checkpoint|这是共同实验模型；不是改名后的原论文作者网络；支持全部上表数学路线|
|`dit_grid`|迁移RDD原DiT的实际Transformer blocks；原图像网格保留原forward；新空间大小重建PE；视频加入时间PE并联合所有帧token|支持单张量路线；只有B时钟，拒绝PyramidalWan不相同的首帧/尾帧时钟；不伪造联合band接口|
|`wan_grid`|真正的Diffusers Wan Transformer构造，保留自身RoPE与文本cross-attention；支持B或B,T时钟|必须显式提供本地text embeddings；默认是小架构构造示例；没有下载、默认预训练权重或隐式text encoder/VAE|
|`grid_conv`|可配置宽度的Conv3d共同小骨干，图像视为T=1|用于调试单张量数学路线，非论文架构；不支持联合band输入|

DFM、LapFlow、Non-Uniform、MDM需要product-state读写，当前用`grid_dit`，或用`--model-json`提供自己注册的联合模型。程序明确拒绝用普通Wan/原DiT独立调用每个band来冒充联合模型。`dit_grid`/`wan_grid`的位置编码由原架构决定；`--position`是`grid_dit`的选择。普通tensor方法也可通过自定义模型注册扩展；接口为`model(state, time, stage=..., **conditioning)`。

`image_shape`历史字段现在表示`[C,H,W]`或`[C,T,H,W]`。视频主干始终联合处理帧，只有空间resize、DCT/颜色codec逐帧；有单帧精确等价及跨帧非零梯度测试。空间金字塔保持T不变；时空金字塔显式写`grids`，首帧保护显式写`first_frame=true`，不自动猜padding/cropping。所有stage网格必须整除patch；保留首帧的时钟配置建议temporal patch=1。

### 使用示例

```bash
cd /scr/yuheng/bugswarm/rdd_workspace_20261005
source unified_multires/activate_cpu.sh

# 图像方法f-DM，使用联合时空DiT训练视频；仅生成配置。
"$UMDIFF_PYTHON" -m umdiff configure \
  --method fdm --shape 4,9,32,32 --backbone grid_dit \
  --width 192 --depth 4 --attention-heads 6 --position rope \
  --output unified_multires/configs/my_fdm_video.json

# 视频论文的数学路径用于图像，保留三阶段自然/全局时钟。
"$UMDIFF_PYTHON" -m umdiff configure \
  --method pyramidal_wan_path --shape 4,32,32 --backbone dit_grid \
  --output unified_multires/configs/my_pwan_image.json

# 显式首帧保护的真正时空网格，不只是视频逐帧resize。
"$UMDIFF_PYTHON" -m umdiff configure \
  --method pyramidal_wan_path --shape 4,9,32,32 \
  --grids '[[3,8,8],[5,16,16],[9,32,32]]' --first-frame \
  --output unified_multires/configs/my_pwan_spacetime.json

# CPU短检验。正式输入换成 --data /path/to/local_tensor_or_shards。
"$UMDIFF_PYTHON" -m umdiff train \
  --config unified_multires/configs/cross_modal/rdd_video_decoupled.json \
  --synthetic --steps 4 --batch-size 2 --device cpu \
  --output unified_multires/runs/my_video_check
"$UMDIFF_PYTHON" -m umdiff sample \
  --checkpoint unified_multires/runs/my_video_check/checkpoint.pt \
  --count 2 --output unified_multires/runs/my_video_check/sample.pt
```

`--separate-stages`生成每阶段独立参数的同架构模型；默认共享。`--decoupled`选择RDD的`main`/`drift`头；f-DM自己使用`epsilon`/`residual`头，二者语义不同。`--process-args`传JSON覆盖过程参数，如stage权重、f-DM rescaling/eta、RDD target/metric、采样步数或PDDPM训练表。已存在配置文件不会被`configure`覆盖。

要用准确的自定义DiT/Wan结构，`--model-json /path/model.json`读取完整registry模型spec；例如`{"name":"wan_grid","args":{"architecture":{...完整Wan构造参数...},"time_scale":1000}}`。更换architecture时应同时核对channel数、head语义、patch与text embedding维度。已有本地权重通过`training.initialization`显式加载，支持`path/key/prefix/submodule/strict`，默认严格检查；原生Wan wrapper可直接加载底层模型state_dict。已有unified训练继续使用`--resume`，不要把只加载权重称为恢复optimizer/EMA。

本地数据格式：`torch.save({"latents": BCTHW, "conditions": {"text_embeddings": ...}}, path)`，也接受`images`或直接浮点tensor。图像使用BCHW。字段中的tensor均为同样的样本数；输入应已经采用实验声明的RGB归一化或latent scaling。`.pt`也可按`umdiff_shards_v1`分片避免一次读取全数据。Wan生成时另用`sample --conditioning LOCAL_PT`提供`{"text_embeddings": ...}`；不会用随机文本代替正式条件。输入为latent时输出也仍是latent；视频压缩/解码器不由这里自动构造。

若需要Pyramid Flow/PyramidalWan的RGB先逐尺度resize、再分别VAE编码，请在process中设置`clean_pyramid_fields`，在`training.conditioning`将对应本地tensor字段送至`destination: "context"`。不提供时走已声明的模型域平均池化路径；二者不能混称为相同codec配方。

### 新范围的验证入口

机器证据：[audit.json](evidence/cross_modal_20261013/audit.json)、[41配置实形状预检](evidence/cross_modal_20261013/config_preflight.json)。运行：

```bash
"$UMDIFF_PYTHON" -m pytest unified_multires/tests/test_cross_modal.py \
  unified_multires/tests/test_cross_modal_cli.py \
  unified_multires/tests/test_component_contracts.py -q
"$UMDIFF_PYTHON" unified_multires/tools/validate_cross_modal.py
"$UMDIFF_PYTHON" unified_multires/tools/check_cross_modal_configs.py
```

新证据分别记录：作者Pyramid Flow原函数/时钟与新目标；PyramidalWan与独立本地paper实现；DiT/Wan适配器逐层F/B；f-DM、Edify、PDDPM、Non-Uniform的独立方程检查；SPEED原函数；图像/视频与不同backbone组合、噪声协方差、位置编码/跨帧依赖、保存/新进程恢复/采样。正式训练质量和全部paper-specific trick没有因此变成已验证。

RALU/JiT源码已先获取并审阅，但本轮不把不规则、区域自适应token布局硬改成规则resize；LSSGen额外学习型latent上采样、NAMI learned bridge、RMD teacher蒸馏、MrFlow外部超分保留为单独扩展/原路线。Blurring Diffusion与Inverse Heat是相关频率退化背景，本轮未新增完整训练路线。PPFlow/DiMR的核心架构创新也保留原native路线，不伪装成同一固定patch DiT。Diffusability/CAT按用户决定排除出主比较范围。

本轮最终结果：**167项pytest、41份配置实形状更新、9种stage程序各128步固定夹具训练全部通过**。原DiT/Wan所测模块F/B和Pyramid Flow所测三阶段原目标对照逐元素相同；PyramidalWan独立paper对照最大FP64误差约3.6e-15。完整汇总见[SUMMARY.json](evidence/cross_modal_20261013/SUMMARY.json)。历史native证书与本次共同模型证据分别保留。

新增多阶段采样默认关闭autograd，并仅在`return_trajectory=True`时保留中间状态；开启/关闭trace的最终输出精确相同，视频内存不随保存的完整采样轨迹增长。
