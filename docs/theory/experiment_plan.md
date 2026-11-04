# RDD 2D / DiT 实验计划

最后更新：2026-08-10

本计划只覆盖 ImageNet-256 的 2D / DiT 实验。Video 实验有独立计划，本文件不调整其内容或优先级。

权威结果总账见 2d/ALL_RESULTS.md。任何新结果进入论文前必须同时登记：原生数据来源、checkpoint、模型类型与参数量、训练 stage split、采样 split 及其顺序、sampler、scheduled steps、真实 NFE、CFG、跨尺度初始化、样本数和完整指标。

## 0. 证据边界

- 只有 checkpoint 路径含 ImageNet_original、使用原生 ImageNet ILSVRC 并完成 50K ADM-reference 评估的结果，才可进入论文主表。
- 旧 Hugging Face 预处理数据训练结果只用于提出假设，不再补跑、不进入论文。
- DDPM 的 scheduled steps 等于 NFE；当前三阶段 DPM2 的真实 NFE 为 2 × scheduled total − 6。
- 比较 sampler 时必须匹配真实 NFE；禁止再比较 “DPM2 scheduled250” 与 “DDPM250” 后声称等计算量。
- inference-time stage boundary mismatch 不是 training-split ablation。

投稿目标说明：下面的“核心闭环”针对最可能的 NeurIPS 审稿质疑设计，但不存在一份实验清单能保证录取或 Spotlight。与其堆很多弱实验，本计划只保留能回答因果性、先行工作重叠、公平效率和可组合性四个关键问题的实验。

## 1. 已完成并可直接用于论文

### 1.1 原生 ImageNet convergence

- Unified XL 与 DiT-XL/2：50K–500K checkpoint，DDPM250、CFG1.0。
- RDD450K 缺失，图中保持空白。
- 关键结果：RDD400K FID7.065，DiT400K FID19.781；RDD150K FID13.755 已优于 DiT500K FID17.464。

### 1.2 Unified XL 主消融

- DDPM250、CFG1.0/1.5/2.0/2.5/4.0。
- Sampling allocation：[32,16,8] = 150/50/50、130/60/60、90/80/80、70/80/100、60/70/120。
- Noise filling 对 bilinear。
- DPM2 scheduled250/125/65/35；真实 NFE 分别为494/244/124/64。
- Inference boundary mismatch：matched 300/300/400 与若干错误边界。

### 1.3 其他已有结果

- Multistage 300/300/400 的主结果、allocation 和低步 DPM2。
- Unified L 400K、DDPM250、CFG1.0。
- 250-step 与1000-step speed microbenchmark；其中正式论文应优先使用250-step结果并明确 benchmark 限制。

## 2. P0：投稿阻塞项

### 2.1 先补评估，不重新训练

这些任务最便宜，而且可能直接补出最强的效率结果；在启动新训练前先完成。

#### P0.1 Unified L CFG1.5

- Checkpoint：ImageNet_original Unified L 400K。
- 设置：DDPM，90/80/80，noise filling，CFG1.5，50K。
- 当前问题：样本评估因 NumPy 的 np.long 报错。
- 动作：修 evaluator 后直接重评，先检查已有样本是否完整，不要盲目重采样。
- 价值：Unified L 训练 microbenchmark 比 DiT 快约10%，250-step采样快约2.56倍；若 CFG1.5 FID有竞争力，它会成为最强 quality-efficiency point。

#### P0.2 Multistage 100/300/600 正确 FID

- Checkpoint：ImageNet_original MS-100 400K。
- 设置：DDPM，90/80/80，noise filling，CFG1.0 与1.5，各50K。
- 当前状态：配置和采样日志存在，完整指标未归档。
- 价值：决定1000-step benchmark中的大幅加速是否具有质量意义。

#### P0.3 DiT500K CFG1.5

- Checkpoint：ImageNet_original DiT-XL/2 500K。
- 设置：DDPM250，CFG1.5，50K。
- 价值：补齐 operating-point comparison；不能再用旧 HF baseline 替代。

#### P0.4 已生成样本的补评估

- 检查 ablation_cfg1.jsonl 对应目录。
- 优先补 Unified XL CFG1.0 的 DPM2/低步结果，以及 Multistage CFG1.0 DPM2。
- 有完整50K样本时只跑 evaluator；样本不足时再 resume。

### 2.2 最小因果矩阵：RDD 到底比 f-DM 多了什么

这是当前最重要的新训练，优先级高于继续扩展 sampler、CFG 或 stage split。f-DM 已经包含阶段内连续插值、精确反向条件、跨尺度 noise rescaling 和边界 noise resampling；论文不能再靠“f-DM 是离散跳变”区分。

先在原生 ImageNet、Unified L/2、200K updates、相同 batch/优化器/VAE/参数预算下做四组：

| 组别 | Forward / target | 回答的问题 |
|---|---|---|
| A. fixed-resolution DiT-L/2 | 标准单分辨率噪声预测 | 同规模基线 |
| B. progressive `d=0` | 相同三分辨率训练和计算，去掉 drift | 收益是否只是多尺度联合监督/额外 loss |
| C. f-DM-style adapted | 同一 latent resolution schedule，双预测 signal + degradation，并使用其 noise rescaling/resampling 原则 | RDD 单一 effective-noise 参数化是否优于最接近先行工作 |
| D. full RDD | 当前 effective-noise 单头 + noise filling | 完整方法 |

执行约束：

- C 必须称为“在相同 DiT/latent 协议下的 f-DM-style adaptation”，不能冒充官方复现；f-DM 无官方训练代码且原论文架构/空间不同。
- 四组先做 10K 采样筛选，200K 端点全部做 50K FID/sFID/IS/P/R。
- 若 B 与 D 接近，停止“drift curriculum 是主要原因”的表述；若 C 与 D 接近，创新应定位为工程简化和 solver 适配而非训练收敛机制。
- 若差异清晰，只将 B、C、D 中最强两组升级到第二训练 seed；不需要所有四组各跑三次。
- 如果预算只能做一个新增 XL/2 训练，应选 B（`d=0`）跑到 400K，并与现有 full RDD-XL/2 直接比较。

### 2.3 一次正交性验证，不做方法堆叠动物园

只选择一条主线：

1. **首选 RDD-FM / SiT**：固定 SiT objective/backbone，比较 SiT 与 RDD-SiT；不依赖外部视觉编码器，最干净地证明 RDD 不限于 DDPM 参数化。
2. **备选 RDD + REPA**：如果 FM 实现不能在短期闭环，在 32×32 fine stage 的指定层加入 REPA alignment，比较 RDD 与 RDD+REPA。必须同时报告 DINO 特征预计算/训练开销。

REG 不作为投稿必做组合：它同时引入 noised class token、class-token velocity head 和 REPA loss，工程量更高，而且其低频语义加速与 RDD 的 coarse-to-fine curriculum 可能重叠。只有 RDD-FM/REPA 组合已经稳定且资源充足时才做 RDD+REG。

### 2.4 最低统计要求

- 主 XL operating points 使用至少3个独立采样 seeds，报告均值、标准差和完整50K指标。
- 训练随机性使用 2 个 seeds 覆盖“full RDD vs 最强因果 baseline”即可；其余矩阵单 seed，并明确探索性。
- 不把小于约0.1的 FID差异解释为机制收益，除非多训练 seed 支持。

## 3. P1：核心机制和公平比较

### P1.1 Matched-NFE sampler matrix

目的：确认当前 RDD 的 DPM2 优势不是由额外模型调用造成，并定位低 NFE 稳定性拐点。

固定条件：

- Model：Unified XL 400K。
- Data：原生 ImageNet。
- Boundary：noise filling。
- CFG：先做1.0；只把64、124、244 NFE中最有价值的 operating points补到1.5。
- Allocation：固定按90/80/80比例缩放。150/50/50只在最终约250 NFE最强点复核一次。

| 目标真实 NFE | DDPM | DPM2 | DPM1 | 目的 |
|---:|---|---|---|---|
| 64 | 必跑 | 已有/复核 | 仅附录可选 | 失稳端点 |
| 124 | 必跑 | 已有/复核 | 仅附录可选 | 中低预算 |
| 244/250 | 必跑 | 已有/复核 | 仅附录可选 | 主论文公平点 |

执行策略：

1. 全部先用10K样本筛选。
2. 三个锚点先10K；64、124、244/250中能改变结论的点和最终主表点升级到50K。
3. 采样脚本直接计数 model forward；文件名同时写 scheduled 与 NFE。

论文判据：

- 如果 DPM2 在 CFG1.0 和1.5、124与244 NFE均优于DDPM，可主张 RDD 与 drift-corrected second-order solver 有稳定协同。
- 如果优势只在CFG1.5出现，应改写为 solver–guidance interaction。
- 如果优势只在noise filling出现，应归因于 boundary consistency，而不是泛化到所有RDD设置。

### P1.2 Boundary × sampler 2×2

目的：解释为什么旧 bilinear/constant 实验几乎总是DDPM更好，而当前 Unified XL 的DPM2更好。

固定 Unified XL 400K、CFG1.5、约250真实NFE：

| Boundary | DDPM | DPM2 |
|---|---|---|
| noise filling | 必跑 | 必跑 |
| bilinear | 必跑 | 必跑 |

若资源允许，再在CFG1.0复现一次。这个实验优先于继续扩展 nearest/bicubic 等更多上采样器。

### P1.3 Checkpoint × sampler sweep

目的：直接检验“训练尚不充分时DDPM更好、训练成熟后DPM2翻转”的假设。

- Checkpoint：100K、300K、400K、500K。200K只在曲线形状不清楚时补。
- Sampler：DDPM、DPM2。
- 真实 NFE：约250。
- CFG：先1.0；只对发生 solver 排名翻转的checkpoint补1.5。
- Boundary：noise filling。

全部先10K筛选；只把翻转两侧和最终主表点升级到50K。没有这组实验前，论文不得把solver翻转归因于训练成熟度。

### P1.4 DPM2 midpoint drift 机制消融

固定 Unified XL 400K、CFG1.5、约250真实NFE：

1. 当前实现：midpoint重新估计噪声和drift。
2. midpoint噪声，但复用起点drift。
3. 不使用drift correction的错误控制组，只作为机制诊断。

这是验证DPM2收益来源的最直接实验，比新增更多通用sampler更有价值。

### P1.5 正式 wall-clock benchmark

当前速度数据可作为microbenchmark，但论文最终版还需要统一协议：

- 单张同型号GPU；
- 真实checkpoint；
- 相同batch、精度、CFG与sampler；
- 生产版noise filling；
- 记录scheduled steps、真实NFE、model forwards、端到端latency、throughput、peak memory；
- 分别报告DDPM250和DPM2约250 NFE；
- 至少30次timed batches并给标准差。

训练速度另做真实8-GPU DDP benchmark；当前542.73/682.82ms是单进程microbenchmark，不能标为8-GPU实测。

## 4. P2：强化论文结论

### P2.1 Limited-interval guidance（低成本正交组合）

现有CFG sweep已经证明高CFG损害Recall，不再做1.1--1.6的密集全程扫描。参考 NeurIPS 2024 limited-interval guidance，只在中间噪声区间启用CFG，是比任意 stage-wise CFG排列更可信、更便宜的组合。

- 固定Unified XL 400K、DDPM250与DPM2约244 NFE。
- 用CFG1.5做全程 baseline，再测试3个预注册区间：按logSNR定义的中间50%、中间70%，以及仅16×16 stage启用。
- 10K筛选后只将最佳区间升级到50K。
- 同时报FID、sFID、Precision、Recall和实际条件/无条件forward次数。

### P2.2 Sampling allocation停止规则

现有150/50/50已经形成明确最优点。默认不再新增180/35/35、170/40/40等密集局部搜索；只在matched-NFE结果显示最优allocation随sampler改变时，在140/55/55、150/50/50、160/45/45三点做10K复核。allocation tuning不能替代机制实验。

### P2.3 真正的training stage-split ablation

已有250/250/500、200/300/500等结果只是推理时改变边界，不回答训练分段问题。

旧HF数据中有一个值得用正确数据复核的稳定现象：在同为400K、DDPM、constant boundary、80/80/90采样时，300/300/400对100/300/600的FID在CFG1.5下为8.865对10.680，在CFG1.25下为11.645对15.216。训练代码确认每个global step都会分别更新三个stage，因此100/300/600的32分辨率模型确实在0--100区间获得约3倍的per-timestep覆盖；结果仍更差，说明局部t覆盖密度不是主要瓶颈。

优先检查的解释是：100-step fine stage使drift rate放大、留给高频纠错的reverse-time区间过短；600-step coarse stage让最小的DiT-B进入更高SNR区间并稀释每个t的训练样本；以及8分辨率状态造成不可逆的信息瓶颈。MS-100从250-step增加到匹配训练区间的1000-step后旧FID仍约10.7，也说明采样离散化不是唯一原因。

建议先用Unified L做100K–200K pilot：

- 100/300/600
- 200/300/500
- 300/300/400

必须从头训练并在matched inference boundaries下比较，同时记录held-out MSE vs global t、stage-boundary前后的高低频重建误差、每stage gradient norm和pred-x0误差。只有pilot显示明确收益后，才扩展到Unified XL。

### P2.4 d^c=0 curriculum control

已提升为P0最小因果矩阵，不再作为“有额外预算才做”的实验。

### P2.5 多seed统计

已提升为P0最低统计要求。主表只要求覆盖结论所依赖的full RDD与最强因果baseline，不对所有探索配置重复训练。

### P2.6 机制分析

以下实验保留为加分项，但不得阻塞主实验：

- 理论noise-filling方差附近的窄范围 sigma sweep；
- drift multiplier在0.75/1.0/1.25附近的窄范围 sweep；
- stage boundary前后的PSD / frequency residual可视化；
- per-class FID，用于检查CFG和高频类别偏差；
- Precision–Recall Pareto图，直接使用已有CFG结果即可。

## 5. P3：范围扩展

### 5.1 ImageNet-512

有价值，但必须在ImageNet-256公平性与机制实验完成之后。先用Unified L或较小模型做200K pilot，提供同架构、同训练预算baseline。没有真实结果前，论文只能写future work，不能写preliminary evidence。

### 5.2 Flow Matching

已作为P0正交性验证首选。先完成一个可信checkpoint的50K FID；若RDD-FM不优于同协议SiT，就诚实报告负结果并改用RDD+REPA作为唯一组合实验，不扩展完整FM曲线。

### 5.3 更小模型

优先补Unified L，不再立即扩展大量B/4、L/4组合。只有Unified L确认形成quality–compute Pareto后，才增加B规模。

### 5.4 RDD-Wan / video flow matching

Video是高价值的跨模态检验，但Pyramidal Flow、PyramidalWan、TPDiff和Spectral Progressive Diffusion已经覆盖“视频按时空分辨率渐进生成”这一宽泛想法。因此最小有意义实验不是只展示能生成视频，而是同一Wan 2.1-1.3B checkpoint下的四路因果矩阵：Vanilla Wan、PyramidalWan、相同stage但`d=0`、完整RDD-Wan。

执行顺序：

1. 先运行`rdd_video/run_rdd_wan_smoke.sh`，再用更新后的生成脚本缓存包含`latent + T5 context + prompt`的离线数据。缺失T5 context必须fail fast。
2. 当前RDD-FM以latent average-pool/repeat幂等投影为主协议。Pixel-space downsample再VAE encode是非线性变换，只用于PyramidalWan数据或未来的transform-specific inverse实验，不能直接套用解析clean reconstruction。所有方法仍共享来源视频/文本、stage boundaries、batch、optimizer和0/1K/2K/5K checkpoints。
3. 以VBench quality/semantic及temporal子指标为主，报告真实NFE、TFLOPs、端到端latency、peak memory和tokens/update。
4. Euler、AB2、Heun必须同时用于RDD和PyramidalWan；边界重采样后重置multistep history。不得声称PyramidalWan结构上不能使用高阶solver。
5. 只有RDD在matched compute下显示更快收敛或更好NFE--quality frontier，video才进入主贡献；否则作为附录通用性结果。

当前`online_synthetic`模式只是teacher生成终点样本后的普通RDD-FM，不是step distillation，而且每个DDP rank同时保存student、EMA、teacher和T5，显存代价过高。真正few-step distillation推迟到稳定RDD-Wan checkpoint之后，并且必须对RDD和PyramidalWan使用同一蒸馏recipe。

## 6. 删除或停止的实验

以下实验不再占用当前2D论文资源：

1. 继续评估旧HF数据checkpoint：绝对指标无效。
2. Decouple续训到700K：成本高，且不能回答当前最关键的Unified/noise-filling/DPM2机制问题。
3. Extended DiT或RDD训练到7M以追求“iso-quality”：资源成本过高，且当前论文应使用matched training compute/convergence curve。
4. 大规模DDIM/任意sampler zoo：先完成DDPM/DPM1/DPM2的真实NFE公平比较。
5. 更多inference-only stage mismatch：结论已经明确；下一步应做真正的training-split ablation。
6. CFG≥2的更密集搜索：已明确造成FID和Recall恶化。
7. 当前论文内的Consistency/TRACT/distillation组合训练：与核心贡献正交、工程量大，且现阶段只有推测性乘法加速。可作为独立后续工作。
8. CIFAR-10 unconditional scope-defense：低分辨率任务不能充分体现RDD的多分辨率计算价值，优先级低于ImageNet-512。
9. 同时做RDD+REPA、RDD+REG、RDD+DiMR、RDD+FasterDiT：只能选择一个正交组合，避免论文退化为技巧堆叠。
10. 把f-DM、DFM或LapFlow描述成“不连续/启发式”以制造差异：这在文献事实层面不可防守。

## 7. 收敛后的最小投稿包

### 阶段A：立即完成的零训练闭环

1. P0.1--P0.4全部补评估。
2. 在采样代码中加入真实NFE计数。
3. 完成matched-NFE的64/124/244三个锚点。
4. 复核并修正speed benchmark元数据和MS-100 FLOPs。
5. 完成正式wall-clock、peak memory和训练step-time协议。

### 阶段B：投稿必做训练

1. Unified L/2 200K最小因果矩阵：DiT、`d=0`、f-DM-style、full RDD。
2. 将最强因果baseline与full RDD补到第二训练seed；若差异不清晰，优先升级`d=0`到XL/2 400K。
3. 完成boundary × sampler 2×2和精简checkpoint × sampler筛选。
4. 只完成一个正交组合：RDD-FM/SiT首选，RDD+REPA备选。
5. 完成limited-interval guidance最佳区间的50K评估。

### 阶段C：冻结主表前

1. 关键NFE点、因果矩阵端点、正交组合全部使用50K ADM-reference评估。
2. 完成精简midpoint drift消融。
3. 主XL结果补3个采样seed；full RDD与最强因果baseline补2个训练seed。
4. 冻结数据manifest、配置、checkpoint hash和绘图脚本。

### Spotlight stretch：最多再选一个

1. **首选 ImageNet-512 matched pair**：Unified L/2、200K、固定分辨率baseline与RDD各一组。如果效果不清晰，不扩XL。
2. **备选长训曲线**：继续现有RDD-XL到750K/1M，只在能补同等协议baseline或明确画compute-normalized趋势时使用。

不要两个都做。Spotlight更依赖一个清楚、可信、可复现的新结论，而不是更多表格。即使全部结果理想，也不能保证录取或Spotlight；它只意味着当前最明显的先行工作、因果性和公平性漏洞已被覆盖。

## 8. 最终论文应形成的四张核心证据图

1. FID vs training steps / measured training time，并同时标出REPA/REG公开结果但注明协议不同。
2. 最小因果矩阵：DiT / `d=0` / f-DM-style / RDD。
3. FID vs真实NFE的DDPM/DPM2 Pareto，附boundary × sampler机制小图。
4. FID–latency–memory Pareto，突出Unified XL/L，并用一张小表报告唯一正交组合。

旧HF结果只在内部分析中用于解释实验设计，不出现在这四张图中。
