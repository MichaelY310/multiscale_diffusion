# 共同架构实验与逐论文核心复现的范围

## 2026-10-14 更新：DFM 时间采样修正与全部主候选的训练策略审计

下节记录的是修正前的审计；**DFM common-DiT 主配置现在已恢复官方 lognormal**，同时保留 NumPy 阶段抽样、早期 band 的 shifted-logit-normal 时间和作者日志 RNG 消耗。REPA 仍为官方示例默认的 off。作者原 loss 与同一个缩小共同 DiT 的连续64次CPU更新，模块输入3776对、输出3584对、activation梯度3584对、参数梯度1984对逐字节相等；权重/Adam/RNG也一致。实际Trainer两次累积更新、30/10采样和7项相关回归通过。只修改该主配置、生成器与manifest，其余17份配置（含RDD）hash不变；历史报告不覆盖。

[逐方法完整表、packing/共享噪声含义、理论期望与计算预算、LapFlow差异](evidence/training_policy_audit_20261014/REPORT_ZH.md)，[机器证据](evidence/training_policy_audit_20261014/AUDIT.json)。PixelFlow/Pyramid Flow每图全阶段、SSD整批同时间、f-DM分层积分、Haar联合目标是不同策略；DFM/LapFlow并未全部改成同一种阶段平均。不保证同batch/updates等计算或变化对质量影响小。PixelFlow共同采样入口按用户要求保留Euler；不开展Dopri5接入。


## 2026-10-14 表示空间、codec与公平性复核

本轮为只读配置/源码及原论文复核，不变更已验配置/生产代码，不启动训练或下载模型/数据。证据见[evidence/representation_review_20261014/AUDIT.json](evidence/representation_review_20261014/AUDIT.json)。18份main/controlled配置的VAE声明与input_transform完全相同：SD-VAE-ft-ema、posterior sample、scale0.18215、4×32×32。但真实资产由启动参数绑定，现有入口没有内容hash锁定，不能把配置一致写成已测真实权重一致；整个unified的历史native/视频配置仍保留各自codec。

- **DFM更正**：作者SiT公开复现例`enc_type:none, proj_coeff:0`，默认已关闭REPA。原论文正文/附录未找到REPA on/off消融，不能编造关闭后损失数值。论文SE-FLUX与公开SD-VAE-SiT要区分；后者源码用`sd-vae-ft-mse`。当前共同配置另把作者时间采样`weighting:lognormal`改为`uniform`，是独立的有意共同设置，不能由“REPA默认关闭”推导成整个配方原样。[官方仓库](https://github.com/snap-research/dfm)、本地`source/dfm`实际路径为`sources/dfm/experiments/train/dfm_sit_b_256.yaml:35`。
- **PixelFlow更正**：原collate给不同图像平衡分配阶段，packing隔离样本注意力，loss按图像平均；当前StageProgram每张图在所有阶段计算、共享噪声金字塔、阶段平均后一次optimizer update。仅取消packing布局可以等价，但当前还改了估计器相关性/方差和成本。论文表3b固定CFG1.5：Euler FID2.51、Dopri5 2.43；表3c阶段CFGmax2.4结合Dopri5为1.98。[原表](https://arxiv.org/html/2504.07963#S4)。新latent路径Euler/CFG1不属于这份最佳配方。
- **表示影响有直接证据**：LapFlow表2a在CelebA-HQ256中SD→EQ使LapFlow FID4.37→3.53，而LFM5.26→7.77；codec与方法存在交互，不保证人人同涨同跌。[论文](https://arxiv.org/html/2602.19461)。DFM表4在ImageNet512、DiT-B400K中原FLUX→SE-FLUX使DFM34.00→32.89、FM54.50→43.16；这些不是本项目结果，也不能直接预测换SD-VAE的影响。[论文附录D/表4](https://arxiv.org/html/2506.19839)。
- **EMA/MSE不能与完整换表示混淆**：Stability官方说明ft-ema与ft-mse只微调decoder，不能仅因名字不同就认定latent encoder也换了；decoder仍会影响RGB评价。本轮没有下载这两套权重做文件/张量检查。[模型卡](https://huggingface.co/stabilityai/sd-vae-ft-mse)。
- **其他术语**：SSD关闭的是采样中clean预测的[-1,1]截断，latent没有该范围约束；PF的temporal pyramid是压缩较早的历史帧作为自回归condition，当前图像空间路径不包括它；LapFlow原MoT本已基于Transformer，改动是取消尺度专家参数并保留联合band/causal mask；Haar细节LH/HL/HH形成3×4通道，额外输入/输出head含调制共2,768,304参数（主体674,834,720的0.41%）。

实验解释：同DiT/同codec可以公平回答“固定表示与骨干时，哪个过程更好”；不能仅凭该表宣称原完整方法排名。先保留旧RDD/SD-VAE锚点；建议另用相同EQVAE比较RDD/DFM/LapFlow，并为pixel原生方法保留pixel-DiT组（至少RDD与PixelFlow），各组重新匹配预算、允许等预算调参。此处为建议，尚未添加或执行新实验。一般非线性encoder满足`E(Down(x)) != Down(E(x))`，latent分解与RGB尺度语义不自动相同；随机CPUparity无法证明质量损失为零。PixelFlow工程可跑的推荐不等同最忠实原表示的推荐。

截至2026-10-14可核查的公开系统：Wan2.2、HunyuanVideo1.5、LTX2、FLUX2、Qwen-Image2.1、SANA都使用学习压缩后的latent（SANA是DC-AE，不必统称VAE）；JiT/PixelDiT/PixelFlow代表活跃的pixel研究。来源URL逐项保存在AUDIT；没有猜测未公开的商业模型内部架构。

## 2026-10-13 已准备：ImageNet-256 主实验和受控配置

新配置和运行说明见[README的同DiT实验章节](README_ZH.md#imagenet-256-同-dit-主实验与受控实验2026-10-13)，准确清单为[comparison_imagenet256/manifest.json](configs/comparison_imagenet256/manifest.json)：12份main、6份controlled，9种多尺度方法+f-DM另一日程候选+DDPM/flow基线。使用旧RDD的DiTMultiRes-XL/2主体、SD-VAE-ft-ema、ImageNet256、global256、AdamW1e-4、400K；其余方法的UNet/专用Transformer、pixel表示、codec、teacher等取舍已逐一写在每个JSON的`comparison.changes`。

RDD的模型、1000步表、fine→coarse 300/300/400目标、各阶段独立取样、sum_stage_means和optimizer数学配方不变；新表采样锚点是**实际旧400K DDPM实验的coarse90/mid80/fine80、CFG1**，不是下文旧通用250步例子的100/75/75，也不是paired模板的DPM2。当前新入口单进程micro32×accum8，不冒称原8-rank RNG、data顺序和all-reduce逐位重放。

main是各方法日程的预先登记候选，共同的是数据暴露量，**未声称等训练FLOPs或已经调到最佳**。controlled的5个标量过程固定8/16/32、名义400/300/300与90/80/80调用；第6项SSD只匹配尺度/分区，保留1000步算子采样，明确不是250NFE对照。联合band方法保留必要接口，未强塞统一标量时钟；Haar额外投影参数量单列。PyramidalWan是path数学移植；f-DM cosine为明确假设候选，另备论文允许的linear，均不能叫作者最佳原配方。

CPU验证：18配置完整网格/日程、缩小网络各两次累积更新和完整预算采样通过；RDD与原目标的144模块输出、144activation gradients、31parameter gradients逐字节相同；174项pytest通过；完整XL meta构建的共同主体674,834,720参数及键/shape一致。新增入口实际验证了本地随机VAE的RGB训练、跨进程恢复和采样；没有正式长训、预训练资产质量、GPU显存/速度或50K评分结论。首次RDD夹具的平均池化运算顺序错误与修复证据保留，生产RDD未修改。

证据：[PREFLIGHT](evidence/main_comparison_20261013/PREFLIGHT.json)、[结构](evidence/main_comparison_20261013/ARCHITECTURE.json)、[主体限定FLOPs](evidence/main_comparison_20261013/SCALAR_CORE_FLOPS.json)、[回归](evidence/main_comparison_20261013/pytest.log)、[最终摘要/源码检查](evidence/main_comparison_20261013/SUMMARY.json)。新入口`python -m umdiff.experiment plan/train/sample --config ...`；原GPU guard仍在，当前仅CPU。下文旧“尚未实施共同骨干”等按日期视为历史。

## 2026-10-13 最新实验筛选与阶段时钟核查（覆盖下文旧建议）

用户明确移除 **PDDPM、Relay Diffusion、MrFlow、Bottleneck Sampling**：当前主实验和推理补充均不再选择这四项，保留源码、配置和历史证据。下文曾建议将它们放入主表/补充表的内容仅为历史记录。旧`run_*_papers.sh`与parity清单用于历史复现，未改成当前主实验启动器，不能使用其中的`all`代替本节筛选。

另经论文§3核查，**完整Edify 256→1024系统**也先完成256采样，再前向加噪并运行1024 upsampler，按相同标准排除。框架中`edify_laplacian`只是内部dimension-varying EDM的受控实例，保留为数学组件，不称完整Edify主结果复现。Relay/PDDPM有上采样、噪声及路径衔接设计；排除理由不是“没有处理transition”，也不是证明这些方法必然更慢。

### 阶段配置：原论文、作者发布默认与本地测试值分开

`1000`可能指训练时间坐标，并不等于推理调用1000次网络。下表连续区间包含共享边界；只有RDD/SSD行是离散整数索引。分辨率为正方形边长，latent与pixel明确区分。

|方法、所核配置|尺度与原始时间|换成高噪声→低噪声的1000刻度／实际预算|结论|
|---|---|---|---|
|RDD：此前讨论的三阶段DDPM/jump配置|latent 8→16→32；fine→coarse stage_lengths=[300,300,400]|999→600；599→300；299→0；未缩减400+300+300次，250配置100+75+75次|全局反向步划分；不是三个完整去噪循环|
|SSD：作者ssd.yaml默认4尺度ConvexDecay_0.5，64图像实例|pixel 8→16→32→64；tau=t/999，阶段位置x=1-sqrt(1-tau)|999→937；936→750；749→438；437→0；分别63/187/312/438个时间索引|作者分辨率函数CPU枚举所得，不声称该默认等于所有主表最佳配方|
|SSD：相同schedule改2尺度|pixel 32→64|999→750；749→0，即250/750|可表达的两尺度配置，不因可表达就认证所有主表结果；equal四尺度则各250|
|f-DM-DS：论文256实验|pixel 16→32→64→128→256；5阶段cosine；论文采样250个时间步|**精确数字切点尚不能认证**：正文打印tau_k=cos(1-k/(K+1))，与tau_0=0矛盾，缺作者可执行源码；不能擅自补pi/2后称原版值|单个全局过程，机制符合；本地FDMProcess默认等分不是论文cosine配方|
|Pyramid Flow：作者三阶段scheduler，shift=1,gamma=1/3|相对空间尺度1/4→1/2→1；stage_range=[0,1/3,2/3,1]|真实噪声系数sigma×1000分别1000→667；800.240→334；500.750→0。网络time另映射约1000→744→386→1；示例首视频单元20/20/20步，后续单元10/10/10|不是物理sigma的三个无重叠区间；边界噪声校正后继续，没有先完成低尺度完整循环|
|PixelFlow：作者XL C2I yaml，4阶段，gamma=-1/3|pixel 32→64→128→256；signal系数s分别0→.25、1/7→.5、1/3→.75、.6→1|以q=1000(1-s)表示：1000→750；857.143→500；666.667→250；400→0。模型time边界0/175/426/719/999；CLI Euler默认每阶段30步，共120，Dopri5自适应|阶段末点名义四等分，但边界校正后起点不同，不能写成物理噪声无回跳|
|PyramidalWan：论文非蒸馏多步配置|natural noise zeta：1→.9412→.5858→0；3个时空尺度|natural刻度1000→941.2→585.8→0；论文20/20/10步。zeta不是Wan网络sigma，需按时空缩放omega转换|paper实现，非作者仓库parity；首帧/后续帧映射也须保留|
|DFM：公开SiT-256两尺度|latent 16→32；先粗band的t1:1→.1，后粗t1:.1→0与新细band t2:1→0联合演化|只看粗时钟可写1000→100→0；第二段同时有细band1000→0。generation yaml步数30/10，train yaml监控采样40/10|多个band时钟，不能伪装成所有分量共享同一个t；粗band在边界未完成|
|LapFlow：论文256主设置|latent 16→32；noise→data时间t:0→.5→1|令q=1000(1-t)，为1000→500→0；高分辨率三尺度t:0→.33→.67→1，对应1000→670→330→0。主表使用自适应ODE，不是1000次调用|后段联合演化活跃bands；论文T=.5、本地fixture .4、作者CLI默认.9分别记录，不互相冒充|
|Non-Uniform Diffusion：论文128多尺度实验|3级Haar即4个尺度16→32→64→128，反向t:1→.75→.5→.25→epsilon|1000→750→500→250→约0；论文总256 Euler–Maruyama步|频带逐渐加入；作者4个独立score模型，旧band继续变化，不是干净低图作为额外condition|
|SPEED：作者FLUX 1024、scales=[.5,1]、delta=.01|latent64→128；切点由实测频谱P(w)=A w^-beta及容差计算|连续阈值sigma*=.852662598；边界对齐后sigma~=.920472620，即1000→852.663，再920.473→0。作者默认总网格50步；实际触发在网格首个sigma<=阈值处，故实际边界不等于连续公式值|推理扩展，切点随模型/尺寸/容差改变；不提供跨模型固定600/300切分|

Pyramid Flow和PixelFlow的网络time表不是整数DDPM索引，精确端点（包括最后一次模型评估与积分终点之别）见JSON，勿将约值取整后充当作者采样代码。SPEED的50是原scheduler网格配置，不据此保证时间重映射后的真实NFE仍恰好50。

f-DM的歧义：若**额外假设**正文遗漏了pi/2，使用tau_k=cos((1-k/5)pi/2)，反向1000刻度会是1000/951.057/809.017/587.785/309.017/0。这只是明确标注的候选解释，未被作者源码认证，不写入原版主配方、不覆盖现有parity配置。论文可以确认的内容是五尺度、cosine划分与250步采样。

MDM是联合嵌套多尺度；DiMR主要是模型内多分辨率；DCTdiff是固定DCT表示；PPFlow是patch/token阶段机制；RMD另有蒸馏teacher；NAMI有学习bridge/结构变化。它们不因此成为“完整低图→重新加噪”方法，但也不应被列成已确认的RDD式变网格1000步分割。本轮不扩大此前固定DiT主表范围。

当前优先比较：RDD、f-DM-DS、SSD、Pyramid Flow空间路径、PixelFlow、PyramidalWan path；DFM/LapFlow/Non-Uniform作为联合频带组；SPEED放推理组。f-DM精确原论文日程仍有上面明确的歧义，不能将“可进入方法比较”写成“原论文所有配置已经确认”。

### 本轮证据

- CPU执行作者scheduler或其原函数AST，输出精确端点、SSD全部1000个索引分组与源码SHA256：`evidence/stage_clock_audit_20261013/audit.py`、`RESULT.json`。CUDA未初始化。Pyramid Flow使用已固定的作者实现快照（位于LapFlow的old_baselines内），没有下载新代码或权重。
- 作者文件：`sources/ssd/configs/ssd.yaml`、`sources/ssd/guided_diffusion/ssd_utils.py`；`sources/pixelflow/configs/pixelflow_xl_c2i.yaml`、`pixelflow/scheduling_pixelflow.py`、`sample_ddp.py`；`sources/lapflow/old_baselines/Pyramid-Flow/diffusion_schedulers/scheduling_flow_matching.py`、`inference_multigpu.py`；`sources/dfm/experiments/{generation,train}/dfm_sit_b_256.yaml`；`sources/speed/{configs.yaml,utils.py,latent_image_gen.py}`。
- 论文：[f-DM §3.1/3.3/4.1](https://arxiv.org/html/2210.04955)、[PyramidalWan 附录训练与采样](https://arxiv.org/html/2601.04792)、[LapFlow §5.1及表2d](https://arxiv.org/html/2602.19461)、[Non-Uniform §4.1](https://arxiv.org/html/2207.09786)、[Edify §3](https://arxiv.org/html/2411.07126)。
- 本次只改范围/说明并新增小型CPU日程证据；没有重新训练、下载模型/数据、修改作者源码/生产算法或覆盖历史parity结论。

## 2026-10-13 采样循环与“完整去噪”筛选更正

用户所指是实际执行完整反向采样循环的计算，而非在中间时刻代数预测一次 `pred_xstart`。不能仅凭边界加噪、局部时间到0/1、或训练噪声表有1000项，就认定某方法额外跑了一遍1000步去噪。

新增可复跑CPU证据：`evidence/sampling_loop_audit_20261013/audit.py` 与 `RESULT.json`。执行原采样循环，使用记录调用次数的占位模型，包含完整调用时间轨迹和源码hash；不是实际模型质量、完整网络parity或GPU耗时测试。CUDA未初始化，无模型/数据下载，无训练或checkpoint写入。审计夹具初次运行的Relay尺寸限制和f-DM命名输出接口修正也记在JSON中，未改生产采样器。

- **RDD保留**：正式 `sampler=ddpm` 调用jump sampler。旧RDD三阶段分别遍历999→600、599→300、299→0，未缩减时实测400+300+300=1000次模型调用；所测250步分配100+75+75=250次，DPM2同250调度点为494次调用。边界复用已有`pred_xstart`，没有额外求解到干净端点的循环。先前将其归类成Relay式额外完整采样是错误的；局部噪声重新归一化并不能支持该分类。此结论不扩展成噪声幅度连续或所有solver端点相同的声明。
- **PDDPM是最明确的完整低分辨率循环例子**：论文Algorithm 1及§4.1–4.2设置Tf=1000、Ts=100、delta_ts=.3，三尺度预算1000+30+30。当前本地paper实现默认restart_steps为1000/250/250，未缩减实测1500次；已保存小配置steps=20实测20+20+20=60次。这些本地配置是controlled预算，不是论文主表原配方，不可混称。按用户实验范围移出主线，不删除实现。
- **Relay官方ImageNet链式流程**：先完整低分辨率EDM到sigma=0，再上采样并运行截断高分辨率blur采样。作者CLI默认低256区间，Heun实测511次调用；高200网格点经rho=100、sigma<.9筛选后剩100区间，199次去噪器评估（CFG=3.5时cond/uncond分开为398次），总909次模型forward。这个CLI数值不是最佳FID预算认证，也不是1000步+1000步。按“同一过程分配分辨率”的主线列为接续生成对照。
- **MrFlow与Bottleneck不能以必需千步重跑为由淘汰**：MrFlow作者FLUX/Qwen脚本都是12步低分辨率生成+1步高分辨率refine，另有VAE/SR成本；Bottleneck当前小配置是20+10+10=40次调用，阶段末端归零再加噪。两者有生成后细化/阶段重启，但属于短流程推理扩展，不能据此声称计算必然高于RDD。Bottleneck论文表4列N=[6,20,8]且另有strength，正文与算法截断约定存在歧义，没有作者可执行源码证据时不捏造精确论文NFE。
- **LapFlow保留方法、记录实现冗余**：两尺度作者/兼容采样先积分粗尺度0→1，却仅以中间t=.4状态进入联合阶段。dt=.05实测粗尺度20次+联合12次=32次，存在对最终主输出非必要的粗尺度尾段；三尺度相同dt实测6+6+8=20次，无该分支。当前未删除兼容计算，后续效率结果必须计入或明确声明去除附加输出计算。
- Pyramid Flow、PyramidalWan path、PixelFlow、f-DM的所测6+6+6预算均为18次，无隐藏完整采样循环；RDD-FM为18+2次边界预测；Edify默认Heun为36+1次最终去噪预测。端点是否带噪应读物理系数，不能只看局部积分变量。

筛选建议：PDDPM和Relay官方链式流程不列当前单过程主训练比较；MrFlow/Bottleneck仅作为短流程推理补充，不以“必须1000步”描述；RDD以及其余中间状态/联合频带方法不因`x0`预测被过滤。实际效率用各分辨率NFE×单次成本及codec/边界开销衡量，以上未测真实端到端速度。

论文证据：[PDDPM](https://arxiv.org/html/2208.01864)、[Bottleneck](https://arxiv.org/html/2503.18940)。原版代码证据在`sources/relay/generate_imagenet.py`、`sources/mrflow/flux1_mrflow.py`、`sources/mrflow/qwen_image_mrflow.py`、`sources/lapflow/sampler/ours_sample_joint.py`。

本次用户接受：仅CPU、共同非零随机权重的计算对照作为方法实现的准入；不把未测试真实预训练权重、GPU/BF16/Flash等作为当前核心完成的阻塞。正式需要teacher/codec的实验仍要使用有学习能力的模型；随机资产只用于正确性测试。不下载模型或数据。

## 共同架构的含义

共同Transformer主体、规模、数据与预算可以用于controlled comparison。各方法必需的patch投影、多尺度输出头、时间映射、网络路由及噪声/损失仍应保留。全部改成固定patch和单输出的原始DiT，会删除PPFlow/DiMR/MDM等方法的一部分。已有作者backbone证据不自动成为新Wan/DiT移植的证据；应对新适配器另外做小量对照。

## 问题逐项结论

- SSD：已测CelebA64四尺度是主表配置之一，不是已验全部主表或最佳FID配方。主表还含CelebA64两/三尺度、CelebA128三/五尺度、CelebA256三/四/五/六尺度（五尺度部分结果仅报计算成本）和ImageNet64两/四尺度。官方发布checkpoint覆盖其中七组。当前可变`resolutions`列表、beta、loss权重及Flexi-UNet配置能表达同一SSD族的这些变化；仍需建立具体配方并逐项对照。Equal/ConvexDecay/TanhLikeDecay可编码成显式分辨率列表。附录替代退化和非SSD消融不自动都已实现。论文表2的CelebA64 SSD(2L) FID2.14、SSD(4L) FID4.28；这里是引用论文，不是本项目成绩。
- DFM：teacher来自可选REPA特征对齐，通常是冻结视觉特征编码器；核心DFM flow目标不要求生成teacher。源train.py中`proj_coeff<=0`或`enc_type=none`不加载REPA encoder。可以将无REPA作为共同实验设置；若与REPA配置比成绩，要单列额外监督及成本。
- PPFlow：四/五“层”是patch尺度阶段数量，不是Transformer深度。公开训练入口的构造默认`learn_sigma=True`，输出多余通道在flow读出中被丢弃；不能解读为论文额外优化方差loss。公开两/三阶段主分支已测，no-sigma不作为当前实验前提。统一主体仍需保留可变patch、阶段投影和packing。
- DCTdiff：已测公开UViT实现保留DCT表示和目标；未公开DiT分支不阻塞数学方法的共同架构实验。UViT是具有浅/深层长skip的Transformer生成骨干。DCTdiff在RGB频域工作，若直接改成VAE latent DCT则是新变体，不能称原RGB DCTdiff。
- MDM：已测共同非零随机网络权重，模块输出/梯度相同，可按当前标准完成核心准入。随机测试不是任意权重的数学证明；真实checkpoint加载仍应检查键名、shape、buffer等，但不要求现在下载。嵌套结构保留，私有Flickr与公开CC12M的质量差异不作为本轮代码阻塞。
- DiMR：新增CPU RGB→原版随机VAE→moments/posterior/scale→DiMR训练与采样→RGB的验证。报告入口`evidence/dimr_rgb_20261012/REPORT_ZH.md`。VAE按作者冻结，实际训练只更新DiMR；codec输入导数另外诊断，不伪称作者训练VAE。
- Diffusability：主创新是codec的尺度等变正则，不是新的多阶段生成目标。论文式(2)要求低分辨率latent解码能重建低分辨率RGB。当前已验SE等组件，模板明确不是定位到的最佳配方（例如模板SE weight=1而论文主要实验alpha=.25）；完整codec微调→重编码→下游生成器训练未执行。完整GAN并非主创新的必要前提，不能把所有仓库可选分支列成主结果必须项。
- RMD：接受已测Wan非GAN分支作为完成。冻结生成teacher是方法必需，随机teacher仅测算术，正式实验需有能力的teacher。其他backbone不继承当前证明；参考仓库官方归属未独立确认，来源记录保留。
- NAMI：数学上适配阶段path/learned boundary/深度路由；BridgeFlow是可学习的线性阶段映射，论文还逐阶段增加Transformer模块。更窄的固定深度、无额外bridge训练主表可暂不纳入；公开推理已验，作者完整训练器未提供，训练复现不能声称完成。
- MrFlow：方法本身不增加训练。预训练flow低分辨率生成→VAE解码→预训练Real-ESRGAN超分→VAE编码→少量噪声→原生成器高分辨率refinement。结合Pi-Flow/FLUX-schnell也是加载已经蒸馏的模型，不是MrFlow又训练了模型。
- SPEED（本轮原论文复核更正）：论文第4.3节明确给出频谱变换后的分阶段微调目标，不能称“论文没有训练目标”。阶段输入由推理同款频谱扩张及时间对齐构造，起止端点共享噪声，监督分阶段直线速度；论文还做LoRA微调。固定官方仓库README明确只支持training-free inference。现有unified认证只覆盖推理；论文微调配方尚未实现并认证，不能由通用Trainer的存在推导为已覆盖。来源：https://arxiv.org/html/2605.18736#S4.SS3 ，`sources/speed/README.md`。
- CAT：依内容复杂度选择压缩比的tokenizer/codec；caption+LLM选择复杂度，nested VAE产生不同token数量，另训练DiT。已测codec，未提供的作者训练器不能被诊断`output.square().mean()+KL`代替。适合表示/codec扩展，不必挤入多阶段diffusion主方法表。

## 当前用户范围与接口澄清（本轮追加）

- 用户将Diffusability和CAT排除出当前主研究范围；保留历史源码和证据，不继续扩展这两项。NAMI作为可选结构/learned-boundary扩展讨论。
- MDM的NestedUNet和DiMR的多分支Transformer/ConvNeXt及TD-LN均属结构贡献；将其换为未经改造的普通DiT会舍弃方法核心。PPFlow可以共享Transformer主体，但必须保留变patch及阶段输入/输出投影；DFM、LapFlow也有多band读写接口。
- 数学框架包含layout/位置编码与参数共享；当前工程中这些由`model`、native adapter及部分batch控制，尚无独立通用的position-embedding registry或适配全部作者架构的一个开关。
- 通用`path/target/metric/boundary`可组合；native `batch`与配套`sampler`为了保留RNG、packing、loss归约而采用受约束的组合，不能随意用顶层path/target覆盖。现有native parity不自动证明改为统一DiT/Wan后仍相同。
- 当前`rdd_decoupled`定义是effective-epsilon主头加raw-drift辅助头，不是pure-epsilon与drift两个独立头。RDD图像默认阶段边界是上采样预测clean后重新加噪，而非只resize带噪状态。

上述为论文/代码审阅与范围文档修正；本轮没有执行新的训练、GPU任务或parity测试，也不改写历史测试结果。

## 固定DiT主体后，各方法还保留什么（2026-10-12本轮审阅）

用户明确希望实验集中比较多分辨率diffusion过程，预计使用统一DiT。以下是对研究范围的分析建议，尚未修改训练配方或实施新的骨干移植。

建议固定DiT的主体块、深度、宽度和基础位置编码规则，保留表达方法所必需的输入/输出接口、多尺度时间条件与条件依赖。单一输入张量、单一时间标量、完全相同的输出头，并不是所有多尺度数学方法都能无损满足的约束。相同架构也不等于必须将所有阶段的独立参数强行共享。

|方法|替换专用架构后的判断|建议实验位置|
|---|---|---|
|RDD|resolution drift、预测参数化、噪声耦合和边界都独立于特定DiT结构|主表|
|Relay|patch级模糊路径、block noise及relay连接仍在；将多个同结构DiT强行合为一份参数是另一个变量|主表，标注DiT移植|
|SSD|失去Flexi-UNet的结构效率，但线性退化扩散和算子后验仍是主要贡献；保留输入/输出尺度转换|主表，标注DiT移植|
|PixelFlow|多阶段flow端点、再加噪及时间对齐仍在；RGB换latent是额外的表示变化|主表；latent版本明确标为路径移植|
|DFM|分量分解、独立时钟及渐进训练采样仍在；保留多分量读写、时间条件，REPA可关闭|主表，注明无REPA及codec设置|
|LapFlow|Laplacian分量和延迟flow仍在；论文主方法的MoT及因果注意力是重要结构贡献，不能笼统称为无关trick|保留shared-DiT数学变体；主表或补充表显式标注w/o MoT；因果mask需保留或单列消融|
|MDM|去掉NestedUNet损失嵌套结构和效率，但extended-space联合多尺度扩散及尺度SNR仍有独立意义；不能退化为各尺度独立训练|补充的MDM-style joint diffusion on DiT，不称完整MDM|
|DiMR|去掉多分支Transformer/ConvNeXt、TD-LN、特征级联，主要只剩普通VP和可选多尺度辅助loss|从该主表排除；若保留辅助loss，作为明确命名的自身消融|
|PPFlow|若保留变patch仍保留核心；若固定patch且去掉阶段投影，主要剩普通flow|严格固定patch主表排除；可另列同DiT主体的token效率比较|
|DCTdiff|DCT表示、频率处理与SNR设计仍在，UViT不是方法成立的必要条件|表示空间补充实验；不是最直接的渐进分辨率基线|
|NAMI|固定深度去掉模型划分收益，但分段flow和learned BridgeFlow仍有独立意义；不能说全部核心消失|learned-boundary补充实验，注明固定深度；完整训练仍有发布限制|
|SPEED|频谱扩张、时间对齐及谱驱动日程不依赖专用生成骨干|同一已训练DiT上的推理表；论文微调配方仍未实现并认证|
|MrFlow|不依赖专用生成骨干；依赖额外VAE/SR流程，不能删掉SR后仍称原方法|推理流程补充表，计入SR/VAE成本|
|RMD|跨分辨率分布匹配蒸馏不依赖特定Wan结构；teacher是目标所必需|蒸馏专表，不混入从头训练同监督主表|

论文依据：[MDM §3.1–3.3](https://arxiv.org/html/2310.15111)、[DiMR §4](https://arxiv.org/html/2406.09416)、[PPFlow §3](https://arxiv.org/html/2506.23543v3)、[DFM §3.3–3.4](https://arxiv.org/html/2506.19839)、[LapFlow §4.4](https://arxiv.org/html/2602.19461)、[NAMI §3–4](https://arxiv.org/html/2503.09242)、[SSD作者仓库](https://github.com/prateksha/ScaleSpaceDiffusion)。LapFlow源码`models/MultiScaleDiT.py`分别提供`use_mot`与`use_mask`，README将MoT+causal列为论文主方法，关闭MoT分支列为progressive/ablation；不能把开关存在解读为该贡献不重要。

与此研究问题紧密相关的另外两个候选是[f-DM](https://arxiv.org/abs/2210.04955)的变换均值/噪声尺度/反向过程，以及[Pyramidal Flow Matching](https://arxiv.org/abs/2410.05954)的空间金字塔flow/阶段连接。后者的图像实验只能代表其空间部分，不能冒称完整视频时空方法；前者当前是paper-component候选，不增加native认证状态。

结论必须限定为“共同DiT和实验预算下，多分辨率过程的受控比较”。不据此宣称击败被删除结构贡献后的完整原论文，也不自动继承作者骨干的逐元素parity。本轮只做论文/代码审阅及文档更新，无训练、GPU或新增parity实验。

## 文献和固定代码

- SSD：[论文](https://arxiv.org/abs/2603.08709)、[官方仓库](https://github.com/prateksha/ScaleSpaceDiffusion)，本地论文审阅文本`/tmp/rdd_ssd_audit.txt`表2/3/5与附录8.3。
- DFM：`sources/dfm/train.py`的enable_repa分支、`sources/dfm/README.md`的SiT发布说明。
- PPFlow：[论文](https://arxiv.org/html/2506.23543v3)，`sources/ppflow/models.py`构造/flow读出，`sources/ppflow/train.py`模型构造。
- UViT：[论文](https://arxiv.org/abs/2209.12152)；DCTdiff：`sources/dctdiff/README.md`。
- Diffusability：[论文](https://arxiv.org/html/2502.14831v3)式(2)、`configs/paper_recipes/diffusability_se24.json`的明确component-template声明。
- NAMI：[论文](https://arxiv.org/html/2503.09242)第3节。
- MrFlow：[论文](https://arxiv.org/html/2607.01642v1)第3节及表2说明。
- CAT：[论文](https://arxiv.org/html/2501.03120)，已有固定补充材料与codec诊断报告。

Codec在本项目指图像/视频表示的编码与解码组件，常见为VAE/AE；teacher可能是蒸馏生成器或REPA特征编码器，两者与codec、文本编码器、用于初始化的checkpoint均应区分。

## 补充：以扩散过程为核心的文献候选（2026-10-13）

用户要求补回f-DM、Pyramid Flow、PyramidalWan，并继续寻找主要贡献不在模型架构的工作。上面的共同DiT表不是完整文献目录。本节是论文/公开项目页审阅及实验范围建议，不是新增实现或parity认证；不更新历史`PAPER_COVERAGE.json`的审计计数。

### 训练与微调方法

|方法与论文|核心数学/过程|共同DiT保留范围与建议|当前证据边界|
|---|---|---|---|
|[f-DM: A Multi-stage Diffusion Model via Progressive Signal Transformation](https://arxiv.org/abs/2210.04955)，2022/ICLR2023|沿扩散时间变换干净信号，联合处理信号变换、噪声尺度、预测参数化及反向过程|最高优先级的RDD近邻；保留变换相关输出、尺度条件和阶段重采样，不必保留特定UNet|已有数学映射；`fdm_decoupled_clean`是组件移植，不是完整f-DM。此次[官方项目页](https://jiataogu.me/fdm/)仍标Code & Data即将发布，未获得作者可执行代码|
|[Pyramidal Flow Matching for Efficient Video Generative Modeling / Pyramid Flow](https://arxiv.org/abs/2410.05954)，2024/ICLR2025|不同空间分辨率的flow段、噪声与时间对齐、阶段连接；视频还含历史帧的时间金字塔和自回归机制|高优先级；图像共同DiT可以比较空间路径，但必须标spatial transplant，不能声称复现完整视频机制。它不是PPFlow|[官方仓库](https://github.com/jy0205/Pyramid-Flow)存在；本项目旧视频/FastVideo工作有单独实现和证据，不自动继承为当前图像DiT parity|
|[PyramidalWan: On Making Pretrained Video Model Pyramidal for Efficient Inference](https://arxiv.org/abs/2601.04792)，2026|把预训练Wan微调成时空金字塔flow；推广正交变换下的阶段转换；另研究teacher与步数蒸馏|视频高优先级；选择真实分辨率金字塔分支，不选择另行讨论的patch-pyramidal分支。主去噪骨干可沿用，但首帧/时空条件、teacher监督和训练预算要保留或显式消融|此次[官方项目页](https://qualcomm-ai-research.github.io/PyramidalWan/)未找到作者训练源码。已有本地paper implementation，公开公式与作者未披露选择见根目录`PAPER_IMPLEMENTATION_CONTRACT_ZH.md`；不能称native作者复现|
|[Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models](https://arxiv.org/abs/2411.07126)，2024|Dimension-Varying EDM：Laplacian各频带不同消失日程；移除已无信号的高频以减小状态，逆向恢复高频及噪声|高优先级的频域/路径对照；抽取Laplacian扩散过程放在同DiT，保留频带日程、SNR及边界。不能称完整Edify级联系统复现|已有paper mapping，未确认可执行作者完整训练发布；数学路径重实现与作者最优系统成绩分开|
|[Non-Uniform Diffusion Models](https://arxiv.org/abs/2207.09786)，2022|不同分量使用不同扩散速度/SDE；§3.3以多层Haar分解让高频更快到达噪声，逆向逐级加入细节|很相关的理论与训练基线；需要各频带时钟、非均匀loss权重和多尺度读写。原实验按阶段使用独立score网络；同DiT架构可保留，强制一份共享权重是新增变量|本轮补充候选，未接入/测试。论文提MSDiff，但脚注写待发布；此次未确认可执行官方库，不能据摘要称已经拿到源码|
|[Pyramidal Denoising Diffusion Probabilistic Models / PDDPM](https://arxiv.org/abs/2208.01864)，2022|先完成低分辨率去噪，再上采样、前向加噪，运行较短的高分辨率逆过程；单个score网络配坐标编码|适合简单且可解释的低成本基线；换DiT后保留多尺度坐标语义。它是逐层重新加噪/去噪，不等于一条连续resolution-drift路径|已有paper mapping，未确认可执行作者仓库；不能标native验证完成|

固定DiT主体不等于强制所有方法同一份权重、同一输出头或同一标量时间。上述“适配”是根据方法数学作出的工程判断，不是已有共同骨干数值测试结论。

### 同一预训练模型上的推理候选

|方法与论文|改变什么|共同DiT的限制/额外组件|公开源码检查|
|---|---|---|---|
|[Training-free Diffusion Acceleration with Bottleneck Sampling](https://arxiv.org/abs/2503.18940)，2025|高→低→高的非单调分辨率轨迹；优化转换时机与时间步重映射|不需新去噪网络；适合检验低→高是否必然最优|论文所给`https://github.com/tyfeld/Bottleneck-Sampling`本轮返回404，不将“论文有代码链接”写成“可执行源码已公开”|
|[RALU: Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers](https://ignoww.github.io/RALU_project/)，2025/2026|低分辨率→边缘区域先加密的混合分辨率→全分辨率；噪声与时间匹配|无需重训主模型，但要混合token坐标/输入布局；并非当前规则矩形grid日程的无改动配置|[官方仓库](https://github.com/ignoww/RALU)可访问；原arXiv题名为Upsample What Matters，ID2507.08422，是同一工作|
|[Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers / JiT](https://arxiv.org/abs/2603.10744)，2026|空间近似生成ODE、稀疏/自适应token及确定性Micro-Flow，处理增加空间自由度时的一致性|无需修改或重训主模型；需要输入布局和局部ODE插入逻辑，可作为噪声边界之外的确定性边界对照|[官方仓库](https://github.com/Wenhao-Sun77/Just-in-Time)发布FLUX推理。此JiT不是Just image Transformers那篇|
|[LSSGen: Leveraging Latent Space Scaling in Flow and Diffusion for Efficient Text to Image Generation](https://arxiv.org/abs/2507.16154)，2025|低分辨率早期采样，经学习型latent上采样与日程调整进入高分辨率|不改主去噪器，但额外训练依赖VAE的轻量ResNet上采样器；作为learned-boundary补充，不能称完全无需额外训练|本轮论文审阅，未确认可执行官方发布；[ICCV2025 workshop论文](https://openaccess.thecvf.com/content/ICCV2025W/AIGENS/papers/Tang_LSSGen_Leveraging_Latent_Space_Scaling_in_Flow_and_Diffusion_for_ICCVW_2025_paper.pdf)|

这些与SPEED/MrFlow一起属于推理过程比较；不直接拿未改动预训练权重的方法去回答“RDD从头训练是否更好”。本轮未下载这些新候选仓库或执行测试。

### 数学框架应引用、但未必放入变网格主表的两个早期工作

- [Blurring Diffusion Models](https://arxiv.org/abs/2209.05557)，2022/ICLR2023：把模糊/热扩散与Gaussian diffusion联系起来，并构造频率相关的模糊加噪过程；主要贡献在退化过程，适合路径理论比较。
- [Generative Modelling With Inverse Heat Dissipation](https://arxiv.org/abs/2206.13397)，2022/ICLR2023：随机反演热方程，从低频结构恢复高频细节；[官方代码](https://github.com/AaltoML/generative-inverse-heat-dissipation)存在。

这两者的“尺度”首先是信号/频率尺度，不自动意味着模型运行时张量分辨率递增或token减少。若加状态压缩，是需单独证明/测试的移植，不能默认作者已有相同计算加速。

### 建议优先级与框架含义

1. 图像训练补充优先：f-DM、Pyramid Flow空间路径、Edify Laplacian路径；Non-Uniform Diffusion作频带时钟基线，PDDPM作简单阶段重启基线。
2. 视频微调补充优先：PyramidalWan；teacher/蒸馏分支单列额外监督和计算预算。
3. 推理补充优先：Bottleneck Sampling、RALU、JiT；LSSGen作为额外学习边界实验。
4. 框架需区分信号频率尺度、张量网格尺度、局部token布局。非均匀时钟/协方差、非单调分辨率日程、非均匀空间布局和确定性/学习型阶段映射都有文献依据，不能仅用“换upsample算子”概括所有差异。

本轮仅文献核查和范围文档更新，无训练、GPU、模型或数据下载，也无新增parity结论。

## 2026-10-13 共同模型与跨图像／视频实现验收

新增可执行过程：f-DM-DS、Pyramid Flow空间路径、PyramidalWan path、Edify dimension-varying EDM实例、Non-Uniform Haar、PDDPM；补充Bottleneck推理策略。已有RDD/PixelFlow/Relay/SSD/DFM/LapFlow/MDM/DCTdiff及普通flow/DDPM接入共同模型配置；SPEED边界可用于共同模型推理。总计16种训练配置选择、2种可选推理策略、41份小型配置。入口`python -m umdiff configure`，具体命令与各方法边界见`README_ZH.md`末尾。

- `grid_dit`支持轴分离sin-cos/3D RoPE、joint时空attention、product-state多band、stage/逐帧clock；`dit_grid`复用原DiT blocks；`wan_grid`调用真正的Wan Transformer；`grid_conv`用于小型调试。可指定宽度、深度、heads、patch或完整model JSON，并选择共享／独立stage参数。不是每个backbone都强行支持每个方法：普通DiT/Wan的单张量API不支持联合band，原DiT不支持PyramidalWan首帧／尾帧不同clock，组合会明确报错。
- 167项CPU pytest全部通过；41份已保存配置按实际模型和网格各一次更新通过；9种stage程序各128步固定随机小视频拟合，final/initial loss为0.0356–0.6397。这只检查可训练性，不是泛化或生成质量。
- 原DiT图像/T=1、Wan图像/T=3的模块输入、输出、activation gradient、parameter gradient全部已测张量相同。Pyramid Flow原作者`add_pyramid_noise`、原scheduler与新实现的三阶段输入、目标、clock、loss、网络逐层输出和梯度全部相同；网络是两边共用的可控小DiT，不是对作者完整视频DiT的声明。修正过FP32标量提前cast的2e-7量级差异，最终没有保留该失配。
- PyramidalWan对照迁移项目的独立paper math，FP64最大差异约3.6e-15，来自池化/代数等价式的浮点次序；这不是作者仓库parity。f-DM(v1)、Edify(v1)、Non-Uniform(v1)、PDDPM按已注明方程及具体日程检查。SPEED三种spectral boundary在图像/视频上与原函数相同。
- Relay共同模型路线包含低EDM训练/采样→repeat→高blur diffusion，未把随机图像替代低模型输出。PDDPM/DDPM采样预算与1000步训练噪声表分离。局部Video shard→训练→新进程resume→采样精确相同；T=1与image相同，T>1有跨帧非零导数。
- 回归原通用组件：Relay 84、LapFlow 715、RDD 412项通过；LapFlow两项原已注明的reconstruction恒等式舍入仍为2.38e-7，不是native/unified失败。旧76项整套作者架构认证没有被冒称为本轮全部重跑。

证据：`evidence/cross_modal_20261013/SUMMARY.json`、`audit.json`、`config_preflight.json`、`logs/`及`regression/`。22份外部source-only快照的2113个选中文件hash通过；新增RALU/JiT/Inverse Heat只用于预审，不算实现。全部CPU、无预训练模型/数据下载、无GPU、无共享dirty FastVideo修改。

本轮按用户允许的统一范围排除：不规则token的RALU/JiT、codec专用学习upsampler的LSSGen；PF自回归历史、PW teacher/DMD、SPEED论文微调、Edify未披露最优日程；其余结构/蒸馏/外部SR路线保留原实现或明确独立。新图像／视频互换都是controlled transplant，不能凭CPU测试宣称作者最佳结果或跨任务迁移质量已经复现。f-DM默认边界离散化、Edify线性attenuation、Bottleneck论文步数歧义等已列在README，不隐藏为原版配置。
