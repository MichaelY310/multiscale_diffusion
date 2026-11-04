<!-- SUPPORTED_SCOPE_CURRENT_20261012 -->
> **2026-10-12 当前支持范围验收已完成**：固定最终代码的 76 项 CPU 任务及全部补充回读通过。12项生成/codec/蒸馏核心的已测原版与 unified 模块输入、输出、activation gradient、处理前 parameter gradient 逐字节一致；SPEED/NAMI/MrFlow 是推理对照，backward N/A；CAT 是 codec 加诊断 adjoint，不冒称作者训练目标。31份完整 image 配置各做两次独立 forward/backward、不更新 optimizer；另验完整 codec、缩小视频网格的30层 Wan，以及12方法和12扩展组的缩小网络各128步。
>
> [最终逐方法报告](/scr/yuheng/bugswarm/rdd_workspace_20261005/unified_multires/evidence/unified_scope_20261012/REPORT_ZH.md)、[机器结果](/scr/yuheng/bugswarm/rdd_workspace_20261005/unified_multires/evidence/unified_scope_20261012/AUDIT.json)、[统一点与明确排除项](/scr/yuheng/bugswarm/rdd_workspace_20261005/unified_multires/SUPPORTED_REPRODUCTIONS_ZH.md)。报告记录原版细节、共享 backbone/solver、显式源码缺陷兼容和修补。已测支持范围没有未解释数值失配；论文最佳质量、真实资产训练、原 CUDA kernel 和全部作者 CLI 不在本证书内。历史148项清单继续保留，不被整体改成完成；下文旧 pending、旧源文件计数和 GPU 授权均为历史记录，当前以本块和最终报告为准。
>
> 新增证据包含31配置新进程恢复、真实 CPU Gloo/worker/GradScaler、SSD TrainLoop 与 PPFlow main、RDD27组可选目标及两rank自适应采样器、PixelFlow Dopri5四阶段34项原始轨迹/像素/RNG检查。PixelFlow提供与作者相同的int32标签，之前仅索引dtype不同的尝试保留；两项LapFlow重构恒等式约2.38e-7舍入单列，不算原版/unified失配。19库2077个选定源码文件及本地RDD源码结束时哈希检查通过。
>
> 全部测试进程已结束。已回读检查点后清理 447 个临时文件、2.528 GiB，保留报告/日志/配置/hash。重跑：`bash unified_multires/run_supported_reproduction_checks.sh --workers 2`（`--plan`仅查看）。仅CPU，无模型/数据下载，无共享 dirty FastVideo/环境修改，不恢复旧监控或GPU授权。
>
> 初轮75/76通过；旧回归脚本按失效行号提取reader导致NameError，修为AST命名定位后该项和pytest重测通过。生产实现、配置及其他测试文件未变，旧失败与精确hash差异保存在[测试修复记录](/scr/yuheng/bugswarm/rdd_workspace_20261005/unified_multires/evidence/unified_scope_20261012/harness_repair.json)；未重写历史结果。
<!-- /SUPPORTED_SCOPE_CURRENT_20261012 -->

<!-- CPU_PARITY_20261012 -->
**2026-10-12 当前状态更新**：本文件下方是2026-10-10的原始审计条目与反例描述，保留为历史记录。它们的**当前状态以 [148项逐项处理记录](CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md) 为准**，不能再把旧“尚未修复”文字当作当前实现结论。

本轮已实际完成12种训练/codec/蒸馏方法、33份选中配置的当前逐层forward/backward对照：31份完整image backbone/输入网格、完整SD式VAE、完整30层Wan（缩视频网格）；实际捕获的输出、activation gradient、处理前parameter gradient全部逐字节相同、max_abs=0。12方法128步及12扩展128步全部重跑，39对完整状态checkpoint已重载验证并清理临时payload。另3个公开推理workflow通过，作者训练backward N/A；CAT新公开补充codec完成缩小/完整512诊断对照，但不是其diffusion trainer复现。见 [完成报告](CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md)。

**未把148项全部标成关闭**：这份原清单同时包含私有/未发布trainer、未知best-run、真实资产/数据、CUDA多卡、正式长训练和主表FID；仅CPU forward/backward不能完成这些验收。完整官方CLI仍0/33认证、最佳主表仍0行正式复现。所有已经修掉的CPU错误与限定实测、仍缺的前提分别列在新ledger。
<!-- /CPU_PARITY_20261012 -->

# Original / Unified：未完成事项、偏离与验证缺口清单

整理日期：2026-10-10。状态依据同日[严格审计报告](evidence/strict_original_unified_audit_20261010/REPORT_ZH.md)和[AUDIT.json](evidence/strict_original_unified_audit_20261010/AUDIT.json)。此文档用于后续逐项修复和验收；**写入清单不表示已经修复，也不表示此次重新执行了全部训练实验**。

## 1. 范围、状态与已有证据

范围是14种外部发布方法加本地RDD：Relay、SSD、PixelFlow、DFM、LapFlow、PPFlow、DCTdiff、MDM、DiMR、Diffusability、RMD、SPEED、NAMI、MrFlow、RDD。33份成对配置属于其中12种训练/codec/蒸馏方法，SPEED/NAMI/MrFlow另走推理专用对照。另8篇仅数学映射的文献列在第5节。此前Pyramid Flow/PyramidalWan视频工程不属于33份清单，其接入和验收须单独检查。

当前不能承诺“原论文最佳结果可以一键完整复现”，也不能承诺“所有入口、所有forward/backward和真实最终像素逐位相同”。

- `matched-core`的original侧调用作者目标/path，unified侧调用我们的组件，但共用作者backbone、数据读取和Trainer。双方共同偏离官方完整CLI时，仍可能相等。
- `native-cli`的33/33配对均未认证，参考命令存在未完成适配，严格模式拒绝启动。
- 近期30份paper、11份fair短训，每份2次缩小网络/合成输入更新：**实际捕获的模块输入、输出、activation梯度、参数梯度相等**；不能外推未执行的分支、完整真实训练或不同kernel。
- 13份现存小采样tensor重新载入后逐字节相等；包含latent、uint8和小随机模型像素，不能当作全部真实预训练模型的最终浮点pixel认证。
- 通用Relay/PPFlow/LapFlow/RDD组件检查存在容差内非零差，详见第4节。
- 17个源快照中1211个选定文件与基线87个实现文件hash核对通过；这不表示1211文件全部逐行审完或完整Git/权重全部认证。

状态标签：

| 标签 | 含义 |
|---|---|
| 已确认偏离 | 有具体代码、参数或CPU反例证明不同；尚未修复 |
| 未验证 | 功能/配置可能已有，但没有完成所要求的完整对照 |
| 未定位 | 最佳run的配方、seed、预算、资产或评估行尚未确定 |
| 实现缺口 | 指定入口或分支尚未完整接入 |
| 发布限制 | 所检查发布不含必要训练器/私有分支/配方；不能伪造认证 |
| 范围限制 | 已有通过仅覆盖限定测试，必须保留说明 |

执行顺序：先处理CPU可修的真实错误和CLI/数据生命周期，再绑定准确配方与资产、补当前完整模型/采样测试；之后才是重新获授权的CUDA/分布式验证和正式质量实验。当前只用CPU，不下载模型或数据，不启动正式训练、不恢复旧监控、不改dirty FastVideo环境。

## 2. 所有方法共有的待办

以下均未达到完整验收；部分有源码核心证据，仍不应勾选完成。

| 编号 | 状态与具体问题 | 后续应完成什么 |
|---|---|---|
| C01 | 范围限制：original短测是作者核心加我们的训练循环，并非逐篇原样官方CLI | 对每个已发布训练入口建立真正original CLI与unified CLI的完整配对；保留`native`与`controlled transplant`标签 |
| C02 | 未验证：33个native配对全部`certified:false` | 完成真实入口参数、依赖、数据、初始化、更新、保存、恢复和采样小流程验收后，才逐配置认证 |
| C03 | 已确认偏离：参考CLI没有完整表达所选合同，部分参数遗漏/错误 | 逐项检查作者parser、默认值、官方脚本；落盘最终完整配置和原版命令，逐字段比较，禁止仅比较合同SHA |
| C04 | 实现缺口：unified参考仍指向准备时JSON，没有落盘合同中替换资产/改变控制项后的配置 | 原版与unified都写出最终resolved配置；实际入口加载它，并核对加载后的参数值 |
| C05 | 已确认控制差异：paper模式改成world1，普通micro1、PPFlow8、PixelFlow4再累积 | 单独记录原拓扑、单进程替代和累积方式；比较原rank-local/reduced梯度及image/update时钟，不宣称与原多卡轨迹逐位相同 |
| C06 | 已确认控制差异：fair模式global256/world1/micro16/accum16/FP32/seed0/400k/公共AdamW，不是所有论文原配方 | 保留“原配方复现”和“RDD预算对齐新实验”两组，分别验收；不得把fair结果当作论文原成绩 |
| C07 | 未定位：部分作者最佳global batch、seed、更新/epoch/image预算未知 | 找到与具体论文表格行对应的原run配置/记录；找不到就保留未知和阻塞，不补成默认0或任意步数 |
| C08 | 已确认偏离：共同reader和普通unified CLI跨epoch拼接尾部，多个原loader用`drop_last` | 每方法匹配丢尾/跨epoch策略及每epoch更新数；用不足整batch的数据检验epoch边界 |
| C09 | 未验证：原DistributedSampler/shuffle/shard/worker/prefetch RNG没有逐入口对齐 | 比较样本ID、rank分配、batch顺序、worker随机流；恢复后重新验证下一批完全一致 |
| C10 | 未验证：真实数据集、split、JPEG/PNG/ZIP/moments布局和预处理未全部端到端比较 | 使用已有本地同一小数据检查crop/resize/flip/normalize/label/统计，不以合成输入替代 |
| C11 | 已确认偏离：条件DCTdiff ImageNet和Relay缺label时会补`arange(batch)`，未被必需标签guard覆盖 | 真实条件训练必须报错或读取真实label；测试缺失label、micro1和多个类别，禁止补造类0 |
| C12 | 未验证：共同LocalVAE先编码逻辑global batch再切micro；原版通常按rank/局部batch编码/采后验 | 匹配编码调用边界、posterior抽样、预计算moments/在线编码、AMP与RNG时序，并比较真实latent |
| C13 | 未验证：资产通常只检查路径存在，未强制绑定论文对应内容hash和确切变体 | 为backbone/VAE/teacher/text/tokenizer/SR/bridge/初始化建立来源、版本和内容hash清单；不同变体不能悄悄替代 |
| C14 | 范围限制：两侧复用作者网络，不是独立网络架构重写的认证 | 在结果中保留backbone复用说明；若以后声称独立架构重写，需另做架构/权重映射/层级对照 |
| C15 | 已采用替代：CPU attention、源码提取、reader/distributed stub及兼容shim不等于原生运行 | 每方法记录实际替代；对目标kernel/精度另测，不能把共同替代后的相等外推到原生实现 |
| C16 | 已确认环境不同：当前依赖版本与作者示例不同 | 在独立环境固定作者依赖和源码commit，记录必要补丁；共享FastVideo环境保持只读 |
| C17 | 已确认比较器盲点：通用`bitwise`只是`torch.equal`，可能忽略dtype和signed-zero原始字节 | 同时检查shape/dtype/原始字节、非有限值；增加已知dtype反例，区分容差相等、数值相等、字节相等 |
| C18 | 未验证：model/Adam/EMA多数比较数值和dtype，未全部比较原始字节；冻结参数group被规范化过滤 | 比较完整状态并明确active/frozen参数group差异；区分tensor内容相同与checkpoint序列化文件相同 |
| C19 | 未验证：参数grad多数在clip/nan_to_num后比较 | 另捕获unscale前后、clip/repair前后原始梯度及修复mask，防止处理步骤掩盖差异 |
| C20 | 范围限制：module hooks不覆盖所有functional/ATen操作、非Tensor结果和未执行分支 | 导出逐分支覆盖表和首次不同位置；对影响目标、边界、CFG和更新的关键functional单独对照 |
| C21 | 范围限制：paper289、fair70条未执行模块记录；DiMR各profile有4对requires_grad输出两侧都未反传 | 区分容器与实际未执行层；解释无grad原因，并用覆盖关键分支的输入补测，不能计作已验证grad |
| C22 | 范围限制：最新成对训练主要是缩小网络、合成输入、随机初始化、2次更新 | 使用准确初始化和当前完整网络做小量训练/推理对照；不能以短测证明最终质量或收敛 |
| C23 | 未验证：此次未重跑41份训练整套、此前全部128步或全部完整架构 | 修复后按当前代码/配置重新运行必要对照，绑定新hash；旧结果只保留为旧范围证据 |
| C24 | 未验证：历史16份完整image架构证据未绑定当前harness，3份模板后来变化 | 14份旧harness SHA不同、2份缺SHA；MDM256/1024和SSD模板变化，详见第6节。补当前配置对应的完整网络短测 |
| C25 | 未验证：所有native的初始化、warmup/curriculum、callback/eval RNG、精度、冻结策略未整体对齐 | 比较完整调用顺序和状态，不只比较单独loss；将必要callback随机数纳入恢复证据 |
| C26 | 未验证：真实完整CLI的checkpoint和新进程resume未逐方法完成 | 比较保存的model/optimizer/EMA/scaler/RNG/数据游标，再比较恢复后的下一步；旧payload删除后不能用于恢复 |
| C27 | 未验证：许多推理对照止于latent、uint8或tiny/random VAE | 比较真实完整pipeline在uint8量化前的浮点pixel，以及全部阶段状态；latent或PNG相同不足以证明全链相同 |
| C28 | 未验证：原版与unified的API类型、张量layout、输出dtype、文件内容/metadata未完整统一 | 逐方法记录输出契约；PixelFlow ndarray/Tensor差异已确认，必须明确适配并测试 |
| C29 | 未验证：CUDA FP16/BF16/原FP32、Flash/xformers、scaler/overflow/TF32/CUDA RNG | CPU可预检的先做；真实CUDA kernel需要重新获GPU授权后专门验证，不能继承CPU零差 |
| C30 | 未验证：实际DDP/FSDP/all-reduce、分布式累积、rank seed和通信顺序 | 按作者拓扑验收；小两卡smoke不能认证原8/16卡最佳拓扑，更不能用world1结果代替 |
| C31 | 未验证：全部正式训练、论文预算、多seed和最佳质量主表 | 明确每行原结果的任务/数据/初始化/预算，然后训练并用官方评测协议验收；当前未复现任何最佳主表行 |
| C32 | 未验证：FID/IS/precision-recall/50k采样、参考统计、CFG选择预算和视频评估 | 统一公平组评估协议，同时保留原论文协议；先确认数据split/统计匹配，不能只比较示例图片 |
| C33 | 未验证：实际训练/推理速度、显存、FLOPs、NFE及预训练成本 | 在同硬件、精度、任务和质量预算下计量，包含VAE/text/SR/bridge与预训练；CPU时长不能当GPU性能 |
| C34 | 范围限制：hash仅覆盖选定文件，论文全部附录、所有源文件和未选分支没有逐字/逐行审完 | 保留审查范围与源文件列表；进一步审未选关键分支，不能宣称不存在其他差异 |
| C35 | 已确认说明错误：GPU准备旧清单把MDM统一写native FP16，RMD资产写Real video/prompts | 修正后续生成器/计划：MDM逐YAML选择精度；RMD论文主实验是image-free/prompt-only。旧报告保留历史身份 |
| C36 | 实现缺口：未来GPU规范是计划，没有已认证CUDA runner | 补实际执行adapter、资产检查、独立guard和结果采集；此次不启动GPU，也不把66条计划说成66条可执行实验 |
| C37 | 配置元信息不够精确：部分dataset字段是整篇论文数据集汇总，不是该profile实际训练数据；例如DFM字段列512/1024/Kinetics，而公开模板是256 latent | 每profile单独记录实际dataset/resolution/任务、codec压缩率和输入字段；不能从论文汇总字段推断该配置覆盖所有任务 |

公共证据：

- [native_cli.json](evidence/strict_original_unified_audit_20261010/native_cli.json)、[cli_inventory.json](evidence/strict_original_unified_audit_20261010/cli_inventory.json)：真实入口拒绝与33配置参考参数。
- [data_order.json](evidence/strict_original_unified_audit_20261010/data_order.json)、[labels.json](evidence/strict_original_unified_audit_20261010/labels.json)、[comparator.json](evidence/strict_original_unified_audit_20261010/comparator.json)：epoch、条件和比较器反例。
- [AUDIT.json](evidence/strict_original_unified_audit_20261010/AUDIT.json)：覆盖、历史配置delta、各profile阻塞和源码hash。

## 3. 逐方法的偏离、缺项和未执行实验

以下勾选框是未完成清单。官方没有发布训练器的方法，不能靠自造训练器宣称原样复现；如果只验收公开推理，应明确限定范围。

### 3.1 Relay Diffusion（2配置）

- [ ] REL01【已确认偏离】恢复官方默认增强`augment=0.12`及完整AugmentPipe；模板目前没有增强。
- [ ] REL02【已确认偏离】网络增加与原CLI对应的`augment_dim=9`；目前默认0，模型参数结构不同，需匹配真实权重。
- [ ] REL03【已确认偏离】高阶段EMA半衰期模板500,000,000与原CLI默认500,000相差1000倍；修正并检查EMA rampup/image clock。低阶段50M已有对应记录，不作为同一错误重复修。
- [ ] REL04【已确认偏离】修正成对计划覆盖loss_multiplier后的归一化；常数loss CPU检查相对官方8rank公式low/high为1/16、1/2。需同时匹配microbatch、DDP平均、FP16 loss scale和更新时机，不能只调一个常数后就宣布认证。
- [ ] REL05【实现缺口】完整参考CLI补FP16、ls25/1、dropout.1、efficient attention、blur/up-scale/block/prob等官方flags；当前参考命令不是完整官方复现命令。
- [ ] REL06【未验证】native ZIP reader、dataset.json真实标签、数据增强随机流与数据顺序。
- [ ] REL07【未定位】最佳总图像预算的low/high分配和确切checkpoint对；目前1250M总体信息不足以认证两阶段训练run。
- [ ] REL08【未验证】真实64→256双网络cascade、最佳权重和完整sampling设置；现存8/16尺寸随机像素对照不能替代。
- [ ] REL09【未验证】原xformers/FP16、gradient repair、按images计LR/EMA及8rank完整流程。
- [ ] REL10【数值差异】通用block DCT与作者FFT DCT有FP32约1.67e-6/FP64约1.78e-15差，需保持相同运算顺序并验证差异消除。

证据：[relay.json](evidence/strict_original_unified_audit_20261010/relay.json)、[component_relay.json](evidence/strict_original_unified_audit_20261010/component_relay.json)；源入口[sources/relay/train.py](sources/relay/train.py)。

### 3.2 Scale Space Diffusion / SSD（1配置）

- [ ] SSD01【范围/实现缺口】当前只选CelebA64四层；尚未覆盖64的2/4层、128的3/5层、256的3/6层及ImageNet64等全部主结果配置。
- [ ] SSD02【未定位】对应最佳行的训练checkpoint、停止预算和评测协议；发布64的1M、128/256的300k信息不能自动认证当前选中run。
- [ ] SSD03【未验证】官方`dataset_split_file`及CelebA真实train split；普通ImageFolder没有替用户落实原split。
- [ ] SSD04【实现缺口】独立停止上限；不能把`lr_anneal_steps`当作只停止训练的参数，因为它改变LR。
- [ ] SSD05【已采用替代/未验证】CPU补BilinearOp输出尺寸属性、模拟rank0/barrier；原Hydra训练入口和真实多卡完整流程未对照。
- [ ] SSD06【未验证】全部分辨率的完整模型、数据、原DDPM采样和最终像素/质量。

当前source script batch32×2=global64的元信息已经修正；不是仍待修的global32问题。源脚本：[train_ssd.sh](sources/ssd/bash_scripts/train_ssd.sh)。

### 3.3 PixelFlow（Euler、Dopri5两配置）

- [ ] PIX01【未定位】论文最佳训练global batch、完整预算和最终run；公开8rank×4、10epochs、seed42示例没有被确认是最佳训练行。
- [ ] PIX02【已采用替代/未验证】CPU packed varlen SDPA替换原Flash变长注意力；必须验证原生Flash/BF16，不能把共享SDPA零差外推。
- [ ] PIX03【已采用替代/未验证】Diffusers/NumPy RoPE兼容shim与作者依赖环境；固定作者版本或明确修改点后再测试。
- [ ] PIX04【实现缺口】原CLI YAML完整overlay，尤其fair控制项；原epoch-only入口缺匹配累积/独立max-update适配。
- [ ] PIX05【已确认接口差异】original输出NumPy ndarray、unified输出Tensor；当前pixel数值相同，但返回类型未统一。
- [ ] PIX06【未验证】真实最佳权重、原生训练全部生命周期、50k FID及text-to-image主结果。
- [ ] PIX07【未验证】完整真实pipeline下Euler/Dopri5 solver容差、CFG、NFE和RNG；当前256浮点pixel零差只认证已测随机模型配置。

源入口：[train.py](sources/pixelflow/train.py)、[公开YAML](sources/pixelflow/configs/pixelflow_xl_c2i.yaml)。

### 3.4 DFM（XL、B、REPA三个配置）

- [ ] DFM01【发布限制】公开仓库明确是SiT/REPA重实现，架构和结果不同于论文DiT/SE-FLUX；不能把公开分支认证说成论文原架构认证。
- [ ] DFM02【已知配方差异】论文beta(.9,.99)、wd.01、warmup10k与公开模板beta(.9,.999)、wd0等不同；逐组保留原配方，不能混写。
- [ ] DFM03【已知预算/任务差异】论文512的500k、1024的350k、video的200k与当前256/400k公开配置不同；完整论文分辨率和视频主结果未覆盖。
- [ ] DFM04【发布限制】私有SE-FLUX finetune数据、确切autoencoder及原run资产未发布/未绑定，无法据当前发布承诺原样最佳复现。
- [ ] DFM05【已确认CLI错误】参考缺`--exp-name`和`--model`，作者parser会exit2；补准确B/XL模型、config及必要flags。
- [ ] DFM06【实现缺口】REPA成对launcher仍被阻塞；真实DINO/teacher、projector、teacher预处理及训练全链未对照，合成teacher特征不足以认证。
- [ ] DFM07【未验证】source预计算VAE mean/std、sd-vae-ft-mse训练与生成VAE选择；在线RGB编码不能自动视为等价。
- [ ] DFM08【未验证】原Accelerate精度/梯度累积、warmup、保存/恢复、阶段采样权重、CFG和完整生成质量。

此前source NumPy debug draw已经保留并有对应验证，不应重新列为仍缺失；后续回归仍需保留。来源：[作者README](sources/dfm/README.md)、[train.py](sources/dfm/train.py)。

### 3.5 LapFlow（CLI默认与论文flags两个配置）

- [ ] LAP01【已确认CLI错误】参考dataset=`imagenet`无效，源码为`imagenet_256`；已CPU复现UnboundLocalError。
- [ ] LAP02【已确认参数遗漏】参考缺model_type/image_size/f/num_levels/in-out channels/use_mask/time_seg_small/label_dropout/num_classes/num_epoch等；默认ADM/32/3层/3通道/mask关/T.9/无类别/1200epochs不对应所选论文XL/256/两层/4通道/mask开/T.5/120epochs。
- [ ] LAP03【已知配方差异】CLI默认LR5e-4、EMA.9、T.9与论文ImageNet256 LR1e-4、EMA.9999、T.5不同；论文flags模板已有，但真实原CLI尚未认证。
- [ ] LAP04【未验证】准确EQVAE和posterior；不能换SD-VAE或random初始化仍称论文原实验。
- [ ] LAP05【未定位】最佳CFG；当前CFG2是明确评估选择，不是已确认最优FID配置。
- [ ] LAP06【实现缺口】source square sampler缺陷/未port；须说明不可用或完成有证据的修补，不能静默宣称覆盖。
- [ ] LAP07【范围限制】两层专用对照已有；三层只有额外组件级4步，未做完整native三层训练/采样验收。
- [ ] LAP08【实现缺口/未验证】HQ512/1024、对应三尺度、cosine2e-4→1e-6及batch256/256/128等主结果配方未完整纳入并运行。
- [ ] LAP09【未验证】epoch预算和数据量、原cosine/constant策略、更新数量、恢复与正式质量。
- [ ] LAP10【数值差异】generic residual分解→重构对原图恒等式约2.38e-7误差；这是有限精度恒等式检查，不应误写为两层native训练已失配。

证据：[lap_cli.json](evidence/strict_original_unified_audit_20261010/lap_cli.json)、[component_lapflow.json](evidence/strict_original_unified_audit_20261010/component_lapflow.json)。

### 3.6 PPFlow（两层、三层两个配置）

- [ ] PPF01【未验证】真实SiT预训练初始化、权重转换和后续fine-tune；随机架构fixture禁用了这一步。
- [ ] PPF02【已知预算差异】公开CLI1400epochs未对齐论文ImageNet256 fine-tune 1M更新；源数据量/停止规则需明确，不能直接等同。
- [ ] PPF03【实现缺口】原CLI缺匹配gradient accumulation、max-update、本地VAE路径适配；原版和unified尚无完整同合同入口。
- [ ] PPF04【实现缺口】4/5层、FLUX、512分辨率和B从头11M等论文变体未全部公开/纳入，逐分支区分发布限制和工程缺失。
- [ ] PPF05【未验证】真实VAE、原batch packing、任意microbatch以及learn_sigma=False分支；当前micro8适配通过不能认证任意组合。
- [ ] PPF06【未定位/未验证】对应最佳checkpoint、CFG与完整评测；现有Heun2/250点和stage CFG小测不代表最佳质量。
- [ ] PPF07【数值差异】generic Flow与作者stage端点插值的运算顺序不同，371项非逐值相等，最大9.54e-7；包括prediction/gradient/weight，需对齐运算树并验证差异消除。
- [ ] PPF08【未计量】公平实验应计入SiT预训练成本，不能只对齐fine-tune batch/步数就声称总成本相同。

README的`PPF_XL_2`与源码`PPFlow_XL_2`名称不一致；我们的参考采用源码有效名称，不把它误列成当前model名无效。证据：[component_ppflow.json](evidence/strict_original_unified_audit_20261010/component_ppflow.json)、[train.py](sources/ppflow/train.py)。

### 3.7 DCTdiff（13个公开UViT配置）

- [ ] DCT01【发布限制】论文最强DiT分支没有在所检查发布中提供；不能以UViT全部模板覆盖替代该结果。
- [ ] DCT02【已确认接口缺陷】条件ImageNet缺真实label时补造arange；修必需label检查，并测试真实类别映射。
- [ ] DCT03【未验证】原JPEG/数据导出布局、预先resize、split、augmentation和FID reference statistics；特别是导出CIFAR到ImageFolder不能只看图像尺寸。
- [ ] DCT04【未验证】四A100/FP16/Accelerate、worker和完整原CLI生命周期。
- [ ] DCT05【范围限制】13个配置只有缩小训练；历史全架构仅CIFAR与ImageNet64两个，其他分辨率完整网络未全面测试。
- [ ] DCT06【未验证】所有数据集的真实权重、DCT统计/EBFR、DPM或EM完整采样与质量；不同数据集必须使用对应统计，不能复制CIFAR常数。
- [ ] DCT07【验证缺口】两个现存decode是uint8相等，尚未由这些记录证明量化前浮点decode逐像素相同。

当前频率统计/EBFR已转入模板，不列为整项缺失。来源：[公开配置目录](sources/dctdiff/configs)、[train.py](sources/dctdiff/train.py)。

### 3.8 Matryoshka Diffusion Models / MDM（256、1024两个配置）

- [ ] MDM01【已确认RNG偏离】每次setup为导入类构建/丢弃tiny网络，输入全部提供仍额外消耗Torch RNG；移出microbatch绑定路径并比较原生命周期。
- [ ] MDM02【数据/配方差异】CC12M公开YAML不同于发布权重50M Flickr数据及训练curriculum；不能视为同一个best run。
- [ ] MDM03【未定位】最佳global batch、rank topology和完整阶段预算；注释batch字段不是权威配置。
- [ ] MDM04【实现缺口/未验证】256模板未强制原vis_model_256x256.pth初始化；需显式绑定并检验exact权重。
- [ ] MDM05【未验证】1024真实inner256 checkpoint初始化和递进curriculum；旧random测试明确禁用inner初始化。
- [ ] MDM06【未验证】在线Flan-T5-XL/tokenizer、negative text条件、streaming MLX/S3 reader、buffer、mixed-resolution比例、micros和scale conditioning。
- [ ] MDM07【已采用替代】导入使用reader/MLX/S3/torchinfo兼容stub；strict local inner load与source捕获异常fallback策略不完全相同，须明确并对照实际行为。
- [ ] MDM08【说明错误/未验证】所选256源YAML`fp16:0`，旧GPU准备文字写native FP16；逐配置确认真实精度，不可强行统一。
- [ ] MDM09【未验证】真实T5/vision/inner初始化后的完整256/1024推理与最终画质，CFG/eta/resampled steps/clip策略和完整原CLI。
- [ ] MDM10【历史证据缺口】MDM模板后来新增sampler和unconditional配置，旧全架构harness/config hash不能充当当前完整配置验收；需要当前回归。

证据：[mdm_rng.json](evidence/strict_original_unified_audit_20261010/mdm_rng.json)。此前fixture通道问题已修正，见第6节，**不作为当前仍坏的通道错误**。

### 3.9 DiMR（G/2R、XL/2R两个配置）

- [ ] DIM01【未验证】原16rank/2node FP16/Accelerate完整训练与数据worker/RNG；缩小CPU网络不能认证。
- [ ] DIM02【未验证】CompVis SD-VAE真实mean/logvar npy数据、label、posterior与每epoch重采样，不能以其他VAE/在线RGB默认替换。
- [ ] DIM03【未验证】真实pipeline中的Python CFG dropout与DPM guidance scale.4/power-1语义，不能当普通CFG乘法替代。
- [ ] DIM04【未定位】800epoch发布权重与1M更新的确切数据量、更新数量和停止关系；没有核算前不宣称同预算或确定不同最佳run。
- [ ] DIM05【未验证】当前完整DiMR-XL网络；旧完整架构测试只有G，G/XL两种缩小测试不能外推。
- [ ] DIM06【覆盖缺口】各profile有4对requires_grad输出两侧都未反传；解释其分支/是否影响loss，补必要反向覆盖。
- [ ] DIM07【未验证】真实VAE解码后的浮点pixel、最佳G/XL权重、原评测协议与正式FID。

已有global1024、LR2e-4、betas(.99,.99)、wd.03、warmup5k等公开参数已录入；不能把这部分参数录入视为完整生命周期认证。源入口：[train.py](sources/dimr/train.py)。

### 3.10 Improving the Diffusability of Autoencoders（SE2/4组件配置）

- [ ] DIF01【发布限制/未定位】README训练/eval/checkpoint说明仍TODO；最佳完整实验配置、资产、batch/预算未确定。
- [ ] DIF02【已知模板差异】source autoencoder默认KL1、SE0/empty factors；当前SE2/4模板KL0、SE1，LR/freezing/预算不是定位到的best recipe。
- [ ] DIF03【实现缺口】完整Hydra入口所需model/data/loss/teacher配置组合尚未落盘并认证；单codec组件完整不等于全训练入口完整。
- [ ] DIF04【未验证】真实VAE/VGG或其他perceptual variant、encoder冻结/训练策略与全部loss开关的真实初始化。
- [ ] DIF05【实现缺口】GAN、时序SE、tokenizer、distillation-AE、相关regularizer完整分支与训练pipeline未全部纳入。
- [ ] DIF06【未验证】更换/微调codec后的下游生成器重训和质量主表，622组件检查不能代替。

作者默认与我们的SE模板属于不同实验选择；不能把“有差异”写成SE数学组件已全部失配。来源：[作者README](sources/diffusability/README.md)、[autoencoder loss配置](sources/diffusability/configs/loss/autoencoder.yaml)。

### 3.11 RMD（Wan1.3B发布配置）

- [ ] RMD01【模型/预算差异】当前1.3B/global8/300步模板不同于论文Wan14B主实验；完整14B未验收。
- [ ] RMD02【batch差异】release14B示例8rank×micro4×accum2=global64，不等于论文video batch4；不能混成一个配置。
- [ ] RMD03【数据差异】论文JourneyDB prompts与release VidProM2000/synthetic条件不同；绑定同一prompt集合/负条件/text模型。
- [ ] RMD04【说明错误】论文主实验image-free/prompt-only；旧GPU资产项“Real video/prompts”不准确，不能强制真实视频训练集作为复现条件。
- [ ] RMD05【未验证】真实teacher/fake/student初始化、完整Wan latent网格和warmup后的全模型分支；tiny17步覆盖warmup15，完整1.3B backbone只2步且缩网格，没有通过完整模型warmup。
- [ ] RMD06【未验证】原BF16/FSDP/AMP、双optimizer、完整checkpoint/恢复和真实解码。
- [ ] RMD07【实现缺口/未验证】可选GAN，以及SDXL/PixArt/SD3.5图像backbone等论文分支未完整接入/对照。
- [ ] RMD08【来源待确认】仓库作者归属缺paper作者链接佐证；数值结果只能归于固定发布，不能无条件标官方作者实现。
- [ ] RMD09【未验证】论文主表质量/推理加速以及完整native CLI对unified CLI。

曾单独执行source实际train()两次tiny CPU更新，不应说“从未执行任何作者训练函数”；但它不认证最佳完整CLI。来源：[发布YAML](sources/rmd/configs/train_wan_1_3b.yaml)、[作者归属有限制的源README](sources/rmd/README.md)。

### 3.12 SPEED（推理专用）

- [ ] SPD01【范围/实现缺口】当前主要tiny FLUX；source PixelGen和Wan视频完整pipeline未全部统一/对照。
- [ ] SPD02【未验证】真实pretrained backbone、text/tokenizer、VAE、输入prompt和原主表采样配置。
- [ ] SPD03【未验证】真实DCT/Haar/FFT采样、CFG/NFE、完整decode浮点pixel及视频输出。
- [ ] SPD04【未验证】正式质量和GPU速度/显存，tiny随机VAE相等不能代替。
- [ ] SPD05【发布/任务边界，2026-10-12更正】固定官方代码只发布training-free推理，但论文第4.3节明确包含频谱变换微调目标。该微调配方尚未实现并认证；不把现有推理通过写成training/backward parity通过。

### 3.13 NAMI（推理专用）

- [ ] NAM01【发布限制】论文有训练batch768信息，但所检查发布只有推理，不能据此恢复完整训练配方/作者trainer。
- [ ] NAM02【未验证】真实训练过的progressive Transformer和learned bridge权重；random bridge不替代。
- [ ] NAM03【未验证】vendored Diffusers版本、真实text encoders/tokenizer/VAE和原prompt条件；不能用全局安装Diffusers或普通FLUX自动认证。
- [ ] NAM04【未验证】全部阶段、原采样步数和真实pixel/最佳FID；当前每stage3步、合成embedding仅tiny推理对照。
- [ ] NAM05【发布/任务边界】没有作者训练/backward parity证据；若只验收公开推理应明确这一范围。

### 3.14 MrFlow（推理专用）

- [ ] MRF01【实现缺口】当前FLUX工作流不认证Qwen完整路线；Qwen分支需单独模型、接口和对照。
- [ ] MRF02【未验证】真实FLUX/Qwen、T5/tokenizer、VAE和RRDB/RealESRGAN-SR资产及原prompt。
- [ ] MRF03【未验证】真实分辨率下VAE→PIL uint8→tile SR→resize→posterior→refinement的完整chain、sigma、步数和随机流。
- [ ] MRF04【未验证】正式质量、速度、显存，当前patched asset loader下tiny真实workflow通过不替代。
- [ ] MRF05【任务边界】公开training-free推理没有训练/backward对照，不能把推理parity写成训练复现。

三种推理方法现有报告：[SPEED](evidence/paired_experiments_20261010/auxiliary/inference/speed_inference.json)、[NAMI](evidence/paired_experiments_20261010/auxiliary/inference/nami_inference.json)、[MrFlow](evidence/paired_experiments_20261010/auxiliary/inference/mrflow_inference.json)。

### 3.15 本地RDD（L、XL两个配置）

- [ ] RDD01【已确认控制差异】历史8×32与fair world1/micro16/accum16不同，400k公共预算不等于历史L492000/XL407600；不能宣称历史轨迹相同。
- [ ] RDD02【已确认入口遗漏】original参考没有`--data-source imagenet`，source默认hf，与共同ImageFolder读入不同；应按实际绑定明确选择。
- [ ] RDD03【未验证】真实ImageNet label、ADM变换、准确SD-VAE-ema、在线posterior与原rank-local随机流。
- [ ] RDD04【未验证】当前完整RDD-L；旧完整架构只有XL随机网络，L/XL缩小短测不替代。
- [ ] RDD05【未验证】所有扩展REPA/minSNR/loss-aware/其他native CLI可选分支；六种历史目标变体不等于所有开关组合已测。
- [ ] RDD06【数值差异】generic DDPM loss reduction和FM clean代数重排约4.77e-7；需要运算顺序对齐/差异消除验证。
- [ ] RDD07【资产/恢复缺口】历史数据/权重不是全部迁移；已删除GPU验证payload只有metadata，不能用于恢复训练。
- [ ] RDD08【未验证】完整原CLI与unified CLI的真实初始化、训练/采样/新进程resume及正式同预算质量。

证据：[component_rdd.json](evidence/strict_original_unified_audit_20261010/component_rdd.json)、[global256小对照](evidence/paired_experiments_20261010/full_batch/rdd_unified_xl_300_300_400.json)。

## 4. 已测但不严格相同的数值/API，以及未验证像素

| 项目 | 已知差异/限制 | 尚需完成 |
|---|---|---|
| Relay generic DCT | 84检查中2项非逐值相等，max1.6689301e-6 | 对齐DCT运算顺序，报告首次差异及修补后对照 |
| PPFlow generic | 484中371项非逐值相等，max9.5367432e-7；包含小训练prediction/grad/weight | 保持作者stage插值/速度运算树，再证明差异消除 |
| LapFlow reconstruction identity | 715中2项非逐值相等，max2.3841858e-7 | 明确有限精度分解/重构误差，不能混写为native训练失败 |
| RDD generic | 412中4项非逐值相等，max4.7683716e-7 | 对齐loss reduction和clean重构运算顺序并回归 |
| PixelFlow输出API | NumPy ndarray vs Tensor，已测数值/dtype相同 | 明确/统一API适配，并保持浮点pixel对照 |
| DCTdiff终态decode | 两个保存decode是uint8 | 另存、比较量化前浮点decode |
| DFM/DiMR/LapFlow/RDD现存采样 | 部分记录止于latent | 接准确真实VAE后比较浮点pixel与后处理 |
| Relay现存采样 | 小8/16尺寸随机pixel | 官方64→256最佳权重cascade采样/像素 |
| SPEED/NAMI/MrFlow现有decode | tiny/random VAE、合成/替代资产 | 真实pretrained/text/VAE/bridge/SR全chain |
| checkpoint容器 | 文件SHA可受ZIP文件名/metadata影响，状态比较未全部字节级 | 分别认证状态张量、完整param group和文件/API语义，不混为同一个bitwise结论 |

**当前没有做过“修改原论文小地方后，全部上述非零差都消失”的统一验证**。已有原因分析不能当作该实验已完成。

## 5. 尚无作者数值复现的文献，以及本次未覆盖的视频工程

| 文献 | 当前登记状态 | 未完成事项 |
|---|---|---|
| f-DM | 仅数学映射，未确认可执行官方实现 | 重新检查发布；若可获得源码，再审入口/配方并逐组件对照 |
| Edify Image | 未确认技术报告对应可执行官方实现 | 同上；不能用其他开源模型代替该论文认证 |
| Pyramidal Denoising Diffusion Probabilistic Models | 未确认可执行官方实现 | 发布/源码与配方确认、forward/backward/采样对照 |
| From Sketch to Fresco | 已登记下载仓库只有README/LICENSE | 重新确认是否新增实现；此前没有作者数值对照 |
| AViTS | 已登记仓库为README-only | 重新确认发布；训练/推理实现与数值对照未做 |
| DDiT | 未确认可执行官方实现 | 发布确认与作者数值对照 |
| CAT | 未确认可执行官方实现 | 发布确认与作者数值对照 |
| LoT | 前次登记项目无可执行代码链接 | 重新确认新发布；作者数值对照未做 |

这些“未确认”来自已有源登记，**严格审计这一轮没有逐个重新查询最新发布**，不等于永久没有开源。

Pyramid Flow/PyramidalWan及其FastVideo接入不在此次33份配置中，不能继承此处的image核心parity结论。后续仍需依据[项目交接](../multiscale_diffusion/docs/PROJECT_HANDOFF_ZH.md)分别核对官方native、论文实现、controlled transplant、FastVideo适配和真实视频训练/采样/质量；本清单不声称已经重新完成那一部分审查。

## 6. 历史报告、环境和恢复证据的缺口

### 6.1 不应误列为当前未修代码错误的历史事项

- 初次整套paper检查保留MDM fixture通道错误的failed SUMMARY；最终根索引引用修正后的mdm_final通过报告。**通道问题已有修正，整套旧SUMMARY的failed是历史记录**；修复后没有把30项整套全部重跑并覆盖成passed。
- DFM source NumPy debug draw已保留；DCT频率统计/EBFR已转入；SSD source全局batch64元信息已修正。这些已补项不能重复说成尚未实现。
- 近期捕获的专用网络模块值/梯度零差是有效范围证据；不能因其他入口存在差异就把它们全部否定，也不能将其外推到完整原CLI。

### 6.2 仍未补的当前完整架构回归

- 16份历史完整image架构报告中，14份所存harness SHA不等于当前版本；DCTdiff/SSD两份没有存harness SHA。不是当前harness全量运行证明。
- MDM256/1024模板后来加入CFG/eta/resampling/clip/unconditional；SSD后来修正batch与备注，共3份模板hash变化；`AUDIT.json`有逐字段delta。
- 完整DiMR-XL、RDD-L、全部DCT数据集完整模型、新真实初始化与完整后处理没有全面重新执行。
- 完整Wan1.3B backbone历史只2步、缩小视频网格，不认证warmup后全模型分支或官方原始latent尺寸。
- 新近训练终态payload按用户要求已清理，不能现在从它们重新加载逐tensor比对；保留报告/hash/配置。需要复跑时重新生成必要临时payload，验收后再清理。

### 6.3 依赖环境与临时产物

此次审计读取的安装版本：Torch2.12.0+cu130、torchvision0.27.0+cu130、diffusers0.40.0、transformers5.15.1、accelerate1.0.1、timm1.0.30、NumPy2.5.3、flash-attn2.8.4。DCT README示例Torch2.4/torchvision.19/accelerate.33/transformers4.23.1；PixelFlow示例Torch2.6/Flash2.7.4。**未复刻同一作者环境**。NAMI实际使用vendored Diffusers，不能用全局版本号替代其源版本。

原17库hash只覆盖选定1211文件，未以完整Git对象、全部未选文件和真实权重hash认证整个原仓库。此前已清理的GPU检查checkpoint不能恢复；正式训练成果、用户数据和原版权重不能当临时验证payload删除。新临时验证产物在报告/配置/hash保留和数值/恢复检查完成后清理。

## 7. 完成标准与执行顺序

1. **修实际错误**：Relay增强/augment_dim/EMA/缩放；LapFlow、DFM及RDD入口；完整resolved配置；条件label guard；MDM setup RNG；drop_last和严格byte comparator。
2. **完成原配方定位**：每个论文结果行绑定数据/split、网络/VAE/初始化、优化器、精度/拓扑、完整预算和评估；不可得项保持发布限制。
3. **补CPU完整生命周期**：真实本地小数据、样本顺序、posterior/teacher/text、raw grad、参数更新、callback、保存及新进程resume；逐配置比较真正原CLI和unified CLI。
4. **补采样/像素**：准确solver/CFG/边界与真实资产的浮点decode，对API/后处理差异单独验收。
5. **补原生GPU/分布式**：只有重新获授权后执行；独立guard、获准卡号与当时deadline不可省略，不能恢复旧授权。
6. **跑正式research实验**：分别完成原配方复现和公平预算对照，官方质量指标、多seed、成本和必要消融；没有正式结果前不得承诺效果至少与论文相同。

逐配置关闭条件：最终resolved配置可审阅；原版/统一入口都实际加载该配置和同一资产；所选真实小数据上的forward、处理前后backward、更新/状态、采样pixel和恢复有相应范围证据；差异若不能消除必须定位、量化并保留；最佳成绩另需正式训练与评估证明。

下方附录逐项保留33份配置的已知控制项和入口阻塞，避免只看方法名遗漏某个profile。

## 附录A：33份paper模式配置的控制项与完整入口阻塞

以下直接从严格审计保存的`AUDIT.json`提取，避免手工遗漏某份配置。列出的控制项是准备的**共同源码核心合同**，不是官方最佳拓扑/配方认证。全部world_size=1；最佳未知保留“未知”。`required_assets`只表示配置占位符提取结果，空列表不表示该论文无需预训练/VAE/teacher等真实资产；数据内嵌条件和其他必要资产仍须按正文绑定。

| 配置ID | 方法 | global batch | micro × accumulation | 预算 | seed | 精度合同 | 原CLI认证 |
|---|---|---:|---|---|---|---|---|
| dctdiff_afhq512_uvit_mid_8by8 | dctdiff | 128 | 1 × 128 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_celeba64_uvit_small_2by2 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_cifar10_uvit_mid_2by2 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_cifar10_uvit_mid_deep_2by2 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_cifar10_uvit_small_2by2 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_ffhq128_uvit_mid_4by4 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_ffhq128_uvit_mid_deep_4by4 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_ffhq128_uvit_small_4by4 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_ffhq256_uvit_mid_4by4 | dctdiff | 256 | 1 × 256 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_ffhq512_uvit_mid_8by8 | dctdiff | 128 | 1 × 128 | 500000 | 1234 | float16 | 未认证 |
| dctdiff_imgnet64_uvit_mid_2by2 | dctdiff | 1024 | 1 × 1024 | 500000 | 123456 | float16 | 未认证 |
| dctdiff_imgnet64_uvit_mid_deep_2by2 | dctdiff | 1024 | 1 × 1024 | 500000 | 123456 | float16 | 未认证 |
| dctdiff_imgnet64_uvit_small_2by2 | dctdiff | 1024 | 1 × 1024 | 800000 | 123456 | float16 | 未认证 |
| dimr_DiMR-G-2R_imagenet256 | dimr | 1024 | 1 × 1024 | 1000000 | 1234 | float16 | 未认证 |
| dimr_DiMR-XL-2R_imagenet256 | dimr | 1024 | 1 × 1024 | 1000000 | 1234 | float16 | 未认证 |
| pixelflow_xl_c2i | pixelflow | 未知 | 未知 × 未知 | 10 epochs | 42 | bfloat16 | 未认证 |
| ppflow_xl_2 | ppflow | 256 | 8 × 32 | 1400 epochs | 0 | 由默认/具体实现决定 | 未认证 |
| ppflow_xl_3 | ppflow | 256 | 8 × 32 | 1400 epochs | 0 | 由默认/具体实现决定 | 未认证 |
| dfm_sit_xl | dfm | 256 | 1 × 256 | 400000 | 0 | float16 | 未认证 |
| dfm_sit_xl_repa_localfeatures | dfm | 256 | 1 × 256 | 400000 | 0 | float16 | 未认证 |
| dfm_sit_b_released_yaml | dfm | 256 | 1 × 256 | 400000 | 0 | bfloat16 | 未认证 |
| lapflow_mot_xl_2 | lapflow | 未知 | 未知 × 未知 | 未知 | 1024 | 由默认/具体实现决定 | 未认证 |
| lapflow_imagenet_xl_2_paper | lapflow | 256 | 1 × 256 | 120 epochs | 1024 | 由默认/具体实现决定 | 未认证 |
| ssd_celeba64 | ssd | 64 | 1 × 64 | 未知 | 未知 | 由默认/具体实现决定 | 未认证 |
| mdm_cc12m256 | mdm | 未知 | 未知 × 未知 | 1000000 | 未知 | 由默认/具体实现决定 | 未认证 |
| mdm_cc12m1024 | mdm | 未知 | 未知 × 未知 | 1000000 | 未知 | float16 | 未认证 |
| rdd_unified_xl_300_300_400 | rdd | 未知 | 未知 × 未知 | 未知 | 0 | 由默认/具体实现决定 | 未认证 |
| diffusability_se24 | diffusability | 未知 | 未知 × 未知 | 未知 | 未知 | 由默认/具体实现决定 | 未认证 |
| relay_imagenet_low | relay | 4096 | 1 × 4096 | 未知 | 未知 | 由默认/具体实现决定 | 未认证 |
| relay_imagenet_high | relay | 4096 | 1 × 4096 | 未知 | 未知 | 由默认/具体实现决定 | 未认证 |
| rmd_wan13b | rmd | 8 | 1 × 8 | 300 | 未知 | 由默认/具体实现决定 | 未认证 |
| rdd_unified_l_300_300_400 | rdd | 未知 | 未知 × 未知 | 未知 | 0 | 由默认/具体实现决定 | 未认证 |
| pixelflow_xl_c2i_paper_dopri5 | pixelflow | 未知 | 未知 × 未知 | 10 epochs | 42 | bfloat16 | 未认证 |

### 每份配置仍待解除的入口阻塞

下面保留审计机器记录的英文原文；中文问题与验收要求已在第2、3节展开。共同核心“无阻塞”仅表示该受控入口没有这类已登记阻塞，**不表示官方CLI已认证或正式质量已复现**。

#### dctdiff_afhq512_uvit_mid_8by8

- 原配置文件：[`dctdiff_afhq512_uvit_mid_8by8.json`](configs/paired_experiments/dctdiff_afhq512_uvit_mid_8by8.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_celeba64_uvit_small_2by2

- 原配置文件：[`dctdiff_celeba64_uvit_small_2by2.json`](configs/paired_experiments/dctdiff_celeba64_uvit_small_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_cifar10_uvit_mid_2by2

- 原配置文件：[`dctdiff_cifar10_uvit_mid_2by2.json`](configs/paired_experiments/dctdiff_cifar10_uvit_mid_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_cifar10_uvit_mid_deep_2by2

- 原配置文件：[`dctdiff_cifar10_uvit_mid_deep_2by2.json`](configs/paired_experiments/dctdiff_cifar10_uvit_mid_deep_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_cifar10_uvit_small_2by2

- 原配置文件：[`dctdiff_cifar10_uvit_small_2by2.json`](configs/paired_experiments/dctdiff_cifar10_uvit_small_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_ffhq128_uvit_mid_4by4

- 原配置文件：[`dctdiff_ffhq128_uvit_mid_4by4.json`](configs/paired_experiments/dctdiff_ffhq128_uvit_mid_4by4.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_ffhq128_uvit_mid_deep_4by4

- 原配置文件：[`dctdiff_ffhq128_uvit_mid_deep_4by4.json`](configs/paired_experiments/dctdiff_ffhq128_uvit_mid_deep_4by4.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_ffhq128_uvit_small_4by4

- 原配置文件：[`dctdiff_ffhq128_uvit_small_4by4.json`](configs/paired_experiments/dctdiff_ffhq128_uvit_small_4by4.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_ffhq256_uvit_mid_4by4

- 原配置文件：[`dctdiff_ffhq256_uvit_mid_4by4.json`](configs/paired_experiments/dctdiff_ffhq256_uvit_mid_4by4.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_ffhq512_uvit_mid_8by8

- 原配置文件：[`dctdiff_ffhq512_uvit_mid_8by8.json`](configs/paired_experiments/dctdiff_ffhq512_uvit_mid_8by8.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_imgnet64_uvit_mid_2by2

- 原配置文件：[`dctdiff_imgnet64_uvit_mid_2by2.json`](configs/paired_experiments/dctdiff_imgnet64_uvit_mid_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_imgnet64_uvit_mid_deep_2by2

- 原配置文件：[`dctdiff_imgnet64_uvit_mid_deep_2by2.json`](configs/paired_experiments/dctdiff_imgnet64_uvit_mid_deep_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dctdiff_imgnet64_uvit_small_2by2

- 原配置文件：[`dctdiff_imgnet64_uvit_small_2by2.json`](configs/paired_experiments/dctdiff_imgnet64_uvit_small_2by2.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dimr_DiMR-G-2R_imagenet256

- 原配置文件：[`dimr_DiMR-G-2R_imagenet256.json`](configs/paired_experiments/dimr_DiMR-G-2R_imagenet256.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dimr_DiMR-XL-2R_imagenet256

- 原配置文件：[`dimr_DiMR-XL-2R_imagenet256.json`](configs/paired_experiments/dimr_DiMR-XL-2R_imagenet256.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### pixelflow_xl_c2i

- 原配置文件：[`pixelflow_xl_c2i.json`](configs/paired_experiments/pixelflow_xl_c2i.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Requires a source YAML overlay; epoch-only source has no matching accumulation/max-update controls; best training budget is unknown.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### ppflow_xl_2

- 原配置文件：[`ppflow_xl_2.json`](configs/paired_experiments/ppflow_xl_2.json)。
- 必须绑定的配置资产占位符：`sit`、`vae`
- 共同源码核心入口阻塞：
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Author CLI has no accumulation/max-update/local-VAE-path flags matching this contract.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### ppflow_xl_3

- 原配置文件：[`ppflow_xl_3.json`](configs/paired_experiments/ppflow_xl_3.json)。
- 必须绑定的配置资产占位符：`sit`、`vae`
- 共同源码核心入口阻塞：
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Author CLI has no accumulation/max-update/local-VAE-path flags matching this contract.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dfm_sit_xl

- 原配置文件：[`dfm_sit_xl.json`](configs/paired_experiments/dfm_sit_xl.json)。
- 必须绑定的配置资产占位符：`vae`
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Select the exact SiT config, VAE/teacher/latent data and source Accelerate lifecycle; original paper FLUX/DiT branch differs.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dfm_sit_xl_repa_localfeatures

- 原配置文件：[`dfm_sit_xl_repa_localfeatures.json`](configs/paired_experiments/dfm_sit_xl_repa_localfeatures.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - REPA source-core frontend/teacher pipeline not certified by this launcher.
- 原作者完整CLI与unified CLI配对阻塞：
  - Select the exact SiT config, VAE/teacher/latent data and source Accelerate lifecycle; original paper FLUX/DiT branch differs.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### dfm_sit_b_released_yaml

- 原配置文件：[`dfm_sit_b_released_yaml.json`](configs/paired_experiments/dfm_sit_b_released_yaml.json)。
- 必须绑定的配置资产占位符：`vae`
- 共同源码核心入口阻塞：
  - 无此类已登记阻塞；仅共同源码核心范围。
- 原作者完整CLI与unified CLI配对阻塞：
  - Select the exact SiT config, VAE/teacher/latent data and source Accelerate lifecycle; original paper FLUX/DiT branch differs.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### lapflow_mot_xl_2

- 原配置文件：[`lapflow_mot_xl_2.json`](configs/paired_experiments/lapflow_mot_xl_2.json)。
- 必须绑定的配置资产占位符：`vae`
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Add exact paper path/stage flags; source epoch-only trainer lacks matched accumulation/max-update controls and best CFG is unidentified.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### lapflow_imagenet_xl_2_paper

- 原配置文件：[`lapflow_imagenet_xl_2_paper.json`](configs/paired_experiments/lapflow_imagenet_xl_2_paper.json)。
- 必须绑定的配置资产占位符：`eqvae`
- 共同源码核心入口阻塞：
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Add exact paper path/stage flags; source epoch-only trainer lacks matched accumulation/max-update controls and best CFG is unidentified.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### ssd_celeba64

- 原配置文件：[`ssd_celeba64.json`](configs/paired_experiments/ssd_celeba64.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
  - Author best-run seed not identified; do not silently replace it with0.
- 原作者完整CLI与unified CLI配对阻塞：
  - Native Hydra train loop has no independent exact update cap; lr_anneal_steps would change LR.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### mdm_cc12m256

- 原配置文件：[`mdm_cc12m256.json`](configs/paired_experiments/mdm_cc12m256.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Author best-run seed not identified; do not silently replace it with0.
- 原作者完整CLI与unified CLI配对阻塞：
  - Original streaming/T5/vision curriculum/data/global batch must be recovered; CC12M is not the released Flickr training run.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### mdm_cc12m1024

- 原配置文件：[`mdm_cc12m1024.json`](configs/paired_experiments/mdm_cc12m1024.json)。
- 必须绑定的配置资产占位符：`mdm_inner256`
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Author best-run seed not identified; do not silently replace it with0.
- 原作者完整CLI与unified CLI配对阻塞：
  - Original streaming/T5/vision curriculum/data/global batch must be recovered; CC12M is not the released Flickr training run.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### rdd_unified_xl_300_300_400

- 原配置文件：[`rdd_unified_xl_300_300_400.json`](configs/paired_experiments/rdd_unified_xl_300_300_400.json)。
- 必须绑定的配置资产占位符：`vae`
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Historical epochs do not identify an exact continuation/update budget.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### diffusability_se24

- 原配置文件：[`diffusability_se24.json`](configs/paired_experiments/diffusability_se24.json)。
- 必须绑定的配置资产占位符：`vae`、`vgg`
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
  - Author best-run seed not identified; do not silently replace it with0.
  - Requires a separate codec/distillation or full pretrained inference pipeline; common scratch trainer is inapplicable.
- 原作者完整CLI与unified CLI配对阻塞：
  - Best released full pipeline/config is not identified.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### relay_imagenet_low

- 原配置文件：[`relay_imagenet_low.json`](configs/paired_experiments/relay_imagenet_low.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
  - Author best-run seed not identified; do not silently replace it with0.
- 原作者完整CLI与unified CLI配对阻塞：
  - Exact cascade allocation/image-clock/EMA/loss-scale/architecture flags and native ZIP label reader required.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### relay_imagenet_high

- 原配置文件：[`relay_imagenet_high.json`](configs/paired_experiments/relay_imagenet_high.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
  - Author best-run seed not identified; do not silently replace it with0.
- 原作者完整CLI与unified CLI配对阻塞：
  - Exact cascade allocation/image-clock/EMA/loss-scale/architecture flags and native ZIP label reader required.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### rmd_wan13b

- 原配置文件：[`rmd_wan13b.json`](configs/paired_experiments/rmd_wan13b.json)。
- 必须绑定的配置资产占位符：`wan`
- 共同源码核心入口阻塞：
  - Author best-run seed not identified; do not silently replace it with0.
  - Requires a separate codec/distillation or full pretrained inference pipeline; common scratch trainer is inapplicable.
- 原作者完整CLI与unified CLI配对阻塞：
  - Teacher/student/fake/FSDP/local pretrained assets and paper14B main-table settings are not source-core scratch training.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### rdd_unified_l_300_300_400

- 原配置文件：[`rdd_unified_l_300_300_400.json`](configs/paired_experiments/rdd_unified_l_300_300_400.json)。
- 必须绑定的配置资产占位符：`vae`
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Historical epochs do not identify an exact continuation/update budget.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

#### pixelflow_xl_c2i_paper_dopri5

- 原配置文件：[`pixelflow_xl_c2i_paper_dopri5.json`](configs/paired_experiments/pixelflow_xl_c2i_paper_dopri5.json)。
- 必须绑定的配置资产占位符：无登记占位符；仍须检查真实数据/条件/初始化资产。
- 共同源码核心入口阻塞：
  - Released/paper global batch not identified.
  - Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.
- 原作者完整CLI与unified CLI配对阻塞：
  - Requires a source YAML overlay; epoch-only source has no matching accumulation/max-update controls; best training budget is unknown.
  - Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.

## 附录B：文档证据身份

- 提取源：`evidence/strict_original_unified_audit_20261010/AUDIT.json`。
- 提取时SHA256：`1d8b6d51e1eb2180e5659ff805af1ffa6208b95cfad6910ab04615c9f3139968`。
- 配置数量：33；原CLI认证数：0。
- 此文档整理时没有执行训练、下载资产或修改算法；关闭事项须提交新代码/配置和新范围证据，不能只修改勾选框。
