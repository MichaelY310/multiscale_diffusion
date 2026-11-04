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

# 2026-10-12：原清单148项逐项处理记录

> **用户已调整项目验收范围**：优先统一核心数学与主要训练/推理算法，允许共同训练工程；不要求每篇的专有CLI、环境、数据服务和全部特殊分支完全契合。148项保留为严格复现审计，不再全部作为项目完成门槛。后续按所选公开配置区分“核心已融入并验证”“可选或排除分支”“质量实验未执行”；必要teacher/codec及改变目标分布的差异不能静默忽略。

> **后续代码变更后的验收边界**：下表及原 full-backbone 证书保留其运行时范围，不自动认证后来新增的 RDD loss weighting/REPA/adaptive sampler 和 checkpoint 状态。新增 CPU worker、真实 Gloo DDP、CPU GradScaler、部分作者入口及新进程恢复证据在 `evidence/nontraining_closure_20261012/`。最新31配置恢复矩阵为30通过、1次状态字段差异；PixelFlow 的差异仅为测试过程中新增的两个空 checkpoint 字段，固定代码单独重跑已通过，原失败矩阵保留。当前27项组件测试通过；全部方法在固定最终代码下的回归及新增 RDD 分支仍待完成，不能宣称全部验收结束。详见 [当前验收审计](evidence/nontraining_closure_20261012/CURRENT_REPRODUCTION_GATE_AUDIT.json)。

此表区分已修复的CPU错误、已经实测的限定范围和仍缺少外部前提的验收。**148项并未全部关闭**；CUDA、多卡、私有代码、真实资产和正式主表实验没有被合成CPU测试代替。全部实测矩阵见 [完成报告](CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md)。

| 编号 | 本次状态 | 已做与保留范围 |
|---|---|---|
| C01 | partial_cpu | 已直接执行作者 forward/loss 和原 backward 语句；没有调用共同 Trainer。完整官方 CLI 的数据、保存、评估生命周期仍未认证。 |
| C02 | requires_assets_or_formal_run | 33 个完整 native CLI 认证仍为 false；本报告认证的是源码核心逐层对照，不篡改 CLI 认证字段。 |
| C03 | partial_cpu | 修正 DFM-SiT 模型/exp-name、LapFlow 全部选中 flags、RDD ImageNet、Relay flags；实际执行 DFM/LapFlow/RDD parser。完整资产/lifecycle overlay 不等同于 parser 成功。 |
| C04 | fixed_and_verified_cpu | 两个 backend 的 33 份最终 resolved 配置已落盘，并验证入口引用及实际解析控制项。 |
| C05 | requires_cuda_distributed | CPU 对照使用 micro1/2/3/4/8，逐报告记录；未冒充官方 8/16 rank 或全局 batch 的轨迹。 |
| C06 | boundary_acknowledged | paper 源码合同和 fair-rdd 新实验仍分别保存；公共 AdamW 128 步检查明确标为组件回归。 |
| C07 | requires_assets_or_formal_run | 不能从没有发布的 best-run 日志推测 seed/batch/预算；未知项保留。 |
| C08 | partial_cpu | 修复 drop_last、epoch 尾部和恢复游标；实际两个 production reader 索引片段对齐单 rank 官方 DataLoader 的不足整 batch 样本顺序。 |
| C09 | partial_cpu | 单 rank shuffle/3 个 epoch/恢复下一批已逐样本比对；真实多 worker、shard、rank 分配未认证。 |
| C10 | partial_cpu | 独立执行 DCTdiff 13 配置作者 PNG loader、flip、编码和真实类别字段；其他真实数据布局/统计/reader 未完整端到端运行。 |
| C11 | fixed_and_verified_cpu | 真实条件输入缺 label、错误 dtype/shape/range 均拒绝；合法多个类别通过。仅显式 synthetic 模式生成夹具 label。 |
| C12 | requires_assets_or_formal_run | 各核心测试直接使用同一模型域输入；没有声称真实 VAE rank-local posterior 编码生命周期相同。 |
| C13 | requires_assets_or_formal_run | 源码、架构 JSON、配置和 harness 已绑定 hash；没有取得的 pretrained VAE/teacher/text/SR 内容 hash 不能补造。 |
| C14 | boundary_acknowledged | 两侧共用作者 backbone 实现，独立组装路径/目标/loss/backward；明确是 integration parity，没有声称独立重写网络架构。 |
| C15 | boundary_acknowledged | CPU attention、AST 提取、接口适配和导入 stub 都在报告/source_executions 中披露；原 CUDA kernel 单列。 |
| C16 | requires_assets_or_formal_run | 复用只读现有环境，记录依赖版本；未安装/改造官方旧版环境，更未修改 dirty FastVideo。 |
| C17 | fixed_and_verified_cpu | 比较 shape、dtype 和 uint8 原始字节；negative controls 检查 signed zero、dtype、NaN/Inf。 |
| C18 | partial_cpu | 128 步loss/原始grad/权重/EMA逐字节相同；重读39对临时checkpoint，完整model/EMA/Adam状态、全部param_groups及typed scalars未过滤即相同。原正式CLI/scaler/文件序列化整体仍未认证。 |
| C19 | fixed_and_verified_cpu | 在任何 clip/repair/optimizer 操作前捕获参数 grad 和模块输出的 activation grad；没有用 nan_to_num 掩盖差异。CUDA unscale 另属 C29。 |
| C20 | partial_cpu | 导出每个执行模块、重复调用、输出 Tensor 路径、None-grad 原因和关键 functional 组件对照；没有声称捕获全部 ATen 操作。 |
| C21 | fixed_and_verified_cpu | 明确区分 ModuleList 等容器与未执行实体层；DiMR 相关 view 来自 checkpoint 首次 no_grad pass，重算路径实际捕获梯度。None 从未计作零梯度通过。 |
| C22 | partial_cpu | 33 份选中配置均完成当前 full-backbone 短测；合成模型域数据/随机共用非零权重、CPU FP32、RMD 缩网格等限制逐项保留。 |
| C23 | fixed_and_verified_cpu | 修补后 31 配置缩小网络、31 配置完整网络、12 方法各128步、12扩展各128步全部重新执行并做严格 byte gate。 |
| C24 | fixed_and_verified_cpu | 当前完整架构覆盖全部13 DCT、MDM256/1024、DiMR G/XL、RDD L/XL 等31配置，加完整 codec 和30层 Wan。旧16份历史 hash 不再承担本次结论。 |
| C25 | partial_cpu | 修 MDM setup RNG、FP32 累积 backward 次序、Relay 增强/loss 汇总、RMD warmup；真实初始化/callback/eval/原 AMP 完整生命周期未认证。 |
| C26 | requires_assets_or_formal_run | 临时随机检查 checkpoint 已验状态并清理；没有把它们当作各篇真实官方 CLI 的新进程 resume 证明。 |
| C27 | partial_cpu | DCT 13 配置量化前 float decode、PixelFlow 浮点 API、三种 inference workflow 的浮点输出已比对；真实 pretrained 完整像素/质量仍缺资产。 |
| C28 | partial_cpu | PixelFlow 新 ndarray BHWC/float32 adapter 经5种 Euler/Dopri5/CFG/shift 配置测试；NAMI bridge 显式接口映射。所有原文件 metadata 未统一认证。 |
| C29 | requires_cuda_distributed | 本次没有 GPU 授权且仅用 CPU；CUDA AMP/scaler/Flash/xformers/TF32 不能用 CPU 检查替代。 |
| C30 | requires_cuda_distributed | 原 8/16 卡 DDP/FSDP/all-reduce/rank RNG 未运行。 |
| C31 | requires_assets_or_formal_run | 没有启动百万步正式训练，没有重新训练所有论文主表和多 seed；forward/backward 零差本身不证明最佳 FID。 |
| C32 | requires_assets_or_formal_run | 没有下载真实数据/参考统计或执行50k FID/IS/视频主表评测。 |
| C33 | requires_cuda_distributed | CPU 报告只记录检查耗时；不把它称为 GPU 速度、显存或 FLOPs/质量比较。 |
| C34 | boundary_acknowledged | 17 源快照1211个选定文件重新核 hash；CAT补充4个源码另有 hash清单。没有声称全部未知/未发布分支被审完。 |
| C35 | fixed_and_verified_cpu | MDM256 原 FP32/1024 原 FP16 分开标注；RMD 为 prompt-only/image-free，模板元信息已修正。 |
| C36 | requires_cuda_distributed | 未来 GPU 计划不是执行证据；本次 CPU runner 可执行，未认证原 CUDA runner。 |
| C37 | fixed_and_verified_cpu | 33 profile 逐项绑定实际数据集/分辨率/任务；不再使用整篇论文数据集汇总替代 profile 数据。 |
| REL01 | fixed_and_verified_cpu | 恢复实际 AugmentPipe p=.12/原随机 draw 次序，包含完整网络当前对照。 |
| REL02 | fixed_and_verified_cpu | 两配置 augment_dim=9，当前完整 backbone/layer/grad 对照通过。 |
| REL03 | partial_cpu | 修 high EMA half-life 为500000、low为50M；完整真实多 rank image clock/rampup 另需验证。 |
| REL04 | partial_cpu | CPU microbatch 直接执行作者 sum(loss)*loss_multiplier.backward，low25/32、high1/256；原8rank/FP16 loss scale 未执行。 |
| REL05 | partial_cpu | 参考 CLI 已补原选中 flags；未加载真实 ZIP/权重启动完整8rank CLI。 |
| REL06 | requires_assets_or_formal_run | CPU AugmentPipe 已测；真实 ZIP/dataset.json reader/label/data order 尚缺同数据完整 CLI 对照。 |
| REL07 | requires_assets_or_formal_run | 最佳阶段预算/low-high checkpoint 对尚无作者可绑定 run 记录。 |
| REL08 | requires_assets_or_formal_run | 随机小 cascade 回归通过；真实64→256最佳权重 cascade 未测。 |
| REL09 | requires_cuda_distributed | 原 xformers/FP16/8rank training lifecycle 未执行。 |
| REL10 | fixed_and_verified_cpu | FFT DCT 运算树对齐；84/84组件检查 byte exact，max_abs=0。 |
| SSD01 | partial_cpu | 当前所选完整 CelebA64 四层架构及全部7条训练分辨率路径通过；其他论文主表配置仍未形成最佳训练结果。 |
| SSD02 | requires_assets_or_formal_run | 最佳 checkpoint/停止预算与评测行未定位到正式 run。 |
| SSD03 | requires_assets_or_formal_run | 元信息保留官方 split 要求；没有真实 CelebA split 数据验证。 |
| SSD04 | partial_cpu | 受控 runner 独立 steps 上限已可用；作者无限循环 Hydra 原入口尚未建立正式 stop-only adapter。 |
| SSD05 | boundary_acknowledged | CPU H_out/W_out 属性 shim 与 rank0 依赖明确列出；不是原 Hydra/DDP CLI 的认证。 |
| SSD06 | partial_cpu | 完整64架构/7路径和缩小128步采样通过；未下载其它主表权重/真实数据运行质量评测。 |
| PIX01 | requires_assets_or_formal_run | 公开示例不自动等于 best run；最佳 budget/batch 未补造。 |
| PIX02 | requires_cuda_distributed | 完整31配置含 PixelFlow backbone 和 packed CPU SDPA，原 Flash/BF16 未运行。 |
| PIX03 | boundary_acknowledged | 实际 NumPy RoPE shim 明确记录，未认证作者旧依赖的原 CUDA 环境。 |
| PIX04 | partial_cpu | resolved contracts 已落盘并通过解析；epoch-only 原 CLI 的全部 fair overlay/lifecycle 未认证。 |
| PIX05 | fixed_and_verified_cpu | ndarray BHWC float32 adapter 与原 pipeline 的5个配置逐像素、dtype/API 完全相同。 |
| PIX06 | requires_assets_or_formal_run | 没有真实 pretrained checkpoint/text-to-image完整训练/50kFID。 |
| PIX07 | partial_cpu | 完整随机256模型 forward/backward，Euler/Dopri5 小采样/API/RNG 回归通过；真实最佳权重 sampler 仍无证据。 |
| DFM01 | unavailable_release | 发布 README 的 SiT/REPA reimplementation 与论文 DiT/SE-FLUX 差异保留，未伪造私有架构复现。 |
| DFM02 | boundary_acknowledged | 分别保留公开模板与论文正文 optimizer 配方，不混写为同一 best run。 |
| DFM03 | requires_assets_or_formal_run | 仅已发布256 latent配置完整 backbone；论文512/1024/video主表未正式复现。 |
| DFM04 | unavailable_release | 私有 SE-FLUX 数据/确切 codec/run 未发布，不能取得原 best 资产。 |
| DFM05 | fixed_and_verified_cpu | 补 exp-name 与 DFM-SiT-B/2、DFM-SiT-XL/2，实际 parser/model 类型检查通过；不会误走普通 SiT baseline。 |
| DFM06 | partial_cpu | local teacher_features 的成对入口接通，REPA scalar reduction 与作者 train.py 相同；完整 XL/projector raw grad 通过。真实 DINO/预处理仍缺资产。 |
| DFM07 | requires_assets_or_formal_run | 未以随机latent替代真实预计算VAE mean/std或 SD-VAE-ft-mse。 |
| DFM08 | partial_cpu | 当前实际 loss/backward/阶段两分支/REPA 和128步通过；原 AMP/warmup/full checkpoint/质量未认证。 |
| LAP01 | fixed_and_verified_cpu | dataset 修为 imagenet_256；实际作者 parser 验证。 |
| LAP02 | fixed_and_verified_cpu | 所选完整模型/尺度/通道/mask/T/class/epoch flags 已补齐并核实 parser 结果。 |
| LAP03 | boundary_acknowledged | CLI default 和 paper flags 两配置分别对照，原正式 recipe/FID 仍未认证。 |
| LAP04 | requires_assets_or_formal_run | 未取得真实EQVAE与posterior资产。 |
| LAP05 | requires_assets_or_formal_run | CFG2 明确是测试选择，不宣称最优 FID。 |
| LAP06 | unavailable_release | 发布 square sampler 存在源码调用缺陷；没有静默修作者文件/把失败分支伪报认证。 |
| LAP07 | partial_cpu | 补 native 三层训练+CFG采样128步；仍是缩小网络/合成latent，非真实最佳三层训练实验。 |
| LAP08 | requires_assets_or_formal_run | HQ512/1024主表recipe/完整训练与最佳权重未纳入正式实验。 |
| LAP09 | requires_assets_or_formal_run | 原完整epoch/data/cosine/resume/正式质量未运行。 |
| LAP10 | mathematical_roundoff_explained | 715项中仅两项是原图分解后重构恒等式的FP32舍入2.3841858e-7；native两侧所有逐层值/grad byte exact。 |
| PPF01 | partial_cpu | SiT shape/权重转换小分支回归通过；未取得真实SiT预训练权重进行fine-tune。 |
| PPF02 | requires_assets_or_formal_run | 1400epoch与1M更新正式预算没有据真实数据量等同。 |
| PPF03 | partial_cpu | CPU runner 使用官方packing micro8直接对照；原CLI完整accum/max-update/localVAE overlay未认证。 |
| PPF04 | unavailable_release | 公开2/3层均完整 backbone 和128步通过；未发布FLUX/4-5层及其他论文主结果不能由配置名补造。 |
| PPF05 | partial_cpu | 原2/3层packing与真实backward语句通过；发布learn_sigma=False unpatchify有源码缺陷，任意micro或资产组合不宣称通过。 |
| PPF06 | requires_assets_or_formal_run | 小solver/CFG分支通过不等于最佳checkpoint与完整评测。 |
| PPF07 | fixed_and_verified_cpu | stage端点插值/velocity运算树修正，484/484 byte exact，prediction/grad/weight max_abs=0。 |
| PPF08 | requires_assets_or_formal_run | 未做正式成本计量；记录必须包含SiT pretraining。 |
| DCT01 | unavailable_release | 已发布13个UViT完整网络通过；论文未发布最强DiT分支没有作数值复现。 |
| DCT02 | fixed_and_verified_cpu | 必需真实label guard及作者PNG loader真实类别937已对照。 |
| DCT03 | partial_cpu | 13 profile 作者PNG loader/flip/DCT编码对照完成；真实JPEG导出、split/FID stats未认证。 |
| DCT04 | requires_cuda_distributed | 原4A100/FP16/Accelerate/DataLoader完整CLI未运行。 |
| DCT05 | fixed_and_verified_cpu | 全部13配置均重新运行当前完整UViT架构，逐层forward/activation grad/parameter grad全byte exact。 |
| DCT06 | requires_assets_or_formal_run | 配置统计/EBFR已保留，随机训练/采样回归不等于所有数据集真实权重与正式质量。 |
| DCT07 | fixed_and_verified_cpu | 13 profile作者decode仅去最终uint8 cast暴露float64边界；量化前与uint8两种逐像素bytes相同。 |
| MDM01 | fixed_and_verified_cpu | 拆分仅导入类与构建模型，setup不再构建丢弃网络消耗RNG；256/1024输入全提供时三类RNG保持不变。 |
| MDM02 | unavailable_release | CC12M YAML不等于Flickr50M最佳模型/完整curriculum；公开资料缺口保留。 |
| MDM03 | requires_assets_or_formal_run | 最佳global batch/rank/阶段budget缺作者run不能推测。 |
| MDM04 | requires_assets_or_formal_run | 未下载vis_model_256x256.pth；随机网络检查不认证真实初始化。 |
| MDM05 | requires_assets_or_formal_run | 完整1024随机模型明确禁用inner pretrained；真实checkpoint/curriculum未测。 |
| MDM06 | partial_cpu | mixed-resolution/VDM/scale/CFG等组件128步回归；在线T5/tokenizer/negative text/streaming真实reader缺资产。 |
| MDM07 | boundary_acknowledged | reader/MLX/S3/torchinfo导入stub明确披露；未以stub替代真实数据调用。 |
| MDM08 | partial_cpu | 精度元信息已修，FP32 backward先于accum除法的原次序也已对照；1024原FP16 scaler尚未执行。 |
| MDM09 | requires_assets_or_formal_run | 未取得T5/vision/inner最佳权重进行正式1024像素/画质验证。 |
| MDM10 | fixed_and_verified_cpu | 当前256/1024 full backbone与loss/backward、128步和采样分支重新执行，替代历史不匹配hash证据。 |
| DIM01 | requires_cuda_distributed | 原16rank/FP16/Accelerate/worker整CLI未执行。 |
| DIM02 | requires_assets_or_formal_run | 未加载真实 CompVis mean/logvar/label npy 数据验证 posterior 生命周期。 |
| DIM03 | partial_cpu | 已发布 CFG dropout/guidance 组件当前回归通过；真实pipeline文本/资产质量未认证。 |
| DIM04 | requires_assets_or_formal_run | 800epoch与1M更新关系没有原数据量/run日志不能认证。 |
| DIM05 | fixed_and_verified_cpu | 完整DiMR G/XL两个当前模型层级forward/rawgrad检查均通过。 |
| DIM06 | fixed_and_verified_cpu | 逐Tensor无grad原因记录：reentrant checkpoint首次no_grad的ChannelFirst view；重算中真正损失路径的梯度均已捕获并相等。 |
| DIM07 | requires_assets_or_formal_run | 真实VAE/最佳G-XL weights和正式FID未执行。 |
| DIF01 | unavailable_release | 作者训练/eval/checkpoint说明缺完整best recipe；完整codec组件不能补出best run。 |
| DIF02 | boundary_acknowledged | SE2/4 KL0-SE1属于明确组件实验，与source默认KL1-SE0分开；不冒称定位到最佳配置。 |
| DIF03 | requires_assets_or_formal_run | 完整Hydra/data/teacher训练合同未认证；本次直接作者codec/loss函数对照。 |
| DIF04 | partial_cpu | 补全SD式83.65M VAE/256grid +真实VGG19代码三尺度/冻结encoder，全部层及rawgrad相等；权重随机，非真实pretrained资产。 |
| DIF05 | unavailable_release | 时序/GAN/tokenizer/distill-AE等论文扩展整pipeline未全部接入；不把codec分支证据外推。 |
| DIF06 | requires_assets_or_formal_run | 没有codec换后下游正式生成器重训和主表质量。 |
| RMD01 | requires_assets_or_formal_run | 完整公开1.3B backbone对照不认证论文14B最佳实验。 |
| RMD02 | boundary_acknowledged | 发布global64和论文video batch4分别保留，未混成一种配置。 |
| RMD03 | requires_assets_or_formal_run | JourneyDB/VidProM真实prompts、负条件和text模型未绑定/加载。 |
| RMD04 | fixed_and_verified_cpu | 修为prompt-only/image-free，不再要求真实video作为论文主实验必需数据。 |
| RMD05 | partial_cpu | 30层Wan完整1.3B结构、transition1/2各17步，覆盖15步warmup后的fake/student backward及双Adam；真实21x60x104网格/预训练初始化未测。 |
| RMD06 | partial_cpu | CPU双optimizer/clip/state+floatdecode通过；原BF16/FSDP/AMP/正式checkpoint resume未认证。 |
| RMD07 | unavailable_release | GAN和SDXL/PixArt/SD3.5各实验pipeline未完整接入/对照。 |
| RMD08 | boundary_acknowledged | 数值结论归于固定发布commit，作者归属限制仍明确保留。 |
| RMD09 | requires_assets_or_formal_run | 主表质量/加速/完整CLI未正式跑。 |
| SPD01 | partial_cpu | FLUX DCT/DWT/FFT当前源码workflow逐层前向+浮点decode通过；PixelGen/Wan完整路线未认证。 |
| SPD02 | requires_assets_or_formal_run | 真实pretrained/text/tokenizer/VAE缺绑定，不下载替代。 |
| SPD03 | partial_cpu | 三个变换BF16 CPU随机fixture完整浮点/uint8像素及逐层forward相同；真实video/pretrained采样未测。 |
| SPD04 | requires_assets_or_formal_run | 正式质量/原GPU速度显存未测。 |
| SPD05 | boundary_acknowledged | 公开training-free流程无训练loss，backward标N/A，不把0个grad计为通过。 |
| NAM01 | unavailable_release | 公开只有推理，没有作者trainer，不能从batch768恢复训练。 |
| NAM02 | requires_assets_or_formal_run | 随机bridge已测，真实训练bridge和transformer权重未取得。 |
| NAM03 | partial_cpu | 实际使用作者vendored Diffusers，模型/VAE一起clone保持SiLU别名；真实text/tokenizer/VAE权重未测。 |
| NAM04 | partial_cpu | 三阶段固定16x32/64/128几何，CFG1/3、模块逐层与浮点像素相同；每stage3步/小宽度非最佳pipeline认证。 |
| NAM05 | boundary_acknowledged | 无发布作者训练器，backward N/A；不伪造一项作者训练验证。 |
| MRF01 | partial_cpu | 源码FLUX完整随机workflow逐层通过；Qwen分支尚未认证。 |
| MRF02 | requires_assets_or_formal_run | FLUX/Qwen/T5/VAE/RRDB真实权重缺绑定，不下载。 |
| MRF03 | partial_cpu | actual main完整VAE→PIL→tile SR→resize→posterior→refinement、sigma.12/RNG对照通过；小随机资产，非原真实分辨率最佳模型。 |
| MRF04 | requires_assets_or_formal_run | 正式质量/速度显存未执行。 |
| MRF05 | boundary_acknowledged | training-free source没有作者loss，backward N/A。 |
| RDD01 | boundary_acknowledged | 旧8x32与fair预算不同如实保留；CPU检查不宣称复现历史492k/407.6k训练轨迹。 |
| RDD02 | fixed_and_verified_cpu | original参考补data-source imagenet，实际parser正确。 |
| RDD03 | requires_assets_or_formal_run | 真实ImageNet/ADM/SD-VAE-ema/posterior rank-local生命周期未完整认证。 |
| RDD04 | fixed_and_verified_cpu | 当前完整L和XL逐层forward/原始backward、阶段汇总均完全相同。 |
| RDD05 | partial_cpu | 六历史目标变体+multires mapped-time原始loss/grad/三solver回归；所有可选REPA/minSNR/loss-aware组合未穷举。 |
| RDD06 | fixed_and_verified_cpu | DiscreteVP effective-noise、FM clean重构、native stage mean汇总运算树对齐；412/412通用组件byte exact，native/扩展loss与grad也全byte exact。 |
| RDD07 | requires_assets_or_formal_run | 已删旧GPU payload不能恢复，未补造历史数据/权重；新临时检查产物验收后清理。 |
| RDD08 | requires_assets_or_formal_run | 未运行真实完整CLI正式训练/新进程resume/同预算quality。 |
