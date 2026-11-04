# CPU执行策略与后续GPU验证准备（2026-10-10）

**本日后续严格审计纠正**：[完整差异报告](evidence/strict_original_unified_audit_20261010/REPORT_ZH.md)确认多个CPU可修的原配方/生命周期缺陷，必须先处理再执行未来CUDA规范。历史机器清单把MDM写为native FP16，但所选256源YAML为fp16:0，应逐配置确定精度；RMD的“Real video/prompts”资产描述不准确，论文主实验是image-free/prompt-only，不应强制真实视频训练集。旧plan与绑定代码保留供追溯，尚未重生成或启动GPU。

用户最新指示：“只用cpu。但是如果有一定要用gpu的地方那就一次性全部准备好”。当前只执行CPU工作，不启动GPU，不下载模型或数据，不启动正式训练，不恢复旧监控。旧卡号/截止时间不作为新的授权。

准备清单覆盖14种有可执行公开实现的外部方法及本地RDD，共15种方法；绑定33份成对配置，包含11份RDD预算对齐配置。未来CUDA核验分为66个方法/类别规范。**这是完整覆盖当前方法清单的测试计划；没有经过认证的CUDA测试runner，也不是66个已经可以执行或通过的实验。**

机器清单：[configs/gpu_checks/plan.json](configs/gpu_checks/plan.json)。本轮CPU审计：[evidence/gpu_preparation_20261010/SUMMARY.json](evidence/gpu_preparation_20261010/SUMMARY.json)；执行约束测试：[POLICY_CHECK.json](evidence/gpu_preparation_20261010/POLICY_CHECK.json)。既有数值对照：[evidence/paired_experiments_20261010/REPORT_ZH.md](evidence/paired_experiments_20261010/REPORT_ZH.md)。两个原版/unified成对脚本的使用范围仍以[PAIRED_EXPERIMENTS_ZH.md](PAIRED_EXPERIMENTS_ZH.md)为准。

## 可以继续用CPU完成的事情

- 数学路径、噪声协方差、上采样、预测目标、阶段时间、loss、采样器和配置检查；已有小网络activation/gradient对照证据继续有效。
- 原版完整CLI的参数适配、数据读入/预处理、累积更新、优化器/EMA、保存恢复和随机状态的工程补齐。完整作者CLI尚未认证，不能把这个缺口归因于没有GPU。
- 精确确定论文最佳配置、初始化与评估设置；不存在的私有分支或缺失配方不能通过GPU测试补出。
- 显式绑定现有本地资产，核对checkpoint/codec/text encoder/tokenizer/SR、数据与预处理；不自动搜索或下载资产。缺真实资产时，合成输入不能解除对应阻塞。

已有对照是执行过的模块边界和分支的检查。它们没有检查所有functional/ATen节点，也不能证明真实数据训练的最终FID。

## 需要实际GPU才能验证的范围

| 类别 | 要比较的内容 | 前置条件与结果口径 |
|---|---|---|
| CUDA FP32核心 | 实际CUDA attention/packed kernels、activation、activation gradient、参数gradient、loss、连续RNG | 必须先补两侧device传播；当前源码适配器仍有CPU张量/上下文。不能带着CPU attention shim认证FlashAttention |
| 原生精度 | CUDA FP16/BF16或原版FP32，autocast/master参数/scaler、overflow、unscale、clip/repair、Adam/EMA | CPU同dtype可预检，CUDA实际kernel仍需实测；保持原版precision策略，不能沿用FP32CPU结论 |
| 采样与真实解码 | 各阶段边界、CFG/noise、solver/NFE、final latent与真实VAE浮点pixel；需要时包含text/learned bridge/SR | 两侧必须使用同一真实本地资产与输入。检查uint8量化前像素，不能只比较latent或保存PNG |
| 分布式完整生命周期 | 原版CLI与unified CLI的累积/all-reduce、FSDP/NCCL、clock、保存与新进程恢复、worker/data RNG | 原版CLI适配先在CPU补齐。两卡仅是smoke，不能代替原最佳拓扑；单卡不能认证多卡通信 |
| 实际成本 | synchronized时间、NFE/forward计数、显存峰值；训练/推理/预训练成本分别记录 | 数值检查先通过，固定backend/dtype/shape/资产后测量；CPU时间/FLOPs不能替代CUDA结果 |

默认设计的核心数值检查仅两次更新，采样使用选定原版solver行；正式训练与50k样本质量评估属于另外的实验，不会被这个准备入口启动。实际显存和磁盘按模型、optimizer/EMA、双侧比较和恢复payload估算，没有统一80GiB要求。两个对照可顺序跑并保存小型hash/统计，避免为了同时放两套状态增加显存。

## 逐篇必须保留的特殊检查

| 方法 | 特别容易被“统一配置”掩盖的地方 |
|---|---|
| Relay | 两网络cascade、FP16 loss scaling与gradient repair、按images计数的LR/EMA；原参考8-rank不是本次1进程 |
| SSD | 原DDPM与scale/time分组；不能借lr_anneal_steps截断训练而改变constant LR |
| PixelFlow | Packed variable-length attention、跨阶段双份CFG、Euler与Dopri5；最佳训练batch/budget未知 |
| DFM | Uniform/lognormal/REPA、原NumPy debug draw、Accelerate累积；公开SiT/SD-VAE不同于论文DiT/SE-FLUX |
| LapFlow | MOT层数、EQVAE/SD-VAE posterior、stage/path/CFG；公开默认与论文flags不同 |
| PPFlow | 两/三层batch packing、SiT初始化、Heun CFG；原CLI缺匹配累积/更新上限/本地VAE的适配 |
| DCTdiff | FP16、逐数据集DCT统计/EBFR、真实DCT decode；未发布DiT分支不能用UViT替代认证 |
| MDM | 多层内模型初始化、在线Flan-T5、流式mixed-resolution/curriculum；发布权重50M Flickr不同于CC12M训练配置 |
| DiMR | Moments/labels、Python CFG dropout、原DPM guidance语义；发布16-rank/2-node FP16生命周期 |
| Diffusability | SE2/4、VGG、冻结/训练codec与下游生成器；最佳完整配方缺失，GAN/temporal尚未纳入 |
| RMD | BCTHW、teacher/fake/student、warmup15、两optimizer、两transition、AMP/FSDP/GAN；发布global64不同于论文batch4 |
| RDD | Decoupled/joint、32/16/8 latent与300/300/400、真实SD-VAE；历史8×32不同于fair 1×16×16 |
| SPEED | 公开推理的DCT/Haar/FFT、FLUX/VAE、CFG与NFE；没有公开训练对照 |
| NAMI | 训练过的transformer与learned bridge、vendored Diffusers；不能用普通FLUX认证，无公开trainer |
| MrFlow | VAE/PIL uint8/RRDB tile/resize/posterior/refinement全链路；FLUX与Qwen需分别绑定真实资产 |

每个方法的source commit、论文配方缺口、本地资产要求、原版CLI参考参数以及每份paper/fair合同SHA都在机器清单中。没有可执行源码的8篇文献另保留为未复现，不拼造GPU测试。

## 误差判定和随机性

同一GPU、相同kernel、相同输入与draw的配对先要求逐字节相等，同时报告第一个不同的层/阶段、max absolute/relative、RMS及非有限值。FP32、FP16、BF16分别记录，不合并为一个passed。

切换Flash/SDPA等kernel后若不同，必须定位到具体操作，并复现“原版切换到相同backend后差异消失”的证据。不同mask/layout/数学定义不属于浮点舍入。没有实测前不设置一个宽松的全局容差让全部论文自动通过；数值接近和逐字节相等是两个结论。

CPU和CUDA RNG不能假定相同seed生成同一噪声。跨设备数学对照应显式提供同一noise/times/posterior draws；同设备真实生命周期对照应保存CPU、全部可见CUDA、NumPy、Python与数据游标状态。不同多卡归约顺序也需单独定位舍入，不能直接宣称单进程结果等于历史8卡轨迹。

## 现在可执行的CPU准备入口

```bash
cd /scr/yuheng/bugswarm/rdd_workspace_20261005
bash unified_multires/prepare_gpu_checks.sh --action plan
bash unified_multires/prepare_gpu_checks.sh --action check
```

入口只使用Python标准库和不加载Torch的计划代码，强制CUDA_VISIBLE_DEVICES为空；没有run动作，不启动nvidia-smi/NVML、CUDA、训练子进程或联网下载。check检查清单是否过期、33配置/11fair覆盖、15方法/66规范、原17库1211源文件hash、50份既有CPU报告与其绑定的87个实现文件以及工具hash；它不是新的模型数值检查。

CPU执行约束测试已通过：调用者即使设置CUDA_VISIBLE_DEVICES=7也会在Python启动前清空；禁止Torch/CUDA导入、子训练进程和网络连接的审计hook下仍可完成清单检查；`--action run`会被拒绝。计划中的原版CLI参考把Python解释器保留为`@experiment_python@`，避免CPU检查使用python/python3不同符号链接时误报配方变化；未来实际运行环境仍需单独固定。

本地资产绑定沿用[configs/paired_experiments/assets.example.json](configs/paired_experiments/assets.example.json)中的data/assets映射。需要时使用同一个显式本地bindings文件分别执行plan/check的`--bindings /absolute/path/assets.json`。该步骤只展开路径与记录缺项，不代表资产内容/权重已经验收。

未来实际GPU执行还需要补齐CUDA runner和原版CLI适配、完整资产检查，并确定当时获准的资源/绝对deadline。已有独立guard位置记录在清单中，此次不启动它，不自动续用旧卡号或到期授权。数值对照完成后保留配置/hash/小报告，立即清理临时checkpoint/media；保留正式训练成果、用户数据和原版权重。
