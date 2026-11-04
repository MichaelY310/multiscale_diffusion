# 成对实验入口：论文配方与 RDD 对齐配方

逐项修复和验收请看[未完成事项、偏离与验证缺口清单](ORIGINAL_UNIFIED_GAPS_TODO_ZH.md)，其中含33配置的完整入口阻塞附录。

**2026-10-10 严格审计补充**：当前original是作者源码核心加共同Trainer，不能作为官方完整CLI验收。确认Relay增强/网络条件、high EMA1000倍和归一化差异；LapFlow参考dataset错误、DFM参考漏必需参数；data/drop_last、条件label和MDM setup RNG也有差异。专用小配置的已捕获activation/gradient仍相等；generic组件有非零误差；33/33 native入口未认证。完整逐篇缺口、像素/反向范围及13项CPU检查见[严格审计报告](evidence/strict_original_unified_audit_20261010/REPORT_ZH.md)。尚未修复的项已列入交接，勿以合同SHA相等推断论文配方相等。

**2026-10-10 最新设备约束**：本轮及后续接续只执行CPU。GPU必需的验证范围已集中准备，见[GPU_CHECKS_PREPARATION_ZH.md](GPU_CHECKS_PREPARATION_ZH.md)及`prepare_gpu_checks.sh`（仅plan/check，不执行GPU）。旧GPU时间窗/卡号没有恢复；CUDA runner与作者完整CLI仍待适配，不能把计划视为已验证或已可正式运行。

两个入口为 `run_original_papers.sh` 和 `run_unified_papers.sh`，读取同一份
`configs/paired_experiments/manifest.json`。默认只生成计划，不启动正式训练。

**当前不能声称所有论文的最佳结果都可完整复现。** 默认 `native-cli` 模式列出真实作者训练入口和 unified CLI，保留每个入口尚未解决的配置、数据、训练生命周期差异，并拒绝正式启动未认证的配对。作者 CLI 的参数参考不是已完成的运行命令。

`matched-core` 是本次实现的受控实验入口：原版一侧直接调用固定版本作者源码的 loss/path；unified 一侧调用我们的 Method 组件；两侧使用相同作者 backbone、数据读取、初始化、优化器、EMA、microbatch 和训练循环。它确实执行两种不同的目标构建实现，但**不等于未经修改的作者完整 CLI**。当前仅支持单进程 CPU；GPU 原授权已过期，没有隐式续期，没有多卡训练认证。

## 两种配置含义

- `paper`：保留已公开的配方参数。公开示例与论文主表不同、原训练 seed/batch/预算未知时都明确记录。它不是“论文最佳成绩已复现”的认证。
- `fair-rdd`：ImageNet-256 类条件、global batch256、AdamW、LR1e-4、betas(.9,.999)、wd0、常数LR、EMA.9999、FP32、seed0、400k更新。单进程 microbatch16 × accumulation16。历史 RDD 是8卡，每卡32、不累积；全局 batch相同不代表与历史多卡轨迹逐位相同。400k是本次选定的共同预算；历史L到492000、XL到407600。

对齐仅修改实验控制项。各论文的 backbone、VAE、预训练初始化、目标、阶段路径、阶段采样和推理策略保留。默认没有把 PPFlow 的 SiT预训练成本忽略为零，也没有把 PixelFlow变成latent模型。预训练成本、参数量、FLOPs和收敛调参必须另报。

ImageNet-256同任务的11配置可做受控组：DiMR-G/XL、PixelFlow Euler/Dopri5、PPFlow2/3、DFM-XL/B公开版、LapFlow论文flags、RDD-L/XL。这是6种方法。其余数据集、文本生成、视频蒸馏、自编码器和 inference-only方法不会被悄悄改写为这个组。

不建议把“所有配方都和RDD一样”作为复现原成绩的唯一实验。batch变化会改变更新次数与梯度噪声；优化器、EMA时钟、预训练、VAE、采样器和训练样本数同样影响结果。统一batch/预算的结果应作为新的对照组；原论文成绩应由原配方单独验收。公平组还需要相同评估集、参考统计、50k样本协议、CFG选择预算、多训练seed，以及等FLOPs/预训练成本表。

## 审阅和小测试

从工作区运行：

```bash
bash unified_multires/run_original_papers.sh --mode paper --action plan
bash unified_multires/run_unified_papers.sh  --mode paper --action plan
bash unified_multires/run_original_papers.sh --mode fair-rdd --action check
bash unified_multires/run_unified_papers.sh  --mode fair-rdd --action check
```

计划中的 `production_entrypoints` 给出作者/unified真实入口参考及阻塞原因。`contract_sha256` 对整个配方、数据绑定、资产绑定计算哈希；原版和unified对应配置必须相同。计划检查不加载模型，不下载任何资产。

明确的小CPU测试：

```bash
bash unified_multires/run_unified_papers.sh --mode fair-rdd --action validate
bash unified_multires/run_original_papers.sh --mode paper --action validate
```

`validate`始终成对测试作者源码后端与unified后端：缩小的非零网络、合成输入、FP32CPU、两次更新、梯度累积、连续RNG。作者模块名字冲突，因此每配置独立进程。原版优化器/EMA/累积由测试中独立实现，与我们的Trainer比较；没有逐步重新seed掩盖状态差异。`--action validate`不认证native CLI。

已有 `full_batch/` 检查另外保留RDD实际batch256/micro16/accum16，但网络仍缩小，没有真实VAE/数据。PixelFlow Dopri5和Euler分别与作者pipeline比较原始浮点像素与RNG；SPEED/NAMI/MrFlow另有本地随机VAE/SR的解码像素检查。latent-only测试不能冒充真实VAE像素一致性，更不能推断FID。

真实数据绑定见 `configs/paired_experiments/assets.example.json`。例如：

```json
{
  "data": {
    "rdd": "/absolute/path/imagenet/train",
    "ppflow": "/absolute/path/imagenet/train",
    "dimr": "/absolute/path/original_vae_moments"
  },
  "assets": {
    "vae": "/absolute/path/original_sd_vae",
    "sit": "/absolute/path/original_sit_tensor_state.pt",
    "eqvae": "/absolute/path/original_eqvae"
  }
}
```

数据可按 profile ID覆盖同论文绑定，例如DCT的不同数据集。没有资产时不回退到随机权重，也不替换VAE。真实训练必须显式指定一份profile和 `--entrypoint matched-core --action run`；此入口当前是受控CPU实验工具，不能替代未完成的原CLI复现或生产GPU训练。默认native-cli的 `run`会解释阻塞并退出。没有自动全量任务队列。

本次证据在 `evidence/paired_experiments_20261010/`；主要结果配方、源代码版本、逐项不足和验收范围见该目录 `REPORT_ZH.md`。小测试完成后保留JSON、配置、张量统计和SHA，删除本次临时checkpoint/生成媒体；正式结果和用户数据不在清理范围。
