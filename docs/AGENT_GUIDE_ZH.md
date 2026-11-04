# 新机器与新 agent 的仓库操作指南

本指南记录仓库中可定位的入口、实验约定和验证证据，不依赖旧聊天。
所有命令从 Git 根目录执行；`/path/to/...` 必须替换为本机路径。
训练示例是运行配方，阅读文档不等于启动实验。

## 1. 先确定要用哪套实现

| 目的 | 使用什么 |
|---|---|
| 在旧实现上训练/采样 RDD DDPM 或 FM | `2d/` 入口，下文第 2–3 节 |
| 在新框架上复现旧 RDD | `unified_multires/configs/rdd_reproduction/`；保留 native 模型、路径和边界 |
| 不同论文共用 DiT/VAE 做公平比较 | `unified_multires/configs/comparison_imagenet256/`；这是方法移植实验 |
| 运行作者公开配置的兼容实现 | `unified_multires/configs/paper_recipes/`，逐项看支持范围与证据 |
| 已有 FM-XL v2 做 fresh 采样，然后训练 PF | `unified_multires/experiments/imagenet256_fm_pf/`；优先用现成完整流程 |

`main` 已包含旧 RDD 和新 unified；不需要另找分支或另 clone 一个 unified 仓库。
新框架是自建组件系统，部分 native 网络/求解器复用作者实现，不是 FastVideo 的包装。
旧实验的 “Unified DiT” 只是指多个分辨率共用一个 RDD 模型。

开始工作先看 `git status`，保留已有修改。普通环境可用：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './unified_multires[rdd,test]'
```

按设备选择兼容的 PyTorch。上述依赖不是历史环境的精确锁定；FM/PF 正式配对实验用其
`setup_environment.sh` 固定环境。不要为跑本项目修改其他项目的共享环境。
CPU 测试使用 `CUDA_VISIBLE_DEVICES='' UMDIFF_CPU_ONLY=1`；正式 GPU 任务重新选择获准设备。
旧机器的卡号、监控与截止日期不构成本仓库的新机器运行规则。

## 2. 旧仓库训练

两个共享模型入口都支持 `--vae-path` 本地 Diffusers VAE 目录，支持 torchrun/DDP。
ImageNet 使用原始 train 类别目录，显式选择 `--data-source imagenet`，不要沿用默认 hf loader。
下面以一张获准 GPU、global batch 256、microbatch 1 写出参数；可按设备调整卡数和 microbatch。

```bash
export IMAGENET_TRAIN_DIR=/path/to/imagenet/train
export SD_VAE_EMA_DIR=/path/to/sd-vae-ft-ema

# DDPM：stage-lengths 按高/中/低分辨率排列。
torchrun --standalone --nproc_per_node=1 2d/train_RDD_imagenet256_unified.py \
  --model DiT-XL/2 --resolutions 32,16,8 --stage-lengths 300,300,400 \
  --diffusion-steps 1000 --image-size 256 --data-source imagenet \
  --data-path "$IMAGENET_TRAIN_DIR" --vae ema --vae-path "$SD_VAE_EMA_DIR" \
  --global-batch-size 256 --micro-batch-size 1 --global-seed 0 \
  --max-train-steps 400000 --ckpt-every 50000 \
  --results-dir unified_multires/runs/legacy_ddpm

# FM-XL v2 时钟：stage-lengths 按低/中/高排列，对应 0/.3/.6/1。
torchrun --standalone --nproc_per_node=1 2d/train_RDD_FM_imagenet256_unified.py \
  --model DiT-XL/2 --resolutions 32,16,8 --stage-lengths 300,300,400 \
  --diffusion-steps 1000 --image-size 256 --data-source imagenet \
  --data-path "$IMAGENET_TRAIN_DIR" --vae ema --vae-path "$SD_VAE_EMA_DIR" \
  --global-batch-size 256 --micro-batch-size 1 --global-seed 0 \
  --time-sampler uniform --repa-weight 0 \
  --max-train-steps 400000 --ckpt-every 50000 \
  --results-dir unified_multires/runs/legacy_fm
```

**这两条都是正式长训练命令，不是 CPU smoke test。** 需真实数据、VAE、设备授权和容量预检。
microbatch 1 不保证任意 GPU 都能装下 XL 的参数、梯度、optimizer 和 EMA。
`--micro-batch-size` 是全部 rank 合计的 microbatch。例如四卡、micro32、global256，
每卡每次前向 8 张，累积 8 次。global、micro 与 rank 数必须满足代码的整除要求。
单卡累积与历史八卡训练有效 batch 可相同，但不承诺 RNG 和每次更新逐位相同。

恢复时使用同一配方，加 `--resume /path/to/checkpoint.pt`；确认日志中的恢复 step 和目标总步数。
旧脚本会在 `results-dir` 下创建实验子目录，以日志记录的 checkpoint 路径为准。
独立三模型、decoupled 和其他历史变体不要简单改共享模型名字；先查具体入口与对应配置。

## 3. 旧仓库采样

`2d/sample_fid_unified.py` 读取 **JSONL，每行一个实验**。例如先创建自己的 run 目录，保存为
`unified_multires/runs/legacy_fm_fresh.jsonl`，将 checkpoint 和输出路径替换成本机路径：

```json
{"model_type":"fm_unified","model":"DiT-XL/2","ckpt":"/path/to/RDD-FM-Unified-XL-v2/0400000.pt","resolutions":"32,16,8","stage_lengths":"300,300,400","num_steps":"80,80,90","sampler":"euler","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"/path/to/output/fm_fresh"}
```

```bash
python 2d/sample_fid_unified.py \
  --config unified_multires/runs/legacy_fm_fresh.jsonl \
  --rank 0 --world-size 1 --total-samples 64 --batch-size 4 --vae-batch-size 1 \
  --seed 0 --vae-path "$SD_VAE_EMA_DIR" --no-eval
```

单 rank 时自动选择可见 CUDA，否则 CPU。必须显式写 `--world-size 1`，默认值是 8。
多卡采样给每个进程单独的可见卡和 `--rank r --world-size N`；不要仅套 torchrun，
因为该入口不会自动将 `LOCAL_RANK` 当成这些参数。正式评估需保存完整配置和分片完成情况。

旧 DDPM 的现成 JSONL 模板是 [eval_unified_l_400k.jsonl](../configs/eval_unified_l_400k.jsonl)，
模型为 L/2，checkpoint 路径须对应，不能错配 XL 权重。它的 `model_type=unified`、
`sampler=ddpm`，阶段长度按高→低，采样步数按低→高。

当前 FM 训练和 `model_type=fm_unified` 采样的阶段长度都按低→高。
corrected FM-L 不能只看标签猜时钟；查
[原生配方表](../unified_multires/configs/rdd_reproduction/README_ZH.md)。
旧 sampler 默认加载 EMA；正式复现仍需核实 checkpoint 的实际字段、模型与日程。

## 4. unified 训练与采样

完整 CPU 两步训练、恢复、采样、视频 tensor 示例见
[unified README](../unified_multires/README.md)，不需要下载模型或数据。

| 配置 | 入口 | 恢复语义 |
|---|---|---|
| raw method JSON | `python -m umdiff train/sample` | train 的 `--steps` 是本次新增更新数 |
| `umdiff_experiment_v1` 完整 JSON | `python -m umdiff.experiment plan/train/sample` | train 自动跑到配置规定的总更新数 |

完整 ImageNet 比较可先只解析计划：

```bash
EXPERIMENT=unified_multires/configs/comparison_imagenet256/main/rdd_ddpm_xl_legacy.json
python -m umdiff.experiment plan --config "$EXPERIMENT"

# 资产/设备已准备且任务要求正式训练时：
python -m umdiff.experiment train --config "$EXPERIMENT" \
  --data "$IMAGENET_TRAIN_DIR" --vae "$SD_VAE_EMA_DIR" \
  --device cuda --output unified_multires/runs/imagenet_rdd
```

复现旧 FM-XL v2 fresh 采样，用对应 native 配方导入原权重：

```bash
python unified_multires/tools/import_legacy_rdd_checkpoint.py \
  --config unified_multires/configs/rdd_reproduction/rdd_fm_xl_v2_euler_cfg1p5_fresh.json \
  --checkpoint /path/to/0400000.pt --weights ema \
  --output unified_multires/runs/fm_imported.pt

python -m umdiff sample --checkpoint unified_multires/runs/fm_imported.pt \
  --count 1 --labels 17 --seed 0 --device cpu \
  --output unified_multires/runs/fm_latents.pt

python -m umdiff decode --latents unified_multires/runs/fm_latents.pt \
  --vae "$SD_VAE_EMA_DIR" --scale 0.18215 --device cpu \
  --output unified_multires/runs/fm_rgb.pt
```

此例 CFG1.5，真实 XL 在 CPU 上也需足够 RAM；不是小模型验证命令。
导入仅转换推理外壳，不能当作恢复旧 optimizer/RNG/数据游标。
独立模型用 `import_legacy_rdd_multistage.py` 分别指定 32/16/8 的 checkpoint。
原生训练先复制配方并把 `@sd_vae_ema@` 替换成本地 VAE；详见原生配方 README。

unified 默认使用 EMA。原生步数在配置的 `sampler.args.steps` 或 `process.steps`；
不要用通用 `sample --steps` 覆盖原生 solver。普通 `--count` 是**一个 batch**，
不能写 50000 就期望自动分批。当前公开入口是单进程/单设备，不支持直接 torchrun 多卡。

## 5. 现成 FM fresh → PF 训练 → 双方评估

使用 [实验包 README](../unified_multires/experiments/imagenet256_fm_pf/README_ZH.md)，
具体步骤在 [AGENT_PROMPT_ZH.md](../unified_multires/experiments/imagenet256_fm_pf/AGENT_PROMPT_ZH.md)。
只有用户要求执行该实验时才将这份 prompt 当作执行计划；普通改代码任务不启动其中的长训练。

| 步骤 | 文件/动作 |
|---|---|
| 独立环境 | `setup_environment.sh`，固定依赖；设置 `UMDIFF_PYTHON` |
| FM checkpoint、VAE | `workflow.py assets`；`assets.lock.json` 固定 HF revision/hash |
| 64 张预览、50K FM fresh | `run_fm_fresh.sh`；分批、校验、断点恢复 |
| 原始 ImageNet | `prepare_imagenet.py`，已有 tar 或有权限的 HF 数据；不替用户接受访问条款 |
| PF 实际第 1 步预检、400K 训练 | `run_pf_train.sh --preflight-only`，然后同目录 `run_pf_train.sh` |
| PF 50K | `workflow.py pf-sample` |
| 双方 ADM 五指标 | `setup_evaluation.sh`、`run_adm_evaluation.sh RUN_ROOT` |

FM 模型来源为 [HF 归档](https://huggingface.co/yang335/search_agent_hl_to_i18n_transport)，
路径 `checkpoints/RDD-FM-Unified-XL-v2/0400000.pt`。只下载 lock 中的文件，不拉整个大仓库。
VAE 为 `stabilityai/sd-vae-ft-ema`。FM/PF 主对比是 CFG1、同 DiT-XL/2 主体、同 VAE、
latent8→16→32、名义阶段 0/.3/.6/1；PF 保留自身位置编码、路径、补噪和实际时钟修正。
PF 从头训练，不用 RDD 权重初始化。双方实际 NFE 均 252，但仍需测耗时。
这不是 PF 原视频论文的原生架构/最佳结果复现。

PF 完整 XL 真实数据预检是正式第 1 次更新，后续从它继续，不重新初始化。
训练/采样仍是单卡；正式任务中途改变 batch、seed、模型或配置需新 run，不能混入旧输出。

## 6. 指标、验证与实验记录

正式 ImageNet 指标用 50K 生成图和 ADM 的 FID/sFID/IS/Precision/Recall。
[EVALUATION_ZH.md](../unified_multires/experiments/imagenet256_fm_pf/EVALUATION_ZH.md)
提供官方 Inception GraphDef、ImageNet-256 参考 NPZ 的下载与校验，使用独立 TensorFlow CPU 环境。
不要把旧 `tensorflow-gpu` 环境说明作为新机器安装方案。

FM/PF 的 `run_adm_evaluation.sh` 校验该包的 `SUMMARY.json` 和逐批 hash，不能直接传旧 sampler 的
PNG 目录。旧输出需按[旧 ADM 评估入口](../guided-diffusion/evaluations/README.md)调用
`2d/png_to_npz.py` 与 `guided-diffusion/evaluations/evaluator.py`，使用同一 reference/GraphDef，
并记录样本数和来源。打包 50K RGB NPZ 约 9.83GB，需预留临时空间和足够 RAM。

已发布的依据：

- [旧 RDD ↔ native unified 对照](validation/20261104/README.md)：module activation、activation/参数梯度、更新/EMA/RNG及完整采样轨迹；按每份 JSON 的具体配置/源码 hash 解读。
- [合并后 CPU 验收](../unified_multires/evidence/monorepo_20261104/REPORT_ZH.md)：安装入口、迁移、图像/视频小管线等。
- [FM/PF 实验包验收](../unified_multires/experiments/imagenet256_fm_pf/VALIDATION_ZH.md)：随机小 DiT/VAE、完整 NFE 采样与恢复；未在本机跑正式模型质量实验。
- [各论文支持范围](../unified_multires/SUPPORTED_REPRODUCTIONS_ZH.md)：native 配置和有意保留的边界。

“同输入、权重、RNG 下逐元素一致”不等于重新训练 400K 后 FID 一样，也不等于任意 GPU/BF16
或全部论文配置一致。修改数值代码时跑相关 CPU 测试，不需要为文档修改启动正式训练。
部分原始大型 trace 未进入 Git；报告中的历史绝对路径是 provenance，不是新机器必备目录。

每个正式 run 记录：Git commit、配置、数据/VAE/模型来源与 hash、EMA/raw、dtype、设备、
global/micro/累积、seed、阶段时钟、fresh/filling、solver、分阶段 steps/实际 NFE、CFG、
生成张数、指标命令、结果与耗时。运行中另记 PID/session、最新 step 和恢复命令。
保留正式 checkpoint 与指标；自己的临时 tensor/测试图/中间文件用完清理，避免再次堆积。

## 7. 长期约定与导航

- 边界定义以 [BOUNDARY_NAMES_20261104_ZH.md](BOUNDARY_NAMES_20261104_ZH.md) 为准：
  fresh/refresh 丢弃旧噪声；noise_filling 保留由旧 noisy 和预测 clean 恢复的残差。
  旧 DDPM 的同名误用已明确迁移；历史 FM fresh 与 filling 不能共用一份 FID。
- 共同比较外层 loss 按 RDD：同一批图覆盖全部阶段，每阶段 mean，再以系数 1 求和。
  不悄悄改成随机单阶段、除以阶段数或照搬作者 packing 权重；内部目标/权重按所选配方。
- 主实验关注单条生成路径中改变分辨率；Relay 等已排除候选虽留有代码，不自动放回主表。
- 作者源码用 `unified_multires/tools/fetch_sources.py` 按 lock 下载/校验，RDD 直接用本仓库父目录。
  源码缓存、数据、checkpoint、env、运行日志不属于源码提交；遵守各来源许可证。
- 视频入口、3D 和理论草稿见 [内容索引](REPOSITORY_CONTENTS_ZH.md) 与 [理论索引](README.md)。
  `wan_grid` 不是真实 Wan，能跑视频 tensor 不表示完整视频 VAE/文本条件系统已认证。
  历史理论草稿不是最新证明，不能拿讨论中的猜测直接宣称算法优越。

本仓库的 [根 AGENTS.md](../AGENTS.md) 是自动入口，`2d/AGENTS.md` 和
`unified_multires/AGENTS.md` 补充局部规则。根据
[Codex 官方说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)，
在 clone 后的仓库内启动新的 Codex 会话即可加载项目指令；无需安装个人 skill。
自动入口会要求按任务读取本指南，具体长文不会全部塞进启动上下文。
已打开的旧会话需重新启动才能重新构建指令链。若未加载，检查工作目录、上层
`AGENTS.override.md` 和本地 Codex 配置。其他工具是否自动读取取决于其对 AGENTS.md 的支持。

今后修改 CLI、实验规则或支持范围时，同时更新这些入口与对应操作文档；不靠旧聊天保存知识。

文档核对记录（2026-11-04）：新指南及三份 AGENTS 的 34 个相对链接均指向随 Git 发布的文件；
5 个 Bash 代码块通过语法检查，4 个脚本示例的参数与实际 argparse 定义一致。
CPU/离线环境实际执行 8 个相关 CLI `--help` 和上述 ImageNet 配置的 `plan`，全部成功。
本次仅改文档，没有下载模型/数据、使用 GPU 或重跑训练质量实验。
