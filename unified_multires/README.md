# Unified Multires：配置驱动的多分辨率 diffusion / flow 框架

这是旧 RDD 仓库里的新统一训练、采样和对照框架。今后两者一起维护：

新 agent 从仓库根目录的 [AGENTS.md](../AGENTS.md) 自动获得导航；本目录另有
[开发规则](AGENTS.md)。旧/新入口选择、训练采样和评估约定见[操作指南](../docs/AGENT_GUIDE_ZH.md)。

**新服务器实验入口：** [RDD-FM fresh采样 → ImageNet-256 Pyramid Flow训练](experiments/imagenet256_fm_pf/README_ZH.md)，包含环境安装、固定模型/VAE下载、断点续跑与[给agent的完整prompt](experiments/imagenet256_fm_pf/AGENT_PROMPT_ZH.md)。

```text
multiscale_diffusion/
├── 2d/、DiT/、diffusion/       # 旧 RDD 训练、网络、采样实现
├── rdd_video/、RDD_DiT_3D/     # 旧视频和 3D 研究代码
└── unified_multires/
    ├── umdiff/               # 新框架
    ├── configs/              # 实验配置
    ├── tools/                # 原生对照、权重导入、源码获取
    └── tests/                # CPU 回归测试
```

**先选实验目的，再选配置。** 原生复现保留选定原实现；共同 DiT 实验有明确的模型、VAE和训练估计器改动，两者不能混称。随机小测试通过也不等于已经复现论文 FID。

## 1. 安装与环境

以下命令从 Git 仓库根目录运行。Python 3.10+；建议使用独立虚拟环境，并按自己的设备安装合适的 PyTorch。

```bash
git clone https://github.com/MichaelY310/multiscale_diffusion.git
cd multiscale_diffusion
python -m venv .venv
source .venv/bin/activate

# 基础框架 + 旧RDD/共同DiT所需依赖 + 测试工具
python -m pip install -e './unified_multires[rdd,test]'
```

只运行自有小模型和通用组件，可用`pip install -e ./unified_multires`。更多作者网络按需安装`./unified_multires[authors,test]`。`requirements-authors.txt`记录历史实际环境，含当时预发布版本，不是保证所有机器都能安装的通用锁文件。

本仓库依赖源码目录，使用上面的**editable install**。不安装时也可：

```bash
export UMDIFF_PYTHON=python
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" -m umdiff --help
```

`activate_cpu.sh`隐藏GPU并设置`UMDIFF_CPU_ONLY=1`；`activate.sh`只设置Python搜索路径和离线模型加载，不改变设备。默认`--device cpu`。需要自己启动GPU训练时，在另一个正常环境的shell里选择设备，例如`CUDA_VISIBLE_DEVICES=0`并传`--device cuda`；不要沿用CPU-only shell。CLI是**单进程、单设备**，通过microbatch累积实现有效batch，不能直接把命令套进torchrun当多卡训练。

本次迁移/验收没有使用GPU。历史临时任务的主机名和2026年10月截止日期已从通用设备选择函数中移除；任务的资源授权和deadline由外部启动环境管理。

### 作者源码依赖

RDD的原网络和solver直接使用本仓库上层的`DiT/`、`diffusion/`，无需额外下载。需要其他作者实现的配方，先准备固定源码：

```bash
python unified_multires/tools/fetch_sources.py --list
python unified_multires/tools/fetch_sources.py --repo pixelflow --repo lapflow
# 离线检查这两个来源；缺文件或hash不符都会报错：
python unified_multires/tools/fetch_sources.py --repo pixelflow --repo lapflow --verify
# 若确实需要全部作者快照：
# python unified_multires/tools/fetch_sources.py --all
```

`sources.lock.json`固定23个来源及每个源码文件SHA256。下载器只获取列出的源码，拒绝不同hash的已有文件，不下载模型或数据。各来源许可证独立，见[SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md)。第三方源码、`.git`历史、本地依赖缓存均不重复提交到此仓库。

## 2. 先跑一个无需数据和权重的CPU闭环

下面生成一个小型RDD配置，训练2次更新，再采样。它验证管线，生成的是随机数据训练的小模型，不是正式图像效果。

```bash
python -m umdiff configure \
  --method rdd --shape 3,16,16 --backbone grid_dit \
  --width 24 --depth 1 --attention-heads 3 --patch 1,2,2 \
  --grids '[[8,8],[16,16]]' --sampling-steps 4 \
  --output unified_multires/runs/demo/rdd.json

python -m umdiff train \
  --config unified_multires/runs/demo/rdd.json \
  --synthetic --steps 2 --batch-size 2 --device cpu \
  --output unified_multires/runs/demo/train

python -m umdiff sample \
  --checkpoint unified_multires/runs/demo/train/checkpoint.pt \
  --count 2 --seed 0 --device cpu \
  --output unified_multires/runs/demo/samples.pt
```

RGB输出同时保存PNG。训练目录保存`checkpoint.pt`与`trace.json`；checkpoint带配置、模型、EMA及训练恢复状态。临时测试结束后可删除自己的`runs/demo/`。

恢复**再训练2步**：

```bash
python -m umdiff train \
  --config unified_multires/runs/demo/rdd.json \
  --resume unified_multires/runs/demo/train/checkpoint.pt \
  --synthetic --steps 2 --batch-size 2 --device cpu \
  --output unified_multires/runs/demo/resumed
```

底层`umdiff train --steps`表示本次新增更新数；下一节的`umdiff.experiment train`则自动训练到配置中的总预算。不要混淆。

## 3. 选择哪类配置

| 目录 | 用途 | 能声称什么 |
|---|---|---|
| `configs/rdd_reproduction/` | 18份旧RDD DDPM/FM、共享/独立模型原生配方 | 对齐对应旧RDD算法；仍须匹配数据、权重、clock、精度与评估协议 |
| `configs/comparison_imagenet256/main/` | 12份主实验，保留方法自己的尺度/日程 | 共同DiT/VAE/训练规则下比较方法，不是各论文完整系统的最佳结果 |
| `configs/comparison_imagenet256/controlled/` | 6份额外受控配置 | 在可比方法间固定尺度、日程等变量 |
| `configs/paper_recipes/` | 选定公开作者配置的原生组件配方 | 范围按配置的`evidence`和支持文档判断，不能默认每篇最佳结果全覆盖 |
| `configs/native_fixtures/`、`configs/controlled/` | 小网络数学/训练测试 | 实现验证，不作质量比较 |
| `configs/cross_modal/` | 图像/视频、共享/独立模型与预测头组合 | 共同模型上的方法移植；不是完整Wan/T2V产品管线 |
| `configs/inference_only/` | 特定公开推理流程 | 是否含训练器由具体方法决定 |

主实验当前包括RDD、PixelFlow、Scale Space、DFM、LapFlow、Pyramid Flow空间路径、PyramidalWan论文路径、f-DM、Non-Uniform/Haar及普通DDPM/flow基线。精确清单见[manifest](configs/comparison_imagenet256/manifest.json)。Relay等历史排除候选仍有源码/测试，但不重新进入主实验。

### 配置的两个层级

- **method JSON**：直接给`python -m umdiff train --config ...`；包含`model`、过程/目标/采样组件和`training`。
- **完整实验 JSON**：`format=umdiff_experiment_v1`，含`method`、`run`、`assets`、`sampling`、`comparison`；给`python -m umdiff.experiment ...`。

| 字段 | 控制内容 |
|---|---|
| `method.model` | 骨干、宽深、patch、位置编码、共享或独立阶段模型 |
| `image_shape` | 模型域的`[C,H,W]`或`[C,T,H,W]`，latent尺寸不等于最终RGB尺寸 |
| `task/process/grids/batch` | diffusion/FM路径、阶段网格、时钟、噪声结构与训练样本构造 |
| `target`或过程专用目标字段 | epsilon、velocity、joint/decoupled等，按所选实现的schema填写 |
| `boundary`或过程专用边界字段 | fresh、noise_filling、作者补噪/重标定等转阶段规则 |
| `sampler`或`process.solver/steps` | 求解器、各阶段步数、CFG；不同方法使用各自显式schema |
| `training` | 数据预处理、VAE、时间采样、loss与optimizer/EMA/累积 |
| `run` | 总更新数、global/micro batch、seed、存档频率；仅完整实验JSON使用 |

不是所有配方都用一个相同的`process`键。建议复制最接近的现成配置；入口会拒绝不支持的字段，而不是忽略拼错的参数。

## 4. 正式 ImageNet-256：plan → train → sample → decode

准备本地ImageNet训练目录（每类一个子文件夹）、本地SD-VAE-ft-ema目录。框架不会自动下载这些资产。主实验默认global256、micro32、累积8、400K更新，实际以选中JSON为准。

```bash
export IMAGENET_TRAIN_DIR=/path/to/imagenet/train
export SD_VAE_EMA_DIR=/path/to/sd-vae-ft-ema
EXPERIMENT=unified_multires/configs/comparison_imagenet256/main/rdd_ddpm_xl_legacy__train_r8-400_r16-300_r32-300__sample_r8-90_r16-80_r32-80.json

# 只解析数学和预算，不加载完整模型，也不开始训练。
python -m umdiff.experiment plan --config "$EXPERIMENT"

# 有效训练数据和VAE准备好之后执行；设备可按自己的环境改为cuda。
python -m umdiff.experiment train --config "$EXPERIMENT" \
  --data "$IMAGENET_TRAIN_DIR" --vae "$SD_VAE_EMA_DIR" \
  --device cpu --output unified_multires/runs/imagenet_rdd

# 中断后恢复到JSON指定的总updates，而不是额外再跑400K。
python -m umdiff.experiment train --config "$EXPERIMENT" \
  --data "$IMAGENET_TRAIN_DIR" --vae "$SD_VAE_EMA_DIR" \
  --resume unified_multires/runs/imagenet_rdd/checkpoint.pt \
  --device cpu --output unified_multires/runs/imagenet_rdd

python -m umdiff.experiment sample --config "$EXPERIMENT" \
  --checkpoint unified_multires/runs/imagenet_rdd/checkpoint.pt \
  --count 4 --labels 17 --sample-seed 0 --device cpu \
  --output unified_multires/runs/imagenet_rdd/sample_latents.pt

python -m umdiff decode \
  --latents unified_multires/runs/imagenet_rdd/sample_latents.pt \
  --vae "$SD_VAE_EMA_DIR" --scale 0.18215 --device cpu \
  --output unified_multires/runs/imagenet_rdd/sample_rgb.pt
```

这里列CPU命令是为了不隐式使用设备，不表示XL正式训练在CPU上实用。正式训练前先复制配置、缩小宽深和预算做小数据检查；修改后的配置不能继续引用原完整配置的数值证书。

**共同训练规则**：同一批图片覆盖所有阶段，每阶段先取batch mean，再以系数1求和；阶段内部目标、时间采样、噪声结构、位置编码和采样规则按配置保留。改变阶段数会改变计算成本，需同时记录质量与NFE/耗时。[详细比较协议](README_ZH.md)。

## 5. 复现旧RDD，而不是共同主实验

使用[原生复现配置与命令](configs/rdd_reproduction/README_ZH.md)。比如旧FM-XL权重：

```bash
python unified_multires/tools/import_legacy_rdd_checkpoint.py \
  --config unified_multires/configs/rdd_reproduction/rdd_fm_xl_v2_euler_cfg1p5_noise_filling__train_r8-300_r16-300_r32-400__sample_r8-80_r16-80_r32-90.json \
  --checkpoint /path/to/0400000.pt --weights ema \
  --output unified_multires/runs/imported_fm.pt

python -m umdiff sample --checkpoint unified_multires/runs/imported_fm.pt \
  --count 1 --labels 17 --seed 0 --device cpu \
  --output unified_multires/runs/imported_fm_latents.pt
```

导入只转换推理权重外壳，严格检查key/shape和可得clock信息；不能当作旧optimizer/数据游标的训练resume。独立三模型使用`import_legacy_rdd_multistage.py`，逐个指定`32=...`、`16=...`、`8=...`。

- `fresh`与`refresh`相同：丢弃旧残差，整张重新抽噪声。
- `noise_filling`：旧noisy和预测clean恢复残差，归一化保留，再补新增子空间。
- 旧DDPM主结果原先叫noise_filling，实际是fresh；旧FM同名选项是恢复残差。
- 原DDPM主评估低→高步数是`80/80/90`；共同主实验里的`90/80/80`是另一配方。
- FM Euler`80/80/90`和midpoint`40/40/45`均252NFE。FM旧`dpm2`别名是midpoint RK2，不是DDPM DPM-Solver。

各阶段步数在原生配方的`sampler.args.steps`或`process.steps`中设置。`sample --steps`只用于通用solver，有原生sampler的配方会明确拒绝它。默认使用EMA；`--raw-weights`才选择非EMA。

## 6. 视频与自己的组件组合

```bash
python -m umdiff configure \
  --method rdd --shape 4,3,16,16 --backbone grid_dit \
  --width 24 --depth 1 --attention-heads 3 --patch 1,2,2 \
  --grids '[[3,8,8],[3,16,16]]' --sampling-steps 4 \
  --output unified_multires/runs/video_demo.json
python -m umdiff train --config unified_multires/runs/video_demo.json \
  --synthetic --steps 2 --batch-size 1 --device cpu \
  --output unified_multires/runs/video_demo
python -m umdiff sample --checkpoint unified_multires/runs/video_demo/checkpoint.pt \
  --count 1 --seed 0 --device cpu --output unified_multires/runs/video_demo/latents.pt
```

形状含义是4个通道、3个latent帧、16×16网格。方法与骨干是分开的；`grid_dit/dit_grid/wan_grid/grid_conv`可按接口选择。`wan_grid`是共同测试骨干，不是自动加载真实预训练Wan。视频数据使用本地BCTHW tensor或shards；真实文本条件/视频VAE需要在配置和数据里显式提供，图像`decode`子命令不能直接解码视频。

数据可为本地图像目录、`.pt`中的`images`或`latents`，以及`umdiff_shards_v1`目录。Tensor输入可带`labels`和`conditions`；类别训练必须提供真实标签，文本字段必须与`training.conditioning`对应。

实现组件在`umdiff/`中按registry组合：`paths.py/multires_process.py`定义过程，`grids.py/operators.py`定义投影，`targets.py`定义预测目标，`boundaries.py`定义边界，`solvers.py/grid_sampling.py/samplers.py`定义采样，`training.py`管理更新、EMA与恢复。扩展示例：[examples/custom_components.py](examples/custom_components.py)。

## 7. 数值验证与质量评估

```bash
python -m pytest unified_multires/tests -q
```

部分测试需要相应作者源码/可选依赖；缺少时先按报错运行固定源码获取工具。完整逐层/作者对照脚本在`tools/`，它们可能需要本地历史fixture和证据目录，不能把全部脚本视为无需资产的一键测试。

已发布的紧凑RDD证据见[旧/新计算对照](../docs/validation/20261104/README.md)，数学/方法范围见[SUPPORTED_REPRODUCTIONS_ZH.md](SUPPORTED_REPRODUCTIONS_ZH.md)。本地`evidence/`的约2GB历史记录仍保留；Git不携带全部逐层trace、运行tensor和网站打包副本。旧文档中指向未发布原始证据的路径是归档引用，不是当前clone自带的数据。

此次合并的[CPU验收报告](evidence/monorepo_20261104/REPORT_ZH.md)记录183项测试，以及独立临时目录中的图像/视频训练采样、恢复和旧RDD路径验证。可用`python unified_multires/tools/validate_monorepo.py --output /path/to/check_report`重跑入口验证；临时checkpoint和tensor会自动清理，只保留报告和文本日志。

`sample`每次生成一个batch，不自动跑完整FID。正式比较需固定checkpoint、EMA、类别序列、seed、batch、采样配置和样本数，分批生成与同一VAE解码后再评估。不要把`--count`直接设成50000而一次分配所有样本。

已有RGB PNG之后，复用旧仓库的ADM评估：

```bash
python 2d/png_to_npz.py /path/to/generated_pngs \
  --num-samples 50000 --output /path/to/samples_50k.npz
# 在已准备ADM Inception图和TensorFlow依赖的环境里执行：
python guided-diffusion/evaluations/evaluator.py \
  /path/to/VIRTUAL_imagenet256_labeled.npz /path/to/samples_50k.npz
```

Inception图需放在评估器当前工作目录的`classify_image_graph_def.pb`；原评估器缺失时有下载行为，本框架离线设置并不会替它拦截。数据、VAE、ADM图及参考统计均由运行者提前准备。本次仅验证代码迁移和CPU入口，没有下载这些资产，也没有重新跑正式FID/VBench。
