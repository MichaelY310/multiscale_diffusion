# 在新服务器上先跑 RDD-FM fresh，再训练 Pyramid Flow

两个正式入口：`run_fm_fresh.sh`、`run_pf_train.sh`。交给服务器agent的完整任务见[AGENT_PROMPT_ZH.md](AGENT_PROMPT_ZH.md)。所有下列命令从Git仓库根目录执行。

## 固定实验协议

| 项目 | RDD-FM fresh采样 | PF训练与后续采样 |
|---|---|---|
| 模型 | 已有FM Unified-XL v2、400K、EMA | 同尺寸DiT-XL/2从头训练；不会加载RDD训练好的权重 |
| 架构 | width1152/depth28/heads16/patch2，保留原8通道输出中的前4个FM通道 | 相同主体和输出槽；使用已有PF RoPE移植，额外sigma槽不参与FM目标 |
| 表示 | 同一SD-VAE-ft-ema，latent×0.18215 | 同一VAE，每次访问posterior抽样后构建latent金字塔 |
| 网格 | 8→16→32，解码256×256 | 相同 |
| 名义阶段 | 0/.3/.6/1 | 相同；保留PF补噪后的实际起点修正及作者network clock |
| 数学 | 原RDD-FM路径、fresh丢弃旧噪声 | PF noisy endpoints插值、原补噪缩放；native_psd实现原奇异协方差的平方根 |
| 外层训练loss | 旧模型每阶段mean求和 | 同一图片覆盖三阶段，三个mean以系数1相加 |
| Euler更新数 | 80/80/90，另有两次边界预测 | 81/81/90 |
| 每尺度实际NFE | 81/81/90，共252 | 相同 |
| CFG | 默认1；可单独运行1.5作为额外实验 | 当前该入口只支持1，传1.5会明确拒绝 |
| 优化/预算 | 已完成的历史训练 | AdamW lr1e-4、betas0.9/0.999、eps1e-8、weight_decay0、EMA0.9999、固定lr、400K更新 |
| batch/精度 | 默认采样batch4，VAE解码batch1 | global256，默认micro1/累积256，FP32、允许TF32 |
| 主评估 | 50K，标签image_index%1000，固定batch/seed | 后续使用相同标签、batch/seed、VAE与评估实现 |

PF原作者名义阶段是三等分；作者训练脚本阶段样本比例1:2:1。本实验显式更改为上述共同设置，不称作PF原论文最佳系统复现。保留PF RoPE意味着比较仍含PE差异，不称“只有概率路径不同”。微批次/单卡RNG和归约与旧RDD八卡训练不同；旧RDD权重作为历史参考，不冒称一对同初值同硬件新训练。双方阶段内部velocity单位和时间映射仍按各自数学定义。

原作者也是**先VAE编码，再在latent上构建金字塔**；此前主配置里“每个RGB尺度分别编码”的说明已更正。这里只更换共同VAE及latent缩放，不把这一点误报为原版先后顺序不同。

官方来源：[训练入口](https://github.com/jy0205/Pyramid-Flow/blob/a012faa1dc4d71301a7a153c7f9554c081947ea2/train/train_pyramid_flow.py)、[阶段scheduler](https://github.com/jy0205/Pyramid-Flow/blob/a012faa1dc4d71301a7a153c7f9554c081947ea2/diffusion_schedulers/scheduling_flow_matching.py)、[latent构造](https://github.com/jy0205/Pyramid-Flow/blob/a012faa1dc4d71301a7a153c7f9554c081947ea2/pyramid_dit/pyramid_dit_for_video_gen_pipeline.py)。这两项实验不需要下载作者PF模型或完整作者源码。

## 1. 环境

Linux，Python3.10–3.12（推荐3.11/3.12），NVIDIA CUDA环境。训练入口目前是**单进程、单GPU**，不要套`torchrun`。建议至少32GB主机RAM，64GB更从容；GPU能否容纳完整XL必须实测，不能只凭容量承诺。先用micro1检查，显存不足时不要悄悄改成小模型或BF16。

```bash
EXP=unified_multires/experiments/imagenet256_fm_pf
bash "$EXP/setup_environment.sh"
export UMDIFF_PYTHON="$PWD/.venv-fm-pf/bin/python"
```

脚本建立独立`.venv-fm-pf`，安装固定Torch2.8.0/torchvision0.23.0和[依赖](requirements.txt)。默认CUDA12.8 wheel；根据驱动可设置`TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`；纯CPU检查用`https://download.pytorch.org/whl/cpu`。不要修改其他项目的共享环境。Python版本通过`BOOTSTRAP_PYTHON=/path/to/python3.12`选择。自定义环境目录用`FM_PF_ENV_DIR`，并相应设置`UMDIFF_PYTHON`。

没有FlashAttention、编译扩展或作者权重的硬依赖。新服务器先做无需模型/数据下载的CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' UMDIFF_CPU_ONLY=1 \
  "$UMDIFF_PYTHON" -m pytest unified_multires/tests/test_fm_pf_launch.py -q
```

这里环境变量仅作用于该条命令，不会让后续GPU任务保持CPU-only。正式GPU命令前选择一张允许使用的空闲卡，例如`export CUDA_VISIBLE_DEVICES=0`；编号必须以新服务器实况为准。

## 2. 下载精确资产

```bash
"$UMDIFF_PYTHON" "$EXP/workflow.py" assets
# 已有文件时只校验：
"$UMDIFF_PYTHON" "$EXP/workflow.py" assets --verify
```

默认保存到`checkpoints/fm_pf_assets/`；可用`--assets /large/disk/fm_pf_assets`指定大磁盘，后面的采样/训练命令也传同一参数。

| 资产 | 精确来源 | 大小 |
|---|---|---:|
| 原RDD-FM XL v2 400K | [yang335/search_agent_hl_to_i18n_transport](https://huggingface.co/yang335/search_agent_hl_to_i18n_transport/blob/848d8a1eccaab8c1807ab792c4ec403dde25288d/checkpoints/RDD-FM-Unified-XL-v2/0400000.pt)，路径`checkpoints/RDD-FM-Unified-XL-v2/0400000.pt` | 10,797,841,648bytes |
| SD-VAE-ft-ema | [固定revision](https://huggingface.co/stabilityai/sd-vae-ft-ema/tree/f04b2c4b98319346dad8c65879f680b1997b204a)，只取`config.json`和`diffusion_pytorch_model.safetensors` | 334,643,823bytes |

[assets.lock.json](assets.lock.json)固定revision、大小和hash。两个模型源在准备时均为公开非gated，不需要下载完整HF大仓库。下载器遇到已有错误hash会停止，不覆盖。只有显式`assets`动作联网；训练/采样离线。网路需要账户时使用服务器自己的HF登录，不能把token写进脚本、Git或聊天。

首次采样严格导入EMA，生成`fm_ema_inference.pt`，约2.7GB；原10.8GB文件含optimizer等状态，不代表DiT需要10.8GB权重。导入只用于推理，不用于PF初始化或训练resume。

## 3. 先跑FM fresh采样

先64张检查，接着50K正式样本：

```bash
bash "$EXP/run_fm_fresh.sh" --count 64 --cfg 1 \
  --output unified_multires/runs/fm_pf/fm_preview

bash "$EXP/run_fm_fresh.sh" --count 50000 --cfg 1 \
  --output unified_multires/runs/fm_pf/fm_fresh_cfg1
```

产物：`preview.png`、`png/000000.png…`、`run.json`、逐批hash记录和`SUMMARY.json`。默认batch4/解码batch1；显存不足可在正式运行前改`--batch-size 1`，之后必须固定。每批seed为`base_seed+batch_start`，标签为全局序号%1000；这是明确的新共同评估协议，不保证与旧评估器随机标签抽取逐样本对应。PNG量化沿用旧评估器的截断方式。

重跑**同一条命令**会校验并跳过完整批次；中途只写了一部分的批次会重生成。更改checkpoint、cfg、batch、count、环境或seed必须使用新输出目录。脚本每次只加载一次模型和VAE，不保存每批latent/tensor。它输出图像，不伪造FID。

额外CFG1.5实验：加`--cfg 1.5`并使用另一个目录。该结果可以与旧noise_filling的CFG1.5作参考，但本次PF主对比是双方CFG1。新fresh结果不能继承旧noise_filling的已报告FID。

## 4. ImageNet数据准备

首选已有原始ImageNet-1K train目录：

```text
/datasets/imagenet/train/
  n01440764/*.JPEG
  n01443537/*.JPEG
  ... 共1000类、1,281,167张训练图
```

不要离线预缩放成256、不要使用固定抽样的latent代替在线VAE。loader执行ADM crop/flip，VAE每次访问重新抽posterior。不同缓存方式可以另做实验，不能默默更改本配方。

如果已有合法取得的`ILSVRC2012_img_train.tar`：

```bash
"$UMDIFF_PYTHON" "$EXP/prepare_imagenet.py" \
  --archive /datasets/ILSVRC2012_img_train.tar --output /datasets/imagenet/train
```

脚本流式解包内层class tar，保留原编码字节，不留额外class tar副本，也不会覆盖不同内容的文件。原压缩包不会自动删除。

没有原tar时，可使用[HF官方ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)。用户必须先在该页获得访问权限，服务器登录有权限的账号；agent不能代替用户接受条款。它的parquet不能直接交给ImageFolder，因此提供原字节导出器：

```bash
"$UMDIFF_PYTHON" -m pip install 'datasets==4.1.1'
# 必要时交互登录：.venv-fm-pf/bin/hf auth login
"$UMDIFF_PYTHON" "$EXP/prepare_imagenet.py" --hf --output /datasets/imagenet/train
```

仅流式读取train split并保存原图字节，不下载validation/test，不重新JPEG编码。固定首次访问的dataset revision，保留数据来源和label顺序；使用`0000…0999`文件夹，按官方label索引排序。中断可重跑，已存在图像会比较原字节。权限缺失时必须说明阻塞原因，不能改用其他数据集、预裁剪镜像或合成数据开始正式训练。

原数据是百GB级；原tar与解包同时保留需要两份空间。除已有ImageNet外，建议为模型、环境、PNG和训练checkpoint预留至少100GB。采样不保存latent，训练只保留latest及50K/100K/200K/400K里程碑；一个完整XL训练checkpoint约10.8GB，原子保存还需要临时空间。runner在训练chunk前要求至少32GiB可用空间。

## 5. 再跑PF训练

```bash
export IMAGENET_TRAIN_DIR=/datasets/imagenet/train

# 完整XL + 真实图片 + 真实VAE，实际做一次参数更新；属于正式训练第1步。
bash "$EXP/run_pf_train.sh" --preflight-only

# 同一目录自动从第1步/最近checkpoint恢复，最终总计400K次更新。
bash "$EXP/run_pf_train.sh"
```

默认输出`unified_multires/runs/fm_pf/pf_train/`。预检包含类别/图片数量、32张图片解码、完整网络forward/backward、AdamW/EMA和checkpoint；数据检查不是全量内容认证。`progress/`记录loss、实际运行环境、耗时和峰值显存。`PREFLIGHT.json`只有成功完成实际更新后才生成。

GPU有余量时可在**第一次参数更新之前**使用`--micro-batch 2/4/8/...`，必须整除256；后续命令保持相同参数。不能中途改microbatch并声称RNG轨迹不变。`--updates`表示目标总更新数，用于显式短期pilot；主实验默认400000，不能擅自改小后称完成。

建议使用调度系统或tmux持久运行，例如：

```bash
# 在已选GPU、配置好环境变量的shell中启动；tmux窗口里运行上面的正式命令。
tmux new -s pf-imagenet
```

不要只报告“已启动”就说训练完成。保存PID/session、完整命令、日志、checkpoint路径和当前step。中断后重新执行同一训练命令，会从latest继续。每1000步保存latest，每5000步结束一个worker并记录进度；外层继续下一段。milestone保留50K/100K/200K/400K，正式checkpoint不是临时垃圾。

## 6. PF采样与质量比较

训练结束后，以相同EMA、batch、标签和seed生成PF图像：

```bash
"$UMDIFF_PYTHON" "$EXP/workflow.py" pf-sample \
  --checkpoint unified_multires/runs/fm_pf/pf_train/checkpoint.pt \
  --count 50000 --cfg 1 --batch-size 4 \
  --output unified_multires/runs/fm_pf/pf_400k_cfg1
```

如果FM用了另一个batch，这里也同步。两边共252NFE但仍需报告实际采样时间，不能把调用次数说成完整系统耗时相等。当前PF sampler仅CFG1，显式拒绝其他CFG，避免仅修改metadata却未改变模型输出。

正式指标评估现已补齐独立环境安装、官方下载、文件校验和评估入口。完整说明见[EVALUATION_ZH.md](EVALUATION_ZH.md)。FM50K完成后先评估FM；PF训练及50K采样完成后用同一协议评估PF：

```bash
BOOTSTRAP_PYTHON="$PWD/.venv-fm-pf/bin/python" bash "$EXP/setup_evaluation.sh"
bash "$EXP/run_adm_evaluation.sh" unified_multires/runs/fm_pf/fm_fresh_cfg1
# PF训练和采样完成之后：
bash "$EXP/run_adm_evaluation.sh" unified_multires/runs/fm_pf/pf_400k_cfg1
```

输出每个run下的`evaluation/metrics.json`和`adm.log`，含FID、sFID、IS、Precision、Recall。下载官方Inception约96MB和参考NPZ约2.04GB；CPU评估环境与训练环境分开，正式NPZ临时文件约9.83GB。只有5项指标保存成功后才清理该临时文件。当前训练和采样仍为单进程单GPU，不支持直接套torchrun。

## 本地验证范围

见[VALIDATION_ZH.md](VALIDATION_ZH.md)。此处CPU验证采用随机小DiT和随机小VAE，实际执行保存/加载、编解码、完整252NFE采样、PF更新/恢复。没有在本机下载正式VAE/FM权重或ImageNet，没有使用GPU或开始正式训练。新服务器必须执行真实XL预检。
