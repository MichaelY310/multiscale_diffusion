# README 历史快照（不作为当前验收结论）

这是整理公开入口前保留的历史文本。当前范围、命令和结果以 [README_ZH.md](README_ZH.md) 及其链接的支持约定为准。文中的“本轮/最新”均指当时记录，测试payload可能已按用户要求删除。

<!-- SUPPORTED_SCOPE_20261012 -->
**当前支持约定**：[保留原版核心的统一范围](SUPPORTED_REPRODUCTIONS_ZH.md)、[逐方法机器合同](SUPPORTED_REPRODUCTIONS.json)。按用户最新要求，统一数学组件、主要训练目标和采样，使用公共训练工程；不承诺复制作者全部服务、私有分支及最佳成绩。新一轮固定代码验收正在运行，最终结果将位于 [逐仓库报告](evidence/unified_scope_20261012/REPORT_ZH.md)。下文按日期保留历史检查，不能用旧测试认证后来的代码。
<!-- /SUPPORTED_SCOPE_20261012 -->

<!-- CPU_PARITY_20261012 -->
**2026-10-12 当前CPU逐层对照已执行完成**：[完整结果](CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md)、[原148项逐项处理](CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md)。12种训练/codec/蒸馏方法、33份配置（31完整image网络+完整SD式codec+完整30层Wan；Wan缩小网格）捕获的模块输入/输出、activation gradient和处理前parameter gradient均逐字节相同、max_abs=0；12方法128步和12扩展128步全部重新通过。共用作者backbone、随机非零权重、合成输入、CPU FP32；不是独立网络重写或最佳质量认证。

SPEED/NAMI/MrFlow公开推理通过、作者backward N/A。CAT官方补充4份源码新增codec适配：缩小网络16case、完整classification.py512配置8case逐层/诊断adjoint通过，无公开diffusion trainer，未计入12种训练核心。33个完整native CLI仍未认证，论文最佳主表未正式复现。原文未修复条目以本轮ledger为准。

重跑：`bash unified_multires/run_cpu_layer_parity.sh`；只用现有CPU环境，完整检查可能持续数小时。39对临时checkpoint已实际重载并比较完整model/EMA/Adam/param_groups/scalars，之后清理6.953GiB测试payload，保留全部报告/hash。下文历史checkpoint路径不保证仍有payload，不能用其恢复正式训练。
<!-- /CPU_PARITY_20261012 -->

# Unified Multiresolution Diffusion / Flow

**2026-10-11 方法等价与候选设计**：[等价层级、RDD 选择理由、领域痛点及更好性质的配置](METHOD_EQUIVALENCE_AND_DESIGN_20261011_ZH.md)。VP effective 主头等价于匹配读出的 SNR 加权 full-clean 重建；不同 product-state 噪声通常不等价。提出保留 coarse noisy state 的正交 lift、band-normalized 目标、误差/成本预算和相关误差校准方向，分别标明可证明性质、已有先例和质量未验证范围。14 组 CPU toy 代数检查通过，不是作者 parity/新算法质量认证；未修改生产实现或运行 GPU/训练。

**2026-10-11 论文定位复核**：[统一数学框架—训练实现—RDD 的近邻与理论评估](PAPER_POSITIONING_REVIEW_20261011_ZH.md)。EDM、Score-SDE、OT-CFM、SSD、DFM 等已有同样的三项贡献结构；SSD/DFM 是多分辨率任务的直接先例。补充目标/损失等价与奇异端点、Gaussian 边界先例和有限预测误差的分析，区分“包含 RDD”与“分析指导 RDD”。录用判断明确假设 parity 和公平效果优势已经完成；本次不改变实际认证状态。仅 CPU 文献与代数检查，无训练/GPU/模型数据下载或生产代码修改。

**2026-10-10 相关统一框架检索**：[已有框架与本项目比较](RELATED_UNIFIED_FRAMEWORKS_20261010_ZH.md)。算子Stochastic Interpolants与我们的路径核心有直接重叠；TorchCFM、Meta Flow Matching、DiffuseNNX、Modular Diffusers、CleanDiffuser已有多方法/模块化工程先例。当前不能声称首个通用统一框架或最高完成度；本次只读文献和公开说明，不改变原生入口认证及质量实验状态。

**未完成事项与偏离总清单**：[ORIGINAL_UNIFIED_GAPS_TODO_ZH.md](ORIGINAL_UNIFIED_GAPS_TODO_ZH.md)。按公共问题、15种方法、8篇未做作者数值复现的文献整理，并逐份列出33配置的控制项与入口阻塞。

**2026-10-10 严格复核更新**：[Original/Unified完整差异审计](evidence/strict_original_unified_audit_20261010/REPORT_ZH.md)、[机器证据](evidence/strict_original_unified_audit_20261010/AUDIT.json)。已测源码核心模块值/梯度相等，但确认Relay增强/augment_dim、EMA和缩放，LapFlow/DFM参考CLI、drop_last、条件label、MDM setup RNG等仍有实际差异；四类generic检查仅容差通过。33个native配对均未认证，不能据此前通过状态承诺原论文完整结果。此轮仅CPU审计，生产实现与历史证据未改。

**最新执行约束：只用CPU。** [GPU验证准备](GPU_CHECKS_PREPARATION_ZH.md)集中列出15种方法的待检规范、真实资产与CUDA/native-CLI适配缺口；`prepare_gpu_checks.sh`仅生成/检查计划，不执行GPU。测试清单不代表最佳结果已经复现。

**2026-10-10 成对实验入口与主要结果配方**：[使用说明](PAIRED_EXPERIMENTS_ZH.md)、[当前报告](evidence/paired_experiments_20261010/REPORT_ZH.md)。两脚本默认仅计划；33配置，30公开配方/11RDD对齐小CPU检查通过。原作者完整CLI与最佳画质仍未认证，不能用源码核心后端短跑代替原CLI复现。

本目录是独立的、离线运行的研究框架。它覆盖普通 diffusion/flow、band path、joint/decoupled target、product-state、多种跨尺度边界，以及十四篇可执行公开源码的方法（RMD 的作者归属确认有限制）和现有 RDD 的数值对照。

**先看 [逐篇复现报告](REPRODUCTION_REPORT_ZH.html)、[完整数学框架](FRAMEWORK_MATH_ZH.md) 和 [组件映射](PAPER_COVERAGE.json)。** 报告区分 paper-level 映射、source-native 数值复现、controlled transplant 和真实数据质量结论。

本轮全部使用 CPU。没有下载模型权重或训练数据。小模型训练与作者源码逐步对照通过，不表示已获得论文规模的生成质量；这些小配置也不是论文最终训练 recipe。

**产物保留策略（用户2026-10-09明确要求）**：大型临时验证 checkpoint/生成媒体在数值与恢复检查完成、报告/配置/哈希和小型证据保存后立即删除。现有 `gpu_checks_20261006` 权重与媒体已清理，保留的历史 checkpoint 元信息不能用于恢复训练；当前 unified 小型证据未删除。正式训练成果、原版权重和用户数据不按此临时策略清理。

**2026-10-09 最新状态**：[修改后逐篇正式配置复核](evidence/current_recipe_parity_20261009/REPORT_ZH.md)、[机器证据](evidence/current_recipe_parity_20261009/SUMMARY.json)、[31份配方模板及用法](configs/paper_recipes/README_ZH.md)。16个完整图像骨干配置和1个完整Wan-1.3B骨干配置各2次配对更新，实际执行共享模块的activation、activation gradient、参数gradient及模型/优化器状态逐字节一致。图像保留正式架构/输入网格；Wan视频网格缩小。缩小非零作者网络128步回归与公开采样策略另外验证；DFM uniform/lognormal及REPA各连续128步、不逐步重设seed也一致。全部CPU、随机初始化、合成输入；不是最佳FID、完整训练预算或预训练质量复现。

本轮修正PixelFlow正式patch/head packing与MSE归约、DFM公开B YAML/lognormal/原版logging RNG、PPFlow原版Heun2/250点及SDE后端、MDM250步重采样与1024三级配置、Relay完整网络/全局图像时钟/梯度修复、LapFlow论文与CLI不同flags、DCT原版预先resize契约。逐篇JSON记录所选配置、实际执行差异、未认证范围和源hash。不能从已测配置推断任意组件组合一致。

前一次 [配方补齐报告](evidence/recipe_extensions/REPORT_ZH.md) 与其28份模板/17变体记录是历史证据，保留原样。当前31模板为27份发布配置、1份本地RDD、3份最佳run未完全定位的组件/论文flags模板。真实数据、原版权重/teacher、GPU/DDP与正式质量评估尚未执行。

下述10月8日“完整配方审计”是修复前的历史状态。保留原24项发现和12个反例；当前15项具体公开分支已关闭，8项部分补齐并列出剩余范围，1项因源码未公开延后。不要在修复后的代码上把旧反例脚本当成功能回归测试；使用上面的新增验证入口。

2026-10-08 完整配方审计：[逐篇兼容性细节](evidence/compatibility_audit/REPORT_ZH.html)、[24项代码证据与待补工作](evidence/compatibility_audit/AUDIT.json)、[12个CPU反例](evidence/compatibility_audit/PROBES.json)。发现原版 DCTdiff 频率 loss 权重、PPFlow/DiMR/DFM/LapFlow/MDM 的引导采样、MDM mixed-resolution batch、Relay 双模型完整链、微调初值转换等未完整接入；SSD rescale 和 RDD 非恒等 timestep_map 还存在训练/采样网络时间契约缺口。此前128步及逐层一致性只覆盖其专用配置，不外推这些分支。所有12类训练方法的 `full_paper_recipe_coverage` 均标为false；这不撤销已测分支的数值证据。

本次审核包含10个隔离分支探针，另以真实缩小的作者 UViT 验证频率权重会改变参数梯度，以真实 PPFlow 两/三层网络和作者ODE sampler验证CFG会改变终态。没有修改生产算法或作者文件；确认差异并不表示缺项已修复。复跑：使用下述CPU环境后依次运行 `"$UMDIFF_PYTHON" unified_multires/tools/audit_compatibility_gaps.py` 和 `"$UMDIFF_PYTHON" unified_multires/tools/build_compatibility_audit.py`。

2026-10-08 补验：[逐层 activation 报告](evidence/activation_parity/REPORT_ZH.html)与[机器汇总](evidence/activation_parity/SUMMARY.json)。11 篇训练论文和本地 RDD，共 21 种变体各 128 步；390,544 对模块输入、348,384 对模块输出、227,552 对 activation 梯度及 178,336 对参数梯度均通过 shape/dtype、数值和原始字节一致性检查，最大差 0。21 对最终模型、EMA（如有）和完整优化器状态也与此前无 hooks 的结果一致。每个执行模块的调用次数及未执行模块均保存在逐篇 JSON 中。

范围是**实际执行的共享作者网络模块边界**，不是 `forward` 内每个 functional/ATen 运算，也不覆盖未执行分支。Diffusability 比较共享 VAE 的内部执行模块；原版 BTCHW facade 根接口和 unified BCHW 接口不同，根接口不作同名模块匹配，已有 mean/logvar/latent/reconstruction 断言仍保留；额外无 loss 的 encoder 诊断调用不计入 trace。DiMR 的 backward checkpoint 重算同样逐层比较。所有作者源码保持原样。

补验复跑：先使用下述 CPU 环境，再运行 `"$UMDIFF_PYTHON" unified_multires/tools/validate_activations.py --selftest`、`validate_activations.py --all --steps 128 --workers 2` 和 `build_activation_report.py`（后两个脚本同样使用完整 tools 路径）。可用 `--method rdd` 等逐篇执行。补验 checkpoint 保存在 `runs/activation_parity/activation_parity/`，此前 `runs/native_parity/` 结果保留。

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

embedding 字典至少含 `prompt_embeds`、`pooled_prompt_embeds`；NAMI CFG>1 另需 negative 两项。文本编码器不在这个入口内隐式构造。`runs/offline_entrypoints/{speed,nami,mrflow}/` 中有经过测试的本地权重/embedding/config 加载示例；那些权重是本轮从零随机初始化的 fixture。

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
"$UMDIFF_PYTHON" unified_multires/tools/validate_all.py --training-steps 128
"$UMDIFF_PYTHON" unified_multires/tools/validate_cli.py
"$UMDIFF_PYTHON" unified_multires/tools/validate_offline_entrypoints.py
"$UMDIFF_PYTHON" unified_multires/tools/build_report.py
```

`validate_all.py` 逐个独立进程运行 15 组组件、12 类实际训练、3 类完整 inference-only workflow 、RMD 原版训练入口、DFM 连续 RNG 对齐与数学契约测试。每一步 loss/gradient/parameter/EMA 检查保存在 `evidence/*_training.json`；成对原版/统一实现的本地训练 checkpoint 在 `runs/native_parity/`；实际采样 tensor 对及 native/unified/绝对差分 PNG 在 `runs/parity_outputs/`，报告可直接预览并下载原始 tensor；CLI 恢复样例在 `runs/cli_validation/`。

需要只重跑一个方法时可用 `validate_all.py --filter training/rdd`；保持原有 `--training-steps`，否则需完整重跑。

`evidence/sources.json` 记录源码 commit 与文件 hash；`PREIMPLEMENTATION_REVIEW.json` 保留实现前审读时间；`NUMERICAL_DIFFERENCES.json` 记录发现、原因与修复。上游源码未改写，必要兼容修复均位于 adapter/oracle namespace；源文件完整性由最终审计检查。

训练 oracle 执行作者 loss、模块方法或原始训练语句体，使用相同骨干与受控 AdamW/EMA；除 RMD 另测实际 `train()` 外，各仓库分布式 launcher 未纳入本轮执行范围。

CPU 数值测试固定实际输入、stage/time/noise，而不仅设置相同 seed；原仓库的日志抽样、数据 loader、CPU/GPU RNG 与内存布局也会影响同 seed 的序列。DFM 的 `loss.py:202/233` 两处随机 debug logging 会推进 NumPy RNG；连续 global-seed 对齐可停用这两个随机日志条件，或为日志另设 RNG。该操作只作为复现说明，当前固定源码未改写。

CPU 数值测试使用与作者一致的算术顺序。PixelFlow 两端共同使用 CPU SDPA，尚未验证与作者 GPU FlashAttention kernel 的 bitwise 一致性。DCTdiff/DiMR 复用作者 DPM-Solver backend，LapFlow 使用相同 torchdiffeq 求解器；报告明确标注了这些复用。

## 依赖、授权和论文实验范围

当前可执行环境和依赖版本见 `evidence/ENVIRONMENT.json`；安装信息见 `requirements-authors.txt`。基础框架依赖 torch/numpy/scipy/Pillow，作者网络额外依赖 diffusers/transformers/timm/einops 等。`_deps/` 只放本次隔离安装的少量 Python 依赖，没有改共享环境。`sources/` 保留各作者许可证，详见 [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md)。

全部验证默认强制 CPU/offline。当前任务 GPU 授权仅 surfi1 的物理 7 号卡、最多一张，截止 2026-10-08 洛杉矶 09:00（16:00 UTC）。若本次任务中确需使用，必须经独立 deadline guard；截止后 CLI 拒绝 GPU。当前证据没有使用 GPU，未启动长训练或旧监控，也未修改共享 dirty FastVideo。

本轮交付基础训练/采样框架与数值复现。仍需用户已有的真实数据、匹配的完整模型规模/训练预算/codec/teacher/评测资产，才能检验论文 FID 和“至少与论文一样好”。`PAPER_COVERAGE.json`、报告及 checkpoint metadata 都保留这一范围，不把小模型或合成像素的相同说成真实生成质量已经认证。
