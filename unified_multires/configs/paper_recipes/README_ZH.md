# 正式配方模板与本地运行

这些是结构与组件配置，不是已经获得论文指标的 checkpoint。31份模板中，27份导出发布配置，1份来自本地RDD训练器，3份是尚未完全定位论文最佳实验的组件/论文flags模板（LapFlow CLI与论文flags、Diffusability）。每份 JSON 的 `evidence.notes` 列出仍需匹配的条件。没有模型或数据下载。

先看 [最新复核报告](../../evidence/current_recipe_parity_20261009/REPORT_ZH.md)、[独立源配置审计](../../evidence/current_recipe_parity_20261009/config_audit.json) 和 [INDEX.json](INDEX.json)。全部完整架构的 meta 构造通过；16个完整图像骨干及完整Wan骨干另有2步CPU配对更新。原版权重没有在本次测试中加载，随机初始化短跑和meta检查不能代替完整训练。

|方法|模板|必须准备的本地输入|当前边界|
|---|---|---|---|
|DCTdiff|13份 `dctdiff_*`|正确split的图像；新数据的bound/entropy统计|UViT发布配置；论文未公开DiT不覆盖|
|DiMR|`dimr_DiMR-XL-2R_imagenet256`、`dimr_DiMR-G-2R_imagenet256`|作者 `.npy` moments+label；原版VAE用于解码|包含发布LR/EMA、dropout、CFG；完整FID未跑|
|PixelFlow|`pixelflow_xl_c2i`|ImageNet原split图片|4stage、patch4/head72、30步/阶段与CFG2.4是公开eval默认；CPU SDPA不认证GPU FlashAttention|
|PPFlow|`ppflow_xl_2`、`ppflow_xl_3`|ImageNet、原VAE、SiT初始化|本地SiT严格转换；原版Heun2/250点；4/5stage/FLUX发布缺失|
|DFM|`dfm_sit_xl`、`dfm_sit_b_released_yaml`、`dfm_sit_xl_repa_localfeatures`|原VAE；REPA另需带teacher features的latent分片|B YAML使用lognormal/BF16、generation30+10；native_debug_rng显式兼容logging draw；released architecture与paper不同|
|LapFlow|`lapflow_mot_xl_2`、`lapflow_imagenet_xl_2_paper`|原版VAE/EQVAE、数据；CLI另冻结epochs与steps_per_epoch|CLI T.9/LR5e-4/EMA.9与论文T.5/LR1e-4/EMA.9999分开；最佳CFG未定位；source square sampling有bug|
|MDM|`mdm_cc12m256`、`mdm_cc12m1024`|RGB、已编码Flan-T5-XL特征与mask；1024严格本地inner256权重|完整2/3级网络与250步resampling已测；真实预训练课程/在线reader未认证|
|RDD|`rdd_unified_xl_300_300_400`|ImageNet与原VAE|本地DiTMultiRes三损失程序；完整三层质量尚无结果|
|Diffusability|`diffusability_se24`|原VAE、VGG19参数与图像|codec训练；之后必须重提latent并重训生成器才能评价生成质量|
|Relay|`relay_imagenet_low/high`|64/256原预处理数据、两个网络初始化或各自训练结果|README完整低/高网络及图像时钟；全局4096/DDP/xformers未执行|
|RMD|`rmd_wan13b`|Wan transformer、prompt/negative embeddings|Wan发布配方；GPU BF16、GAN/其它骨干及视频质量未认证|

模板中的 `@vae@`、`@eqvae@`、`@sit@`、`@vgg@`、`@wan@`、`@mdm_inner256@` 必须指向**已有本地资产**。用 `--asset mdm_inner256=/absolute/path/to/inner256_state.pt` 等绑定。文件存在只说明能够尝试加载；仍需核对版本、精度和实际权重。不要把项目里的合成权重当作这些资产。

```bash
cd /scr/yuheng/bugswarm/rdd_workspace_20261005
source unified_multires/activate_cpu.sh

# 例：先绑定原版本地资产，并核验固定源文件的hash。
"$UMDIFF_PYTHON" unified_multires/tools/render_paper_recipe.py ppflow_xl_2 \
  --data /absolute/path/to/imagenet_train \
  --asset vae=/absolute/path/to/original_diffusers_vae \
  --asset sit=/absolute/path/to/sit_state_dict.pt \
  --output unified_multires/runs/formal/ppflow2.json
```

CPU功能检查使用缩小 `configs/native_fixtures`，不要直接在CPU启动XL网络的大训练。纸面模板拒绝 `--synthetic`。正式训练需要在新GPU授权及独立guard下运行；原授权已经在2026-10-08 LA09:00结束，CLI不会自动续到第二天。本文不提供可绕过期限的GPU命令。

小真实数据pilot需逐篇检查：正确预处理/label或text features→初始化加载→forward/backward→optimizer/EMA→checkpoint恢复→guided inference→使用同一VAE解码。随后按原版有效global batch、训练预算和官方评估器运行。修改component组合时结论属于controlled transplant，不再自动继承native一致性。

数据入口支持本地图像目录、float BCHW `.pt`（含labels/conditions）、DiMR作者moments记录，以及分片tensor目录。分片只保留一个shard常驻，单个shard需能装入内存；全局随机shuffle可能跨shard频繁读取，应按实际I/O选择布局。它不是原版CC12M线上reader的逐位复现。

```json
{
  "format": "umdiff_shards_v1",
  "shards": [
    {"file": "00000.pt", "count": 1024},
    {"file": "00001.pt", "count": 1024}
  ]
}
```

每个 `.pt` 为 `{"images": tensor, "labels": labels, "conditions": {...}}` 或将images换成latents。MDM训练条件字段为 `lm_outputs` 和 `lm_mask`；CFG采样另有 `negative_lm_outputs` 和 `negative_lm_mask`（sample-only，模板映射原版negative-first顺序）；DFM REPA为 `teacher_features`。全量 `.pt` 会整体加载，较大数据用分片。训练恢复核对路径/文件size/mtime、manifest、batch和seed；这不是整个数据集内容hash，应额外冻结数据清单。旧无data_binding的checkpoint无法直接通过新版绑定CLI恢复；原数值证据仍可用Trainer/native后端读取。

latent训练模板默认 `.18215` 缩放及本地SD系VAE，不可用任意VAE替换而仍声称原版质量。导入已缩放latent时移除 `training.input_transform`，保持尺度一致。推理输出latent后可用同一原版image VAE解码：

```bash
"$UMDIFF_PYTHON" -m umdiff.cli sample \
  --checkpoint /absolute/path/to/checkpoint.pt --labels 0 1 --count 2 \
  --output unified_multires/runs/formal/latents.pt

"$UMDIFF_PYTHON" -m umdiff.cli decode \
  --latents unified_multires/runs/formal/latents.pt \
  --vae /absolute/path/to/original_diffusers_vae --scale .18215 \
  --output unified_multires/runs/formal/pixels.pt
```

`decode`只支持BCHW image AutoencoderKL；Wan视频需要原版Wan VAE及其latent规范化，不适用上述image命令。MDM生成RGB无需VAE，采样使用text conditioning文件。RMD条件格式见distillation_cli和发布源配置。

Relay分别训练low/high后，使用 `umdiff.workflows.run` 的 `name=relay` 工作流；提供 `models.low/high`、严格tensor state字典 `checkpoints.low/high` 和one-hot `class_labels`。该工作流保留原版EDM→nearest→blur顺序。源训练器按图像数LR/EMA和特殊pixel-sum归约已提供；模板loss multiplier按单进程有效batch512换算，image_count_multiplier=8模拟源码全局图像时钟；单进程梯度与8-rank DDP不等价，改变batch/world size必须重新计算。模板显式配置原版nan_to_num梯度修复。

LapFlow epoch scheduler可配置 `training.optimizer.lr_schedule={"name":"cosine_epoch","steps_per_epoch":...,"epochs":...,"minimum_ratio":...}`；这里采用显式epoch clock，原版Accelerate包裹scheduler的历史行为需要另外确认。MDM VDM可将time_sampler设为one-based discrete并传 `vdm_gammas`，optimizer的 `loss_reduction=normalized_weights`；发布CC12M配置默认关闭VDM，不要自动改成开启。

复跑新增CPU分支：

```bash
"$UMDIFF_PYTHON" unified_multires/tools/validate_recipe_extensions.py dctdiff --steps 128
# 其它位置参数：dimr ppflow mdm mdm_vdm ssd dfm lapflow lapflow3 relay rdd diffusability
"$UMDIFF_PYTHON" unified_multires/tools/validate_recipe_runtime.py
"$UMDIFF_PYTHON" unified_multires/tools/validate_recipe_auxiliary.py
"$UMDIFF_PYTHON" unified_multires/tools/validate_recipe_entrypoint.py
"$UMDIFF_PYTHON" unified_multires/tools/check_paper_recipes.py
"$UMDIFF_PYTHON" unified_multires/tools/audit_current_profile_configs.py
"$UMDIFF_PYTHON" unified_multires/tools/validate_dfm_current_rng.py
"$UMDIFF_PYTHON" unified_multires/tools/build_current_profile_report.py
```

`prepare_dct_statistics.py --arrays Y.npy Cb.npy Cr.npy --block 2 --output stats.json`使用作者原始bound/entropy公式。字段虽然叫 `*_std`，实为100bin histogram entropy并按原版保留3位小数，不能用通常的标准差替代。输入是本地raw raster-order DCT系数数组，不是已归一化tokens。
