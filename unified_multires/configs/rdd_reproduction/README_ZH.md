# 旧 RDD 图像实验的原生配方

这些raw method JSON包含原 DiTMultiRes 共享模型，以及每个分辨率单独一个原 DiT 的 MultiStage 模型。FM使用 `task=rdd_fm_native`，保留原阶段独立噪声、时间抽样、全阶段loss求和及Euler/midpoint；不要改用generic `process=rdd`后仍称历史配方。DDPM复制现有旧RDD锚点。

`manifest.json` 给出18份配置：FM-XL v2三组历史采样setting及对应fresh消融；corrected FM-L两套clock和两种noise边界；DDPM-XL CFG1/1.5历史fresh设置；另含4份 DDPM MultiStage XL/L/B（300/300/400、100/300/600，CFG1/1.5，fresh）及2份FM独立XL/L/B（fresh/noise_filling）。所有采样 `steps` 按 coarse→fine 写。独立FM未找到正式历史FID，两份边界是显式选择，不表示已有相应质量成绩。

| 模型/配置 | FM边界 | 采样更新 | CFG | 历史边界 |
|---|---|---|---|---|
| FM-XL v2 | 0/.3/.6/1 | Euler80/80/90 | 1或1.5 | noise_filling |
| FM-XL v2 | 0/.3/.6/1 | midpoint40/40/45 | 1.5 | noise_filling |
| corrected FM-L标签300_300_400 | 0/.4/.7/1 | Euler80/80/90 | 1 | noise_filling |
| corrected FM-L标签400_300_300 | 0/.3/.6/1 | Euler80/80/90 | 1 | noise_filling |
| DDPM-XL300/300/400 | 不适用 | DDPM80/80/90 | 1或1.5 | fresh（旧名称noise_filling） |

DDPM表中的80/80/90是旧评估器实际执行的coarse→fine分配。旧汇总CSV的fine_middle_coarse90/80/80是反向展示，不能当作coarse→fine。此前comparison/main锚点中的90/80/80是另一种分配，本目录已纠正为历史主FID分配；没有改旧comparison配置。

FM两种求解器都实际调用252次模型；midpoint不是DPM-Solver。fresh=丢弃旧噪声、全新IID；noise_filling=由旧noisy与predclean恢复旧粗噪声，归一化保留并补新增子空间。FM fresh配置是新消融，没有把旧noise_filling的FID算给它。

先从已有HF档案取需要的权重到本地（本任务没有代为下载），再导入：

```bash
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" unified_multires/tools/import_legacy_rdd_checkpoint.py \
  --config unified_multires/configs/rdd_reproduction/rdd_fm_xl_v2_euler_cfg1p5_noise_filling__train_r8-300_r16-300_r32-400__sample_r8-80_r16-80_r32-90.json \
  --checkpoint /your/local/0400000.pt --weights ema \
  --output /your/local/fm_xl_inference.pt

"$UMDIFF_PYTHON" -m umdiff sample \
  --checkpoint /your/local/fm_xl_inference.pt --count 1 --labels 17 \
  --seed 0 --device cpu --output /your/local/sample_latents.pt

"$UMDIFF_PYTHON" -m umdiff decode \
  --latents /your/local/sample_latents.pt --vae /your/local/sd-vae-ft-ema \
  --scale 0.18215 --device cpu --output /your/local/sample_rgb.pt
```

导入器读取本地可信checkpoint，严格检查键/形状；有resolution/clock元数据时也检查。`--weights model`显式选择非EMA权重，缺所选字段会报错，没有静默fallback。它只转换推理外壳，不恢复旧optimizer/数据游标/RNG；导入结果禁止当训练resume使用。

训练配置内VAE路径为 `@sd_vae_ema@` 占位符，先复制配置并替换成自己的本地VAE目录；训练CLI `--batch-size 256`，共享FM配方accumulation8，独立FM配方accumulation1；有效batch为256。多rank数据顺序与随机流并未因此逐位复现。旧正式XL数GB权重在CPU加载和推理仍需要足够RAM；本轮测试仅用随机小网络，不是下载/运行原权重。

证据：`../../evidence/rdd_reproduction_20261104/REPORT_ZH.md`、`fm_native_v1/RESULT.json`、`import/RESULT.json`。不要把随机共同权重的数值等价表述成历史50K FID已重跑。

## 独立 MultiStage 模型

`rdd_ddpm_multistage_xl_l_b_*` 保留原 32×32 DiT-XL/2、16×16 DiT-L/2、8×8 DiT-B/2 三套独立参数与固定网格位置编码；不把三份权重塞入同一个 DiTMultiRes。`stage_lengths` 按高→中→低，采样 `steps=[80,80,90]` 按低→中→高。

训练从同一批VAE latent开始，每阶段分别计算原loss。新框架用一个AdamW管理互不共享的三组参数；已经与原三个AdamW逐参数比较4次更新，activation、gradient、Adam状态、EMA及RNG一致。该结论限于相同超参数、同步更新三阶段的设置；不承诺独立历史多rank数据顺序或三个模型异步训练的随机轨迹。

先将每个分辨率对应的checkpoint显式指定，再使用同一个采样CLI：

```bash
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" unified_multires/tools/import_legacy_rdd_multistage.py \
  --config unified_multires/configs/rdd_reproduction/rdd_ddpm_multistage_xl_l_b_ddpm_cfg1p5_fresh__train_r8-400_r16-300_r32-300__sample_r8-80_r16-80_r32-90.json \
  --checkpoint 32=/your/local/checkpoints_32x32/0400000.pt \
  --checkpoint 16=/your/local/checkpoints_16x16/0400000.pt \
  --checkpoint 8=/your/local/checkpoints_8x8/0400000.pt \
  --weights ema --output /your/local/multistage_inference.pt

"$UMDIFF_PYTHON" -m umdiff sample \
  --checkpoint /your/local/multistage_inference.pt --count 1 --labels 17 \
  --seed 0 --device cpu --output /your/local/multistage_latents.pt

"$UMDIFF_PYTHON" -m umdiff decode \
  --latents /your/local/multistage_latents.pt --vae /your/local/sd-vae-ft-ema \
  --scale 0.18215 --device cpu --output /your/local/multistage_rgb.pt
```

对原100/300/600混合400K/600K实验，选同名100/300/600配置，并显式把 `8=.../0600000.pt`、16/32各400K填入；工具记录每份输入的SHA256和可得训练步，不猜测路径。只合并推理权重，不转换旧optimizer或恢复训练游标。上述导入工具与真实 `umdiff sample` CLI 已用随机小模型、完整3/3/3采样执行，并与原阶段循环逐元素一致；临时模型/采样张量已删除，见 `../../evidence/rdd_reproduction_20261104/ddpm_boundary/multistage_import/SUMMARY.json`。

独立FM同样用此导入器，配置换为 `rdd_fm_multistage_xl_l_b_euler_cfg1p0_fresh__train_r8-300_r16-300_r32-400__sample_r8-80_r16-80_r32-90.json` 或 `..._noise_filling.json`。导入器支持原checkpoint中的`resolution="32x32"`与`argparse.Namespace`；检查其中可得的clock/阶段区间信息。最新真实CLI对照为 `ddpm_boundary/multistage_import_v2/SUMMARY.json` 与 `fm_multistage/SUMMARY.json`。独立FM原逐阶段backward/step和unified一次总loss更新已逐参数核对一致；另8条完整采样轨迹覆盖两种边界、Euler/midpoint与CFG1/1.5。

总验收：[FINAL_SUMMARY_ZH.md](../../evidence/rdd_reproduction_20261104/FINAL_SUMMARY_ZH.md)。
