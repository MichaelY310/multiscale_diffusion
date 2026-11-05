# RDD DDPM / DPM-Solver 两种边界的无 CFG 对照

下面正文可直接复制给持有 GPU 的服务器 agent。

---

请在当前 multiscale_diffusion 仓库完成两组 ImageNet256 采样与官方 ADM 五指标评估，每组 50,000 张。

目标是复查历史效果最好的 RDD DDPM 共享 XL 模型，在 DPM-Solver-2 下使用 fresh/refresh 与真正恢复旧残差的 noise_filling 的差别。不训练模型，不启动 PF 训练，不下载 ImageNet 训练集。

先读仓库 AGENTS.md、2d/AGENTS.md、docs/AGENT_GUIDE_ZH.md；保留已有 FM 实验、环境、未提交修改和其他人的任务。检查设备使用情况，使用获准空闲 GPU。可复用之前 FM 任务的四进程调度、VAE、ADM 资产与独立环境，但不能直接调用那个强制 FM/fresh 的入口来跑 DDPM。

## 模型身份：必须选这一份

- 模型：RDD-Unified-XL，DiTMultiRes-XL/2，一个模型共享三个分辨率，400K EMA。
- DDPM 阶段长度按高/中/低：300/300/400。latent 顺序 8→16→32。
- 这是历史 DPM2、CFG1.5 跑出 FID 2.6256746（40/40/45）与 2.6166875（80/80/90）的模型。
- 不要选 100/300/600、三个独立 B/L/XL 模型、Unified-L 或 FM-XL-v2。

只下载以下 checkpoint（若已存在且校验正确则复用）：

```text
HF repo: yang335/search_agent_hl_to_i18n_transport
revision: 848d8a1eccaab8c1807ab792c4ec403dde25288d
path: checkpoints/RDD-Unified-XL/0400000.pt
bytes: 10797841584
SHA256: 5c90e03baea7d1ed00ab6e9b8251cd66757e4e64548797e6adc14b9800cb361a
```

下载可用 huggingface_hub.hf_hub_download，并显式传 repo_id、revision、filename、local_dir；不得下载整个 HF repo。逐字节校验 SHA256，检查 checkpoint 中可得的模型/阶段/训练步 metadata。严格取 checkpoint['ema']，不得 fallback 到 model。若为省内存导出推理文件，逐张量比对它与原 EMA、记录源/目标 hash，不改变参数或模型结构。

复用已验证的 stabilityai/sd-vae-ft-ema，revision=f04b2c4b98319346dad8c65879f680b1997b204a，权重 SHA256=32db726da04f06c1b6b14c0043ce115cc87a501482945c5add89a40d838fcb46，latent_scale=0.18215。

## 两组正式配置

两组都用 DPM-Solver 二阶，CFG=1.0：保留 ImageNet 类别条件，不做额外 CFG；不是 unconditional label，也不是 CFG=0。

```json
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"CHECKPOINT_PATH","num_steps":"80,80,90","sampler":"dpm2","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"FRESH_OUTPUT"}
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"CHECKPOINT_PATH","num_steps":"80,80,90","sampler":"dpm2","cfg_scale":1.0,"upsample_mode":"noise_filling","boundary_semantics":"recovered_v2","output_dir":"FILLING_OUTPUT"}
```

将占位符替换为实际绝对路径，保存为 JSONL。80/80/90 是低→中→高的原生 solver 配置，历史记录实际 NFE 为 494；用 hooks 实测并报告，不能当作 250 NFE，也不能误调用 FM 中名为 dpm2 的 midpoint。

推荐独立目录：unified_multires/runs/ddpm_dpm2_boundary/ddpm_fresh_cfg1 与 ddpm_noise_filling_cfg1。两组都 seed=0、相同 per-GPU batch（优先16）、相同标签及初始噪声、相同分片与精度，VAE decode batch=1。使用 FP32，TF32 策略两组相同并记录；不改为 BF16/FP16。

优先保留 2d/sample_fid_unified.py 的原生 DDPM 阶段求解器和边界函数。它支持显式 --rank/--world-size，每个进程一张可见卡；不要只套 torchrun。每个 rank 的命令形式是：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID "$UMDIFF_PYTHON" 2d/sample_fid_unified.py \
  --config /absolute/path/to/pair.jsonl \
  --rank RANK --world-size WORLD_SIZE \
  --total-samples 50000 --batch-size 16 --vae-batch-size 1 \
  --seed 0 --vae-path /absolute/path/to/verified/vae --no-eval
```

原入口随机抽取类别，可保留；两组必须固定同一批次编号、seed、标签和所有噪声调用顺序。若采用之前的 index%1000 类别协议，则两组一起采用，并先与原生 sampler 在固定输入下核对。不能把“同 seed”直接当作标签/噪声已相同的证据。只改边界策略，其他数学组件均保持一致。

## 必须先确认两种边界确实不同

- fresh/refresh：上采样预测 clean，重新抽整张高分辨率 IID 噪声，丢弃旧残差。
- noise_filling：从旧 noisy 阶段出口与预测 clean、该出口真实 alpha/sigma 恢复残差；2× 空间上采样后除以2保留，再添加新噪声的块均值为零的细节部分。
- 不要拿 legacy_initial_noise / innovation_preserving（保留阶段最初抽的噪声）冒充恢复末端残差。
- 系数必须来自当前 DDPM diffusion 对象，不能直接套 FM 的0.3/0.6；不修改阶段累计 alpha、drift、clean 恢复、裁剪和 DPM 算法。
- 检查 _do_upsample、_redraw_upsample、_noise_filling_upsample、调用方的 previous_state/previous_diffusion，以及最终旧状态对应的实际 timestep。noise_filling 所用状态与系数必须属于同一时刻。

为回答“以前到底用了什么”，另审查旧代码，而不是根据文件夹名字或这次 FID 倒推。可检查 git show c90213b:2d/sample_fid_unified.py：该归档中名为 noise_filling 的分支调用 _redraw_upsample。对照实际历史评估配置及能找到的执行记录，给出函数/行号证据；若没有原实验逐次源码快照，要明确这个证据范围。

先用 CPU 合成 tensor 验证两种边界，再用真实 EMA 做至少一个 batch 的原生与实际运行封装对照。记录每个阶段的输入、出口、clean 预测和转阶段结果，确认首次转阶段前两实验完全相同，转阶段后按预期不同。只为预检临时保存少量 tensor，报告完成后清理。

编译仅作为可选加速：如果使用 torch.compile，先在真实权重上做完整采样轨迹 eager/compiled 数值检查，记录误差与 PNG 差异；预热不得消耗正式随机流。通过后固定后端，不在50K中途切换。

## 运行、评估与交付

1. 先各生成64张预览，确认模型、VAE、分支、CFG、实际NFE与日志，再各跑50K。正式运行不混入预览样本。
2. 为每批记录全局图片编号、标签、seed、文件hash和完成状态；两组批次定义必须相同。恢复时按原完整批次重建随机流，不能因为跳过已有图片而改变剩余图片的随机数。不要沿用旧入口仅按文件存在跳过的行为却声称严格配对。
3. 等全部 worker 成功退出，校验每组恰好50K个编号，无缺失/覆盖/损坏，统计类别数量、重复PNG及hash。保留源码commit与完整diff。
4. 用之前已验证的官方 ADM evaluator、Inception pb 和 ImageNet256 reference 计算 FID/sFID/IS/Precision/Recall，使用同一评估环境。PNG→NPZ必须 RGB uint8、(50000,256,256,3)、arr_0，latent除以0.18215后VAE解码，沿用旧uint8截断量化。
5. 如果输出是旧 sampler 的 PNG 目录，就直接使用 2d/png_to_npz.py 和 guided-diffusion/evaluations/evaluator.py；FM包 run_adm_evaluation.sh 要求专用manifest，不能直接套用或伪造完成标记。可复用其检查/打包/评估逻辑，适配真实 DDPM run 格式。
6. 不新增训练，不覆盖旧FM结果。保留最终模型、图片、配置、日志、指标和紧凑证据；可删除评估成功后可重建的巨大NPZ、临时tensor。任务结束检查自己的GPU进程全部退出。

最终给我两组五指标对照表、实际NFE、耗时、模型和VAE身份、两组唯一数学差异的证据，以及历史边界语义的核查结论。历史2.62是CFG1.5，不得把它当成本次CFG1的目标分数。不要仅生成配置就结束，需要完成两组50K和评估。
