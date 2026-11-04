# RDD / DiT 实验总账

最后核对：2026-09-03

这份文件是 2D ImageNet-256 实验的单一总账。结果按证据等级严格分区：

- **A — 可用于论文**：原生 ImageNet ILSVRC 数据，checkpoint 路径含 ImageNet_original，并能与评估日志对应。
- **B — 可能有价值但指标未闭环**：配置和采样任务使用原生 ImageNet checkpoint，但没有找到完整 50K 指标，或评估程序失败。
- **S — 仅速度有效**：速度不依赖训练数据内容，可以用于工程对比；但当前脚本不是完整端到端论文 latency。
- **H — 仅历史启发**：旧 Hugging Face 预处理数据训练；数值绝不能与 A 级结果混表或写入论文主结论。

除非单独说明，A 级 FID/IS/sFID/Precision/Recall 均为 50K 样本、ADM ImageNet-256 reference statistics、VAE EMA latent、center crop 的评估结果。

## 1. 参数与记号

### 1.1 训练公共参数

| 参数 | 值 |
|---|---|
| 数据 | 原生 ImageNet ILSVRC；data-source=imagenet |
| 图像 / latent 分辨率 | 256×256 / 32×32 |
| VAE | ema |
| 类别数 | 1000 |
| diffusion horizon | T=1000 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 0 |
| global batch | 256，8 GPUs |
| seed | 0 |
| epochs 参数 | 1400 |
| augmentation | center crop + random horizontal flip |

训练入口：train_RDD_imagenet256_300_300_400_original.sh、train_RDD_imagenet256_100_300_600_original.sh、train_RDD_imagenet256_unified_XL_original.sh、train_RDD_imagenet256_unified_L_original.sh、DiT/train_imagenet256_original.sh。

### 1.2 模型登记

| 简称 | 模型类型 | 每个分辨率的 backbone | 训练分段 [32,16,8] | 参数量 |
|---|---|---|---:|---:|
| DiT | baseline | DiT-XL/2 at 32 | 单阶段 1000 | 675,129,632 |
| RDD Unified XL | unified/shared | DiTMultiRes-XL/2，共享跨尺度 | 300/300/400 | 674,834,720 |
| RDD Unified L | unified/shared | DiTMultiRes-L/2，共享跨尺度 | 300/300/400 | 457,840,672 |
| RDD MS-300 | multistage | 32: XL/2；16: L/2；8: B/2 | 300/300/400 | 1,263,363,936 |
| RDD MS-100 | multistage | 32: XL/2；16: L/2；8: B/2 | 100/300/600 | 1,263,363,936 |
| Decouple | multistage/decoupled | 32: XL/2；16: L/2；8: B/2 | 300/300/400 | 速度文件未单列 |

MS-300 的参数拆分为 675,129,632 + 457,906,208 + 130,328,096。

### 1.3 最容易混淆的两个口径

1. 训练分段统一写成 **[32,16,8]**。采样实际顺序是 8→16→32；sample_fid_unified.py 和 JSONL 的 num_steps 按 **[8,16,32]** 保存。因此 JSON 的 80,80,90 在本文件写成 [32,16,8]=90/80/80。
2. JSON 的 sampler=dpm2 中，“scheduled steps”不等于 NFE。代码在二阶步同时评估端点与中点；三阶段实际 **NFE = 2 × scheduled total − 6**。因此：

| DPM2 scheduled total | 实际 NFE |
|---:|---:|
| 250 | 494 |
| 125 | 244 |
| 65 | 124 |
| 35 | 64 |

DDPM 的 scheduled total 等于 NFE。后文凡比较 DDPM 与 DPM2，都同时标 scheduled steps 和真实 NFE。

CFG=1.0 表示没有 guidance amplification；早期文件名中的 cfg0、nocfg 与它等价。

## 2. A 级：原生 ImageNet 正确结果

### 2.1 DiT-XL/2 baseline

| Ckpt | 采样 | Scheduled / NFE | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ | 备注 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 400K | DDPM | 250 / 250 | 1.0 | **19.781288** | 6.163877 | 67.587814 | 0.6348 | 0.6221 | 用户确认的 dit400k_250step_ddpm_cfg0.txt |
| 400K | DDPM | 250 / 250 | 1.25 | 10.261 | 5.189 | — | — | — | 只找到 FID/sFID |
| 500K | DDPM | 250 / 250 | 1.0 | **17.474150** | 6.1855 | 75.1660 | 0.6426 | 0.6287 | eval_results_dit500k |
| 500K | DDPM | 250 / 250 | 1.0 | 17.463568 | — | — | — | — | exp04 独立重复；与上一行差 0.011 |

dit500k_eval.jsonl 还配置了 500K、DDPM250、CFG1.5，但未找到完整指标，归入 B 级。

### 2.2 RDD Unified XL，400K，训练分段 [32,16,8]=300/300/400

#### 主结果

| 采样 | Sampling split [32,16,8] | Scheduled / NFE | CFG | 边界 | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| DDPM | 90/80/80 | 250 / 250 | 1.0 | noise filling | **7.06155** | 4.33458 | 130.0995 | 0.7167 | 0.6303 |
| DDPM | 90/80/80 | 250 / 250 | 1.5 | noise filling | **3.537371** | 5.165530 | 275.5237 | 0.86806 | 0.5070 |
| DPM2 | 90/80/80 | 250 / 494 | 1.5 | noise filling | **2.617** | 4.534 | 245.61 | 0.837 | 0.541 |

第一行的远端重复 rdd_400K.txt 为 FID 7.06155；收敛曲线的独立评估为 7.06508。

#### DDPM / DPM2 采样预算

| Sampler | Split [32,16,8] | Scheduled | 真实 NFE | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPM | 90/80/80 | 250 | 250 | 1.5 | 3.537 | 5.166 | 275.52 | 0.868 | 0.507 |
| DDPM | 45/40/40 | 125 | 125 | 1.5 | 3.888 | 6.122 | 270.40 | 0.864 | 0.499 |
| DDPM | 25/20/20 | 65 | 65 | 1.5 | 4.429 | 8.052 | 258.67 | 0.853 | 0.479 |
| DDPM | 15/10/10 | 35 | 35 | 1.5 | 5.973 | 12.461 | 228.76 | 0.820 | 0.444 |
| DPM2 | 90/80/80 | 250 | 494 | 1.5 | 2.617 | 4.534 | 245.61 | 0.837 | 0.541 |
| DPM2 | 45/40/40 | 125 | 244 | 1.5 | 2.626 | 5.090 | 237.54 | 0.825 | 0.554 |
| DPM2 | 25/20/20 | 65 | 124 | 1.5 | 3.325 | 7.648 | 218.91 | 0.800 | 0.575 |
| DPM2 | 15/10/10 | 35 | 64 | 1.5 | 11.761 | 24.489 | 149.94 | 0.672 | 0.573 |

计算匹配后仍有一个很强的结论：DPM2 125 scheduled / 244 NFE 的 FID=2.626，优于 DDPM 250 NFE 的 3.537，降低 25.8%；DPM2 65 scheduled / 124 NFE 的 3.325，也优于 DDPM 125 NFE 的 3.888，降低 14.5%。但 DPM2 在约 64 NFE 出现明显稳定性崩塌：11.761 对 DDPM 65 NFE 的 4.429。

所以“RDD 400K 时 DPM2 比 DDPM 好”成立，而且在约 250 NFE 的公平比较下仍成立；但不能把 DPM2 的 2.617 写成“250 NFE”。

#### CFG sweep：DDPM250，noise filling

| CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 7.061 | **4.335** | 130.10 | 0.717 | **0.631** |
| 1.5 | **3.537** | 5.166 | 275.52 | 0.868 | 0.507 |
| 2.0 | 8.335 | 7.177 | 364.36 | 0.921 | 0.394 |
| 2.5 | 13.184 | 9.503 | 412.62 | 0.942 | 0.303 |
| 4.0 | 20.594 | 14.813 | 464.30 | 0.947 | 0.162 |

CFG1.5 是 FID 最优点。继续增大 CFG 只提高 IS/precision，却快速损害 FID、sFID 和 recall。

#### Sampling-step allocation：DDPM，total=250，CFG1.5，noise filling

| Split [32,16,8] | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---:|---:|---:|---:|---:|
| 150/50/50 | **3.404** | **4.907** | 274.93 | 0.864 | 0.508 |
| 130/60/60 | 3.500 | 4.973 | 274.10 | 0.865 | 0.510 |
| 90/80/80 | 3.537 | 5.166 | 275.52 | 0.868 | 0.507 |
| 70/80/100 | 3.624 | 5.464 | 275.27 | 0.866 | 0.507 |
| 60/70/120 | 3.701 | 5.648 | 273.13 | 0.865 | 0.502 |

在固定 250 步下，把更多预算给最终 32×32 阶段更好。此前“低分辨率越多越好”的文字结论是反的。

#### 推理分段与训练分段不匹配

模型 checkpoint 固定为训练分段 300/300/400，只修改推理时传入的 stage_lengths；DDPM250、CFG1.5、noise filling。

| 推理 stage_lengths [32,16,8] | FID↓ |
|---:|---:|
| 300/300/400（matched） | **3.537** |
| 250/250/500 | 9.183 |
| 400/300/300 | 25.023 |
| 200/300/500 | 40.282 |

这是 inference-boundary mismatch，不是重新训练的 training-split ablation。结果说明模型高度依赖训练时的时间边界，论文中不能把它描述为“训练分段消融”。

#### Stage boundary 初始化

DDPM250、CFG1.0、同一 Unified XL 400K checkpoint：

| 边界初始化 | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---:|---:|---:|---:|
| noise filling | **7.061** | **4.335** | **130.10** | **0.717** | 0.631 |
| bilinear | 9.543 | 4.847 | 116.70 | 0.683 | **0.640** |

noise filling 将 FID 降低约 26.0%，是最稳定、最值得保留的机制消融之一。

### 2.3 RDD Unified L，400K，300/300/400

| 采样 | Split [32,16,8] | Scheduled / NFE | CFG | 边界 | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| DDPM | 90/80/80 | 250 / 250 | 1.0 | noise filling | **8.461** | 4.429 | 120.10 | 0.705 | 0.630 |

CFG1.5 的样本曾进入评估，但 evaluator 因 NumPy 的 np.long 已移除而失败；没有合法指标。

### 2.4 RDD Multistage，XL+L+B

#### 主要 checkpoint / 边界结果

| Train split [32,16,8] | Ckpt | Sampler | Sampling split [32,16,8] | Scheduled / NFE | CFG | 边界 | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 300/300/400 | 400K | DDPM | 90/80/80 | 250/250 | 1.5 | noise filling | **3.814852** | 5.263469 | 218.8609 | 0.84202 | 0.5083 |
| 300/300/400 | 400K | DDPM | 90/80/80 | 250/250 | 1.0 | noise filling | 11.829104 | 4.575259 | 98.4415 | 0.68026 | 0.6282 |
| 300/300/400 | 400K | DDPM | 90/80/80 | 250/250 | 1.5 | bilinear | 9.611802 | 6.02734 | 181.1417 | 0.77694 | 0.4203 |
| 300/300/400 | 400K | DPM1 | 90/80/80 | 250/250 | 1.5 | bilinear | 11.060892 | 6.41520 | 152.8729 | 0.73502 | 0.4502 |
| 300/300/400 | 400K | DDPM | 90/80/80 | 250/250 | 1.0 | bilinear | 14.367 | 5.194 | 90.23 | 0.651 | 0.632 |
| 100/300/600 | 400K | DDPM | 80/80/90 | 250/250 | 1.0 | noise filling | **17.332152** | 5.023189 | 79.0045 | 0.63302 | 0.6274 |
| 100/300/600 | 400K | DDPM | 80/80/90 | 250/250 | 1.5 | noise filling | **5.042842** | 5.723004 | 181.8104 | 0.80612 | 0.5121 |
| 100/300/600（仅 8×8 继续到 600K；16/32 仍为 400K） | mixed | DDPM | 80/80/90 | 250/250 | 1.0 | noise filling | **16.852790** | 5.043862 | 80.6173 | 0.63572 | 0.6210 |
| 100/300/600（仅 8×8 继续到 600K；16/32 仍为 400K） | mixed | DDPM | 80/80/90 | 250/250 | 1.5 | noise filling | **4.974628** | 5.930731 | 184.3072 | 0.80786 | 0.5132 |

400K CFG1.0 的正确 FID 是 11.829，不是旧汇总里带星号的 5.282。原始采样输出目录虽然含 `380k`，日志实际加载三个 `0400000.pt`；checkpoint 元数据显示它们从 `0380000.pt` 恢复后继续训练并保存，因此这里应标作 400K。

100/300/600 两项均由 50K 完整样本计算；原始 NPZ 仍保存在 surfi2。早期 evaluator 在 Precision/Recall 阶段使用了 NumPy 已移除的 `np.bool`，因此旧日志只留下 FID、sFID、IS。评估器现已改为 `np.bool_`，两组都已直接复用原 NPZ 补齐五项，无需重新采样。仅继续训练 8×8 的 mixed 对照原本就有完整五项。

仅把 8×8 从 400K 继续到 600K，CFG1.0 FID 只改善 0.479（17.332→16.853），CFG1.5 只改善 0.068（5.043→4.975）。相对于同为 250-step 的 MS-300 baseline FID 差距，它们分别只追回约 8.7% 和 5.6%；而 CFG1.0 recall 从 0.6274 变为 0.6210，反而略降。因此“100/300/600 很差主要因为 8×8 checkpoint 少训练 200K”不受数据支持。

在相同 250-step CFG1.5 设置下，100/300/600 的 FID=5.043，较 300/300/400 的 3.815 高 32.2%；CFG1.0 则为 17.332 对 11.829，高 46.5%。因此 schedule 劣化是真实的，不是单一 CFG 偶然现象。与此同时，它比错误 HF 数据训练的旧 MS-100、80/80/90、CFG1.5（FID=10.680）改善 52.8%，所以不能把模型判定为训练失败。

注意：`FID=3.627037`、sFID=4.847330、IS=218.577、P=0.84316、R=0.5176 属于下面的 **200/200/200、总 600-step** 采样，不能填入 90/80/80、总 250-step 行。早期汇总曾把这两行混在一起，现已按 Hugging Face 原始 evaluator 日志纠正。

#### 400K DDPM allocation，CFG1.5，noise filling

| Split [32,16,8] | Total NFE | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 200/200/200 | 600 | **3.627037** | 4.847330 | 218.577 | 0.84316 | 0.5176 |
| 150/50/50 | 250 | 3.700 | — | — | — | — |
| 130/60/60 | 250 | 3.793 | — | — | — | — |
| 120/120/160 | 400 | 3.715 | — | — | — | — |
| 70/80/100 | 250 | 3.905 | — | — | — | — |
| 60/70/120 | 250 | 4.013 | — | — | — | — |
| 50/40/40 | 130 | 4.196 | — | — | — | — |
| 40/30/30 | 100 | 4.411 | — | — | — | — |

#### 400K DPM2，CFG1.5，noise filling

| Split [32,16,8] | Scheduled | 实际 NFE | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 45/40/40 | 125 | 244 | **4.379** | 5.438 | 180.61 | 0.789 | 0.558 |
| 25/20/20 | 65 | 124 | 5.855 | 8.458 | 165.38 | 0.766 | 0.570 |
| 15/10/10 | 35 | 64 | 17.442 | 26.933 | 111.10 | 0.632 | 0.554 |

当前没有 Multistage 400K、DPM2 scheduled250 的合法结果。

### 2.5 正确数据的训练收敛曲线

统一设置：DDPM250、CFG1.0、RDD 使用 noise filling。这里纳入所有已有完整曲线且训练达到至少 400K 的正确原生 ImageNet 模型：RDD Unified L/2、RDD Unified XL/2 和 DiT-XL/2。Unified XL 的 450K 按用户确认视为丢失，留空且不插值。

| Checkpoint | RDD Unified L FID↓ | RDD Unified XL FID↓ | DiT-XL/2 FID↓ |
|---:|---:|---:|---:|
| 50K | 50.943079 | 47.411347 | 77.149702 |
| 100K | 23.186205 | 20.488017 | 47.327367 |
| 150K | 16.098189 | 13.754593 | 35.741985 |
| 200K | 12.840071 | 10.663906 | 29.661975 |
| 250K | 11.222006 | 9.250047 | 26.055215 |
| 300K | 10.080764 | 8.149241 | 23.125130 |
| 350K | 9.215080 | 7.510125 | 20.913408 |
| 400K | 8.456287 | 7.065080 | **19.781288** |
| 450K | 8.130312 | — | 18.599495 |
| 500K | 7.749682 | 6.460807 | 17.463568 |

远端存在名为 rdd_450K.txt、FID=6.7301 的对象，但用户明确确认 450K 正确结果已经丢失；它不进入 A 级表或论文曲线。

三模型完整图：paper/figures/fid_convergence_all_original.{svg,pdf,png}；数据：eval_results/convergence_all_original/fid_convergence_all_original.tsv；生成脚本：paper/experiments_scripts/plot_fid_convergence_all_original.py。旧的双模型论文图仍保留在 paper/figures/fid_convergence.pdf。

包含 Unified-L 三种三阶段分配、两阶段 500/500、Unified-XL diffusion、Original DiT-XL/2 和 RDD-FM 400K 单点的扩展图见：`eval_results/unified_l_stage_comparison/unified_l_stage_allocation_fid_with_dit_fm.{png,pdf,svg}`。FM 只有 400K 完整评估，因此图中只画单点，不连接或插值成收敛曲线。

### 2.6 RDD-FM Unified XL v2，400K

这组 checkpoint 使用原生 ImageNet 和 Unified XL/2。它是 legacy v2：`stage_lengths=300,300,400` 在 FM 代码里按 **coarse/middle/fine** 时间顺序解释，得到边界 `[0,0.3,0.6,1]`；它不是 DDPM `[32,16,8]=300/300/400` 的严格时间镜像。严格镜像应使用 FM coarse/middle/fine=`400/300/300`，边界 `[0,0.4,0.7,1]`。

| Solver | Sampling steps [32,16,8] | Updates / actual NFE | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Euler | 90/80/80 | 250 / 252 | 1.0 | **14.617391** | 5.844362 | 92.5354 | 0.65072 | 0.6390 |
| Euler | 90/80/80 | 250 / 252 | 1.5 | 3.358714 | **4.591564** | 235.0467 | 0.84276 | 0.5062 |
| Midpoint RK2 | 45/40/40 | 125 / 252 | 1.5 | **3.290897** | 4.620143 | 235.4825 | 0.84292 | 0.5122 |

采样脚本的 `num_steps` 按实际 coarse→fine 顺序 `[8,16,32]` 保存，因此表中 90/80/80 对应 JSON `80,80,90`；表中 45/40/40 对应 JSON `40,40,45`。Euler 比 updates 多出的 2 NFE，以及 Midpoint 的 252 NFE，都包含前两个 stage 结束时用于 drift-aware clean inverse 的 endpoint evaluation。

在与收敛曲线一致的 CFG1.0 口径下，legacy RDD-FM 400K FID14.617，明显差于 Unified-XL diffusion 400K 的 7.065，也差于 Unified-L diffusion 的 8.456；因此扩展图不能把 FM 描述成更快收敛。CFG1.5 下 FM 的结果很强，但它属于不同 guidance operating point，不能直接放进 CFG1.0 收敛曲线。

## 3. B 级：2d/exp 中尚未闭环的正确数据实验

检查到的 2d/exp JSONL 全部指向 ImageNet_original；下面只表示“配置正确”，没有指标时绝不补数。

| 文件 | 配置覆盖 | 状态 |
|---|---|---|
| ablation_cfg1.jsonl | Unified XL 400K：DPM2 与 DDPM，scheduled 250/125/65/35，CFG1.0；三个 inference split mismatch；Multistage 400K DPM2 250/125/65/35，CFG1.0 | 15 项。只有与 A 级主结果重合的 Unified DDPM250 有可靠 FID；其余未在远端结果目录找到完整指标 |
| ablation_batch_400k.jsonl | Multistage / Unified allocation；bilinear；Multistage 100/130/400/600/1000-step | 15 项。多数已映射到 A 级表；第 15 项 num_steps JSON=[300,300,400] 在采样时失败 |
| new_ablation_400k.jsonl | Unified / Multistage DPM2 和 DDPM 低步数，CFG1.5；Unified DPM2-250；三种 mismatch；CFG2/2.5/4 | 16 项，已映射到 A 级 solver、mismatch、CFG 表 |
| exp04_rdd_convergence.jsonl | Unified XL 50/100/150/200/250/300/350/500K | 8 项，已映射到收敛表；400K 来自主实验 |
| dit500k_eval.jsonl | DiT500K DDPM250 CFG1.0 / 1.5 | CFG1.0 已有；CFG1.5 未找到完整指标 |
| ms100300600_400k_noisefilling.jsonl | MS-100 400K、DDPM250、CFG1.0 / 1.5、noise filling | 2 项；采样配置/日志存在，未找到完整评估指标 |
| main_model_*.jsonl | Multistage、Unified L、Unified XL 主配置 | 合法结果已映射到 A 级 |

ablation_batch_400k.jsonl 第 15 项失败原因是每个 stage 的 scheduled section 与训练区间不整除，日志报错：“cannot divide section of 300 steps into 400”。它没有 FID。

最值得立即补跑/补评估的是：

1. Unified XL 400K、CFG1.0 的 DDPM/DPM2 全预算矩阵；
2. Multistage 400K、CFG1.0 的 DPM2 矩阵；
3. MS-100 400K、CFG1.0/1.5 的 50K FID；
4. DiT500K CFG1.5；
5. Unified L CFG1.5（修复 np.long 后重评，无需重采样）。

## 4. S 级：speed_benchmark

这些速度仍有参考价值，因为前向/反向计算量不依赖模型由哪一版数据训练；但必须准确描述脚本口径。

### 4.1 训练 microbenchmark

benchmark_training_speed.py 是 **单张 H20、per-GPU batch=32** 的模型计算 microbenchmark，5 次 warmup、20 次计时。它没有实际运行 8-GPU DDP；元数据中的 “8 GPUs / global batch 256” 只是目标训练配置。因此相对时间、显存和 FLOPs 可用，绝对 8-GPU throughput 不可直接引用。

| 模型 | ms/step↓ | std | image/s/GPU↑ | peak GB↓ | 估算 FLOPs/step | 相对 DiT 时间 |
|---|---:|---:|---:|---:|---:|---:|
| DiT-XL/2 | 542.73 | 0.53 | 59.0 | 27.71 | 1.037e12 | 1.000× |
| MS-300 | 665.26 | 0.90 | 48.1 | 36.49 | 1.225e12 | 1.226× |
| MS-100 | 668.39 | 0.48 | 47.9 | 36.49 | 1.225e12 | 1.232× |
| Unified XL | 682.82 | 3.45 | 46.9 | 38.80 | 1.360e12 | 1.258× |
| Unified L | **488.18** | 1.43 | **65.5** | 29.06 | 0.923e12 | **0.899×** |

按实测 microbenchmark wall-clock，Unified XL 400K 约等于 DiT 503.2K 的训练时间；按脚本估算 FLOPs，约等于 DiT 524.8K。此时正确收敛表仍是 RDD400K FID7.065 对 DiT500K FID17.464，优势很大。

### 4.2 采样 250 scheduled steps

benchmark_paper.py：每个 worker 独占一张 H20，batch=16，3 warmup + 10 timed batches；DDPM p_sample_loop，单次 class-conditional forward，**没有 CFG 双前向**。RDD 固定 [32,16,8]=90/80/80，边界实现为 bilinear 后加固定标准差噪声，接近但不等同完整评估的 noise-filling 公式。因此主要用于比较模型计算结构。

| 模型 | batch ms↓ | ms/image↓ | 相对 DiT 加速↑ |
|---|---:|---:|---:|
| DiT-XL/2 | 21,335.0 | 1,333.4 | 1.000× |
| MS-300 | 9,544.9 | 596.6 | **2.235×** |
| MS-100 | 9,576.6 | 598.5 | 2.228× |
| Decouple | 9,574.6 | 598.4 | 2.228× |
| Unified XL | 11,254.8 | 703.4 | 1.896× |
| Unified L | **8,335.1** | **520.9** | **2.560×** |

all_speed_benchmark_results.json 仍把 250-step 标成 NOT_MEASURED，但根目录 benchmark_250_speed.log 是后续成功运行，应以后者为准。paper_benchmark.json 只持久化了 Original 这一行，结构化结果并不完整。

### 4.3 采样 1000 scheduled steps

单张 H20、batch=256、1 warmup + 3 timed batches；同样是 DDPM 单前向、无 CFG。

| 模型 | ms/image↓ | 相对 DiT 加速↑ |
|---|---:|---:|
| DiT-XL/2 | 9,917.31 | 1.000× |
| MS-300 | 3,468.09 | 2.860× |
| MS-100 | **1,517.12** | **6.537×** |
| Decouple | 3,468.13 | 2.860× |
| Unified XL | 3,906.03 | 2.539× |
| Unified L | 2,723.73 | 3.641× |

1000-step 的 MS-100 与 MS-300 巨大差异来自训练分段/实际各 stage 执行长度；250-step benchmark 对两者都强制 90/80/80，所以延迟几乎相同。

### 4.4 脚本估算的采样 FLOPs

| 模型 | 250 scheduled | 1000 scheduled |
|---|---:|---:|
| DiT | 86.417 T | 345.666 T |
| MS-300 / MS-100 / Decouple | 36.133 T | 122.952 T |
| Unified XL | 39.734 T | 138.206 T |
| Unified L | 26.958 T | 93.766 T |

这些 FLOPs 是脚本近似值。DPM2 必须按真实 NFE 重新计数，不能把 scheduled steps 直接代入。

## 5. H 级：旧 HF 数据训练结果，仅用于产生假设

**警告：本节模型使用错误的 Hugging Face 预处理数据。所有绝对 FID 都不能用于论文、不能与第 2 节做数值比较。** 旧文档里的 “original” 只表示原始 DiT 架构，不表示原生 ImageNet 数据。

历史原始表保存在 docs/research_docs/existing_experiments.txt 和 docs/theory/experiment_plan.md。本节保留与 sampler、分段、边界和 CFG 相关的完整关键网格。

### 5.1 旧 DiT baseline

#### 400K

| Scheduled | Sampler | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1000 | DDPM | 1.5 | 12.38 | 5.85 | 133.79 | 0.754 | 0.432 |
| 1000 | DDPM | 1.25 | 17.64 | 6.73 | 95.19 | 0.669 | 0.485 |
| 1000 | DDPM | 1.0 | 28.20 | 8.15 | 59.58 | 0.565 | 0.539 |
| 250 | DDPM | 1.5 | 12.61 | 6.26 | 132.76 | 0.746 | 0.434 |
| 250 | DDPM | 1.25 | 18.19 | 7.38 | 93.52 | 0.662 | 0.483 |
| 250 | DDPM | 1.0 | 28.80 | 9.02 | 58.78 | 0.561 | 0.544 |

#### 500K

| Scheduled | Sampler | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | DDPM | 1.0 | 34.34799 | 15.80877 | 56.76685 | 0.53168 | 0.5264 |
| 50 | DPM2 | 1.0 | 28.09873 | 11.15973 | 56.31889 | 0.55550 | 0.5911 |
| 100 | DDPM | 1.0 | 28.83111 | 11.04257 | 62.17416 | 0.56194 | 0.5372 |
| 100 | DPM2 | 1.0 | 28.45803 | 9.44943 | 57.77638 | 0.55186 | 0.5824 |
| 250 | DDPM | 1.0 | **26.26764** | 8.79077 | 65.5040 | 0.57396 | 0.5467 |
| 250 | DDPM | 1.25 | 16.10691 | 7.17228 | 106.1851 | 0.68190 | 0.4862 |
| 250 | DDPM | 1.5 | **11.15917** | 6.07187 | 150.7715 | 0.76342 | 0.4227 |
| 250 | DPM1 | 1.0 | 28.55195 | 8.42660 | 58.8481 | 0.55154 | 0.5734 |
| 250 | DPM2 | 1.0 | 28.29295 | 8.78245 | 58.4571 | 0.55362 | 0.5757 |
| 250 | DPM1 | 1.5 | 12.69678 | 6.19152 | 126.8378 | 0.72588 | 0.4663 |
| 250 | DPM2 | 1.5 | 12.44501 | 6.40073 | 126.6596 | 0.72496 | 0.4795 |

旧 DiT 的现象不是简单的“DDPM 永远更好”：无 CFG、50/100 scheduled 时 DPM2 有帮助；到 250 scheduled，尤其 CFG1.5 时 DDPM 更好。由于旧文档没有可靠的真实 NFE/实现版本登记，这只能启发当前的 NFE×CFG sweep。

### 5.2 旧 Multistage 400K，训练分段等价于当前 [32,16,8]=300/300/400

以下 source step 顺序保持历史文件原样，均为 **[8,16,32]**。

#### Sampling split [8,16,32]=80/80/90

| 边界 | CFG | Sampler | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---|---:|---:|---:|---:|---:|
| bilinear | 1.5 | DDPM | **9.322807** | 6.01456 | 182.7489 | 0.78104 | 0.4255 |
| bilinear | 1.5 | DPM1 | 10.717529 | 6.43690 | 156.0523 | 0.73558 | 0.4561 |
| bilinear | 1.5 | DPM2 | 10.313238 | 6.50190 | 156.1015 | 0.74048 | 0.4711 |
| bilinear | 1.25 | DDPM | **13.004807** | 6.44227 | 128.6087 | 0.69884 | 0.4829 |
| bilinear | 1.25 | DPM1 | 15.964749 | 7.22343 | 109.9339 | 0.64820 | 0.5064 |
| bilinear | 1.25 | DPM2 | 15.538524 | 7.47234 | 109.6244 | 0.65258 | 0.5204 |
| bilinear | 1.0 | DDPM | **22.151179** | 7.23775 | 78.8312 | 0.58602 | 0.5417 |
| bilinear | 1.0 | DPM1 | 26.303664 | 8.27400 | 67.6355 | 0.54430 | 0.5633 |
| bilinear | 1.0 | DPM2 | 25.851063 | 8.74331 | 67.5657 | 0.54856 | 0.5755 |
| constant | 1.5 | DDPM | **8.864813** | 5.60170 | 193.5644 | 0.80334 | 0.4159 |
| constant | 1.5 | DPM1 | 9.536919 | 5.59117 | 170.4140 | 0.76690 | 0.4513 |
| constant | 1.5 | DPM2 | 9.108075 | 5.80968 | 168.7808 | 0.76056 | 0.4666 |
| constant | 1.25 | DDPM | **11.644734** | 5.68558 | 139.1958 | 0.72498 | 0.4784 |
| constant | 1.25 | DPM1 | 13.591450 | 6.09918 | 122.1192 | 0.68778 | 0.4965 |
| constant | 1.25 | DPM2 | 13.476994 | 6.47675 | 119.7535 | 0.68584 | 0.5131 |
| constant | 1.0 | DDPM | **19.788986** | 6.28376 | 85.9055 | 0.61568 | 0.5305 |
| constant | 1.0 | DPM1 | 22.752114 | 7.10843 | 75.9937 | 0.58778 | 0.5631 |
| constant | 1.0 | DPM2 | 22.827783 | 7.66284 | 73.9555 | 0.58440 | 0.5752 |

#### Sampling split [8,16,32]=100/100/50

| 边界 | CFG | Sampler | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---|---:|---|---:|---:|---:|---:|---:|
| bilinear | 1.5 | DDPM | **9.719502** | 6.35206 | 179.3329 | 0.77788 | 0.4128 |
| bilinear | 1.5 | DPM1 | 11.013554 | 6.46500 | 154.8838 | 0.73248 | 0.4498 |
| bilinear | 1.5 | DPM2 | 10.265136 | 6.54735 | 155.9052 | 0.74164 | 0.4743 |
| bilinear | 1.25 | DDPM | **13.598743** | 6.56615 | 125.2802 | 0.69400 | 0.4719 |
| bilinear | 1.25 | DPM1 | 16.320504 | 7.19533 | 109.1543 | 0.64638 | 0.5036 |
| bilinear | 1.25 | DPM2 | 15.539745 | 7.59582 | 109.4422 | 0.65482 | 0.5238 |
| bilinear | 1.0 | DDPM | **23.062262** | 7.17829 | 77.6164 | 0.58052 | 0.5332 |
| bilinear | 1.0 | DPM1 | 26.780986 | 8.17944 | 67.1284 | 0.54070 | 0.5605 |
| bilinear | 1.0 | DPM2 | 25.923621 | 8.96720 | 67.3702 | 0.54964 | 0.5812 |
| constant | 1.5 | DDPM | 9.205242 | 6.05795 | 188.8285 | 0.79804 | 0.4160 |
| constant | 1.5 | DPM1 | 9.659915 | **5.61216** | 171.2453 | 0.76380 | 0.4452 |
| constant | 1.5 | DPM2 | **9.112026** | 6.05032 | 167.8310 | 0.76202 | 0.4741 |
| constant | 1.25 | DDPM | **12.100662** | 5.93851 | 136.6205 | 0.72152 | 0.4583 |
| constant | 1.25 | DPM1 | 13.816363 | 6.20087 | 122.0105 | 0.68512 | 0.4980 |
| constant | 1.25 | DPM2 | 13.575317 | 6.99795 | 119.3255 | 0.68334 | 0.5277 |
| constant | 1.0 | DDPM | **20.146991** | 6.28025 | 84.7130 | 0.61380 | 0.5233 |
| constant | 1.0 | DPM1 | 23.214855 | 7.20979 | 75.5205 | 0.58138 | 0.5526 |
| constant | 1.0 | DPM2 | 23.262851 | 8.37775 | 73.1783 | 0.58540 | 0.5723 |

在这 12 个同配置单元格中，DPM1 相对 DDPM 为 12/12 更差，FID 中位劣化约 15.1%；DPM2 为 11/12 更差，中位劣化约 13.3%。唯一例外是 constant、100/100/50、CFG1.5：DPM2 9.112 对 DDPM 9.205，只好约 1.0%。所以用户观察到的旧实验趋势非常强，但不是字面意义上的“永远”。

#### 旧 allocation：DDPM、CFG1.0、constant

| Split [8,16,32] | FID↓ |
|---:|---:|
| 100/100/50 | 20.14699 |
| 80/80/90 | 19.78899 |
| 60/60/130 | 19.59756 |
| 50/50/150 | 19.42996 |
| 40/40/170 | **19.29334** |
| 20/30/200 | 19.65978 |

尽管绝对值无效，这个方向与当前正确数据一致：在一定范围内，增加最终 32×32 阶段的采样预算更有利；过度压缩低分辨率阶段后又会回退。

#### 旧 1000-step

| Split [8,16,32] | Sampler | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 400/300/300 | DDPM | 1.5 | 8.520285 | 5.33339 | 193.987 | 0.8041 | 0.4346 |
| 400/300/300 | DDPM | 1.25 | 11.54665 | — | — | — | — |
| 400/300/300 | DDPM | 1.0 | 19.78743 | — | — | — | — |

### 5.3 旧 MS-100、Decouple、Unified

#### Multistage，训练分段等价于当前 [32,16,8]=100/300/600，400K

| Split [8,16,32] | Sampler | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150/75/25 | DDPM | 1.5 | 11.62164 | 7.43990 | 158.913 | 0.75278 | 0.4066 |
| 150/75/25 | DDPM | 1.25 | 16.61724 | — | — | — | — |
| 150/75/25 | DDPM | 1.0 | 26.67779 | — | — | — | — |
| 80/80/90 | DDPM | 1.5 | 10.67977 | 6.34347 | 163.826 | 0.76104 | 0.4206 |
| 80/80/90 | DDPM | 1.25 | 15.21561 | — | — | — | — |
| 80/80/90 | DPM1 | 1.0 | 24.97352 | — | — | — | — |
| 600/300/100 | DDPM | 1.5 | 10.78128 | — | — | — | — |
| 600/300/100 | DPM1 | 1.0 | 25.67838 | — | — | — | — |

#### 旧 700K 冲突记录

原文 heading 写 multistage-400-300-400，行内 model type 却写 multistage-600-300-100，不能可靠推断训练分段；只保留数值，不用于 schedule 结论。

| Source label | Split [8,16,32] | Sampler | CFG | FID↓ |
|---|---:|---|---:|---:|
| conflict | 80/80/90 | DDPM | 1.5 | 7.74112 |
| conflict | 80/80/90 | DDPM | 1.25 | 9.52178 |
| conflict | 80/80/90 | DPM1 | 1.0 | 16.29336 |
| conflict | 400/300/300 | DDPM | 1.5 | 7.41317 |
| conflict | 400/300/300 | DPM1 | 1.0 | 16.54438 |

#### Decouple 400K，300/300/400，noise filling

| Scheduled | Split [8,16,32] | Sampler | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1000 | 400/300/300 | DDPM | 1.5 | 8.338 | 5.345 | 202.842 | 0.808 | 0.428 |
| 1000 | 400/300/300 | DDPM | 1.0 | 19.008 | — | — | — | — |
| 250 | 80/80/90 | DDPM | 1.5 | 8.734 | — | — | — | — |
| 250 | 80/80/90 | DDPM | 1.25 | 11.309 | — | — | — | — |
| 250 | 80/80/90 | DDPM | 1.0 | 19.109 | — | — | — | — |

#### Unified XL 400K，300/300/400，noise filling

| Scheduled | Sampler | CFG | FID↓ | sFID↓ | IS↑ | P↑ | R↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000 | DDPM | 1.5 | 7.426 | 5.765 | 235.749 | 0.813 | 0.435 |
| 1000 | DDPM | 1.0 | 16.731 | — | — | — | — |
| 250 | DDPM | 1.5 | 7.747 | — | — | — | — |
| 250 | DDPM | 1.25 | 9.562 | — | — | — | — |
| 250 | DDPM | 1.0 | 16.955 | — | — | — | — |

#### Unified L 400K，300/300/400，noise filling

| Scheduled | Sampler | CFG | FID↓ |
|---:|---|---:|---:|
| 1000 | DDPM | 1.5 | 7.916 |
| 1000 | DDPM | 1.0 | 18.640 |
| 250 | DDPM | 1.5 | 8.170 |
| 250 | DDPM | 1.25 | 10.845 |
| 250 | DDPM | 1.0 | 18.864 |

## 6. 关键发现与解释

### 6.1 当前 RDD 的 DPM2 优势是真的，但原先的计算口径不公平

- 同 scheduled250：DPM2 FID2.617 比 DDPM3.537 好 26.0%，但 DPM2 实际是 494 NFE。
- 近似同 NFE：DPM2 scheduled125 / 244 NFE 的 FID2.626，仍比 DDPM250 / 250 NFE 的 3.537 好 25.8%。
- DPM2 从 124 NFE 的 3.325 到 64 NFE 的 11.761，存在明确的稳定性拐点。

这比“训练没充分所以 DDPM 更好 / 训练充分所以 DPM 更好”的单因果解释更具体。当前证据说明 **模型、跨尺度边界、CFG、NFE 与二阶中点 drift 重算共同决定结果**；仅用 checkpoint step 不能解释旧/新差异。

### 6.2 为什么旧 HF 实验几乎总是 DDPM 更好，而当前 Unified XL 相反

可检验的解释，按优先级排序：

1. **边界分布匹配**：当前 noise filling 显著优于 bilinear；旧实验大量使用 bilinear/constant。高阶 solver 对 stage boundary 的 off-manifold 初值更敏感。
2. **统一模型的向量场更连续**：Unified 跨尺度共享参数，可能比三个独立模型在交界处更平滑，更适合二阶局部近似。
3. **当前 DPM2 的 midpoint drift 重算**：实现不是简单复用起点 drift；这个细节可能是主要收益来源。
4. **CFG 交互**：旧 DiT 在低步、无 CFG 时 DPM2 反而改善，而 250-step CFG1.5 时 DDPM 更好，说明 guidance 会改变 solver 稳定性。
5. **数据与预处理校准**：原生 ImageNet 改变了训练分布、FID reference 对齐和 score calibration。
6. **训练成熟度**：可能有影响，但旧/新同时改变了数据、边界、架构和实现，现有证据无法单独归因。

### 6.3 其他值得注意的点

- RDD Unified XL 与 DiT-XL/2 参数量几乎相同，却在每个正确 checkpoint 都显著更快收敛；即使按训练 wall-clock 或估算 FLOPs 对齐，400K RDD 对 500K DiT 仍有大幅 FID 优势。
- Unified XL CFG1.5 的 FID 最好，但 CFG1.0 的 recall 和 sFID 更好；论文应同时报告 precision/recall，避免把 CFG 带来的 mode dropping 当成纯质量提升。
- 当前正确 allocation 与旧错误数据的方向一致：更多最终 32×32 预算通常更有价值。这是少数可以从旧数据迁移为新实验设计的稳定信号。
- Unified L 只有 457.8M 参数，正确数据下 CFG1.0 FID8.461，且速度 benchmark 最快；它很适合作为 efficiency point，而不应只把 Unified XL 放在主表。
- Multistage 参数总量约 1.263B，显著大于 DiT/Unified XL。它的质量比较必须同时给出训练显存/时间，不能只看 sampling FLOPs。
- speed benchmark 的 250-step 结果确实存在；旧 JSON 的 NOT_MEASURED 是过期状态。

## 7. 最有价值的下一轮实验

1. **严格 matched-NFE sampler matrix**：Unified XL 400K，在 CFG1.0 与 1.5 下各跑 DDPM NFE={64,96,124,160,244,250}；DPM1 同 NFE；DPM2 用 scheduled 使实际 NFE 尽量匹配。所有日志直接写真实 model-call count。
2. **checkpoint × sampler 二维 sweep**：100K/200K/300K/400K/500K，每个 checkpoint 固定 matched NFE 比 DDPM、DPM1、DPM2。这才可以检验“训练成熟度使 DPM2 翻转”的假设。
3. **2×2 边界消融**：同一正确 Unified XL checkpoint，noise filling / bilinear × DDPM / DPM2。优先用约 125 和 250 NFE 两档。
4. **DPM2 机制消融**：midpoint 重新计算 drift vs 冻结起点 drift；这能直接判断当前 solver 翻转是否来自实现细节。
5. **定位稳定性拐点**：在真实 NFE 64 与 124 之间增加约 80、96、112 NFE。
6. **补齐 B 级指标**：先修 Unified L 的 evaluator，再评 MS-100、DiT500 CFG1.5 和 CFG1.0 sampler grids。
7. **正式 latency**：真实 checkpoint、生产 noise filling、CFG1.5、固定 50K 图、记录单图 wall-clock/吞吐/peak memory；DPM2 单独按真实 NFE 测。另做真实 8-GPU DDP training throughput。

## 8. 数据来源索引

- 本地配置：2d/exp/*.jsonl
- 正确数据日志：2d/exp、2d 根目录日志，以及 Hugging Face 的 eval_results_unified、eval_results_dit500k、eval_results_exp04
- 速度：2d/speed_benchmark/*.json、2d/benchmark_paper.py、2d/benchmark_training_speed.py、benchmark_250_speed.log
- 旧 HF 数据：docs/research_docs/existing_experiments.txt、docs/theory/experiment_plan.md
- DPM2 NFE 实现：diffusion/gaussian_diffusion_with_drift.py

当新结果加入时，必须同时登记：数据源、模型类型、参数量、checkpoint、训练 split、采样 split 及顺序、sampler、scheduled steps、真实 NFE、CFG、stage boundary、样本数、reference statistics 和五项指标。缺任一关键项时先放 B 级，不进入论文主表。
