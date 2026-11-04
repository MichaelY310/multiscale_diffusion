# RDD 三步 Phased-DMD：推导、实现与实验协议

## 结论先说

目标不是把普通 DMD 生硬地重复三次，而是把 RDD 的三个分辨率阶段直接当成三个有物理意义的 distillation phase：

1. 第 1 次 DiT 调用在 8×8 latent 上完成全局结构生成；
2. noise filling 后，第 2 次调用在 16×16 上补充中频结构；
3. 再次 noise filling 后，第 3 次调用在 32×32 上补充最终纹理。

因此推理严格是 **3 次 conditional DiT forward（3 conditional NFE）**。VAE decode 和两次无需网络的 noise filling 不计入 NFE。

RDD 和普通 diffusion 的关键区别是：重新加噪时不能写成普通的 `alpha*x0 + sigma*noise`。RDD 的 clean signal 在每个阶段内还会被 resolution drift 改变。因此，teacher score、fake score，以及它们对 student clean output 的梯度，都必须经过同一个 RDD forward operator。忽略这一点会让 distillation objective 与 teacher 的真实路径不一致。

本实现是 **stage-aligned / phased DMD**，采用 DMD2 中稳定而重要的 data-free self-rollout 和多次 fake-score 更新思路，但目前没有实现 DMD2 的 GAN head，所以论文中应称为 “RDD-Phased-DMD” 或 “DMD-style distribution matching”，不能直接声称是完整 DMD2。

## 1. 为什么 RDD 应该做三阶段 distillation

300/300/400 Unified-L teacher 在正向扩散的 finest-first 记法下是：

| 全局 timestep | 负责的 latent 分辨率 | 正向扩散方向 |
|---:|---:|---|
| 0–299 | 32×32 | 逐渐删除 32→16 的细节 |
| 300–599 | 16×16 | 逐渐删除 16→8 的细节 |
| 600–999 | 8×8 | 最终走向纯噪声 |

生成沿反方向进行，所以顺序恰好是：

```text
z_8 at t=999
  -> G_theta(8×8, t=999)      -> x0_hat_8
  -> upsample + noise filling -> state_16 at t=599
  -> G_theta(16×16, t=599)    -> x0_hat_16
  -> upsample + noise filling -> state_32 at t=299
  -> G_theta(32×32, t=299)    -> x0_hat_32
  -> VAE decode
```

这比任意选三个 timestep 更自然：RDD 的边界本来就把生成任务分成“布局—中频结构—纹理”三个子问题。它也与 [Phased DMD](https://arxiv.org/abs/2510.27684) 把 score matching 限制在多个 SNR 子区间的思想一致，但 RDD 的子区间不是额外拍脑袋划分，而是由 resolution transition 决定。

## 2. 普通 DMD 到底在做什么

令 student generator 为 `G_theta(z)`，它产生 clean sample `x0^g`。在随机噪声等级 `t` 上重新加噪得到 `x_t`。然后使用两个 score：

- `s_real(x_t,t)`：冻结的 diffusion teacher，描述 teacher/data 分布希望往哪里走；
- `s_fake(x_t,t)`：持续在 student 样本上训练的 fake-score critic，描述 student 当前分布往哪里走。

DMD 使用二者之差更新 student：

```text
student update direction ~ s_fake - s_real
```

直观地说：如果 fake score 与 real score 相同，student 的 noisy distribution 已经与 teacher 一致；如果不同，这个差值告诉 student 应该把输出分布往哪里推。

[原始 DMD](https://arxiv.org/abs/2311.18828) 从近似 KL 的梯度得到这一规则，并额外使用 teacher 多步输出的 regression loss 稳定训练。[DMD2](https://arxiv.org/abs/2405.14867) 移除了预先生成 regression dataset 的依赖，并加入更稳定的 two-time-scale 更新及 GAN 相关改进；其[官方实现](https://github.com/tianweiy/DMD2)在 ImageNet 配置中也使用多次 fake-score 更新对应一次 generator 更新。

## 3. RDD forward operator

### 3.1 一个 stage 内的 forward

对某个需要从当前分辨率退化到下一层分辨率的 stage，定义：

- `D`：average pooling；
- `U`：nearest-neighbor repeat upsampling；
- `P = U D`：先降采样再升回当前形状；
- `lambda_t`：本 stage 内从 0 线性增长到 1 的 drift 比例；
- `alpha_t = sqrt(alpha_bar_t)`；
- `sigma_t = sqrt(1-alpha_bar_t)`。

RDD drift 是

```text
d(x0) = P x0 - x0.
```

所以 RDD marginal 为

```text
x_t = alpha_t [x0 + lambda_t(Px0-x0)] + sigma_t epsilon
    = A_t x0 + sigma_t epsilon,

A_t = alpha_t M_t,
M_t = I + lambda_t(P-I)
    = (1-lambda_t)I + lambda_t P.
```

在 8×8 最粗 stage 没有下一层需要退化，`P` 项不存在，退化回普通的 `A_t=alpha_t I`。

### 3.2 coarse 与 residual 两个正交子空间

average-pool + nearest-repeat 构成的 `P` 满足：

```text
P^T = P,    P^2 = P.
```

因此任何 latent 都可写成：

```text
x = P x + (I-P)x.
```

- `Px` 是下一层仍能表达的 coarse 部分；
- `(I-P)x` 是当前高分辨率独有的 residual/detail。

RDD signal operator 对二者的作用分别是：

```text
M_t P x       = P x,
M_t (I-P) x   = (1-lambda_t)(I-P)x.
```

也就是说 coarse 信号不被 resolution drift 删除，而新增细节会按 `1-lambda_t` 逐步衰减。这正是 RDD 相对普通 diffusion 需要特殊处理的数学原因。

## 4. RDD-DMD 梯度推导

### 4.1 用 clean prediction 写 score

对于线性 Gaussian forward

```text
x_t = A_t x0 + sigma_t epsilon,
```

一个 clean predictor `x0_hat(x_t,t)` 对应的 score 可写为

```text
s(x_t,t) = [A_t x0_hat(x_t,t) - x_t] / sigma_t^2.
```

teacher 与 fake critic 接收完全相同的 `x_t`，所以 `-x_t` 抵消：

```text
s_fake - s_real
  = A_t (x0_hat_fake - x0_hat_real) / sigma_t^2.
```

记

```text
Delta_x0 = x0_hat_fake - x0_hat_real.
```

### 4.2 拉回 student clean output

重新加噪时 `x_t` 对 student clean output 的 Jacobian 是 `A_t`。所以 score difference 拉回 clean space 后为

```text
grad_x0 L_DMD
  proportional to A_t^T (s_fake-s_real)
  = A_t^T A_t Delta_x0 / sigma_t^2.
```

普通 diffusion 中 `A_t=alpha_t I`，这只留下一个可被 DMD weighting/normalization 吸收的标量。但 RDD 中 `A_t` 不是标量，所以不能直接丢掉。

由于 `P` 对称且幂等：

```text
M_t^T M_t
  = P + (1-lambda_t)^2 (I-P).
```

因此 exact RDD-DMD 梯度会：

- 完整保留 coarse 方向的 distribution correction；
- 把 residual/detail 方向的 correction 乘以 `(1-lambda_t)^2`；
- 避免 teacher forward 已经几乎删除某个细节时，student 仍被一个普通 DMD 梯度强行追逐该细节。

实现中先计算 `Delta_x0`，再连续应用两次 `A_t`，等价于 `A_t^T A_t Delta_x0`。随后使用与 DMD 常见实现一致的 per-sample clean-error normalizer；整体 `alpha_t^2/sigma_t^2` 标量由 weighting/normalization 吸收，而分辨率相关的 `(1-lambda_t)^2` 不能吸收、必须保留。

## 5. 训练算法

每个 outer step 循环选择一个 phase：`8 -> 16 -> 32 -> 8 -> ...`。

### 5.1 生成当前 phase 的 on-policy 输入

```text
function rollout_to_phase(j):
    state = Gaussian noise at 8×8
    for phase k = 0 ... j:
        clean = student(state, top_timestep_of_phase_k)
        if k < j:
            state = noise_fill(upsample(stopgrad(clean)))
    return clean, state
```

只有当前目标 phase 的 student call 保留梯度；以前 phase 用 student 自己的输出构造 on-policy 输入，但 stop-gradient。这样不会把三张巨大 computation graph 同时留在显存里，也避免只用 teacher trajectory 训练造成 inference exposure bias。

### 5.2 训练 fake-score critic

对当前 student clean sample：

1. 在该 phase 的 local timestep 范围内随机采样 `t`；
2. 使用**精确 RDD marginal** `x_t=A_t x0+sigma_t epsilon` 重新加噪；
3. 让 critic 做与原 teacher 相同参数化的 denoising loss；
4. 每个 student update 前更新 critic 5 次。

这对应 DMD/DMD2 中让 fake score 紧跟 student 当前分布的 two-time-scale 思路。

### 5.3 更新 student

1. 对 student sample 再随机取一个当前 phase timestep并精确 RDD re-noise；
2. frozen teacher（训练时 CFG=1.5）给出 `x0_hat_real`；
3. fake critic 给出 `x0_hat_fake`；
4. 用上一节的 `A_t^T A_t` 计算 stop-gradient DMD direction；
5. 加一个权重 0.05 的 on-the-fly teacher endpoint preservation MSE；
6. 更新 student，并维护 EMA student。

小 preservation 项只负责训练早期的结构锚定，不需要预先缓存 teacher 多步生成数据。它也是为什么当前实现应诚实称为 DMD-style / phased DMD，而不是声称完全复现 DMD2。

## 6. Noise filling 为什么必须与现有 RDD 一样

从 coarse clean `x0_c` 切到更细分辨率时：

```text
u = nearest_upsample(x0_c)
x_start_next = q_RDD(u, top_timestep_next, fresh_noise).
```

在匹配边界，`P u = u`，所以 resolution drift 项恰好为零；这保证跨阶段输入与该 finer stage 的 forward marginal 相容。若直接把 `u` 送入下一步，训练时 student 看到的分布与推理时/teacher forward 都不一致；若用普通 diffusion 加噪，也会在离边界位置破坏 RDD 路径。

## 7. 当前正式配置

| 项目 | 配置 |
|---|---|
| Teacher | RDD Unified DiT-L/2 300/300/400 @ 400K |
| Student 初始化 | teacher EMA |
| Fake critic 初始化 | teacher EMA |
| 推理 NFE | 3 conditional DiT calls |
| 分辨率顺序 | 8×8 -> 16×16 -> 32×32 |
| 跨阶段 | nearest upsample + noise filling |
| GPU | physical 4,5,6,7 |
| micro-batch | 64/GPU |
| gradient accumulation | 1（代码支持显存不足时保持 global batch 不变） |
| effective global batch | 256，与正常 RDD 训练一致 |
| fake critic : student | 5 : 1 |
| critic warmup | 100 outer steps |
| student / critic LR | 2e-6 / 2e-6 |
| teacher CFG | 1.5，蒸馏进 student；推理只做 conditional call |
| preservation weight | 0.25（当前为 latent MSE teacher anchor） |
| gradient correction | exact `A_t^T A_t` |
| EMA | 0.9999 |
| 训练长度 | 350K outer optimizer steps |
| checkpoint | 每 5K（与正常 RDD launcher 一致）；`latest.pt` 保存精确 optimizer/critic/student/EMA 状态 |
| 自动停止 | 每次启动后的下一个 America/Los_Angeles 09:00；08:55 只保存 checkpoint 且继续训练，09:00 私有进程组硬停止 |

需要澄清：原始 DMD 的 ImageNet 实验使用 EDM/ADM denoiser，并没有一套“DMD DiT 配置”。当前配置把其 ImageNet recipe 中可迁移的部分用于 RDD DiT-L/2：AdamW `2e-6`、betas `(0.9,0.999)`、weight decay `0.01`、warmup 500、loss 权重 `1/1/0.25`、score 范围 `[0.02T,0.98T]`、clip 10、mixed precision 和 350K steps。原文使用 global batch 336；这里按 RDD parity 和本实验要求使用 256。原文还使用 25K 离线 teacher noise-image pairs 和 LPIPS regression；当前 data-free 实现以 online teacher endpoint latent MSE 代替，因此不能声称是原 DMD 的逐项复现。

[DMD2 官方 ImageNet recipe](https://github.com/tianweiy/DMD2/blob/main/experiments/imagenet/imagenet_lr2e-6_scratch.sh)采用 5:1 fake-score/generator 更新比；[FastVideo 的 DMD2 文档](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/distillation/dmd.md)展示了 3-step video DMD。这里没有直接复制它们的 timestep，因为 RDD 的三个 resolution boundary 已经定义了三个 phase。

## 8. 必须做的评估与 ablation

正式结论至少需要：

1. 50K ImageNet samples：IS、FID、sFID、Precision、Recall；
2. 同一硬件实测 latency、吞吐、peak memory；
3. 与原 teacher 的 DDPM-250 及 DPM-Solver 配置比较；
4. checkpoint curve：1K、5K、10K、20K、30K；
5. 相同 class labels / seeds 的可视化对比。

最重要、数量仍可控的 ablation：

| Ablation | 回答的问题 |
|---|---|
| exact `A_t^T A_t` vs simple clean difference | RDD-specific 数学修正是否真的必要 |
| phase-aligned 3-step vs 三个普通全局 timestep | RDD boundary 是否带来优势 |
| noise filling vs deterministic upsample | 随机补充 finer detail 是否必要 |
| preservation 0 vs 0.25 | 结构锚定是否防止早期坍塌 |
| critic ratio 1:1 vs 5:1 | fake score 是否跟得上 student distribution |

不要一开始同时扫很多 CFG/LR/边界。先用 10K samples 做上述筛选，再只给胜者生成 50K。

## 9. 与相近工作的关系及诚实边界

- [DMD](https://arxiv.org/abs/2311.18828)：提供 distribution matching 的基本 score-difference 梯度。
- [DMD2](https://arxiv.org/abs/2405.14867)：改进 data-free 稳定性并加入 GAN 等机制；当前实现只采用其中 self-rollout / fake-score TTUR 的一部分。
- [Phased DMD](https://arxiv.org/abs/2510.27684)：把 few-step DMD 分成多个 score/SNR 子区间，是最接近当前 three-phase 训练逻辑的工作。
- [PyramidalWan](https://openaccess.thecvf.com/content/CVPR2026/papers/Korzhenkov_PyramidalWan_On_Making_Pretrained_Video_Model_Pyramidal_for_Efficient_Inference_CVPR_2026_paper.pdf)：研究预训练视频模型的 pyramidal inference，并报告了 few-step pyramidal distillation；它说明“多分辨率 + few-step”本身不能作为独占 novelty。
- [Cross-Resolution Distribution Matching](https://arxiv.org/abs/2603.06136)：直接研究跨分辨率 DMD，是论文写作时必须正面比较的新竞争者。

RDD 这里真正可以强调的技术点是：**同一个 resolution-drift forward process 同时定义 teacher training path、跨阶段 marginal 和 DMD clean-space pullback；由此得到显式的 `A_t^T A_t = alpha_t^2[P+(1-lambda_t)^2(I-P)]` 修正。** 是否形成有效贡献最终取决于 exact-vs-simple ablation 和 3-NFE 质量，而不是仅靠公式名称。

## 10. 代码、命令与日志

- RDD-DMD operator：`diffusion/rdd_phased_dmd.py`
- 数学单元测试：`2d/test_rdd_phased_dmd_math.py`
- 训练器：`2d/train_rdd_phased_dmd3.py`
- 3-NFE sampler：`2d/sample_rdd_phased_dmd3.py`
- 50K 全指标评估：`2d/exp/eval_rdd_unified_l_phased_dmd3_4gpu.sh`
- 4–7 卡 launcher：`2d/exp/train_rdd_unified_l_phased_dmd3_dmdpaper_4567.sh`
- tmux payload：`2d/exp/run_rdd_unified_l_phased_dmd3_dmdpaper_tmux_payload.sh`
- 正式 tmux：`rdd_unified_l_phased_dmd3_gb256`
- 正式 log：`logs/rdd_unified_l_phased_dmd3_dmd_imagenet_gb256_4567.log`
- watchdog log：`logs/rdd_unified_l_phased_dmd3_dmd_imagenet_gb256_la9_watchdog.log`
- 输出：`results_rdd_dmd3/unified_l_300300400_400k_phased_dmd3_dmd_imagenet_gb256`

查看进度：

```bash
tmux attach -t rdd_unified_l_phased_dmd3_gb256
tail -f /scr/yuheng/DevOpsBench/multiscale_diffusion/logs/rdd_unified_l_phased_dmd3_dmd_imagenet_gb256_4567.log
```
