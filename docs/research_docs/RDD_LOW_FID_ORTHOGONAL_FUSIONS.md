# RDD 低 FID 路线：可信方法、正交性、融合设计与可运行实验

最后更新：2026-08-11

## 1. 结论先行

RDD 目前最值得走的路线不是把方法改造成另一个 SiT、EDM2 或 RAE，而是保留“随时间逐级改变空间自由度 + 跨尺度 drift/posterior”这个主干，在它上面叠加三类已经被反复验证的通用组件：

1. **先优化旧 checkpoint 的采样**：Limited-Interval Guidance（LIG）和 Autoguidance。它们不改变 RDD 训练目标，最便宜，也最容易证明与 RDD 正交。
2. **再优化 400K 内的训练效率**：Min-SNR、P2、loss-second-moment timestep sampling。它们改变损失权重或抽样方差，不改变 RDD 的 forward process、drift 或 stage 边界。
3. **最后做一个强而且有 RDD 味道的表征融合**：多尺度 REPA。不是只在 32×32 latent 上对齐一次，而是把同一个干净图像的 DINOv2 patch grid 池化成 16×16、8×8、4×4 token grids，分别监督 RDD 的 32、16、8 latent stages。这保留 RDD 主干，并把“语义由低分辨率先形成、细节由高分辨率后补齐”的叙事变成可直接测量的训练约束。

本轮代码已经实现以上三类。所有新开关默认关闭；不传新参数时，模型结构、state dict 键、旧 CFG forward 和 timestep RNG 路径保持原样。本地 `pytorch` Conda 环境已通过全部 Python 编译检查和 4 组 CPU 单元测试，包括旧 state dict 严格加载、REPA opt-in、loss 公式与 guidance 张量组合；真实 CUDA、旧 checkpoint byte parity 和 50K FID 仍必须在训练服务器验证，不能把 CPU 测试表述成最终数值验证。

## 2. 为什么别人的 FID 会非常低

不能把“方法名”和最终 FID 一一对应。公开的低 FID 通常是以下因素的乘积：

| 因素 | 典型做法 | 对 RDD 的含义 |
|---|---|---|
| 训练量 | 更长训练、更大模型、更大 global batch | 400K 对收敛速度有意义，但不能和 4M/7M 终局数值直接比较 |
| Guidance | classifier guidance、CFG、LIG、Autoguidance | 可能大幅降 FID，同时牺牲 recall；必须同时报告 precision/recall |
| 表征先验 | REPA/DINO、VA-VAE、RAE | 外部预训练 encoder 注入强语义先验；应明确标注，不冒充纯 ImageNet-from-scratch |
| Tokenizer 上限 | SD-VAE、VA-VAE、RAE 的 reconstruction FID 不同 | 生成器无法稳定突破很差的 reconstruction ceiling |
| 采样器/NFE | DDPM、SDE、Heun、DPM-Solver、Restart、优化 schedule | 同一 checkpoint 可差很多；表格必须写 solver 和真实 NFE |
| EMA 选择 | 固定 EMA、post-hoc EMA、多个 seed/EMA 中选最好 | 选择过程本身也是超参搜索，必须披露 |
| 评测口径 | train vs validation reference、50K 数量、crop、Inception 实现 | 这是最常见的“看起来进步很大”来源；RDD 必须固定 ADM ImageNet-256 reference |

可信的历史锚点如下。不同分辨率、训练量或 guidance 的结果不能横向当作严格排名：

- ADM 在 ImageNet-256 使用 classifier guidance 报告 FID 4.59；它说明 guidance 本身就能显著移动 FID。[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)
- DiT-XL/2 在 ImageNet-256 报告 FID 2.27，但依赖大模型和长训练；DiT 论文明确观察到 Gflops/训练量增加会持续降低 FID。[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)
- Min-SNR 报告 ImageNet-256 FID 2.06，并把改善归因于缓和不同 timestep 的优化冲突；官方仓库再结合 LIG 报告 1.57。[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)、[官方代码](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training)
- REPA 报告 ImageNet-256 FID 1.42，并显著加快 SiT 收敛；它用了冻结的视觉 foundation encoder，所以属于“外部表征先验”路线。[ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf)、[官方代码](https://github.com/sihyun-yu/REPA)
- LightningDiT + VA-VAE 报告 64 epochs、CFG FID 2.11，以及 800 epochs、CFG FID 1.35。低 FID 同时来自 tokenizer 和生成器 recipe，不应把 1.35 归功于单一 loss。[官方代码/CVPR 2025 Oral](https://github.com/hustvl/LightningDiT)
- EDM2 在 ImageNet-512 报告 CFG FID 1.81，核心包含 magnitude-preserving architecture 和 post-hoc EMA；这不是能无损塞进现有 RDD checkpoint 的小开关。[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf)、[官方代码](https://github.com/NVlabs/edm2)
- Autoguidance 使用较小或训练不足的同分布模型作 guide，报告 ImageNet-512 FID 1.25。其官方实现就是 `bad.lerp(good, scale)`。[NeurIPS 2024](https://arxiv.org/abs/2406.02507)、[官方代码](https://github.com/NVlabs/edm2)

## 3. 方法优先级与正交性

### A 级：应立即跑

#### 3.1 Limited-Interval Guidance

LIG 的观察是：高噪声早期 guidance 有害，低噪声末期 guidance 多余，中间区间最有益。原论文把 ImageNet-512 的记录从 1.81 改进到 1.40，并在多种网络和数据上验证。[论文](https://arxiv.org/abs/2404.07724)

它与 RDD 高度正交：RDD 决定状态空间和 drift 如何随时间改变；LIG 只决定 conditional/unconditional score 在哪些时间被外推。RDD 甚至提供了一个额外问题：最佳 guidance 区间是否跨越 8→16 或 16→32 边界。这可以成为论文里的 RDD-specific analysis。

实现：`2d/sample_fid_unified.py` 的 JSONL 新字段：

```json
{
  "guidance_low": 0.10,
  "guidance_high": 0.70,
  "guidance_schedule": "constant",
  "guidance_min_scale": 1.0,
  "guidance_schedule_power": 1.0
}
```

时间定义固定为 `0=clean, 1=highest noise`，并使用全局 1000-step RDD 时间，因此边界和 300/300/400 stages 可以直接对应。

#### 3.2 Autoguidance

Autoguidance 与 RDD 的兼容条件是：good 和 bad model 必须学习**相同 RDD forward process、相同 stage boundaries 和相同条件分布**。最佳 guide 顺序是：

1. 同一 Unified-XL run 的较早 checkpoint（最干净的 parity）；
2. 同样 300/300/400 RDD 训练的 Unified-L；
3. 同样 RDD 训练但明显 undertrained 的 XL。

不要首先用 no-drift DiT 或错误 HF 数据集模型做 guide；那会把“bad quality”和“不同目标分布”混在一起。

实现字段：

```json
{
  "cfg_scale": 1.0,
  "autoguidance_ckpt": "/path/to/weaker_rdd.pt",
  "autoguidance_model": "DiT-XL/2",
  "autoguidance_scale": 2.0,
  "autoguidance_low": 0.0,
  "autoguidance_high": 1.0
}
```

当前实现有意把 CFG 和 Autoguidance 分开做 parity，避免一次实验同时引入两个外推方向。它只 guidance 4 个 latent epsilon channels，learned variance 保留 good model 的输出。

#### 3.3 Min-SNR-γ

RDD 目前是 epsilon prediction。实现使用

\[
w_t=\frac{\min(\operatorname{SNR}_t,\gamma)}{\operatorname{SNR}_t},\qquad \gamma=5.
\]

对每个 RDD stage 使用该 stage 的局部 conditional diffusion SNR。只缩放 denoising MSE，不缩放 learned-variance VB 项。这个细节如果做错，可能让 variance head 退化。

它非常适合 RDD 的 400K 叙事：RDD 的目标正是更合理地分配不同尺度/噪声阶段的学习，而 Min-SNR 是缓和 timestep 梯度冲突。两者结合是否叠加，能说明 RDD 的收益不是普通 loss weighting 的替代品。

#### 3.4 多尺度 REPA

官方 REPA 在 DiT/SiT 中间层投影到 DINO patch features，以 negative cosine loss 对齐。RDD 融合版对每个 stage 做：

\[
L_{\text{total}}=\sum_k L_{\text{RDD},k}
+\lambda\sum_k L_{\text{cos}}
\left(P(h_k),\operatorname{Pool}_k(z_{\text{DINO}})\right).
\]

这里 `Pool_k` 把 16×16 DINO patch grid 自适应平均池化到 RDD student 的 16×16、8×8、4×4 token grids。这个设计比“RDD + 原版 REPA”更有论文价值：它显式约束低分辨率 stage 学语义、高分辨率 stage 逐步恢复细节。

默认 `repa_weight=0` 时模型没有 projector 参数；开启后 checkpoint 会带 `repa_projector.*`，采样器会自动识别并构造 projector，然后在普通 forward 中不计算它。旧 checkpoint 仍 strict load。

### B 级：便宜但预期收益低于 A 级

#### 3.5 P2 weighting

P2 使用

\[
w_t=(k+\operatorname{SNR}_t)^{-\gamma}
\]

减少高 SNR clean-up 阶段的权重，强调感知内容阶段。[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html)

它与 Min-SNR 是竞争性 loss recipes，不应一开始叠加。先做 baseline / Min-SNR / P2 三选一。

#### 3.6 Loss-second-moment timestep sampling

Improved Diffusion 用历史 loss 二阶矩选择 timestep，并用 importance weights 保持目标无偏；主要价值是降低训练估计方差，而不一定直接改变最优解。[ICML 2021](https://proceedings.mlr.press/v139/nichol21a.html)、[官方代码](https://github.com/openai/improved-diffusion)

仓库原本已经有 OpenAI sampler，本轮修复了 NumPy 1.24 以后失效的 `np.int`。不要在第一轮同时和 Min-SNR 组合；先单独判断它是否对 RDD stage loss 有稳定收益。

#### 3.7 FM stage-local logit-normal time sampling

对每个固定 RDD-FM stage 先采样 `tau=sigmoid(N(mu,sigma))`，再映射到该 stage 的 `[s_k,s_{k+1}]`。它**不会**重新分配 coarse/middle/fine 的 400/300/300 边界，因此与修正后的公平 FM 时间镜像不冲突。

这是改变 FM 训练分布的 recipe，不应直接冒充 RDD novelty；价值是让 diffusion/FM 两条主线都有可信的优化 baseline。

### C 级：有价值，但不应现在直接改主线

- **EDM2 magnitude preservation**：会重写 linear/attention/residual 的参数化，旧 checkpoint 不能直接复用。适合单独的 `RDD-EDM2 backbone` 长期分支，不适合当前论文主表的首要补强。
- **Post-hoc EMA**：很正交且可信，但需要从训练开始保存 power-function EMA 所需快照；不能从目前单一 EMA checkpoint 精确恢复。下一次长跑值得加入，当前不能伪造。
- **VA-VAE / RAE / DC-AE**：可能显著降低 tokenizer ceiling，但会改变 latent channels、scale、分布和网络输入。它们应作为 tokenizer ablation，不能替换主线后还把全部收益归给 RDD。RAE 结果很强但较新；先把 SD-VAE 主线做扎实。
- **AYS / Restart sampling**：都是正式会议方法，并且与 RDD 正交。[AYS, ICML 2024](https://proceedings.mlr.press/v235/sabour24a.html)、[Restart, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html)。但 RDD 有 stage transitions，直接照搬单尺度 schedule 不严谨。应先为每个 stage 拟合/搜索 schedule，并把跨尺度 transition 的 NFE 算清楚。
- **Zero-terminal-SNR**：是合理的训练修正，[WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf)，但会改变 forward schedule 和 RDD posterior；不能用于旧 checkpoint，也不是一个只改 sampler 的小开关。

## 4. 最小而有说服力的实验矩阵

不要做完整笛卡尔积。推荐顺序：

### Phase 0：兼容性与口径

1. 用旧 JSONL、不含任何新字段，固定 seed 生成 256 或 1K 张。
2. 与改代码前保存的同配置目录做 SHA-256 byte parity：

```bash
python scripts/compare_sample_dirs.py OLD_DIR NEW_DIR
```

3. 服务器跑 CPU/GPU smoke tests：

```bash
python tests/test_rdd_improvements.py
```

### Phase 1：无需训练的 10K screening

用当前最佳 RDD Unified-XL 400K，固定 DPM2 80/80/90、noise filling、相同 seed：

- legacy CFG 1.5；
- LIG `[0.05,0.80]`；
- LIG `[0.10,0.70]`；
- LIG `[0.20,0.80]`；
- cosine-ramp LIG `[0.10,0.80]`；
- 如果有合格 weak RDD checkpoint：Autoguidance 1.5 / 2.0 / 2.5，以及 2.0 + interval。

命令：

```bash
python 2d/exp/build_rdd_fid_improvement_parity.py \
  --ckpt /path/to/RDD_GOOD_400K.pt \
  --guide-ckpt /path/to/SAME_RDD_WEAK.pt \
  --output-root /scr/yuheng/DevOpsBench/multiscale_diffusion/eval_results/rdd_improvement_screen \
  --config-out 2d/exp/rdd_improvement_screen.jsonl

TOTAL_SAMPLES=10000 CONFIG=$PWD/2d/exp/rdd_improvement_screen.jsonl \
  bash 2d/exp/run_rdd_fid_improvement_parity_2gpu.sh
```

10K 只筛选。取最多两个 winner 跑 50K，报告 FID、sFID、IS、precision、recall、真实 NFE 和 wall-clock。不要只挑最低 FID 而隐藏 recall 崩溃。

### Phase 2：训练 recipe 的受控比较

优先 50K step early read，再把最好的两个跑到 100K/200K/400K：

1. baseline RDD；
2. RDD + Min-SNR；
3. RDD + P2；
4. RDD + loss-second-moment；
5. RDD + finest-only REPA（普通 REPA 对照）；
6. RDD + multiscale REPA；
7. 只有 2 和 6 都单独有效时，再跑 RDD + Min-SNR + REPA。

准备 DINO：

```bash
bash scripts/setup_rdd_improvements.sh
```

训练例子：

```bash
VARIANT=minsnr MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
VARIANT=p2 MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
VARIANT=repa_finest MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
VARIANT=repa MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
VARIANT=minsnr_repa MAX_STEPS=400000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
```

所有 run 必须固定原版 ImageNet、ADM crop、global batch 256、stage boundaries、模型大小、VAE 和 eval seed。REPA 额外报告 DINO 模型和外部预训练来源。

### Phase 3：FM

只做三条：corrected FM baseline（400/300/300 coarse→fine）、logit-normal、multiscale REPA。不要一开始把 diffusion 的 Min-SNR 生搬到 velocity loss。

```bash
VARIANT=fm_logit_normal MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
VARIANT=fm_repa_finest MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
VARIANT=fm_repa MAX_STEPS=100000 bash 2d/exp/train_rdd_improvement_parity_2gpu.sh
```

## 5. 论文如何保持 RDD 是主心骨

主结果至少保留两层：

1. **Plain RDD vs plain baseline**：同 backbone、tokenizer、训练量、loss recipe 和 sampler。这个表负责证明 RDD 本身。
2. **RDD + accepted recipes**：证明 RDD 能吸收通用强组件并达到有竞争力的绝对 FID。这个表负责绝对性能，不把通用组件冒充 RDD contribution。

最好的叙事不是“RDD 因为加了 REPA 所以新”，而是：

> RDD reorganizes the generative path by matching spatial degrees of freedom to noise-dependent information. This path-level change is complementary to optimization weighting, pretrained representation supervision, and post-training guidance. Across plain and strengthened recipes, RDD preserves its convergence advantage; its multistage structure further enables scale-aligned representation supervision unavailable to a single-resolution path.

必须准备的审稿人防御实验：

- RDD 增益在 legacy loss 和 Min-SNR 下都存在；
- RDD 增益在无 REPA 和有 REPA 下都存在；
- LIG 最优区间与 stage boundaries 的关系图；
- 多尺度 REPA vs 只对齐 finest stage；
- FID 改善没有完全来自 precision 上升/recall 下降；
- 参数量、训练 FLOPs、采样 NFE 和 VAE 完全对齐。

## 6. 本轮代码清单与兼容性边界

- `diffusion/rdd_improvements.py`：Min-SNR/P2、FM time sampler、LIG schedule、CFG/Autoguidance 组合、REPA pooling/loss。
- `diffusion/repa.py`：官方 DINOv2 teacher 加载和 patch-token 提取。
- `DiT/models_multires.py`：可选 projector 和 `return_features`；默认无新参数、普通 forward 不变。
- `2d/train_RDD_imagenet256_unified.py`：Min-SNR/P2、loss-second-moment、多尺度 REPA、精确 max-step stop。
- `2d/train_RDD_imagenet256.py`：multistage Min-SNR/P2/loss-second-moment。
- `2d/train_RDD_FM_imagenet256_unified.py`：stage-local logit-normal 和多尺度 REPA。
- `2d/sample_fid_unified.py`：Unified RDD 的 LIG/scheduled CFG/Autoguidance，自动加载带 REPA projector 的 checkpoint。
- `2d/exp/*improvement*`：配置生成、2-GPU sampling 和训练 variant launcher。
- `tests/test_rdd_improvements.py`：公式、张量组合、state-dict opt-in 测试。

明确限制：

- 目前 LIG/Autoguidance parity 只接到 `model_type=unified` diffusion；multistage 和 FM 会显式报错，不会悄悄忽略字段。
- 当前不允许 CFG 与 Autoguidance 同时开启；先获得各自独立结论，再决定是否值得增加组合。
- REPA 从旧 checkpoint fine-tune 时需要 `--reset-optimizer-on-resume`；加载器只允许缺少新 projector 键，其他 checkpoint 键不匹配仍会报错。
- 只有服务器完成真实 PyTorch/GPU smoke test 后，才能开始昂贵的 400K run。
