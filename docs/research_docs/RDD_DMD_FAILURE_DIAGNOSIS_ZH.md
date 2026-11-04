# RDD 三步 DMD 失败诊断

更新时间：2026-08-28

## 结论

当前三步 DMD 效果差，不是教师 checkpoint 损坏，也不是 EMA、采样权重选择、阶段 timestep 映射或 noise-filling 接错。主要是两个训练目标层面的结构性问题：

1. `exact` 路径校正实际施加了 `alpha_local^2 * M_t^2`，把本来就困难的 8×8 全局布局梯度严重压小；
2. `preservation loss` 的目标不是完整教师轨迹生成的清晰样本，而是每个 stage 最高噪声端点的一次 `predicted x0`。这个目标本身是模糊的条件均值，权重 0.25 会持续把三步学生拉回模糊解。

两者组合后形成直接的失败机制：正确的分布匹配力被压弱，错误的模糊锚点持续存在；8×8 第一阶段先失败，16×16 和 32×32 又在它的模糊输出上继续 rollout，误差逐级传递。

继续按当前目标训练到 350K，较大概率只会在同一个模糊盆地里缓慢移动，不会自动恢复教师的 250-step 生成质量。

## 直接视觉证据

同一个 `RDD-Unified-L/0400000.pt` 教师权重：

- 正常 RDD DDPM250、CFG 1.5、noise filling 可以生成清晰图像：
  `../../eval_results/diagnostics/rdd_dmd_checkpoint_previews/teacher_400k_ddpm250_cfg1.5/000000.png`
- 将教师直接作为每阶段一步的三步初始化时，输出已经非常模糊：
  `../../eval_results/diagnostics/rdd_dmd_checkpoint_previews/teacher_init_400k/000000.png`
- 训练约 57K 后的 raw student 仍然基本是同一团模糊图：
  `../../eval_results/diagnostics/rdd_dmd_checkpoint_previews/dmd_latest_raw_student/000000.png`

同种子 8 张图片的简单清晰度统计：

| 权重/采样 | RGB 标准差 | 平均绝对梯度 | Laplacian 方差 |
|---|---:|---:|---:|
| 三步教师初始化 | 0.0813 | 0.0112 | 0.00403 |
| DMD 5K EMA | 0.0816 | 0.0118 | 0.00470 |
| DMD 50K EMA | 0.0889 | 0.0134 | 0.00720 |
| DMD 最新 EMA | 0.0889 | 0.0133 | 0.00710 |
| DMD 最新 raw student | 0.0916 | 0.0135 | 0.00770 |
| 教师正常 DDPM250 | 0.2101 | 0.0260 | 0.01373 |

这说明：

- 训练并非完全没有更新；
- 但 50K 到约 57K 已经基本平台化；
- raw student 只比 EMA 略清晰，EMA lag 不是根因；
- 正常教师采样明显更清晰，因此教师模型本身不是根因。

## 根因一：`exact` 梯度错误地额外衰减了低分辨率阶段

当前代码中的 stage-local 信号算子是

\[
A_t = \alpha_t M_t,
\qquad
M_t = I + \lambda_t(P-I).
\]

`exact_rdd_dmd_gradient()` 连续调用两次 `apply_signal_operator()`，所以实际得到

\[
A_t^\top A_t\Delta
=
\alpha_t^2 M_t^2\Delta,
\qquad
\Delta=\hat x_{0,\mathrm{fake}}-\hat x_{0,\mathrm{teacher}}.
\]

问题不是说任何推导里都绝对不能出现 `alpha^2`，而是当前实现：

- 把它作为所谓 RDD path correction 的一部分；
- 没有配套 `sigma_t` 权重或其它时间权重；
- normalizer 也不会消除这个随 timestep 变化的缩放；
- 在没有 resolution drift 的 8×8 stage，`exact` 本应退化到普通 DMD 的 `simple` 方向，当前却仍多乘了 `alpha_t^2`。

因此它违反了最重要的实现不变量：关闭 drift 后，RDD-DMD 应退化成标准 DMD。

按当前 300/300/400 stage schedule 和 score timestep 范围 `[0.02, 0.98]` 计算，代码施加的梯度幅度衰减为：

| stage | 分量 | 最小 | 中位数 | 平均 | 最大 |
|---|---|---:|---:|---:|---:|
| 32×32 | coarse/all | 0.411 | 0.787 | 0.759 | 0.999 |
| 32×32 | 新细节 | 0.000224 | 0.198 | 0.298 | 0.959 |
| 16×16 | coarse/all | 0.0702 | 0.319 | 0.386 | 0.958 |
| 16×16 | 新细节 | 0.000038 | 0.0803 | 0.201 | 0.920 |
| 8×8 | 全空间 | 0.00183 | 0.0587 | 0.172 | 0.896 |

8×8 stage 的中位梯度只剩 `simple` DMD 的约 5.9%，即约小 17 倍。这个 stage 恰好负责类别、物体数量、整体轮廓和布局，所以它一旦学不好，后两阶段很难补救。

相关代码：

- `diffusion/rdd_phased_dmd.py:99-107`
- `diffusion/rdd_phased_dmd.py:150-180`

现有单元测试只检查 surrogate gradient 能反传，没有检查 `c=None` 时 `exact == simple`，所以这个问题没有被测试捕获。

## 根因二：preservation target 本身就是模糊的

当前 preservation target 是：

```python
top = top_timestep(phase_diffusion, ...)
teacher_endpoint = predict_clean(
    phase_diffusion, teacher_forward, stage_input, top, labels
)
preserve_loss = mse(generated, teacher_endpoint)
```

这不是原始 DMD 的 regression target。原始 DMD 的 regression 使用教师多步确定性采样轨迹得到的配对清晰输出；DMD2 则直接移除了 regression loss。当前实现只让教师在 stage 的最高噪声端点看一次输入，再取一次 predicted-x0。

各 stage 顶端的一步信噪比约为：

| stage | 顶端一步 SNR |
|---|---:|
| 32×32 | 0.657 |
| 16×16 | 0.0698 |
| 8×8 | 0.00156 |

8×8 顶端几乎是纯噪声。给定这种输入，MSE 训练的 predicted-x0 倾向于条件均值；存在多种可能图像时，均值在像素/latent 空间里必然模糊。于是 preservation loss 权重 0.25 实际在做：

> 要求学生保留教师“一步去除极高噪声”的模糊输出。

这与“保留教师的高质量生成能力”不是一回事。

## 为什么三个 stage 会把错误放大

三步学生的 rollout 是：

\[
z_{8}\rightarrow \hat x_0^{8}
\rightarrow \text{noise filling}_{16}
\rightarrow \hat x_0^{16}
\rightarrow \text{noise filling}_{32}
\rightarrow \hat x_0^{32}.
\]

第一阶段 8×8 的正确 DMD 梯度最弱，同时模糊 preservation target 最危险。因此第一阶段先生成模糊的大色块。第二阶段和第三阶段看到的是学生自己的错误分布，而不是教师的正确中间轨迹；它们只能在错误布局上增加有限细节。这就是为什么最新图比初始化略锐，但物体结构没有真正恢复。

## 次要问题

### fake-score 的 5 次更新复用了同一批生成图

当前每个 outer step 先缓存一次 student rollout，然后对同一批 `critic_clean` 做 5 次 critic optimizer update，只重采样 timestep/noise。

这不等价于官方 DMD2 ImageNet recipe 的 two-time-scale 调度。官方 `dfake_gen_update_ratio=5` 是连续更新 fake score，而 generator 每五个 outer step 更新一次；fake score 会看到持续变化/重新生成的数据。当前实现使 fake-score 在每 5 次更新中看到的生成样本多样性降低约 5 倍，容易过拟合当前小批假样本，降低 DMD 梯度质量。

它很可能加重问题，但单独不足以解释从一开始就存在的严重模糊。

### 当前日志无法可靠做 per-phase 分析

每一行日志平均了前 10 个 step（混合三个 phase），却只打印当前 step 的 phase。因此日志上的 `phase=0/1/2` 不是相应 phase 的独立 loss。需要把三个 phase 分桶记录，才能判断哪一阶段先崩。

## 已排除的原因

1. **教师 checkpoint 损坏：排除。** 同一 400K 教师正常 DDPM250 输出清晰。
2. **采样加载了错误权重：排除。** 默认确实加载 EMA；显式加载 raw student 同样模糊。
3. **只是 EMA 太慢：排除。** raw 仅略好于 EMA，远低于正常教师。
4. **stage timestep offset/顺序错误：未发现。** 8→16→32 采样顺序与 global 999→599→299 顶端映射一致。
5. **noise filling 接错：未发现。** stage 边界使用 clean upsample 加新噪声，与既有 RDD sampler 一致。
6. **没有 GAN loss：不是首要原因。** 官方 DMD2 ImageNet 基线可以不启用可选 GAN classifier；当前失败在进入 GAN 讨论前就已经发生。
7. **batch size 太小：排除为首因。** 当前 global batch 是 256，与目标设置一致。
8. **纯粹训练不够久：不支持。** 50K 和最新 EMA 几乎无改善，loss 也没有下降趋势。

## 日志证据

5K step 分桶后：

- critic loss 约 0.035，基本不变；
- DMD surrogate 从约 0.056 上升并平台在 0.061–0.062；
- preservation 从约 0.0123 上升并平台在约 0.014；
- 没有 NaN 或数值爆炸；
- 但也没有显示目标在持续改善。

这更像“稳定地优化了不合适的目标”，不是普通的数值不稳定。

## 建议修复顺序

### P0：先修正确性

1. 将 path correction 改成只保留 resolution-specific 的
   \(M_t^\top M_t=M_t^2\)，不要在这里重复乘 `alpha_t^2`。
2. 加不变量测试：`c=None` 时 `exact` 必须与 `simple` 完全相同。
3. 加频带测试：coarse 子空间系数为 1，新细节子空间系数为 `(1-lambda_t)^2`。
4. 当前 preservation 先设为 0，跑一个干净的 DMD2-style 对照。

### P1：如果确实需要 preservation

不要使用最高噪声端点的一步 predicted-x0。只能选择下面之一：

- 原始 DMD 风格：预计算或在线运行教师完整多步 RDD/DDIM 轨迹，为相同初始噪声提供配对清晰 target；
- consistency/trajectory distillation 风格：蒸馏相邻教师轨迹映射；
- DMD2 风格：完全移除 regression/preservation，只依赖修正后的 distribution matching。

### P1：修 fake-score 调度

采用官方语义：fake-score 每个 outer step 看新的 student rollout；student/generator 每 5 个 outer step 更新一次。不要对同一生成 batch 连续做 5 个 optimizer step。

### P2：短跑筛选，不要立刻重跑 350K

从同一个教师重新初始化，先做三个 5K–10K 小实验：

| 试验 | path correction | preservation | 目的 |
|---|---|---:|---|
| A | simple | 0 | 最小标准 DMD2 基线 |
| B | 仅 `M_t^2` | 0 | 验证 RDD 几何校正是否有益 |
| C | 仅 `M_t^2` | 真正多步教师配对 target | 判断高质量 regression 是否进一步稳定 |

每 1K/5K/10K 固定相同 64 个 seeds/classes，比较：

- raw 与 EMA 网格；
- 8×8、16×16、32×32 各阶段输出；
- 10K FID/precision/recall；
- 每 phase 的 DMD loss、fake-score loss、梯度范数和 clamp 比例。

若 A/B 在 5K 已明显摆脱模糊初始化，再扩大训练；否则先继续修目标，不要烧 350K。

## 复现工具

- 梯度衰减分析：`2d/diagnostics/analyze_rdd_dmd_gradient_weighting.py`
- checkpoint 同种子采样：`2d/exp/diagnose_rdd_dmd_checkpoint_previews.sh`
- 图像清晰度统计：`2d/diagnostics/analyze_rdd_dmd_checkpoint_previews.py`
- 日志分桶：`2d/diagnostics/analyze_rdd_dmd_training_log.py`
- 显式选择 EMA/raw 权重：`2d/sample_rdd_phased_dmd3.py --weight-source ...`

## 主要参考

- DMD: <https://arxiv.org/abs/2311.18828>
- DMD2: <https://arxiv.org/abs/2405.14867>
- DMD2 官方实现：<https://github.com/tianweiy/DMD2>
- DMD2 官方 ImageNet 配置：<https://github.com/tianweiy/DMD2/blob/main/experiments/imagenet/imagenet_lr2e-6_scratch.sh>
- DMD2 官方梯度实现：<https://github.com/tianweiy/DMD2/blob/main/main/edm/edm_guidance.py>
