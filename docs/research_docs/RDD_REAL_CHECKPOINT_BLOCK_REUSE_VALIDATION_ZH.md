# 真实 RDD checkpoint：跨尺度 block reuse 的最终验证

> **最终判断：当前 training-free 整块 CSBR 不适合作为论文主方法；RDD delta transport 已出现稳定的机制证据，训练版值得做，而且训练目标已经被实验限定得很清楚。** 真实 `RDD-Unified-XL/2 @ 400K` 表明，16×16 block residual 确实包含 32×32 residual 的一部分低频方向，但不足以直接代替整个 fine block。时间条件校准、选层和 transport 可在三个 seed 上把同预算 stale-cache 误差降低约 15–28%，但目前只覆盖 3 层，整体速度不到论文目标。下一步不应继续枚举手工 scale，而应训练一个便宜的、显式保留 fine 高频输入的 cross-scale residual adapter，把该优势扩展到更多层。

## 1. 这次究竟回答了什么

此前 FastWan 上的 faux-RDD 视频实验回答的是：**如果 coarse residual 可以替代 fine block，那么能省多少时间？** 它证明了跳 6–24 个完整 fine block 有明显速度上限，但 checkpoint 并未按 RDD/CSBR 训练，不能证明画质。

这次改用真正训练过的 ImageNet RDD checkpoint，回答另外三个问题：

1. coarse residual 是否真的能预测 fine residual；
2. training-free 对齐能否把这种相关性变成稳定的实际采样加速；
3. 与普通全分辨率 previous-timestep cache 相比，RDD 是否已经有独有优势。

答案依次是：**部分能；整块替代不能，但 delta transport 可以稳定改善同预算 cache；目前还没有达到可发表的速度—质量优势。**

## 2. 严格实验口径

| 项目 | 设置 |
|---|---|
| RDD | `RDD-Unified-XL/2`, 400K checkpoint |
| 强基线 | original `DiT-XL/2`, 400K checkpoint |
| 规模 | 两者均为 XL/2：hidden=1152、28 blocks、16 heads、patch=2 |
| RDD sampler | DDPM 250：coarse/middle/fine=`80/80/90` |
| stage boundary | 全局 `1000→600→300→0` |
| 边界转换 | canonical `noise_filling` |
| CFG | 1.0 |
| 比较精度 | 所有 baseline/replacement 都走同一个 manual BF16/autocast forward；只改变指定 block 是否被替换 |
| 计时 | 单进程、同 batch、交错执行、7 repeats、报告 median |

两者是**匹配容量**，不是逐算子完全相同：RDD 使用 multiresolution RoPE，原始 DiT 使用固定 2D sin-cos position embedding。因此下文可将 DiT cache 称为强基线，不能把两者绝对秒数差异解释成方法速度差异。

## 3. 第一层结论：直接搬 coarse hidden/residual 不成立

设第 $l$ 个 Transformer block 的更新为

$$
h_{l+1}=h_l+\Delta_l(h_l,t,c).
$$

naive CSBR 用已经算过的 16×16 residual 替代 32×32 residual：

$$
\widehat\Delta_l^{32}(t)=U\!\left(\Delta_l^{16}(T_{16})\right).
$$

真实 checkpoint 的最佳单层不是完全无关，但只达到中等相关：

| layer | predicted-$x_0$ relative L2 | residual cosine | low-frequency cosine |
|---:|---:|---:|---:|
| 16 | 0.0476 | 0.488 | 0.563 |
| 15 | 0.0495 | 0.555 | 0.619 |
| 13 | 0.0558 | 0.542 | 0.616 |
| 8 | 0.0625 | 0.471 | 0.542 |
| 11 | 0.0670 | 0.652 | 0.713 |

而连续替换 tail-4/6/8 时，predicted-$x_0$ relative L2 分别变成 `0.733/0.803/0.876`。原因不是“cosine 不够高”这么简单：

$$
\Delta_l^{32}
=P_{\rm low}\Delta_l^{32}+P_{\rm high}\Delta_l^{32}.
$$

coarse residual 只能提供 cell 间的低频更新；它看不到同一 coarse cell 内 token 的不同纹理。并且 fine block 的输入 $h_l^{32}$、timestep 和上游误差都与 coarse stage 不同。替换一个 block 的误差进入下一个 block 后会再次经过 attention/MLP，因此多层误差不是简单相加，而会被非线性传播。

![直接复用的速度—扰动 Pareto](assets/rdd_real_checkpoint_validation/quality_speed_pareto.svg)

## 4. 时间条件 affine：方向对，但容量远远不够

我们没有再手调一个固定 scale，而是在 16 个与采样图片不重叠的 ImageNet calibration images 上，对每层、每通道、五个 fine timestep knot `0/75/150/225/299` 拟合：

$$
\widehat\Delta_{l,\rm low}^{32}(t)
=a_l(t)\odot U(\Delta_l^{16})+b_l(t).
$$

两个独立生成 seed 上，连续替换 3 层时：

| 方法 | seed 42424 latent L2 | seed 51515 latent L2 |
|---|---:|---:|
| static affine | 0.3543 | 0.3570 |
| time-conditioned affine | **0.2906** | **0.2838** |

也就是说 timestep conditioning 稳定降低约 `18–21%` 的替换误差，证明 timestep mismatch 确实存在；但绝对误差仍很大，图像会明显变化。只每三次 fine NFE 替换一次 6 个较安全层时，latent L2 降到 `0.183/0.165`，画面更接近 baseline，但稳定计时只节省 **2.1%**。

![时间条件与低频校准对比](../../eval_results/rdd_final_fair_time_seed_42424/comparison_grid.png)

另一 seed：

![时间条件与低频校准对比 seed 2](../../eval_results/rdd_final_fair_time_seed_51515/comparison_grid.png)

### 七次交错稳定计时

| 方法 | median denoiser | 延迟减少 | 该 seed latent L2 |
|---|---:|---:|---:|
| dense RDD | 4.631 s | 0 | 0 |
| static stagewide-3 | 4.461 s | 3.66% | 0.398 |
| time-conditioned stagewide-3 | 4.515 s | 2.49% | 0.328 |
| time-conditioned period-3 safe-6 | 4.533 s | 2.11% | 0.205 |
| time-conditioned stagger-6 | 4.549 s | 1.77% | 0.233 |

结论很直接：training-free 版本要么快得太少，要么破坏太大；没有合格 Pareto 点。

## 5. 真实生成轨迹揭示的 pattern

为了避免用 forward-noised ImageNet latent 代替真实 sampling distribution，我们又在**模型自己的 dense reverse trajectory**上，记录 90 次 fine NFE、28 层 residual，并比较：

$$
\text{generic}(t)=\Delta_l^{32}(t_r),
$$

$$
\text{RDD-transport}(t)
=\Delta_l^{32}(t_r)
+\widehat\Delta_l^{c}(t)-\widehat\Delta_l^{c}(t_r).
$$

后式不是直接用 coarse residual 替换 fine residual，而是只利用 coarse 预测的**时间变化量**修正 stale fine cache。

![transport 随 refresh gap 的变化](assets/rdd_real_checkpoint_validation/transport_improvement_by_gap.svg)

两个真实生成 seed 给出一致 pattern：

- gap=1 时平均改善约为零；generic cache 已经非常准，coarse 修正没有空间发挥。
- gap=4–5 后，transport 开始改善 low-frequency residual。
- gap=16 时，完整 residual relative-L2 改善约 `0.006–0.009`，低频改善约 `0.013–0.018`。
- high-frequency 改善始终接近零或略为负数；coarse trajectory 不会凭空产生 fine texture。
- 最稳定受益层是 `1/2/3/14/17/20`，不是简单的“所有中层”或“所有尾层”。
- 在 fine stage 后半段（较小 timestep）transport 更可靠；最前面的 fine NFE 只约一半比较获益。

这支持一个有限但真实的说法：**RDD coarse cache 能解释 stale fine residual 的一部分低频、长时间间隔变化。** 它不支持“coarse residual 可直接替代 fine block”。

## 6. 实际 rollout：delta transport 仍未稳定胜出

我们用真实轨迹选出的 6 层，比较相同 layer、相同 period、相同跳块数的 generic cache 和 RDD transport。最有希望的是 period=5，并且只在 $t\le180$ 时启用 transport：

| seed | generic latent L2 | RDD late-transport L2 | 相对变化 |
|---:|---:|---:|---:|
| 60013 | 0.04421 | **0.04235** | -4.2% |
| 70019 | **0.11371** | 0.11421 | +0.4% |
| 80021 | 0.04843 | **0.04560** | -5.9% |

六层混合配置平均只有一点改善，而且一个 seed 反转，说明“受益层”和“可安全跳层”不能混为一谈。

![late transport seed 60013](../../eval_results/rdd_delta_late_seed_60013/comparison_grid.png)

![late transport seed 70019](../../eval_results/rdd_delta_late_seed_70019/comparison_grid.png)

![late transport seed 80021](../../eval_results/rdd_delta_late_seed_80021/comparison_grid.png)

### 6.1 只保留最稳定的 3 层后，RDD-specific 正信号成立

根据真实 dense trajectory，而不是根据 forward-noised proxy，固定选择最稳定受益的 `layer 1/2/3`。在相同跳层、相同 period 下，transport 与 generic cache 的计算预算几乎相同，区别仅是前者加入 coarse-predicted time delta：

| seed | period | generic L2 | RDD transport L2 | 误差相对降低 |
|---:|---:|---:|---:|---:|
| 60013 | 5 | 0.04634 | **0.03794** | **18.1%** |
| 70019 | 5 | 0.08823 | **0.06315** | **28.4%** |
| 80021 | 5 | 0.04836 | **0.03978** | **17.8%** |
| 60013 | 8 | 0.07049 | **0.05984** | **15.1%** |
| 70019 | 8 | 0.14150 | **0.10308** | **27.2%** |
| 80021 | 8 | 0.08766 | **0.06762** | **22.9%** |
| **3-seed mean** | **5** | **0.06098** | **0.04695** | **23.0%** |
| **3-seed mean** | **8** | **0.09988** | **0.07685** | **23.1%** |

这是目前最重要的正结果：**相同 full-resolution block skip 数下，真实 RDD coarse trajectory 能稳定修正 stale fine cache，而不是只在 teacher-forced 指标上好看。** 三个 seed 的图像肉眼都与 dense baseline 很接近。

七次交错稳定计时进一步把速度上限校正为：

| 设置 | median denoiser | 相对 dense RDD 延迟减少 |
|---|---:|---:|
| dense RDD | 4.759 s | 0 |
| generic period-5 top-3 | 4.659 s | 2.10% |
| RDD transport period-5 top-3 | 4.638 s | 2.54% |
| generic period-8 top-3 | 4.639 s | 2.51% |
| RDD transport period-8 top-3 | 4.659 s | 2.11% |

generic 与 transport 的约 `±0.4` percentage-point 差异属于计时波动/轻微 adapter overhead，不能写成 transport 自身加速。真实结论是：它在**几乎相同执行成本**下稳定降低误差，但只跳 3/28 层，整体仅快约 2–2.5%。它证明机制，尚未达到用户要求的 10–20% 系统加速。训练目标就是把这条曲线从 3 层扩展到 9–12 层。

![top-3 transport seed 60013](../../eval_results/rdd_delta_top3_seed_60013/comparison_grid.png)

![top-3 transport seed 70019](../../eval_results/rdd_delta_top3_seed_70019/comparison_grid.png)

![top-3 transport seed 80021](../../eval_results/rdd_delta_top3_seed_80021/comparison_grid.png)

## 7. 必须承认的强基线：原始 DiT timestep cache

原始 DiT 的相邻 timestep block residual cosine 约为 `0.9998–0.99998`。在 7 次交错稳定计时中：

| 方法 | 跳块规则 | median denoiser | 延迟减少 | latent L2 |
|---|---|---:|---:|---:|
| dense DiT | 无 | 3.091 s | 0 | 0 |
| temporal stagger-9 | 每次轮换跳 3 层 | 2.920 s | 5.53% | 0.0496 |
| temporal stagger-12 | 每次轮换跳 4 层 | 2.839 s | 8.16% | 0.0841 |

![原始 DiT temporal-cache 对比](../../eval_results/dit_fair_temporal_stagger_stable_400k/comparison_grid.png)

此前一次未交错的计时曾显示 `20–22%`，七次稳定结果证明那是顺序/热状态偏差，不能使用。更重要的是：当前 RDD training-free 方法没有在相同扰动下稳定胜过这个通用 cache 基线。

## 8. Go / No-Go 决策

### No-Go

- 不再把 naive CSBR、固定 scale、整段 hidden replacement 或直接 stagewide residual replacement 当论文主方法。
- 不再通过继续枚举 scale/period 来寻找偶然的 4-image 最优点；只保留跨 seed 一致的 top-3 transport 作为 adapter 初始化证据。
- 不把 FastWan faux-RDD 的 12–61% denoiser 节省写成已经达到匹配画质的结果。
- 不声称“coarse residual 天生比 full-resolution timestep residual 更准确”。真实结果相反。

### Go

训练一个很小的 **Cross-Scale Residual Adapter（CSRA）**。它仍以 RDD coarse stage 为独有输入，但必须让当前 fine hidden 提供 coarse cache 无法拥有的高频：

![CSRA 训练版结构](assets/rdd_real_checkpoint_validation/csra_training_method.svg)

$$
\widehat\Delta_l^f(t)
=A_l\!\left(U\Delta_l^c,e_t\right)
+B_l\!\left(P_{\rm high}h_l^f,e_t\right).
$$

![CSRA 训练版结构](assets/rdd_real_checkpoint_validation/csra_training_method.svg)

- $A_l$：low-rank channel projection + timestep FiLM，修复通道语义与 timestep mismatch；
- $B_l$：便宜的 depthwise 2D/3D convolution + low-rank projection，只看当前 fine high-frequency hidden；
- 每 3–5 个 NFE 或交错 layer group 运行一次真实 fine block，刷新 correction，阻止误差积累；
- 先候选 `1/2/3/14/17/20`，训练时再用 gate/validation 淘汰，而不是硬编码所有 28 层。

这与普通 full-resolution cache 的关键区别不是“别人绝对不能加 adapter”，而是 RDD 已经免费提供真实生成过的 coarse trajectory；全分辨率模型若额外生成 coarse trajectory，就不再是相同执行图。

## 9. 建议的最小训练实验

先只做 2D，冻结 `RDD-Unified-XL/2 @ 400K` 主干，验证 adapter 值不值得搬到视频。

| 项目 | 建议 |
|---|---|
| 可训练参数 | 6 层 CSRA；rank=64 起步，另测 rank=128；timestep FiLM；depthwise 3×3 |
| backbone | 全冻结；不要一开始全模型 finetune |
| 数据 | 原生 ImageNet-256，沿用正确 original pipeline |
| teacher | 同 checkpoint 的 dense fine block residual / 最终 epsilon |
| skip schedule | period 3 与 5；stagger groups；训练时随机采样 refresh mask |
| batch | effective 256，与现有主实验口径一致 |
| optimizer | AdamW，adapter LR `2e-4`，weight decay `0.01`，grad clip `1.0` |
| 预算 | 5K smoke → 20K 决策点 → 最多 50K；不是重训 400K backbone |
| loss | block residual MSE + low/high band MSE + final epsilon consistency；20K 后才加短 rollout consistency |
| 主基线 | dense RDD、generic temporal cache、naive CSBR、CSRA；相同 NFE/skip 数 |
| 早停 | 20K 时若 CSRA 在相同 skip 下不能稳定优于 generic cache，则停止，不进视频 |

在 XL/2 hidden=1152 下，两条 rank-64 low-rank branch 加一个 depthwise 3×3，约为每层 `0.3–0.45M` 参数；6 层约 `2–3M`，不到 675M backbone 的 `0.5%`。显存不需要保存 backbone 权重梯度，但仍要做 dense teacher forward。4 张 80GB GPU 的建议起点是每卡 micro-batch 8、gradient accumulation 8，得到 effective batch 256；若实际显存富余再增大 micro-batch，不改变 effective batch。

2D 成功门槛：

1. 3 个以上 seed、10K image FID proxy 均优于同 compute generic cache；
2. 50K image FID 相对 dense RDD 恶化不超过 `+0.5`，Precision/Recall 无单边坍塌；
3. 至少能安全跳 9–12 个 fine blocks，稳定 denoiser 延迟减少 `≥5%`。2D token 少，因此不强求 10%；视频端再要求 `≥10%`。

只有通过这个门槛，才值得安排真实 RDD-Wan：先得到按 RDD/noise-filling 训练或微调的 Wan checkpoint，再训练 3D CSRA；FastWan 的 DMD/VSA 可以作为后续正交加速，而不能代替真实 RDD teacher。

## 10. 需要你安排的下一项算力

**现在确实需要训练，但只需要 adapter pilot，不需要重训 XL/2。** 建议你找一个 4-GPU、连续 6–12 小时的窗口跑 5K smoke；确认 loss、显存和 sample 正常后，再决定是否给 20K/50K。当前 training-free 路线已经验证充分，继续做手工 sweep 的科研收益很低。

## 11. 原始结果与代码

- 直接复用与 DiT 单步诊断：`eval_results/rdd_vs_dit_block_reuse_400k`
- timestep-conditioned 多 seed：`eval_results/rdd_final_fair_time_seed_42424`、`...51515`
- RDD 七次稳定计时：`eval_results/rdd_time_conditioned_stable_400k`
- 真实 dense trajectory：`eval_results/rdd_generation_trajectory_seed_91001`、`...92003`
- late transport 三 seed：`eval_results/rdd_delta_late_seed_60013`、`...70019`、`...80021`
- top-3 transport 三 seed：`eval_results/rdd_delta_top3_seed_60013`、`...70019`、`...80021`
- top-3 七次交错稳定计时：`eval_results/rdd_delta_top3_stable_400k`
- 原始 DiT 七次稳定对照：`eval_results/dit_fair_temporal_stagger_stable_400k`
- 核心诊断代码：`2d/attention_analysis/probe_rdd_vs_dit_block_reuse.py`、`sample_rdd_time_conditioned_reuse_preview.py`、`probe_rdd_generation_trajectory_transport.py`、`sample_rdd_delta_transport_preview.py`

所有新增实验入口默认关闭，不改变旧 checkpoint 格式或已有训练/采样行为。
