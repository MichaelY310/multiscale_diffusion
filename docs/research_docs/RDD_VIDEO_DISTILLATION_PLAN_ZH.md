# RDD-Video 少步蒸馏：调研结论、最终方案与执行手册

更新日期：2026-08-17

本文只讨论 **distillation**。不使用稀疏注意力、缓存 hidden state、量化、编译器、kernel 或任何硬件专用加速。目标是把一个已经训练好的 RDD-Wan 教师从 `20/20/10` 等多步采样，压缩成真正的 `2/2/1` 五次条件模型调用，同时尽量保持画质、运动和多样性。

## 1. 先给结论

最值得做的不是“直接照抄某一种蒸馏”，而是下面这个两阶段方案：

1. **阶段内 trajectory/consistency 初始化**：每个 student step 模仿 teacher 在同一个 RDD resolution stage 内走很多小步后的终点。它负责让训练稳定。
2. **RDD-aware DMD2 分布修正**：让 student 的整体输出分布接近 teacher，而不是死记 teacher 的某条确定轨迹。它负责恢复画质和多样性。
3. student 固定使用 **coarse/middle/fine = `2/2/1`**，任何蒸馏区间都不能跨越 RDD resolution boundary。
4. teacher 必须先是一个收敛的 **RDD-Wan teacher**。不能直接拿原始 Wan 当 RDD 各分辨率的 teacher。PyramidalWan 和 SwD 都观察到：原始 full-resolution teacher 在最低分辨率上可能根本不给出合理结果，直接做 DMD 会失败。
5. student、fake-score critic 都从同一个 RDD teacher 初始化，用 LoRA 训练。默认 rank 128。
6. DMD re-noise、critic flow-matching target、clean reconstruction 全部沿 **RDD path** 计算，不能偷偷换回标准 rectified-flow path。
7. inference 时 teacher CFG 已经被蒸进 student，因此使用 `guidance_scale=1`。旧采样器每个 stage 额外做 endpoint forward；新实现复用该 stage 最后一个 student call 的 clean prediction，因此 `2/2/1` 确实是 **5 conditional NFE**。

我把这个方案简称为 **RDD-Phased DMD2**。这不是把 DMD2 原样套上去：RDD path 对 clean recovery 和 distribution gradient 都有额外的投影算子，且 rollout 必须包含 RDD 的分阶段升分辨率过程。

## 2. “最好”到底指什么

蒸馏没有一个对所有模型都绝对最好的方法。需要区分三个目标：

- **最稳定、最好理解**：Progressive Distillation、Consistency Distillation、LCM。
- **少步生成绝对质量更强的主流路线**：DMD/DMD2 及其视频变体。
- **与 RDD 的多分辨率结构最接近**：SwD、PyramidalWan 的 pyramidal DMD。

因此本文不建议只做 trajectory loss，也不建议从第 0 步就只跑 DMD。前者容易继承 teacher 的数值误差且过度平均，后者容易因 fake-score critic 不准而崩。先 trajectory，后 DMD2，风险最低。

## 3. 经典方法与当前强方法

| 方法 | 核心想法（人话） | 优点 | 对 RDD 的判断 |
|---|---|---|---|
| [Progressive Distillation, ICLR 2022](https://arxiv.org/abs/2202.00512) | teacher 走两步，student 学成一步；反复把步数减半 | 经典、清楚、稳定 | 适合 Phase A，但单独使用不是最终质量最强路线 |
| [Consistency Models, ICML 2023](https://arxiv.org/abs/2303.01469) | 同一条 probability-flow ODE 上不同噪声位置应映射到同一 clean endpoint | 可一到少步生成 | 可做阶段内 consistency；不能跨 RDD stage 直接套公式 |
| [LCM, 2023](https://arxiv.org/abs/2310.04378) | 在 latent space 蒸馏 guided PF-ODE，通常 2--4 步 | 训练便宜、成熟 | 可作为 trajectory baseline，不解决 RDD path 的分布匹配问题 |
| [DMD, CVPR 2024](https://arxiv.org/abs/2311.18828) | 用“目标分布 score - 当前生成分布 score”直接推 student | 一步质量强 | 思路适用，但原版依赖 teacher 生成的回归数据，且不是多 stage |
| [DMD2, NeurIPS 2024](https://arxiv.org/abs/2405.14867) | 在线训练 fake critic；用 two-time-scale 更新稳定 critic；训练时模拟 inference rollout | 当前最可靠的分布蒸馏骨架之一 | 作为 RDD 最终主方法；必须改写 re-noise/path/rollout |
| [DOLLAR, 2024/2026](https://arxiv.org/abs/2412.15689) | variational score distillation + consistency，再做 reward optimization | 4-step 视频指标强 | 证明“trajectory + distribution”合理；reward 部分是额外变量，主实验先不加 |
| [Scale-wise Distillation (SwD), 2025/2026](https://arxiv.org/abs/2503.16397) | 少步模型同时逐渐提高 resolution，并用 patch MMD 做分布对齐 | 与多分辨率生成高度相关 | 强竞争对手；其 Wan 实验中 DMD 在低分辨率 teacher 不可靠时失败，正好支持“先训 RDD teacher” |
| [PyramidalWan, 2026](https://arxiv.org/abs/2601.04792) | 先把 Wan 微调成 pyramidal teacher，再做 pyramidal DMD；使用 2/2/1 | 与本项目结构最接近 | 最重要的直接对比；它也说明 original Wan teacher 不能直接指导最粗 stage |
| [FastVideo DMD2](https://haoailab.com/FastVideo/distillation/dmd/) | 面向 Wan 的工程化 student/teacher/critic DMD2 训练 | 官方代码成熟、角色划分清楚 | 参考 optimizer、critic update 和 rollout；本项目不使用其中的稀疏/硬件部分 |

补充客观判断：

- DMD2 在 ImageNet-64 报告 one-step FID 1.28、COCO zero-shot FID 8.35，但这是图像任务，不能直接拿数值预测 RDD-Wan。
- DOLLAR 报告 4-step VBench 82.57，但其 reward optimization 改变了目标，不能把全部增益都归因于 distillation。
- PyramidalWan 最适合做结构性参考。其论文使用 2 张 H100、每 GPU batch 6、5K steps、80K Wan-14B synthetic videos，并测试 `2/2/1`。这与当前两张 96GB GPU 的资源最接近。
- FastVideo 公布的正式 Wan DMD 配置规模可达到 32 张 H200 和大规模 synthetic latents，不适合直接照搬到本项目的两卡实验。

## 4. 需要哪些前置知识

不需要先把所有论文从头读完。按下面顺序掌握即可。

### 4.1 必须懂的最小知识

1. **Flow Matching 的直线路径**

   标准 rectified flow 常写成

   ```text
   x_t = t x_1 + (1-t) epsilon
   ```

   其中 `t=0` 是噪声，`t=1` 是 clean data。Wan 使用相反的 noise time：`sigma=1-t`，网络预测 `dx/dsigma`。

2. **Euler/Heun ODE solver**

   需要理解“网络给出当前位置的速度；Euler 用一次速度向前走一步；Heun 再看一次终点速度做平均”。不需要高等数值分析。

3. **CFG**

   teacher 同时算 conditional/unconditional，组成 guided prediction。蒸馏后 student 可以把这个 guided behavior 内化，因此 inference 不再做双分支 CFG。

4. **score 是什么**

   score 是 noisy distribution 的 log-density 对输入的梯度。直观上，它告诉 noisy sample 往哪里移动会更像该分布。DMD 不要求你知道真实 score，而是分别用 teacher 和 fake critic 估计它们。

5. **stop-gradient**

   DMD 先算一个“希望 student 往哪里改”的梯度方向，再把这个方向当常量构造 surrogate loss。teacher 与 critic 的这部分预测不反传。

6. **LoRA、EMA 与 two-time-scale update**

   LoRA 只训练低秩增量；two-time-scale 指 critic 比 student 更新更频繁或更快，避免 student 被一个过时的 fake critic 误导。

### 4.2 最小阅读清单

只读以下部分就足够理解代码：

1. Progressive Distillation：方法图和“teacher 两步变 student 一步”的章节。
2. DMD2：DMD gradient、fake critic、two-time-scale update、multi-step rollout mismatch 四部分。
3. PyramidalWan：pyramidal teacher、DMD 蒸馏、step-distillation 实验和 ablation。
4. 回看本仓库 `rdd_video/flow_matching_video.py` 的 `q_sample`、`compute_velocity_target`、`predict_x1_from_velocity`。

Consistency Models/LCM、SwD、DOLLAR 可以在理解主方案后再读，不是开始实验的前置阻塞项。

## 5. RDD-aware 蒸馏的数学核心

### 5.1 RDD stage 内的路径

一个 RDD stage 中：

```text
x_t = t x_1 + (1-t) epsilon + f(t) (P x_1 - x_1)
f(t) = t (s_next - t) / (s_next - s_start)
```

`P` 是当前 resolution 下的 avg-pool 再 nearest-repeat，`[s_start,s_next]` 是这个 stage 负责的 data-time 区间。

定义线性 signal operator：

```text
A_t x_1 = t x_1 + f(t)(P-I)x_1
```

那么 RDD re-noise 是：

```text
x_t = A_t x_1 + (1-t) epsilon
```

这就是为什么普通 DMD 的 `t*x1+(1-t)*noise` 不能直接用于 RDD：它漏掉了 resolution drift。

### 5.2 Phase A：阶段内轨迹压缩

固定 student schedule 为：

```text
coarse: 2 calls
middle: 2 calls
fine:   1 call
```

每个 student call 都严格位于一个 stage 内。对一个区间 `[t_a,t_b]`：

1. 从真实 latent `x_1` 和同一份 noise 构造正确的 RDD `x_ta`。
2. teacher 用 8 个 Heun substeps 从 `t_a` 走到 `t_b`，得到 `x_teacher_end`。
3. student 只调用一次并做一个 Euler update，得到 `x_student_end`。
4. 最小化两者的 endpoint MSE。

它教 student “一次大步应该落在哪里”。区间绝不能跨 boundary，因为 boundary 处 tensor resolution 和 transition distribution 都变了。

### 5.3 Phase B：RDD-aware DMD2

训练 student 时，先用 student 自己模拟 inference rollout。目标 call 以前的所有 call 都 detach，只对当前 call 反传。这解决 DMD2 所说的 train/inference input mismatch。

得到当前 call 的 clean prediction `x_gen` 后：

1. 在当前 stage 内随机选一个 score time `t_s`。
2. 使用 RDD path 重新加噪：`x_s=A_s x_gen+(1-t_s)epsilon`。
3. frozen RDD teacher 在 `x_s` 上给出 guided clean estimate `x_real`。
4. fake critic 在 `x_s` 上给出当前生成分布的 clean estimate `x_fake`。
5. 简化 DMD2 梯度方向是 `x_fake-x_real`。

对 RDD 更严格的 path correction 是：

```text
g_RDD proportional to A_s^T A_s (x_fake - x_real)
```

在当前 latent avg-pool/repeat 的假设下，`P` 是对称幂等投影，因此 `A_s^T=A_s`。代码中 `exact` 模式把 `A_s` 连续应用两次；`simple` 模式保留普通 DMD2 的 clean-difference approximation。

这条 correction 是 RDD-specific derivation，应该在论文里作为方法点，但必须通过 `exact` vs `simple` ablation 验证，不能只凭理论宣称一定更好。PyramidalWan 也观察到某些理论更简化的 DMD 版本反而更好，因此实证比较很重要。

fake critic 自己不是随便训练的。它对 `x_gen` 沿同一条 RDD path 做普通 RDD-FM：

```text
L_critic = || w_fake(x_t,t) - w_RDD_target(x_gen,epsilon,t) ||^2
```

student 每 5 个 critic optimizer steps 更新一次，避免 fake critic 追不上 student。

最后保留一个权重 `0.01` 的 teacher-preservation loss，防止 DMD 只追逐静态观感而损害 Dynamic Degree。这对应 PyramidalWan 中 `L_teach` 的功能角色。

## 6. 最终推荐训练配置

### 6.1 前提

- Backbone：Wan2.1-T2V-1.3B。
- Teacher：已经用 RDD path 微调并收敛的 RDD-Wan 1.3B。
- Data：当前 paper recipe 的 80K Wan-14B synthetic videos，预先缓存三个 resolution 的 latent 和 T5 embedding。
- RDD boundaries：保持 teacher 的 `[0,0.06,0.41,1]`，即 stage lengths `60,350,590`。蒸馏阶段不再同时搜索 boundary。

### 6.2 两卡配置

```text
GPU                         2,3
student calls               2/2/1
optimizer steps             5000
Phase A trajectory steps    1000
Phase B DMD2 steps          4000
teacher substeps/call       8, Heun
student LoRA                rank 128, alpha 128
critic LoRA                 rank 128, alpha 128
student LR                  1e-5
critic LR                   1e-5
critic betas                (0.0, 0.999)
student update interval     5 critic steps
teacher CFG                 5.0
inference CFG               1.0
teacher preservation        0.01
per-GPU microbatch          1
gradient accumulation       6
effective global batch      12
precision                   bfloat16
```

PyramidalWan 的公开配置同样是两卡、5K steps、6 samples/GPU，因此这是最有根据、又能在现有资源上运行的首个正式配置。它不是保证最优的超参数，只是最合理的起点。

## 7. 必做实验与停止条件

### 7.1 最小论文矩阵

| ID | 模型 | NFE | 是否需要新训练 | 目的 |
|---|---|---:|---|---|
| A | RDD teacher | 20/20/10 | 否 | quality upper bound |
| B | RDD teacher 直接少步采样 | 2/2/1 | 否 | 证明不是改 sampler 就够了 |
| C | trajectory-only student | 2/2/1 | 是，Phase A checkpoint | 轨迹蒸馏 baseline |
| D | RDD-Phased DMD2 exact | 2/2/1 | 是 | 主方法 |
| E | RDD-Phased DMD2 simple | 2/2/1 | 从 C 分叉 | 验证 RDD path correction |
| F | D 去掉 0.01 teacher loss | 2/2/1 | 从 C 分叉 | 验证运动保持项 |

最少需要一个 teacher、一个 Phase-A student、三个从 Phase-A checkpoint 分叉的 Phase-B run。先做 D/E；若两者都明显不工作，不要急着做 F。

### 7.2 评估

- VBench 总分及 16 个维度，特别是 Dynamic Degree、Motion Smoothness、Temporal Flickering、Subject Consistency。
- VBench2，避免只针对旧 benchmark 调参。
- 固定 prompt/seed 的 blinded human preference：主方法 vs teacher；主方法 vs PyramidalWan-DMD。
- 多 seed diversity，防止 DMD 用 mode collapse 换取观感。
- 报告 scheduled updates、真实 conditional NFE；不要只报脚本中的 step 字符串。

### 7.3 Pilot gate

正式跑 5K 前先跑 200--500 steps：

1. critic loss 有下降且不是快速趋近 0；
2. DMD gradient norm 有限，没有 NaN/Inf；
3. 生成 latent 的 mean/std 不持续爆炸；
4. `exact` 与 `simple` 至少有一条在 100 prompts 上优于未蒸馏 2/2/1；
5. Dynamic Degree 不出现明显坍缩。

## 8. 已实现的代码

- `rdd_video/rdd_distillation.py`：RDD path、固定阶段内 2/2/1 schedule、teacher integration、trajectory loss、exact/simple DMD surrogate。
- `rdd_video/lora_utils.py`：Wan attention/FFN 的依赖无关 LoRA 注入、保存和合并。
- `rdd_video/distill_rdd_wan.py`：student/teacher/fake critic 双阶段 DDP trainer。
- `rdd_video/export_rdd_wan_distilled.py`：把 student LoRA 合并成旧 sampler 可读的普通 RDD checkpoint。
- `rdd_video/sample_rdd_wan.py`：新增 true-five-NFE boundary clean reuse。
- `rdd_video/run_rdd_wan_distill_23.sh`：正式 GPU 2、3 启动脚本。
- `rdd_video/smoke_rdd_wan_distill_23.sh`：允许 base Wan 仅做 plumbing smoke；不代表生成质量。
- `rdd_video/sample_rdd_wan_distilled.sh`：真正 `2/2/1`、CFG=1 的采样脚本。
- `rdd_video/test_rdd_distillation_cpu.py`、`test_lora_utils_cpu.py`：数学与 LoRA contract tests。

## 9. 执行命令

### 9.1 CPU 数学测试

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
source .venv-rdd-video/bin/activate
python rdd_video/test_rdd_distillation_cpu.py
python rdd_video/test_lora_utils_cpu.py
```

### 9.2 双卡 plumbing smoke

这个测试只确认真实 Wan forward、DDP、LoRA、backward 和 checkpoint 能跑。它可以暂用 base Wan，因此不能评价 RDD distillation 质量。

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
CUDA_VISIBLE_DEVICES=2,3 bash rdd_video/smoke_rdd_wan_distill_23.sh
```

### 9.3 正式训练

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion

TEACHER_CHECKPOINT=/absolute/path/to/converged_rdd_wan_teacher.pt \
DATA_DIR=/absolute/path/to/multires_latents_latent_wan14b_80k \
CUDA_VISIBLE_DEVICES=2,3 \
bash rdd_video/run_rdd_wan_distill_23.sh
```

正式脚本在没有 RDD teacher 时会 fail fast，不会把原始 Wan 冒充 RDD teacher。

### 9.4 从 trajectory checkpoint 分叉 simple ablation

```bash
TEACHER_CHECKPOINT=/absolute/path/to/converged_rdd_wan_teacher.pt \
DATA_DIR=/absolute/path/to/multires_latents_latent_wan14b_80k \
RESUME=/absolute/path/to/phase_a_trajectory_checkpoint.pt \
EXP_NAME=rdd_wan_dmd2_221_simple \
PATH_CORRECTION=simple \
TRAJECTORY_STEPS=1000 \
MAX_STEPS=5000 \
CUDA_VISIBLE_DEVICES=2,3 \
bash rdd_video/run_rdd_wan_distill_23.sh
```

注意：resume checkpoint 中的 `step` 应正好等于 Phase-A 结束点。这样两个 DMD branch 才有完全相同的初始化。

### 9.5 合并并采样

```bash
source .venv-rdd-video/bin/activate
python rdd_video/export_rdd_wan_distilled.py \
  --distill-checkpoint results_rdd_video_distill/rdd_wan_dmd2_221_exact/checkpoints/0005000.pt \
  --output checkpoints/RDD-Wan-Distilled-221/0005000.pt

CHECKPOINT=checkpoints/RDD-Wan-Distilled-221/0005000.pt \
DEVICE=2 \
bash rdd_video/sample_rdd_wan_distilled.sh
```

## 10. 实际验证记录（surfi2，GPU 2/3）

这套实现已经在真实 Wan2.1-T2V-1.3B 上做过两类双卡 DDP 冒烟测试，不只是 toy tensor：

1. **Phase A / trajectory**：完成 teacher 多子步积分、student LoRA 前反传、DDP 同步和 checkpoint 保存；一步耗时约 0.59 秒（使用极小 dummy latent，仅测试程序链路）。
2. **Phase B / DMD2**：完成 detached student rollout、RDD re-noise、teacher/fake critic 三模型 forward、fake-critic flow-matching backward、student DMD2/preservation backward 和 checkpoint 保存；一步耗时约 1.29 秒，初始 critic loss 为 1.12014。
3. 两次测试分别生成了约 44.6 MB 和 67.0 MB 的 checkpoint；测试结束后 GPU 2、3 均回到 3 MiB、0% utilization。
4. 初始化的第一步中 student、teacher、fake critic 权重相同，因此 DMD 与 preservation 数值可接近零。这只能证明 plumbing 正确；随着 fake critic 更新与 student rollout 偏离，正式训练中这两项应变为非零。Pilot 必须额外记录高精度 loss 和 gradient norm，若持续为零则立即停止排查。

远端日志与产物：

- Phase A：`/scr/yuheng/DevOpsBench/multiscale_diffusion/results_rdd_video_distill_smoke/gpu23_contract/`
- Phase B：`/scr/yuheng/DevOpsBench/multiscale_diffusion/results_rdd_video_distill_smoke/gpu23_contract_dmd2/`
- 独立 RDD 数学/DDP 核心测试：`/scr/yuheng/DevOpsBench/multiscale_diffusion/logs/rdd_distillation_core_gpu23.log`

## 11. 当前诚实限制

1. 数学 contract 和 LoRA merge 已通过 CPU 测试，但真实质量实验必须等收敛的 RDD-Wan teacher 与 80K cached latent 准备好。
2. 使用原始 Wan 做一两步 smoke 只能证明代码链路正确，不能证明 RDD-Phased DMD2 有效。
3. `A_t^T A_t` correction 是针对当前 latent avg-pool/repeat 投影推导的。若以后把 transform 改成 VAE re-encode 或非线性 transform，不能继续复用该公式。
4. DMD 训练不稳定是正常风险。正式论文必须同时报告失败率、seed 和 `simple` ablation，不能只挑最好 checkpoint。
5. 现在最有价值的 milestone 不是直接追求 one-step，而是先让五步 `2/2/1` 在质量上接近 `20/20/10` teacher。达到后再讨论 `1/1/1` 或总 3-step。
