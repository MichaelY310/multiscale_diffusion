# 给跨尺度复用增加 hidden 蒸馏：实现与独立对照方案

<style>
#content { min-width: 0; overflow-wrap: anywhere; }
#content table { display: block; overflow-x: auto; }
#content p code, #content td code { white-space: normal; overflow-wrap: anywhere; }
</style>

## 0. 最新执行安排：按用户要求改为无限续训

2026-09-17（北京时间）：原2000步 pilot 已正常完成，累计压缩训练7000步。用户随后明确要求停止定时查看并继续不限步数训练；下文的2000步设计属于已经完成的首轮对照预算，不再是当前停止条件。

- 已删除 Codex 每30分钟检查的定时任务，不再自动唤醒对话。
- 从本实验自己的完整 `pilot_hidden/checkpoints/latest.pt` 接着第2001步训练；恢复生成器、critic、两个优化器及每个rank随机数状态，不重新开始warm-up或loss ramp。
- 使用 `--steps 0` 表示真正的无限更新循环，直到用户主动停止或发生硬件／数值／磁盘错误；不是用一个很大的有限数字假装无限。
- 仍为0123卡、每卡64、全局256，hidden权重0.05，原loss、复用层、学习率均不变。4567及其他旧实验不改。
- 每500步继续验证并保存原子替换的完整latest；服务器上的预览与限定范围清理仍保留，不是Codex定时监控。旧权重不重新累积。

服务器启动／恢复命令（同一输出目录，不能重复开两份）：

```bash
bash 2d/exp/run_dmd_reuse_hidden_ablation_20260916.sh hidden continuous
```

tmux仍为 `rdd_reuse_hidden_pilot`；日志位于 `/home/yuheng/rdd_experiments/dmd_reuse_hidden_20260916/pilot_hidden/launcher.log`。需要主动暂停时，可在此实验目录创建 `STOP_REQUESTED`，程序在完成当前更新后保存并退出；下次恢复前须先移走该标记。训练不会由已取消的定时任务替用户停止或重新调参。

## 1. 一句话说明

**保留现有跨尺度复用和全部训练目标，让原来不跳层的 DMD 模型额外指导三个关键位置的 hidden。整模型继续微调，不加 adapter，不增加推理层数。**

这是待验证的补救措施，不是已经修复画质的结论。之前的诊断说明：既有“可以表达、但还没学好的更新”，也有“低分辨率缓存无法直接表达的细粒度更新”。这次主要处理前者。

## 2. 老师是谁，教什么

老师是原始、冻结的三步 RDD-DMD Unified-L/2 模型，不是未蒸馏的 250 步 diffusion 模型。后者仍作为原 DMD loss 的 score teacher，二者用途不同。

最重要的是输入对齐：学生先按照真实三步过程生成自己的 stage 输入，再把这个输入的 **detach 版本**、相同类别、相同 timestep 交给不跳层教师。不能拿教师自己轨迹中的另一个输入对应的 hidden 直接硬对齐。

下面的层号从 1 开始；hidden 是 Transformer block 输出，不是 attention probability、RGB 图片或最终 VAE latent。

| 监督点 | 比较方式 | 希望解决什么 |
|---|---|---|
| 16×16 stage，第 24 层后 | 完整 hidden 对齐 | 中间阶段既是生成步骤，又是 fine 缓存的来源，先把出口校准 |
| 32×32 stage，第 22 层后 | 双方都在 token 网格上做 2×2 平均池化，再对齐 | 校准大段复用结束时的粗尺度状态，不强迫缓存表示 cell 内细节 |
| 32×32 stage，第 24 层后 | 完整 hidden 对齐 | 指导末尾仍计算的两层修复状态，并把梯度传回前面的共享模型 |

32×32 latent 经 patch=2 后是 16×16 token 网格。第 22 层的池化将它变成 8×8 token 网格，**不是把 RGB 图片缩小**。保留的是每个小格平均信息，因此这里的“低频”是粗 cell 投影意义，不是精确 Fourier 频率截断。

教师不更新；学生的 hidden、缓存、跨分辨率 noise-filling 都保留梯度。训练可以反过来调整低分辨率阶段输出的缓存。

## 3. 新增的 loss

为避免某层仅因为 hidden 数值大就支配训练，每个样本先用教师 hidden 的平均平方做归一化，再对 batch 平均：

$$
D(S,T)=\frac{1}{B}\sum_{b=1}^{B}
\frac{\operatorname{mean}\left[(S_b-\operatorname{stopgrad}(T_b))^2\right]}
{\max(\operatorname{mean}(T_b^2),10^{-6})}.
$$

这里分子表示“差值逐元素平方后求平均”，不是“先求平均再平方”。它不是 cosine，也不会忽略学生特征的幅度错误。

令 $P$ 为上述 token 平均池化：

$$
L_{hidden}=\frac{1}{3}\left[
D(h^{16}_{24,S},h^{16}_{24,T})+
D(Ph^{32}_{22,S},Ph^{32}_{22,T})+
D(h^{32}_{24,S},h^{32}_{24,T})\right].
$$

$$
L_{new}=L_{existing}+\lambda(u)L_{hidden},\qquad
\lambda(u)=0.05\min(u/500,1).
$$

`0.05` 是整个三项平均 loss 的权重，不是每项各乘 0.05 再相加。它是保守的起始超参数，**没有被验证为最优，也不代表占总梯度 5%**。日志分别记录三项误差与当前权重，后续可以独立比较 0.01 / 0.05，而不是同时修改结构和一堆目标。

原有 loss 的 ramp 完成后仍为：最终 clean MSE×0.1，8×8 和 16×16 clean MSE 各×0.5，DMD surrogate×1，LPIPS×0.25，variance-head MSE×0.01。原 on-policy clean MSE 的权重在原 1000 步 ramp 后仍为 0。这次不顺手改变它，避免混淆原因。

## 4. 哪些不变，为什么不会牺牲推理速度

- 仍是三步，global t=999/599/299，8→16→32，noise filling。
- 仍是共享 24 层 DiT-L/2。8×8 全计算；16×16 复用第 3、6、8、11、14、17、19、22 层；32×32 复用第 3—22 层。
- 仍微调整个生成器，不是只训练 adapter；原 score critic 训练也保留。
- hidden capture 仅训练时显式打开，不注册永久 hook，不增加参数，不改 state_dict 键。
- 利用原训练已经执行的 same-input 教师前向取得 hidden，不再增加一套教师模型或额外教师前向；会增加少量特征保留、loss 和反向计算开销。
- 旧命令默认 `--hidden-weight 0`，完全关闭新增分支。已有训练目录或 checkpoint 的 hidden 目标不匹配时直接报错，要求独立输出目录。

因此**推理计算图没有改变**。不把这说成额外提速，也不声称视频质量已验证。

## 5. 如何判断是否真的有用

新实验从 A@5000 状态开始，与旧无hidden的 DMD＋LPIPS 实验采用相同模型、AdamW 状态、每个 rank 的 paired 随机数状态。**正式执行时直接复用旧实验的同更新数验证结果作为对照，不重跑一遍基线。** 两套重跑脚本仍保留以备需要。

| 配置 | 基线 | 新方案 |
|---|---|---|
| 初始状态 | A@5000 | 同一个 A@5000 |
| 目标 | 原 DMD＋LPIPS | 原 DMD＋LPIPS＋hidden |
| hidden 权重 | 0 | 500 步升至 0.05 |
| 本轮比较更新数 | 复用旧实验的 1000 / 2000 步结果 | 新训练 2000 步 |
| GPU | 旧实验为 4、5、6、7 | 新实验为 0、1、2、3 |
| batch | 每卡 64，全局 256，不拆生成器 microbatch | 同左 |
| 优化器 | AdamW，lr=1e-5，betas=(0.9,0.999)，wd=0.01，clip=1 | 同左 |
| 精度 | FP32 参数/优化器，BF16 autocast，TF32 关 | 同左 |
| score critic warm-up | 200 步 | 同左 |
| 原目标 ramp | 1000 步 | 同左 |

比较 500/1000/1500/2000 步的固定种子图像与 held-out 256 张结果。新训练额外记录三个 hidden NMSE。**hidden loss 降低不是过关标准**，还需要对象形状、结构和图像指标一起改善；256 张诊断也不能替代正式 FID。

如果只有 hidden NMSE 下降而图片没变化，不默认继续堆训练。下一步应单独测试恢复 fine 第 3—22 层中的少量实际计算层，让模型具有产生 cell 内新更新的能力；那属于另一个结构实验，需要重新测加速。

## 6. 代码和命令

| 文件 | 修改内容 |
|---|---|
| `2d/attention_analysis/dmd_cascade_reuse.py` | 可选 post-block hidden capture；原调用默认不变 |
| `2d/attention_analysis/dmd_hidden_distillation.py` | 三监督点、池化、归一化、stop-gradient、权重 ramp |
| `2d/attention_analysis/train_dmd_reuse_objective.py` | 可开关 loss、same-input 教师、日志/验证、resume 防误改 |
| `tests/test_dmd_hidden_distillation.py` | 输出/梯度一致性、跨 stage 梯度、教师冻结、池化/归一化测试 |
| `2d/attention_analysis/verify_dmd_hidden_smoke.py` | 比较真实四卡开关 hidden 的 smoke，检查初始原 loss 完全一致 |
| `2d/exp/run_dmd_reuse_hidden_ablation_20260916.sh` | 独立四卡 smoke 和配对 2000 步 pilot |

服务器根目录 `/scr/yuheng/DevOpsBench/multiscale_diffusion`：

```bash
# 三步全 batch 工程检查，不是画质训练。
bash 2d/exp/run_dmd_reuse_hidden_ablation_20260916.sh hidden smoke

# 两个独立 pilot；共用 GPU 锁，因此同时启动也会串行。
bash 2d/exp/run_dmd_reuse_hidden_ablation_20260916.sh baseline pilot
bash 2d/exp/run_dmd_reuse_hidden_ablation_20260916.sh hidden pilot
```

所有输出位于 `/home/yuheng/rdd_experiments/dmd_reuse_hidden_20260916/{smoke_hidden,pilot_baseline,pilot_hidden}/`，每组有 `launcher.log`、`train.jsonl`、`progress.json`、`protocol.json`、checkpoints 和验证 latent。`pilot_baseline` 是备用命令，不会为了本轮重复运行；2026-09-16 已按用户新指令启动 `pilot_hidden`，tmux 名称为 `rdd_reuse_hidden_pilot`。

新增独立 `rdd_reuse_hidden_monitor`：自动解码新目录的验证 latent、计算类别一致性，只在 0123 有足够显存时执行。新增验证从 500 步开始有独立256张集，之后每500步检查；模型只保存可恢复的完整 `latest.pt`，保留所有验证图片和指标。

例如解码新分支 2000 步的 256 张验证集（另选有至少 10 GiB 空余的卡，不与满载训练抢显存）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-rdd-eval/bin/python \
  2d/attention_analysis/decode_dmd_cascade_previews.py \
  --run /home/yuheng/rdd_experiments/dmd_reuse_hidden_20260916/pilot_hidden/holdout256 \
  --step 2000
```

把 `pilot_hidden` 换成 `pilot_baseline` 得到同种子基线；输出 `preview_0002000/comparison_first16.jpg` 和所有样本单图，`complete.json` 记录分类一致性（本机已缓存分类模型）。这不是 FID。

## 7. 验证状态

CPU 回归已通过：6 个新增测试＋5 个既有测试。包括 hidden 捕获前后默认输出、参数键、RNG、原 loss 的参数梯度完全一致；单独 hidden loss 能传回三个阶段的输入和所有 24 个 producer block；教师无梯度；池化项排除 cell 内零均值细节。

四卡真实 smoke 已通过：物理 0/1/2/3，每卡 64 张，总 batch 256，无生成器梯度累积；三个更新分别覆盖 fine/middle/coarse 的 score 训练阶段，DMD、LPIPS 和 hidden loss 都开启。所有 24 层的 QKV 权重均有有限、非零梯度，教师冻结。整个 smoke 日志中最高 allocated 显存 **25.429 GiB/卡**，没有 OOM；三步结束正常退出，不影响原 4567 上的训练。

注意 `progress.json` 的 `peak_gib` 是最后一次更新的峰值 23.721 GiB，不能误当作全程峰值；上面的 25.429 是对三次日志中的 `max_rank_peak_gib` 取最大。

64 张固定验证的 fine 最终 hidden NMSE 从 0.2283 变为 0.2198，但最终 latent MSE 仍为约 0.0750。**这只是三步工程 smoke，不是画质改善证据，也不能归因于 hidden loss。** 正式2000步训练的状态单独见下方第8节。

随后无 hidden 的同配置三步 smoke 也正常完成。初始各项旧 loss（各 stage MSE、on-policy、variance、DMD、LPIPS）逐项一致；总 loss 唯一的初始差值等于新增 `0.05 × hidden_loss`。无 hidden 的最高 allocated 显存为 25.319 GiB，新增监督在本次测试中多用约 112 MiB/卡。没有用这两次共享 GPU 的短跑比较训练吞吐。

### 本次启动前补充检查与空间清理

- 默认旧行为不变，给新 pilot 打开 `--latest-only`，避免每次验证额外生成约1.83GB权重快照。
- 提前到500步做独立256张评估，新增自动预览队列；不增加推理层或新的模型参数。
- 清理严格限旧 `control/checkpoints` 和 `dmd_lpips/checkpoints` 的编号历史权重。先用CPU读取并校验完整latest及模型head有限性，再删不超过latest更新数的历史文件；跳过临时文件、刚写入的文件和任何软链接。
- 2026-09-16首次清理16个文件，约29.303GB（27.291GiB）。旧control的latest为新增30000步，旧dmd_lpips首次清理时latest为新增10000步。日志、图片、验证latent、A@5000起点与所有教师权重均保留。
- 历史权重没有额外备份，删除不可直接恢复。后台仅继续执行这一明确范围的“保留latest”清理，不停止或更改旧训练。

<!-- HIDDEN_PILOT_LIVE -->

## 8. 本次正式 pilot：实测进度与图片

证据刷新（非实时）：2026-09-16T17:22:52+00:00。

状态：`training`；新增 2010 步（无步数上限），累计 7010 步。

最近训练 batch 的 hidden NMSE：{"middle_block24": 0.03356967493891716, "fine_block22_lowpass": 0.0919588953256607, "fine_block24": 0.10987231135368347}；不是画质分数。

最近一次更新，各 rank 最大 allocated 显存：23.721 GiB。

本轮只在旧目标上增加 hidden 监督。使用原无hidden实验的同起点、同训练步数、同seed结果作为对照，避免重复训练对照组。旧组不重新启动、不改变其目标。

| 新增更新 | 新 hidden：64张 MSE | 旧无hidden：64张 MSE | 新／旧类别Top-1 |
|---:|---:|---:|---|
| 0 | 0.07501 | 0.07501 | 11/64 / 11/64 |
| 1 | 0.07459 | 0.07432 | 13/64 / 12/64 |
| 200 | 0.07486 | 0.07437 | 14/64 / 14/64 |
| 500 | 0.07370 | 无同step验证 | 19/64 / 尚未解码 |
| 1000 | 0.07673 | 0.07596 | 27/64 / 27/64 |
| 1500 | 0.07747 | 无同step验证 | 28/64 / 尚未解码 |

### 独立256张验证（不是FID）

| 新增更新 | 新 hidden：MSE | 旧无hidden：MSE | 新／旧类别Top-1 |
|---:|---:|---:|---|
| 500 | 0.08098 | 无同step验证 | 68/256 / 尚未解码 |
| 1000 | 0.08392 | 0.08366 | 88/256 / 87/256 |
| 1500 | 0.08450 | 无同step验证 | 92/256 / 尚未解码 |

### 固定64张的hidden误差轨迹

这里只记录新组对同输入冻结教师的hidden误差；旧组没有对应记录，不能当作两组hidden误差的比较。

| 新增更新 | 16×16末层 NMSE | 32×32第22层池化 NMSE | 32×32末层 NMSE |
|---:|---:|---:|---:|
| 0 | 0.05901 | 0.12627 | 0.22826 |
| 1 | 0.05949 | 0.12605 | 0.22840 |
| 200 | 0.05232 | 0.12312 | 0.16588 |
| 500 | 0.04240 | 0.11271 | 0.12559 |
| 1000 | 0.03659 | 0.10007 | 0.11551 |
| 1500 | 0.03460 | 0.09341 | 0.10942 |

**1000步同预算检查：目前没有证据表明额外hidden监督优于原配方。** 已逐一对照两组固定64张与独立256张的前16组图片。主要物体布局、细节变化和错误形态很接近；犬类／人物五官、婴儿、灯罩和文字仍明显损坏，没有看到稳定一致的视觉优势。

同seed独立256张：新hidden组Top-1为88/256，旧无hidden组87/256；Top-5为146/256与149/256；最终latent MSE为0.08392与0.08366。固定64张两组Top-1均为27/64，Top-5为39/64与40/64。差一张不能当作有效提升；这些指标也不是FID。

新组固定64张fine末层hidden NMSE从初始0.22826降到0.11551（约下降49.4%），但这一下降尚未转化为相对无hidden基线的画质收益。不能把新组相较自身500步的类别改善全部归因于hidden loss，因为旧配方随训练也会改善。

当前刚完成原目标1000步ramp；按计划继续到2000步，在完整权重下再检查同预算独立256张对照。不改变损失、复用结构或预算，也不保证再训练就会改善。

### 同新增1000步、相同种子的固定前16组

每一对左边为原始不跳层DMD，右边为该组复用学生。所有图固定取前16组，不挑选成功样本。

#### 原目标，无hidden监督：固定64张中的前16组

![原目标，无hidden监督](../../eval_results/dmd_reuse_objective_20260915/dmd_lpips/preview_0001000/comparison_first16.jpg)

#### 原目标，无hidden监督：独立256张中的前16组

![原目标，无hidden监督](../../eval_results/dmd_reuse_objective_20260915/dmd_lpips/holdout256/preview_0001000/comparison_first16.jpg)

#### 新增hidden监督：固定64张中的前16组

![新增hidden监督](../../eval_results/dmd_reuse_hidden_20260916/pilot_hidden/preview_0001000/comparison_first16.jpg)

#### 新增hidden监督：独立256张中的前16组

![新增hidden监督](../../eval_results/dmd_reuse_hidden_20260916/pilot_hidden/holdout256/preview_0001000/comparison_first16.jpg)

### 新组最新：新增1500步（没有对应旧组同step预览）

下图仍是左原始不跳层DMD、右新增hidden的复用学生；不能与上方不同更新数的旧组作同预算胜负结论。

**1500步人工检查：仍未看到明显的整体画质恢复。** 已查看固定64张、独立256张各自的前16组。铠甲、栗子等部分对象仍可辨认，但蚂蚁、犬类五官、人物、婴儿、灯罩、眼镜盒和马桶等样本仍有显著结构丢失或涂抹。这里是固定展示，不是只挑失败图。

固定64张Top-1为28/64、Top-5为43/64，最终latent MSE为0.07747；独立256张Top-1为92/256、Top-5为148/256，MSE为0.08450。旧组没有1500步的验证，不能据此宣布hidden监督优于旧目标；1000步同预算对照仍是当前可用的直接比较。

固定64张fine末层hidden NMSE继续降到0.10942（相对初始下降约52.1%），仍然不等于画质恢复。按既定计划完成2000步后再做同种子、同预算的最终比较，不自动加训练步数或修改复用层。

#### 固定64张中的前16组

![新hidden组最新预览](../../eval_results/dmd_reuse_hidden_20260916/pilot_hidden/preview_0001500/comparison_first16.jpg)

#### 独立256张中的前16组

![新hidden组最新预览](../../eval_results/dmd_reuse_hidden_20260916/pilot_hidden/holdout256/preview_0001500/comparison_first16.jpg)

**评价原则：hidden误差下降不等于画质变好；三步smoke不是质量实验。** 当前不作训练后视频质量声明。执行安排以文首最新用户授权为准；预算改变不意味着已经证明了画质收益。
