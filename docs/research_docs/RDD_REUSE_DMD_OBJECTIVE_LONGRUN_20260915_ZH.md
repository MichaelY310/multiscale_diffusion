# RDD两级复用：长训练与分布匹配对照

<style>#content{overflow-wrap:anywhere}#content table{display:block;max-width:100%;overflow-x:auto}#content img{max-width:100%}</style>

本轮两组都用A的推理结构：图像8→16复用8/24层，16→32复用20/24层；无adapter，整个共享生成器可训练。视频对应10/30、25/30层。结构不再改变，比较的是**训练配方**，不是重新搜索跳层数量。

上一轮同路径RDD＋VSA基线下，A在1280×768×61帧的采样＋VAE延迟减少22.17%。这是已有视频计算成本证据，不是本轮训练后的视频画质或速度验证。

## 1. 两组怎么安排

| 参数 | MSE对照组 | DMD＋LPIPS新方案 |
|---|---|---|
| GPU | 0,1,2,3 | 4,5,6,7 |
| 起点 | 同一个A@5000的生成器、AdamW状态、四rank训练噪声状态 | 完全相同 |
| 训练长度 | 再训练30,000更新，累计35,000 | 相同；另有200次fake-score预热 |
| 每次有效batch | 256；优先每卡64 | 256；优先每卡64 |
| 生成器优化器 | AdamW，lr=1e-5，betas=.9/.999，wd=.01，clip=1 | 相同 |
| 推理 | 3 NFE，999/599/299，noise filling | 相同 |
| 参数与精度 | FP32参数／优化器，BF16 autocast，TF32关闭 | 相同 |
| 损失 | 保持上一轮MSE配方 | 逐步减弱fine逐点MSE，加入分布匹配与图像LPIPS |
| 新增推理参数／计算 | 0 | 0；额外模型只在训练时使用 |

这是有明确预算的长期实验，不保证增加步数一定能恢复画质。不得仅凭训练loss下降报告成功。

## 2. MSE对照组不变的目标

```text
final clean MSE
+ 0.5 × (coarse clean MSE + middle clean MSE)
+ 0.25 × on-policy clean MSE
+ 0.01 × variance-head MSE
```

MSE目标来自冻结的原三步DMD生成器。验证与新一轮全部训练均使用学生自己完整三步的输入，不重新启用teacher-input混合。

## 3. 新方案：让压缩生成器恢复生成分布，而不只拟合每个latent数值

有三个角色，不能混淆：

1. **原三步DMD生成器**：冻结，给相同噪声／类别提供参考图和适度的阶段约束。
2. **原400K RDD diffusion模型**：冻结，在任意score timestep提供目标分布的score；不是三步DMD模型。沿用原实验CFG1.5及其epsilon前三通道指导行为。
3. **fake-score模型**：独立dense DiT-L/2，学习当前复用生成器输出的分布。它从原diffusion模型初始化，先训练200次；之后每次生成器更新配一次fake-score更新，AdamW lr=1e-5，clip=10。

分布匹配作用在**完整8→16→32 rollout产生的最终样本**上。查询低分辨率score时，平均池化最终样本，而不是把中间阶段输出当作最终生成分布。梯度通过pool、两次noise filling和全部跨stage缓存回传。score全局时间范围为20…979，按各stage覆盖长度加权抽样，沿用仓库已验证的RDD drift梯度公式。

前1000个生成器更新，令r从0逐渐增加到1：

```text
(1 - 0.9r) × final paired MSE
+ 0.5 × (coarse paired MSE + middle paired MSE)
+ 0.25(1-r) × on-policy MSE
+ 0.01 × variance-head MSE
+ r × RDD distribution matching
+ 0.25r × decoded-image LPIPS
```

fake-score预热与额外score随机数使用独立随机数流，不消耗两组用于配对训练的共同噪声流。

**这不是原DMD论文或DMD2的逐项复现。** 它是本项目的目标函数对照：图像LPIPS参考是在线的原DMD输出，不是原训练那25,000组多步教师固定pair；没有GAN、没有新增adapter。不能把组合收益单独归因于某一个loss。

感知损失涵盖每卡的全部64个样本，但每次仅解码4个，算出对latent的梯度并释放VAE/VGG计算图，最后按链式法则传回生成器。已用独立单元测试比较分块和整体损失的梯度；这不改变生成器有效batch，也不是只抽四张图计算loss。已通过先2张、再4张解码的四卡测试，后者峰值分配显存23.62GiB。

## 4. 先做的粗阶段替换诊断

固定64个验证seed，在未做本轮新训练的A@5000上：

| 验证路径 | 最终latent MSE↓ |
|---|---:|
| 完整A学生 | 0.0750074 |
| 仅把8×8的输出＋对应层缓存换回原DMD，再继续学生16/32阶段 | 0.0843111 |

替换后没有改善，反而变差。因此不能因coarse相对原模型出现误差就认定它被损坏；它可能已与后续学生共同适应。这个干预也会引入组合分布变化，不能证明粗阶段本身更好。**本轮不强制冻结或永久换回原coarse模型。**

## 5. 检查什么才算改善

- 保留上一轮固定64张验证图用于连续对比。
- 1000、2000及后续5000倍数更新增加256张新的独立验证图，不用这些seed训练。
- 对比完整图片、类别一致性与latent保真；这些都不是FID。
- 完整采样明显改善后，再做同样样本数的FID、sFID、IS、precision和recall。当前不把64/256张小样本测试冒充正式分布评估。
- 记录梯度范数、DMD梯度RMS、fake-score损失、显存和完整checkpoint，异常只处理本任务。

## 6. 日志、checkpoint与监控

结果根目录：`/home/yuheng/rdd_experiments/dmd_reuse_objective_20260915`

```bash
tmux attach -t rdd_reuse_obj_control
tmux attach -t rdd_reuse_obj_dmd

tail -f /home/yuheng/rdd_experiments/dmd_reuse_objective_20260915/control/launcher.log
tail -f /home/yuheng/rdd_experiments/dmd_reuse_objective_20260915/dmd_lpips/launcher.log
```

日志step从0计本轮新增更新；`total_compression_updates`才是含前5000步的累计值。DMD组启动时`critic_warmup`是只训练fake-score，不代表生成器卡住。

每1000更新保存可恢复的完整最新状态；1、200、1000、2000以及5000倍数保存生成器快照并验证。保留两组全部历史实验；不会为了腾空间删除其他结果。原用户checkpoint及上一轮A/B checkpoint不改。

远程监控只读取这两个任务并按显存余量解码其预览，不发送进程终止信号。本机定时监控负责看日志、实际图片和修复本任务错误；常规健康进度不反复通知。

<!-- LIVE_EVIDENCE -->

## 7. 实际进度与最新图片（非实时页面）

本地证据刷新：2026-09-16T14:14:25+00:00。检查日志时以服务器最新内容为准。

| 组别 | 新增／累计更新 | 状态 | 实测峰值分配显存 |
|---|---|---|---|
| control | 30000／35000 | completed | 10.85 GiB |
| dmd_lpips | 10340／15340 | training | 23.61 GiB |

**MSE对照组已完成全部预算：新增30000步，累计35000步，正常结束并保存最终checkpoint。** 固定64张和独立256张最终预览均已完成并人工检查。与早期相比有局部边缘／可辨识度改善，但人物、婴儿、犬类面部和灯罩结构仍明显破碎或涂抹，不能称为恢复原模型质量。

最终独立256张：类别Top-1 81/256（31.6%），原DMD 208/256（81.2%）；最终latent MSE 0.06590。这些不是FID。该结果是对照组预算终点，不能与仍在训练的新配方作同预算胜负结论；不自动延长对照组训练。

[对照组最终独立验证前16组图片](../../eval_results/dmd_reuse_objective_20260915/control/holdout256/preview_0030000/comparison_first16.jpg)；两组等步数比较仍见下文。

两组目标函数不同，**不能比较训练total loss大小判断谁的图更好**。下表使用相同固定验证集的MSE；分布匹配可能允许图像偏离原轨迹，需要同时看感知与分布指标。

| 组别 | 新增更新 | 8／16／32 MSE | 类别Top-1（64张） |
|---|---:|---|---|
| control | 0 | 0.0253／0.0224／0.0750 | 11/64 |
| control | 1 | 0.0258／0.0218／0.0743 | 11/64 |
| control | 200 | 0.0243／0.0225／0.0750 | 14/64 |
| control | 1000 | 0.0238／0.0213／0.0729 | 12/64 |
| control | 2000 | 0.0233／0.0208／0.0718 | 15/64 |
| control | 5000 | 0.0238／0.0202／0.0702 | 17/64 |
| control | 10000 | 0.0244／0.0183／0.0667 | 20/64 |
| control | 15000 | 0.0252／0.0167／0.0635 | 18/64 |
| control | 20000 | 0.0261／0.0165／0.0629 | 20/64 |
| control | 25000 | 0.0246／0.0169／0.0625 | 19/64 |
| control | 30000 | 0.0250／0.0163／0.0616 | 23/64 |
| dmd_lpips | 0 | 0.0253／0.0224／0.0750 | 11/64 |
| dmd_lpips | 1 | 0.0256／0.0218／0.0743 | 12/64 |
| dmd_lpips | 200 | 0.0235／0.0221／0.0744 | 14/64 |
| dmd_lpips | 1000 | 0.0232／0.0205／0.0760 | 27/64 |
| dmd_lpips | 2000 | 0.0232／0.0219／0.0782 | 32/64 |
| dmd_lpips | 5000 | 0.0232／0.0208／0.0762 | 30/64 |
| dmd_lpips | 10000 | 0.0235／0.0182／0.0711 | 31/64 |

下面主图统一使用两组都完成的**新增10000步**，避免把较早完成的对照组与训练步数更少的新方案直接比较。各自更晚的单组预览另给链接。

**10000步人工检查：类别可辨识度继续缓慢增加，但没有明显的整体画质跃升。** 已核对两组同seed、同更新数的固定前16组和独立验证前16组，并回看新方案5000步图片。新方案在鬣狗毛发、汽车格栅与鞋子边缘等局部仍优于同一步数MSE对照；与自身5000步相比，大部分构图和失真模式几乎不变。婴儿、人物和犬类五官、蚂蚁、灯罩花纹与文字仍明显破损，不能称为恢复原DMD质量。

独立256张类别Top-1：对照65/256（25.4%），新方案117/256（45.7%），原DMD208/256（81.3%）。新方案5000步为105/256（41.0%），增加12张；固定64张仅从30/64变为31/64，同一步数对照20/64。这仍是小样本诊断，不是FID或统计显著性结论。

独立256张最终latent MSE：对照0.07289、新方案0.07822，新方案自身5000步为0.08038。继续原定30000新增更新预算，下一次计划等步数检查15000步；没有因为这次小幅变化启动大规模FID，也不声称视频质量已恢复。

**这里仍是没有hidden蒸馏的原长训练。** 另行实现的hidden loss只完成独立三步工程smoke，未并入本组，不能把本页10000步结果归因于新增hidden监督。

**相同新增10000步、相同独立256个seed**（未用于训练；不是FID）：

| 组别 | 最终latent MSE↓ | 类别Top-1 | 类别Top-5 |
|---|---:|---:|---:|
| 原DMD参考 | — | 208/256 | 240/256 |
| control | 0.07289 | 65/256 | 112/256 |
| dmd_lpips | 0.07822 | 117/256 | 173/256 |

### control：新增10000步，固定前16组

每组左边原DMD、右边复用学生；不挑图。

![原DMD与复用学生](../../eval_results/dmd_reuse_objective_20260915/control/preview_0010000/comparison_first16.jpg)

该组另已完成新增30000步：[最新单组预览](../../eval_results/dmd_reuse_objective_20260915/control/preview_0030000/comparison_first16.jpg)，目前不能与另一组不同更新数的图作等步数结论。

独立验证集：同为新增10000步的固定前16组，每组左原DMD、右复用学生。

![独立验证：原DMD与复用学生](../../eval_results/dmd_reuse_objective_20260915/control/holdout256/preview_0010000/comparison_first16.jpg)

额外独立验证：新增30000步，256张，完整指标[JSON](../../eval_results/dmd_reuse_objective_20260915/control/holdout256/preview_0030000/complete.json)。这也不是FID。

### dmd_lpips：新增10000步，固定前16组

每组左边原DMD、右边复用学生；不挑图。

![原DMD与复用学生](../../eval_results/dmd_reuse_objective_20260915/dmd_lpips/preview_0010000/comparison_first16.jpg)

独立验证集：同为新增10000步的固定前16组，每组左原DMD、右复用学生。

![独立验证：原DMD与复用学生](../../eval_results/dmd_reuse_objective_20260915/dmd_lpips/holdout256/preview_0010000/comparison_first16.jpg)

额外独立验证：新增10000步，256张，完整指标[JSON](../../eval_results/dmd_reuse_objective_20260915/dmd_lpips/holdout256/preview_0010000/complete.json)。这也不是FID。
