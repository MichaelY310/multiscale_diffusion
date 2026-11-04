# RDD 步数分配诊断：运行与复现

本目录用于回答两个问题：

1. `100/300/600` 为什么明显差于 `300/300/400`？
2. 只把 `8×8` stage 从 400K 继续训练到 600K，是否足以补回差距？

所有训练分段在文档中统一按 `[32×32, 16×16, 8×8]` 书写；采样顺序相反，为 `8→16→32`。

## 标准 50K 指标

`guided-diffusion/evaluations/evaluator.py` 计算五项：

- Inception Score；
- FID；
- sFID；
- Precision；
- Recall。

旧 evaluator 在 P/R 使用 `np.bool`，NumPy 1.24+ 会在前三项完成后崩溃。现已改为 `np.bool_`。`2d/sample_fid_unified.py` 也只有在五项全部存在时才认定评估完成，并使用临时文件原子写日志与 `.metrics.json`；只有 FID 的半截日志会自动重算。

## 不占 GPU 的诊断

入口：

```bash
bash 2d/diagnostics/run_stage_allocation_cpu_diagnostics.sh \
  2>&1 | tee logs/stage_allocation_cpu_driver.log
```

它依次完成：

1. 将既有 hybrid 50K PNG 打包为 NPZ；
2. 在固定 10K 子集上计算 64/128/256 Laplacian-pyramid 能量；
3. 计算去均值亮度图的 radial FFT；
4. 写出 64px、128px 低通 NPZ；
5. 用 ADM evaluator 分别计算各组 10K 低通五项指标。

训练日志另由 `analyze_training_losses.py` 解析，输出每个 stage 的 moving-average 曲线、最后两个 10K 窗口均值与近期斜率。它只判断“同一 run 是否还在收敛”；由于不同步数分配让 stage 负责不同的 global timestep 区间，不能用 raw loss 绝对值跨 schedule 排名。

低通 10K FID 只用于定位大结构/中等结构问题，不能和论文的 50K 标准 FID 横向比较。多样本 evaluator 会在同一尺度复用一次 reference activations，指标定义没有改变。

### Laplacian 到底在看什么

可以把一张图想成三张叠在一起的透明纸：

1. 第一张只画最大的色块和轮廓（把图缩到 64px 再放回 256px）；
2. 第二张只画从 64px 增加到 128px 才出现的眼睛、腿和中等边缘；
3. 第三张只画从 128px 增加到 256px 才出现的毛发、草叶和细边。

程序分别把这三张“透明纸”的像素平方后取平均，得到三层能量及其比例。这回答的是“模型把多少变化放在大结构、中等结构、细节中”。另外还算传统四邻居 Laplacian：`上 + 下 + 左 + 右 - 4×中心`。平坦区域会互相抵消为 0，急剧变化的边缘或噪点会变大，所以它只是一把粗略的“局部锐利程度尺”。值大不必然更好，因为噪声同样会让它变大。

### FFT 到底在看什么

FFT 可以理解成图片版的音乐均衡器。音乐可拆成低音、中音和高音；图片也可拆成：

- 低频：天空、背景、主体位置和大轮廓；
- 中频：局部形状和较粗边缘；
- 高频：毛发、草叶、细线和锐边，也可能是噪点。

脚本先去掉每张图的平均亮度，再统计低/中/高频各占总变化能量的比例，并画完整 radial curve。FFT 与 Laplacian 是两把不同的尺：前者按“变化快慢”分解，后者按 64/128/256 的空间层级分解。二者都只能解释现象，不能单独判定生成质量，因此还要同时看 VAE control、低通 FID、标准 50K 指标和 pred-x0 图。

原始 100/300/600 用 2 个 sampling rank，8×8-continued 用 4 个。采样器按 rank 交错分配索引并按 batch 起点设种，所以相同 NPZ index 不是同一随机轨迹，不能做逐图 paired test。报告使用两组独立 10K 样本的均值差及 Welch 标准误。

## 小规模 GPU 诊断

这些脚本在物理 GPU 0 上串行运行；若 GPU 0 有任何 compute PID，它们只等待，不杀进程。

```bash
bash 2d/diagnostics/run_pred_x0_after_gpu0_idle.sh
bash 2d/diagnostics/run_teacher_forced_after_pred_x0.sh
bash 2d/diagnostics/run_local_solver_error_after_teacher_forced.sh
bash 2d/diagnostics/run_class_consistency_after_local_solver.sh
```

顺序与用途：

1. `pred-x0`：固定 8 个类别、随机种子、DDPM 80/80/90、noise filling、CFG 1.0/1.5；比较所有 schedule 边界及 `boundary-1`。
2. `teacher-forced`：在真实 ImageNet validation latent 的正确 RDD forward 上，分开“模型自身预测误差”和“当前分辨率天然表示不了的误差”。
3. local solver error：一次大步与两次半步的差异，只是数值难度 proxy，不是 FID。
4. class consistency：用 ResNet-50 V2 检查旧 50K 样本是否遵循抽到的类别；标签恢复分别使用原始实验的 2-rank 和 continued 实验的 4-rank RNG 配置，不能混用。它是语义辅助指标，不代表视觉质量。

另有一个必要的 VAE control：

```bash
bash 2d/diagnostics/run_vae_frequency_control_after_gpu1_idle.sh
```

它把同一批 10K 真实参考图做确定性的 SD-VAE posterior-mean encode→decode，然后重做 FFT/Laplacian。没有这个 control，就不能区分“RDD 丢了高频”和“所有 latent diffusion 都会受到 VAE 瓶颈”。脚本等待物理 GPU 1 空闲，VAE reconstruction 完成后又等待主 CPU 诊断结束才开始 FFT。

当前三组 checkpoint：

- `RDD_300_300_400`，三 stage 均 400K；
- `RDD_100_300_600`，三 stage 均 400K；
- `RDD_100_300_600_8x8continued600K`，32/16 为 400K，只有 8 为 600K。

## 600/300/100

`600/300/100` 必须训练到与前两者相同的 400K 才能公平进入主比较。完成后运行：

```bash
bash 2d/diagnostics/run_pred_x0_add_600_when_400k.sh
```

脚本会先验证三个 `0400000.pt` 都存在，并等待 GPU 0 空闲；它不会拿早期 checkpoint 冒充公平结果。

## 报告

所有任务完成后：

```bash
python 2d/diagnostics/build_stage_allocation_report.py \
  --repo-root /scr/yuheng/DevOpsBench/multiscale_diffusion \
  --bundle-dir eval_results/diagnostics/RDD_STAGE_ALLOCATION_REPORT_BUNDLE
```

输出为自包含 HTML bundle，包含图、表、完整日志、CSV/JSON、配置与源码快照。`manifest.json` 的八个 completion 字段必须全部为 `true`。

测试：

```bash
python tests/test_stage_allocation_diagnostics.py
```

测试覆盖金字塔重构/正交能量、五项指标完整性、独立样本统计不误作 paired test，以及所有 producer schema 到最终 HTML 的集成。
