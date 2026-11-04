# RDD 视频 attention 加速：确定性边界、VSA 对比与 training-free 实验

> **2026-09 无 checkpoint Wan2.1 速度复核：** 已新增同骨干 FLOP 模型、真实 FastVideo sparse executor、RDD route-lifting Triton kernel、RDD output-reuse Triton kernel和不同 C/M/F step allocation。完整结论与可视化见 [RDD 视频速度方法对比](RDD_VIDEO_SPEED_METHOD_COMPARISON_ZH.html)。新基准不改变下文已有 2D 质量实验；它只回答视频形状下的速度潜力。

## 先给确切答案

- **RDD 用到视频上，在固定网络、固定总采样步数和固定 stage token schedule 时，一定减少算术量；但在同等视频质量下是否一定减少端到端时间，不能在视频实验前声称为定理。** 质量可能要求更多训练或更多采样步；数据搬运、VAE、通信和 kernel 利用率也会影响真实时间。
- **若比较的是 RDD+VSA 与 RDD 原生上一 stage route，在相同 fine sparsity、相同输出质量、相同 block-sparse kernel、且原生 route 不需要额外频繁刷新时，RDD 原生 route 严格更省。** 上一 stage attention 本来就是 RDD 生成的一部分；VSA 则每个 fine layer、每个 timestep 都要对当前 fine Q/K/V 再做 pooling、coarse attention、Top-K 和 coarse output。
- 但这个增量优势没有想象中大。VSA 论文报告：87.5% sparsity 时 coarse router 约占 attention runtime 的 14%。理想地完全删掉它，attention 子系统相对 VSA 最多约 **1.163×**；VSA 优化后 attention 只占端到端约 20%，因此额外端到端上界约 **1.029×**。这是“完全不再算 current-fine coarse branch”的上界；若质量补偿需要每层重新计算当前 fine centroid QK/V，则会加回 VSA coarse branch 的大部分工作。RDD 真正可能带来大收益的是多分辨率生成本身，而不是只省掉 VSA router。
- **同 stage 复用 attention probability 可以继续提速，但 2D 中有效的完整 A-cache 不能直接当成视频方案。** 长视频的 A 是 O(H·rho·N²·layers) 显存；应优先复用 O(H·M·K) 的 block route，或只在少数层/短窗口缓存 probability。

## 为什么 RDD 的多分辨率本身有确定 FLOPs 优势

固定采样步分配时，attention pair 计算量近似：
```
C_RDD = Σ_s L_s · N_s²
C_all_fine = (Σ_s L_s) · N_fine²
```
示例 token 数 [4096, 9216, 16384]、步数 [20, 15, 15] 时，RDD 的 pairwise-attention FLOPs 仅为全程 fine-resolution 的 **0.420**。这是算术量结论，不等于同质量端到端 speedup。

## RDD 原生 route 和 VSA 的真实区别

- VSA 对**当前 fine hidden state**做 cube pooling，当前 timestep 重新计算 coarse Qc/Kc/Vc、Ac、Top-K，并把 Oc 和 fine sparse output 经过 learned gate 融合；这不是 training-free。
- RDD 原生方案把**生成过程中本来已经算过的上一 resolution attention**变成下一 stage 的候选 block route；fine stage 只在选中的 block 内算当前 fine QK/AV。
- 因而在前述强假设下，RDD-native = VSA 减去 online coarse router；但若跨 stage route 漂移严重，需要 current-fine refresh 或保留更多 block，优势会缩小甚至消失。我们的 2D 结果正说明了这个限制。
- VSA 的 Oc 还承担被删区域的全局信息。若 RDD 用当前 fine K/V centroid 做同样的 global tail，就不再是“完全免费 coarse”；若直接复用上一 stage 的 coarse summary，计算最省，但过去 2D probe 中 parent-summary 明显弱于 current-fine summary。因此‘稳定快于 VSA’必须以同质量实验为条件。

## 为什么 FastVideo/VSA 能明显加速，而当前 RDD 2D 稀疏几乎不能

- 序列长度完全不是一个量级。RDD Unified XL 的8/16/32 latent经过patch=2后只有16/64/256 tokens；FastVideo官方VSA kernel benchmark使用 `q_seq_lens=49152`。后者比RDD最细stage长192倍。
- QKV/MLP等线性层成本随N增长，attention QK/AV随N²增长。因此N=256时QK/AV只占很小部分，删掉attention pairs并不会删掉QKV projection、output projection和MLP；N约5万时二次attention才成为主要瓶颈。
- VSA不是training-free硬删dense模型的route。它把coarse router、fine sparse attention和coarse/fine gate一起训练或微调，让模型适应高稀疏率；当前RDD实验直接复用原模型，固定上一stage route在部分seed会漏掉决定构图的block。
- VSA论文应区分两个数字：单独VSA把Wan-2.1 attention约加速6×，端到端从31秒降到18秒，约1.72×；FastVideo文档中的几十倍denoising加速还叠加了3-step DMD/sparse-distill，不能全部归因于稀疏attention。
- 原始依据：[VSA论文](https://arxiv.org/abs/2505.13389)、[FastVideo VSA文档](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md)、[49,152-token kernel benchmark](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/README.md)、[DMD sparse-distill配置](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/distillation/dmd.md)。

### 当前RDD kernel和官方VSA kernel是不是同一个

- 不是同一份代码。当前RDD的`packed64`是VSA风格实现：同样接收每个query block的selected key-block indices、用online softmax只计算选中QK/AV，但它支持2x2或4x4空间micro-block，并把若干小block打包到64-token compute tile。
- 官方FastVideo Triton VSA固定以64-token Q/KV blocks执行，输入是`q2k_idx`、`q2k_num`和`variable_block_sizes`，Q/KV长度必须是64的倍数。当前RDD 16-stage只有64 tokens，因此官方VSA只看到1个block，无法稀疏；32-stage有256 tokens，也只有4个block，最小非零保留率是25%。这是2D实验必须使用更小自定义block的原因。
- route来源与执行kernel可以解耦。到RDD video时，把上一resolution attention lift到fine的64-token时空cube网格，转换成VSA的三组metadata后，可以直接调用官方`block_sparse_attn_from_indices`；执行kernel不在乎route来自VSA current-fine coarse-router还是RDD上一stage。
- 但替换kernel只解决速度，不解决质量。低密度实验中legacy与packed输出cosine几乎完全相同，却一起崩坏，说明误差来自route遗漏和模型未适应稀疏attention，而不是Triton数值实现。视频方案仍需要coarse/global输出补偿、route refresh或sparse finetune。

### 官方VSA kernel会不会在不够稀疏时反而变慢：实测答案是会

以下是同一张RTX PRO 6000 Blackwell上直接加载FastVideo官方、未经修改的Triton64 forward kernel，与PyTorch dense SDPA的CUDA-event中位数。route metadata已预先构造，不计router成本；测试时该GPU还在运行训练，因此绝对毫秒数只能视作共享GPU初测，但跨N一致的交叉趋势非常明确。1.0×以下表示VSA比dense更慢。

![官方VSA密度—速度交叉曲线](../../eval_results/rdd_vsa_kernel_density_bench/vsa_density_speedup.svg)

| N tokens | dense SDPA ms | 实际保留率 | VSA ms | VSA/dense 加速 | 判断 |
|---:|---:|---:|---:|---:|---|
| 256 | 0.0133 | 25.0% | 0.0086 | 1.540× | 更快 |
| 256 | 0.0133 | 50.0% | 0.0108 | 1.228× | 更快 |
| 256 | 0.0133 | 75.0% | 0.0128 | 1.040× | 更快 |
| 256 | 0.0133 | 100.0% | 0.0148 | 0.898× | 更慢 |
| 1,024 | 0.0481 | 12.5% | 0.0202 | 2.381× | 更快 |
| 1,024 | 0.0481 | 25.0% | 0.0267 | 1.804× | 更快 |
| 1,024 | 0.0481 | 50.0% | 0.0431 | 1.116× | 更快 |
| 1,024 | 0.0481 | 75.0% | 0.0596 | 0.807× | 更慢 |
| 1,024 | 0.0481 | 87.5% | 0.0678 | 0.710× | 更慢 |
| 1,024 | 0.0481 | 100.0% | 0.0757 | 0.635× | 更慢 |
| 4,096 | 0.4468 | 12.5% | 0.1008 | 4.431× | 更快 |
| 4,096 | 0.4468 | 25.0% | 0.1825 | 2.448× | 更快 |
| 4,096 | 0.4468 | 50.0% | 0.3437 | 1.300× | 更快 |
| 4,096 | 0.4468 | 75.0% | 0.5055 | 0.884× | 更慢 |
| 4,096 | 0.4468 | 87.5% | 0.5860 | 0.762× | 更慢 |
| 4,096 | 0.4468 | 100.0% | 0.6673 | 0.669× | 更慢 |
| 16,384 | 9.8995 | 12.5% | 1.1589 | 8.542× | 更快 |
| 16,384 | 9.8995 | 25.0% | 2.9466 | 3.360× | 更快 |
| 16,384 | 9.8995 | 50.0% | 9.2827 | 1.066× | 更快 |
| 16,384 | 9.8995 | 75.0% | 13.9783 | 0.708× | 更慢 |
| 16,384 | 9.8995 | 87.5% | 15.0505 | 0.658× | 更慢 |
| 16,384 | 9.8995 | 100.0% | 18.4710 | 0.536× | 更慢 |
| 49,152 | 90.2844 | 12.5% | 19.3734 | 4.660× | 更快 |
| 49,152 | 90.2844 | 25.0% | 44.6974 | 2.020× | 更快 |
| 49,152 | 90.2844 | 50.0% | 84.9170 | 1.063× | 更快 |
| 49,152 | 90.2844 | 75.0% | 129.6249 | 0.697× | 更慢 |
| 49,152 | 90.2844 | 87.5% | 152.6315 | 0.592× | 更慢 |
| 49,152 | 90.2844 | 100.0% | 177.9170 | 0.507× | 更慢 |

- N>=1,024时，12.5%和25%保留率稳定更快；50%仍约1.06–1.30×；75%开始全面跌到0.70–0.88×。因此官方VSA也存在清晰的稀疏率交叉点，不是‘只要调用稀疏kernel就不会慢’。
- 100%保留时，官方VSA只有dense SDPA约0.51–0.67×；原因是它做了全部QK/AV后仍支付block-index读取、动态稀疏循环和更弱的连续访存/调度效率。
- N=256时官方64-token block只有4个KV blocks，最低非零密度25%；25%约1.54×、50%约1.23×、75%约1.04×、100%约0.90×。它并没有提供RDD 2D想要的12.5%细粒度route。
- 当前RDD 16-token packed kernel在N=256的25%/50%/75%初测为2.03×/1.29×/1.08×，与官方VSA的交叉形状一致；这不支持‘RDD kernel根本写坏了’。4-token kernel在密度升高后更差，主要是query tile利用率太低。共享GPU造成个别非单调离群点，正式绝对数字需独占复测。
- 因而现在没有理由把官方64-token kernel生硬魔改回RDD 2D：它会失去route粒度，且无法解决高密度本来就不划算的问题。到长视频N很大时，应直接复用官方64/128-token执行kernel，只替换route来源；若继续优化2D micro-kernel，目标应是16-token block的低密度路径，而不是追求75–100%密度胜过FlashAttention。
- 原始数据：[metrics.json](../../eval_results/rdd_vsa_kernel_density_bench/metrics.json)；官方实现：[FastVideo block-sparse API](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py)、[Triton64 kernel](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py)。

## 方法原语：先明确每个字母到底做什么

下面的字母是本报告内部的短名称，不是新的模型名。一个 setting 可以组合多个原语；例如 `R + S + P + F` 就是“上一 stage 给初始 route、按 block 稀疏计算、同 stage 复用 A、并周期性用 current-fine pooled QK 刷新 route”。

| 编号 | 方法 | 实际做法 | 主要参数 |
|---|---|---|---|
| R | 跨 stage route | 用 8-stage 末尾的 attention 决定 16-stage 候选 block，再用 16-stage 末尾 attention 决定 32-stage 候选 block | density、block_side、self block、route是否在stage内更新 |
| S | block-sparse attention | 只对 route 选中的 query-key block 运行 QK/softmax/AV | density16/32、block_side16/32 |
| P | 同 stage probability 复用 | 缓存 A=softmax(QK)，后续跳过Q/K并计算 A_old V_current | stage、layers、刷新周期K |
| F | current-fine route 刷新 | 对当前 fine Q/K 做 block pooling，每R步重新计算一次候选 route | pooled_route_refresh16/32 |
| C | centroid/global-tail 补偿 | 选中block逐token精算；未选block用当前fine K/V均值表示 | density或route_mass、block_side |
| E | 输出残差补偿 | 每R步算一次 O_dense-O_sparse，随后加到 sparse 输出 | residual_refresh16/32 |
| G | CFG probability 共享 | conditional 和 unconditional 分支共用同一个 A | stage、layers、跨步周期K |
| A | 自适应刷新判据 | latent相对RMS变化小于tau时命中P-cache，否则刷新 | tau、max_interval、sketch_stride |
| H | 时间分段周期 | 在stage前20%/中65%/后15%采用不同的A刷新周期 | time_score_intervals=(early,middle,late) |
| U | 本 stage route 更新 | 第一次由上一stage给route，后续用本stage上一步的attention更新route | temporal_route_update |
| W | dense warm-up | stage开头一段calls或模型最前几层保持dense | dense_warm_fraction、dense_warm_layers |
| M | Top-P route | 每个query保留累计route mass达到阈值的最小block集合 | route_mass_threshold |
| V | 方差加权route | K/V块内越不均匀，越提高该block被精算的优先级 | variance_route_beta |

### family 只是实验批次，不等于一种方法

| family | 包含的方法 | 数据规模 | 是否可用真实时间 |
|---|---|---|---|
| adaptive_probability | P固定周期，或P+A自适应周期 | 第一轮4 classes | 是；完整A-cache只适合作为2D control |
| probability_tuning16 | 与adaptive_probability相同，但专门细调16/32 stage的K和层数 | 第二轮16 classes | 是；tuning16指16个类别，不是16x16分辨率 |
| true_kernel | P+H，或R+S+P+F，或R+S+E | 第一轮4 classes | 是；空间稀疏配置走实际Triton kernel |
| cfg_probability_share | G，部分配置再叠加跨timestep的P | 第一轮4 classes | 是 |
| centroid_oracle | R+C及其W/M/V变体 | 第一轮4 classes | 否；先构造dense logits，只看质量上限 |
| centroid_tuning16 | 与centroid_oracle相同的扩大测试 | 第二轮16 classes | 否；tuning16仍表示16个类别 |

### 通用参数中最容易混淆的地方

- `density32=0.5`：目标是每个 query block 保留约50%的 key blocks；强制 self block 后实际密度可能略高。
- `block_side32=4`：一个空间 block 是4x4=16个tokens，不是总共4个block。
- `score_interval=3`：每3个 denoiser calls 重算一次 A；中间使用 A_old V_current。
- `score_layers=7`：只在前7个 Transformer blocks 复用A；不是前7个采样steps。
- `middle_density` 和 `middle_score_interval`：这里的 middle 指模型深度15%到85%的 Transformer blocks。
- `time_score_intervals=(5,3,2)`：这里才是采样时间轴，表示stage前20%每5步、中65%每3步、后15%每2步刷新。
- `pooled_route_refresh=6` 更新的是“算哪些block”；`score_interval=3` 更新的是“已选block里的attention probability”。
- `temporal_route_update=false`：除显式的F刷新外，route保持上一stage边界给出的版本；true表示后续用本stage最近一次A更新。

## 全部 training-free 实验：按方法组合重新分组

完整 seed 数：`{'centroid_oracle': 3, 'true_kernel': 3, 'cfg_probability_share': 3, 'adaptive_probability': 3, 'centroid_tuning16': 3, 'probability_tuning16': 3}`。speedup 是 denoiser wall-clock；cosine/PSNR 是同 class、同初始噪声、同 transition noise 相对 dense RDD 的逐像素相似度，不是 FID。centroid 分支先构造 dense logits，是质量 oracle，不能用其时间声称加速。

![速度—质量 Pareto](../../eval_results/rdd_training_free_attention_pareto.svg)

### P：固定周期复用 attention probability

这些 setting 不做空间稀疏。它们缓存完整A，跳过部分timestep的Q/K和softmax；适合验证时间冗余，但完整A的O(N²)显存不适合作为长视频最终实现。

| 实验批次(family) | setting | 方法组合 | 精确参数 | speedup | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|---:|
| adaptive_probability | `fixed_8_k2` | P 概率复用 | stage=8; 前28层; K8=2; 最后一步强制刷新 | 1.017× | 0.998654 | 0.997694 | 38.06 |
| adaptive_probability | `fixed_8_k3` | P 概率复用 | stage=8; 前28层; K8=3; 最后一步强制刷新 | 1.053× | 0.995869 | 0.993240 | 33.98 |
| adaptive_probability | `fixed_all_k2` | P 概率复用 | stage=8/16/32; 前28层; K8=2, K16=2, K32=2; 最后一步强制刷新 | 1.146× | 0.995073 | 0.993604 | 32.74 |
| adaptive_probability | `fixed_all_k3` | P 概率复用 | stage=8/16/32; 前28层; K8=3, K16=3, K32=3; 最后一步强制刷新 | 1.226× | 0.991005 | 0.985696 | 29.97 |
| probability_tuning16 | `fixed_16_32_k2` | P 概率复用 | stage=16/32; 前28层; K16=2, K32=2; 最后一步强制刷新 | 1.311× | 0.992019 | 0.988013 | 31.39 |
| probability_tuning16 | `fixed_16_32_k3` | P 概率复用 | stage=16/32; 前28层; K16=3, K32=3; 最后一步强制刷新 | 1.124× | 0.983246 | 0.973398 | 27.78 |
| probability_tuning16 | `fixed_16_32_k3_first14` | P 概率复用 | stage=16/32; 前14层; K16=3, K32=3; 最后一步强制刷新 | 1.026× | 0.990281 | 0.981727 | 30.66 |
| probability_tuning16 | `fixed_16_32_k3_first21` | P 概率复用 | stage=16/32; 前21层; K16=3, K32=3; 最后一步强制刷新 | 1.021× | 0.988259 | 0.979751 | 29.50 |
| probability_tuning16 | `fixed_16_32_k3_first7` | P 概率复用 | stage=16/32; 前7层; K16=3, K32=3; 最后一步强制刷新 | 0.912× | 0.995992 | 0.992403 | 35.07 |
| probability_tuning16 | `fixed_16_32_k4_first7` | P 概率复用 | stage=16/32; 前7层; K16=4, K32=4; 最后一步强制刷新 | 0.926× | 0.995179 | 0.993505 | 33.41 |
| probability_tuning16 | `fixed_16k2_32k3` | P 概率复用 | stage=16/32; 前28层; K16=2, K32=3; 最后一步强制刷新 | 1.119× | 0.991354 | 0.987457 | 30.98 |
| probability_tuning16 | `fixed_16k3_32k2` | P 概率复用 | stage=16/32; 前28层; K16=3, K32=2; 最后一步强制刷新 | 1.283× | 0.983665 | 0.973842 | 27.93 |

### P + A：用 latent 变化自动决定 cache hit/miss

| 实验批次(family) | setting | 方法组合 | 精确参数 | speedup | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|---:|
| adaptive_probability | `adaptive_16_32_tau02_max5` | P 概率复用 + A 自适应刷新 | stage=16/32; 前28层; tau=0.02; 最多5步不刷新; sketch_stride=2 | 0.988× | 1.000001 | 1.000000 | 126.02 |
| adaptive_probability | `adaptive_16_32_tau04_max5` | P 概率复用 + A 自适应刷新 | stage=16/32; 前28层; tau=0.04; 最多5步不刷新; sketch_stride=2 | 0.990× | 0.999997 | 0.999996 | 63.44 |
| adaptive_probability | `adaptive_16_32_tau06_max5` | P 概率复用 + A 自适应刷新 | stage=16/32; 前28层; tau=0.06; 最多5步不刷新; sketch_stride=2 | 0.999× | 0.999968 | 0.999961 | 54.34 |
| adaptive_probability | `adaptive_16_32_tau08_max5` | P 概率复用 + A 自适应刷新 | stage=16/32; 前28层; tau=0.08; 最多5步不刷新; sketch_stride=2 | 1.009× | 0.999309 | 0.998976 | 41.07 |
| adaptive_probability | `adaptive_all_tau02_max5` | P 概率复用 + A 自适应刷新 | stage=8/16/32; 前28层; tau=0.02; 最多5步不刷新; sketch_stride=2 | 0.994× | 1.000001 | 1.000000 | 126.02 |
| adaptive_probability | `adaptive_all_tau04_max5` | P 概率复用 + A 自适应刷新 | stage=8/16/32; 前28层; tau=0.04; 最多5步不刷新; sketch_stride=2 | 1.001× | 0.999262 | 0.998914 | 40.64 |
| adaptive_probability | `adaptive_all_tau06_max5` | P 概率复用 + A 自适应刷新 | stage=8/16/32; 前28层; tau=0.06; 最多5步不刷新; sketch_stride=2 | 1.011× | 0.998474 | 0.997179 | 38.39 |

### P + H：按采样早/中/晚阶段改变复用周期

| 实验批次(family) | setting | 方法组合 | 精确参数 | speedup | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|---:|
| true_kernel | `dense_time_score_4_3_2` | P 概率复用 + H 时间分段周期 | 16/32均100% dense; 前28层; stage时间前20%/中65%/后15%的A刷新周期=(4, 3, 2) | 1.157× | 0.993985 | 0.990723 | 31.51 |
| true_kernel | `dense_time_score_5_3_2` | P 概率复用 + H 时间分段周期 | 16/32均100% dense; 前28层; stage时间前20%/中65%/后15%的A刷新周期=(5, 3, 2) | 1.157× | 0.994665 | 0.991189 | 32.07 |
| true_kernel | `dense_time_score_5_4_2` | P 概率复用 + H 时间分段周期 | 16/32均100% dense; 前28层; stage时间前20%/中65%/后15%的A刷新周期=(5, 4, 2) | 1.181× | 0.985522 | 0.974107 | 28.49 |

### R + S + P + F：跨stage route、真实块稀疏、概率复用、current-fine route刷新

| 实验批次(family) | setting | 方法组合 | 精确参数 | speedup | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|---:|
| true_kernel | `route_refresh3_aggressive_score3` | R 跨stage路由 + S 块稀疏 + P 概率复用 + F fine路由刷新 | density16=87.5%; density32=75%; 中间层density32=87.5%; block16=2x2; block32=4x4; A刷新K16=3, K32=3; pooled-QK route刷新R16=3, R32=3; 强制self block | 1.067× | 0.987740 | 0.977843 | 28.34 |
| true_kernel | `route_refresh6_aggressive_score3` | R 跨stage路由 + S 块稀疏 + P 概率复用 + F fine路由刷新 | density16=87.5%; density32=75%; 中间层density32=87.5%; block16=2x2; block32=4x4; A刷新K16=3, K32=3; pooled-QK route刷新R16=6, R32=6; 强制self block | 1.027× | 0.987930 | 0.977846 | 28.44 |
| true_kernel | `route_refresh6_conservative_score3` | R 跨stage路由 + S 块稀疏 + P 概率复用 + F fine路由刷新 | density16=93.75%; density32=87.5%; block16=2x2; block32=4x4; A刷新K16=3, K32=3; pooled-QK route刷新R16=6, R32=6; 强制self block | 1.066× | 0.990420 | 0.982512 | 29.46 |
| true_kernel | `stage32_route_refresh3_keep50_score3` | R 跨stage路由 + S 块稀疏 + P 概率复用 + F fine路由刷新 | density16=100%; density32=50%; block16=4x4; block32=4x4; A刷新K16=1, K32=3; pooled-QK route刷新R16=0, R32=3; 强制self block | 1.072× | 0.983402 | 0.973070 | 27.32 |
| true_kernel | `stage32_route_refresh6_keep50_score3` | R 跨stage路由 + S 块稀疏 + P 概率复用 + F fine路由刷新 | density16=100%; density32=50%; block16=4x4; block32=4x4; A刷新K16=1, K32=3; pooled-QK route刷新R16=0, R32=6; 强制self block | 1.069× | 0.983495 | 0.972742 | 27.37 |

### R + S + E：跨stage route、真实块稀疏、dense-output residual补偿

| 实验批次(family) | setting | 方法组合 | 精确参数 | speedup | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|---:|
| true_kernel | `cascade_keep25_residual10` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=25%; density32=25%; block16=2x2; block32=4x4; dense residual刷新R16=10, R32=10; 强制self block | 0.947× | 0.976062 | 0.959050 | 26.76 |
| true_kernel | `stage32_keep125_residual10` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=100%; density32=12.5%; block16=4x4; block32=4x4; dense residual刷新R16=0, R32=10; 不强制self block | 1.002× | 0.994252 | 0.989005 | 32.04 |
| true_kernel | `stage32_keep125_residual5` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=100%; density32=12.5%; block16=4x4; block32=4x4; dense residual刷新R16=0, R32=5; 不强制self block | 0.985× | 0.998213 | 0.996662 | 36.89 |
| true_kernel | `stage32_keep25_residual10` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=100%; density32=25%; block16=4x4; block32=4x4; dense residual刷新R16=0, R32=10; 强制self block | 0.984× | 0.997080 | 0.994371 | 35.04 |
| true_kernel | `stage32_keep25_residual15` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=100%; density32=25%; block16=4x4; block32=4x4; dense residual刷新R16=0, R32=15; 强制self block | 0.961× | 0.995797 | 0.991945 | 33.55 |
| true_kernel | `stage32_keep25_residual5` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=100%; density32=25%; block16=4x4; block32=4x4; dense residual刷新R16=0, R32=5; 强制self block | 0.968× | 0.998835 | 0.997340 | 40.26 |
| true_kernel | `stage32_keep50_residual10` | R 跨stage路由 + S 块稀疏 + E 输出残差 | density16=100%; density32=50%; block16=4x4; block32=4x4; dense residual刷新R16=0, R32=10; 强制self block | 0.975× | 0.999492 | 0.999182 | 42.46 |

### G（部分再加P）：CFG cond/uncond probability共享

| 实验批次(family) | setting | 方法组合 | 精确参数 | speedup | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|---:|
| cfg_probability_share | `cfg_share_16_32_k1` | G CFG共享 | stage=16/32; 前28层; timestep K=1; cond的A供uncond使用 | 1.008× | 0.981562 | 0.977409 | 26.50 |
| cfg_probability_share | `cfg_share_16_32_k3` | G CFG共享 + P 跨步复用 | stage=16/32; 前28层; timestep K=3; cond的A供uncond使用 | 1.157× | 0.980762 | 0.969578 | 26.44 |
| cfg_probability_share | `cfg_share_all_k1` | G CFG共享 | stage=8/16/32; 前28层; timestep K=1; cond的A供uncond使用 | 0.976× | 0.976193 | 0.970614 | 25.58 |
| cfg_probability_share | `cfg_share_all_k1_first7` | G CFG共享 | stage=8/16/32; 前7层; timestep K=1; cond的A供uncond使用 | 0.993× | 0.997916 | 0.997285 | 35.56 |
| cfg_probability_share | `cfg_share_all_k2` | G CFG共享 + P 跨步复用 | stage=8/16/32; 前28层; timestep K=2; cond的A供uncond使用 | 1.104× | 0.973894 | 0.965785 | 25.20 |
| cfg_probability_share | `cfg_share_all_k3` | G CFG共享 + P 跨步复用 | stage=8/16/32; 前28层; timestep K=3; cond的A供uncond使用 | 1.182× | 0.968824 | 0.958900 | 24.50 |

### R + C：Centroid/global-tail质量oracle及其U/W/M/V组合

Centroid表中的speedup故意为空：这些实验先计算dense logits再模拟稀疏，只能回答质量是否值得继续实现，不能回答速度。

| 实验批次(family) | setting | 方法组合 | 精确参数  | cosine mean | cosine min | PSNR |
|---|---|---|---|---:|---:|---:|
| centroid_tuning16 | `stage32_centroid75` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=75%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.996839 | 0.996243 | 35.71 |
| centroid_tuning16 | `stage32_centroid625` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=62.5%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.993292 | 0.991137 | 32.18 |
| centroid_oracle | `stage32_centroid50` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=50%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.993185 | 0.988916 | 31.97 |
| centroid_tuning16 | `stage32_centroid50_warm20_layer1` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + W dense预热 | 仅32 stage; density32=50%; block32=4x4; 未选block用当前fine K/V均值补偿; stage前20% calls全层dense; 前1个Transformer block始终dense; 逐步用本stage的A更新route; 强制self block | 0.992986 | 0.990202 | 32.03 |
| centroid_tuning16 | `stage32_centroid_mass95` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + M Top-P route | 仅32 stage; 每个query保留累计route mass达到95%的最小block集合; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.992788 | 0.992093 | 31.76 |
| centroid_tuning16 | `stage32_centroid50_warm10_layer1` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + W dense预热 | 仅32 stage; density32=50%; block32=4x4; 未选block用当前fine K/V均值补偿; stage前10% calls全层dense; 前1个Transformer block始终dense; 逐步用本stage的A更新route; 强制self block | 0.991532 | 0.988608 | 31.12 |
| centroid_tuning16 | `stage32_centroid50` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=50%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.989557 | 0.984907 | 30.27 |
| centroid_tuning16 | `stage32_centroid375_warm20_layer1` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + W dense预热 | 仅32 stage; density32=37.5%; block32=4x4; 未选block用当前fine K/V均值补偿; stage前20% calls全层dense; 前1个Transformer block始终dense; 逐步用本stage的A更新route; 强制self block | 0.988893 | 0.984243 | 29.97 |
| centroid_tuning16 | `stage32_centroid_mass90` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + M Top-P route | 仅32 stage; 每个query保留累计route mass达到90%的最小block集合; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.988084 | 0.986734 | 29.51 |
| centroid_oracle | `stage32_centroid375` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=37.5%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.987936 | 0.979572 | 29.15 |
| centroid_oracle | `stage32_centroid25_warm20_layer1` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + W dense预热 | 仅32 stage; density32=25%; block32=4x4; 未选block用当前fine K/V均值补偿; stage前20% calls全层dense; 前1个Transformer block始终dense; 逐步用本stage的A更新route; 强制self block | 0.985689 | 0.976433 | 28.11 |
| centroid_tuning16 | `stage32_centroid_mass85` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + M Top-P route | 仅32 stage; 每个query保留累计route mass达到85%的最小block集合; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.984357 | 0.981454 | 28.23 |
| centroid_tuning16 | `stage32_centroid375` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=37.5%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.983729 | 0.976409 | 28.42 |
| centroid_oracle | `stage32_drop50` | R 跨stage路由 + S 仅删除(oracle) + U 本stage路由更新 | 仅32 stage; density32=50%; block32=4x4; 未选block直接删除; 逐步用本stage的A更新route; 强制self block | 0.982911 | 0.969510 | 27.17 |
| centroid_oracle | `stage32_centroid25_var10` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + V 方差加权route | 仅32 stage; density32=25%; block32=4x4; 未选block用当前fine K/V均值补偿; K/V块内误差route权重beta=1; 逐步用本stage的A更新route; 强制self block | 0.981616 | 0.969859 | 27.11 |
| centroid_oracle | `stage32_centroid25_var05` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + V 方差加权route | 仅32 stage; density32=25%; block32=4x4; 未选block用当前fine K/V均值补偿; K/V块内误差route权重beta=0.5; 逐步用本stage的A更新route; 强制self block | 0.981601 | 0.969835 | 27.11 |
| centroid_oracle | `stage32_centroid25` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=25%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.981052 | 0.969270 | 26.94 |
| centroid_oracle | `stage32_centroid125_warm20_layer1` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 + W dense预热 | 仅32 stage; density32=12.5%; block32=4x4; 未选block用当前fine K/V均值补偿; stage前20% calls全层dense; 前1个Transformer block始终dense; 逐步用本stage的A更新route; 不强制self block | 0.973610 | 0.956161 | 25.38 |
| centroid_oracle | `cascade_centroid25` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 16和32级联; density16=25%; density32=25%; block16=2x2; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 强制self block | 0.962755 | 0.942978 | 23.93 |
| centroid_oracle | `stage32_centroid125` | R 跨stage路由 + C centroid补偿 + U 本stage路由更新 | 仅32 stage; density32=12.5%; block32=4x4; 未选block用当前fine K/V均值补偿; 逐步用本stage的A更新route; 不强制self block | 0.962178 | 0.937566 | 23.78 |
| centroid_oracle | `stage32_drop25` | R 跨stage路由 + S 仅删除(oracle) + U 本stage路由更新 | 仅32 stage; density32=25%; block32=4x4; 未选block直接删除; 逐步用本stage的A更新route; 强制self block | 0.948544 | 0.922175 | 21.76 |
| centroid_oracle | `stage32_drop125` | R 跨stage路由 + S 仅删除(oracle) + U 本stage路由更新 | 仅32 stage; density32=12.5%; block32=4x4; 未选block直接删除; 逐步用本stage的A更新route; 不强制self block | 0.906962 | 0.871628 | 18.96 |

## 已经能下的实验结论

- 固定短周期 `A_old @ V_current` 在早期测试中最有希望：4-class round 1 的全 stage K=2 为1.146× / cosine 0.9951；16-class tuning 为1.311× / cosine mean 0.9920 / min 0.9880。但新增 seed 20260828 的 dense-K2 control 只有 cosine 0.9621，出现明显姿态/构图分叉。因此它是有收益的近似，不是跨 seed 稳健的 training-free 默认项；正式论文需要50K样本FID/precision/recall或至少更大seed集。
- 全 stage K=3 更激进：16-class tuning 只有1.124× / cosine 0.9832，困难 seed 的 min 0.9734；`16-stage K=3, 32-stage K=2` 同样较差，说明 16×16 stage 的误差会被32×32继续放大。若做 training-free control，K=2仍优于K=3，但现在不能把K=2表述为无损。
- 直接把 route 压到 50% 再定期用 current-fine pooled QK 刷新，只得到约 1.07×，同时 cosine 约 0.983；route refresh 没有自动解决遗漏的全局信息。
- centroid compensation 显著优于直接丢弃，但 training-free 下仍需较高密度：32-stage 保留 75% 时 cosine 0.9968，62.5% 时 0.9933，50% 时 0.9896。给第一层加入 20% dense warm-up 可把 50% 提到 0.9930，但这些数值来自先构造 dense logits 的质量 oracle，不能声称真实加速。
- Top-P attention-mass 路由没有胜过固定密度：mass 85/90/95 平均实际保留约 40.4%/47.7%/58.9% block，cosine 为 0.9844/0.9881/0.9928；与相近固定密度相比没有质量优势。当前不值得为 Top-P 增加动态排序和不规则 kernel 复杂度。
- dense-output residual 可恢复质量，却因定期 dense refresh 和额外 kernel 调度而慢于 dense；它更像 distillation target 或误差分析工具。
- CFG cond/uncond probability 共享会造成明显分布误差。只共享前 7 层接近保真，却没有净提速。
- 以 latent 相对 RMS 决定 cache hit 的自适应原型命中率低且包含同步开销；固定周期优于这个判据。

## 新增重点实验：8→16→32 原生级联 route 与 FastVideo 风格 packed kernel

### 这轮到底测试了什么

- `triton_dense_score2_all`：16/32 都不做空间删除，只做同 stage 的 probability K=2 复用，即刷新步计算 A=softmax(QK)，下一步计算 A_old V_current。它是判断空间稀疏是否真的额外有益的关键 control。
- `legacy_cascade_safe_k2`：8-stage 最后得到的 route 决定 16-stage block；16-stage 的稀疏结果再决定 32-stage block。stage 内不做 dense route refresh；16/32 都每2步刷新一次已选 block 内的 A。目标密度为93.75%/87.5%，加 self block 后实测约93.9%/87.6%。
- `packed_cascade_safe_k2`：数学与 legacy-safe 完全相同，只把选中 block 打包成较大的 tile，用 FastVideo/VSA kernel 同类的 online-softmax Triton 计算，`kv_tile=64`。
- `*_cascade_safe_nocache`：与 safe 级联保留完全相同的空间 route，但每一步都重新计算当前 QK/softmax（K=1）。它专门用来隔离困难 seed 中的误差究竟来自空间删 block，还是 K=2 probability cache。
- `*_cascade_keep125/1875/50_nocache`：16和32两个stage都严格使用上一stage给出的固定route，只保留12.5%、18.75%、50%；不强制self block、不复用A、不做current-fine route refresh、不加centroid/residual。当前每stage有16个空间route blocks，所以严格20%无法表示，最接近的是3/16=18.75%。
- `hybrid32_cascade_safe_k2`：16-stage 保留适合小 query-block 的 legacy micro-kernel，32-stage 改用 `kv_tile=32` 的 packed kernel。`balanced` 把目标密度进一步降到87.5%/75%。
- 所有图片均使用 Unified XL、DDPM 80/80/90、CFG 1.5、同 class/同初始噪声/同 transition noise。表中 cosine/PSNR 是相对 dense 的逐像素指标，不等于 FID。

### 每一轮的原始结果

| run | setting | backend | 实际保留block 16/32 | K16/K32 | denoiser加速 | cosine | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| explore-2cls | `packed_cascade_balanced_k2` | packed64 | 87.7%/75.1% | 2/2 | 0.900× | 0.990362 | 27.40 |
| explore-2cls | `packed_cascade_balanced_nocache` | packed64 | 87.7%/75.1% | 1/1 | 0.981× | 0.989009 | 26.82 |
| explore-2cls | `packed_cascade_equal875_k2` | packed64 | 87.7%/87.5% | 2/2 | 0.893× | 0.990476 | 27.46 |
| explore-2cls | `packed_cascade_middle_sparse_k2` | packed64 | 81.9%/71.5% | 2/2 | 0.965× | 0.985845 | 25.76 |
| explore-2cls | `packed_cascade_moderate_k2` | packed64 | 75.4%/62.7% | 2/2 | 0.847× | 0.983689 | 25.13 |
| explore-2cls | `packed_cascade_safe_k2` | packed64 | 93.9%/87.6% | 2/2 | 0.836× | 0.994956 | 30.22 |
| explore-2cls | `packed_cascade_safe_nocache` | packed64 | 93.9%/87.6% | 1/1 | 0.879× | 0.994467 | 29.80 |
| explore-2cls | `packed_stage32_75_k2_control` | packed64 | 100.0%/75.2% | 1/2 | 0.906× | 0.997602 | 33.48 |
| layout-4cls | `legacy_cascade_balanced_k2` | legacy | 87.7%/75.2% | 2/2 | 0.940× | 0.971147 | 25.48 |
| layout-4cls | `legacy_cascade_safe_k2` | legacy | 93.9%/87.6% | 2/2 | 1.098× | 0.991452 | 30.42 |
| layout-4cls | `packed_cascade_balanced_k2` | packed64 | 87.7%/75.2% | 2/2 | 1.029× | 0.971039 | 25.47 |
| layout-4cls | `packed_cascade_safe_k2` | packed64 | 93.9%/87.6% | 2/2 | 0.940× | 0.991412 | 30.39 |
| layout-4cls | `packed_stage32_75_k2_control` | packed64 | 100.0%/75.2% | 1/2 | 1.042× | 0.997815 | 35.44 |
| layout-4cls | `packed_strided_cascade_balanced_k2` | packed64_strided | 87.7%/75.2% | 2/2 | 0.900× | 0.971039 | 25.47 |
| layout-4cls | `packed_strided_cascade_safe_k2` | packed64_strided | 93.9%/87.6% | 2/2 | 0.783× | 0.991412 | 30.39 |
| layout-4cls | `triton_dense_score2_all` | legacy | 100.0%/100.0% | 2/2 | 1.046× | 0.998458 | 37.80 |
| hybrid-4cls | `hybrid32_cascade_balanced_k2` | hybrid_packed32 | 87.7%/75.2% | 2/2 | 1.100× | 0.989572 | 28.54 |
| hybrid-4cls | `hybrid32_cascade_safe_k2` | hybrid_packed32 | 93.9%/87.6% | 2/2 | 1.050× | 0.995124 | 31.87 |
| hybrid-4cls | `legacy_cascade_safe_k2` | legacy | 93.9%/87.6% | 2/2 | 1.132× | 0.995122 | 31.87 |
| hybrid-4cls | `packed_cascade_safe_k2` | packed64 | 93.9%/87.6% | 2/2 | 1.137× | 0.995219 | 31.96 |
| hybrid-4cls | `triton_dense_score2_all` | legacy | 100.0%/100.0% | 2/2 | 1.207× | 0.997691 | 36.44 |
| final-4cls | `legacy_cascade_safe_k2` | legacy | 93.9%/87.6% | 2/2 | 0.881× | 0.942637 | 22.60 |
| final-4cls | `packed_cascade_safe_k2` | packed64 | 93.9%/87.6% | 2/2 | 1.010× | 0.943313 | 22.65 |
| final-4cls | `triton_dense_score2_all` | legacy | 100.0%/100.0% | 2/2 | 1.016× | 0.962134 | 24.99 |
| nocache-4cls | `legacy_cascade_safe_nocache` | legacy | 93.9%/87.6% | 1/1 | 0.994× | 0.941494 | 22.36 |
| nocache-4cls | `packed_cascade_safe_nocache` | packed64 | 93.9%/87.6% | 1/1 | 1.020× | 0.941888 | 22.40 |
| density-4cls | `legacy_cascade_keep125_nocache` | legacy | 12.5%/12.5% | 1/1 | 1.037× | 0.792175 | 16.58 |
| density-4cls | `legacy_cascade_keep1875_nocache` | legacy | 18.8%/18.8% | 1/1 | 0.963× | 0.807309 | 16.97 |
| density-4cls | `legacy_cascade_keep50_nocache` | legacy | 50.0%/50.0% | 1/1 | 0.891× | 0.885532 | 19.51 |
| density-4cls | `packed_cascade_keep125_nocache` | packed64 | 12.5%/12.5% | 1/1 | 1.003× | 0.792162 | 16.58 |
| density-4cls | `packed_cascade_keep1875_nocache` | packed64 | 18.8%/18.8% | 1/1 | 1.008× | 0.807333 | 16.97 |
| density-4cls | `packed_cascade_keep50_nocache` | packed64 | 50.0%/50.0% | 1/1 | 0.973× | 0.885661 | 19.52 |

`explore-2cls` 只有2类且计时重复1次，只用于筛参数；下表聚合只使用4-class确认轮。由于同机 wall-clock 波动明显，应同时报告区间，不能只挑最快一次。

### 4-class 确认轮聚合

| setting | 确认runs | speedup中位数 [min,max] | cosine均值/min | PSNR均值 |
|---|---:|---:|---:|---:|
| `hybrid32_cascade_balanced_k2` | 1 | 1.100× [1.100,1.100] | 0.989572/0.989572 | 28.54 |
| `hybrid32_cascade_safe_k2` | 1 | 1.050× [1.050,1.050] | 0.995124/0.995124 | 31.87 |
| `legacy_cascade_balanced_k2` | 1 | 0.940× [0.940,0.940] | 0.971147/0.971147 | 25.48 |
| `legacy_cascade_keep125_nocache` | 1 | 1.037× [1.037,1.037] | 0.792175/0.792175 | 16.58 |
| `legacy_cascade_keep1875_nocache` | 1 | 0.963× [0.963,0.963] | 0.807309/0.807309 | 16.97 |
| `legacy_cascade_keep50_nocache` | 1 | 0.891× [0.891,0.891] | 0.885532/0.885532 | 19.51 |
| `legacy_cascade_safe_k2` | 3 | 1.098× [0.881,1.132] | 0.976404/0.942637 | 28.30 |
| `legacy_cascade_safe_nocache` | 1 | 0.994× [0.994,0.994] | 0.941494/0.941494 | 22.36 |
| `packed_cascade_balanced_k2` | 1 | 1.029× [1.029,1.029] | 0.971039/0.971039 | 25.47 |
| `packed_cascade_keep125_nocache` | 1 | 1.003× [1.003,1.003] | 0.792162/0.792162 | 16.58 |
| `packed_cascade_keep1875_nocache` | 1 | 1.008× [1.008,1.008] | 0.807333/0.807333 | 16.97 |
| `packed_cascade_keep50_nocache` | 1 | 0.973× [0.973,0.973] | 0.885661/0.885661 | 19.52 |
| `packed_cascade_safe_k2` | 3 | 1.010× [0.940,1.137] | 0.976648/0.943313 | 28.33 |
| `packed_cascade_safe_nocache` | 1 | 1.020× [1.020,1.020] | 0.941888/0.941888 | 22.40 |
| `packed_stage32_75_k2_control` | 1 | 1.042× [1.042,1.042] | 0.997815/0.997815 | 35.44 |
| `packed_strided_cascade_balanced_k2` | 1 | 0.900× [0.900,0.900] | 0.971039/0.971039 | 25.47 |
| `packed_strided_cascade_safe_k2` | 1 | 0.783× [0.783,0.783] | 0.991412/0.991412 | 30.39 |
| `triton_dense_score2_all` | 3 | 1.046× [1.016,1.207] | 0.986094/0.962134 | 33.08 |

### 客观结论

- safe 级联在 seed 20260826/20260827 上保持主体与布局，但在困难 seed 20260828 上发生构图分叉：legacy/packed K=2 cosine 约0.943。它不是一个跨 seed 稳健的 training-free 配置。balanced 更不稳定。
- 困难 seed 的无-cache隔离仍只有 cosine 约0.9415–0.9419，说明主因是固定上一 stage route 删错关键 block，而不是 K=2 probability cache。作为对照，同 seed 的纯 dense-K2 为0.962，证明概率复用也有误差，但不是级联失真的主要来源。
- safe 实际只删除约6.1%的16-stage block和12.4%的32-stage block，空间稀疏很弱；即使如此也会在某些 seed 触发明显轨迹分叉。当前不能把固定边界 route 直接作为无需训练的默认推理方法。
- 低密度完整级联给出了更明确的负结果：12.5% 的 legacy/packed cosine约0.792，18.75%约0.807，50%约0.886；12.5/18.75%基本破坏主体，50%虽保留语义但纹理和构图明显损坏。三档端到端速度都约等于或慢于dense，说明当前2D短序列里索引/调度开销大于稀疏QK/AV节省。
- 三轮确认中，dense-K2、legacy-safe-K2、packed-safe-K2 的 wall-clock 区间高度重叠，且排名会反转。结合 high-density 的小算术节省，当前不能声称 coarse-route sparsity 在2D短序列上带来稳定的额外净加速。
- packed64 在算子级 microbenchmark 能减少 legacy kernel 的索引/softmax开销，但 16-stage 每个 route block 只有4个 query tokens，填充到较大计算 tile 会浪费算力；packed32 hybrid 也没有稳定胜过 legacy。短序列 N=64/256、高保留率下，Flash/SDPA 或小型 micro-kernel 已经很强。
- 因而这轮真正的硬件结论是：代码路径已实现且数值正确，但**当前2D尺度没有足够长的序列，也没有足够高且跨 seed 安全的稀疏率，让 FastVideo 风格 packed kernel 稳定形成额外端到端优势**。下一步应先解决 route 可靠性（current-fine轻量刷新、teacher route蒸馏或sparse finetune），再把同一kernel移到视频长序列；不应继续对4-token query block做过拟合优化。

### 对比图片

![Packed cascade exploratory sweep](../../eval_results/rdd_packed_cascade_search/triton_hierarchical_comparison.png)

![Packed layout confirmation](../../eval_results/rdd_packed_layout_confirm/seed_20260826/triton_hierarchical_comparison.png)

![Hybrid32 confirmation](../../eval_results/rdd_hybrid32_confirm/seed_20260827/triton_hierarchical_comparison.png)

![Final packed-vs-legacy confirmation](../../eval_results/rdd_packed_final_confirm/seed_20260828/triton_hierarchical_comparison.png)

![No-cache isolation on the difficult seed](../../eval_results/rdd_nocache_isolation/seed_20260828/triton_hierarchical_comparison.png)

![Strict cascade 12.5, 18.75 and 50 percent](../../eval_results/rdd_strict_cascade_density_sweep/seed_20260828/triton_hierarchical_comparison.png)

## 视频上应该怎么组合

推荐的可扩展顺序是：

1. RDD 多分辨率 video denoising，先拿到基础 FLOPs 和 wall-clock 收益。
2. 8→16、16→32 只传 block route，不传完整 A；进入 fine stage 后保留一个短 dense/current-fine warm-up，随后周期性刷新 route。
3. 对遗漏 block 加 parameter-free centroid/global-tail；若质量仍不足，再做 sparse finetune 或 teacher distillation。
4. 同 stage 首先复用 route；只在 token 较少的 coarse stage、少量层或短时间窗口复用完整 probability。
5. 写融合 kernel：route lifting + Top-K/index packing + sparse QK/AV；不要把 Python gather 和 host-side decision 算进最终方法。

## Probability cache 的视频显存警告

以 fp16、12 heads、30 layers、12.5% density 为例：N=23,296 时全层 probability cache 约 45.5 GiB；coarse route cache 约 0.089 GiB；attention-output residual 约 2.0 GiB。N=61,440 时 probability cache 约 316 GiB。因此 2D 的完整 A 复用不能直接宣称为长视频方案。

## 主要原始文献

- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)
- [DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance](https://arxiv.org/abs/2505.14708)
- [DiTFastAttn: Attention Compression for Diffusion Transformer Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/0267925e3c276e79189251585b4100bf-Paper-Conference.pdf)
- [SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation via Error-aware Routing](https://arxiv.org/abs/2603.08982)
- [FastVideo 官方实现](https://github.com/hao-ai-lab/fastvideo)

## Round-1 对比图片

![Centroid compensation](../../eval_results/rdd_training_free_centroid_sweep/seed_20260819/centroid_compensation_comparison.png)

![True Triton kernel](../../eval_results/rdd_training_free_kernel_sweep/seed_20260819/training_free_kernel_comparison.png)

![Adaptive probability cache](../../eval_results/rdd_adaptive_probability_sweep/seed_20260819/adaptive_probability_cache_comparison.png)

![CFG probability sharing](../../eval_results/rdd_cfg_probability_share_sweep/seed_20260819/cfg_probability_share_comparison.png)

## 16-class tuning 对比图片

下图为 seed 20260822 的轻量预览；完整分辨率 PNG 保留在 surfi2 对应结果目录。其余两个 seed 的预览也已下载到本地结果目录。

![Centroid tuning 16 classes](../../eval_results/rdd_centroid_tuning16/seed_20260822/centroid_compensation_comparison_preview.webp)

![Probability reuse tuning 16 classes](../../eval_results/rdd_probability_tuning16/seed_20260822/adaptive_probability_cache_comparison_preview.webp)

## Wan2.1 零微调伪 RDD 端到端视频检查

现在已经在 Wan2.1-T2V-1.3B 上跑通完整三阶段视频采样，并将前一 stage 的 `4×4×4` block route 接到 FastVideo Triton64 sparse kernel。结果表明：原始 Wan 权重在 drift=0 的 boundary-only 三阶段轨迹上就会崩坏；单独使用 75%/50% coarse-guided route 对伪 RDD dense 只造成很小附加误差，而全层 `reuse_every=2` 的 attention-output 复用明显过激。

- [可直接播放七组视频的 HTML 报告](RDD_WAN_ZERO_FINETUNE_SAMPLING_SANITY_ZH.html)
- [方法、指标与诊断的 Markdown 报告](RDD_WAN_ZERO_FINETUNE_SAMPLING_SANITY_ZH.md)
