# 详细实验方案文档

> 本文档为每个必须完成的实验提供完整规划，按优先级排序。
> 所有实验脚本位于 `experiments_scripts/` 目录下。

> **2026-08-10审计说明：** 当前执行优先级以 docs/theory/experiment_plan.md 为准。本文件保留仍有研究价值的详细设计；旧HF数据只用于提出假设。实验4（convergence）已完成，实验2（iso-compute）已有主要证据但仍需正式DDP统计，实验8需按统一生产协议重测。实验10已改为真正的training-split ablation。原实验11（训练到7M）和实验15（当前论文内做distillation/TRACT）已删除。Video计划也已重写：不再预设RDD收敛快2倍，不把普通合成数据微调称为蒸馏，不再声称PyramidalWan不能使用高阶solver。

---

## P0 级别（阻塞性，必须做）

---

### 实验1: d^c=0 Progressive Resolution Baseline (Curriculum without Drift)

#### 1. Critique原文

- **Round 12 (Ablation)**: 要求factorial decomposition，拆分RDD各组件贡献
- **Round 35 (Controls)**: 质疑是否设置了proper controls来隔离drift的贡献
- **Round 40 (Mechanism)**: 追问RDD的改进到底来自drift本身，还是仅仅来自多分辨率训练的curriculum效应

核心质疑：**"你怎么知道不是multi-resolution curriculum本身带来的提升，而drift只是搭了curriculum的便车？"**

#### 2. 实验目的

通过训练一个**有curriculum但无drift**的模型，将RDD的性能增益分解为两部分：
- 多分辨率progressive training的贡献（curriculum效应）
- Drift项本身的贡献（频率分解指导）

这是经典的factorial decomposition实验设计。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 模型架构 | DiT-XL/2 |
| 训练步数 | 400K steps |
| Batch size | 256 |
| 硬件 | 8×A100 |
| 分辨率时间轴 | 与RDD一致：Stage 1 (64×64, t∈[0,300]), Stage 2 (128×128, t∈[300,600]), Stage 3 (256×256, t∈[600,1000]) |
| **关键区别** | Forward process中设置 d^c = 0，即去除所有drift项 |
| 其他所有超参数 | 与RDD Unified XL完全一致（lr, EMA decay, augmentation等） |

具体实现：在forward process计算中，将drift coefficient设为0，保留multi-resolution transition结构但移除drift的信号引导。

#### 4. 预期结果

| 模型 | FID@cfg=1.0 | FID@cfg=1.5 |
|------|-------------|-------------|
| DiT-XL/2 (baseline) | ~10.26 | ~3.2 |
| **d^c=0 (本实验)** | ~7-9 | ~4.5-6.0 |
| RDD Unified XL | ~4.5 | ~2.62 |

预期：d^c=0模型介于DiT和RDD之间，说明curriculum有一定贡献，但drift是主要驱动力。

#### 5. 如果未达预期

- **情况A: d^c=0与RDD差距很小（ΔFID < 0.5）**
  - 说明drift贡献不大，curriculum是主要功臣
  - 应对：论文需要重新定位为"多分辨率curriculum学习框架"，drift作为理论elegant的bonus
  - 写法调整：弱化§3的理论motivation，强化§4中curriculum设计的贡献

- **情况B: d^c=0比DiT还差**
  - 说明多分辨率without drift反而harmful（分辨率突变引入训练不稳定）
  - 应对：这反而支持drift的必要性——drift是连接不同分辨率的"胶水"
  - 写法调整：强调drift不仅提供频率指导，还稳定multi-resolution transition

- **情况C: d^c=0显著好于DiT（ΔFID > 3）但仍明显差于RDD（ΔFID > 1）**
  - 最理想情况：两者都有贡献，drift在curriculum基础上进一步提升
  - 写法：直接报告factorial decomposition，量化各组件贡献百分比

#### 6. 论文写法

- **位置**: §4.3 新增 "Factorial Decomposition" 小节
- **内容**: 一张ablation table，4行：
  1. DiT-XL/2 (standard)
  2. DiT-XL/2 (multi-res curriculum, d^c=0) ← 本实验
  3. RDD (drift only, no curriculum) ← 如果可行的话
  4. RDD Unified XL (full method)
- **叙述**: "To disentangle the contributions of progressive resolution training and the drift term, we conduct a factorial decomposition..."
- 计算contribution percentage: `(FID_DiT - FID_{d^c=0}) / (FID_DiT - FID_RDD) × 100%` 为curriculum贡献

#### 7. 对应脚本名

`experiments_scripts/exp01_curriculum_no_drift.sh`

---

### 实验2: Iso-Compute Fair Comparison

#### 当前状态

主要证据已经具备：

- 单进程训练microbenchmark：Unified XL / DiT per-step time = 1.258×。
- 估算FLOPs ratio = 1.312×。
- 因此RDD400K约等于DiT503K的microbenchmark时间，或DiT525K的估算FLOPs。
- 正确DDPM250/CFG1.0结果：RDD400K FID7.065，DiT500K FID17.464。

这已经表明优势不是同step下多用约25%计算所能解释。

#### 仍需完成

1. 在相同8-GPU DDP环境正式测量两者训练throughput、通信开销和peak memory。
2. 如果DiT525K checkpoint存在，补一次DDPM250/CFG1.0的50K评估；没有checkpoint时不为此训练到750K。
3. convergence图增加training steps、estimated wall-clock和estimated FLOPs三个横轴版本。
4. 报告microbenchmark与真实DDP的区别。

#### 论文判据

主结论使用RDD400K对DiT500K的近似time-matched比较。
不再使用RDD400K对公开DiT7M结果推导“3.7× wall-clock”或“17.5× less compute”。

### 实验3: Multi-seed Statistical Evaluation

#### 1. Critique原文

- **Round 9 (Statistical Rigor)**: "单次FID评估不提供置信区间，如何判断差异是否统计显著？"
- **Round 14 (Methodology)**: 要求报告统计显著性检验

核心质疑：**"FID差异可能只是随机波动，你的结果可复现吗？"**

#### 2. 实验目的

通过多seed评估提供FID的置信区间，证明RDD vs DiT的差异是统计显著的，而非随机变异。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 模型 | RDD Unified XL (已训练好) + DiT-XL/2 (已训练好) |
| 采样数 | 每次50K样本 |
| Random seeds | 3个不同seed (42, 123, 456) |
| CFG设置 | cfg=1.0 和 cfg=1.5 各做一组 |
| 总评估次数 | 3 seeds × 2 models × 2 CFG = 12次 |
| 采样器 | DDPM 250步 |
| 统计检验 | Welch's t-test (不假设等方差) |

#### 4. 预期结果

| 模型 | FID@cfg=1.5 (mean±std) |
|------|------------------------|
| DiT-XL/2 | 3.20 ± 0.05 |
| RDD Unified XL | 2.62 ± 0.04 |
| **p-value** | < 0.01 |

预期：
- 各模型的FID标准差 < 0.1（50K样本足够稳定）
- 两模型差异的t-test p-value < 0.01

#### 5. 如果未达预期

- **情况A: std很大（> 0.5）**
  - 应对：增加到5个seed以获得更可靠的估计
  - 检查是否50K样本不够，考虑增加到100K

- **情况B: 差异不显著（p > 0.05）**
  - 应对：honest报告，可能需要增大样本量或更多seed
  - 在论文中transparently报告p-value

- **情况C: RDD的std > DiT的std**
  - 应对：讨论RDD可能有更高的generation diversity导致FID评估波动大

#### 6. 论文写法

- **位置**: 贯穿全文——所有主要FID数字改为 `mean ± std` 格式
- **具体修改**:
  - Table 1: 所有FID/sFID/IS列改为 mean±std
  - §4.2 文字中："RDD achieves FID of 2.62±0.04 (mean over 3 seeds)"
  - 添加footnote或appendix报告完整统计检验结果
- **Appendix**: 完整的3-seed数据表 + t-test结果

#### 7. 对应脚本名

`experiments_scripts/exp03_multi_seed_eval.sh`

---

## P1 级别（重要，不做会降分）

---

### 实验4: Convergence Curve（已完成）

#### 协议

- Native ImageNet ILSVRC。
- Unified XL与DiT-XL/2。
- DDPM250、CFG1.0。
- 50K样本、ADM reference。
- Checkpoint：50K到500K；RDD450K缺失并保持空白。

#### 关键结果

| Checkpoint | RDD FID | DiT FID |
|---:|---:|---:|
| 50K | 47.411 | 77.150 |
| 100K | 20.488 | 47.327 |
| 150K | 13.755 | 35.742 |
| 200K | 10.664 | 29.662 |
| 300K | 8.149 | 23.125 |
| 400K | 7.065 | 19.781 |
| 500K | 6.461 | 17.464 |

RDD在整个区间均更优；RDD150K已优于DiT500K。
生成脚本为fig02_convergence_curve.py，图已插入论文。

#### 剩余工作

- 增加按microbenchmark wall-clock和估算FLOPs重标横轴的版本。
- 对关键checkpoint做多sampling seed，避免过度解释小于0.1 FID的差异。
- 不再声称固定的3.7×收敛倍数；报告具体checkpoint和compute-matched比较。

### 实验5: Per-stage CFG Schedule

#### 1. Critique原文

- **Round 32 (CFG)**: 指出"不同stage的signal-to-noise ratio不同，uniform CFG可能不是最优的；per-stage CFG可能进一步提升FID"

核心质疑：**"既然各stage的噪声水平不同，为什么用同一个CFG值？"**

#### 2. 实验目的

验证per-stage CFG调度是否能进一步压低FID，同时为RDD框架的灵活性提供额外证据。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 模型 | RDD Unified XL (已训练好，无需重新训练) |
| 类型 | **Inference-only 实验**（成本低） |
| CFG配置grid | 见下表 |
| 每组评估 | 50K样本FID |

CFG Schedule Grid（[Stage1/Stage2/Stage3]）：

| 编号 | Stage 1 (低频) | Stage 2 (中频) | Stage 3 (高频) | 备注 |
|------|---------------|---------------|---------------|------|
| A | 1.5 | 1.5 | 1.5 | Uniform baseline |
| B | 2.0 | 1.5 | 1.25 | 递减（高频少引导） |
| C | 2.0 | 1.5 | 1.0 | 强递减 |
| D | 1.5 | 1.25 | 1.0 | 温和递减 |
| E | 2.5 | 1.5 | 1.0 | 极端递减 |
| F | 1.0 | 1.5 | 2.0 | 递增（对照） |

#### 4. 预期结果

| 配置 | 预期FID | 说明 |
|------|---------|------|
| A (uniform 1.5) | 2.62 | 现有结果 |
| B (2.0/1.5/1.25) | ~2.48 | 最优候选 |
| C (2.0/1.5/1.0) | ~2.50 | 次优 |
| D (1.5/1.25/1.0) | ~2.55 | 温和改善 |
| E (2.5/1.5/1.0) | ~2.60 | 可能过强 |
| F (1.0/1.5/2.0) | ~2.80 | 预期变差 |

#### 5. 如果未达预期

- **情况A: Uniform已是最优（所有per-stage都不如uniform）**
  - 解释：drift对CFG的放大效应在各stage一致，无需差异化
  - 论文中作为"有意义的负面结果"报告

- **情况B: 改善很小（< 0.05 FID）**
  - 说明：在统计误差范围内，不能声称per-stage CFG有效
  - 需要结合实验3的std判断是否显著

- **情况C: 递增(F)反而最好**
  - 反直觉但可能有道理：高频细节需要更强引导
  - 需要revisit理论解释

#### 6. 论文写法

- **位置**: §4.6 或新增小节 "Per-stage Classifier-free Guidance"
- **内容**: 
  - 一张表格展示grid search结果
  - 分析最优CFG schedule的物理含义
  - 讨论：低频stage需要更强引导（structure重要），高频stage可以放松（细节多样性）
- **如果改善显著**: 更新main table中RDD的最佳FID
- **如果改善不显著**: 在appendix报告，main text提一句"uniform CFG is near-optimal"

#### 7. 对应脚本名

`experiments_scripts/exp05_per_stage_cfg.sh`

---

### 实验6: Drift Magnitude λ Sweep (Inference-only)

#### 1. Critique原文

- **Round 12 (Ablation)**: 要求理解drift的"剂量效应"
- **Round 48**: 追问drift correction在inference时的敏感性

核心质疑：**"Drift correction的强度是否可调？模型对drift magnitude有多敏感？"**

#### 2. 实验目的

验证drift在inference中的必要性和敏感性，理解最优drift强度是否恰好是训练时使用的值。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 模型 | RDD Unified XL (已训练好) |
| 类型 | **Inference-only 实验** |
| λ值 | {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5} |
| 含义 | 采样时将drift correction乘以系数λ: `drift_applied = λ × drift_predicted` |
| 每个λ | 50K样本FID |
| CFG | cfg=1.5 (固定) |

#### 4. 预期结果

| λ | 预期FID | 含义 |
|---|---------|------|
| 0.00 | > 5.0 | 完全移除drift correction |
| 0.25 | ~4.0 | 严重不足 |
| 0.50 | ~3.2 | 不足 |
| 0.75 | ~2.8 | 接近 |
| **1.00** | **2.62** | **标准（训练时匹配）** |
| 1.25 | ~2.9 | 略过强 |
| 1.50 | ~3.5 | 过强 |

预期曲线呈U型，最低点在λ=1.0附近。

#### 5. 如果未达预期

- **情况A: λ=0也还不错（FID < 4.0）**
  - 严重问题：说明drift correction在采样时贡献有限
  - 应对：检查是否模型已经"内化"了drift信息到score function中
  - 论文中需要讨论：训练时drift改变了学习到的score，即使inference不显式加drift也有效果

- **情况B: 最优λ不在1.0（如0.8或1.2）**
  - 有趣发现：train-test mismatch可能导致最优drift不是训练值
  - 应对：报告最优λ值，讨论可能的原因

- **情况C: λ对FID几乎没影响（曲线平坦）**
  - 说明模型对drift magnitude鲁棒
  - 这是好结果：意味着deployment时不需要精确调drift

#### 6. 论文写法

- **位置**: §4.x "Drift Magnitude Analysis" 或整合到ablation section
- **内容**:
  - 折线图：FID vs λ
  - 讨论：λ=0的严重退化证明drift correction是必要的
  - 讨论：U型曲线说明drift存在最优剂量
  - 理论联系：λ=1是SDE理论精确匹配值

#### 7. 对应脚本名

`experiments_scripts/exp06_drift_lambda_sweep.sh`

---

### 实验7: σ_h Sweep (Noise Filling Validation)

#### 1. Critique原文

- **Round 15 (Theory)**: 质疑σ_h=0作为最优noise filling参数的理论推导
- 论文§3.4已有承诺："We validate this choice empirically in §4"

核心质疑：**"σ_h=0的理论最优性是否有经验支撑？"**

#### 2. 实验目的

经验验证§3.4中关于σ_h=0是最优noise filling参数的理论推导，完成论文中已做出的承诺。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 模型 | RDD Unified XL (已训练好) |
| 类型 | **Inference-only 实验** |
| σ_h值 | {0, 0.05, 0.1, 0.2, 0.5, 1.0} |
| 含义 | 在upsampling时null space方向添加N(0, σ_h²)噪声 |
| 每个σ_h | 50K样本FID |
| CFG | cfg=1.5 |

#### 4. 预期结果

| σ_h | 预期FID | 说明 |
|-----|---------|------|
| **0.00** | **2.62** | **理论最优** |
| 0.05 | ~2.65 | 几乎无影响 |
| 0.10 | ~2.75 | 轻微退化 |
| 0.20 | ~3.00 | 明显退化 |
| 0.50 | ~4.00 | 严重退化 |
| 1.00 | ~6.00 | 噪声淹没信号 |

预期：FID单调递增with σ_h，验证理论。

#### 5. 如果未达预期

- **情况A: σ_h=0.05-0.1反而更好**
  - 说明轻微的null space噪声有正则化效果
  - 应对：报告实际最优值，软化"唯一最优"声明
  - 论文改为："σ_h=0 is near-optimal, with marginal improvement possible at σ_h≈0.05"

- **情况B: σ_h对FID影响极小（全部在2.62±0.1内）**
  - 说明模型对noise filling鲁棒
  - 论文强调：理论预测verified（σ_h=0最优），且方法对此超参数不敏感

- **情况C: 非单调（σ_h=0.2比σ_h=0.1好）**
  - 异常情况，需要debug
  - 检查实现是否正确

#### 6. 论文写法

- **位置**: §3.4 中已有placeholder，填入实验结果
- **内容**:
  - 表格：σ_h vs FID
  - 一句总结："As predicted by Proposition 3.2, σ_h=0 yields optimal FID, with monotonic degradation as σ_h increases"
  - 可选：appendix中给出更细粒度的sweep

#### 7. 对应脚本名

`experiments_scripts/exp07_sigma_h_sweep.sh`

---

### 实验8: Production Wall-Clock and Memory Benchmark

#### 已有microbenchmark

250-step、单H20、batch16结果：

| 模型 | ms/image | Speedup |
|---|---:|---:|
| DiT-XL/2 | 1333.4 | 1.00× |
| Unified XL | 703.4 | 1.90× |
| MS-300 | 596.6 | 2.24× |
| MS-100 | 598.5 | 2.23× |
| Unified L | 520.9 | 2.56× |

这些结果证明模型计算结构确实更快，但脚本没有CFG双前向，边界实现也不是production noise filling，因此不能直接写matched-quality deployment speedup。

#### 正式实验协议

- 同一张GPU、同一软件环境和精度。
- 真实checkpoint。
- Batch至少测试1、8、16。
- 10 warm-up + 30 timed batches。
- 同一CFG、同一boundary实现。
- DDPM使用相同真实NFE。
- DPM2按实际model forward计数，并与DDPM约250 NFE匹配。
- 记录mean、std、p50、p95、throughput和peak memory。
- 同时给pure model latency与包含VAE decode的end-to-end latency。

#### 必测配置

1. DiT DDPM250。
2. Unified XL DDPM250。
3. Unified L DDPM250。
4. MS-300 DDPM250。
5. Unified XL DPM2约244 NFE。
6. 最终quality-optimal allocation和CFG。

#### 论文输出

- FID vs latency Pareto。
- FID vs NFE Pareto。
- Latency / throughput / memory表。
- 清楚区分scheduled steps、NFE和wall-clock。

## P2 级别（加分项）

---

### 实验9: PSD/Frequency Analysis

#### 1. Critique原文

- **Round 30 (Frequency)**: "drift d^c据说捕获高频信息，有经验证据吗？"
- **Round 23 (Visual Quality)**: 要求分析RDD生成图像的频率特性

核心质疑：**"论文声称drift编码高频细节，需要频谱分析验证"**

#### 2. 实验目的

通过Power Spectral Density分析，经验验证drift term d^c确实主要包含高频信息，支撑论文的理论narrative。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 数据 | ImageNet验证集随机选取1000张图片 |
| 分辨率 | 各stage对应的分辨率 |
| 分析内容 | 对每张图计算d^c = x - Downsample(Upsample(x))，然后做2D FFT |
| 对比对象 | (a) 原图x的PSD, (b) low-freq部分的PSD, (c) d^c的PSD |

具体步骤：
1. 对1000张256×256图片，分别downsample到128×128再upsample回256×256
2. 计算drift: d^c = x_256 - Upsample(Downsample(x_256))
3. 对d^c做2D FFT → |FFT|²得到PSD
4. 计算radial average PSD（按频率距离average）
5. 统计高频(f > Nyquist/2)能量占总能量比例

#### 4. 预期结果

- d^c的PSD能量 > 70%集中在高频区域 (f > Nyquist/2)
- 原图x的高频能量约30-40%
- Radial PSD曲线：d^c在高频段显著高于原图的归一化PSD
- d^c的PSD应该近似flat或increasing with frequency（白噪声或蓝噪声特征）

#### 5. 如果未达预期

- **情况A: d^c也有显著低频成分（高频占比 < 50%）**
  - 说明drift不仅是"高频residual"，还包含structure信息
  - 应对：修改论文narrative，d^c是"multi-scale residual"而非纯"high-frequency detail"
  
- **情况B: 结果因图片类型而异**
  - 某些图片的d^c主要是高频（纹理丰富的），某些有低频成分（大面积smooth区域的边缘）
  - 应对：按ImageNet类别分组报告，讨论content-dependent特性

#### 6. 论文写法

- **位置**: §3.3 或 Appendix (Figure)
- **内容**:
  - 图：Radial PSD curves (3条线：original, low-freq, drift d^c)
  - 统计：高频能量占比的直方图
  - 可视化：几张示例图的d^c（应该看起来像edge map）
- **叙述**: "Figure X confirms that the drift term d^c predominantly captures high-frequency content, with XX% of its energy above the Nyquist/2 threshold..."

#### 7. 对应脚本名

`experiments_scripts/exp09_psd_analysis.py`

---

### 实验10: Native-ImageNet Training Stage-Split Ablation

#### 1. 已验证的旧数据现象

旧HF数据只作方向性参考，但在完全可比的400K、DDPM、constant boundary、80/80/90采样设置下：

| Training split [32,16,8] | CFG | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 300/300/400 | 1.5 | 8.865 | 5.602 | 193.56 | 0.8033 | 0.4159 |
| 100/300/600 | 1.5 | 10.680 | 6.343 | 163.83 | 0.7610 | 0.4206 |
| 300/300/400 | 1.25 | 11.645 | 5.686 | 139.20 | 0.7250 | 0.4784 |
| 100/300/600 | 1.25 | 15.216 | 6.432 | 115.41 | 0.6775 | 0.4768 |

100/300/600虽然让32分辨率模型只覆盖t=0--100、每个离散t平均获得约3倍样本，但整体质量反而更差。
这不是单纯采样步数问题：它在匹配自身区间的1000-step采样下FID仍为10.781，与250-step的10.680相当。

#### 2. 关键解释假设

1. Drift stiffness：高分辨率段长度从300缩到100，使drift rate中的1/ΔT放大3倍，边界误差更难修正。
2. Information bottleneck：8分辨率段扩展到t=400--1000，进入更高SNR区间，可能过早丢弃中高频信息。
3. Capacity mismatch：最小的DiT-B负责600个局部t，且每个t看到的样本更少；最大的DiT-XL只负责100个t，训练预算分配可能失衡。
4. Boundary entropy：t=100的noise-filling方差更小，16→32边界提供的高频随机自由度可能不足。
5. Per-timestep density is not task difficulty：更密集的局部t覆盖不能抵消更强的drift、更短的修正路径和上游不可逆信息损失。

#### 3. 实验设计

先完成已有原生ImageNet checkpoint的相同设置评估：

| 模型 | Checkpoint | Sampler | Sampling split [32,16,8] | CFG | Boundary |
|---|---:|---|---:|---:|---|
| MS-300 | 400K | DDPM | 90/80/80 | 1.0, 1.5 | noise filling |
| MS-100 | 400K | DDPM | 90/80/80 | 1.0, 1.5 | noise filling |

然后用Unified L进行低成本training pilot：

- 100/300/600
- 200/300/500
- 300/300/400

每个训练到100K和200K，在matched inference boundaries下评估。

#### 4. Stage-wise diagnostics

对MS-100与MS-300记录：

- held-out denoising MSE vs global timestep；
- 每个stage结束后的pred-x0误差；
- 8→16和16→32边界前后的low-pass/high-pass重建误差；
- 每个stage的gradient norm和loss scale；
- 按局部t统计有效样本数与误差；
- 10K stage-wise FID或特征距离，定位质量首先在哪个stage分叉。

#### 5. 判据

- 若差异在8分辨率stage已经出现，支持coarse information bottleneck / capacity mismatch。
- 若8和16阶段相近、进入32阶段后才分叉，支持short fine-stage / drift stiffness解释。
- 若改变sampling allocation无法弥补差异，说明问题主要来自训练分段而不是离散化。
- 只有原生ImageNet结果复现后，才能把这一规律写入论文定量结论。

#### 6. 对应脚本

- exp10_stage_boundary_train.sh：改为真正从头训练不同split。
- exp10_stagewise_diagnostics.py：新增逐t和逐stage诊断。
- 现有inference-only mismatch脚本保留作敏感性结果，不再扩展更多网格。

## 图表相关实验

---

### 图1: Method Overview Figure

#### 1. Critique原文

- 多轮critique反映current figure不够直观

#### 2. 实验目的

重新设计方法概述图，清晰展示RDD的forward process如何在时间轴上从高分辨率过渡到低分辨率。

#### 3. 实验设计

- 不需要跑模型实验
- 制作示意图（matplotlib或TikZ）
- 关键元素：
  - 时间轴t: 0→T
  - 三个stage的分辨率标注（256→128→64）
  - Drift term的可视化（作为"信息桥梁"连接不同分辨率）
  - 对比：标准diffusion vs RDD的forward trajectory

#### 4. 预期结果

一张清晰的方法示意图，能让reviewer一眼理解RDD的核心idea。

#### 5. 如果未达预期

反复迭代直到图表足够清晰。可参考Progressive GAN、Imagen等论文的方法图设计。

#### 6. 论文写法

- **位置**: Figure 1 (paper首图)

#### 7. 对应脚本名

`experiments_scripts/fig01_method_overview.py`

---

### 图2: Convergence Curve

#### 1. Critique原文

依赖实验4的输出。

#### 2. 实验目的

将实验4的数据可视化为publication-quality figure。

#### 3. 实验设计

- 读取实验4产出的FID数据
- 使用matplotlib绘图
- 配色：配合论文整体风格
- 双X轴：steps + FLOPs

#### 4. 预期结果

清晰的收敛曲线对比图。

#### 5. 如果未达预期

调整可视化参数。

#### 6. 论文写法

- **位置**: Figure 2

#### 7. 对应脚本名

`experiments_scripts/fig02_convergence_curve.py`

---

### 图3: Effective Noise Visualization

#### 1. Critique原文

帮助理解drift-to-noise ratio随时间的变化。

#### 2. 实验目的

展示不同timestep下effective noise的drift-to-noise ratio，帮助理解RDD在不同阶段的行为。

#### 3. 实验设计

- 不需要模型推理
- 仅使用noise schedule参数计算
- 画drift magnitude vs noise magnitude随t变化的曲线
- 以及DNR (Drift-to-Noise Ratio) 随t变化

#### 4. 预期结果

- Stage transition点有明显的DNR跳变
- 低t（高分辨率stage）drift占主导
- 高t（低分辨率stage）noise占主导

#### 5. 如果未达预期

检查noise schedule参数设置是否正确。

#### 6. 论文写法

- **位置**: Figure 3 (§3.x 理论部分)

#### 7. 对应脚本名

`experiments_scripts/fig03_effective_noise_dnr.py`

---

### 图4: PSD Analysis

#### 1. Critique原文

依赖实验9的输出。

#### 2. 实验目的

将实验9的PSD分析结果可视化。

#### 3. 实验设计

与实验9合并执行。

#### 4. 预期结果

清晰的PSD对比图。

#### 5. 如果未达预期

调整可视化。

#### 6. 论文写法

- **位置**: Figure in §3.3 or Appendix

#### 7. 对应脚本名

`experiments_scripts/fig04_psd_analysis.py`（与exp09合并）

---

## 当前依赖关系与执行优先级

### 已完成

- 实验4：native-ImageNet convergence curve。
- Unified XL主CFG、allocation、noise-filling和solver scheduled-step结果。
- 250-step采样microbenchmark。

### 立即补齐

1. Unified L CFG1.5 evaluator。
2. MS-100 native-ImageNet CFG1.0/1.5。
3. DiT500K CFG1.5。
4. 已有ablation_cfg1样本的缺失评估。

### 核心链路

1. 实验12 matched-NFE Pareto。
2. Boundary × sampler 2×2。
3. Checkpoint × sampler训练成熟度测试。
4. DPM2 midpoint drift消融。
5. 实验8正式wall-clock。
6. 实验13 CFG与stage-wise CFG。

### 训练型实验

1. 实验1 drift-free causal control。
2. 实验10 native-ImageNet training split pilot。
3. 实验3 multi-seed。

### 非阻塞加分项

- 实验6窄范围drift multiplier。
- 实验7理论值附近sigma sweep。
- 实验9 PSD/frequency analysis。
- 实验14 downsampling operator，仅在核心链路完成后。
- ImageNet-512与Flow Matching范围扩展。

## 资源控制原则

- 所有新采样先10K筛选，只有主结论点升级到50K。
- 已有完整样本优先只重跑evaluator。
- 不再为旧HF checkpoint、CFG大于2、更多inference mismatch、7M extended training或当前论文的distillation组合分配资源。
- 任何solver表都同时记录scheduled steps与真实NFE。

## 补充实验（Critique 遗漏项）

---

### 实验12: Matched-NFE DDPM / DPM1 / DPM2 Pareto

#### 1. 已知事实

当前DPM2中，三阶段scheduled total为S时，真实NFE为2S−6：

| DPM2 scheduled | 真实NFE | FID |
|---:|---:|---:|
| 250 | 494 | 2.617 |
| 125 | 244 | 2.626 |
| 65 | 124 | 3.325 |
| 35 | 64 | 11.761 |

因此旧表把250/125/65/35直接写作NFE是错误的。
最有价值的现有公平比较是DPM2 244 NFE的FID2.626，对DDPM 250 NFE的3.537。

#### 2. 实验目的

- 构建FID vs真实模型调用数，而不是FID vs scheduled steps。
- 验证DPM2优势是否跨CFG、checkpoint和boundary稳定。
- 定位64--124 NFE之间的稳定性拐点。

#### 3. 实验设计

固定Unified XL 400K、native ImageNet、noise filling。

| 真实NFE目标 | DDPM | DPM1 | DPM2 |
|---:|---|---|---|
| 64 | ✓ | ✓ | ✓ |
| 80 | 可选 | 可选 | ✓ |
| 96 | ✓ | ✓ | ✓ |
| 112 | 可选 | 可选 | ✓ |
| 124 | ✓ | ✓ | ✓ |
| 160 | ✓ | ✓ | ✓ |
| 244/250 | ✓ | ✓ | ✓ |

CFG分别为1.0和1.5。
先10K样本筛选；64、96、124、244/250以及所有发生sampler排序翻转的点升级到50K。

#### 4. 记录要求

每个结果必须写：

- 各stage scheduled steps；
- 各stage实际model forwards；
- 总NFE；
- sampler order；
- CFG；
- boundary；
- wall-clock和peak memory；
- FID、sFID、IS、Precision、Recall。

#### 5. 论文输出

主图横轴使用真实NFE。
DPM2 scheduled250 / 494 NFE可作为best-quality点，但不能与DDPM250称为等计算比较。
约250 NFE的主结论使用DPM2 scheduled125 / 244 NFE对DDPM250 / 250 NFE。

### 实验13: Fine-Grained and Stage-Wise CFG

#### 1. 已有结果

Unified XL、DDPM250、noise filling：

| CFG | FID | Precision | Recall |
|---:|---:|---:|---:|
| 1.0 | 7.061 | 0.717 | 0.631 |
| 1.5 | 3.537 | 0.868 | 0.507 |
| 2.0 | 8.335 | 0.921 | 0.394 |
| 2.5 | 13.184 | 0.942 | 0.303 |
| 4.0 | 20.594 | 0.947 | 0.162 |

CFG高于1.5已经明确恶化FID和Recall，不再进行2--4之间的密集搜索。

#### 2. 实验目的

- 定位1.1--1.6之间的真实FID最优点。
- 检验分阶段CFG能否保留低分辨率多样性，只在最终32分辨率阶段增强条件一致性。
- 与正确ImageNet DiT baseline做相同CFG协议比较。

#### 3. 实验设计

全阶段统一CFG：

- 1.0、1.1、1.2、1.3、1.4、1.5、1.6。

Stage-wise [CFG32,CFG16,CFG8]：

- [1.5,1.0,1.0]
- [1.5,1.25,1.0]
- [1.5,1.25,1.25]
- [1.5,1.5,1.5]

Sampler分别使用DDPM250和最终选定的DPM2约250真实NFE。
先10K筛选，最佳统一CFG、最佳stage-wise CFG和baseline升级到50K。

#### 4. 报告指标

必须同时报告FID、sFID、IS、Precision和Recall。
目标不是证明RDD在每个CFG都占优，而是给出透明的quality-diversity Pareto，并使用各方法的最佳合法 operating point。

#### 5. 论文输出

- FID / Recall vs CFG曲线；
- stage-wise CFG小表；
- 主表保留CFG1.0公平比较，并另报各模型最佳operating point。

### 实验14: Downsampling Operator Ablation (Inference-only)

#### 1. Critique原文

- **Round 1**: 质疑 avg_pool 作为 downsampling operator 的选择
- **Round 30**: PSD分析中 downsampling 方式影响 drift 的频谱特性

核心质疑：**"Average pooling是否是最佳downsampling选择？其他算子是否能更好地捕获高频？"**

#### 2. 实验目的

验证downsampling算子选择对最终生成质量的影响，证明avg_pool的合理性或发现更好的选择。

#### 3. 实验设计

| 配置项 | 设置 |
|--------|------|
| 模型 | RDD Unified XL (已训练好，avg_pool训练) |
| 类型 | **Inference-only 实验** |
| 算子替换 | 仅在drift correction的采样路径中替换算子 |
| 测试算子 | avg_pool (baseline), Gaussian blur (σ=1.0), bilinear downsample |
| 每组 | 50K样本FID |
| CFG | cfg=1.5 |

注意：模型是用avg_pool训练的，inference时替换会引入train-test mismatch。因此这个实验主要验证avg_pool的鲁棒性，而非寻找"最优"算子。

#### 4. 预期结果

| Downsample Operator | 预期FID | 说明 |
|---------------------|---------|------|
| avg_pool (train=test) | 2.62 | Baseline (匹配) |
| Gaussian blur σ=1.0 | ~2.70 | 轻微mismatch |
| bilinear | ~2.65 | 接近avg_pool |

预期：avg_pool接近最优，替换造成轻微退化，证明选择合理但不关键。

#### 5. 如果未达预期

- **情况A: 某个算子显著更好（FID差>0.3）**
  - 不太可能（因为train-test mismatch）
  - 如果确实更好：说明应该用该算子重新训练

- **情况B: 所有算子结果几乎一样（ΔFID < 0.05）**
  - 好消息：drift correction对具体downsampling算子不敏感
  - 论文声称鲁棒性

#### 6. 论文写法

- **位置**: Appendix (鲁棒性分析)
- **内容**: 小表格，3行
- **叙事**: "Replacing the avg_pool operator at inference with alternative downsampling operations (Gaussian blur, bilinear) produces FID within ±0.08, confirming the method's robustness to operator choice."

#### 7. 对应脚本名

`experiments_scripts/exp14_downsample_ablation.sh`

---

## 评估指标升级说明

**所有FID评估实验（exp01-07, exp10-14）必须同时报告以下指标：**

| 指标 | 工具 | 说明 |
|------|------|------|
| FID-50K | ADM evaluator | 标准 |
| sFID | ADM evaluator | 空间频率敏感FID |
| IS (Inception Score) | ADM evaluator | 多样性+质量 |
| Precision | ADM evaluator | 保真度 |
| Recall | ADM evaluator | 多样性 |
| Clean-FID | `clean-fid` package | 消除resize artifact |

实现方式：修改评估pipeline，在现有 `evaluator.py` 输出基础上额外运行 `clean-fid`。

**在 exp03 (multi-seed) 中**：所有6个指标都报告mean±std。

---

## 更新后的计算资源预估

| 实验 | GPU需求 | 预估时间 | 类型 |
|------|---------|---------|------|
| 实验1 | 8×A100 | ~4天 | 训练 |
| 实验2 | 8×A100 | ~1-3天 | 继续训练 |
| 实验3 | 1×A100 | ~2天 | 仅采样评估 |
| 实验4 | 1×A100 | ~3天 | 仅采样评估 |
| 实验5 | 1×A100 | ~1天 | 仅采样评估 |
| 实验6 | 1×A100 | ~1天 | 仅采样评估 |
| 实验7 | 1×A100 | ~1天 | 仅采样评估 |
| 实验8 | 1×A100 | ~2小时 | Benchmark |
| 实验9 | 1×GPU | ~1小时 | 数据分析 |
| 实验10 | 8×A100 | ~8天 | 训练 |
| 实验11 | 8×A100 | ~3-6天 | 继续训练 |
| **实验12** | 1×A100 | ~2天 | 仅采样评估 |
| **实验13** | 1×A100 | ~2.5天 | 仅采样评估 |
| **实验14** | 1×A100 | ~0.5天 | 仅采样评估 |
| **实验15a** | 1×A100 | ~1天 | 仅采样评估 |
| **实验15b** | 8×A100 | ~2-3天 | 蒸馏训练 |
| **总计** | | **~28-36 A100-days** | |

---

## 论文中推理速度声明的重要修正

当前可使用的250-step单H20 microbenchmark：

| 模型 | ms/image | 相对DiT |
|---|---:|---:|
| DiT-XL/2 | 1333.4 | 1.00× |
| Unified XL | 703.4 | 1.90× |
| MS-300 | 596.6 | 2.24× |
| MS-100 | 598.5 | 2.23× |
| Unified L | 520.9 | 2.56× |

限制：

- 当前脚本是单次class-conditional forward，不包含CFG双前向；
- RDD边界是bilinear加固定噪声，不是生产版noise filling；
- 这是模型计算microbenchmark，不是完整生成pipeline；
- 1000-step的MS-100 FLOPs元数据与实际stage allocation冲突，暂不进入论文主表；
- DPM2必须按真实NFE单独测量。

论文当前只能写“在该microbenchmark协议下，Unified XL为1.90×、Unified L为2.56×”。
在完成真实checkpoint、production noise filling、相同CFG和matched-NFE的正式测试前，不写matched-quality speedup，也不写6.5×主结论。

---

## Experiment 21: Controlled RDD-Wan Fine-tuning (exp21)

### 1. Question

在相同Wan 2.1-1.3B初始化、离线video--caption数据、stage shapes、stage boundaries、optimizer、global batch和update数下，RDD的stage-interior residual path是否比普通pyramidal path带来可测收益？不能预设“快2倍”。

### 2. Minimum causal matrix

| Method | Purpose |
|---|---|
| Vanilla Wan | 原始checkpoint和完整分辨率参考 |
| PyramidalWan reproduction | 最接近的公平baseline |
| Staged, drift-free | 隔离低分辨率token节省本身 |
| RDD-Wan | residual interior path + analytic clean inverse |

- 数据：所有训练方法读取完全相同的离线缓存样本和T5 context；teacher生成成本单独报告。
- Checkpoint：0、1K、2K、5K；只有趋势仍未饱和时才加10K。
- 主指标：VBench quality/semantic以及关键子维度；每个prompt至少3个生成seed。
- 训练指标：held-out conditional FM loss、每stage loss/gradient norm、真实tokens/update、wall-clock和peak memory。不同path的raw MSE只作诊断，不能单独证明生成质量更好。
- 数据模式：当前RDD-FM以latent average-pool/repeat幂等投影为主，因为解析clean inverse在这里严格成立。Pixel downsample + VAE re-encode是非线性变换，只用于PyramidalWan baseline的数据准备；在增加x-prediction head或transform-specific inverse前，不作为RDD的即插即用消融。没有`text_emb`的数据必须报错，不能静默用全零context。

### 3. Decision rule

- RDD若在相同update和compute下更快达到同一VBench，才写“video fine-tuning convergence improved”。
- 若RDD只改善低NFE质量，则叙事改为更好的path/solver interaction。
- 若RDD和drift-free接近，则视频只证明通用性，不作为headline novelty。
- 若PyramidalWan更好，保留负结果于附录并停止扩大视频训练规模。

### 4. Scripts

- `rdd_video/setup_wan21.sh`
- `rdd_video/prepare_rdd_wan_data.sh`
- `rdd_video/train_rdd_wan.sh`
- `rdd_video/run_rdd_wan_smoke.sh`

---

## Experiment 22: Matched-NFE Video Solver and Latency Curve (exp22)

### 1. Design

对PyramidalWan和RDD-Wan使用同一组Euler、AB2和Heun stage-local solver。每次noise-resampling boundary都重置multistep history。横轴同时给出scheduled updates、真实network calls、TFLOPs和wall-clock；Heun每update为2 NFE，endpoint clean reconstruction的额外前向也必须计入。

推荐最小anchor：

- Euler 20/20/10；
- AB2 20/20/10；
- Euler 10/10/5；
- AB2 10/10/5；
- Heun 10/10/5（与50个update的一阶方法约为matched NFE）。

官方Wan Flow-DPM++只作为vanilla baseline。若用于RDD，必须把标准`x_0=x_sigma-sigma*w`替换为RDD projection clean inverse，不能直接套scheduler后宣称正确。

### 2. Evaluation

主结果为VBench score--NFE与score--latency曲线；同时报告subject/background consistency、temporal flicker、motion smoothness、aesthetic quality和text alignment。FVD可作为补充，但不作为唯一或首要video metric，因为它对实现和特征网络敏感。

### 3. Script

`rdd_video/sample_solver_sweep.sh`。相同的config矩阵必须在PyramidalWan官方/复现代码上运行；在该baseline接通前不得做“solver兼容性独占”结论。

### 4. Distillation gate

Teacher生成clean videos再做RDD-FM是synthetic-data fine-tuning，不是step distillation。只有matched-NFE的RDD-Wan先显示稳定收益后，才选择一种相同的DMD/progressive-distillation recipe同时蒸馏RDD-Wan和PyramidalWan。蒸馏不进入当前最小投稿包。

---

## Experiment 23: ImageNet 512×512 Scaling Verification (exp23)

### 1. Objective
Verify that RDD's training acceleration advantage generalizes from 256×256 to 512×512, addressing reviewer concern about single-resolution validation.

### 2. Targeted Critique
"All image experiments are on 256×256. The method might not scale to higher resolutions."

### 3. Design
- **Model**: DiT-XL/2 (same architecture, operates on 64×64 latent for 512×512)
- **Stage config**: 4 stages — ÷8 → ÷4 → ÷2 → ÷1 (additional stage needed for larger resolution)
- **Baseline**: Standard DiT-XL/2 on ImageNet 512×512
- **Training budget**: Match iso-FLOP — train both for equivalent A100-hours
- **Metrics**: FID-50K, sFID, IS, Precision, Recall at 100K/200K/400K steps

### 4. Expected Outcome
RDD achieves comparable FID to baseline at significantly fewer steps (>2× speedup), consistent with 256×256 findings. The larger resolution should benefit MORE from the multi-resolution curriculum since there are more frequency bands to exploit.

### 5. Failure Contingency
If advantage diminishes at 512: report honestly and discuss in limitations. The 256×256 results + video results still provide sufficient evidence for the paper.

### 6. Hardware
- 8×A100 80GB
- Estimated time: ~200 A100-hours (100K steps at higher resolution)

### 7. Script
`experiments_scripts/exp23_imagenet512_scaling.sh`

---

## Experiment 25: Unified-L Complete Evaluation (exp25)

### 1. Objective

Complete the native-ImageNet quality-efficiency point for the already trained Unified L model before expanding to additional B/4 or L/4 variants.

### 2. Current Evidence

- Unified L: 457.8M parameters.
- DDPM250, CFG1.0 FID8.461.
- Training microbenchmark: 0.899× DiT time per step.
- 250-step sampling microbenchmark: 2.56× faster than DiT.
- CFG1.5 evaluation failed only because of the evaluator's NumPy np.long incompatibility.

### 3. Design

1. Repair the evaluator and reuse the existing CFG1.5 sample set.
2. Report FID/sFID/IS/Precision/Recall.
3. Run the final matched-NFE DPM2 setting only if Unified L's DDPM CFG1.5 result lies on the quality-efficiency Pareto.
4. Do not add DiT-B/4 experiments until the L result is complete and a correct native-ImageNet B checkpoint is confirmed.

### 4. Paper Output

A quality–training-time–sampling-latency table comparing DiT-XL/2, Unified XL, and Unified L.

---

## Priority Classification (Scope Defense Experiments)

| Exp | Priority | Compute | Addresses |
|-----|----------|---------|-----------|
| exp25 | P0 (evaluator fix) | Low | Unified-L efficiency point |
| exp23 | P3 (after core ablations) | High | Resolution scaling |

