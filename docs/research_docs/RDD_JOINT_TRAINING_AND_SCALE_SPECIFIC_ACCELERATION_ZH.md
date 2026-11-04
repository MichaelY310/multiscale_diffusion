# RDD 联合训练与跨尺度加速方案

> **2026-09-15 路线调整：暂停 adapter/reader 续训，转向不加 adapter 的直接跨尺度残差复用＋共享模型整体微调。** 新实现、配置和结果独立记录在[新实验页面](RDD_DIRECT_REUSE_FULLMODEL_20260915_ZH.html)。下方仍保留旧路线的历史结果，不代表它正在继续训练。

## 1. 当前判断

### 2026-09-15 更新：视频计算预算通过，但续训没有解决2D质量问题

**目前仍不是“只差视频训练”。** 新双reader在1280视频上的完整去噪计算有约13%节省空间，但2D继续训练到1600后并未改善分类表现；此外本轮严格FP32 VAE使去噪+解码只快约1.5%。下面分别记录，不把两个不同验证拼成一个成功结论。

#### 同预算800→1600：loss更低，但生成分类结果没有跟着变好

三组均从各自固定800模型恢复优化器和随机状态，再训练**固定800更新**，最终1600，不挑best；原800checkpoint和训练协议未改。统一4×64＝256、无累计、相同数据和学习率。原8/16冻结，reader及执行的fine主干共同训练。续训代码单独保存：[实现](../../2d/attention_analysis/continue_csra_layout_to1600.py) · [后台脚本](../../2d/exp/run_csra_layout_continuation1600_20260915.sh)。

继续使用前轮同256个开发样本、DDPM80/80/90、CFG1、noise filling。新生成1024张PNG全部下载并核验；256张原RDD PNG与800阶段逐文件哈希一致，排除了样本改变。

| 方法 | 更新 | Top-1 | Top-5 | 目标类别概率 | latent L2偏差 | teacher状态验证epsilon MSE |
|---|---:|---:|---:|---:|---:|---:|
| 原RDD | — | 69.14% | 88.28% | 0.27676 | 0 | — |
| 连续跳6层 | 800 | 60.55% | 81.25% | 0.24005 | 0.26867 | 0.012157 |
| 连续跳6层 | 1600 | 62.11% | 79.30% | 0.24660 | 0.26069 | 0.009804 |
| split3+3正确coarse | 800 | 64.84% | 79.69% | 0.25398 | 0.24168 | 0.006769 |
| **split3+3正确coarse** | **1600** | **60.16%** | **80.47%** | **0.24444** | **0.23345** | **0.005977** |
| split3+3零coarse | 800 | 55.47% | 76.56% | 0.19878 | 0.26747 | 0.008547 |
| split3+3零coarse | 1600 | 57.42% | 78.13% | 0.20922 | 0.25619 | 0.007241 |

split正确coarse的Top-1变化为**−4.69个百分点，配对95%区间[−8.59,−1.17]**。Top-5略升但区间包含0；目标类别概率下降的区间也包含0。因此不是所有指标都一致恶化，但**不能把loss和latent偏差下降当作已改善画质**。这里仍是分类筛选，不是FID；同一开发集、单训练seed、未做多重比较校正。

1600时正确coarse相对零coarse的目标类别概率仍高0.03522，区间[0.02064,0.05071]；但Top-1差仅+2.73个百分点，区间[−2.34,+7.81]。coarse内容的价值没有消失，但不能一直引用800时的+9.38个百分点，假装后续训练仍保持同样幅度。

四张固定三列图均已检查：小鸟、狗、蝴蝶、食物等主体仍可辨认；车辆几何、键盘结构、人物脸部与瓶面内容的变化仍在。1600并未肉眼统一恢复这些问题，也不是每张图都更差。归档SHA256：`e1d36b0da2b717ea8afd1ce9edc86aa63d82460c7ac57b766f199ccfb1be1f6b`。[完整对照及哈希审计](../../eval_results/csra_skip_layout_20260915/split1600_quality256/paired_continuation_summary.json)。


<details><summary>1600更新固定三列图1：原RDD／split正确coarse／split零coarse</summary><img src="../../eval_results/csra_skip_layout_20260915/split1600_quality256/shard_0/source_control_comparison.png" style="max-width:100%;width:950px" alt="1600更新固定分片0"></details>

<details><summary>1600更新固定三列图2：原RDD／split正确coarse／split零coarse</summary><img src="../../eval_results/csra_skip_layout_20260915/split1600_quality256/shard_1/source_control_comparison.png" style="max-width:100%;width:950px" alt="1600更新固定分片1"></details>

<details><summary>1600更新固定三列图3：原RDD／split正确coarse／split零coarse</summary><img src="../../eval_results/csra_skip_layout_20260915/split1600_quality256/shard_2/source_control_comparison.png" style="max-width:100%;width:950px" alt="1600更新固定分片2"></details>

<details><summary>1600更新固定三列图4：原RDD／split正确coarse／split零coarse</summary><img src="../../eval_results/csra_skip_layout_20260915/split1600_quality256/shard_3/source_control_comparison.png" style="max-width:100%;width:950px" alt="1600更新固定分片3"></details>

#### 已完成：双reader的视频全成本计时

本轮四个独立GPU worker仅用0–3卡；4–7留给其他人。设备为NVIDIA RTX PRO 6000 Blackwell Server Edition。FastWan视频主干H1536、30层，候选fine执行24层；2D的L/2则是24层中执行18层，并不是同一个已训练模型直接加载到video。每格预热2轮、8组ABBA/BAAB，正式总计256条三次NFE轨迹。所有执行block保留**native VSA20、coarse output和gate**；两份模型、两个reader、VAE在所有arm都驻留。计入两个reader、coarse缓存、noise filling边界、同步，每次真实VAE decode。原生模型、副本和不启用reader的封装逐元素一致；核验跳层数量、两个reader调用数和完整视频文件均通过。

| 输出／stage | reader配置 | 同RDD基线→候选去噪中位数 | 配对去噪延迟下降 [95%区间] | 基线→候选去噪+VAE中位数 | 配对总延迟下降 |
|---|---|---:|---:|---:|---:|
| 832×448 Spatial421 | 连续跳6层，1个reader | 0.4147→0.3641 s | 12.17% [11.35,13.05] | 3.9509→3.9017 s | 1.26% |
| 同上 | split3+3，2个reader | 0.4138→0.3714 s | 9.97% [9.66,10.33] | 3.9534→3.9121 s | 1.01% |
| 832×448 ST421 | 连续跳6层，1个reader | 0.4052→0.3551 s | 12.50% [11.82,13.45] | 3.9491→3.9001 s | 1.27% |
| 同上 | split3+3，2个reader | 0.4062→0.3621 s | 10.62% [9.74,11.44] | 3.9587→3.9160 s | 1.08% |
| 1280×768 Spatial421 | 连续跳6层，1个reader | 1.2147→1.0449 s | 14.10% [13.67,14.52] | 10.5415→10.3654 s | 1.70% |
| **同上** | **split3+3，2个reader** | **1.2157→1.0635 s** | **12.89% [12.69,13.09]** | **10.5438→10.3869 s** | **1.55%** |
| 1280×768 ST421 | 连续跳6层，1个reader | 1.1181→0.9446 s | 15.68% [15.34,16.00] | 10.3683→10.1970 s | 1.66% |
| **同上** | **split3+3，2个reader** | **1.1182→0.9642 s** | **13.41% [13.13,13.80]** | **10.3704→10.2208 s** | **1.52%** |

百分比是交错配对组的平均延迟下降，不等于两列中位数相除；区间对组重采样，不把同组调用当独立重复。单reader与双reader分别配对，不能用两个候选时间的极小差值宣称一个更优kernel。双reader确实增加成本：1280 Spatial的reader合计21.68→43.27ms，ST19.29→38.45ms。新方案有计算预算，但没有达到“端到端含VAE快10%”的目标。

**为什么总收益这么小：** 本轮VAE是FP32、TF32关闭，832解码约3.54s、1280约9.25–9.33s，占总时间近89%。reader只省去噪，不能省同分辨率的最终解码。这个精度口径两边一致，比较是配对公平的，但**不是已优化的生产视频管线**；不能与旧报告另一种VAE口径的1.x秒混在一起，也不能把通用VAE精度优化收益算作RDD贡献。后续若改VAE精度/实现，两边必须同时改并检查输出差异。

全部29帧、16fps（约1.81秒）；FastWan DMD 3 NFE，t=1000/757/522，seed20660913，prompt为雪林中的红狐狸。Spatial421为`(4,1)→(2,1)→(1,1)`，ST421为`(4,4)→(2,2)→(1,1)`，factor第一项空间、第二项时间；最终latent分别8×56×104及8×96×160。No offload，不含T5、加载、预热编译和MP4编码。

**视频质量警告：这些video reader是随机权重，从未训练；FastWan本身也未训练RDD drift。** 1280两组基线尚可见狐狸但有伪影，随机reader输出基本是颗粒/色块。它们只验证实际计算成本，绝不能与上面的已训练2D结果混称为“训练后视频质量不错”。


<details><summary>成本审核视频：832×448，Spatial421；三个视频均29帧、16fps</summary>

<p>左：同RDD+VSA基线；中：随机单reader；右：随机双reader。后两者没有视频训练，不能用于判断方法训练后的质量。</p>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/832_spatial/baseline_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/832_spatial/prefix6_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/832_spatial/split3_3_UNTRAINED_COST_ONLY.mp4"></video>

</details>

<details><summary>成本审核视频：832×448，ST421；三个视频均29帧、16fps</summary>

<p>左：同RDD+VSA基线；中：随机单reader；右：随机双reader。后两者没有视频训练，不能用于判断方法训练后的质量。</p>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/832_st/baseline_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/832_st/prefix6_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/832_st/split3_3_UNTRAINED_COST_ONLY.mp4"></video>

</details>

<details><summary>成本审核视频：1280×768，Spatial421；三个视频均29帧、16fps</summary>

<p>左：同RDD+VSA基线；中：随机单reader；右：随机双reader。后两者没有视频训练，不能用于判断方法训练后的质量。</p>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/1280_spatial/baseline_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/1280_spatial/prefix6_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/1280_spatial/split3_3_UNTRAINED_COST_ONLY.mp4"></video>

</details>

<details><summary>成本审核视频：1280×768，ST421；三个视频均29帧、16fps</summary>

<p>左：同RDD+VSA基线；中：随机单reader；右：随机双reader。后两者没有视频训练，不能用于判断方法训练后的质量。</p>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/1280_st/baseline_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/1280_st/prefix6_UNTRAINED_COST_ONLY.mp4"></video>
<video controls preload="none" style="width:32%;min-width:220px" src="../../eval_results/csra_skip_layout_20260915/split_video_cost/1280_st/split3_3_UNTRAINED_COST_ONLY.mp4"></video>

</details>

[全部计时与审计](../../eval_results/csra_skip_layout_20260915/split_video_cost/summary_and_audit.json)。完整12个视频已下载；归档SHA256：`ef458daa9d754fecabdce984a77d46d0078eccad93f647eece03370809858bd6`。

#### 新诊断：空间对齐很重要，但“学生走偏后单步误差爆炸”未被支持

4种模型各16个相同新样本，不训练；每个fine调用都检查。比较学生在原RDD轨迹上的输出误差，以及在自己轨迹上相对原RDD对同一输入的输出误差。额外诊断调用与无诊断的完整采样逐元素一致，不影响随机数/轨迹。

| 模型 | 原RDD轨迹epsilon MSE | 自己轨迹epsilon MSE | 自己／原轨迹 |
|---|---:|---:|---:|
| split800 | 0.007710 | 0.006980 | 0.91 |
| split1600 | 0.006771 | 0.006249 | 0.92 |
| 连续跳6层800 | 0.013944 | 0.012441 | 0.89 |
| 连续跳6层1600 | 0.011228 | 0.010218 | 0.91 |

这里teacher也不是真实epsilon，指标只是与原RDD的一致程度。**没有看到换到学生轨迹后单步预测误差增大；因此不能继续把这一种“误差爆炸”解释当事实。** 小误差仍可能改变整条随机轨迹，最终分类也不必随平均MSE改善，但具体因果尚未确定。

在10个预定fine调用上，暂时改变coarse输入，其他设置不变；下面是自己轨迹上的epsilon MSE倍数：

| 修改 | split800 | split1600 |
|---|---:|---:|
| 原样正确coarse | 1.00× | 1.00× |
| 换成同batch另一张图的coarse | 1.56× | 1.60× |
| 所有coarse位置变成全局平均 | 3.05× | 3.39× |
| coarse网格空间循环平移1格 | 4.73× | 5.26× |
| 将attention读出分支置零，保留对齐affine与fine MLP | 1.026× | 1.052× |

这支持**coarse的空间对应关系是当前reader真正使用的信息**，不是只增加了一个无关小网络。注意这些干预也会引入分布变化，不能证明空间对齐是唯一机制；读出分支单步影响较小，也不等于可以安全删除。

还检查了2×2平均与剩余细节的误差分解。约75%误差落在细节子空间，**但细节本来占3/4自由度，不能据此宣布错误主要来自高频**；按自由度归一后，整体并没有显著单边集中。这里是latent网格的局部分解，不直接等于人类感知的语义/纹理。

[诊断完整曲线与原始指标](../../eval_results/csra_skip_layout_20260915/split_rollout_errors/summary.json)。下一轮是小规模实际出图：在800和1600父模型上，将attention读出乘1、0.5、0，保持coarse对应位置的affine及fine MLP。**只改变输出的门控筛选，暂不宣称节省kernel时间**；通过画质筛选后才实现真正跳过相应计算的版本。当前不再自动延长失败配置的训练。

#### 已完成小规模读出门控：值得尝试简化，但没有质量保证

每个父模型32个相同新样本；共256张PNG已下载并逐文件核验。这里只将两处reader的attention输出乘1、0.5、0，空间对齐的coarse affine和当前fine MLP保持不变。**门控为0时原attention仍在计算，所以本表完全不是加速测试。**

| 父模型 | 原RDD Top-1 | 完整读出 Top-1 / Top-5 | 半读出 Top-1 / Top-5 | 零读出 Top-1 / Top-5 |
|---|---:|---:|---:|---:|
| split800 | 78.13% | 68.75% / 81.25% | 71.88% / 81.25% | 68.75% / 78.13% |
| split1600 | 78.13% | 71.88% / 87.50% | 75.00% / 84.38% | 71.88% / 87.50% |

每32张的一个样本就是3.125个百分点；不能把半读出的“高一个样本”当确定改善。完整/零读出Top-1相同也**不等于质量等价**。固定图中橙子、玩具熊、狗大体接近，车辆与人物脸部问题仍在，门控没有明显救回。两种训练更新的表现也随小样本集变化，不能用这32张推翻前面256张的负结果。


<details><summary>读出门控固定图1：原RDD／完整读出／半读出／零读出（800更新）</summary><img src="../../eval_results/csra_skip_layout_20260915/attention_gate_pilot/shard_0/comparison.png" style="max-width:100%;width:1150px" alt="读出门控固定分片0"></details>

<details><summary>读出门控固定图2：原RDD／完整读出／半读出／零读出（800更新）</summary><img src="../../eval_results/csra_skip_layout_20260915/attention_gate_pilot/shard_1/comparison.png" style="max-width:100%;width:1150px" alt="读出门控固定分片1"></details>

<details><summary>读出门控固定图3：原RDD／完整读出／半读出／零读出（1600更新）</summary><img src="../../eval_results/csra_skip_layout_20260915/attention_gate_pilot/shard_2/comparison.png" style="max-width:100%;width:1150px" alt="读出门控固定分片2"></details>

<details><summary>读出门控固定图4：原RDD／完整读出／半读出／零读出（1600更新）</summary><img src="../../eval_results/csra_skip_layout_20260915/attention_gate_pilot/shard_3/comparison.png" style="max-width:100%;width:1150px" alt="读出门控固定分片3"></details>

[完整小样本指标及审计](../../eval_results/csra_skip_layout_20260915/attention_gate_pilot/summary_and_audit.json)。当时实现了真正不执行reader QKV/SDPA的[空间对齐pointwise版本](../../2d/attention_analysis/csra_aligned_pointwise_reader.py)，保留同位置coarse映射及fine MLP；CPU单元测试证明它等于原reader的零读出，并断言没有调用SDPA。该256样本对比属于旧路线的历史工作，不再表示正在运行。**2026-09-15已按用户要求暂停adapter/reader续训，转向[不加adapter的直接复用＋整网微调](RDD_DIRECT_REUSE_FULLMODEL_20260915_ZH.html)；新页面记录实际训练、图像和视频结果。**

### 2026-09-15 新结果：拆分跳层的 coarse 信息有用，但还没有达到保质结论

**已完成：四种布局各800更新、最初64图筛选、split同预算零coarse对照，以及全新256图配对验证。** 不是只写了脚本。当前最值得保留的线索是：两个reader之间插入真实fine计算，能保留部分结构；正确coarse内容比同大小的全零输入明显有帮助。然而，新256图没有重现最初64图“Top-1和原RDD相同”的结果，不能说已经追平原RDD。

#### 方法到底是什么

以下用从第1层开始的编号，原RDD Unified-L/2共24层。两种新模型都省掉6层：

```text
连续跳6层：
fine输入 → reader替代1–6层 → 真算7–24层

拆分3+3：
fine输入 → reader A替代1–3层 → 真算4–6层
         → reader B替代7–9层 → 真算10–24层
                 ↑                    ↑
        两个reader都读取同一份原16×16阶段第14层的hidden
```

reader不是直接复制旧attention，也不是复制完整旧输出。它结合**当前fine特征、当前timestep/类别条件、真实coarse hidden**，执行全通道线性映射、fine查询到原生coarse K/V的attention和小MLP。中间真算的3层让模型在第二次替代前处理当前高分辨率信息；这是结构假设，不是已证明的误差修复定理。

原8×8、16×16模型独立冻结。微调reader以及真正执行的18个fine block和fine输入/输出部分。split有**两个独立完整reader**，不能把之前单reader的视频提速直接贴过来。

#### 全新256图：五组完全配对，实际保存1280张PNG

统一DDPM 80/80/90、CFG1、noise filling；所有90次fine调用均使用对应学生，原8/16端点及标签逐元素一致。四卡各64张，没有梯度累计。下面Top-1/Top-5是ResNet50对生成图的目标类别识别率，**不是FID，也不是“有多少图质量好”**。

| 模型／方法 | 新训练更新 | 类别Top-1 | 类别Top-5 | 目标类别平均概率 | 相对原latent L2 |
|---|---:|---:|---:|---:|---:|
| 原RDD Unified-L/2、400K | 0 | 69.14% | 88.28% | 0.27676 | 0 |
| 旧连续跳6层，历史参考 | 1600 | 64.45% | 81.64% | 0.25203 | 0.26061 |
| 新连续跳6层，正确coarse | 800 | 60.55% | 81.25% | 0.24005 | 0.26867 |
| **拆分3+3，正确coarse** | **800** | **64.84%** | **79.69%** | **0.25398** | **0.24168** |
| 拆分3+3，coarse输入全零 | 800 | 55.47% | 76.56% | 0.19878 | 0.26747 |

正确/全零coarse两组的模型大小、fine-only初始化、数据、训练seed、优化器和800更新一致；零coarse分支并未被删掉，仍执行相同形状的计算。它排除了“只是多放了两个reader”的主要容量混杂，但还没有排除coarse只是某种一般样本条件、而非特定空间结构的解释；后续可以加错配coarse控制。

配对bootstrap（256张、4000次，固定一套训练seed）：

- **正确coarse vs 全零coarse：Top-1 +9.38个百分点，95%区间[+4.30,+14.45]；目标类别概率+0.05521，区间[+0.03813,+0.07224]。** 这支持使用真实coarse内容有价值。
- split vs 同800更新连续跳6层：Top-1 +4.30个百分点，但区间[−0.39,+8.98]包含0；不能宣布拆分确定胜出。
- split vs 旧1600更新参考：Top-1只+0.39个百分点，区间[−4.69,+5.47]；训练预算不同且没有显著优势。
- **split vs 原RDD：Top-5 −8.59个百分点，区间[−12.89,−4.30]；目标类别概率−0.02278，区间[−0.04117,−0.00619]。** 仍存在明确不足。Top-1区间含0不等于已证明等价。

这些是开发集探索区间，未做多重比较校正，也未覆盖重训seed方差。不能把分类差异写成FID改善，更不能据此宣布视频训练成功。

#### 看图：哪些保留了，哪些仍然错了

四张固定grid已全部检查，每张是相应分片前8个样本，不挑成功图：

- 第一组：小鸟、杯子、大象、狗的主体大致保留；车辆在两个学生中明显变形，沙发和杯面结构也有变化。
- 第二组：海边人物的脸及帽子显著改变，全零coarse更严重；建筑、机械部件仍有变形。几列原图文字本就不可靠，不能全部归罪于reader。
- 第三组：木板在正确coarse中较接近原图，全零中变化很大；蝴蝶较稳定，但键盘结构和桥下人物仍变。**这不只是少了高频纹理。**
- 第四组：食物、虫子、蜘蛛较接近；人物面部和姿态改变，瓶身文字几列都不可靠。换姿态本身不等于画质下降。


<details><summary>256图验证：固定三列对比 1（原RDD／split正确coarse／split零coarse）</summary><img src="../../eval_results/csra_skip_layout_20260915/split_quality256/shard_0/source_control_comparison.png" style="max-width:100%;width:950px" alt="固定分片0前三列，不筛选成功样本"></details>

<details><summary>256图验证：固定三列对比 2（原RDD／split正确coarse／split零coarse）</summary><img src="../../eval_results/csra_skip_layout_20260915/split_quality256/shard_1/source_control_comparison.png" style="max-width:100%;width:950px" alt="固定分片1前三列，不筛选成功样本"></details>

<details><summary>256图验证：固定三列对比 3（原RDD／split正确coarse／split零coarse）</summary><img src="../../eval_results/csra_skip_layout_20260915/split_quality256/shard_2/source_control_comparison.png" style="max-width:100%;width:950px" alt="固定分片2前三列，不筛选成功样本"></details>

<details><summary>256图验证：固定三列对比 4（原RDD／split正确coarse／split零coarse）</summary><img src="../../eval_results/csra_skip_layout_20260915/split_quality256/shard_3/source_control_comparison.png" style="max-width:100%;width:950px" alt="固定分片3前三列，不筛选成功样本"></details>

原始数据：[完整数值与服务器协议审计](../../eval_results/csra_skip_layout_20260915/split_quality256/summary_and_audit.json) · [本地1280张PNG哈希验证与配对区间](../../eval_results/csra_skip_layout_20260915/split_quality256/paired_summary_local.json)。归档SHA256：`ce72bbd624709aabe335422bbeaad177f18e4cb036670b95381f953383427ef2`。checkpoint留在服务器，未把训练权重伪装成已经训练好的视频模型。

#### 为什么没有采用线性测试最好的“后移跳层”

最初64张筛选（与上面256张不是同一批）：

| 方法 | Top-1 | Top-5 | latent L2 |
|---|---:|---:|---:|
| 原RDD | 68.75% | 85.94% | 0 |
| 旧prefix6、1600更新 | 62.50% | 84.38% | 0.25816 |
| 新跳第1–6层 | 59.38% | 89.06% | 0.26625 |
| 新跳第4–9层 | 60.94% | 82.81% | 0.26374 |
| 新跳第7–12层 | 48.44% | 76.56% | 0.27325 |
| 新拆分3+3 | 68.75% | 90.63% | 0.23302 |

**“跳第7–12层”在线性teacher状态诊断上最容易，实际rollout却最差。** 固定图里蛇/车结构改变明显；split也有人脸、白犬脸丢失等失败例。说明不能靠单步MSE选最终生成方法。四种布局全部图片保留如下：


<details><summary>最初64图筛选：分片0全部5种候选的固定对比（各图左边都是原RDD）</summary>

<p>previous_prefix6_1600</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_0/previous_prefix6_1600_comparison.png" style="max-width:100%;width:650px" alt="previous_prefix6_1600固定分片0">

<p>front0_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_0/front0_6_comparison.png" style="max-width:100%;width:650px" alt="front0_6固定分片0">

<p>front3_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_0/front3_6_comparison.png" style="max-width:100%;width:650px" alt="front3_6固定分片0">

<p>front6_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_0/front6_6_comparison.png" style="max-width:100%;width:650px" alt="front6_6固定分片0">

<p>split3_3</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_0/split3_3_comparison.png" style="max-width:100%;width:650px" alt="split3_3固定分片0">

</details>

<details><summary>最初64图筛选：分片1全部5种候选的固定对比（各图左边都是原RDD）</summary>

<p>previous_prefix6_1600</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_1/previous_prefix6_1600_comparison.png" style="max-width:100%;width:650px" alt="previous_prefix6_1600固定分片1">

<p>front0_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_1/front0_6_comparison.png" style="max-width:100%;width:650px" alt="front0_6固定分片1">

<p>front3_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_1/front3_6_comparison.png" style="max-width:100%;width:650px" alt="front3_6固定分片1">

<p>front6_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_1/front6_6_comparison.png" style="max-width:100%;width:650px" alt="front6_6固定分片1">

<p>split3_3</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_1/split3_3_comparison.png" style="max-width:100%;width:650px" alt="split3_3固定分片1">

</details>

<details><summary>最初64图筛选：分片2全部5种候选的固定对比（各图左边都是原RDD）</summary>

<p>previous_prefix6_1600</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_2/previous_prefix6_1600_comparison.png" style="max-width:100%;width:650px" alt="previous_prefix6_1600固定分片2">

<p>front0_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_2/front0_6_comparison.png" style="max-width:100%;width:650px" alt="front0_6固定分片2">

<p>front3_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_2/front3_6_comparison.png" style="max-width:100%;width:650px" alt="front3_6固定分片2">

<p>front6_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_2/front6_6_comparison.png" style="max-width:100%;width:650px" alt="front6_6固定分片2">

<p>split3_3</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_2/split3_3_comparison.png" style="max-width:100%;width:650px" alt="split3_3固定分片2">

</details>

<details><summary>最初64图筛选：分片3全部5种候选的固定对比（各图左边都是原RDD）</summary>

<p>previous_prefix6_1600</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_3/previous_prefix6_1600_comparison.png" style="max-width:100%;width:650px" alt="previous_prefix6_1600固定分片3">

<p>front0_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_3/front0_6_comparison.png" style="max-width:100%;width:650px" alt="front0_6固定分片3">

<p>front3_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_3/front3_6_comparison.png" style="max-width:100%;width:650px" alt="front3_6固定分片3">

<p>front6_6</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_3/front6_6_comparison.png" style="max-width:100%;width:650px" alt="front6_6固定分片3">

<p>split3_3</p><img src="../../eval_results/csra_skip_layout_20260915/layout_quality/shard_3/split3_3_comparison.png" style="max-width:100%;width:650px" alt="split3_3固定分片3">

</details>

#### 已完成：测两个完整reader的视频实际成本（实验设计存档，结果见页首）

已增加独立的双reader视频实现与两项scope/reset测试，测试通过；不修改FastVideo本身或旧2D checkpoint相关源码。四个后台worker仅用GPU0–3，测试832×448和1280×768、29帧，各自Spatial421及ST421。每种对比预热2轮、8组交错ABBA/BAAB，基线始终为**同shape、同stage的RDD+native VSA-20**。

两份模型、两个reader和VAE在两边都驻留；所有实际执行block保留VSA coarse output和gate。计入两个reader、缓存、边界、同步，并每次真实decode：分别报告去噪与去噪+VAE，不再把一个外部VAE常数加到去噪时间上。视频reader目前没有训练，输出只能检查接口、有限值和成本；**即使速度通过，也不能宣称视频质量已通过**。

实现：[双段调度](../../rdd_video/attention_analysis/csra_split_video_bridge.py) · [完整视频计时](../../rdd_video/benchmark_csra_split_feasibility.py) · [四卡后台脚本](../../2d/exp/run_csra_split_video_cost_20260915.sh)。运行tmux仍为`csra_layout_training_20260915`；日志在服务器本轮root下`split_video_cost_launcher.log`及`split_video_cost_*.log`。

**当前未宣布任务完成：双reader的视频成本已测完，仍缺更可靠的2D保质证据和真正训练后的video质量；最新结果见页首。**

### 2026-09-15 继续探索：改变跳层位置，而不只是继续训练旧配置

本轮按用户要求继续实验，**没有改动旧checkpoint、旧训练实现或FastVideo最小教学repo**。新增代码都是独立模块。只使用surfi2物理GPU0–3，不停止其他训练。`/scr`只剩约14GiB，因此新结果/模型改存到`/home/yuheng/rdd_experiments/csra_skip_layout_20260915`，所在文件系统当时可用约332GiB。

#### 已完成：在哪些去噪步使用旧的跳6层reader

固定旧front0、prefix6、1600更新的aligned模型，9种配置各64张，共576张新配对开发样本。原8×8和16×16不变；DDPM80/80/90、CFG1、noise filling。**下表的“步”指32×32阶段的90次模型调用，不是训练步数，也不是全局timestep。** 未使用reader的调用运行原始完整RDD；因此这里同时涉及调用位置和原始/微调fine权重的切换，不应把结果归因于某个单独机制。

| 32×32的复用配置 | 使用reader的调用数 | 类别Top-1 | 类别Top-5 | 相对原latent的L2偏差 |
|---|---:|---:|---:|---:|
| 原RDD，全部完整计算 | 0/90 | 65.63% | 84.38% | 0 |
| 90次全部reader | 90/90 | 59.38% | 79.69% | 0.2460 |
| 前75次reader，最后15次原RDD | 75/90 | 59.38% | 79.69% | 0.2400 |
| 前60次reader，最后30次原RDD | 60/90 | 59.38% | 79.69% | 0.2281 |
| 前30次原RDD，后60次reader | 60/90 | 57.81% | 82.81% | 0.1912 |
| 每3次：reader、reader、原RDD | 60/90 | 57.81% | 82.81% | 0.1920 |
| 前后各15次原RDD，中间60次reader | 60/90 | 59.38% | 82.81% | 0.2152 |
| reader和原RDD逐次交替 | 45/90 | 62.50% | 81.25% | 0.1589 |
| 仅前30次reader | 30/90 | 62.50% | 82.81% | 0.1635 |

**暂时没有发现“后期恢复完整计算，就能把损失明显修好”的证据。** 更保守的配置偏差较小，但使用reader更少，也就少省计算。64张只适合筛选：一个样本就相差1.56个百分点，分类正确率不是画质，L2接近也不代表FID更好。本轮不据此宣布赢家或视频提速。特别是FastWan DMD3NFE只有一次fine调用，不能照搬这里“前60/90次”的设置。

4项调度单元测试通过；0号分片的all90新旧实现逐元素一致；所有配置的原8/16端点和标签逐元素一致。完整数字见[调度筛选汇总](../../eval_results/csra_skip_layout_20260915/schedule_summary.json)。所有样本和固定前4张对比图已在服务器保存，不只留成功图片。

#### 已完成：固定跳6层，先检查哪一组层更容易替代

下面使用同一批192张拟合、64张选ridge参数、128张验证数据，每张8个t。所有方案仍执行18个fine block。先以线性映射预测被跳过部分，再运行剩余真实网络，**比较同一个最终epsilon输出的MSE**；不是直接比较不同深度hidden的原始MSE。

| 配置 | 跳过哪些层（下列从第1层数起） | 仅当前fine信息 | 加正确coarse信息 | 加错配图片的coarse信息 |
|---|---|---:|---:|---:|
| `front0_6` | 第1–6层 | 0.105483 | 0.077799 | 0.102377 |
| `front3_6` | 保留前三层，跳第4–9层 | 0.025345 | 0.020131 | 0.025780 |
| `front6_6` | 保留前六层，跳第7–12层 | 0.016954 | 0.013755 | 0.017225 |
| `split3_3` | 跳第1–3层、计算第4–6层、再跳第7–9层 | 0.055679 | 0.045875 | 0.051664 |

这是一个值得检验的线索：**把被跳的6层往后移，保留前面几层真实fine计算，可能比从patch embedding直接跨过去更容易；正确coarse信息仍有额外帮助。** 但这是teacher状态上的线性诊断，不能推出训练后FID会降低多少。仅fine信息的列参数更少；正确/错配coarse列的参数量相同。拆成两段需要两个完整reader，后续必须计入两个reader的时间，不能只数“都跳6层”。所有teacher输出和使用真实目标hidden的拼接路径均逐元素复现基线。

原始记录：[前6层](../../eval_results/csra_skip_layout_20260915/front0_6_metrics.json)、[第4–9层](../../eval_results/csra_skip_layout_20260915/front3_6_metrics.json)、[第7–12层](../../eval_results/csra_skip_layout_20260915/front6_6_metrics.json)、[拆分3+3](../../eval_results/csra_skip_layout_20260915/split3_3_metrics.json)。

#### 已完成：同预算联合训练与真实出图（启动过程存档）

四种布局从同一原始RDD Unified-L/2 400K出发，各固定800更新，训练reader及未跳过的fine网络；原8/16模型独立冻结。统一4卡×64＝global256，无梯度累计；使用activation checkpointing节省中间激活显存，这不是把batch拆小。AdamW：reader学习率1e-4、fine网络1e-5、weight decay0.01、梯度裁剪1；epsilon MSE＋0.05×variance MSE；BF16 autocast、FP32主权重、TF32关闭。

先对单段和双段方案各跑2步真实四卡训练smoke，再顺序跑四个800更新实验，最后自动生成每方案64张配对图。另放原RDD及旧1600更新的prefix6参考；**旧参考训练预算更大，不纳入四种新布局的同预算消融**。不按验证MSE挑best，比较固定800更新checkpoint。原始smoke在NCCL通信路径报错，已恢复此服务器以往训练使用的`NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1`设置，保留失败日志；后续以实际smoke与训练日志为准，不把脚本已写好等同于训练完成。

**启动核验及整轮训练均已完成：** 4项调度测试、2项布局/reader数学与梯度测试通过；单reader和双reader的四卡2步训练、反向传播、验证均完成，无OOM，allocated峰值分别约8.44/8.73GiB。通信失败的旧smoke协议已另存保留，新smoke用同预算重新验证。四个正式800更新及配对出图均完成，之后又完成split零coarse的同800更新和全新256图验证，结果见页首。每个模型每卡始终64张，显存统计不含其他任务。

服务器训练tmux：`csra_layout_training_20260915`。总日志：`/home/yuheng/rdd_experiments/csra_skip_layout_20260915/training_launcher.log`；每方案日志：`train_front0_6.log`、`train_front3_6.log`、`train_front6_6.log`、`train_split3_3.log`。新启动脚本：[完整后台训练队列](../../2d/exp/run_csra_layout_training_20260915.sh)，模型实现：[分段reader](../../2d/attention_analysis/csra_layout_student.py)，训练实现：[固定预算训练](../../2d/attention_analysis/train_csra_layout_pilot.py)。已启用每15分钟的后台跟进：正常运行不反复通知，遇到故障检查本轮日志并修复；得到有意义的结果后更新此报告和下一轮实验。

后续只有真实配对图/独立小样本评估有改善的方案才扩大训练，并重新测同RDD+VSA基线下的完整视频reader成本。**目标仍是质量与10%–20%视频净加速同时成立；当前尚未实现这个完整结论。**

### 独立10K评估已完成：跳6层reader仍有明显质量损失

2026-09-15北京时间完成。本轮没有延长无可靠收益的coarse记忆修正，而是固定较简单reader，使用不参与前面训练、256图预览或旧10K评估的新种子，与原RDD重新配对生成各10,000张。**结果不支持“reader已经基本保质”：FID增加5.85，Precision下降5.83个百分点。** reader在这里指用小网络读取真实coarse hidden，替代fine第1–6层；不是直接复制attention probability。

| 指标 | 原始RDD | 跳6层reader | reader − 原RDD |
|---|---:|---:|---:|
| FID，越低越好 | 11.6741 | 17.5270 | +5.8529 |
| sFID，越低越好 | 17.5614 | 18.8710 | +1.3096 |
| Inception Score，越高越好 | 115.2414 | 93.6692 | −21.5722 |
| Precision，越高越好 | 70.59% | 64.76% | −5.83个百分点 |
| Recall，越高越好 | 70.99% | 70.11% | −0.88个百分点 |

这是同样10K样本数、同reference下的点估计，**不是论文50K指标，也没有重复训练seed或FID置信区间**。不把FID增幅解释成“画质下降了同样百分比”。Precision下降比Recall明显，提示ADM特征空间中生成质量/逼真度一侧的问题更突出，覆盖面指标改变较小；这只是诊断线索，不能证明原因仅是高频纹理不足，或已经排除语义/布局变化。

**对下一步的判断：这套模型暂不能作为“质量已达标、只差视频训练”的最终方案。** 小样本能生成一些相近的图，不足以替代分布评估；新增记忆修正与50%快照回放也没有可靠改善，不能继续把它们的单步MSE小幅降低写成进展。若下一轮尝试减少跳层或缩短复用时段，必须连同完整bridge开销重新做同RDD基线计时，不能拿保守设置的质量配跳6层的速度。若改变训练目标，也需固定结构和计算预算再做独立评估。本轮不自动新增训练或挑一个更好看的checkpoint重报。这个负结果针对当前结构和训练方案，不证明coarse信息无用，也不证明所有跨尺度复用都不可行。

现有视频证据仍只到**计算可行性**：相同RDD+VSA配置下，未训练视频reader在1280×768×29帧的Spatial421/ST421分别约降低13.68%/14.84%去噪延迟，包含reader等开销，不含VAE、共享GPU。它不是本轮2D训练后权重的真实视频结果，不能与上表拼成已经达成的“保质加速”。

#### 本轮全部固定配置与核验

| 项目 | 固定配置 |
|---|---|
| 基线 | 原始RDD Unified-L/2，400K，训练时间段fine/middle/coarse＝300/300/400 |
| 候选 | front-keep1、跳第1–6层的reader；固定2400父模型＋teacher-only固定400次续训，不选best；没有新coarse记忆修正、没有neighbor分支 |
| 复用内容 | 原始16×16 stage在t=300的第13层完整hidden；fine仍计算第0层和7–23层，reader使用当前fine查询读取该coarse记忆 |
| 采样 | 两组各10,000张；同seed、同类别、DDPM80/80/90、CFG1、noise filling、TF32关闭；同SD-ft-ema VAE/BF16 |
| 资源 | GPU0–3四个独立worker，每卡每批64、末批4；分块保存，可核验后续跑；不停止其他任务 |
| 测量 | 原ADM的IS、FID、sFID、Precision、Recall；共享同一次reference激活计算，指标在CPU上运行 |
| 检查 | 先每卡4张smoke；逐块核验原8/16端点、标签、90次fine调用、checkpoint/代码SHA；正式输出每组10K |
| 完成状态 | 两组各10,000张、五项指标、最终审计均完成；峰值allocated约3.77GiB/卡，无OOM，tmux正常退出 |

最终CPU审计核验全部320个数据块、20,000张计数、标签/seed/原8与16端点配对、数据像素格式、所有模型/代码/参考/评估器SHA及最终NPZ哈希。405文件的预览与元数据归档已下载，SHA256为`c6ca0ef99cb5a1a09ce3b8c7c7470b201e0d4028c6f3b4131d437360fecc5885`；64张固定预览图在服务器与NPZ像素逐项比较、本地逐图SHA核验均通过。最终指标JSON与审计哈希也在本地交叉核验通过。

[完整五项指标](../../eval_results/csra_reader_independent_fid10k_20260914/metrics/summary_fid10k.json) · [最终审计与差值](../../eval_results/csra_reader_independent_fid10k_20260914/quality_audit.json) · [采样核验](../../eval_results/csra_reader_independent_fid10k_20260914/sampling_audit.json) · [指标日志](../../eval_results/csra_reader_independent_fid10k_20260914/metrics.log) · [完整启动日志](../../eval_results/csra_reader_independent_fid10k_20260914/launch.log)。原tmux为`rdd_csra_reader_fid10k_0123_20260914`，已退出；服务器四卡采样日志在该结果目录的`sample_gpu0.log`至`sample_gpu3.log`。本轮没有测生成速度。

#### 独立样本固定对比图：左原始RDD，右reader

以下四图每图固定取对应分片的前8张，共32对，不挑成功或失败样本；四图均已逐张查看。它们说明部分主体、颜色和构图确实保留，但不是统一轻微变模糊：

- 图1：犬、部分物体背景接近；列车前部/侧面结构明显改变，乐器演奏者出现额外重叠的人脸/身体形状。原图本身也有不自然之处，不能全部归因于reader。
- 图2：花瓶、黑犬与室内家具整体接近；船边人物、桨/肢体和细小结构发生改变。瓶身文字两边都不可靠，不能仅凭文字断言候选独有失败。
- 图3：螳螂和显微物体相对稳定；牛身前增加竖直杆状结构，狮子鱼鳍和轮廓改变。熊的姿态也变化，但换姿态本身不等于质量更差。
- 图4：龙虾、部分物体及背景相似；几个玻璃杯变成一个占画面的大杯，猴子由侧视变成偏正视，细部和局部结构变化较大。这些构图变化也不能自动判为不自然。

因此，不能把所有差异都解释为“只损失高频”，也不能只展示接近的犬/花瓶而忽略其余样本。主质量结论以同10K完整指标为准。

<details><summary>独立10K固定对比图 1：含列车与人物变化</summary><img src="../../eval_results/csra_reader_independent_fid10k_20260914/shard_0/comparison_grid.png" style="max-width:100%;width:800px" alt="独立样本分片0，左原RDD，右跳6层reader"></details>
<details><summary>独立10K固定对比图 2：花瓶、黑犬与船边人物</summary><img src="../../eval_results/csra_reader_independent_fid10k_20260914/shard_1/comparison_grid.png" style="max-width:100%;width:800px" alt="独立样本分片1，左原RDD，右跳6层reader"></details>
<details><summary>独立10K固定对比图 3：含牛身竖杆与鱼鳍变化</summary><img src="../../eval_results/csra_reader_independent_fid10k_20260914/shard_2/comparison_grid.png" style="max-width:100%;width:800px" alt="独立样本分片2，左原RDD，右跳6层reader"></details>
<details><summary>独立10K固定对比图 4：含玻璃杯和猴子朝向变化</summary><img src="../../eval_results/csra_reader_independent_fid10k_20260914/shard_3/comparison_grid.png" style="max-width:100%;width:800px" alt="独立样本分片3，左原RDD，右跳6层reader"></details>

### 最新结果：coarse记忆修正已训练完成，没有显示可靠质量收益

**两个小网格block的修正分支已完成固定400更新和768张三列配对图，未OOM。** 同256个开发样本、同parent/训练数据/256batch/更新数，比较原RDD、原reader续训、加记忆修正。原8/16端点逐元素相同，重跑原RDD的256张PNG与上一轮哈希完全相同，教师状态对照图片和分类器也已核验。

| 方法 | 类别Top-1 | 类别Top-5 | 目标类别平均概率 | latent L2偏差 |
|---|---:|---:|---:|---:|
| 原RDD Unified-L/2 | 69.14% | 87.50% | 0.27543 | 0 |
| 原reader＋同预算400更新 | 66.02% | 85.94% | 0.24309 | 0.25569 |
| reader＋coarse记忆修正＋400更新 | 66.02% | 84.77% | 0.24115 | 0.25548 |

修正分支相对原reader：Top-1差为0，配对95%区间[−2.73,+2.73]个百分点；Top-5差−1.17个百分点，区间[−3.13,+0.78]；目标类别概率差−0.00195，区间[−0.00580,+0.00170]。**没有可靠改善。** 不能把“区间含0”当成已证明质量等价。主验证MSE只低约0.29%，最终latent偏差只低约0.08%，也不能称为画质提升。

这不是新增分支没学到任何权重：训练后末端修正权重RMS为0.001422、最大绝对值0.01524，已偏离全零初始化。它增加真实计算，却没有通过当前质量筛选，故本轮不优先保留、不自动延长训练。结论仅针对这套结构、固定400预算和复用的开发样本，不否定所有coarse记忆更新方式。

视频方面的13.09%／15.34%仍只证明**未训练video模块的计算空间**，不能与这组2D结果合并成“训练后视频质量和速度均达标”。下一步若继续，应先确认较简单reader在独立样本上的生成质量/FID，而不是继续把单步MSE的小幅下降当作增加分支的理由。整体质量目标尚未达成。

[配对统计](../../eval_results/csra_memory_refiner_20260914/quality_summary.json) · [完整协议审计](../../eval_results/csra_memory_refiner_20260914/completion_audit.json)。服务器checkpoint/优化器/数据/代码/初始状态/四rank轨迹与768PNG审计通过；四张固定grid已全部查看。完整803文件归档已下载，SHA256为`8c445d3421e509f62e3cf558649a451c14761f731466ef4fb64baa7932541376`；768张PNG在本地逐文件核验哈希全部通过。

**固定图的观察：新增修正与原reader通常肉眼非常接近，没有一致的恢复效果。** 网格1中白犬、围栏稳定，但杯子文字、人物嘴部和仪表刻度的改变仍在；网格2中昆虫、鞋、狮子鱼接近，打字机键盘重排未恢复。网格3中的熊从正视变成侧视、人物身份/表情改变，两种学生几乎相同；这不只是少了细纹理。网格4的女性面部和乐器演奏者仍有明显问题，修正分支没有救回。不能把原RDD本身就不自然的例子算成候选独有问题，也不能把不同构图自动等同于不自然。

以下每张是固定前8个样本，没有按好坏筛选。三列依次是：**原始RDD／同预算教师状态续训的原reader／增加coarse记忆修正的reader**。这些不是三种不同采样参数；三列均DDPM80/80/90、CFG1、noise filling。

<details><summary>coarse记忆修正：固定对比图 1</summary><img src="../../eval_results/csra_memory_refiner_20260914/quality/shard_0/comparison_grid.png" style="max-width:100%;width:1100px" alt="原RDD、原reader、记忆修正reader，固定样本1"></details>
<details><summary>coarse记忆修正：固定对比图 2</summary><img src="../../eval_results/csra_memory_refiner_20260914/quality/shard_1/comparison_grid.png" style="max-width:100%;width:1100px" alt="原RDD、原reader、记忆修正reader，固定样本2"></details>
<details><summary>coarse记忆修正：固定对比图 3，含人物与熊姿态变化</summary><img src="../../eval_results/csra_memory_refiner_20260914/quality/shard_2/comparison_grid.png" style="max-width:100%;width:1100px" alt="原RDD、原reader、记忆修正reader，固定样本3"></details>
<details><summary>coarse记忆修正：固定对比图 4，含人物与乐器失败例</summary><img src="../../eval_results/csra_memory_refiner_20260914/quality/shard_3/comparison_grid.png" style="max-width:100%;width:1100px" alt="原RDD、原reader、记忆修正reader，固定样本4"></details>

### 前一轮：固定快照状态回放没有改善质量

**2026-09-14：两组固定400次新增更新、1024张配对PNG与4张固定grid已完成。** 两组均训练reader与fine主干，原8/16独立冻结。256个新开发样本，每个生成原RDD、父模型、教师状态续训、50%学生状态回放四个版本；同seed/类别、DDPM80/80/90、CFG1、noise filling。

| 方法 | 类别Top-1 | 类别Top-5 | 目标类别平均概率 | 相对原RDD的latent L2偏差 |
|---|---:|---:|---:|---:|
| 原RDD Unified-L/2、400K | 69.14% | 87.50% | 0.27543 | 0 |
| 跳6层reader父模型、固定2400 | 63.67% | 85.16% | 0.24613 | 0.26202 |
| 同父模型＋400更新，教师状态 | 66.02% | 85.94% | 0.24309 | 0.25569 |
| 同父模型＋400更新，50%学生快照状态 | 60.16% | 84.77% | 0.23399 | 0.25286 |

**主对照“50%回放−教师状态”的Top-1为−5.86个百分点，配对bootstrap95%区间[−9.77,−2.34]。** 目标类别概率差值−0.00911，区间[−0.01762,−0.00078]。虽然学生状态验证MSE从0.008608降到0.008365、最终latent偏差少约1.11%，并没有可靠质量改善。不自动延长这组失败配置；但不能据此否定所有动态on-policy训练。

教师状态续训比父模型的Top-1高2.34个百分点，但区间[−2.34,+7.03]包含0，也不是可靠改善。区间仅针对这256图、一个训练seed；分类不是FID、真实感或视频质量，latent更接近不是画质更高。与前一轮不是同一批图，不跨表直接比较高低。

四张固定grid均已检查，每列依次是**原RDD／父模型／教师状态续训／50%回放**。犬类、食物、植物和部分物体可辨认且接近；文字、仪表布局、人物面部和乐器场景仍明显改变。部分变化只是构图差异，不能一概叫崩坏；三种学生之间常肉眼接近，不能凭几个图判定优劣。

[逐图统计及配对区间](../../eval_results/csra_state_replay_20260914/quality_summary.json) · [协议审计](../../eval_results/csra_state_replay_20260914/completion_audit.json)。服务器审计已核验两组checkpoint/优化器/随机状态、数据哈希、预算、1024张PNG哈希及原8/16端点一致性；完整1080文件归档已下载，SHA256为`7a006ec7ca81d3bf4ca82832ab53505ab233f7e783b96ccedf428e6cb1ec2c10`，全部1024张PNG在本地逐一复核哈希通过。

<details><summary>状态回放固定对比图 1</summary><img src="../../eval_results/csra_state_replay_20260914/quality/shard_0/comparison_grid.png" style="max-width:100%;width:1280px" alt="原RDD与三种学生，固定样本1"></details>
<details><summary>状态回放固定对比图 2</summary><img src="../../eval_results/csra_state_replay_20260914/quality/shard_1/comparison_grid.png" style="max-width:100%;width:1280px" alt="原RDD与三种学生，固定样本2"></details>
<details><summary>状态回放固定对比图 3</summary><img src="../../eval_results/csra_state_replay_20260914/quality/shard_2/comparison_grid.png" style="max-width:100%;width:1280px" alt="原RDD与三种学生，固定样本3"></details>
<details><summary>状态回放固定对比图 4，含人物与乐器失败例</summary><img src="../../eval_results/csra_state_replay_20260914/quality/shard_3/comparison_grid.png" style="max-width:100%;width:1280px" alt="原RDD与三种学生，固定样本4"></details>

**本轮结构（已完成）：先在小网格上修正 coarse 记忆，再供 fine 读取。** 当前reader已经读取当前fine查询，但coarse内容本身是上一stage固定留下的。新增分支把当前fine特征按coarse cell求平均，与真实coarse hidden结合，在原生低分辨率网格运行两个受当前时间调制的attention＋FFN block，预测对旧记忆的修正。之后使用原reader；fine仍计算第0层、跳1–6、再计算7–23。

```text
上一stage已有coarse hidden ─┐
                          ├─ 小网格、当前t下的两个block → 修正coarse内容 ─┐
当前fine第0层 → 按cell平均 ─┘                                           │
当前fine第0层的完整特征 ─────────────────────────────── 原reader ←─────┘
                                                       ↓
                                                 fine第7层及以后
```

末端修正投影初始化为0，开始时函数等于父模型，但**完整分支仍真实执行并计时**；每个fine时刻都重新修正。先做真实FastWan全流程计时，计入pool、两个小网格block、原reader与原VSA/gate；若连10%净去噪空间都留不住，不进入大训练。通过后才与已完成teacher-only同父模型/数据/256batch/400更新比较。旧快照回放不继续混入，避免同时改变多个变量。

这是本轮验证过的结构假设，质量结论见上表：在固定400预算下没有可靠改善。额外参数和通用attention不是RDD独有；实际优势必须来自复用已生成的coarse状态，后续还需排除“只是增加小网络容量”。也不能说理论上只有RDD能用。

#### coarse记忆修正的完整实验配置：计算筛选及四卡训练已完成

**真实FastWan的256次交错计时已完成。** 每格8组ABBA/BAAB、32条正式轨迹，预热后测试。模型H1536、30层、DMD三次调用、29帧；相同输出shape和stage配置、native VSA保留20%及coarse output/gate、noise filling、无offload、两份模型均驻留。基线是**同配置RDD+VSA、不跳层**；候选执行fine第0层、跳1–6、再执行7–29。原reader和新reader均完整计入缓存、pool、投影、attention/FFN、边界及同步。不是相对全分辨率模型的数字。

| 输出／stage | 候选 | 同RDD基线／候选去噪中位数 | 配对延迟下降均值 | 配对组95%区间 |
|---|---|---:|---:|---:|
| 832×448 Spatial421 | 原reader | 0.9088／0.7992 s | 13.33% | [11.03,16.48]% |
| 同上 | ＋coarse记忆修正 | 0.9083／0.8031 s | 11.15% | [10.18,12.03]% |
| 832×448 ST421 | 原reader | 0.8722／0.7692 s | 12.51% | [11.71,13.23]% |
| 同上 | ＋coarse记忆修正 | 0.8722／0.7716 s | 11.99% | [11.43,12.59]% |
| 1280×768 Spatial421 | 原reader | 2.6397／2.2817 s | 13.68% | [13.31,14.07]% |
| 同上 | ＋coarse记忆修正 | 2.6369／2.2909 s | 13.09% | [12.93,13.28]% |
| 1280×768 ST421 | 原reader | 2.4541／2.0875 s | 14.84% | [14.47,15.28]% |
| 同上 | ＋coarse记忆修正 | 2.4643／2.0820 s | 15.34% | [14.85,15.92]% |

**修正分支增加成本，不是它本身又省了时间。** 1280 Spatial的bridge中位耗时46.04→59.02ms，ST为41.07→46.05ms；ST整模型看似反而更快，是分别与基线配对时的共享GPU波动，不能称为修正分支加速。表内百分比按配对组计算，不等于两个时间中位数相除。区间不排除共享负载的系统误差；不含VAE，且视频bridge未训练，**仅用于判断质量补偿还有没有计算预算**。

两种1280配置的95%区间下界均超过预先设定10%门槛，故进入训练。新增分支零初始化完整视频轨迹差为0，基线wrapper和两份模型也完全一致；零权重没有被当作不执行分支。数据：[完整计时](../../eval_results/csra_memory_refiner_20260914/video_cost/metrics.json) · [预定计算门槛](../../eval_results/csra_memory_refiner_20260914/cost_gate.json)。当前调用PyTorch SDPA等已有算子，**这一轮没有声称新写CUDA kernel**。

15项单元/回归测试通过。实际4×64、两次反向/AdamW更新的smoke通过，零初始化64样本输出逐元素等于父模型；teacher/student两套初始validation与旧teacher-only对照完全相同，四rank首两步的数据索引、标签和t也完全相同。smoke与正式训练峰值allocated均约9.78GiB/卡，无OOM。首次启动的数据格式校验错误已修复并保留失败日志；没有修改旧数据、旧模型或旧训练器文件。

正式任务`rdd_csra_memory_refiner_train_0123_20260914`已在0–3卡完成固定400新增更新，不选best、不自动延长。主干仍Unified-L/2原400K、300/300/400；同父模型、同4096训练图/128验证图、teacher-only状态、相同seed与batch256；AdamW重置，reader/新分支LR1e-4、fine LR1e-5、WD0.01、clip1、主MSE＋0.05方差头损失，BF16 AMP/FP32主权重，关闭TF32，原8/16独立冻结。

已自动完成256个配对开发样本：重跑原RDD并通过逐PNG哈希检查、生成新候选，复用已核验的teacher-only对照；三列共768PNG、4张固定grid。它们是复用的开发seed，不能包装为全新holdout。最终统计与全部固定grid见本节最上方，没有可靠质量改善；完整归档及768张PNG均已在本地校验通过。

代码：[小网格修正](../../rdd_video/attention_analysis/csra_memory_refiner.py) · [2D接入](../../2d/attention_analysis/csra_refined_student.py) · [训练](../../2d/attention_analysis/train_csra_memory_refiner.py) · [自动训练与采样脚本](../../2d/exp/train_csra_memory_refiner_after_cost.sh)。日志在服务器`eval_results/csra_memory_refiner_20260914/train.log`，后台链路日志为`eval_results/csra_memory_refiner_train_launcher_20260914.log`。独立教学仓库FastVideo-RDD-minimal不变。

### 前一轮结果：邻域修正没有可靠改善画质

**2026-09-14：四组固定2400步、1536张配对图片已全部完成并下载核验。** 没有OOM，原始8/16阶段保持逐元素一致；四组初始权重、数据顺序、优化器重置和预算检查通过。有来源时，加邻域分支让验证主MSE从0.010861降到0.010126（少6.77%），但生成结果并没有相应的可靠改善。

| 固定256张配对开发图 | 类别Top-1 | 类别Top-5 | 相对原RDD的latent L2偏差 |
|---|---:|---:|---:|
| 原始RDD Unified-L/2 | 68.75% | 87.11% | 0 |
| reader＋真实coarse，邻域关 | 62.89% | 79.69% | 0.25510 |
| reader＋真实coarse，邻域开 | 61.72% | 80.86% | 0.25069 |
| reader＋清零coarse，邻域关 | 57.42% | 78.91% | 0.27537 |
| reader＋清零coarse，邻域开 | 53.52% | 80.86% | 0.26879 |

真实来源下，“邻域开−关”的Top-1为−1.17个百分点，配对95%区间[−4.69,+1.95]；Top-5为+1.17个百分点，区间[−1.17,+3.91]。source×neighbor的Top-1交互为+2.73个百分点，但区间[−1.95,+7.42]也包含0。因此**不能宣称邻域带来稳定质量收益或跨尺度特有的协同收益**。latent偏差少1.73%不是画质提高1.73%。这是一个训练seed、反复使用的256张开发图；分类不是FID/真实感。

固定展示图中，海滩、甲虫和部分犬类较稳定；reader开关邻域通常看起来十分接近。复杂人物、钢琴键、部分动物形状仍改变，网格2的鹅在真实来源两组中仍几乎消失。不能只展示稳定图片，把这轮算成成功。

数据：[完整统计](../../eval_results/csra_neighbor_2x2_20260914/factorial_summary.json) · [checkpoint/协议审计](../../eval_results/csra_neighbor_2x2_20260914/completion_audit.json)。1627文件归档SHA256为`60b4ee8b0f9ec1cbd4a027e5a9287fe895e68ba57dfafbc2364067aa1c96dd56`；全部1536张PNG在本地再次逐一校验SHA通过。以下每列依次是原RDD、reader/source、neighbor/source、reader/zero、neighbor/zero；固定展示每shard前8张，没有按好坏挑选。

<details><summary>邻域实验固定对比图 1</summary><img src="../../eval_results/csra_neighbor_2x2_20260914/comparison_shard0.png" style="max-width:100%;width:1280px" alt="邻域实验固定对比图1"></details>
<details><summary>邻域实验固定对比图 2</summary><img src="../../eval_results/csra_neighbor_2x2_20260914/comparison_shard1.png" style="max-width:100%;width:1280px" alt="邻域实验固定对比图2"></details>
<details><summary>邻域实验固定对比图 3，含鹅消失的失败例</summary><img src="../../eval_results/csra_neighbor_2x2_20260914/comparison_shard2.png" style="max-width:100%;width:1280px" alt="邻域实验固定对比图3"></details>
<details><summary>邻域实验固定对比图 4</summary><img src="../../eval_results/csra_neighbor_2x2_20260914/comparison_shard3.png" style="max-width:100%;width:1280px" alt="邻域实验固定对比图4"></details>

**已完成：固定学生快照回放的两组训练和全部配对生成。** 4096训练图和128验证图的两种轨迹均已采齐；两组实际4×64、各2次更新的smoke通过，初始验证逐项一致、四rank的数据索引/类别/t匹配，峰值8.98GiB/卡。不是继续延长失败的邻域分支，而是从同一个“邻域关、真实来源、2400步”权重开始，固定网络和推理开销，只改变部分训练输入：

1. 同一seed/类别/采样噪声，原始8/16不变；分别让原RDD和学生走完90个fine调用。
2. 各取相同8个时刻。对每个实际状态，重新问原RDD应该输出什么；绝不把教师另一条轨迹上的答案硬贴给学生状态。
3. 两组同4096训练图、128验证图、每图8状态，同4卡×64、AdamW重置、LR和400次新增更新。一组全用教师状态；另一组按固定索引/epoch规则用一半教师、一半学生状态。两组都训练reader与fine主干，原8/16冻结。
4. 用同256个新的开发样本比较原RDD、父模型、teacher-only和mixed50，全部保存，不因某组loss好看而改变终点。

**这是一轮固定快照回放，不是每次更新都刷新轨迹的严格on-policy训练。** 旧版诊断已发现学生状态的单步MSE反而较低，所以不预设“误差爆炸”，也不保证回放能救质量。它只测试“同预算时，用学生会遇到的输入再训练是否更有效”；不添加推理算子，也不把通用训练技巧称为RDD专属贡献。相关一般原则见[DAgger原论文](https://proceedings.mlr.press/v15/ross11a.html)；针对生成场、在学生状态上查询教师的相关工作见[DanceOPD](https://arxiv.org/abs/2606.27377)。本轮不是复现它们的全部算法，更没有套用其保证。

本轮代码：`csra_state_replay_data.py`、`collect_csra_state_replay.py`、`train_csra_state_replay.py`、`sample_csra_state_replay_quality.py`。采集与训练两个任务均正常结束，不再重启。仅使用0–3卡，采样显存上限18%、训练24%；无OOM，正式训练峰值8.98GiB/卡。完整数据、日志、结果在`eval_results/csra_state_replay_20260914/`。原父模型的teacher/student状态MSE：训练0.010866/0.008616、验证0.011574/0.009027；没有单步误差爆炸。这是各自状态上的误差，不是质量优势，见[数据摘要](../../eval_results/csra_state_replay_20260914/data_summary.json)。最终生成结果与负面结论见本节最上方。

### 已完成的邻域分支：视频计算可行性（不是训练后视频质量）

fine邻域修正不替换coarse全局读取，而是在bridge输出上增加`LayerNorm → H到128维 → 空间3×3逐通道卷积 → SiLU → 回到H维`的残差。最后投影初始化为0，仍真实执行整条分支。19项相关测试通过，真实FastWan的完整3次调用也确认初始化前后latent逐元素一致。1280×768×29帧下，加入完整新分支后的净去噪延迟比同RDD+VSA少**13.33%（Spatial421）／15.19%（ST421）**；这仍是未训练视频bridge的计算可行性，不含VAE。2D四组训练与质量的最终结论见上方：邻域修正未通过可靠质量改善筛选。

### 邻域修正：独立计时与已完成的四组对照

**四组实际4×64 smoke、四组固定2400端点及全部配对图片均已完成。** 每组零初始化的实际batch64输出差异为0，邻域开/关的初始验证结果完全相同，四组首两次更新的数据索引、t和类别一致。峰值allocated为每卡8.98–9.09GiB，未OOM。旧邻域任务已经退出，不再等待或重启它。

FastWan：H=1536、30层，保留当前native VSA20及其coarse输出/gate。fine执行第0层，bridge替代第1–6层，随后执行7–29层；捕获真实middle第13层完整hidden。原始coarse与fine两份模型权重同时驻留；所有缓存、bridge、padding、noise filling和同步计入。同shape基线仍是**相同多分辨率RDD+native VSA**，不是全分辨率基线。每格12组ABBA/BAAB、48次正式轨迹；表内时间为各组全部轨迹的中位数，百分比及区间按整组配对计算，故不要求它们恰好等于两个时间中位数相除。

| 输出／三阶段 | 修正 | 同RDD基线／候选去噪时间 | 配对延迟下降均值 | 配对组bootstrap 95%区间 |
|---|---|---|---|---|
| 832×448×29 Spatial421 | 原reader | 0.9101／0.8016 s | 12.18% | [11.71,12.68]% |
| 832×448×29 Spatial421 | reader＋邻域 | 0.9103／0.8035 s | 12.01% | [11.39,12.62]% |
| 832×448×29 ST421 | 原reader | 0.8760／0.7626 s | 13.04% | [12.41,13.63]% |
| 832×448×29 ST421 | reader＋邻域 | 0.8739／0.7719 s | 12.22% | [11.47,12.90]% |
| 1280×768×29 Spatial421 | 原reader | 2.6441／2.2741 s | 14.06% | [13.80,14.36]% |
| 1280×768×29 Spatial421 | reader＋邻域 | 2.6383／2.2844 s | 13.33% | [13.04,13.58]% |
| 1280×768×29 ST421 | 原reader | 2.4507／2.0888 s | 14.76% | [14.39,15.19]% |
| 1280×768×29 ST421 | reader＋邻域 | 2.4598／2.0848 s | 15.19% | [14.70,15.70]% |

**邻域分支不是省时算子，它是为争取质量而增加的成本。** 1280 Spatial421的bridge中位耗时由45.84ms增至53.36ms，但仍保留超过10%的总体空间。ST中加入分支后看似更快，不能解释为卷积提高了速度：reader开/关是分别与基线配对、不是二者直接交错对测，且GPU上另有训练。这里的区间也只描述本次配对组波动，不排除共享负载或调度带来的系统偏差。新旧front位置不能混用计时。

原始记录：[384次计时](../../eval_results/csra_neighbor_video_feasibility_20260914/full/metrics.json) · [逐项验收](../../eval_results/csra_neighbor_video_feasibility_20260914/full/audit.json)。审计确认8个entry、384条正式轨迹、实际跳层/bridge调用均正确，峰值allocated为10.71GiB；duplicate、no-op wrapper、零初始化新分支的完整轨迹最大差均为0。

2D新对照固定：Unified-L/2，原400K、300/300/400；每个来源从相应direct-control固定1600权重开始，四组全部重置AdamW状态，额外800次更新到2400。每卡64、4卡实际batch256；bridge LR1e-4、fine LR1e-5、WD0.01、clip1、原MSE＋0.05方差头目标。保留相同4096训练图／128验证图、8个教师时间状态，依然是**离线教师轨迹训练，不是on-policy**。原8/16模型独立冻结。每组开始先验证实际64图的新分支零初始化等价性；四组数据索引、t、类别逐项配对。最终用同256开发图、DDPM80/80/90、CFG1、noise filling比较，必须同时报告source×neighbor交互和失败图片。分类命中不是FID/自然度，不因loss下降就继续盲目延长训练。

独立代码：`csra_fine_neighbor.py`、`csra_neighbor_student.py`、`csra_neighbor_protocol.py`；后台启动器：`2d/exp/train_csra_neighbor_2x2_4gpu.sh`，仅0–3卡。新格式为`csra_front1_skip6_spatial_neighbor_v1`，旧checkpoint仅允许显式父权重迁移；resume与最终评估检查模型格式、源文件SHA、固定父checkpoint SHA和训练协议。不会向单独的FastVideo-RDD-minimal学习仓库添加这些研究代码。

**前一轮1600步结果：分阶段训练没有改善质量。** 相同256张图，真实coarse的直接联合／先固定主干800再联合800，Top-1分别为**61.72%／57.81%**，原RDD为**68.75%**；分阶段的latent偏差也更大。不再延长这组训练。coarse仍有用：直接联合的清零对照为56.25%，但这不等于质量达标。随后完成128图×8时刻×两种来源的只读诊断：分别补回教师特征的区域均值或区域内部差异，均使最终输出误差更大；完整补回才有改善。因此不能归结为“只缺高频”，也不能把分频带oracle当成可部署的新方法。该轮1536张PNG、8张固定对比图和原始统计已下载，下面保留成功与失败例。其后的邻域修正已完成，最新负面结果与状态回放试验见本节顶部。仍要求同RDD+VSA下计入全部开销的真实视频净加速≥10–20%，不能把旧未训练bridge计时拼成当前模型的质量与速度证明。

**已实际做过“主模型+adapter一起训练”，不再只是训练adapter。** 全阶段共享主干联合更新后，完整生成反而更差；保持原8/16权重、另训fine主干有小幅改善，但V2、V5和V6仍没有稳定的coarse来源优势。详细对照与失败图片保留在2和8节。

**前一轮：前缀8、3200步训练和256张新样本/方法的配对评估均完成，共768张图。** 正确coarse来源组的类别Top-5命中率为81.64%，无来源组75.39%，原始RDD为91.41%。这支持“真实coarse信息有用”，但当前替代前缀仍损害条件语义，尚未达到质量目标。Top-1分别60.94%/55.86%/73.44%；有/无来源的Top-1差异区间仍含0，不夸大稳定性。四张固定对比grid与逐图指标见§8末尾。没有FID、净视频加速或“只差训练”结论。

两组3200步主MSE为0.014664/0.015920；旧三组开发预览平均latent偏差0.265944/0.279494。有coarse相较1600步的偏差只再低0.71%，不自动续到6400。**评价口径：偏离dense构图不自动等于类别错误；类别正确也不等于图像自然。** 三seed“恢复fine前8层”的诊断也已完成：部分人物结构回来，但对比度/纹理出现新伪影，平均偏差没有稳定改善。恢复路径更贵、且未联合训练，不能作为加速效果或单一原因证明。

**前一轮完成：hidden对齐监督的四组训练与六组预览。** 各自从3200新增800步，全部选中best4000。与同预算无监督组比，有coarse的hidden误差低16.85%，但主预测误差只低0.19%、最终latent偏差只低0.25%；无coarse的最终偏差反而高0.14%。这轮没有实质性修复，不继续靠加大这个辅助项或原样延长训练来宣称进展。表格和所有六张grid见§8末尾。

**前一轮：前缀6的best800，在256个新样本/方法上的匹配评估。** 原RDD / 有coarse / 清零来源的Top-1为68.75% / 59.38% / 53.13%，Top-5为87.11% / 82.03% / 75.39%。有coarse比清零高6.25个百分点Top-1，配对bootstrap95%区间为[1.17,11.33]个百分点；但仍比原RDD低9.38个百分点。**来源优势得到更多样本支持，质量还没达标。** 四张固定grid与完整配置在§8末尾；全部768张PNG哈希已校验，四张grid和全部指标已下载，单张PNG留在服务器。实际视频bridge净计算检查也已完成，结果如下，但不能与2D质量拼成同一训练后视频模型的结果。

上一轮线性8层前缀的两组1600步和三组生成均已完成。正确coarse组validation主MSE为0.018073，无来源组0.020542（少12.02%）；三组平均final latent偏差0.279849 vs0.292452（少4.31%），但只在2/3组占优。白帽人物由严重塌脸改善为可辨认的人脸，但身份、帽子和表情改变；另外两个人物仍明显失败。**有可用信息的证据，不等于质量达标，更不是加速4.31%。** 更早14层结果也保留；整体质量与视频净加速目标尚未达成。

目前没有证据证明“已有三种只能在 RDD 上使用、且各自有效的视频加速方法”。优先保留三条路线：A跨尺度整块替代、D从旧coarse记忆重新读取、C只精算新增细节；B简单区域路由因诊断收益弱而降级。另有一个精确工程优化。明确区分已运行、待验证和不满足专属性的部分，**不是把普通缓存换个名字，或把三个候选当成三个已证实贡献。**

**新增视频计算验证已完成：384次匹配计时。** 真正计算条件bridge、缓存捕获、上采样和两套权重驻留后，1280×768×29帧下替代fine前6层，Spatial421 / ST421的denoiser延迟中位数分别少13.92% / 15.25%；前8层少19.21% / 20.93%。这些是**未训练Wan bridge的计算可行性**，不是训练后视频画质，也不含VAE。完整表与波动在§8末尾。

**10K质量结论仍未改变。** 原RDD / 有coarse best1600 / 清零best1600的FID为**11.8755 / 16.8040 / 20.3236**，Precision为0.6953 / 0.6495 / 0.6243，Recall为0.7007 / 0.6929 / 0.7007。来源信息有帮助，但质量仍未达到原RDD水平。这是同配置10K开发pilot，不可直接对比论文50K FID。全90时间点诊断未发现同状态预测误差爆炸；随后完成的四组loss对照，在同256张开发图上有来源Top-1仅高0.78个百分点，配对95%区间[-1.95,+3.52]包含0，latent偏差只少0.60%。因此不将新loss称为成功，也不据此继续跑大FID。视频端13.92–15.25%的前缀6净去噪延迟下降仍是**未训练video bridge的计算可行性**，不能与2D图拼成同一个训练后视频方法的质量-速度结论。

判断一个方法是否符合要求，分三层：

1. 是否必须利用前一个真实生成 stage 已有的 coarse 状态，而不是当前 fine 特征再池化？
2. 固定全分辨率模型如果没有这些状态，是否需要额外计算或改成多尺度结构才能实现？
3. 它是否只依赖 RDD 的特定概率路径，连其他多尺度生成方法也无法使用？

前两层符合“利用 RDD 已经做过的工作”的工程目标，但不等于第三层。跨尺度缓存可以推广到别的金字塔模型；不能声称数学上只有 RDD 才可能做。

## 2. 新实验：主模型与 adapter 一起训练

### 2.1 原方法在替换什么

原始 Transformer block 输出为 `h + Δ`。现在选中四层，不再计算这些层完整的 attention 和 MLP，而由小网络根据“上一个分辨率的 Δ、当前 fine 特征、当前时间”预测更新。没有复用完整旧 hidden，也没有缓存 attention probability。

目前的小网络有三条分支：coarse 更新的低秩变换、当前 fine 特征的局部高频修正、当前完整 fine 特征的低秩变换。每条末端投影 rank 为 64，hidden 仍为 1024。其固定输出投影最多张成 192 维空间，这正是上一轮容量分析检查的限制。

### 2.2 为什么不能只把冻结开关关掉

旧训练数据只保存了原教师的 coarse 更新。如果学生主模型已经变化，却一直喂原教师的更新，训练时的缓存和真正采样时学生生成的缓存就不同了。

新流程补齐原教师在 `8×8,t=600` 和 `16×16,t=300` 的输入和输出。每次更新都让**当前学生**重新计算这两个输入，16×16 的更新直接送给 fine adapter，且不截断梯度：

```python
pred16, live_cache = student.dense_stage(x16, t=300, capture_layers=layers)
pred32 = student.fine_stage(x32, t, cache=live_cache)
pred8 = student.dense_stage(x8, t=600)

loss = fine_distillation(pred32, original_teacher_target32)
loss += 0.5 * anchor(pred16, original_teacher_target16)
loss += 0.5 * anchor(pred8, original_teacher_target8)
loss.backward()  # fine loss 也能更新产生 live_cache 的模型参数
```

这里没有训练三份模型：三个 forward 使用同一个 Unified-L。每一层在 coarse forward 中都会执行，包括在 fine 阶段被替换的四层。

**仍然存在的限制：** 输入来自原教师轨迹，而不是当前学生完整生成的轨迹；这里只解决了缓存来源不一致，没有宣称解决所有 rollout distribution shift。低分辨率 anchor 目前只覆盖两个端点，不能保证其他时间点完全不退化。因此最终必须重新跑学生从 8×8 到 32×32 的完整生成，不能只看训练 loss。

### 2.3 配置与公平对照

| 项目 | 联合训练组 | adapter-only 对照组 |
|---|---|---|
| 原始教师 | `RDD-Unified-L/0400000.pt` | 相同 |
| 模型 | DiT-L/2，多分辨率，24 层，hidden 1024 | 相同 |
| 训练时间分配 | finest/middle/coarse = 300/300/400 | 相同 |
| adapter 初始化 | `RDD-CSRA-condition-v2-L400K-20260912/best.pt`，该轮第 800 步 | 相同 |
| 被替换层 | 代码编号 13、15、17、20；即第 14、16、18、21 层 | 相同 |
| 替换阶段 | 32×32 的全部 90 次调用，缓存来自 16×16 的 t=300 | 相同 |
| 可训练主模型参数 | 457,840,672 | 0 |
| 可训练 adapter 参数 | 2,055,168 | 相同 |
| 参数/计算精度 | FP32 主参数、BF16 autocast、FP32 adapter | 相同；原始教师目标保持原 BF16 计算 |
| 数据 | 4096 条 train、128 条 validation 教师轨迹；每条 8 个 fine 状态 | 相同 |
| 全局 batch | 256：GPU 0、1、2、3 每卡 64，不累计 | 相同 |
| 优化器 | AdamW；主模型 LR=1e-5，adapter LR=1e-4，WD=0.01，clip=1 | adapter LR=1e-4，其余相同 |
| 每个预测的损失 | 前 4 通道 MSE + 0.05 × 后 4 通道方差头 MSE | 相同 |
| 总损失 | fine 损失 + 0.5×middle anchor + 0.5×coarse anchor | 相同；冻结组 anchor 不产生主模型梯度 |
| 初始计划 | 800 次更新，每 100 步验证；初始化也参与 best 选择 | 相同 |
| best 标准 | 完整验证目标，包括 coarse anchor，防止只改善 fine 而破坏 coarse | 相同 |
| 随机种子 | 20279000 | 相同 |
| 检查点 | 新目录，包含学生主模型、adapter、optimizer | 独立新目录；不覆盖原 checkpoint |

这里不能直接拿旧 BF16 冻结主模型训练的 loss 当成严格匹配对照。新冻结组也使用 FP32 主参数与 BF16 autocast，只有是否更新主模型这个开关不同；新旧执行路径的少量舍入差异单独记录。

### 2.4 已验证与后台流程

24 项单元测试通过，包括：fine loss 到 coarse 输入的梯度、被替换层通过 coarse 分支收到梯度、冻结组不更新主模型、activation checkpointing 的输出及梯度一致性。

真实 Unified-L 的单卡测试采用 64 张、完整反向传播和 Adam 状态：两次更新均通过，共 459,895,840 个可训练参数；峰值 allocated 10.45 GiB、reserved 10.91 GiB。它只检查能不能正确更新，不是训练效果证据。

四个源数据分片各 132 个文件已补齐；每个文件都检查重放出的 16×16 缓存与原数据逐元素相同。不是只检查第一批。后台依次执行四卡 smoke、联合训练、冻结对照、七组配对生成。

**以下为历史启动记录，最终结果见2.5。** 第 100 步，fine 验证 MSE 从初始化 `0.007162915` 降到 `0.007028686`（约下降 1.87%）；包含 anchor 的总目标从 `0.007488615` 降到 `0.007364993`。当时尚未完成新冻结对照；该早期验证信号不是最终画质结论。正式四卡峰值 allocated 为 12.16 GiB/卡，reserved 13.04 GiB/卡；未 OOM，未使用梯度累计。

**完成检查：两组各 800 步和七组完整配对生成都已完成，结果见下节。** 联合组第 800 步也是总验证目标最优点，fine 验证 MSE 为 `0.006370536`，比初始化下降约 **11.06%**；包含 anchor 的总目标为 `0.006772430`。同时 middle/coarse anchor MSE 增至 `1.3053e-4 / 9.2729e-5`，高于初始化的 `5.0988e-5 / 3.3372e-5`。最终生成确认：单步 MSE 改善没有转化为更接近原教师的完整生成。

```bash
tmux attach -t rdd_csra_joint_pair_0123_20260913
tail -f logs/csra_joint_pair_20260913.log
tail -f logs/csra_joint_joint_20260913.log
```

正式启动器：`2d/exp/train_csra_joint_pair_4gpu.sh`。只使用 0–3 卡，启动训练前每卡需要至少 25,000 MiB 空闲，不会终止别人的进程。

生成对比的三个列是：**原始教师 dense、重新训练的冻结组、联合训练组**。不能把联合训练后的主模型 dense 输出当原教师，那会悄悄换掉参考答案。采样固定 DDPM 80/80/90、CFG=1、noise filling、4 张/seed。七个历史 development seeds 用来观察 pattern，不是未见测试集；本轮没有 FID 或视频正式质量结论。

### 2.5 完成结果：联合训练的单步误差更低，但完整生成更差

两组均完成800次更新；根据总验证目标，冻结组选择第100步，联合组选择第800步。没有为了图片好看手动选 checkpoint。

| 指标 | adapter-only best100 | 联合训练 best800 |
|---|---:|---:|
| fine 主输出验证 MSE | 0.007131910 | 0.006370536 |
| middle anchor MSE | 0.000050988 | 0.000130530 |
| coarse anchor MSE | 0.000033372 | 0.000092729 |
| 总验证目标 | 0.007444507 | 0.006772430 |
| 七组最终 latent relative L2 均值 | **0.1899906** | **0.2230209** |

相对原 dense 的最终 latent 偏差均值增加 **17.38%**，7组中6组增加。这个数不是 FID，也不是“画质下降17.38%”。但实际图像也出现明显失败：白帽人物在联合组严重变形，20275003的桌面物件以及20275004的背景布局明显改变。狗、面包、山谷等样例总体仍较稳定。不能只挑稳定图片宣布成功。

| seed | adapter-only | 联合训练 | 相对 adapter-only 的偏差变化 |
|---|---:|---:|---:|
| 20260921 | 0.189629 | 0.269619 | +42.18% |
| 20260922 | 0.218699 | 0.218292 | -0.19% |
| 20260923 | 0.147798 | 0.156648 | +5.99% |
| 20275001 | 0.180950 | 0.186492 | +3.06% |
| 20275002 | 0.224314 | 0.231273 | +3.10% |
| 20275003 | 0.160545 | 0.228181 | +42.13% |
| 20275004 | 0.207999 | 0.270641 | +30.12% |

当前结论是**这套全模型联合训练配置没有解决问题**，不是“任何全模型训练都没有用”。主要待查：只在teacher输入上训练导致rollout输入分布不匹配；共享主干更新改变了整个8/16路径，而低分辨率anchor只约束两个端点；以及新主干与缓存之间的适应/补偿。现有数据尚不足以区分它们。

已新增小型阶段互换诊断，选上述三个失败seed（20260921、20275003、20275004），比较原dense、冻结两段、联合两段、只换joint fine、只换joint coarse、joint主干不替换block、frozen主干不替换block。每组仍4张，同DDPM80/80/90与noise filling；记录阶段边界偏差与“在当前学生输入上询问原教师”的误差。互换后的内部特征可能不兼容，属于**反事实定位实验**，不是训练过的新方法或独立质量测试集。这里不新增长期训练。

所有七张对比图如下，每张从左到右始终为 **原始 dense / adapter-only / 联合训练**。

<details open><summary>20260921：白帽人物明显退化，狗和摄影器材总体较稳定</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="20260921三个模型对比"></details>
<details><summary>20260922：总体变化较小，不能代表所有样例</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="20260922三个模型对比"></details>
<details><summary>20260923：汽车、人物、鸟和蜗牛</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="20260923三个模型对比"></details>
<details><summary>20275001：总体布局较稳定，局部细节改变</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="20275001三个模型对比"></details>
<details><summary>20275002：动物和人物局部变化</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="20275002三个模型对比"></details>
<details open><summary>20275003：桌面物件明显变化，面包相对稳定</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="20275003三个模型对比"></details>
<details open><summary>20275004：背景布局偏离增加，杯子和动物仍大致保留</summary><img src="../../eval_results/csra_joint_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="20275004三个模型对比"></details>

七组指标、84张单图、7张grid、训练日志与完成标记已下载本机；归档 `eval_results/csra_joint_pair_complete_20260913.tar.gz`，SHA-256 `219039bd2e7f88ca4f5564b2d6a74d1153e291408bb953943ba6cb23c2f9fa17`。预览记录的首次采样秒数包含诊断执行环境影响，**不是可靠速度比较**。

### 2.6 阶段互换已完成：新 coarse 路径是主要嫌疑

三个失败seed的诊断全部完成，原冻结组与联合组的最终latent指标都复现了上一轮（允许CPU/GPU reduction造成的微小舍入差）。它们不是三个随机抽出的测试集，不把下表当总体质量估计。

| 实际使用的模型 | 20260921 | 20275003 | 20275004 |
|---|---:|---:|---:|
| 原主干 dense，BF16 | 0 | 0 | 0 |
| 冻结组：coarse/fine均用旧主干，fine用adapter | 0.189629 | 0.160545 | 0.207999 |
| 联合组：coarse/fine均用更新主干，fine用adapter | 0.269619 | 0.228181 | 0.270641 |
| **旧 coarse + 联合训练后的 fine/adapter** | **0.178800** | **0.146503** | **0.206165** |
| 更新 coarse + 原冻结组 fine/adapter | 0.273213 | 0.231443 | 0.268093 |
| 更新主干所有stage均dense，不替换block | 0.251511 | 0.226985 | 0.235440 |
| 未更新主干所有stage均dense，FP32主参数/BF16计算 | 0.008819 | 0.014553 | 0.006813 |

全部数值是相对原始dense的最终latent relative L2，越小越接近参考。最后一行专门记录精度路径的非零偏差，不能把这点全部算成训练影响。

**共同模式：只换fine都改善，只换coarse都恶化；即使不用adapter，更新后的主干也会产生严重变形。** 在fine开始之前，联合组16×16阶段末尾的输入latent已相对原轨迹偏离约11.25%、7.53%、11.81%。这是同一阶段同一形状的对比，不把不同stage归一化的比例直接拼起来解释增减。

因此，有证据优先怀疑：共享权重为fine任务更新时改变了原8/16生成路径；两个端点的anchor不足以约束整段生成。当前fine模型也不是完全正确，例如保留原coarse后白帽人物不再出现极端拉长，但肤色异常仍在。这不证明所有问题都来自coarse，也不证明更好的全时段保留约束无法让单一共享主干成功。

下面每张图七列顺序与上表一致（最后两列为更新/未更新主干dense）。完整trace和同状态教师误差都在本机 `eval_results/csra_joint_stage_isolation_20260913/`；峰值allocated约4.36GiB/卡，三个worker无OOM。

<details open><summary>20260921：第4列保留原coarse，避免极端头部拉长；红色面部问题仍在</summary><img src="../../eval_results/csra_joint_stage_isolation_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:1792px" alt="七种stage互换配置对比"></details>
<details><summary>20275003：桌面布局主要跟随coarse路径变化</summary><img src="../../eval_results/csra_joint_stage_isolation_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:1792px" alt="20275003阶段定位"></details>
<details><summary>20275004：背景偏差同样主要跟随coarse路径变化</summary><img src="../../eval_results/csra_joint_stage_isolation_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:1792px" alt="20275004阶段定位"></details>

### 2.7 固定原 coarse，训练实际执行的 fine 主干：已完成

基于上面的定位，不直接给失败的共享主干加训练步数。新实验固定8/16阶段为**原始BF16教师**，使用其真实cache；训练32阶段实际执行的20个Transformer block、embedding/output head与四个adapter。四个永远被替换的原fine block不参与优化，不虚报它们为训练参数。另跑完全相同数据/精度/原coarse路径的adapter-only对照。

| 配置 | 新组设置 |
|---|---|
| 模型/初始化 | 原Unified-L 400K + 同一个V2 best800 adapter；不是从失败joint800续训 |
| coarse路径 | 原始模型权重不更新；采样8/16输出与原dense做逐元素parity检查 |
| fine路径 | 联合组训练实际执行的主干+adapter；对照只训练adapter |
| 数据/损失 | 同4096训练+128验证轨迹、每轨迹8个fine状态；主输出MSE + 0.05×variance MSE |
| batch/LR | 0123，每卡64，总256，无累积；主干1e-5，adapter1e-4 |
| 预算 | 两组各800次更新，每100步验证；先四卡两步完整优化器smoke |
| 采样 | 同七个development seed，各4张，DDPM80/80/90、CFG1、noise filling |
| 会话/日志 | `rdd_csra_preserved_coarse_0123_20260913`；`logs/csra_preserved_coarse_pair_20260913.log` |
| 启动器 | `2d/exp/train_csra_preserved_coarse_pair_4gpu.sh` |

**必须说明代价：这已经是coarse/fine分别保留权重的stage-specialized版本，不是单一共享权重Unified模型。** 原coarse路径不增加NFE或重复运算，但与训练后的fine主干一起常驻会增加权重显存。日后视频测速必须把这份显存、可能的offload成本以及adapter实际计算都计入，不能把它称为免费提速。新checkpoint格式为`csra_preserved_coarse_v1`，必须使用`sample_csra_preserved_preview.py`；不能交给旧的联合模型采样器，让它再次用fine权重生成coarse。

新旧路径合计30项回归测试通过，四卡每卡64张的完整优化器smoke也通过。**两组800步、七组完整生成均已完成。** 实际训练主干382,281,760参数、adapter2,055,168参数；峰值allocated **9.74GiB/卡**、reserved11.96GiB/卡，无OOM。旧tmux正常结束，所有原结果保留。

| 指标 | 固定coarse + adapter-only best100 | 固定coarse + fine主干/adapter best800 |
|---|---:|---:|
| fine验证主输出MSE | 0.007132001 | 0.006330443 |
| variance验证MSE | 0.004437217 | 0.003981018 |
| 总验证目标 | 0.007353862 | 0.006529494 |
| 七组最终latent relative L2均值 | 0.1901036 | **0.1805941** |

完整生成偏差均值下降约**5.00%**，6/7组改善；这不是FID，也不是“画质提升5%”。前一轮共享主干训练造成的极端头部拉长没有出现，但白帽人物的面部肤色/形状仍明显不对，尚未通过质量目标。

| seed | adapter-only | fine主干联合训练 | 偏差变化 |
|---|---:|---:|---:|
| 20260921 | 0.190491 | 0.178075 | -6.52% |
| 20260922 | 0.222159 | 0.210202 | -5.38% |
| 20260923 | 0.148275 | 0.150083 | +1.22% |
| 20275001 | 0.181150 | 0.173503 | -4.22% |
| 20275002 | 0.223502 | 0.207935 | -6.96% |
| 20275003 | 0.158810 | 0.141903 | -10.65% |
| 20275004 | 0.206339 | 0.202459 | -1.88% |

七组两个学生的8/16阶段端点输入/输出与原dense的max-abs均为**0**。28张原dense PNG与上轮逐张SHA-256完全相同。此次差异没有混入新的coarse轨迹或不同的原始参照图。

以下每张三列为 **原dense / 固定coarse的adapter-only / 固定coarse的fine主干联合训练**。

<details open><summary>20260921：极端变形避免了，但白帽人物仍不能算修复</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse三模型对比20260921"></details>
<details><summary>20260922：纹理和边缘变化</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse20260922"></details>
<details><summary>20260923：本组latent偏差略增，保留负结果</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse20260923"></details>
<details><summary>20275001：总体布局稳定</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse20275001"></details>
<details><summary>20275002：动物和人物细节</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse20275002"></details>
<details open><summary>20275003：桌面物件没有上一轮的整体布局退化</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse20275003"></details>
<details><summary>20275004：背景恢复稳定，但与dense仍不同</summary><img src="../../eval_results/csra_preserved_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="固定coarse20275004"></details>

84张单图、7张grid、指标及日志全部位于本机`eval_results/csra_preserved_pair_20260913/`；归档`csra_preserved_pair_complete_20260913.tar.gz`，SHA-256 `7739d0b5a52706d079f6bb089a9e77b73f576766e5c3d74738f9ba7958ee9702`。

### 2.8 学生状态诊断：没有看到平均局部误差被放大

额外选32条validation轨迹（4分片各1批8张），每条8个fine观察点，共256状态/模型。分别在原教师的输入和学生自己生成的输入上，询问**同一个原教师在该输入上应该输出什么**。不是拿两个不同输入的预测直接做差。无训练、无VAE；诊断额外调用原教师，因此不计速度。

| 模型 | 教师状态上主输出MSE | 学生状态上主输出MSE | 学生/教师MSE | 全局归一化L2：教师→学生 |
|---|---:|---:|---:|---:|
| 固定coarse，adapter-only | 0.007680149 | 0.006730214 | 0.8763 | 0.11120 → 0.10478 |
| 固定coarse，fine主干联合训练 | 0.006848804 | 0.006093479 | 0.8897 | 0.10501 → 0.09960 |

全局归一化L2按`sqrt(总误差能量/总教师目标能量)`计算，而非逐项比值平均。在这批样本中，绝对MSE和归一化误差都没有出现平均放大；不能据此把“离开教师轨迹后局部预测彻底失效”当作主要已证实原因。**这不等于没有distribution shift或累积误差**：小的有方向误差仍可能逐步改变脸部和纹理，平均指标也可能掩盖个别图像。

误差较大的观察点集中在`t=84/40`，而不是fine刚开始的`t=299`。例如联合组学生状态MSE分别约0.01143/0.01235，t299约0.00144。它提示后半段细节恢复值得关注，但不能只凭MSE大小就直接推导最优loss权重。

所有原教师预测重放、原coarse缓存以及带/不带诊断的最终学生latent都逐元素一致（max-abs=0）。原始指标已下载`eval_results/csra_preserved_shift_20260913/`。这些validation轨迹没有被改标为训练数据。

### 2.9 容量对照已完成：V4逐通道修正没有明显额外收益

先检查容量而不是立即延长V2。保持原coarse固定、四个替换层、rank64、数据和训练设置不变，给V2增加一条很小的当前fine特征修正：

$$\Delta_{V4}=\Delta_{V2}+g(t)\odot\operatorname{LN}(h_{\mathrm{fine}})+b(t).$$

人话：旧的三条低秩分支，在减去偏置后，更新只能落在至多`3×64=192`维的固定输出子空间；新分支允许当前fine特征的1024个通道分别被缩放、平移，不必全部经由192维输出子空间。它仍没有fine token之间的attention，也不能自动补全所有远距离细节。

`g(t), b(t)`由每张图运行一次的`64→64→2048`时间MLP产生，最后一层零初始化。因此从相同V2权重开始时输出完全相同，再由训练决定是否需要这条路。新增4×137,280=549,120参数，adapter总计2,604,288；逐token额外工作主要是乘加。这里没有把rank从64提高，也没有改变采样NFE。

这个逐通道门控**本身不是RDD独有的创新**，它是跨尺度block替代中的修正部件。只有组合后保留质量、并扣除它的真实计算和两套权重开销后仍有视频净提速，才有研究价值。

两组800步与七组配对生成均完成，旧会话`rdd_csra_preserved_v4_0123_20260913`正常结束。33项回归测试和每卡64的完整优化器smoke通过；峰值allocated约9.75GiB/卡，无OOM。所有图片和日志在本机`eval_results/csra_preserved_v4_pair_20260913/`；归档SHA-256为`1baf88cb379c3e3e9f0ed97de8fd632304d05e408298ab29a433dfda845ba3d0`。

| 指标 | V2 adapter-only | V4 adapter-only | V2 fine联合训练 | V4 fine联合训练 |
|---|---:|---:|---:|---:|
| validation选出的步数 | 100 | 100 | 800 | 800 |
| fine主输出验证MSE | 0.007132001 | 0.007129452 | 0.006330443 | 0.006333493 |
| 总验证目标 | 0.007353862 | 0.007351102 | 0.006529494 | 0.006531659 |
| 七组最终latent relative L2均值 | 0.1901036 | 0.1900435 | 0.1805941 | 0.1804812 |

V4相对对应V2，最终偏差均值只降低约0.032%（冻结）和0.062%（联合），不构成有意义的质量突破。联合组6/7 seed稍好，但20275004更差；不能忽略它。逐一查看七张grid后，白帽人物仍红脸/变形，部分布局/纹理仍偏离。**不继续扩大V4实验，也不把它列为有效加速贡献。**

七组两种学生的原coarse端点parity均为0；28张dense PNG与V2原dense逐张SHA-256相同。每张图三列为**原dense / V4 adapter-only / V4 fine联合训练**：

<details open><summary>20260921：脸部失败仍未修复</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V4白帽人物等对比"></details>
<details><summary>20260922：蝴蝶等总体接近，但纹理不同</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="V4第二组"></details>
<details><summary>20260923：汽车、人物、鸟、蜗牛</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="V4第三组"></details>
<details><summary>20275001：乐器、树叶、树林、动物</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="V4第四组"></details>
<details><summary>20275002：动物、人物和局部边界</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="V4第五组"></details>
<details><summary>20275003：桌面、面包、人物和动物</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="V4第六组"></details>
<details><summary>20275004：V4联合相对V2联合的latent偏差增加，不能只选改善样例</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="V4第七组"></details>

### 2.10 续训1600步已完成：平均偏差略降，脸部问题仍在

V2 fine联合训练在100–800的八次验证都更新了best，且配对生成比同架构冻结对照平均偏差降低5%。因此，在V4没有额外收益后，保留更简单V2，只把训练预算延长到**总计1600步**，检验尚未收敛的fine主干能否继续改善完整生成。不是根据训练loss单独作决定，也不是在没有改善的V4上继续堆时间。

两组都从各自`latest800`恢复权重和Adam状态；另复制历史`best`，保留冻结组best100，避免把latest误当best。不改变rank、层、LR、数据、采样、精度或loss。新目录带`-continue1600`，原800步文件不变。每100步validation，结束后按best生成相同七组；若MSE改善仍不转化为图片改善，不自动继续到3200。

| 项目 | 当前续训 |
|---|---|
| 会话 | `rdd_csra_preserved_continue1600_0123_20260913` |
| 启动器 | `2d/exp/continue_csra_preserved_pair_to1600_4gpu.sh` |
| 总日志 | `logs/csra_preserved_continue1600_pair_20260913.log` |
| GPU / batch | 0123，每卡64，总256，无累积；启动前每卡至少25,000MiB空闲 |
| 训练顺序 | fine主干+adapter到1600，adapter-only对照到1600，然后配对生成 |
| 完成检查 | 两组1600步、七组完整生成全部完成，tmux正常退出；联合/冻结峰值allocated约9.74/4.53GiB/卡，无OOM |
| 结果边界 | 整体略改善但脸部未恢复，没有视频10%净提速结论 |

| 指标 | adapter-only best100 | fine联合800 | fine联合1600 |
|---|---:|---:|---:|
| fine验证MSE | 0.007132001 | 0.006330443 | 0.005923353 |
| 总验证目标 | 0.007353862 | 0.006529494 | 0.006096122 |
| 七组最终latent relative L2均值 | 0.1901036 | 0.1805941 | 0.1771461 |

冻结对照虽训练到1600，但最优仍为历史best100，预览也确实加载100。联合组选择best1600。相对冻结对照，1600组平均latent偏差降低约**6.82%**，7/7组都较低；相对自己的800步，仅再降低约**1.91%**，5/7组改善。不是FID，也不是6.82%的画质提升。

| seed | 联合800 | 联合1600 |
|---|---:|---:|
| 20260921 | 0.178075 | 0.180700 |
| 20260922 | 0.210202 | 0.206138 |
| 20260923 | 0.150083 | 0.146958 |
| 20275001 | 0.173503 | 0.165145 |
| 20275002 | 0.207935 | 0.198332 |
| 20275003 | 0.141903 | 0.138300 |
| 20275004 | 0.202459 | 0.204451 |

查看全部七张grid：很多动物、自然场景和物件保持合理，但白帽人物仍有红色脸部、表情/形状变化；20260921和20275004的偏差比800步更高。**这仍不足以宣布质量问题解决，所以不直接继续3200步。**

全部84张单图、7张grid、7份metrics和日志已下载本机；28张dense PNG与800步参考完全相同，14项coarse端点parity均为0。归档`eval_results/csra_preserved_continue1600_complete_20260913.tar.gz`，SHA-256 `3059fafc9744832e0a8ba2ddf46317d7a1ace3c629cb4790c112ad0082225b1f`。

下面三列均为**原dense / adapter-only best100 / fine联合best1600**：

<details open><summary>20260921：白帽人物仍失败，不能只看平均指标</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第一组"></details>
<details><summary>20260922：纹理略有变化</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第二组"></details>
<details><summary>20260923：汽车、人物、鸟和蜗牛</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第三组"></details>
<details><summary>20275001：自然场景总体较稳定</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第四组"></details>
<details><summary>20275002：动物和人物局部变化</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第五组"></details>
<details><summary>20275003：桌面和服装局部变化</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第六组"></details>
<details><summary>20275004：平均偏差相对800步增加</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步第七组"></details>

### 2.11 来源干预：当前adapter真的用到了coarse吗？

不改fine输入、不改模型、不训练，只改变四个adapter收到的coarse缓存。32条validation轨迹、每条8个fine状态，共256状态/组；这些数据已经参与选checkpoint，不是新测试集。每个状态比较正确来源、同图父格错位、换另一张图片、全零。两个单元测试验证干预不改动原缓存，排列保留原数值；运行时检查不消耗CUDA随机数。

| coarse来源 | adapter-only预测MSE | fine联合训练预测MSE | 联合组相对正确来源 |
|---|---:|---:|---:|
| 正确图片、正确父格 | 0.007680149 | 0.006848804 | 基准 |
| 同图片，但父格位置错位 | 0.007930114 | 0.007033790 | +2.70% |
| 另一张图片 | 0.008279598 | 0.007253954 | +5.92% |
| 全零 | 0.008700863 | 0.007543857 | +10.15% |

这说明当前网络不是完全忽略coarse来源；样本和空间对齐有贡献。但提升幅度有限，**不能推出一个充分训练的无coarse小网络一定更差**。换错缓存还可能是分布外扰动。为此完成了2.12节的充分训练对照：无coarse追平，因此已收缩当前跨尺度优势的主张，不能换名字包装普通block压缩。

脚本：`2d/attention_analysis/probe_csra_coarse_source.py`；原始结果：`eval_results/csra_coarse_source_20260913/metrics.json`，已下载本机。只在0卡执行，峰值allocated1.86GiB，未停止任何既有训练。这是局部预测诊断，没有FID、视频或速度结论。

### 2.12 对照已完成：无coarse版本追平，当前跨尺度优势未得到支持

新的输入干预不只在推理时发生，而是贯穿训练与采样。保持V2结构、四层、初始权重、数据、seed、LR、精度、batch、损失和1600步预算不变，让原coarse分支接收全零；另外两条当前fine分支照常工作，fine主干也共同训练。8/16生成仍用原模型，保证fine入口相同；这个实验比较的是**同一RDD内“有无跨stage缓存”**，不是拿不同生成路径混在一起。

参数张量形状和名义数量相同，但全零输入使`low_down.weight`的数据梯度为0；时间与偏置分支仍可学习。因此这是来源移除控制，不是宣称每个分支的有效表示能力都完全相同。两组沿用同一个已经见过coarse的V2初始化，可能偏向有coarse组；若无coarse组反而追平，这会构成重要负面证据；若它较差，仍需更多初始化/强无coarse架构对照，不能直接推导全分辨率模型无法适配。

运行时不再捕获16阶段的cache；fine替换层直接建立零输入，甚至不读取coarse值。新checkpoint格式`csra_preserved_source_control_v1`，防止旧采样器误当作普通coarse模型加载。旧参数默认仍为`aligned`，不改变原实验。

| 项目 | 设置 |
|---|---|
| 训练 | `train_csra_joint.py --arm preserved_joint --coarse-source zero`，从原base+V2初始化训练1600，不从joint1600接着训 |
| 启动器 | `2d/exp/train_csra_no_coarse_control_4gpu.sh`；先四卡每卡64的2步完整优化器smoke |
| 完成状态 | 37项回归测试、2步优化器smoke、1600步训练和七组生成全部完成，tmux正常退出。峰值allocated约9.77GiB、reserved约11.71GiB/卡，无OOM |
| 会话 / 日志 | `rdd_csra_no_coarse_control_0123_20260913`；`logs/csra_no_coarse_control_20260913.log`和`logs/csra_no_coarse_train_20260913.log` |
| 资源 | 0123，每卡64，总256；每卡至少25,000MiB空闲，不停止别人任务 |
| 输出 | `checkpoints/RDD-CSRA-preserved_joint-no_coarse-L400K-20260913` |
| 配对预览 | `sample_csra_preserved_preview.py --source-control`；七组同seed、DDPM80/80/90、CFG1、noise filling |
| 列顺序 | 原dense / 有coarse联合best1600 / 无coarse联合训练后validation best |
| 判断 | 先回答来源贡献，不把局部误差或跳层数当实际视频速度 |

| 指标 | 有coarse、best1600 | 无coarse、best1600 |
|---|---:|---:|
| validation主输出MSE，1024状态 | 0.005923353 | 0.005909712 |
| validation方差输出MSE | 0.003455395 | 0.003452540 |
| validation总目标 | 0.006096122 | 0.006082339 |
| 七组最终latent相对L2平均 | 0.177146145 | 0.176094666 |

无coarse平均偏差略低0.594%，4/7个seed更低；差异很小、没有独立重复训练，**不宣称无coarse显著更好**。真正重要的是：没有出现预期的跨尺度来源优势。这与2.11的推理时清零并不矛盾——已经适应coarse的网络被突然改输入会变差，不代表给无coarse网络充分训练后仍会变差。

| seed | 有coarse | 无coarse |
|---|---:|---:|
| 20260921 | 0.180700 | 0.180546 |
| 20260922 | 0.206138 | 0.215081 |
| 20260923 | 0.146958 | 0.141720 |
| 20275001 | 0.165145 | 0.168969 |
| 20275002 | 0.198332 | 0.191711 |
| 20275003 | 0.138300 | 0.145030 |
| 20275004 | 0.204451 | 0.189606 |

七组均已逐一查看：大部分物体与场景接近，白帽人物两者都有明显红脸/形状问题；无coarse在汽车等部分例子更接近参考，在狐狸朝向等例子偏离更大。不是统一胜负，也不是FID。不能用这些28张开发图片证明生成分布同样好或跨尺度信息在任何架构下都无用。

全部84张单图、7张grid、7份metrics、日志和完成标记已下载。28张dense与28张有coarse参考PNG的SHA-256均与上一轮完全一致；14项coarse端点parity均为0。归档`eval_results/csra_no_coarse_complete_20260913.tar.gz`，SHA-256 `7105ccae69af2c46a959b18a4cbcbf2475aa86e0d034f83b8596e11c891c1d6a`。

以下三列统一为**原dense / 有coarse联合1600 / 无coarse联合1600**：

<details open><summary>20260921：两种训练都没有修复白帽人物</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse完整训练第一组"></details>
<details><summary>20260922：水生动物与蝶翅</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse第二组"></details>
<details><summary>20260923：汽车、人物、鸟与蜗牛</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse第三组"></details>
<details><summary>20275001：场景与动物</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse第四组"></details>
<details><summary>20275002：动物和人物局部变化</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse第五组"></details>
<details><summary>20275003：无coarse的狐狸朝向偏离更多</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse第六组"></details>
<details><summary>20275004：小物体、鸟、杯子与动物</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="有无coarse第七组"></details>

**决策：停止给V2追加相同配置的训练预算。** 保留它作为负面对照，不删除记录。下一项检验第8节的主动全局记忆读取；若这种更直接的信息使用方式同样无法胜过无coarse控制，就进一步收缩这条路线，而不是只换名字。

## 3. 文献对筛选方向的约束

| 工作 | 已经做过的事 | 对本项目的影响 |
|---|---|---|
| HPDM，CVPR 2024 [1] | 将低层级 feature 作为高层级 context；缓存前面 pyramid stages 的激活；higher scales 只执行部分 RIN blocks | **很近的先例。** 跨尺度 feature caching、fine 阶段少算层都不能笼统声称首创 |
| Jenga，2025 [2] | Progressive Resolution + 当前 block 均值产生的稀疏 attention mask | “多分辨率加稀疏 attention”不是新组合；其 mask 不是直接拿已完成生成 stage 的整块更新 |
| VSA，2025 [3] | 当前 Q/K/V 的 coarse tile 分支加 fine sparse 分支，训练门控融合 | 只省 router 不会省掉 QKV、MLP、大部分 fine attention；公平对照不能偷偷删 coarse output |
| SparVAR，CVPR 2026 [4] | next-scale 自回归生成中，利用跨 scale attention 相似性构造 sparse 模式 | 图像领域也已有跨生成尺度的 attention 先例；不能只查 diffusion 文献 |
| Decomposable Flow Matching，2025 [5] | 多尺度分解与每尺度 velocity，前后尺度可以按不同速率继续生成 | 下文“内容/细节分开更新”必须直接对比它，不能当作无人做过的理论 |
| CAT Pruning，2025 [6] | 普通 diffusion 上 token pruning 与缓存 | 单纯“挑一些 token 少算”不符合这里的专属性要求 |
| Tortoise and Hare Guidance，NeurIPS 2025 [7] | 使用多速率积分减少 CFG 部分的计算 | 单纯“每部分不同更新频率”不是 RDD 独有；必须明确更新对象是跨尺度结构与新增细节 |
| InvarDiff，2025预印本 [9] | 所谓cross-scale指时间步和网络层/模块两个粒度的缓存，不是空间分辨率 | 不能只因标题相近就认定它已做coarse生成stage到fine生成stage；但普通跨步/跨层缓存不是我们的独占空间 |
| Helios，2026预印本 [10] | Pyramid UniPC执行低到高空间分辨率采样，另有历史压缩和蒸馏 | 低到高video本身已有直接先例；正文明确跨stage清空UniPC历史，不把它误写成已实现我们的feature-cache传递 |

HPDM 的具体区别值得注意：它缓存旧 pyramid context，供当前 patch 融合，并按尺度分配 RIN 计算；我们要检验的是已训练 RDD 单一轨迹里，上一 stage 的整块更新能否通过小型、时间相关的变换替代当前 fine block。HPDM 正文报告缓存带来约 40% inference speed improvement，附录同时承认 grid sampling、未缓存 tokenizer/context 限制了收益。不能不核对其百分比口径就把它换算成本项目的延迟下降率。[1]

## 4. 方向 A：联合训练的跨尺度整块替代

### 核心思路

coarse 阶段已经做过全局计算，fine 阶段的一些层使用它作为主要更新，再由当前 fine 特征修正；其余层仍完整执行，保留细节与条件交互。联合训练让“产出可复用特征”和“使用这些特征”共同适配。

这与普通 previous-timestep cache 的区别是信息来源、空间大小和搬运方式，不是“普通模型绝对不能缓存”。没有 coarse stage 的固定分辨率生成器不能直接免费取得这份缓存。若它额外运行一个 coarse 网络，新增成本必须算进去，且它已不再是原来的单尺度基线。

### 如何判断能否带来 10%–20%

设目标 fine blocks 原来占 RDD denoiser 时间比例 q，被替换比例 r，小网络连同 cache 搬运只需原 block 时间的 e，则理想延迟下降约为：

$$
\delta\approx q\,r\,(1-e),\qquad S\approx\frac{1}{1-\delta}.
$$

例如 **假设** q=0.8，替换 30 层中的 6 层，且 e=0.1，延迟减少约 14.4%，速度约 1.17×。这是算预算，不是已经测到的视频结果；q、e 都必须实测，不能只数被跳过的层。

当前四层 image pilot 先检查质量能否恢复。下一阶段才考虑更多层，并在实际 Wan block 中计入 adapter、条件处理、cache、插值和 kernel launch。不能拿 raw CSBR 跳层的速度替代带训练 adapter 的速度。

## 5. 方向 B：coarse 保留全局，fine 只精算必要区域

### 方法是什么

不是让所有 fine tokens 都做昂贵 block，再只把 attention pairs 删掉。设一个 coarse cell 对应四个空间子 token，或八个时空子 token。对容易的父格，使用 coarse 驱动的便宜更新；对复杂或不确定的父格，才让其子 token 进入当前 fine 的完整修正路径。

在这个设计中，省的包括一部分 fine QKV、MLP 和 attention，不仅是 QK/AV。全局 context 必须保留，不能因为删除了条件或 coarse 分支才快。

### kernel 要干什么

1. 使用已有 coarse 状态输出父格级别的计算预算；尽量整 tile 选择，不是杂乱的逐像素分支。
2. 将需要精算的子 token 打包成连续数组，使 GEMM 能真正减小矩阵，而不是 dense 算完乘 mask。
3. 保留 coarse 全局特征/其 K/V，再让 active fine queries 获取全局和局部信息。
4. 将细节修正写回相应子 token；其他 token 使用 coarse 更新与便宜修正。

这些操作都要计入端到端时间。普通全分辨率 pruning 可以复制“少算 token”这一部分，因此本方案只有在 coarse 全局替代确实改善相同预算下的质量时，才算满足研究目标。[2][4][6]

### 先做什么检查

新只读探针 `probe_csra_parent_selectivity.py` 使用 64 条已有验证轨迹、512 个 fine 状态。先测“如果知道正确答案，精算最需要的 25% 或 50% 父格，最多能覆盖多少局部 residual 错误”；再测不看 fine 答案、只按 coarse feature 能量选父格的效果，并与随机选择比较。

oracle 看过答案，不能部署，也不代表已经实现加速。若 oracle 本身没有明显集中性，这条路线就缺少空间；若 oracle 好、coarse 简单评分不行，则需要训练预测器，而不是硬说 coarse attention 自然会选对。局部 residual 好也不保证最终图像好，后续仍要跑完整轨迹。

### 本轮诊断结果

| 允许精算的父格比例 | 随机选择期望覆盖的局部错误 | 按 coarse 更新能量选择 | 看过 fine 答案的 oracle |
|---|---:|---:|---:|
| 25% | 25.00% | 28.80% | 33.13% |
| 50% | 50.00% | 53.93% | 58.78% |

结果来自 64 条 validation 轨迹、512 个状态、四个目标层的平均。确实比随机稍好，但**没有出现“少数区域集中了绝大部分误差”**。即使 oracle 选 25% 父格，也只覆盖约 33% 错误。因此不把简单 coarse 能量路由当作高收益方案直接上大规模视频实验。若继续这条路线，需要联合训练让容易区域真的能由 coarse 分支承担，不能指望现有模型天然具备强空间稀疏性。原始记录在 `eval_results/csra_parent_selectivity_20260913/metrics.json`。

## 6. 方向 C：让模型只为新增加的自由度做昂贵计算

### 简单例子

两帧可以表示成“平均内容 m”和“帧间差别 d”：

$$
x_1=m+d,\qquad x_2=m-d.
$$

如果 RDD 已经生成了平均内容，高分辨率阶段不一定还要用同样庞大的网络重复计算这份内容。可以让 coarse 分支维护全局/平均部分，较小 fine 分支负责新增的差别，必要时再用少量完整层融合。这不是把已有某一帧当成精确答案：平均池化得到的是平均，不是选中一帧。

一般写成 `x=Px+(I−P)x`，P 是平均池化再最近邻上采样。真正研究的问题是：能否用较便宜的小网格网络更新 P 部分，用更少的高分辨率层更新其余部分，并保留二者耦合？仅仅做一个 wavelet transform 后仍运行同样大的网络，不会免费减少算力。

### 为什么需要重新训练与数学核对

不能将 coarse 内容永久冻结，因为后面的细节可能反过来改变合理的整体预测。也不能假设两个频带数据独立，或把池化后的噪声仍当成同方差独立像素噪声。

对于 m 个细像素平均，噪声方差会缩小到原来的 1/m；如果希望系数坐标里仍是单位独立 Gaussian，应使用正确归一化的正交基。RDD 当前 noise filling 的 parent/child 噪声关系必须一并保留。这里需要另立显式实验配置，不改变现有 checkpoint 的默认采样。[5]

还有一个容易错误推导的点：RDD-DDPM 训练的目标不是简单的 pure noise。当前 `gaussian_diffusion_with_drift.py` 把目标改成：

$$
\widetilde\epsilon=\epsilon+\frac{\alpha_t\lambda_t}{\sigma_t}(P-I)x_0.
$$

记新增细节 h=(I−P)x₀，则其目标分量为：

$$
(I-P)\widetilde\epsilon=\frac{(I-P)x_t-\alpha_t h}{\sigma_t}.
$$

所以即使边界处 λ=1、forward 的细节信号被完全删除，也**不能**把“该频带边缘分布是 Gaussian”误解为“当前网络高频目标就是已知噪声，不需要预测内容”。后验与反向 drift 仍可能需要细节的条件期望。这条捷径不作为已成立的加速方法。

DFM 已经研究分尺度 velocity 和不同生成速率，因此本方向不是单凭分解公式就有 novelty。值得验证的是与 RDD 已训练路径兼容的低成本实现、coupling 保留方式，以及相对相同 RDD 基线是否真的更好。[5][7]

## 7. 工程优化 K：coarse 分支先在小网格计算，最后才放大

### 为什么可以精确改写

目前 adapter 的 coarse 分支先把 token 复制到 fine 网格，再依次做线性层、时间调节、SiLU、线性层。对最近邻复制而言，同一父格的所有孩子输入完全相同；这些逐 token 运算没有必要重复。

设 U 是最近邻复制，Fₜ 是对每个 token 使用相同参数的函数，则：

$$
F_t(Uc)=U F_t(c).
$$

这个等式对带偏置线性层、SiLU、逐样本相同的时间调节成立。**不对一般双线性插值或跨 token attention 自动成立。**

改成在 coarse 网格计算整条 coarse 分支，每个 fine step 仍使用当前时间，只把最后的输出复制出去。第一层与时间无关，推理时还可以在 stage 边界预先计算一次。训练时参数在变，不能跨 optimizer 更新复用这种投影。

### 自己实现的 kernel

`rdd_video/attention_analysis/csra_coarse_transport_kernel.py` 中，Triton kernel 直接读取父格结果并与当前 fine 修正相加，不产生一个完整的 upsample 临时张量。计算逻辑是：

```python
parent = parent_index[fine_token]
out[fine_token, channel] = coarse_output[parent, channel] + fine_update[fine_token, channel]
```

这不是新 attention kernel，也没有近似 attention。收益来自少做 coarse 重复投影、减少中间张量的读写，与融合算子的基本原则一致。[8]

### 测试口径

使用 Wan-1.3B 的 hidden 宽度 1536、rank64、batch1 的合成张量，测试三种视频 token 网格与空间/时空两类父子关系。固定输入、权重、FiLM 和 FP32/TF32 设置，预热后 100 次顺序交替测试。以下不是实际视频网络计时，其他任务仍在 GPU 上，属于初步算子证据。

对照包括：旧版放大后算、放大但也预投影缓存、小网格不缓存、小网格缓存加普通 PyTorch lift、小网格缓存加 Triton lift-add。这样不会把“只允许我们缓存第一层”当成优势。冷启动 BLAS 初始化与热运行的 setup 时间分开记录。

结果与原始 JSON：`eval_results/csra_coarse_transport_20260913/metrics_matched_tf32.json`。当前测得的 TF32 设置下所有比较输出逐元素一致；关闭 TF32 的独立检查存在约 3e-6 的最大浮点舍入差异，不能因此声称对任意设备、输入和执行模式永远 bitwise 相同。

| fine token 数 | 父格关系 | 放大网格也缓存第一投影：每调用 | coarse 缓存 + fused lift-add：每调用 | 90 次调用且包含双方各一次热 setup：分支加速 |
|---|---|---:|---:|---:|
| 11,648 | 空间约 4 个孩子/父格 | 0.1593 ms | 0.1285 ms | 1.25× |
| 11,648 | 时空约 8 个孩子/父格 | 0.1596 ms | 0.1209 ms | 1.33× |
| 46,592 | 空间 4 个孩子/父格 | 0.8003 ms | 0.5055 ms | 1.59× |
| 46,592 | 时空 8 个孩子/父格 | 0.8008 ms | 0.4524 ms | 1.78× |
| 72,000 | 空间约 4 个孩子/父格，包含边缘取整 | 1.2528 ms | 0.7821 ms | 1.61× |
| 72,000 | 时空约 8 个孩子/父格，包含边缘取整 | 1.2532 ms | 0.6943 ms | 1.82× |

初次执行的一次 BLAS 初始化达到约 89 ms；这不能混入“热 setup”后只扣在某一方法头上。第一版严格 FP32 探针也记录过约 118 ms 冷初始化，因此重测采用了分开记录冷初始化、双方热 setup 和交替重复的口径。**这些数字仍只对应 coarse 分支，不含 fine 分支和完整 denoiser。**

**这项优化依赖已有 coarse 分支，但也是其他金字塔结构可以采用的工程规则。** 它是 A/B 的配套优化，不能单独凑成“第三个只有 RDD 才能做的 10% 加速方法”。

## 8. 方向 D：保留当前 fine query，从旧 coarse 记忆重新读取

### 和之前的C2F-route、直接复用输出有什么不同

C2F-route只用旧coarse来挑“哪些fine块互相看”，fine QKV和大部分网络仍重新计算。CSBR则直接搬旧更新，当前fine子格的不同需求很难表达。这里尝试中间路线：**每个fine token现在重新提出问题Q，但答案库K/V来自已完成的coarse stage**，通过一个小型、时间相关的变换来适配当前阶段。

例如同一个父格复制出的四个子格，Q可以各不相同，所以它们能够从整个coarse记忆中读出不同的组合；不是给四个孩子完全相同的旧attention output。局部细节另用便宜的卷积/逐token修正和保留的完整层处理。视频的文字cross-attention不删除，全局context也保留。这是一种需训练验证的模块替代，不是数学等价的VSA kernel改写。[1][3]

```python
# coarse stage：已经发生的真实生成计算，按选中层保存状态
cache[layer] = project_kv(coarse_residual[layer])

# fine stage：当前子格产生不同问题，读取较小的coarse记忆
q = query_projection(current_fine_hidden, current_t)
k, v = small_time_correction(cache[layer], current_t)
global_update = attention(q, k, v)
local_update = cheap_fine_detail(current_fine_hidden)
h = h + trained_fusion(global_update, local_update)
# 其余完整层与文字条件交互仍执行；不是整模型都只看coarse。
```

只在部分层使用它。如果local分支仍为所有fine tokens计算完整K/V，就不能再宣称省掉了fine K/V投影；如果用局部卷积代替，必须训练并与相同质量预算的基线比较。

### 一个很重要的计算门槛：小K/V不保证比VSA-20少算

设fine有N个token，coarse有N/m个。这个读出需要约N²/m对QK/AV，而VSA-20的fine部分约为0.2N²。先忽略padding、VSA coarse分支和投影，得到：

| 一个父格对应多少孩子m | coarse读出pair数相对dense | 相对VSA-20 fine pair数 | 初步判断 |
|---|---:|---:|---|
| 4：空间长宽各放大2倍 | 25% | 1.25倍 | **attention主体反而更多**，不能直接宣布快 |
| 8：时间、长、宽各放大2倍 | 12.5% | 0.625倍 | attention主体有空间，需抵扣修正/搬运成本 |
| 16：空间长宽各放大4倍 | 6.25% | 0.3125倍 | 计算更少，但来源更粗、信息缺失更严重 |

实际还取决于Q/K/V宽度、head数、哪些层使用、RoPE与memory layout。VSA按tile取整后实际密度也要实测。这解释了为什么“少算一份coarse attention”的直觉不足以保证10%整网收益；更不能把上表pair比例当秒数。

### kernel具体可以优化什么

第一版优先使用成熟的非方形Flash/SDPA attention：Q长、K/V短。以相同张量形状和精度做匹配测试，再决定是否需要定制。可定制点是把时间变换后的coarse K/V组织为连续小tile，供多个fine-query tile重复读取；融合父格索引、必要的子格位置修正和输出写回，避免先复制出完整fine K/V。

不能宣称K/V一定常驻片上SRAM：容量不足时仍从更高层缓存/显存读取。不能把“跨step预投影”的setup漏计。缓存低精度、compile、offload策略也必须给基线相同机会。

### 为什么不能把父格直接当所有孩子的精确attention

如果某个父格的m个孩子真的具有完全相同的K、V，那么合并它们时，softmax logit需要加`log(m)`，以保留重复次数的权重；所有父格m相同则这个常数在softmax中抵消。边缘父格孩子数不同必须单独处理。这是代数性质，不是关于真实网络的假设。

**实际孩子的hidden不同，RoPE位置也不同**，因此真实fine K/V并不相同；旧coarse缓存还有timestep差异。上述合并不再精确，必须通过训练适配、保留fine修正以及实际采样检验。只修log(m)无法解决这些问题。

### 如何证明它不是普通缓存换名

按同一fine输入比较三种来源：已发生的RDD coarse stage、当前fine特征在线池化所得记忆、旧full-resolution timestep缓存压缩所得记忆。所有小变换都接受相同训练预算；在线池化和额外forward计入秒数。只有第一种在质量/总成本上有优势，才支持“利用已有coarse生成工作”的贡献；否则应放弃专属性叙事。HPDM等层级模型也能拥有类似上下文，因此绝对的“其他方法完全不能用”不成立。[1]

### V5小规模实现与匹配控制

2.12节表明旧V2没有得到充分训练后的跨尺度优势，因此本轮改变的是**信息读取方式**，不是继续延长V2。新增`coarse_memory_v5`保留原V2三条路径，另外让当前fine特征通过一个rank64、单头的learned query读取整张coarse residual记忆。不是直接复用原DiT的Q/K/V或attention probability，也不是从no-coarse结果推断它必然有效。

| 项目 | 配置与边界 |
|---|---|
| 输入 | fine 256 tokens；coarse原生64 tokens，H=1024，不把K/V复制成256 |
| 读取 | Q来自当前fine的LayerNorm；K/V来自原16阶段对应层residual的LayerNorm；三者有当前timestep的可学习scale/shift |
| 新增参数 | 四层总计1,284,096；adapter合计3,339,264，已由运行日志复核 |
| 初始化 | memory输出投影为0，V2暖启动输出应逐元素一致；两种来源控制使用相同确定性初始化 |
| 两组 | aligned原coarse / zero从训练开始就不给coarse；两组fine主干都联合训练 |
| 预算 | 各800步，四卡0123×64/global256；其余数据、LR、loss、精度、四层mask与前述相同 |
| 推理 | 同七个seed，原8/16模型固定，DDPM80/80/90、CFG1、noise filling；新参数只在32阶段四个替换层执行 |
| 代码 | `train_rdd_csra_adapter.py`的新opt-in架构，`train_csra_memory_pair_4gpu.sh`启动；旧默认与checkpoint行为保留 |
| 输出 | `checkpoints/RDD-CSRA-memory_v5-{aligned,zero}-L400K-20260913`；`eval_results/csra_memory_pair_20260913` |
| 完成验证 | 41项回归测试、两组smoke、各800步训练和七组配对生成全部完成。两组best均800，tmux正常退出；峰值allocated约9.89GiB/卡，无OOM |
| 已完成会话 / 日志 | `rdd_csra_memory_pair_0123_20260913`；`logs/csra_memory_pair_20260913.log`，分组日志`logs/csra_memory_v5_{aligned,zero}_20260913.log` |

局部父格路径仍保留空间对齐；新增全局读取没有单独加入子格RoPE，只能先检验内容寻址是否带来帮助。它新增一个rank64输出子空间，容量也变了，因此必须主要比较同V5的aligned/zero两组，不能仅因V5优于V2就归因于coarse。

本轮image pilot的adapter仍按原设定用FP32训练/推理，调用PyTorch SDPA；**不是已经完成的高性能视频kernel**。此前小网格lift-add的计时也不能代表这个新增模块。若质量与来源控制有优势，再测实际BF16视频实现、缓存setup/时间修正/搬运及两套权重的成本。

### V5实际结果：没有发现稳定来源优势，不继续堆800→1600

| 指标 | V5有coarse，best800 | V5无coarse，best800 |
|---|---:|---:|
| validation主输出MSE | 0.006343355 | 0.006351495 |
| validation方差输出MSE | 0.004079360 | 0.003988635 |
| validation总目标 | 0.006547323 | 0.006550926 |
| 七组最终latent偏差平均 | 0.181275583 | 0.179627776 |

有coarse的主输出MSE略低约0.128%，但无coarse的完整latent偏差平均略低约0.91%、4/7组更低。没有重复训练的误差条，不称任何一方显著更好。V5有coarse相对同预算V2有coarse800步（0.180594099）还略高约0.38%，没有观察到新增全局读取带来的质量突破。

七组图均已检查：许多物体仍接近dense，但人物脸部失败两组都存在。完整图像的变化与局部MSE方向也不一致。该结果反驳的是“这套具体结构已经有效利用旧coarse带来优势”，不是证明任何跨尺度记忆都没有价值。

| seed | V5有coarse | V5无coarse |
|---|---:|---:|
| 20260921 | 0.177931 | 0.177522 |
| 20260922 | 0.211268 | 0.221848 |
| 20260923 | 0.150686 | 0.140805 |
| 20275001 | 0.173320 | 0.176035 |
| 20275002 | 0.209637 | 0.201108 |
| 20275003 | 0.142477 | 0.149714 |
| 20275004 | 0.203611 | 0.190363 |

全部84张单图、7张grid、7份metrics、日志及完成标记已下载；28张dense PNG的SHA-256与原参考完全相同，14项coarse端点parity为0。归档`eval_results/csra_memory_complete_20260913.tar.gz`，SHA-256 `7bbcb883336aa8ef3e85344095d5df5922be3ead283355008e3d0e0bd878a4d3`。没有FID或视频净速度结论。

<details open><summary>20260921：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第1组"></details>
<details><summary>20260922：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第2组"></details>
<details><summary>20260923：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第3组"></details>
<details><summary>20275001：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第4组"></details>
<details><summary>20275002：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第5组"></details>
<details><summary>20275003：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第6组"></details>
<details><summary>20275004：原dense / V5有coarse / V5无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="V5来源对照第7组"></details>

### 分时间诊断已完成：来源收益确实主要集中在刚升分辨率时，但很小

没有继续给V2/V5扩训。在相同128条验证轨迹的1024个状态上，对V2和V5各自充分训练的aligned/zero版本，按t=299/259/215/171/128/84/40/0分别比较。对主输出预测误差做2×2平均再上采样，分成“局部平均误差”和“局部差异误差”，两者能量之和已通过检查。**这是预测误差分解，不是生成图像的高低频质量，也不是SNR。**

目的是检验：全程平均是否掩盖了刚升分辨率时的局部来源收益，是否有理由只在靠近边界的窗口复用。若没有这种信号，就不能假设缩短窗口必然成为RDD独有方法；若有，也还要重新做同预算训练和完整采样，而非选一个好看的时间点作结论。

实现`probe_csra_time_source_advantage.py`，启动器`probe_csra_time_source_4gpu.sh`；四个只读分片分别用0123，一张卡一个已有模型，不更新权重、不生成大量样本。输出`eval_results/csra_time_source_20260913/{v2_aligned,v2_zero,v5_aligned,v5_zero}/metrics.json`，保存逐轨迹逐时刻结果以便配对分析。

四组全部完成，43项测试通过，峰值显存约1.86GiB/卡，未影响其他训练；数据已下载。下表为 **100×(1−有coarse误差/无coarse误差)**，正数表示有coarse更好。每行使用128条相同轨迹。

| fine t（采样由大到小） | V2总预测误差减少 | V2局部平均误差减少 | V5总预测误差减少 | V5局部平均误差减少 |
|---|---:|---:|---:|---:|
| 299，刚进入fine | +2.616% | +3.625% | +3.131% | +4.319% |
| 259 | +1.658% | +2.242% | +1.877% | +2.551% |
| 215 | +0.154% | +1.100% | +0.424% | +1.466% |
| 171 | +0.042% | +0.536% | +0.218% | +0.901% |
| 128 | −0.366% | −0.060% | +0.011% | +0.084% |
| 84 | −0.422% | −0.369% | −0.136% | −0.150% |
| 40 | −0.497% | −0.696% | −0.064% | −0.316% |
| 0 | −1.853% | −4.344% | −0.849% | −5.073% |

**发现的pattern：** 两种结构在最初fine时刻有小幅来源收益，且更偏向局部平均部分；后半段基本没有优势。t299总误差减少的逐轨迹bootstrap95%区间为V2[0.592%,4.853%]、V5[0.819%,5.611%]。这是对同一训练seed、checkpoint选择validation的描述性区间，没有多重比较校正，也没有重复训练，不是最终显著性或泛化证明。脚本`summarize_csra_time_source.py`对整条轨迹配对采样，完整结果`paired_summary.json`。

这给“更靠近边界时coarse更有用”一点支持，但2%–3%的单步误差变化还不能修复人物，更不代表视频提速。只把复用缩短到开头，反而会减少能省掉的fine计算量。因此暂不把“短窗口”包装为新加速方法。

### V6完整coarse hidden state对照已完成：仍无稳定来源优势

V2/V5的缓存都是`Δ=h_after−h_before`。这一轮改用真正的`h_after`，它还包含进入此层以前累积的特征；它也可能含有噪声，不能预先认定更好。**没有把h冒充Δ直接加到fine上。** 当前fine query经过新的低秩读取分支，从原16阶段的h得到修正，输出仍是fine block的更新。

为了公平，两组都从已训练1600步的V2无coarse学生出发，旧分支继续接收零来源；新增h读取分支的末端投影初始化为零，单位测试证明两组初始输出与这个V2模型严格相同。这样不会一开始就让有coarse组占初始化便宜。随后两组一起更新实际执行的fine主干和全部adapter，唯一来源差别为真实h或零。

| 项目 | 配置 |
|---|---|
| 两组 | `hidden_memory_v6` aligned / zero；新分支结构与参数量相同 |
| 初始化 | 相同`RDD-CSRA-preserved_joint-no_coarse-L400K-20260913/best.pt`，实际1600步；新分支相同随机种子、零输出 |
| coarse | 原Unified-L400K的16×16,t300各选中block输出h；8/16保持原权重 |
| fine | 另存细阶段权重；继续训练实际执行主干及adapter；替换层[13,15,17,20] |
| 数据 / 训练 | 同4096/128轨迹、每轨迹8状态；每组最多新增800步；0123每卡64，总256，无梯度累计 |
| 优化 | backbone LR1e−5、adapter LR1e−4；AdamW、WD.01、clip1；主输出MSE+.05方差输出MSE |
| 新缓存 | 只重放已有16阶段输入的一次forward，不重新采样；逐文件检查旧Δ与整模型输出完全相同，再保存h |
| 测试 / 保护 | 46项测试通过；每组先4卡完整优化器2步smoke；启动前要求每卡至少25000MiB可用，不停止其他任务 |
| 后台 | `rdd_csra_hidden_pair_0123_20260913`已正常结束；主日志`logs/csra_hidden_pair_20260913.log` |
| 结果位置 | `eval_results/csra_hidden_pair_20260913/seed_*/`；相同七组dense对照已全部完成并下载 |

两组各新增800步和七组完整生成全部完成，后台正常结束。四个hidden缓存分片逐文件parity为0，46项测试和两组四卡优化器smoke通过，无OOM。两组初始化validation MSE均为0.005909711675485596，与共同V2零来源起点完全一致。

| 指标 | 有完整coarse h | 训练时无来源 |
|---|---:|---:|
| validation选中新增步数 | 700 | 800 |
| fine主输出validation MSE | 0.005689661 | 0.005663390 |
| 方差输出validation MSE | 0.003285524 | 0.003563695 |
| 选择用总目标 | 0.005853937 | 0.005841575 |
| 七组平均final latent相对L2 | 0.174397288 | 0.173949312 |

无来源平均latent偏差低约0.257%；有来源仅3/7组更低，不能宣称显著胜负，但**没有支持完整h带来稳定收益**。两组相对共同起点0.176094666都略有改善，说明继续优化能降低部分误差，不等于收益来自coarse。白帽人物的脸部与肤色仍明显偏离dense，其他动物/物体/风景大体保留。当前适配器原型没有达到完整质量目标，不继续自动追加V6预算。

| seed | 有h | 无h |
|---|---:|---:|
| 20260921 | 0.177076 | 0.178547 |
| 20260922 | 0.214898 | 0.212249 |
| 20260923 | 0.142829 | 0.142450 |
| 20275001 | 0.165122 | 0.166939 |
| 20275002 | 0.188871 | 0.189502 |
| 20275003 | 0.144478 | 0.142493 |
| 20275004 | 0.187508 | 0.185466 |

全部84张单图、7张grid、metrics、日志和完成标记已取回，七张grid逐一检查；28张dense PNG与V5原参考SHA-256完全相同，14项coarse端点parity为0。归档`eval_results/csra_hidden_complete_20260913.tar.gz`，SHA-256 `2c7314450d1c6ffdbdadec3ab64b0baba5aad4b9451c9b4bba26a97397391fc5`。

<details open><summary>20260921：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照1"></details>
<details><summary>20260922：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20260922/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照2"></details>
<details><summary>20260923：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照3"></details>
<details><summary>20275001：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20275001/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照4"></details>
<details><summary>20275002：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20275002/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照5"></details>
<details><summary>20275003：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20275003/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照6"></details>
<details><summary>20275004：原dense / 完整h / 无来源</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden来源对照7"></details>

该实验是“检查完整h是否含有旧Δ缺少的可用信息”，不是已经成功的第三种方法。依旧有两套权重显存代价，FP32小网络还未作为视频kernel优化；只有完整生成有改善才进入视频净速度测量。

### 保留fine MLP、仅复用coarse K/V：只读诊断与三组预览全部完成

此前V2–V6同时替掉了整层attention和MLP。新的只读检查使用原RDD-L400K，不更新权重，保留当前fine MLP、调制和gate，只改变四层attention的K/V。当前Q不变。它回答“已有coarse信息是否更适合直接提供attention记忆，而不是由很小的网络包办整个block”。

| 分支 | K/V从哪来 | 新增处理 |
|---|---|---|
| `cached_native` | 原16阶段t300的K/V | K保持原coarse RoPE |
| `cached_center` | 同一旧K/V | K换到对应fine孩子中心的坐标；不改变内容 |
| `cached_recondition` | 原coarse归一化hidden | 用当前fine时间/类别调制后，在小网格重新算K/V；这不是零开销复用 |
| `current_parent` | 当前fine K/V按2×2平均 | 同样64个key的同时间控制；不是VSA，不含VSA双分支gate |

四个分片共32条原validation轨迹、256个状态/分支。先检查重写的dense路径与原模型逐位相同，再记录原teacher状态上的最终输出误差。**诊断故意仍计算完整QKV以保持Q的数值完全一致，因此没有加速计时，也不是可直接宣称更快的kernel。** 它只用于决定是否值得训练/优化真正的Q-only缓存路径；若结果差，不直接扩大训练。

实现`probe_rdd_native_coarse_kv.py`，启动器`probe_csra_native_kv_4gpu.sh`，后台`rdd_csra_native_kv_0123_20260913`，输出`eval_results/csra_native_kv_20260913/shard_*/metrics.json`。这个方法也可以被其他真正有coarse stage的网络采用，不作数学排他性声明。

四片全部完成，48项回归测试通过，实际模型256个状态上的dense重写parity为0，数据已下载：

| 分支 | 主预测MSE，256个状态 | 相对同时间pooling控制 |
|---|---:|---:|
| 旧coarse K/V，原坐标 | 0.01574693 | 约2.36倍误差 |
| 旧coarse K/V，孩子中心坐标 | 0.01566411 | 约2.35倍误差 |
| coarse hidden改用当前调制 | 0.01898685 | 约2.85倍误差 |
| 同时间fine K/V池化 | 0.00666103 | 1倍 |

单纯移动RoPE坐标只微调误差；只换时间调制反而更差，说明不能用一个简单“时间校正”就认为旧内容变成了当前内容。该结果没有通过training-free误差筛选，不能作为已成功的速度方案。

为看清局部误差是否会在采样累积，只追加三组既有seed（20260921/20260923/20275004），每组4张，五列：dense、旧KV全90fine步、旧KV仅前20步、同时间pooling全90步、同时间pooling仅前20步。保留全部fine MLP、相同80/80/90和noise filling；第21步开始恢复原dense attention，不改变采样步数。它是少量失败/保守配置可视化，**不是新的训练，更不是将“前20步”当已达到10%加速**。

后台`rdd_csra_native_kv_preview_012_20260913`已完成，脚本`sample_csra_native_kv_3gpu.sh`，日志`logs/csra_native_kv_preview_20260913.log`，输出`eval_results/csra_native_kv_preview_20260913/seed_*/`；只使用012，启动前检查12GiB余量，保持其他任务运行。

| seed | 旧coarse KV，90步 | 旧coarse KV，前20步 | 当前KV池化，90步 | 当前KV池化，前20步 |
|---|---:|---:|---:|---:|
| 20260921 | 0.298097 | 0.150181 | 0.194350 | 0.077336 |
| 20260923 | 0.234396 | 0.104299 | 0.149757 | 0.062133 |
| 20275004 | 0.266598 | 0.182847 | 0.191704 | 0.093483 |
| 三组均值 | 0.266364 | 0.145775 | 0.178604 | 0.077651 |

表中仍是相对原dense的**final latent相对L2偏差**，越小表示轨迹更接近，不是FID、感知质量百分比或速度。四层全90步共替换360次attention；前20步只替换80次，余下70步恢复原模型，不能把两者称为同计算预算。

三张grid均逐一检查。全90步的旧KV会明显改变人物脸部、动物毛发和物体边缘；前20步保留了更多dense结构，白帽人物也明显好于长时间复用。这是有用的**保守窗口可行性现象**，但同窗口的当前KV池化在三组均更接近dense，而且池化不是无需新计算的coarse缓存。因此既不能说旧缓存完全无用，也不能说它已经胜过在线方法。没有在这些预览中计时，不能把20/90的窗口直接换算为视频提速。

全部60张单图、3张grid、3份metrics和4份日志已下载；12张dense PNG与V6原参考SHA-256相同，15项coarse端点parity为0。归档`eval_results/csra_native_kv_preview_complete_20260913.tar.gz`，SHA-256 `764b9f2643b9815d995f62689f025b4bfb54b5a0f0bc821c5be544f5023ca4ce`。

下列五列依次为：**原dense / 旧KV全90步 / 旧KV仅前20步 / 当前KV池化全90步 / 当前KV池化仅前20步**。均为同一原RDD-L400K、同seed/类别、80/80/90、CFG1、noise filling；没有针对这些图微调。

<details open><summary>20260921：长时间复用影响脸部；短窗口保留得更多</summary><img src="../../eval_results/csra_native_kv_preview_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:1280px" alt="原生KV复用窗口对照1"></details>
<details><summary>20260923：汽车、人物、动物和蜗牛</summary><img src="../../eval_results/csra_native_kv_preview_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:1280px" alt="原生KV复用窗口对照2"></details>
<details><summary>20275004：物体、鸟、玻璃杯和毛发</summary><img src="../../eval_results/csra_native_kv_preview_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:1280px" alt="原生KV复用窗口对照3"></details>

### 前缀信息探针已完成：在fine尚未重建全局特征之前使用coarse

V2–V6第一次替换的是0-based第13层；到它之前已经执行了13个原fine block。当前fine输入可能已经包含这些昂贵计算重建的信息，所以此时再加入coarse没有优势，并不直接回答“能否一开始就靠coarse省掉这13层”。本轮已经实现并完成这个不同切入点的探针；它是拟合特征预测器，不更新原teacher，也没有训练整个生成学生。

直观方案：已有16阶段的第13层输出 → 便宜的跨尺度转换，与当前fine的patch输入、时间/类别结合 → 预测32阶段第13层输出 → 继续执行最后10个真实fine block。这样fine仍有完整后半段处理文字/类别、细节和时间变化，不是把旧hidden直接当作当前hidden。真正节省的是前14个fine block，必须扣除转换网络与缓存成本。

实际方法很简单：把当前输入patch特征、当前时间/类别向量和可选的coarse特征拼接，拟合一个线性变换来预测“从输入到第13层输出的更新”，再加回当前输入。这个线性变换比14个Transformer block简单得多，但这里只检查它有没有获得额外信息，没有以省FLOPs推断画质可用。

| 项目 | 实际配置 |
|---|---|
| 教师与位置 | 原RDD-Unified-L400K；coarse是16阶段t300、第13层完整输出；目标是32阶段第13层完整输出，层号均从0开始 |
| 当前信息 | 原模型的patch embedding h0（1024维）及原时间/类别条件c（1024维），不是已经算过13层的fine特征 |
| 额外信息 | coarse完整h（1024维），nearest对应到fine孩子；不改teacher的RoPE或stage transition |
| 拟合数据 | 原训练集选192条轨迹拟合、另外64条校准；原validation128条仅用于报告；每条8个时间状态 |
| 变换 | 全通道仿射ridge回归，训练集中心化/标准化，截距不惩罚；候选正则0.01/0.1/1，仅按64条校准集选择 |
| 参数量 | 无coarse 2,098,176；其他四组均3,146,752。增加1,048,576参数，必须结合等参数来源控制解释 |
| 数值 | FP32关闭TF32的中心化矩阵乘法，FP64合并统计量与求解；防止近常量条件向量导致相减消去误差 |
| 运行 | 0123采集，GPU0合并拟合；52项回归测试通过，所有原模型输出与缓存逐位parity为0 |
| 显存 | 特征采集峰值allocated约1.03GiB/卡；统计/拟合进程约0.72GiB，不是训练整主干的显存 |
| 可复现文件 | `probe_csra_prefix_information.py`、`csra_prefix_probe_math.py`、`probe_csra_prefix_information_4gpu.sh`；指标在`eval_results/csra_prefix_information_20260913/` |

初次启动因未创建输出子目录而失败，已经添加目录创建后重新跑完；首次错误日志保留。没有改变数学计算或数据，也没有OOM或停止其他进程。四片完整采集共384条轨迹/3072个时间状态；划分按整条轨迹，所有分片合并后检查seed无重叠，校准不使用报告validation。

| 预测器的来源 | validation hidden MSE | 选中的正则 | 与正确来源比较的意义 |
|---|---:|---:|---|
| 只有当前patch+时间/类别 | 3.931971 | 1.0 | 没有额外来源的基线，但参数较少 |
| 加入正确且位置对齐的coarse | **3.395780** | 0.1 | 最低，误差比无来源少13.64% |
| 同张图的coarse在网格上错移一格 | 3.709374 | 1.0 | 来源维度/参数相同，保留同图信息；正确对齐仍少8.45%误差 |
| 换成batch内另一张图的coarse | 3.916140 | 1.0 | 来源维度/参数相同，但内容/可能的类别都错；几乎追平无来源 |
| 只给同图coarse的空间平均 | 3.761925 | 0.1 | 保留同图全局/类别信息，没有空间分布；正确来源仍少9.73%误差 |

错移/换图/均值组都是**从拟合开始就使用这种来源并独立选正则**，不是只在推理时突然扰动造成OOD。均值输入秩较低，等参数张量不代表等有效容量；此探针也没有学习非线性的时间/特征交互，所以不能据此断言线性结构就是最优结构。

**最有价值的pattern：改进主要来自低频/全局部分。** 对hidden预测误差按2×2局部平均分解，正确coarse使局部平均误差从2.244328降至1.740780（−22.44%），局部差异误差仅从1.687643降至1.655000（−1.93%）。这是hidden误差的空间分解，不是图像FFT、FID或“低分辨率训练更容易”的证明。它支持继续测试“coarse提供整体更新、保留当前fine计算处理细节”的结构。

| fine t | 无来源hidden MSE | 正确coarse hidden MSE | MSE减少 |
|---|---:|---:|---:|
| 299 | 2.061057 | 1.690571 | 17.98% |
| 259 | 2.097354 | 1.693514 | 19.25% |
| 215 | 2.292332 | 1.840148 | 19.73% |
| 171 | 2.797132 | 2.274022 | 18.70% |
| 128 | 3.689018 | 3.085327 | 16.36% |
| 84 | 5.035447 | 4.359045 | 13.43% |
| 40 | 6.869150 | 6.112940 | 11.01% |
| 0 | 6.614277 | 6.110670 | 7.61% |

按16个原batch seed配对整簇bootstrap（每簇8张、保留错配控制的批内依赖，2000次），总MSE减少的描述性95%区间为12.54%–14.74%。这里只拟合一次，原validation此前也被多个实验用过，没有独立测试集或重复训练，区间不是最终论文显著性声明。完整逐样本结果与重采样规则见`paired_summary.json`、`summarize_csra_prefix_probe.py`。

**不能跳过的限制：** 正确来源的hidden RMS相对误差仍为36.4%，无来源为39.2%；这不是先前表格的final latent误差。13.64%的预测MSE减少不代表画质恢复，更不代表视频提速13.64%。随后已经把拟合结果接入原fine后10层，使用精确teacher hidden作为oracle，1024个验证状态的oracle主模型输出与dense逐位相同。这个只读实验不更新主干、不生成新图、不做速度声明。

| 接入原后10层的前缀预测器 | 主输出MSE | 方差输出MSE |
|---|---:|---:|
| 无来源 | 0.09844676 | 0.43439724 |
| 正确coarse | **0.07720054** | 0.34395316 |
| 同图coarse空间均值 | 0.09400347 | 0.41091225 |

正确来源的主输出MSE比无来源少21.58%、比空间均值少17.87%。这说明来源优势没有在原后缀中消失，但主输出的RMS相对误差仍约35.4%，远不能当成可用采样器；t0处正确来源还略差于无来源，因此也不说每个时刻都受益。

采集/拟合会话`rdd_csra_prefix_information_0123_20260913`与后缀检查`rdd_csra_prefix_suffix_0_20260913`均已完成。后缀只用GPU0，峰值allocated约1.06GiB；结果已下载至`eval_results/csra_prefix_information_20260913/suffix_probe.json`，SHA-256 `b6903fcd11ee9b099466164f014370be2fa7116fb73a35cb66ce3c752c7f6eb5`，日志`logs/csra_prefix_suffix_20260913.log`在服务器。

指标、四片manifest、完整日志和首次目录错误日志均已取回；大特征张量与ridge权重保留服务器以供后续，不重复下载。归档`csra_prefix_probe_results_20260913.tar.gz`，SHA-256 `452658b5910fb08576fbd403f555e52c4c3666ec0402a7e7bf16a2ab6406bdfd`。

这是“复用低分辨率已经完成的前缀计算”，不是普通同分辨率复用的换名；但其他金字塔模型也有同类信息，[HPDM的层级特征缓存](https://arxiv.org/html/2406.07792v1)尤其需要正面对比，不能宣称绝对只有RDD能做。

### V7前缀转换器与fine后缀联合训练：完成，有来源优势但画质失败

这轮不继续扩训V6，而是实际改变计算位置：用一个全通道仿射转换器替代fine前14个block，保留后10个真实block。转换器输入当前patch、当前时间/类别与原16阶段完整h，输出fine第13层的替代hidden；末尾10层负责当前分辨率下的非线性修正。这个配置很激进，不预设800步就能修好图像。

两组**都从同一个无来源ridge函数出发**，新增coarse权重初始化为0。不是把上表各自已经拟合好的有来源/无来源函数直接拿来比赛，因此不会把不同起点当训练收益。两组初始validation主MSE完全相同：0.0984540986828506；第一步同一rank的loss也完全相同。与只读probe的极小差异来自转换成普通仿射参数及BF16端点舍入，不将它称作逐位等价于旧probe；两训练组之间的初始等价已测。

| 项目 | 配置 |
|---|---|
| 两组 | `prefix_v7-aligned`使用真实coarse；`prefix_v7-zero`从训练起无来源；同样的转换器参数张量和优化预算 |
| 原模型 | Unified-L/2 400K、300/300/400；8/16权重固定另存；不是把已更新fine权重用于coarse |
| coarse来源 | 原16阶段t300、第13层完整输出h；不是Δ或attention probability |
| fine执行 | patch与时间/类别embedding → 转换器 → 第14至23层 → 输出头；前14层不执行，不宣称训练了被跳过的参数 |
| 实际可训练参数 | fine主干193,384,480；转换器3,146,752。包括当前embedding、后10层和最终输出头 |
| 初始化 | 原fine主干权重 + 相同无来源ridge仿射参数；coarse列为0；两组没有不同的coarse预训练起点 |
| 数据 / 预算 | 原4096训练、128验证轨迹×8状态；每组最多新增800步；离线教师轨迹，不是on-policy |
| 优化 | AdamW；主干LR1e−5、转换器LR1e−4；WD.01、clip1；主输出MSE+.05方差输出MSE，无额外hidden loss |
| batch / 精度 | 0123，每卡64，总256，无梯度累计；FP32主参数和转换器，主干BF16 autocast；训练用activation checkpoint，不是拆microbatch |
| 测试 / headroom | 54项测试；两组各完成四卡2步真实反向/优化器smoke后才启动正式训练；每卡先要求25,000MiB空余 |
| 实际显存 | 峰值allocated6.14GiB、reserved6.93GiB/卡；核对时仍约30.4GB可用，无OOM，其他训练未停止 |
| 完成情况 | 两组各800/800，best均800；三组原seed各4张的完整生成已完成，不重复启动 |
| 预览设置 | 原dense/有来源/无来源，80/80/90、CFG1、noise filling；所有90次fine NFE均替代前14层；原coarse端点parity为0 |
| 会话 / 日志 | `rdd_csra_prefix_v7_pair_0123_20260913`正常结束；`logs/csra_prefix_v7_pair_20260913.log`与分组`logs/csra_prefix_v7_{aligned,zero}_20260913.log` |
| 代码 / 输出 | `csra_prefix_student.py`、`train_csra_prefix_pair.py`、`sample_csra_prefix_preview.py`；`train_csra_prefix_pair_4gpu.sh`；新checkpoint格式`csra_prefix_linear_v7`避免旧采样器误用 |

有/无来源相同参数张量仍不等于相同有效容量：无来源的coarse列没有信息梯度。最终仍需看来源对照、完整图片和可部署成本。当前参考实现仍把coarse先展开再做矩阵乘法，没有做新的视频融合kernel；如果画质可用，才把coarse线性变换移到小网格、按stage缓存，并测包括两套权重与所有转换成本的净视频延迟。**不能把跳过14层直接写成视频加速58%。**

| 完成后的指标 | 正确coarse来源 | 从训练开始无来源 |
|---|---:|---:|
| best新增步数 | 800 | 800 |
| validation主输出MSE | **0.028560863** | 0.034461967 |
| validation方差头MSE | 0.024976003 | 0.028403351 |
| 选择用总目标 | 0.029809663 | 0.035882134 |
| seed20260921：final latent相对L2 | 0.322588 | 0.344838 |
| seed20260923：final latent相对L2 | 0.281698 | 0.335929 |
| seed20275004：final latent相对L2 | 0.303335 | 0.327285 |
| 三组平均final latent相对L2 | **0.302540** | 0.336017 |

正确来源的validation主MSE减少17.12%，三组平均latent偏差减少9.96%，三组方向一致。但是三组共12个生成实例、只有一组训练随机种子，且这些是已反复查看的开发seed，不能称为独立质量评估或统计显著性证明。**latent偏差不是FID，不是“画质提高9.96%”，更不是“提速9.96%”。**

实际看图：正确来源能保留更多汽车主体、鸟的轮廓和玻璃杯边缘，但人物面部/身体仍严重变形，狗和麝牛形态也明显偏离dense；无来源通常更崩坏。14层替代远比之前四层分散替代激进，因此不能拿V7与V6的偏差直接归因于“前缀比中间层差”。当前同时改变了替代位置、数量和预测器结构。

下面每张图从左到右是 **原dense / 正确coarse+联合训练 / 无coarse+同预算联合训练**，每行同seed同类别。保留全部结果，不只挑汽车等较好的行。

<details open><summary>20260921：人物严重变形，14层替代未通过质量检查</summary><img src="../../eval_results/csra_prefix_v7_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V7前缀14层原dense、有来源、无来源对照1"></details>
<details><summary>20260923：coarse保留了更多汽车结构，但人物仍失败</summary><img src="../../eval_results/csra_prefix_v7_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="V7前缀14层原dense、有来源、无来源对照2"></details>
<details><summary>20275004：鸟和杯子有来源更接近dense，仍有明显形态损失</summary><img src="../../eval_results/csra_prefix_v7_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="V7前缀14层原dense、有来源、无来源对照3"></details>

36张单图、3张grid、3份metrics、训练与预览日志已下载，三张grid逐一检查；12张dense PNG与原生KV预览参考SHA-256完全相同，6项coarse端点parity为0。归档`eval_results/csra_prefix_v7_complete_20260913.tar.gz`，SHA-256 `e21029c7497a9c4c1ba0b68877b605d3b12aec8527765556dc5b03211bd0b9c9`。不继续盲目扩训这个14层版本。

### 8层前缀已完成：部分改善，但仍未达到质量目标

改成“替代fine前8层，保留后16层”，仍从原16阶段t300的第13层完整h读取信息。**来源层仍为13，目标层现在为7，均从0计数**；这是用coarse较深的上下文帮助fine较浅的前缀，不是假装两个网络在同深度。原14层ridge的目标不对，所以这轮重新采集h_after7并拟合；代码会拒绝目标深度不匹配的初始化/恢复/采样。

四片已完成：同192拟合/64校准/128开发验证轨迹，原teacher输出和插入真实h_after7的suffix oracle逐位parity为0。拟合探针hidden MSE为1.796869→1.554155；接入原后16层的主MSE为0.10857668→0.07937066。这仍是只读预测结果。**8层版本的初始最终输出误差并没有小于14层版本**：更早的特征目标、不同的后缀会改变误差传播，不能仅凭“少跳几层”保证训练后质量。

| 项目 | 新实验配置/实际状态 |
|---|---|
| 主干 | 原Unified-L400K；8/16保持原权重，fine另存更新；patch/时间/类别embedding、后16个block和输出头均训练 |
| 实际可训练参数 | fine主干306,722,848 + 转换器3,146,752 |
| 两组初始化 | 同一个新拟合的无来源函数，coarse列全0；两组初始主MSE完全相同0.10857929894700646 |
| 优化与数据 | 同4096/128轨迹×8状态、AdamW、主干1e−5/转换器1e−4、WD.01、clip1、主MSE+.05方差MSE；离线教师轨迹 |
| 预算 | 两组各1600次更新已完成，best均1600；比14层的800步预算大，不能把跨版本变化单独归因于层数；同一8层配对内预算完全相同 |
| GPU / batch | 仅0123、每卡64、总256、无梯度累计；两组均先通过四卡2步真实优化器smoke；55项回归测试通过 |
| 实际显存 | smoke峰值allocated7.83GiB；正式训练allocated7.89GiB、reserved9.17GiB/卡，仍约28.1GB余量；其他约58.6GB训练仍在运行，未OOM |
| 后台 | `rdd_csra_prefix8_pair_0123_20260913`正常结束；两组1600步和三个seed完整生成全部完成，不重复启动 |
| 日志 | `logs/csra_prefix8_pair_20260913.log`；分组`logs/csra_prefix8_{aligned,zero}_20260913.log` |
| 脚本 | `2d/exp/train_csra_prefix8_pair_4gpu.sh`，含新目标采集、五组ridge、suffix parity、两组smoke、训练和预览 |
| 新结果位置 | `eval_results/csra_prefix8_information_20260913/`；`eval_results/csra_prefix8_pair_20260913/seed_*/`；checkpoint是`RDD-CSRA-prefix8-{aligned,zero}-L400K-20260913` |

| 最终指标 | 正确coarse来源 | 从训练开始无来源 |
|---|---:|---:|
| best新增步数 | 1600 | 1600 |
| validation主输出MSE | **0.018073375** | 0.020541827 |
| validation方差头MSE | 0.017167454 | 0.018575913 |
| 选择用总目标 | 0.018931748 | 0.021470623 |
| seed20260921：final latent相对L2 | **0.302247** | 0.347003 |
| seed20260923：final latent相对L2 | **0.255217** | 0.258549 |
| seed20275004：final latent相对L2 | 0.282082 | **0.271805** |
| 三组平均final latent相对L2 | **0.279849** | 0.292452 |

有来源validation主MSE低12.02%，平均latent偏差低4.31%，但方向不是三组一致：第三组无来源反而更接近dense。仍只有三个旧开发seed、12个配对实例和一次训练随机种子，没有FID、独立大样本质量或视频计时。

逐图看：第一组第三行，正确coarse组恢复出了完整可辨认的人脸，而无来源仍是红色变形区域；但这张脸的外貌、帽子和笑容明显不同于dense，不能写成身份或类别保真已解决。第一组第二行、第二组第二行的人物都仍严重失真。狗、汽车、杯子等更像可辨认物体，但细节/几何改变；第三组的鸟、麝牛也不能说有来源稳定更好。这比14层版本保守，却不构成已经成功的方法。

三列仍依次为 **原dense / 正确coarse+联合训练 / 无coarse+同预算联合训练**。原Unified-L400K、80/80/90、CFG1、noise filling；原8/16权重不变，全部90个fine NFE都用8层前缀替代。

<details open><summary>20260921：部分人脸恢复，但仍有明显失败和身份/帽子变化</summary><img src="../../eval_results/csra_prefix8_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="8层前缀1600步原dense、有来源、无来源对照1"></details>
<details><summary>20260923：车辆保留较多结构，人物仍失败</summary><img src="../../eval_results/csra_prefix8_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="8层前缀1600步原dense、有来源、无来源对照2"></details>
<details><summary>20275004：该组无来源latent偏差反而更低</summary><img src="../../eval_results/csra_prefix8_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="8层前缀1600步原dense、有来源、无来源对照3"></details>

36张单图、3张grid、3份metrics和完整训练/预览日志已下载，全部grid已检查；12张dense PNG与14层版本SHA-256相同，6项coarse端点parity为0。归档`eval_results/csra_prefix8_complete_20260913.tar.gz`，SHA-256 `c79bac755a3c80bb7e2508b81502a3a8455525b68c2e795e62cedbb06e7ced28`。

这轮仍不是视频结果。不能把8/24层的省略比例当成净视频提速，也不能只用平均latent偏差较低判定画质合格。下一步应先检查当前前缀转换器的限制，而不是原样追加步数：它对coarse使用的是固定线性变换，时间/类别只通过加法进入这一转换器；其后16层会非线性交互，因此不能说整个模型不具备时间适应能力。可做按时间和频带的来源贡献检查，再决定是否需要时间条件化的非线性转换器、保留少量真实early fine block，或减少替代深度。这些是待验证选择，不是新的已运行训练。

8层前缀的完整探针metrics、suffix检查、四片manifest、日志与两组smoke记录已下载并校验；大特征张量/权重仍留服务器。归档`eval_results/csra_prefix8_probe_results_20260913.tar.gz`，SHA-256 `1959815377dd404bcbb70e71368e645a591455b7871d5ab6008b2b53782e3ec2`。

### 已训练8层前缀的分时间、空间误差诊断

本轮直接读取两组best1600，在同128条开发轨迹的8个teacher状态上检查，不重新训练、不生成新teacher。四片共1024状态/组，58项回归测试通过，诊断峰值仅1.85GiB/卡。主MSE复算0.01807613/0.02054236，与训练时的0.01807338/0.02054183非常接近；诊断batch8、训练validation batch64的BF16舍入略有不同，不能声称逐位相同。

| fine timestep（生成顺序从上到下） | 正确coarse：主MSE | 从训练开始无coarse：主MSE | 正确来源相对少多少误差 |
|---|---:|---:|---:|
| 299 | 0.003253 | 0.004961 | 34.44% |
| 259 | 0.005271 | 0.007415 | 28.92% |
| 215 | 0.010173 | 0.012807 | 20.57% |
| 171 | 0.018226 | 0.021740 | 16.16% |
| 128 | 0.027283 | 0.030776 | 11.35% |
| 84 | 0.035067 | 0.038196 | 8.19% |
| 40 | 0.039362 | 0.042276 | 6.89% |
| 0 | 0.005975 | 0.006168 | 3.13% |

**支持的判断：coarse信息的相对优势更靠近升分辨率的时刻。** 但这不是因果分离时间差、噪声水平和当前任务难度的实验；三者同时变化，不能只归因于“缓存变旧”。t0目标本身功率很小，较低绝对MSE也不代表它最容易。

将每个2×2 latent小格的误差分成“格内平均”和“减掉平均后的差别”，二者平方能量可以严格相加。它是局部正交分解，**不是理想FFT低通，也不能直接等同图像语义/纹理**。正确来源的平均误差为0.003811，格内差异误差0.014265；但对应目标功率也很不相同。除以各自目标功率再开方，两部分相对误差为**17.05% / 17.14%**，整体非常接近。因此不能只因约79%的原始误差落在格内差异上，就断言失败几乎都来自高频。较晚t128、t84确有更高的差异部分相对误差，但不是全部时刻都如此。

推理时把已训练aligned的coarse清零/平移，主MSE变成0.038024/0.035895，比正常0.018076更差，说明学生确实使用了来源和空间对应；它们是分布外干预，**不能替代上表从训练开始就无来源的公平对照**。不比较训练后hidden与原teacher hidden的直接MSE，因为主干联合更新后内部坐标可能改变。

原始指标：[完整逐状态JSON](../../eval_results/csra_prefix8_trained_diagnostic_20260913/metrics.json)。脚本`2d/attention_analysis/diagnose_csra_trained_prefix.py`；归档SHA-256 `5e2bd302f376af90f3aee205a06cd180ea381e6c2fe5f9385829ef0ba7a8ed19`，已下载校验。

### 固定短窗口预览：有改善，但没有通过质量筛选

**没有训练新模型。** 使用上面两组线性prefix8 best1600，仅在32阶段最开始的45或60次NFE执行前缀替代，后面的45或30次恢复原始完整fine模型。8/16阶段、总80/80/90 NFE、CFG1、noise filling、seed/类别不变。不是减少采样步数，不是降低分辨率，也不是根据结果动态退回dense；窗口预先固定，两组完全相同。它会减少可省的计算，因此不能拿全90步的理论节省配短窗口的图片。

| 复用窗口（共90 fine NFE） | seed20260921：有/无来源latent相对L2 | seed20275004：有/无来源 | 两seed均值：有/无来源 |
|---|---:|---:|---:|
| 前45步 | 0.265411 / 0.309500 | 0.243598 / **0.223415** | 0.254504 / 0.266458 |
| 前60步 | 0.295422 / 0.340602 | 0.263349 / **0.245821** | 0.279386 / 0.293212 |
| 全90步，已有参照 | 0.302247 / 0.347003 | 0.282082 / **0.271805** | 0.292165 / 0.309404 |

缩短到45步对两组都减少偏离，却仍只有一个seed支持来源优势。四张grid已逐一检查：45步的狗更接近dense，鸟和杯子的几何也相对保守；白帽人物的aligned脸仍改变外貌并丢掉帽子，无来源脸仍严重变形；纹身人物仍出现不合理的人脸/身体拼接。**不能把缩短窗口视为已经修复画质。** 也没有对这些预览测净视频速度。

每张图三列都是 **原dense / 有coarse / 从训练开始无coarse**；每列4张图。

<details open><summary>seed20260921：仅前45步替代（较保守，但人物仍有失败）</summary><img src="../../eval_results/csra_prefix8_window45_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="前45步prefix8，有来源与匹配零来源对照"></details>
<details><summary>seed20260921：仅前60步替代</summary><img src="../../eval_results/csra_prefix8_window60_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="前60步prefix8，有来源与匹配零来源对照"></details>
<details><summary>seed20275004：仅前45步替代（无来源latent偏差更小）</summary><img src="../../eval_results/csra_prefix8_window45_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="第二seed前45步prefix8配对"></details>
<details><summary>seed20275004：仅前60步替代</summary><img src="../../eval_results/csra_prefix8_window60_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="第二seed前60步prefix8配对"></details>

4份metrics、48张单图和4张grid均已下载；16张dense图SHA-256与原prefix8参照一致，8项coarse端点parity为0，每组实际复用次数经脚本断言。60项测试通过。归档SHA-256 `a0460eba7398be20c601c7c2efd0bb52aad3f1a472b6b8eec47edd2fcc5659f4`。

### 条件化前缀转换器：两组训练与三组预览已完成

前面的时间规律不能证明问题只在仿射转换器，但给了一个有界的下一步：**使转换器能在fine刚开始时，按当前内容和时间重新读取coarse，而不是只做固定的线性相加。** 原来的后16层已经有非线性，因此不是说旧模型完全没有时间适应能力。新的模块本身是常见attention/MLP；研究问题是把它放在尚未执行昂贵fine前缀的位置，是否能用现成coarse信息替代这些计算。不是宣称新发明了cross-attention，也不是重复V5的晚层rank64 residual读取。

```text
原16阶段完整h：64 tokens，边界处保存一次 ── 当前时间/类别调制 ── K、V
当前32阶段patch输入：256 tokens ─────── 当前时间/类别调制 ── Q
                                                               ↓
旧的仿射前缀预测 + 新的Q读取K/V得到的修正 + 当前patch的非线性修正
                                                               ↓
                    仍跳过fine第0–7层，执行第8–23层和输出头
```

新增attention是256 query ×64 native coarse key，16 heads×64维；不先生成256个coarse K/V，也不计算新的fine-to-fine attention。时间/类别生成Q与coarse输入各自的逐通道scale/shift；另外增加1024→2048→1024的SiLU前馈分支。仍保留原来的空间对齐仿射分支，所以新attention没有另加位置/RoPE变换。每个fine NFE重新计算Q/K/V和attention，**不是StepReuse-Out、不是复用attention probability、不是C2F稀疏route**。KV网格较小不等于没有额外成本，视频测速必须扣除这些新增操作。

| 项目 | 实际配置 |
|---|---|
| 名称 / opt-in参数 | 条件化前缀转换器；`--bridge-kind conditional_attention`；旧默认仍是`linear` |
| 基模与跳层 | Unified-L400K 300/300/400；fine前8层替代，后16层真实执行；8/16仍完全原权重 |
| 两组 | 有正确coarse完整h / 从训练开始coarse恒零；两组相同模块和参数张量、数据、更新数 |
| 初始化 | 同一fine-only ridge；新增attention输出层、MLP末层都置零，初始输出与旧线性起点完全相同；不是从两个不同best模型继续训 |
| 初始实测一致性 | 两组validation主MSE均0.10857929894700646；第一步主MSE均0.1192389503121376 |
| 可训练参数 | fine主干306,722,848；转换器15,740,928（旧线性3,146,752） |
| 优化 | fine1e−5 / 整个转换器1e−4，AdamW WD.01，clip1，主MSE+.05方差MSE；种子20279000 |
| 数据 / 预算 | 同4096训练、128开发验证轨迹×8状态；离线teacher蒸馏；两组各1600步，100步验证/保存 |
| GPU / batch | 0–3，共4卡；每卡真实64，总256，无梯度累计；BF16 AMP+FP32主参数，activation checkpointing |
| 已验证 | 62项回归测试；两组四卡2步smoke；梯度连通；无OOM；峰值allocated8.45GiB、reserved约9.85GiB/卡，正式运行时仍约27.4GB余量 |
| 最终状态 | 两组各1600步，best均1600；三组完整预览完成；无OOM、无报错 |
| 训练tmux | `rdd_csra_prefix8_conditional_0123_20260913`已正常退出，不重复启动 |
| 总日志 | `logs/csra_prefix8_conditional_pair_20260913.log`；两组日志`logs/csra_prefix8_conditional_{aligned,zero}_20260913.log` |
| launcher | `2d/exp/train_csra_prefix8_conditional_pair_4gpu.sh` |
| checkpoints | `checkpoints/RDD-CSRA-prefix8-conditional-{aligned,zero}-L400K-20260913/`，新格式`csra_prefix_conditional_v8` |
| 已完成的自动预览 | seeds20260921/20260923/20275004，各4图、dense/有来源/无来源三列；全90 fine NFE使用新转换器；`eval_results/csra_prefix8_conditional_pair_20260913/seed_*/` |

这轮同时改变转换器容量与非线性/读取结构，因此与旧线性版本的改善不能单独归因于某一个新模块；真正的来源证据仍看新配对内的正确coarse/恒零。恒零组相同参数张量不意味着相同有效信息秩。其他原有训练未被停止或改动。

| 完成结果 | 正确coarse来源 | 从训练开始恒零来源 |
|---|---:|---:|
| best更新数 | 1600 | 1600 |
| validation主MSE | **0.016173843** | 0.018354508 |
| validation方差头MSE | 0.015123063 | 0.015864356 |
| 验证总目标 | 0.016929996 | 0.019147726 |
| seed20260921：final latent相对L2 | **0.300925** | 0.344510 |
| seed20260923：final latent相对L2 | **0.235945** | 0.265334 |
| seed20275004：final latent相对L2 | **0.266632** | 0.267171 |
| 三组均值 | **0.267834** | 0.292338 |

正确来源主MSE低11.88%、平均latent偏差低8.38%。相对旧线性prefix8，有来源平均偏差从0.279849降至0.267834（减少4.29%），无来源几乎没变（0.292452→0.292338）。但两个模型的teacher-state主MSE都因升级而下降约一成，不能把这种下降全部算成新增coarse收益。仍只有3个旧开发seed、12个实例、1个训练随机种子，第三组的来源优势只有约0.2%，不足以宣称统计显著或FID改善。

**实际看图仍不合格：** 第二组的汽车结构相对保留较好；第一组正确来源的人脸可辨认而零来源红色塌陷，但外貌/帽子改变。第一组纹身人物的脸部和身体、第二组桌面后的人物仍发生严重变形或消失；第三组鸟的头部/姿态和杯子的形状也改变。仅仅生成了不同外貌本身不一定意味着分布质量下降，但这些明显的局部形状破坏不能靠平均latent偏差掩盖。

三列依次为 **原dense / 条件化前缀＋正确coarse / 同架构＋恒零coarse**。都是原Unified-L400K、80/80/90、CFG1、noise filling，全部90个fine NFE使用新转换器。

<details open><summary>条件化前缀：seed20260921，人物仍有明显失败</summary><img src="../../eval_results/csra_prefix8_conditional_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="条件化prefix8，原dense有来源无来源，seed20260921"></details>
<details><summary>条件化前缀：seed20260923，汽车较保守但人物未修复</summary><img src="../../eval_results/csra_prefix8_conditional_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="条件化prefix8，原dense有来源无来源，seed20260923"></details>
<details><summary>条件化前缀：seed20275004，两组latent偏差非常接近</summary><img src="../../eval_results/csra_prefix8_conditional_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="条件化prefix8，原dense有来源无来源，seed20275004"></details>

全部36张单图、3张grid、3份metrics、训练/smoke/预览日志与完成JSON均已下载校验；12张dense图SHA-256与原prefix8参照一致，6项coarse端点parity为0，metadata确认两组best1600、正确bridge类型和全90 fine NFE。归档`eval_results/csra_prefix8_conditional_complete_20260913.tar.gz`，SHA-256 `cf64431e9d912dc82cf740481adda018b99323df89f3185a578df178aa629c26`。

### 小型轨迹诊断已完成：不支持“学生状态上单步误差爆炸”这个简单解释

用相同seed/噪声分别让原teacher和prefix学生完成轨迹；在8个相同t上，都以原teacher对**各自当前状态**的预测作目标，比较学生误差。这里不是拿学生在自己的状态上的预测，去对比另一个状态的teacher输出。额外teacher调用不改变采样输出、不消耗CUDA RNG；原8/16端点逐位一致，四次完整轨迹复现原预览的final latent相对L2，记录差异均为0。

两个开发seed20260921/20275004、各4张，aligned/zero各两条完整轨迹；不是新的训练、不新增解码图片。62项测试通过，峰值2.687GiB/卡。结果如下，每格都是这8张的平均主预测MSE：

| t | aligned：teacher状态 | aligned：学生状态 | 学生/teacher状态误差比 | zero：学生/teacher状态误差比 |
|---|---:|---:|---:|---:|
| 299 | 0.002929 | 0.002929 | 1.000 | 1.000 |
| 259 | 0.004949 | 0.004109 | 0.830 | 0.827 |
| 215 | 0.008179 | 0.006022 | 0.736 | 0.804 |
| 171 | 0.017852 | 0.011298 | 0.633 | 0.702 |
| 128 | 0.028002 | 0.015798 | 0.564 | 0.630 |
| 84 | 0.033362 | 0.022382 | 0.671 | 0.724 |
| 40 | 0.036734 | 0.025169 | 0.685 | 0.710 |
| 0 | 0.005558 | 0.004579 | 0.824 | 0.845 |

在这些样本上，实际学生状态的单步主预测误差反而较低，但状态与dense的偏离仍逐步累积。**所以不能直接把失败归因于“离线状态没见过，采样就误差爆炸”，也不能因此断言on-policy训练必然修好。** 小样本不排除其他分布变化或稀有失败；较低单步MSE也可能伴随偏向更容易预测的状态，不能当成全局稳定性/质量证明。

数据：`eval_results/csra_prefix8_conditional_rollout_shift_20260913/seed_*_{aligned,zero}.json`。四份JSON和日志均已下载，归档SHA-256 `ca73c75bf0563d1869a773f6b71500a91d063aa9d35463e8fe7d0b3c2aa580ed`。

### 头部互换已完成：当前采样不使用网络方差头

两seed、四列：原dense / 学生两个头 / 学生主头＋teacher方差头 / teacher主头＋学生方差头。只在32阶段混合输出通道，8/16、80/80/90、CFG1和noise filling不变。每次teacher都在**当前这条混合轨迹的状态**上重新计算，而不是使用旧dense轨迹缓存。学生为条件化prefix8、有来源、1600步。

| 最终latent相对dense的L2 | seed20260921 | seed20260923 |
|---|---:|---:|
| 原dense | 0 | 0 |
| 学生两个头 | 0.3009247184 | 0.2359447479 |
| 学生主头＋原teacher方差头 | **0.3009247184** | **0.2359447479** |
| 原teacher主头＋学生方差头 | **0** | **0** |

不是“方差头影响很小”，而是**在实际采样调用链中没有影响**。此前本节只根据`p_mean_variance`内的learned-range分支推断采样使用方差头，这个解释不完整，现更正如下：

```text
sample_fid_unified._run_stage(sampler="ddpm")
  → jump_step_sample_loop(deterministic=False)
    → p_mean_variance：确实计算learned-range variance与pred_xstart
    → jump_step_sample：只取pred_xstart，另用解析Sigma加入随机噪声
                        不读取上一步返回的learned variance
```

所以“网络配置learn_sigma=True”和“这条采样路径实际使用方差预测”不是同一个问题。普通`p_sample`会使用它，不能将这里的结论推广到所有DDPM实现。主头仍是吸收drift的噪声预测。训练中的`.05×方差头MSE`是当前采样下的**辅助目标**，仍可能通过共享参数影响主头；不能据此断言删掉它没有训练影响。本轮没有改损失、采样器或旧checkpoint。

额外确认：最终32阶段local t0的jump是identity，`_sample_batch_unified`最后返回`sample`而非`pred_xstart`，因此这次fine模型调用也不改变最终latent。8/16阶段的`pred_xstart`却参与后续上采样，不能一起省去。这里记录现状，不偷偷改变历史实验的终点定义，也不把省一次调用包装成大幅加速。

<details open><summary>头部互换：seed20260921；第2/3列完全一致，第1/4列完全一致</summary><img src="../../eval_results/csra_prefix8_conditional_head_oracle_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:1024px" alt="主头方差头互换四列，seed20260921"></details>
<details><summary>头部互换：seed20260923；同样是精确一致，而不是肉眼近似</summary><img src="../../eval_results/csra_prefix8_conditional_head_oracle_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:1024px" alt="主头方差头互换四列，seed20260923"></details>

32张单图、2张grid、2份metrics和日志均已下载。32对图像SHA-256比较通过，含两种互换的相等性，以及与旧预览dense/学生的重放一致性；6项coarse端点parity为0。人物恢复只出现在额外运行原teacher主头的oracle列，**不能将该列的画质与学生的速度拼在一起**。

原始输出`eval_results/csra_prefix8_conditional_head_oracle_20260913/`；归档SHA-256 `c02c49b5ee85d3257f8c6a9d98ea30cdacf911253593aae3a6fb2a1741c54174`。`rdd_csra_prefix8_heads_01_20260913`正常结束；脚本`2d/exp/probe_csra_prefix8_heads_2gpu.sh`。

### CPU审计：主头误差经过采样器后，影响并非处处相同

新增`2d/attention_analysis/audit_csra_jump_head_sensitivity.py`，不加载模型、不占GPU：给实际jump采样器注入已知主头扰动，逐一检查全部90个时间点。方差头从−1改到+1时，`p_mean_variance`的log variance确实变化（中间点最大差0.0343046），但固定随机种子的jump输出始终逐元素相同。

主头扰动分成2×2格子内的平均部分和格子内部差别。因为当前jump的上采样/池化是线性的，在**相同输入状态**下，它们进入下一状态的增益可以分别测量：

| 全局t | 格子平均误差的增益绝对值 | 格子内部差别的增益绝对值 |
|---|---:|---:|
| 299 | 0.023379 | 0.007918 |
| 259 | 0.029589 | 0.013594 |
| 215 | 0.021162 | 0.012235 |
| 128 | 0.026018 | 0.020323 |
| 40 | 0.018502 | 0.017460 |
| 0 | 0 | 0 |

人话：fine刚开始时，同样大小的主预测误差，格子平均部分对下一步的影响更大；后面两者逐渐接近。将这两个增益应用到已有学生状态的误差分解后，早期下一状态误差偏平均部分，后期偏格子内部差别。这与“coarse在早期更有用”相容，但**不是证明全部失败都由高频造成，也不是最终图像误差的因果分解**。不同时间的状态误差会继续经过非线性网络传播，不能把本表的平方直接相加当作FID，更不能据此推出最优分辨率边界。

90个时间点的随机混合扰动都与两分量线性响应相符，最大绝对误差5.48e−7。记录`eval_results/csra_prefix8_jump_sensitivity_20260913_metrics.json`包含实际源代码SHA、全部时间点以及已有256条开发诊断记录的转换结果。`tests/test_csra_jump_head_sensitivity.py`锁定这些旧采样行为，防止以后误用配置字段推断行为。

### 有界续训：只验证1600步是否尚不充分，不改变架构

| 主MSE趋势 | 800步 | 1200步 | 1600步 |
|---|---:|---:|---:|
| 有coarse | 0.018467 | 0.017052 | 0.016174 |
| 恒零coarse | 0.022147 | 0.019768 | 0.018355 |

两组都持续改善，故设置**各额外1600更新，总上限3200**的预算对照，而不是无限追加。保持同一架构、数据、优化器状态、种子、学习率、辅助方差损失与采样器；从各自1600步继续，不能将此解释为新的方法贡献。原始两个1600步目录保留不变，新的`conditional3200`目录存续训结果，best不改善时保留1600步参照。

脚本`2d/exp/continue_csra_prefix8_conditional_to3200_4gpu.sh`：0–3四卡，实际64/卡、总256，无梯度累计，25GiB空闲显存与30GiB磁盘余量检查；不停止其他任务。先有来源再无来源；结束自动跑三个原开发seed的完整配对预览。新输出`eval_results/csra_prefix8_conditional3200_pair_20260913/`；checkpoint `checkpoints/RDD-CSRA-prefix8-conditional3200-{aligned,zero}-L400K-20260913/`。

**2026-09-13已完成：** 两组均到3200步，best均为3200；恢复1600步时验证与父checkpoint完全一致。66项测试通过，运行峰值allocated8.454GiB、reserved最高9.178GiB/卡，无OOM。tmux `rdd_csra_prefix8_conditional3200_0123_20260913`正常结束；总日志`logs/csra_prefix8_conditional3200_pair_20260913.log`。

| 结果 | 有coarse：1600 | 有coarse：3200 | 无coarse：1600 | 无coarse：3200 |
|---|---:|---:|---:|---:|
| 验证主MSE | 0.016174 | 0.014664 | 0.018355 | 0.015920 |
| seed20260921：latent相对L2 | 0.300925 | 0.299344 | 0.344510 | 0.336793 |
| seed20260923：latent相对L2 | 0.235945 | 0.226358 | 0.265334 | 0.247401 |
| seed20275004：latent相对L2 | 0.266632 | 0.272131 | 0.267171 | 0.254287 |
| 三组均值 | 0.267834 | **0.265944** | 0.292338 | **0.279494** |

续训使有/无来源验证主MSE分别低9.33%/13.27%，但生成偏差仅分别低0.71%/4.39%。3200步有来源主MSE比无来源低7.88%，生成偏差均值低4.85%；第三个seed无来源更接近dense。有来源的纹身人物仍发生大幅构图/形状改变，白帽人物仍变成另一张脸，杯子的下半部变圆、鸟的头部/姿态偏移。无来源白帽人物对应位置呈哨子状物体。这些观察说明**成对保真度仍不够好**；不能把0.71%写成FID或画质提升，更不能说继续训练已经解决问题。

三列仍是**原dense / 3200步有coarse / 3200步无coarse**；模型、采样与上表1600步完全同配置，只有额外更新数改变。

<details open><summary>3200步：seed20260921</summary><img src="../../eval_results/csra_prefix8_conditional3200_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="3200步原dense有coarse无coarse，seed20260921"></details>
<details><summary>3200步：seed20260923</summary><img src="../../eval_results/csra_prefix8_conditional3200_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="3200步原dense有coarse无coarse，seed20260923"></details>
<details><summary>3200步：seed20275004，这组无coarse偏差较小</summary><img src="../../eval_results/csra_prefix8_conditional3200_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="3200步原dense有coarse无coarse，seed20275004"></details>

全部36张单图、3张grid、3份metrics、两组complete及日志均已下载；12张dense图SHA与1600步参照一致，6项原8/16端点parity为0，metadata核验best3200及90个fine调用。归档SHA-256 `e5c47969036a1c2f10c8873a751cc71997dca4106ec19cd59afe2d91b6638042`。

### 完整90时间点检查：没有发现训练点之间大规模失效

训练和验证只记录了8个固定时间，但生成使用90个。因此补查1600/3200×有/无来源四个模型，在相同原teacher轨迹的**全部90个fine时间点**计算预测误差。两个既有开发seed、每seed4图；共8份JSON、2880条逐图逐时间记录。额外学生调用不消耗CUDA RNG，原teacher最终latent和8/16端点均逐位一致；峰值显存2.687GiB，68测试通过。不是新的训练或视频测速。

图中圆点是8个训练时间；上两行分别是原始主MSE和除以teacher目标功率的误差；最后一行将各时间误差除以相邻两个训练点误差的直线插值。**这条插值只是一条画图参照线，不是正确误差应当服从的理论。**

![8个训练点与全部90个采样点的预测误差](../../eval_results/csra_prefix8_all_times_20260913/all_time_error_curves.png)

| 未训练时间误差/相邻训练点插值（先合计误差再取比值） | 1600有coarse | 1600无coarse | 3200有coarse | 3200无coarse |
|---|---:|---:|---:|---:|
| 全部未训练点 | 1.146 | 1.118 | 1.167 | 1.132 |
| 排除特殊的0–40区间 | 1.066 | 1.042 | 1.090 | 1.056 |
| 仅0–40区间 | 1.606 | 1.587 | 1.635 | 1.651 |

有温和的训练点间起伏，续训后有来源模型的这个相对差距略扩大。但不存在横跨全部时间的倍数级暴涨。最大约2.4×的逐点比值集中在0–40区间，这里原始误差曲线本来就强烈弯曲、t0目标功率也不同；t0主头对最终sample又没有作用，不能用这个比值证明模型“没学过中间t所以崩坏”。这不排除时间覆盖有优化空间，但**不足以把它确定为主要原因或承诺补时间点就能修好**。

数据和图均已下载：`eval_results/csra_prefix8_all_times_20260913/`；归档SHA-256 `ca1792f5b23a15be75519166d31420e5e4d00c94e0e8895e2bb641e0b4da8973`。脚本`diagnose_csra_prefix_all_times.py`与`summarize_csra_prefix_all_times.py`；后台诊断已正常结束，不重复运行。

### 评价口径修正：构图保真、类别一致性、分布质量要分开

核对真实采样label后发现：seed20260921的四类是**Brabancon griffon / punching bag（沙袋）/ whistle（哨子）/ buckle（扣环）**，不是“狗/纹身人物/白帽人物/圆物体”四个类别。另一个seed是hook/toucan/beaker/ox。因此之前把人物消失直接称作质量失败过于笼统：它明确是与dense构图的偏离，但人物可能不是条件要求的主体。模型仍可能把一个合理的含人物构图变成不自然物体，也可能生成了另一种有效构图，需要进一步区分；不能只看像不像dense。

用已有ResNet50 ImageNet1K V2、官方预处理、CPU FP32，对这两组原图及两档训练版本检查类别。只读取已保存PNG和轨迹中实际记录的label，没有重猜CUDA标签、没有下载新权重。

| 8张旧开发图上的诊断 | 原dense | 1600有coarse | 1600无coarse | 3200有coarse | 3200无coarse |
|---|---:|---:|---:|---:|---:|
| Top-1命中数 | 3 | 4 | 2 | 3 | 2 |
| Top-5命中数 | 5 | 5 | 4 | 4 | 4 |
| 平均目标类别概率 | 0.1360 | 0.1126 | 0.0898 | 0.1190 | 0.0980 |

8张远不足以比较模型质量；分类器也可能对不自然图片给高分。这里只纠正“人物不一样=类别错了”的推理，**不据此反过来宣布画质已经达标**。旧章节的质量筛选应理解为保真度与目视筛选，不是FID意义上的分布结论。记录`eval_results/csra_prefix8_preview_semantics_20260913/metrics.json`保留逐图预测、目标名称和图片SHA。

### 已完成：256个新样本的匹配小评估，不再依赖几张旧图

固定3200步模型，原dense/有coarse/无coarse各生成256图，共768图。四个新seed `20360913 + rank×10000`，与保存的训练/开发语料seed检查不重合；每卡生成64个匹配样本，三方法共用labels、初始噪声、80/80/90、CFG1和noise filling，原8/16端点必须逐位相同。VAE按4图解码以控制显存，**不是训练microbatch**。仅0–3，启动前22GiB余量门限，进程内18GiB检查，不触碰其他任务。

保存每张PNG和真实label、逐图类别概率/Top-5、latent偏差。各shard固定展示前8张，顺序在看图前确定；不挑选好看的例子。这是新seed的开发pilot，不是最终独立测试集，更不是256图的数字就可以替代50K FID。分类只检查条件语义，后续仍需分布/真实感评估和实际视频测速。

脚本`2d/exp/sample_csra_prefix_quality256_4gpu.sh`；tmux `rdd_csra_prefix8_quality256_0123_20260913`已正常退出；总日志`logs/csra_prefix8_quality256_pair_20260913.log`。四片完整、768张PNG均逐一通过记录SHA校验，8项原coarse端点parity为0，两个学生均确认best3200。68项测试通过，峰值allocated为3.607GiB/卡，无OOM。没有改训练或模型权重。

用同一个ResNet50 ImageNet1K V2、官方预处理、CPU FP32评估。256张覆盖222个不同目标类别；它不是每类等数量的ImageNet测试集。

| 指标 | 原始RDD dense | 正确coarse＋训练后的前缀替代 | 无coarse＋同配置训练的前缀替代 |
|---|---:|---:|---:|
| Top-1命中 | 188/256，73.44% | 156/256，60.94% | 143/256，55.86% |
| Top-5命中 | 234/256，91.41% | 209/256，81.64% | 193/256，75.39% |
| 平均目标类别概率 | 0.30383 | 0.26083 | 0.21671 |
| 四片batch级latent relative L2的均值 | 0 | 0.27742 | 0.31039 |

**如何读：** 正确来源比无来源Top-5高6.25个百分点，但仍比原RDD低9.77个百分点。按256个匹配图像对bootstrap 10,000次，前者差值的描述性95%区间为`+1.56～+10.94`个百分点，后者为`−14.45～−5.08`个百分点。有/无来源Top-1差值是`+5.08`个百分点，但区间`−0.39～+10.55`包含0。这些是固定checkpoint、四个RNG分片下的开发pilot区间，未做多重比较校正，不反映训练seed变动。**不是FID，也不是质量或速度提升6.25%。** latent一行是四个batch范数比的算术平均，不是256个逐图relative error的均值。

全部四张固定grid已经逐张查看，每张展示前8个匹配样本。三列从左到右都是：**原始RDD / 正确coarse学生 / 无coarse学生**。主要观察：

- 猴子、意面、蝴蝶、勺子、鸟、船和部分狗仍可辨认；正确来源组通常更接近原结构。
- 第1片的橱柜细节消失，车辆轮廓扭曲；第2片房车轮子/车体在无来源组受损更明显，蘑菇斑点与竖琴弦也改变。
- 第4片多张狗脸拼图在两学生中都塌成连片纹理；拿西葫芦的人脸/手部受损，正确来源组相对好些，但并未解决。
- 不能只展示前几种稳定对象宣称成功，也不能因第4片人物变形就认定所有类别都崩坏。

<details open><summary>新样本第1片，固定前8张：蜘蛛猴、橱柜、意面、半履带车、哨子、被子、橄榄球头盔、瞪羚</summary><img src="../../eval_results/csra_prefix8_quality256_20260913/shard_0/comparison_grid.png" style="max-width:100%;width:768px" alt="新seed第1片固定8例，原始RDD、正确coarse、无coarse"></details>
<details><summary>新样本第2片，固定前8张：操纵杆、石鳖、竖琴、房车、蘑菇、马林巴琴、海葵鱼、英国古代牧羊犬</summary><img src="../../eval_results/csra_prefix8_quality256_20260913/shard_1/comparison_grid.png" style="max-width:100%;width:768px" alt="新seed第2片固定8例，原始RDD、正确coarse、无coarse"></details>
<details><summary>新样本第3片，固定前8张：蝴蝶、石墙、理发椅、船屋、西葫芦、长柄勺、河乌、汽车格栅</summary><img src="../../eval_results/csra_prefix8_quality256_20260913/shard_2/comparison_grid.png" style="max-width:100%;width:768px" alt="新seed第3片固定8例，原始RDD、正确coarse、无coarse"></details>
<details><summary>新样本第4片，固定前8张：比格犬、拨片、博美犬、西葫芦、三体船、松鼠猴、望远镜、蜂鸟；包含明显失败</summary><img src="../../eval_results/csra_prefix8_quality256_20260913/shard_3/comparison_grid.png" style="max-width:100%;width:768px" alt="新seed第4片固定8例，保留失败样例"></details>

本机已保存全部逐图指标、日志及四张固定grid；服务器保留全部768张单图，**没有声称768张单图都已下载/目视检查**。指标可查每张图的label、预测Top-5与SHA。[汇总JSON](../../eval_results/csra_prefix8_quality256_20260913/summary.json)；四片metrics在同目录。下载归档`csra_prefix8_quality256_review_20260913.tar.gz`，SHA-256 `a7263bea358721574ce1d6d8ac70d3c42bb8bf271796c72bf3dfdc747544aecf`。标准库汇总脚本`summarize_csra_prefix_quality_pilot.py`，新增3项配对汇总测试通过。

### 已完成的小诊断：恢复真实fine前8层，定位而不是宣称加速

在3200步checkpoint中，前8层一直冻结；被训练的是输入/时间/类别embedding、后16层和末端头，以及替代前缀的转换器。现在比较五列：原RDD；有来源学生；有来源学生恢复前8层；无来源学生；无来源学生恢复前8层。两种恢复列都**不使用转换器，执行完整24层fine**，原8/16生成仍不变。

它能检查“补回真实fine计算是否值得进一步设计”，但这条完整链未联合训练，embedding和后16层已经适配了转换器，因此即使恢复无效，也不能反过来证明前缀没问题。它不是一种新加速方法，更不能把恢复列的质量与省8层的速度拼在一起。

仅三个旧开发seed、每seed4图×5列，共60图；同80/80/90、CFG1、noise filling。脚本`2d/exp/probe_csra_prefix8_bypass_3gpu.sh`，tmux `rdd_csra_prefix8_bypass_012_20260913`已正常退出，日志`logs/csra_prefix8_bypass_pair_20260913.log`，结果`eval_results/csra_prefix8_bypass_20260913/`。71项测试通过，峰值allocated2.687GiB，全部原dense与正常bridge图SHA和既有3200预览相同，12项原coarse端点parity为0；只用了0–2，没有停止其他训练。

| seed | 正常有coarse | 有coarse学生恢复前8层 | 正常无coarse | 无coarse学生恢复前8层 |
|---|---:|---:|---:|---:|
| 20260921 | 0.299344 | 0.264091 | 0.336793 | 0.269853 |
| 20260923 | 0.226358 | 0.281601 | 0.247401 | 0.298589 |
| 20275004 | 0.272131 | 0.258893 | 0.254287 | 0.264429 |
| 三seed的batch级latent relative L2均值 | 0.265944 | 0.268195 | 0.279494 | 0.277624 |

恢复后，纹身人物、白帽人物、桌后人物的原结构部分回来，但出现过强对比度、红斑与颗粒/纹理伪影；牛的头部也有明显新变形。并非简单地算回8层就解决，也不是恢复无效就证明前8层不重要。**桥接输出与训练后suffix已经共同适配**；恢复路径是未联合训练的新组合，结果提示应检查接口处的hidden对齐，但还不能证明这就是根因。

五列顺序：**原RDD / 有coarse正常学生 / 有coarse恢复8层 / 无coarse正常学生 / 无coarse恢复8层**。三图全部展示，不挑选恢复较好的一组：

<details open><summary>恢复诊断seed20260921：人物回来，但纹理/脸部引入新问题</summary><img src="../../eval_results/csra_prefix8_bypass_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:1280px" alt="原RDD与两个学生正常及恢复前8层的五列对比"></details>
<details><summary>恢复诊断seed20260923：latent偏差反而增大</summary><img src="../../eval_results/csra_prefix8_bypass_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:1280px" alt="恢复8层诊断第二组，含负面结果"></details>
<details><summary>恢复诊断seed20275004：不同对象改善不一致</summary><img src="../../eval_results/csra_prefix8_bypass_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:1280px" alt="恢复8层诊断第三组，保留牛头等失败"></details>

全部60张单图、三张grid、metrics和日志已下载并检查三张grid，归档SHA-256 `bd8f7e8fc369c44f78333d776b59a737db58bbd931c878a5c63cd9bc0bec1576`。这12张旧开发图不是分布质量测试，也不取代上面256张的新seed评估。

### 已完成的2×2实验：不增加推理计算，只在训练中约束前缀hidden

人话：之前只要求模型“最后答对”。这次还要求它在“省掉前8层之后，交给后16层的那个中间状态”尽量接近原教师。这样后16层可能不必适应过度偏移的输入，但也可能限制一种本来有用的内部表示，所以必须做有/无监督的匹配对照，不能先认定一定有效。

先做只读审计：从四个语料分片各取16张训练、16张开发图，共64训练+64开发轨迹，每图8个时间、有/无来源各一次，共2048条逐状态记录。原教师前8层在同一`x/t/label`下产生目标，与之前保存的`h_after7`逐位相等；新`return_hidden=True`接口与默认forward的最终输出逐位相等，不改变RNG。峰值allocated2.783GiB。

| 审计指标 | 训练子集：有coarse | 训练子集：无coarse | 开发子集：有coarse | 开发子集：无coarse |
|---|---:|---:|---:|---:|
| 主预测MSE | 0.005697 | 0.007699 | 0.014513 | 0.016537 |
| hidden归一化MSE | 0.225858 | 0.262081 | 0.239056 | 0.279704 |
| hidden cosine | 0.887634 | 0.868418 | 0.880234 | 0.858018 |

有coarse仍较好；主预测训练/开发差距也很明显，但这些子集大小有限、输入难度可能不同，不能单独认定过拟合程度。hidden坐标本来可能随着联合训练变化，所以hidden差异是**可测试的训练约束**，不是已证明的根因。[全部审计与校准记录](../../eval_results/csra_prefix8_hidden_audit_20260913/metrics.json)已经下载；归档SHA-256 `98289f10ca169f59775088cfa3df6a18ab5e43cddbc3eb3b205a25a7c63c4aee`。

辅助损失按每张图归一化：

$$
L_h=\frac{1}{B}\sum_i\frac{\|h_i^{\mathrm{bridge}}-h_i^{\mathrm{teacher}}\|_2^2}{\|h_i^{\mathrm{teacher}}\|_2^2},\qquad
L=L_{\mathrm{main}}+0.05L_{\mathrm{var}}+\lambda L_h.
$$

实现中用两个逐图mean-square的比值，教师功率设`1e-12`下限；教师不收梯度。只用**64张训练图**、有/无来源合计的初始损失校准一个共同权重，使辅助项初始约占主MSE的5%，得到`λ=0.0013727689284662157`。未用开发图或生成图挑权重，不边跑边调。校准JSON SHA-256 `49c66f0940e4883ce92c83fd6b7d156cd972e5adda707d8a0162b09deb50f8b2`记录进每个checkpoint。

| 实验组 | 3200步父模型 | 新hidden监督 | 新增更新数 |
|---|---|---|---:|
| `on_aligned` | 正确coarse组 | 固定上述λ | 800 |
| `off_aligned` | 同一个正确coarse父模型 | λ=0 | 800 |
| `on_zero` | 无coarse组 | 与有coarse组相同λ | 800 |
| `off_zero` | 同一个无coarse父模型 | λ=0 | 800 |

四组都保留父模型优化器、相同数据顺序与学习率；每卡真实64张、总256，无梯度累计，BF16 AMP+FP32主参数，activation checkpointing。最大fine-tune更新计数为4000（原基模仍是400K），不是重新训练一个4000步的ImageNet基模。checkpoint统一按原来的`主MSE+0.05方差MSE`选best，不把辅助项加入某些组的选择标准。无监督组也计算hidden误差用于记录，训练预算按更新数一致，不承诺训练墙钟完全一致。

推理时仍然是同一个8层前缀替代架构，没有新增教师调用。监督本身是常见蒸馏方法，不作为新理论贡献；此处测试它能否让真实coarse信息更有效地省掉fine计算。主代码默认输出和旧checkpoint保持兼容，新增target计算只出现在新训练脚本。

**启动核验：** 74项测试通过；四组分别用4×64完成2次真实优化器更新、验证与梯度检查；初始主MSE/方差MSE与3200父checkpoint一致，教师无梯度，峰值allocated最高9.361GiB、reserved10.006GiB，无OOM。smoke副本不改父checkpoint。第一组`on_aligned`已进入正式训练队列；其余顺序为`on_zero → off_aligned → off_zero`。旧模型不覆盖，新目录每组保存best/latest。

**已完成，四组均选中第4000步。** 正式训练峰值allocated9.361GiB、reserved10.258GiB/卡；74项回归测试及四组真实4×64 smoke通过。六组预览各4图×3列，共72张单图、6张grid；24张原dense PNG与对应3200参考逐文件SHA相同，12项原8/16端点parity均为0，学生均执行90次fine调用及90次前缀替代。

| coarse来源 | hidden监督 | 主预测验证MSE↓ | hidden NMSE↓ | 三seed平均最终latent相对L2↓ |
|---|---|---:|---:|---:|
| 正确来源 | 无（等预算对照） | 0.014254870 | 0.233796966 | 0.262871499 |
| 正确来源 | 有 | 0.014227875 | 0.194393148 | 0.262217621 |
| 清零来源 | 无（等预算对照） | 0.015268142 | 0.277759529 | 0.274788469 |
| 清零来源 | 有 | 0.015283287 | 0.220607448 | 0.275161311 |

正确来源组：hidden NMSE降低16.85%，主MSE仅降低0.19%，平均最终偏差仅降低0.25%。清零来源组：hidden NMSE也明显降低，但最终偏差**增加0.14%**。有coarse组的监督增益只在2/3个seed成立。**把中间特征训得更像teacher，不自动等于最终图更好。** 这些是三个重复使用的开发seed、每组4图的batch相对L2均值，不是逐图均值、FID、类别命中率或加速百分比。

六张grid已逐一查看：有/无hidden监督的图非常接近，原来的人物构图变化、牛头形状异常、玻璃器皿变形仍然存在；车与蜗牛等样本维持较接近的外形。没有目视上的整体恢复。这里评价的是自然程度和对原图的保真度，不把人物消失自动视为该图label错误。

下面全部图片列顺序一致：**原始RDD / 有coarse学生 / 无coarse学生**；按同seed排列有/无监督，方便直接对照，不挑选好看的图。

<details open><summary>seed20260921：无hidden监督，等预算4000步</summary><img src="../../eval_results/csra_prefix8_hidden_off_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="相同种子无hidden监督对照"></details>
<details open><summary>seed20260921：有hidden监督，4000步</summary><img src="../../eval_results/csra_prefix8_hidden_on_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="相同种子增加hidden监督"></details>
<details><summary>seed20260923：无hidden监督</summary><img src="../../eval_results/csra_prefix8_hidden_off_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="第二组无监督对照"></details>
<details><summary>seed20260923：有hidden监督</summary><img src="../../eval_results/csra_prefix8_hidden_on_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="第二组增加hidden监督"></details>
<details><summary>seed20275004：无hidden监督</summary><img src="../../eval_results/csra_prefix8_hidden_off_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="第三组无监督对照"></details>
<details><summary>seed20275004：有hidden监督，整体偏差反而增大</summary><img src="../../eval_results/csra_prefix8_hidden_on_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="第三组增加hidden监督，保留负结果"></details>

```bash
# 服务器工作目录下查看总进度
# 该任务已完成，tmux正常退出；日志仍在。
less logs/csra_prefix8_hidden_2x2_pair_20260913.log
```

启动脚本`2d/exp/train_csra_prefix8_hidden_2x2_4gpu.sh`；训练脚本`2d/attention_analysis/train_csra_prefix_hidden_pair.py`。checkpoint `checkpoints/RDD-CSRA-prefix8-hidden-{on,off}-{aligned,zero}-L400K-20260913/`。预览输出`eval_results/csra_prefix8_hidden_{on,off}_pair_20260913/`，同80/80/90、CFG1、noise filling。[机器可读汇总](../../eval_results/csra_prefix8_hidden_2x2_summary_20260913.json)由`summarize_csra_hidden_2x2.py`校验生成；归档SHA-256为`2e9aa1953b3c14dec96da90328a2103b999f9e3d8131a97aef6e7f5cd81505a3`。没有FID或视频加速结论。

### 已完成：前缀从8层减到6层，来源优势在三个开发seed均出现

前14层→前8层曾有保真度改善，但前8层继续加训练和加hidden监督已接近平台。因此先测试**只替代前6层，保留fine后18层自行计算**；新目标是原teacher的`h_after5`，不能拿原8层的`h_after7`初始化冒充。重新提取目标、只用训练/校准分割选择线性初始化，再训练相同的条件化读取模块与fine后缀。

| 项目 | 本轮配置 |
|---|---|
| 原模型、原coarse路径 | Unified-L/2 400K，300/300/400；原8/16保持不变 |
| 被替代的fine层 | 编号0–5；保留6–23，共18层 |
| 读取的已有信息 | 原16×16端点t300，编号13层完整hidden；不是新增coarse生成 |
| 对照 | 正确coarse / 清零coarse；相同参数量、共同fine-only初始化 |
| 训练 | 各先800次更新，不从8层模型硬续；全局256=4×64、不累计 |
| 数据/优化器 | 相同4096 train /128 dev轨迹，8个时间点；主干1e-5、bridge1e-4、AdamW、WD0.01 |
| 损失 | 主MSE+0.05方差MSE；不加刚才无实质收益的hidden项 |
| 预览 | 三个相同开发seed，DDPM80/80/90、CFG1、noise filling |
| 速度边界 | 跳过6/24层只表示fine层数少25%，**不是整模型快25%**；读取模块、coarse、VAE等仍有成本 |

启动器`2d/exp/train_csra_prefix6_conditional_pair_4gpu.sh`先检查显存和磁盘，再做回归测试、特征提取、初始化、原后缀parity和两组真实4×64 smoke，然后才训练。已有实验没有被覆盖。75项测试通过，四片提取与初始化完成，原后缀oracle逐位parity=0，两组完整4×64 smoke均通过。两组初始主MSE完全相同（0.10548147419467568），共同初始化SHA为`0cd0a31d511002fc087e5f80c39c8fdd3e1dfe172588924d9b2c037c2de88b2c`。**两组均完成且选中best800**，峰值allocated9.016GiB/卡、reserved10.611GiB/卡，没有OOM。

| 训练结果 | 正确coarse来源 | 清零来源 |
|---|---:|---:|
| 主预测验证MSE↓ | 0.012068447 | 0.014489229 |
| 方差头验证MSE↓ | 0.017022263 | 0.018999930 |
| 共同选模目标↓ | 0.012919560 | 0.015439226 |
| 最终checkpoint | 800 | 800 |

| 开发seed，4图/组 | 正确coarse最终latent相对L2↓ | 清零来源↓ |
|---|---:|---:|
| 20260921 | 0.246435612 | 0.296878606 |
| 20260923 | 0.210554808 | 0.216431424 |
| 20275004 | 0.260196745 | 0.275307596 |
| 三个batch统计量的算术平均 | **0.239062389** | **0.262872542** |

同预算下主MSE低16.71%、最终偏差低9.06%，且三个seed方向一致；这比只报告训练loss更有说服力，但样本仍然小且重复用于开发。旧prefix8有coarse3200步的偏差是0.265944，数值上不如本轮；**深度和训练预算同时不同，不能据此量化“减少两层”单独贡献，更不能把9.06%当作加速。** 两组同时减少深度，但仍能看到真实coarse相对清零的帮助；是否维持条件语义与自然度，下一项新样本pilot再检查。

以下三列均为**原始RDD / 正确coarse学生 / 清零来源学生**，全部36张单图与3张grid已收集并查看。12张原dense PNG与旧同seed参考的SHA相同，六项原8/16端点parity=0；两学生均确认实际best800、90fine调用与90次前缀替代。**目视观察：** 第一组戴帽人物和纹身人物的细节在有coarse组恢复更多，第三组鸟的轮廓比清零组完整；但牛头仍严重失真，玻璃器皿形状也变化。第二组的人物有所保留却仍变形，部分图出现文字状伪影。车与蜗牛等相对稳定。不是全部样本都改善，也没有“质量通过”的结论。

<details open><summary>前缀6、800步，seed20260921</summary><img src="../../eval_results/csra_prefix6_conditional_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="前缀6首组，原RDD、有coarse、无coarse"></details>
<details><summary>前缀6、800步，seed20260923</summary><img src="../../eval_results/csra_prefix6_conditional_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="前缀6第二组，保留完整样本"></details>
<details><summary>前缀6、800步，seed20275004</summary><img src="../../eval_results/csra_prefix6_conditional_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="前缀6第三组，保留完整样本"></details>

原始metrics在对应seed目录；训练日志`logs/csra_prefix6_conditional_{aligned,zero}_20260913.log`。归档`csra_prefix6_conditional_complete_20260913.tar.gz`的SHA-256是`2731b895550bff56a687d18ee0d07c7551c4fd1d976f70426e751a40415cdf21`，不包含大型模型权重。

```bash
# 该训练队列已完成，tmux正常退出。
less logs/csra_prefix6_conditional_pair_20260913.log
```

这一步只回答压缩强度与质量的关系。即使图片改善，也必须再与充分训练的清零来源组比较，避免把普通浅层学生的能力误说成RDD的贡献。视频中还需实测同一RDD+VSA基线，把coarse捕获、读取、投影、额外权重驻留与边界操作一起计入；在此之前不报告10%已实现。

### 已完成的新样本评估：固定前缀6的best800，不继续训练

用新seed基数20460913、四片各64图，对比原RDD、正确coarse、清零来源，共256图/方法、768张单图。全部参数沿用上面采样设置；新seeds与先前prefix8的256样本不同，因此只比较本轮内部配对结果，不能直接把两次类别分数相减当作结构的因果增益。

保留真实采样label，每片固定展示前8张；使用已有ResNet50 ImageNet1K V2官方预处理、CPU FP32推理记录Top-1/Top-5和目标类别概率。分类只是条件语义诊断，不是FID或真实感。学生checkpoint固定800，不看新图片后更换模型。新增评估脚本`2d/exp/sample_csra_prefix6_quality256_4gpu.sh`；公共采样/汇总脚本仅增加显式depth、step、seed参数，旧默认prefix8/3200/旧seed保持不变，76项回归测试通过。

任务已正常结束，tmux `rdd_csra_prefix6_quality256_0123_20260913`已退出，不再重启。总日志`logs/csra_prefix6_quality256_pair_20260913.log`，分片日志`logs/csra_prefix6_quality256_shard{0,1,2,3}_20260913.log`，输出`eval_results/csra_prefix6_quality256_20260913/`。峰值allocated3.638GiB/卡，没有OOM。

| 固定256个新样本，CFG1 / DDPM80/80/90 / noise filling | Top-1 | Top-5 | 目标类别概率均值 | 平均分片latent相对L2 |
|---|---:|---:|---:|---:|
| 原始RDD-Unified-L/400K | 176/256 = 68.75% | 223/256 = 87.11% | 0.26939 | 0 |
| 条件前缀6，有正确coarse，best800 | 152/256 = 59.38% | 210/256 = 82.03% | 0.23766 | 0.262601 |
| 相同学生，来源清零且同预算训练，best800 | 136/256 = 53.13% | 193/256 = 75.39% | 0.20336 | 0.292966 |

**这轮回答了什么？** 有coarse相比清零的Top-1高6.25个百分点、Top-5高6.64个百分点；10,000次配对bootstrap的95%区间分别[1.17,11.33]、[1.95,11.33]个百分点。这不是多个训练随机种子的置信区间，而是固定两模型、四片采样RNG条件下，按图片对重采样的区间。Top-1有31对从清零错误变正确、15对反过来，210对相同。四片latent相对L2也全部是有coarse更低。

**没有回答什么？** 相比原RDD，有coarse的Top-1仍低9.38个百分点（区间[-15.23,-3.91]），Top-5仍低5.08个百分点（[-9.38,-1.17]）。不能因此宣布画质达标。类别命中衡量条件语义，不是FID或照片自然度；相对L2是四个batch级比值的平均，不是256个单图误差的平均。旧prefix8 pilot使用不同seeds与训练预算，两轮原RDD分数也不同，不能用原始百分比直接证明“少跳两层提升了多少”。

**固定展示图观察。** 每片前8个样本在看结果前就确定；下面全部保留。图1中有coarse的动物头部同样会变形，不能称每张都占优。图2雪地机械在有coarse组保留得比清零好，但玻璃/蛇形仍变化较大。图3服装的大结构较稳定，鸟与复杂人物仍失败。图4海滩、甲虫总体接近，钢琴键、发饰纹理、球员和衣服文字改变明显。这些例子提示粗结构能继承，但高分辨率精修还不足，不能只挑海滩等稳定样本。

<details open><summary>新样本第0片：左原RDD，中有coarse，右清零</summary><img src="../../eval_results/csra_prefix6_quality256_20260913/shard_0/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6质量256第0片，包含成功与失败"></details>
<details><summary>新样本第1片：同三列</summary><img src="../../eval_results/csra_prefix6_quality256_20260913/shard_1/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6质量256第1片"></details>
<details><summary>新样本第2片：同三列</summary><img src="../../eval_results/csra_prefix6_quality256_20260913/shard_2/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6质量256第2片"></details>
<details><summary>新样本第3片：同三列</summary><img src="../../eval_results/csra_prefix6_quality256_20260913/shard_3/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6质量256第3片"></details>

审计：4份metrics、各192行、256对labels、231个不同类别、全部768张PNG与记录SHA一致、8项原8/16阶段parity为0；所有学生实际使用depth6/best800，分类器SHA一致。全部metrics、日志、四张grid和[机器汇总](../../eval_results/csra_prefix6_quality256_20260913/summary.json)已拉到本机并复算；768张单图仍留服务器，并未逐张人工查看。归档`csra_prefix6_quality256_review_20260913.tar.gz`共10,857,426字节，SHA256 `4f30433486eb06fb423820f97c0db6a3197ecffb2d5e3f1b8f425ba79afb7b25`。

### 已完成的有界验证：视频条件前缀的净计算开销

不是直接继续盲目训练。先检查：**保留同样的RDD+VSA，只把fine前缀改为真正的bridge后，新增开销会不会吃掉省下的时间？** 新代码`rdd_video/benchmark_csra_prefix_feasibility.py`及`attention_analysis/csra_prefix_video_bridge.py`，独立于旧FastVideo源文件；没有修改已有checkpoint或其他AnyFlow工作。

| 项目 | 这次计算可行性pilot的明确设置 |
|---|---|
| 基线 | 同一FastWan2.1-T2V-1.3B，RDD三阶段+原生在线VSA，保留20% |
| 唯一结构变化 | 最后一个NFE的fine前6或8层被条件bridge替代；其余层的VSA coarse输出、gate、cross-attention、MLP不变 |
| bridge输入 | 当前fine patch hidden、当前Wan原始timestep embedding、middle阶段原block13的完整hidden |
| bridge计算 | FP32全通道3H→H仿射；当前fine Q读原生coarse K/V的BF16 attention；H→2H→H点式FFN；全部计时 |
| 分辨率与时间 | 832×448×29帧、1280×768×29帧；Spatial421及ST421 |
| 采样 | 同样DMD 3 NFE，1000/757/522，shift8，noise_filling，native边界；不改步数或offload制造收益 |
| 权重与内存 | 原coarse主干与另一个fine主干分别驻留；两组计时中都保持两套权重驻留；报告额外参数与峰值内存 |
| 计时 | 各shape/方法预热；每个对照12组ABBA/BAAB、每组4次完整采样；保存原始时间及分位数 |
| 边界 | Wan bridge还没训练；这不是2D训练权重移植，更不是视频质量验证。只计完整denoiser+transition，不含VAE |

小型CPU布局/前缀作用域/空开关等6项测试通过。真实模型的两份加载权重和不启用bridge的包装器都与原版逐位一致，最大绝对差均为0。然后完成8个对照，每组12次ABBA/BAAB、每次4条完整采样轨迹，共384次正式测量；所有输出finite，所有实际执行/跳过层数检查通过。单卡GPU0、启动前至少32GiB空闲、PyTorch进程allocator上限为整卡24%；没有停止其他用户任务，没有OOM。共享GPU上的成对测量仍不能视为独占GPU论文最终速度。

启动器`rdd_video/scripts/run_csra_prefix_video_feasibility.sh`，专用tmux `rdd_csra_prefix_video_feasibility_20260913`已正常退出；日志`logs/csra_prefix_video_{launcher,smoke,full}_20260913.log`，输出`eval_results/csra_prefix_video_feasibility_20260913/{smoke,full}/`。完整记录已经下载并在本机复核384行执行计数。

**结果表：基线始终是同一个stage配置的RDD+原生VSA。** 时间包含三个NFE、noise-filling边界、x0重建、实际bridge与缓存，不含模型加载/T5/VAE；每个shape已预热。单次时间列是24次原始调用的中位数；加速/延迟下降是12组成对均值之比的中位数，因此不会与两列时间直接相除完全相同。IQR是配对结果中间50%的范围，不是置信区间。

| 输出（均29帧） | RDD配置 | fine跳过层数 | RDD+VSA中位秒 | +实际bridge中位秒 | 配对加速 | 延迟下降 | 延迟下降IQR |
|---|---|---:|---:|---:|---:|---:|---:|
| 832×448 | Spatial421 | 6 | 0.9064 | 0.8061 | 1.131× | 11.55% | 11.35–12.30% |
| 832×448 | Spatial421 | 8 | 0.9126 | 0.7643 | 1.201× | 16.76% | 15.75–17.27% |
| 832×448 | ST421 | 6 | 0.8707 | 0.7669 | 1.145× | 12.66% | 12.23–13.08% |
| 832×448 | ST421 | 8 | 0.8751 | 0.7253 | 1.212× | 17.51% | 16.66–18.26% |
| 1280×768 | Spatial421 | 6 | 2.6382 | 2.2731 | 1.162× | 13.92% | 13.49–14.47% |
| 1280×768 | Spatial421 | 8 | 2.6413 | 2.1350 | 1.238× | 19.21% | 19.05–19.33% |
| 1280×768 | ST421 | 6 | 2.4607 | 2.0778 | 1.180× | 15.25% | 14.93–16.09% |
| 1280×768 | ST421 | 8 | 2.4586 | 1.9410 | 1.265× | 20.93% | 20.65–21.44% |

Spatial421是`(4,1)→(2,1)→(1,1)`，ST421是`(4,4)→(2,2)→(1,1)`，顺序为空间缩小倍数、时间缩小倍数。不是把同形状全分辨率VSA与RDD换步数混比。Wan实际30层，跳6层是fine阶段20%的层数；与2D的6/24不是同一比例，更不能直接换算成整个视频20%加速。

**没有省略bridge的代价。** bridge自身中位约14ms（832），约46ms（1280 Spatial）/41ms（1280 ST），已包含在上表。它不是零成本搬运，更不是免费跳层。原模型每套参数2,979,643,520字节，两套都实际加载；新增bridge参数141,631,488字节。这个保守原型还驻留了fine中不用的前缀参数，没有用裁剪权重偷偷降低基线成本。全测试峰值allocated最高9.371GiB；这不是基线单模型的最小显存，两个方法计时时都驻留两套权重。

**共享GPU的异常没有删除。** 832-ST的prefix6某一组仅下降0.18%，全组范围0.18–14.45%；不能宣称每次都至少快10%。大分辨率prefix6范围为Spatial13.11–14.62%、ST14.28–17.14%。这些是当前共享负载下的pilot，不是独占GPU最终验收；以后报告论文速度应再在独占GPU复测。

**最重要的限制。** 这是当前结构的真实算力账，不是随机造一个省层数公式；但bridge权重随机、fine主干仍是原始FastWan，尚未做视频RDD/bridge微调。这轮只检查finite与执行正确性，没有生成可评价质量的新视频。VSA路由也会随未训练特征改变，训练后的精确速度仍需重测。2D已经训练的良好/失败图片不能与本表合起来当“同一个模型已经同时达成质量和速度”。目前只确认：**值得继续优化质量，新增模块没有在这些case中吃掉全部计算节省。**

[完整机器记录](../../eval_results/csra_prefix_video_feasibility_20260913/full/metrics.json)。归档`csra_prefix_video_feasibility_review_20260913.tar.gz`共167,177字节、7个JSON/日志文件，SHA256 `8874508f66bfeefb0912990645d9ed02773a575617f4064ca53d6f759ef896a5`；已下载、校验、解包。

### 已完成续训：同一前缀6结构，800→1600，保留匹配清零对照

为什么再给800步，而不是无限扩训？有来源组的验证主MSE在600/700/800步仍连续下降（0.013278/0.012538/0.012068），清零组也下降（0.016215/0.015231/0.014489）；256新样本已显示来源优势，但相对原RDD仍有语义损失。现在视频计算也通过有界检查，因此值得做一次**只改变训练预算**的对照，而不是同时改结构、loss和采样。

| 配置 | 这次续训 |
|---|---|
| 起点 | 两组分别从自己的prefix6 best800恢复模型和AdamW状态 |
| 终点 | 各最多1600次累计微调更新；不是重新训练1600，也不计入原基模400K |
| batch/GPU | 0–3四卡，每卡实际64、全局256，无梯度累计 |
| 主干 | 原8/16冻结；另一个fine主干训练后18层、输入/时间/类别嵌入、输出头和bridge |
| 不变项 | 相同4096训练轨迹、128开发轨迹、8个t、BF16 AMP/FP32主参数、fine LR1e-5、bridge LR1e-4、WD0.01、clip1、mainMSE+0.05varianceMSE |
| checkpoint选择 | 同一开发目标选best；800步父checkpoint仍可被保留为best，不能假定最终best一定1600 |
| 后续检查 | 原来三个开发seed各4图，原RDD/有来源/清零三列；先看实际画质和原8/16逐位parity，不自动增加训练 |

脚本`2d/exp/continue_csra_prefix6_matched1600_4gpu.sh`在视频任务退出、全部8项完成、真实模型parity=0、两个1280-case的prefix6中位延迟下降都≥10%后才启动；因此没有用新增训练干扰上述视频计时。新目录`checkpoints/RDD-CSRA-prefix6-conditional1600-{aligned,zero}-L400K-20260913`，旧best800及图片没有修改。父checkpoint的副本保留为初始best。

专用tmux `rdd_csra_prefix6_matched1600_0123_20260913`已正常退出；总日志`logs/csra_prefix6_conditional1600_pair_20260913.log`，训练日志`logs/csra_prefix6_conditional1600_{aligned,zero}_20260913.log`。两组均完成并实际选中best1600；恢复时800步验证值与原记录完全一致。峰值allocated9.017GiB/卡，reserved有来源9.850GiB/卡、清零9.600GiB/卡，未OOM。没有自动续到3200。

| 验证项 | 有coarse，800 | 有coarse，1600 | 清零，800 | 清零，1600 |
|---|---:|---:|---:|---:|
| 主预测MSE，1024个开发状态 | 0.01206845 | 0.00981108 | 0.01448923 | 0.01124191 |
| 方差头MSE | 0.01702226 | 0.01235029 | 0.01899993 | 0.01292307 |
| best选择目标：主MSE+0.05方差MSE | 0.01291956 | 0.01042859 | 0.01543923 | 0.01188807 |
| 三个开发seed平均latent相对L2 | 0.239062 | 0.223260 | 0.262873 | 0.251200 |

有来源主MSE比同预算清零低12.73%，平均latent偏差低11.12%；但这个均值不是每张图都改善，也不是加速百分比。两个模型增加相同训练预算后主MSE分别少18.70%和22.41%，最终latent偏差只少6.61%和4.44%，不能把训练loss改善幅度照搬成生成质量改善幅度。

| seed（各4图） | 有来源800→1600 | 清零800→1600 | 本轮观察 |
|---|---:|---:|---|
| 20260921 | 0.246436→0.227872 | 0.296879→0.323108 | 有来源更接近原构图；清零的人物/纹理变化更大，但这些图片的目标label未必是人物 |
| 20260923 | 0.210555→0.183932 | 0.216431→0.189451 | 两组都改善；有来源人物的额外覆盖物减少，车和蜗牛较稳定 |
| 20275004 | 0.260197→0.257976 | 0.275308→0.241042 | 清零比有来源更接近原latent；有来源鸟头方向改变、牛头仍严重畸变，不能称已修复 |

**人工核对：** 三张1600步grid及对应800步grid均已查看。白帽人物在有来源组继续保留，第二组人物结构有所恢复；但纹身、文字、鸟头和牛头仍偏离，第一组清零甚至丢掉人物结构。原始图片本身也可能包含不合理内容，偏离原图不必然等于类别错误。因此下一个判断依赖更大固定样本的条件语义对照，而不是对某张人物图下结论。

<details open><summary>1600步，seed20260921：左原RDD，中有coarse，右清零</summary><img src="../../eval_results/csra_prefix6_conditional1600_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6 best1600第一组，三方法全部样本"></details>
<details><summary>1600步，seed20260923：同三列</summary><img src="../../eval_results/csra_prefix6_conditional1600_pair_20260913/seed_20260923/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6 best1600第二组，人物车和蜗牛"></details>
<details><summary>1600步，seed20275004：同三列，保留失败结果</summary><img src="../../eval_results/csra_prefix6_conditional1600_pair_20260913/seed_20275004/comparison_grid.png" style="max-width:100%;width:768px" alt="prefix6 best1600第三组，清零对照反而较接近原latent"></details>

审计全部通过：两组非smoke的1600完成标记、三组实际best1600、90次fine调用/替代、6项原8/16 parity均为0，12张原RDD参考图与800步实验SHA完全一致。36张单图、三张grid、日志和[审计记录](../../eval_results/csra_prefix6_conditional1600_pair_20260913/completion_audit.json)全部在本机。归档`csra_prefix6_conditional1600_review_20260913.tar.gz`含51个文件，共7,862,833字节，SHA256 `b2d4682206186c7bc58265b725301c258b50df2f2511a98ed7b4d870e30a112c`，解包后逐张验证哈希。

### 已完成的配对预算检查：固定同一批256图，比较800与1600

这轮不改变模型、不继续训练。仍使用seed基数20460913、四片各64图、原RDD/有来源/清零三方法、CFG1、DDPM80/80/90和noise filling；只将两个学生从best800换成best1600。旧800步结果直接复用，原RDD也重新生成用于逐图SHA一致性检查。由于800结果已被用来决定续训，这些是**固定开发样本，不是全新独立测试集**。

脚本`2d/exp/sample_csra_prefix6_1600_quality256_4gpu.sh`通过82项回归测试后正常完成，原tmux已退出。四片共768张PNG哈希全部核验，256张原RDD图片及其分类预测与800轮完全一致，8项coarse端点parity全为0，实际checkpoint均best1600。四张grid与完整JSON已经下载；768张单独PNG仍保存在服务器，不声称全部逐张目视检查。结果包10,798,145字节，SHA256 `3a9917dfea56e75d14a5629d3f356482d57521aeff4e2caad24861734d82be8a`。

| 同256个样本 | 原RDD | 有coarse 800 | 有coarse 1600 | 清零800 | 清零1600 |
|---|---:|---:|---:|---:|---:|
| Top-1命中 | 176，68.75% | 152，59.38% | **165，64.45%** | 136，53.13% | 139，54.30% |
| Top-5命中 | 223，87.11% | 210，82.03% | **210，82.03%** | 193，75.39% | 204，79.69% |
| 平均目标类别概率 | 0.26939 | 0.23766 | 0.25315 | 0.20336 | 0.21153 |
| 平均逐图latent relative L2 | 0 | 0.26260 | 0.25316 | 0.29297 | 0.27833 |

**配对后才能说的结果。** 有coarse的800→1600 Top-1净增13/256，即5.08个百分点，按图片对bootstrap的95%区间[1.17,8.98]个百分点；20张变正确、7张变错误。Top-5为7张改善、7张退步，净变化为0。清零Top-1只增1.17个百分点，区间[-2.73,5.08]；但清零Top-5增加4.30个百分点，不能隐藏这个改善。

在1600步同预算下，有coarse比清零Top-1高10.16个百分点，区间[4.69,15.63]；Top-5只高2.34个百分点，区间[-1.95,6.64]。有coarse相对原RDD的Top-1差为-4.30个百分点，区间[-9.77,1.17]；**区间包含0不等于两者质量等价**。Top-5仍低5.08个百分点，区间[-9.38,-1.17]。以上是固定模型下256对图片的描述性区间，不包含训练seed方差，也未作多重比较校正。

**统计口径更正。** 回查实际使用的`sample_csra_adapter_preview.py::relative_l2`，它先逐图计算`norm(candidate-reference)/norm(reference)`，再取均值。四片都是64图，因此再平均四片，确实等于256图的平均逐图误差。此前本文及旧summary warning将其写成“batch范数比、不是逐图均值”是不正确的文字说明；数值没有因此改变，也绝不是质量下降百分比。本文旧表中出现的“batch级”请按此更正理解。

**固定32张展示图中的模式。** 场景大结构、部分犬、海滩和甲虫相对稳定；但bighorn头部、有人的复杂场景和乐器细节仍明显偏离。雪地机械有coarse比清零保留得好；鹅在有coarse样本中几乎消失，不能只展示前者。钢琴、蛇、瓶口与衣服文字仍改变。这提示“来源信息帮助语义”不等于“精细形状已恢复”，下一步必须检查分布质量。

三列固定为原RDD / 有coarse best1600 / 清零best1600；全部四片的前8张均保留：

<details open><summary>固定开发集1：包括动物头部失败</summary><img src="../../eval_results/csra_prefix6_1600_quality256_20260914/shard_0/comparison_grid.png" style="max-width:100%;width:768px" alt="best1600固定样本第1片"></details>
<details><summary>固定开发集2：雪地机械保留，但蛇和瓶口改变</summary><img src="../../eval_results/csra_prefix6_1600_quality256_20260914/shard_1/comparison_grid.png" style="max-width:100%;width:768px" alt="best1600固定样本第2片"></details>
<details><summary>固定开发集3：保留鹅消失等失败</summary><img src="../../eval_results/csra_prefix6_1600_quality256_20260914/shard_2/comparison_grid.png" style="max-width:100%;width:768px" alt="best1600固定样本第3片"></details>
<details><summary>固定开发集4：海滩稳定，钢琴和文字仍变化</summary><img src="../../eval_results/csra_prefix6_1600_quality256_20260914/shard_3/comparison_grid.png" style="max-width:100%;width:768px" alt="best1600固定样本第4片"></details>

原始[1600统计](../../eval_results/csra_prefix6_1600_quality256_20260914/summary.json)、[800→1600逐图配对变化](../../eval_results/csra_prefix6_1600_quality256_20260914/budget_comparison.json)、[文件及coarse一致性审计](../../eval_results/csra_prefix6_1600_quality256_20260914/completion_audit.json)均可追溯。

### 已完成：固定三个模型的10K分布质量pilot

这一轮回答“分类改善是否也伴随分布质量改善”，不再继续靠调分类分数挑模型。三个固定模型：原始`RDD-Unified-L/0400000.pt`、prefix6有来源best1600、同预算清零best1600。保持原始8/16、DDPM80/80/90、CFG1、noise filling；新seed基数20560914，各方法相同10000个实际labels和RNG，四卡各2500张，每批最多64张、VAE每批4张。三个模型各10000张，总计30000张。

| 项目 | 配置及核查 |
|---|---|
| GPU与安全 | 仅0–3；启动22GiB空闲门限、进程内20GiB检查；PyTorch分配器上限为单卡总显存18%，不停止其他任务 |
| 模型 | DiT-L/2、300/300/400；两个学生只替代fine前6层，fine后18层等已训练；coarse原始权重冻结 |
| 断点 | 按64图chunk原子保存NPZ和元数据，重启核对sha；旧实验不覆盖 |
| 配对 | 每batch检查真实label、8/16端点原始字节hash；各rank检查模型文件SHA一致 |
| 评估 | 原有OpenAI ADM五指标，复用同一原生ImageNet参考；CPU执行，避免占其他训练显存 |
| 结果含义 | FID10K开发pilot，不能与论文FID50K直接比较；也不是视频质量或速度实验 |
| 启动脚本 | `2d/exp/eval_csra_prefix6_fid10k_4gpu.sh` |
| tmux | `rdd_csra_prefix6_fid10k_0123_20260914` |
| 总日志 | `logs/csra_prefix6_fid10k_launcher_20260914.log` |
| 分卡日志 | `logs/csra_prefix6_fid10k_shard{0,1,2,3}_20260914.log` |
| 评估日志 | `logs/csra_prefix6_fid10k_metrics_20260914.log` |
| 输出 | `eval_results/csra_prefix6_fid10k_20260914/` |

**任务已正常结束，原tmux已退出，不再轮询它。** 88项测试通过；正式学生64图采样的最高allocated为3.7994GiB/卡，无OOM。四片全部完成；三组各10000张，480个chunk哈希、10000对labels和原8/16端点hash均匹配。三个assembled NPZ的形状、uint8类型、标签和完整文件哈希也逐一核查，不只是计数目录中文件。

| 指标（同参考、同N、同配置） | 原始RDD | 有coarse best1600 | 清零best1600 |
|---|---:|---:|---:|
| Inception Score ↑ | **116.0083** | 97.6982 | 84.6775 |
| FID ↓ | **11.8755** | 16.8040 | 20.3236 |
| sFID ↓ | **17.5159** | 18.3741 | 18.5105 |
| Precision ↑ | **0.6953** | 0.6495 | 0.6243 |
| Recall ↑ | **0.7007** | 0.6929 | **0.7007** |

**能确认的结论：** 在这轮固定预算对照中，有来源比清零FID低3.5196，Precision高0.0252，IS高13.0207；coarse hidden确实提供了有用信息。但有来源仍比原RDD的FID高4.9285、Precision低0.0458、IS低18.3101。不能把“明显胜过删掉来源的对照”写成“质量没有损失”。Recall只比原RDD低0.0078，且清零虽然Recall相同，FID更差；这组结果更符合精度/自然度受损，而不是明显的覆盖率坍缩。**这是指标与图像支持的解释，不是已经证明了失效机制。** 没有多次训练seed或FID置信区间，不把10K单次结果写成确定的总体排序。

参考NPZ实际含10000张uint8图及预存`mu/sigma`、`mu_s/sigma_s`。沿用既有ADM实现：FID/sFID读取该文件预存统计，Precision/Recall使用其图像特征。三组使用同一文件，SHA256 `b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758`；不要把“10000张参考图”误写成“全部预存统计必定由这10000张算得”，文件没有提供这样的生成溯源。

**图片核对（四片各固定前8张，未按效果挑选）。** 低分辨率来源有时能保住清零组丢掉的全局场景：第一组水面动物在清零组变成近似洞穴/陆地背景，而有来源保留水面；第四组人群在清零组出现人物消失，有来源相对保留。另一方面，有来源仍会改变红衣人物/玩偶的脸、细长机械结构、部分动物头部、球场人物、鞋面和车轮细节，不能只展示稳定的食物、龟或网球。原RDD自身也有畸形例子；这里既看与原图一致性，也看自然度，不把原图视为每张都正确的照片。

三列依次是原始RDD / 有coarse / 清零：

<details open><summary>10K固定预览1：场景保留与面部/机械失败并存</summary><img src="../../eval_results/csra_prefix6_fid10k_20260914/shard_0/comparison_grid.png" style="max-width:100%;width:768px" alt="10K同种子第1片三方法比较"></details>
<details><summary>10K固定预览2：毛发、脸部、服装及室内结构</summary><img src="../../eval_results/csra_prefix6_fid10k_20260914/shard_1/comparison_grid.png" style="max-width:100%;width:768px" alt="10K同种子第2片三方法比较"></details>
<details><summary>10K固定预览3：猴脸、球场人物、纹理与轮廓变化</summary><img src="../../eval_results/csra_prefix6_fid10k_20260914/shard_2/comparison_grid.png" style="max-width:100%;width:768px" alt="10K同种子第3片三方法比较"></details>
<details><summary>10K固定预览4：人物消失、车轮及犬脸变化</summary><img src="../../eval_results/csra_prefix6_fid10k_20260914/shard_3/comparison_grid.png" style="max-width:100%;width:768px" alt="10K同种子第4片三方法比较"></details>

原始[五指标JSON](../../eval_results/csra_prefix6_fid10k_20260914/metrics/summary_fid10k.json)与[完整审计](../../eval_results/csra_prefix6_fid10k_20260914/completion_audit.json)可查。结果包599文件、11,754,429字节，SHA256 `b526ca5fe129cfef7a200b5dd66e55c877f36375eb76b667778a143e68e4f827`。包含96张固定单图、480份chunk元数据、模型/数据hash、五指标和日志；大NPZ和全部生成样本留在服务器。四张grid在本机由固定单图原尺寸拼接，没有生成式修改。

### 已完成的小诊断：当前前缀6模型的全90时间点误差预算

旧prefix8的两seed检查没有发现学生状态上单步MSE爆炸，所以不能把这次FID损失直接归因于“训练只看teacher轨迹”。现在换成**当前prefix6 best1600**做小型复核：两组新开发seed20660914/20670914，每seed16张，有来源/清零两个训练后模型；0–3卡各承担一个seed×模型，不继续训练或解码新图。

每组分别沿原teacher和学生自身轨迹走完，并在全部90个fine t上重新计算原teacher对**该条轨迹当前状态**的目标。额外预测必须不消耗RNG，coarse端点与未插桩轨迹输出必须逐位相同；再记录主MSE、2×2格内均值/差分误差，以及真实jump采样器对这两个频带误差的局部放大系数。把MSE乘对应系数平方，只表示这一步传到下一latent的误差能量；**不是最终FID敏感度，也不能把90项直接相加当最终误差**。

脚本`2d/exp/diagnose_csra_prefix6_error_budget_4gpu.sh`已经完成，专用tmux `rdd_csra_prefix6_error_budget_0123_20260914`正常退出，不再轮询。四份结果共11520条逐图逐t记录；插桩前后teacher/学生各自的最终latent逐位一致，原8/16端点也一致。91项回归测试通过，峰值allocated不超过2.883GiB。全部数据与日志已下载，结果包1,050,556字节，SHA256 `f504bd49dd7e9c5cefec1f9ac84d8cd061a3799f05cae97f190e22e71f407956`；[完整汇总](../../eval_results/csra_prefix6_error_budget_20260914/summary.json)可查。

| 32图×全部90个fine t | 有coarse：teacher轨迹 | 有coarse：学生轨迹 | 清零：teacher轨迹 | 清零：学生轨迹 |
|---|---:|---:|---:|---:|
| 平均主头MSE | 0.0120450 | **0.0100046** | 0.0133648 | **0.0113655** |
| 平均局部jump MSE | 4.2835e-6 | 3.5835e-6 | 4.7560e-6 | 4.0627e-6 |
| t299–200，30次：局部误差和占比 | 10.26% | 9.89% | 12.05% | 12.03% |
| t199–100，30次：局部误差和占比 | 29.52% | 27.54% | 29.74% | 29.13% |
| t99–1，29次：局部误差和占比 | **60.22%** | **62.57%** | **58.21%** | **58.84%** |
| t0，1次：局部误差和占比 | 0 | 0 | 0 | 0 |

这轮同样不支持“学生状态上的单步误差爆炸”：两种模型在学生自身状态上的主MSE反而低于teacher轨迹。这不等于轨迹稳定，更不等于已经排除on-policy训练收益；两个目标分别在各自状态重新计算，模型可能走到更容易预测但质量更差的状态。

更具体的线索是：最后约三分之一有效调用占了大约六成的**局部**误差和。当前训练只有8个均匀出现的锚点，实际采样器对它们的预测误差也不是同等敏感，值得用固定采样器系数做一次loss对照。高频误差在主头MSE中占约78%–79%，但2×2差分子空间本来就占75%的自由度，**不能把79%当作“高频特别异常”的证据**；经过jump增益之后高频占约70%–72%。因此不凭这个结果直接加高频权重或增加fine网络层数。

### 已完成：采样器感知loss四组对照，没有可靠画质提升

**2026-09-14四组训练及两组256图/模型的配对评估全部完成。** 保持prefix6 bridge、fine后18层、原8/16、数据、LR、batch和推理计算量不变。四组均从自己的best1600出发，恢复Adam状态与数据顺序，只增加800次更新到固定2400。不把某组更长训练或不同选模策略混进来。

| 实验名 | coarse输入 | 训练loss | 初始化与最终比较 |
|---|---|---|---|
| `old_aligned` | 原16阶段的真实hidden | 原主头MSE＋0.05方差头MSE | 有来源best1600→latest2400 |
| `old_zero` | 清零，同架构读取模块照算 | 原主头MSE＋0.05方差头MSE | 清零best1600→latest2400 |
| `jump_aligned` | 与第一组相同 | 按实际下一步更新系数加权＋0.05方差头MSE | 与第一组相同父模型和更新数 |
| `jump_zero` | 与第二组相同 | 与第三组相同加权 | 与第二组相同父模型和更新数 |

直观解释：预测错误同样大，不代表它们对下一步latent的影响一样大。先按当前DDPM80/80/90真实jump实现，测出这个“影响系数”，再把影响大的错误在训练里看重一些。不是手调“晚期乘几倍”，也不是假定细节一定比整体更重要。

令主头误差为e，P为2×2平均后最近邻放回原网格。P(e)是每个小格的共同偏差，e−P(e)是格内差异，两部分正交。真实下一步样本误差为：

\[
\delta x_{\mathrm{next}}=g_{\mathrm{low}}(t)P(e)+g_{\mathrm{high}}(t)(e-P(e)).
\]

所以新的主头loss是两部分MSE分别乘系数平方，再除以共同常数Z。Z不是根据FID拟合的：假设每个位置都有单位方差独立误差，2×2格内均值/差异各占1/4、3/4自由度；把八个固定训练时间的预期下一步误差取均值，得到 **Z=0.0002625788071**。所有组、所有时间共用同一个Z，保留0.05方差头项。

| 实际训练t | 均值误差权重 | 格内差异权重 |
|---:|---:|---:|
| 299 | 2.081642 | 0.238756 |
| 259 | 3.334375 | 0.703785 |
| 215 | 1.705542 | 0.570117 |
| 171 | 1.566265 | 0.737291 |
| 128 | 2.578118 | 1.572937 |
| 84 | 1.378558 | 1.033442 |
| 40 | 1.303725 | 1.160929 |
| 0 | 0 | 0 |

权重不严格单调，因为90步离散采样的相邻原t间隔有3也有4。最后t=0的主头权重为0，是因为**现有fine采样器返回的sample在这次调用不变**；没有在这轮偷偷修采样器。8/16阶段用于上采样的pred-x0完全不动。局部t=0不监督主头不等于其他时间的主头没有训练；方差辅助项仍保留以匹配旧设置。

| 实施与检查 | 当前记录 |
|---|---|
| 基模与训练量 | 原RDD-Unified-L/2 400K、300/300/400；只续训fine学生；有效fine参数344,502,304，bridge参数15,740,928 |
| 优化器 | AdamW，fine LR 1e-5、bridge LR 1e-4、WD 0.01、clip 1，恢复各自1600步Adam状态 |
| 数据和精度 | 原4096训练/128开发轨迹×8个固定t；离线teacher目标；BF16 AMP、FP32主权重、TF32关闭 |
| 卡与batch | 仅物理0–3；4×64=256，无梯度累计；四组依次使用四卡，不是各一张卡单独训练 |
| 安全 | 启动前至少26GiB空闲/卡、50GiB磁盘；本进程PyTorch分配器上限为卡容量24%；不停止任何其他作业 |
| 测试 | 100项回归测试通过；单位权重loss误差仅浮点舍入、梯度逐位一致；实际jump混合场响应校验通过 |
| 真实短测试 | 四组各做2次4×64更新，均完成；最大allocated约9.019GiB/卡，reserved约9.631GiB/卡，无OOM；父模型验证值一致、恢复数据顺序四组逐卡相同 |
| 正式状态 | 四组`complete.json`均为2400步完成；无OOM，已退出，不重跑 |
| 评估 | `quality_old`与`quality_jump`均完成，各三模型×256张；相同seed/类别、80/80/90、CFG1、noise filling；统一加载**latest2400**，不选best |
| 后续决策 | 新loss改善不可靠，不增加同配方训练预算，不跑本轮大FID；改做同六层替代预算的位置诊断 |

已结束的后台会话：`rdd_csra_prefix6_jump_loss_0123_20260914`。保留日志可查看：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
tail -f logs/csra_prefix6_jump_loss_pair_20260914.log
```

代码入口：[四组启动脚本](../../2d/exp/train_csra_prefix6_jump_loss_2x2_4gpu.sh)、[新loss及系数审计](../../2d/attention_analysis/csra_jump_weighted_loss.py)、[独立续训器](../../2d/attention_analysis/train_csra_prefix6_jump_loss.py)、[新增测试](../../tests/test_csra_jump_weighted_loss.py)。新checkpoint写入`checkpoints/RDD-CSRA-prefix6-jump-loss-{old,jump}-{aligned,zero}-L400K-20260914/`；质量和系数写入`eval_results/csra_prefix6_jump_loss_20260914/`。旧模型、旧训练入口、旧采样行为不变。

这是**现有跨尺度加速器的质量修复假设，不是新的RDD独有加速方法**。它没有新增推理开销，但可能只改善局部误差而不改善FID，甚至让其他t更差。短测试通过只证明代码能运行，不证明画质改善。256张开发图重复使用也不是新的独立测试集。原始1600权重和结果完整保留。

#### 同预算结果：局部loss下降，没有转化为可靠画质提升

| 配方（全部latest2400） | validation主头MSE | 加权主头MSE | Top-1 | Top-5 | 平均逐图latent相对L2↓ |
|---|---:|---:|---:|---:|---:|
| 原RDD，不做替代 | — | — | 68.75% | 87.11% | 0 |
| 原loss，有coarse | 0.0088845 | 0.0102604 | 62.11% | 82.03% | 0.247005 |
| 新loss，有coarse | 0.0088783 | 0.0100939 | 62.89% | 82.42% | 0.245524 |
| 原loss，清零来源 | 0.0099237 | 0.0115228 | 57.03% | 80.08% | 0.271308 |
| 新loss，清零来源 | 0.0098900 | 0.0113167 | 55.08% | 78.91% | 0.270278 |

Top-1/5是现有ResNet50 V2对指定ImageNet类别的命中率，只是语义诊断，**不是FID，也不证明图像自然**。latent L2按每张图先算范数比再平均，不等于视觉质量百分比。训练数据仍是4096条离线teacher轨迹，不是on-policy训练。

- 有coarse：新loss相对原loss的Top-1为**+0.78个百分点**，配对95%区间**[-1.95,+3.52]**；7张由错变对、5张由对变错。Top-5只多0.39个百分点，latent偏差少0.60%。没有可靠优于原loss的证据。
- 清零来源：新loss的Top-1反而少1.95个百分点。因此“新loss让有/无来源差距变大”部分来自清零变差，不能全部算到有来源方法的改善上。四组的来源×loss交互是+2.73个百分点，区间[-0.78,+6.25]，仍含0。
- 新loss有来源仍比原RDD的Top-1低5.86个百分点、Top-5低4.69个百分点。父模型best1600有来源Top-1为64.45%，继续到2400也没有可靠变好。**不再原样加倍训练。**
- 同预算中，有来源仍优于清零：原loss高5.08个百分点，新loss高7.81个百分点。这支持coarse信息有用，但不证明当前读取/替代架构已足够。

区间来自固定模型、固定256张开发图的逐图配对bootstrap；不是多次独立训练的区间，未做多重比较校正。选择ridge/模型和反复观察开发集也会限制结论，不能将这批图叫全新的独立测试集。

**审计：** 所有模型均为固定latest2400；原RDD的256张PNG与两种loss及1600轮参考逐字节相同；1536张PNG哈希已校验，8/16端点差异严格为0；四组前两批逐卡数据索引和t一致。全部结果、图、日志已下载。四个固定五列表如下，列顺序统一为“原RDD、原loss有来源、新loss有来源、原loss清零、新loss清零”。每片固定取前8张，不按效果筛选。

<details open><summary>固定片0：人物脸部、动物和机械细节仍有问题，新loss没有明显修复</summary><img src="../../eval_results/csra_prefix6_jump_loss_20260914/factorial_comparison_shard0.png" style="max-width:100%;width:1280px" alt="四组loss固定对比片0"></details>
<details><summary>固定片1：雪景、黑狗和蛇的构图大体保留，两种loss差别小</summary><img src="../../eval_results/csra_prefix6_jump_loss_20260914/factorial_comparison_shard1.png" style="max-width:100%;width:1280px" alt="四组loss固定对比片1"></details>
<details><summary>固定片2：有来源也可能丢掉草地上的鸟，摩托和多人场景改变</summary><img src="../../eval_results/csra_prefix6_jump_loss_20260914/factorial_comparison_shard2.png" style="max-width:100%;width:1280px" alt="四组loss固定对比片2"></details>
<details><summary>固定片3：海滩稳定，钢琴、人物脸部和篮球动作依然改变</summary><img src="../../eval_results/csra_prefix6_jump_loss_20260914/factorial_comparison_shard3.png" style="max-width:100%;width:1280px" alt="四组loss固定对比片3"></details>

图像观察只说明这些固定例子：两种loss的变化普遍小于有/无来源的变化；原RDD本身也有失败图。不能只拿稳定海滩说所有画质都达标，也不能把与原图不同的每个细节都当作生图失败。

可追溯文件：[四组配对统计及区间](../../eval_results/csra_prefix6_jump_loss_20260914/factorial_summary.json)、[模型/数据顺序审计](../../eval_results/csra_prefix6_jump_loss_20260914/completion_audit.json)、[系数校准](../../eval_results/csra_prefix6_jump_loss_20260914/calibration.json)。结果包`csra_prefix6_jump_loss_review_20260914.tar.gz`有1655个文件、200376025字节，SHA256为`5abfef6ac31bf70bb4dc9d46b1e7033c8f6cf43527be045b3108c7beba6894ff`。本轮不含新的FID或推理速度测量。

### 已完成诊断与800步训练：先让fine看清当前状态，再读取coarse；仍只算18层

现在的bridge直接看patch embedding，读取coarse之前没有任何fine内部的全局交互。一个可检查的假设是：**先执行一层真实fine block，让query带着当前高分辨率的信息，再决定怎样使用旧coarse。** 这不是宣称已经找到原因，也不是把“保留一层”本身包装成RDD专属创新。

| 配置 | 先计算 | 用bridge替代 | 后续正常计算 | 完整fine block总数 |
|---|---|---|---|---:|
| front0，原位置 | 无 | 第1–6层 | 第7–24层 | 18 |
| front1，新位置 | 第1层 | 第2–7层 | 第8–24层 | 18 |

表格为人类从1计数；代码是front0跳过0–5、front1跳过1–6。coarse仍为原16阶段t300的block13完整hidden。**新方案不是“原方案再多算一层”**，因此没有靠多一层计算偷换质量预算；但层数相同不等于实际延迟已经相同，成功后仍要重测视频。

先跑小型信息诊断，不立即开始长训练：用同192张拟合、64张选ridge、128张开发验证，在8个相同t上比较fine-only、正确coarse、空间错位coarse、错图coarse、全局均值coarse。ridge只从0.01/0.1/1选，不能用验证集选。原主干冻结，拟合一个线性读取器；这不是最终的条件attention bridge。

两位置的hidden目标深度不同，所以报告原始误差和相对目标/更新能量的归一化误差；更重要的是，把预测hidden接回各自未改的原模型后半段，**比较同一个最终主头输出**。每个真实teacher hidden接回后必须逐位复现旧teacher输出，front0的全部特征还必须与旧诊断逐位一致。这样不会只靠一个不同深度的hidden更容易拟合，就宣称生图更好。

诊断仅使用0–3卡；每进程分配器上限12%，至少12GiB空闲、25GiB磁盘才开始；未停止别的作业。入口：[诊断脚本](../../2d/attention_analysis/probe_csra_front_keep.py)、[四卡启动器](../../2d/exp/probe_csra_front_keep6_4gpu.sh)、[测试](../../tests/test_csra_front_keep.py)。诊断会话`rdd_csra_front_keep6_probe_0123_20260914`已经正常退出，输出和日志已下载，不再轮询它。

此项先回答“是否值得训练这个替代位置”，不回答最终FID、质量等价、已达到视频10%–20%加速或RDD独占性。新记录使用独立版本，不能悄悄拿来初始化旧prefix6学生。

#### 位置诊断结果：同最终输出目标下有改善，但不是全靠coarse

| 线性读取器可看到什么 | front0最终主头MSE↓ | front1最终主头MSE↓ | 相对减少 |
|---|---:|---:|---:|
| 只看当前fine＋时间/类别 | 0.105483 | 0.073801 | 30.04% |
| 加入正确coarse hidden | 0.077799 | 0.053014 | 31.86% |
| coarse在空间上错位 | 0.094919 | 0.064603 | 31.94% |
| coarse来自另一张图 | 0.102377 | 0.071531 | 30.13% |
| 只保留coarse全局均值 | 0.101886 | 0.072392 | 28.95% |

front1正确来源相对front0的主头MSE差值为-0.024786，128图配对区间[-0.025737,-0.023860]，每张图先平均8个t，128/128张下降。front1内部正确coarse相对fine-only低28.17%，127/128张占优；也优于参数张量同尺寸的错位和错图来源。fine-only比coarse组少约105万线性参数，不能仅凭两者差距就排除容量影响。

**重要反证也要保留：** fine-only同样改善约30%，所以“保留第1层”本身并非RDD独占技巧，也不能说31.86%全部来自跨尺度信息。准确结论是：这个替代位置更适合当前线性近似；在这个位置上，正确coarse还有额外可用信息。能否训练成真正有用的加速器，仍要看完整生成。

hidden的相对指标并非全部改善：正确来源的目标归一化MSE由0.123922降到0.094868，但按被替代更新能量归一化后，从0.131698升到0.193468。原因是第1层已经贡献了较大更新，剩余六层的目标更新能量更小。**不隐藏这一项，也不把不同深度的原始hidden MSE当作公平画质比较**；继续训练的依据主要是同一个最终输出目标的改善。

四片全部48个front0特征记录与原诊断逐位一致；从真实teacher hidden接回的输出误差严格为0，额外计算不改变CUDA RNG。拟合/接回测试每卡峰值allocated约1.081GiB。原始[两个位置统计与逐图区间](../../eval_results/csra_front_keep6_information_20260914/summary.json)、[front0](../../eval_results/csra_front_keep6_information_20260914/front0/metrics.json)、[front1](../../eval_results/csra_front_keep6_information_20260914/front1/metrics.json)已在本地。此处没有生成新视频或测速度。

#### 已完成的受控训练：front1，跳过六层，有/无来源各800步

| 项目 | 固定配置 |
|---|---|
| 基础模型 | 原ImageNet RDD-Unified-L/2，400K，300/300/400；不用FastWan冒充训练后RDD |
| fine路径 | 正常第1层 → 条件化coarse读取器代替第2–7层 → 正常第8–24层 |
| trainable | 原结构18个fine block、patch/time/class embedding、最终头，以及FP32仿射＋fine-Q/coarse-KV读取＋点式FFN |
| coarse路径 | 原8/16独立权重不更新，取原16阶段t300、block13的完整hidden |
| 来源对照 | 真实来源／清零来源；清零不删除读取模块计算；初始化两组输出完全相同 |
| 初始化 | 两组用同一front1 fine-only ridge；coarse仿射列和新增非线性输出层从0起步，不能加载旧位置的initializer |
| 数据 | 同4096训练/128开发teacher轨迹，每条8个t；与旧prefix6同seed20279000、数据顺序；不是on-policy |
| 优化 | 原主头MSE＋0.05方差MSE，不用刚失败的新loss；AdamW，fine LR1e-5、bridge LR1e-4、WD0.01、clip1 |
| batch/精度 | 0–3卡，4×64实际batch=256，无梯度累计；BF16 AMP、FP32主权重；activation checkpointing；TF32关 |
| 预算/选模 | 两组各固定800更新；保留best与latest，但配对pilot统一用latest800；不额外给某组训练步数 |
| 安全 | 启动需26GiB空闲/卡、40GiB磁盘；本进程分配器上限24%；不停止其他训练 |
| 实测短测试 | 两组各2次真实4×64更新已完成；107回归测试通过；四卡前两批数据索引、t、标签完全相同 |
| 正式运行检查 | 两组均完成800步；峰值allocated约8.977GiB/卡、reserved约10.61GiB/卡，无OOM；第一层与后17层均有梯度，被替代六层不训练 |
| 自动评估 | 已完成各模型同256张，DDPM80/80/90、CFG1、noise filling；原8/16端点逐位不变；固定前8/片grid，全768张PNG已下载并校验 |

已结束会话：`rdd_csra_front1_skip6_train_0123_20260914`，正常退出，不再监控或重跑。总日志：`/scr/yuheng/DevOpsBench/multiscale_diffusion/logs/csra_front1_skip6_train_20260914.log`。最终主头验证MSE为有来源0.013474、清零0.015011。

代码：[学生模型](../../2d/attention_analysis/csra_front_keep_student.py)、[训练器](../../2d/attention_analysis/train_csra_front_keep_pair.py)、[启动队列](../../2d/exp/train_csra_front1_skip6_4gpu.sh)、[配对生成](../../2d/attention_analysis/sample_csra_front_keep_quality.py)、[新增模型测试](../../tests/test_csra_front_keep_student.py)。新权重目录`checkpoints/RDD-CSRA-front1-skip6-{aligned,zero}-L400K-20260914/`，质量目录`eval_results/csra_front1_skip6_quality256_20260914/`。旧学生和checkpoint格式保持不变；旧采样器只增加可选loader入口，默认仍用旧loader。

#### 同800步、同256图：局部探针优势没有转成可靠的整体改善

| 模型 | Top-1↑ | Top-5↑ | 平均逐图latent相对L2↓ |
|---|---:|---:|---:|
| 原始RDD，不替代层 | 68.75% | 87.11% | 0 |
| 旧位置front0，有coarse，800 | 59.38% | 82.03% | 0.26260 |
| 新位置front1，有coarse，800 | 60.94% | 79.30% | 0.27088 |
| 旧位置front0，清零，800 | 53.13% | 75.39% | 0.29297 |
| 新位置front1，清零，800 | 51.56% | 75.00% | 0.29561 |

新位置有来源相对旧位置：Top-1增加1.56个百分点，配对95%区间[-3.13,+6.25]；Top-5减少2.73个百分点，区间[-6.64,+1.17]；latent偏差增加3.15%。**分类差异区间包含0，不能称为新位置更好。** 新位置内部，有来源比清零的Top-1高9.38个百分点，区间[4.69,14.06]，支持来源有用；但它仍比原RDD低7.81个百分点。两组之间的gap扩大也可能来自清零组变差，不能代替整体质量改善。

区间是固定checkpoint下对256图的配对bootstrap，不是多次独立训练的置信区间。这些图已经用于开发；Top-1/5不是FID或自然度，latent偏差也不是画质百分比。此次没有新增FID或视频计时。旧/新两套原始RDD的PNG和分类记录完全一致，不能将新800对旧1600/2400混为同预算比较。

**固定图人工检查：** 四张grid全部查看，每张固定展示每片最初8图、三列依次为原RDD/真实coarse/清零。白衣人物在部分例子中比旧位置好，但复杂人脸、商品结构、乐器和动物形状仍变；草地小鸟在有来源组几乎消失。场景、海滩、部分纹理较稳定，不能只选这些成功例。未逐张人工检查全部768张PNG。

<details open><summary>新位置800：第1片，固定8图，原RDD / 真实coarse / 清零</summary><img src="../../eval_results/csra_front1_skip6_quality256_20260914/shard_0/comparison_grid.png" style="max-width:100%;width:768px" alt="新位置第一片，包括人物和商品结构失败"></details>
<details><summary>新位置800：第2片，固定8图</summary><img src="../../eval_results/csra_front1_skip6_quality256_20260914/shard_1/comparison_grid.png" style="max-width:100%;width:768px" alt="新位置第二片，保留蛇和动物形状偏差"></details>
<details><summary>新位置800：第3片，固定8图，包括小鸟消失</summary><img src="../../eval_results/csra_front1_skip6_quality256_20260914/shard_2/comparison_grid.png" style="max-width:100%;width:768px" alt="新位置第三片，包含草地小鸟失败"></details>
<details><summary>新位置800：第4片，固定8图</summary><img src="../../eval_results/csra_front1_skip6_quality256_20260914/shard_3/comparison_grid.png" style="max-width:100%;width:768px" alt="新位置第四片，人物钢琴衣服细节仍偏离"></details>

旧位置同800图：[第1片](../../eval_results/csra_prefix6_quality256_20260913/shard_0/comparison_grid.png) · [第2片](../../eval_results/csra_prefix6_quality256_20260913/shard_1/comparison_grid.png) · [第3片](../../eval_results/csra_prefix6_quality256_20260913/shard_2/comparison_grid.png) · [第4片](../../eval_results/csra_prefix6_quality256_20260913/shard_3/comparison_grid.png)。[配对统计](../../eval_results/csra_front1_skip6_quality256_20260914/placement_comparison.json) · [完成与checkpoint审计](../../eval_results/csra_front1_skip6_quality256_20260914/completion_audit.json)。结果包799文件、100,574,336字节，SHA256为`d83c1fa5c16967d9a67d7f54a0e395a24cc4b652a2b1ed58b365a27ef9863fed`；没有下载大checkpoint。

#### 为什么没有改善：先排查时间覆盖，再检查读取器接口

**时间检查已完成。** 旧/新位置×有来源/清零×两个固定seed，每组16图，沿原teacher和各自student轨迹记录全部90个fine时刻，共23,040条记录。额外预测不改变RNG，原8/16端点和未插桩最终输出逐位一致；每卡峰值allocated2.883GiB。下表均为有coarse模型，越低越好。

| 相同teacher输入上的主头MSE | 旧位置800 | 新位置800 | 新位置变化 |
|---|---:|---:|---:|
| 全部90个t | 0.015554 | 0.016954 | +9.00% |
| 训练见过的8个t | 0.012305 | 0.013858 | +12.62% |
| 中间82个t | 0.015871 | 0.017256 | +8.73% |
| 中段t=201至101 | 0.015219 | 0.017932 | +17.83% |

32图中29图的全90时刻平均误差变差；配对差值0.001400，区间[0.000910,0.001926]。因此不是“训练点都很好，只在点之间失效”。新位置在student自身状态上的全时刻MSE为0.012941，低于teacher状态0.016954，依然没有看到单步误差爆炸；不同轨迹的状态难度不同，这不证明全局稳定或排除了on-policy的可能收益。8个锚点与82个中间点的绝对均值不能直接比较来证明插值不好，因为其时间分布不同。

![相同teacher状态与各自student状态的全部90时刻误差](../../eval_results/csra_front_placement_times_20260914/all90_front_comparison.png)

[90时刻数据汇总及原文件SHA](../../eval_results/csra_front_placement_times_20260914/summary.json) · [诊断代码](../../2d/attention_analysis/diagnose_csra_front_placement_times.py) · [汇总代码](../../2d/attention_analysis/summarize_csra_front_placement_times.py)。这是预测误差，不是FID，也不把各步误差相加冒充最终画质归因。

**接口检查也完成。** 在同128开发图×8个t上，保持各自800步模型不变，只替换一项输入；八个分片均完成，实际forward与拆分计算逐位相同、真实teacher hidden接回原后缀也逐位相同。这些是没有重新训练的反事实，不是可以直接部署的新模型。

| 单项干预，主预测MSE↓ | 旧位置front0 | 新位置front1 |
|---|---:|---:|
| 训练后模型、正确来源 | 0.012066 | 0.013476 |
| 同一模型，把来源清零 | 0.029740 | 0.026204 |
| 同一模型，换另一张图的来源 | 0.016279 | 0.017786 |
| 同一模型，来源网格平移一格 | 0.027478 | 0.026453 |
| 恢复原教师的fine前端输入，其他不变 | 0.012162 | 0.015108 |
| 恢复原教师前端输入及时间/类别condition | 0.022310 | 0.019982 |
| 直接提供教师正确的被省略段末端hidden，仍用训练后后缀 | 0.014424 | 0.006696 |
| 训练后读取器hidden，接回原教师后缀 | 0.041192 | 0.032876 |

这支持三点：①读取器确实利用了coarse内容及空间位置，并非完全忽略来源；②直接恢复原fine输入并没有救回误差，因此不能简单归因于第一层“被训坏”；③新位置若获得正确的中间状态，训练后后缀能把误差降约50.3%，提示读取器接口仍有改进空间。但这个oracle要真实计算被省掉的层，**不是免费加速方案**；不同模块互换有分布不兼容，表格不是因果贡献分解。

[128图接口汇总、条件区间和文件SHA](../../eval_results/csra_front_reader_audit_20260914/summary.json) · [干预代码](../../2d/attention_analysis/audit_csra_front_reader.py)。该实验不更新参数，不生成新图，不测速度。

#### 已完成的训练对照：先把读取器接稳，再让fine主干一起适应

直观想法：先让读取器学会给**固定的原模型后半段**提供有用输入，避免一开始读取器、输入表征和后半段同时变化；随后再一起调整。这只是从接口诊断得到的训练假设，不是已经有效的新贡献。以前的hidden辅助loss没有带来明显画质收益，本轮不重复加该loss，而使用同一个最终输出loss，只改变训练顺序。

| 对照 | 第1–800次更新 | 第801–1600次更新 | coarse来源 |
|---|---|---|---|
| 分阶段组 `staged` | 仅读取器更新，整个fine主干权重固定 | 读取器＋可执行fine主干联合更新 | 真实／清零各一组 |
| 直接联合组 `direct_control` | 已完成的旧front1联合800 | 恢复完整权重和Adam状态，再联合800 | 真实／清零各一组 |

四组都从相同fine-only初始化和原L/2开始，数据顺序、seed20279000、4096训练轨迹、8个t、loss、LR、WD和推理图相同；直接组复用已完成800，不重新浪费它。总更新数均1600，但fine主干更新数有意不同，这正是实验变量；不能称为每个参数训练预算相同。暖身阶段仍计算DDP反向，然后在clip和AdamW之前清空fine梯度，所以不更新其权重/weight decay/Adam状态；这不是训练算力优化。阶段边界对整个fine主干做逐位哈希验证。

仅0–3卡，4×64实际batch256，无累计；BF16 AMP/FP32主权重、TF32关、activation checkpointing。启动要求每卡26GiB空闲及50GiB磁盘，分配器上限24%；未停止其他任务。空间有限，本轮每组只保留latest和完整日志，最终固定latest1600评估，不挑最有利步数。两种训练的256图配对采样均已完成，配置仍DDPM80/80/90、CFG1、noise filling，固定32张/方法展示，全部PNG保留。

已正常退出的会话：`rdd_csra_staged_front_train_0123_20260914`。总日志：`/scr/yuheng/DevOpsBench/multiscale_diffusion/logs/csra_staged_front_train_20260914.log`。[启动脚本](../../2d/exp/train_csra_staged_front_4gpu.sh) · [训练器](../../2d/attention_analysis/train_csra_staged_front_pair.py) · [固定1600采样入口](../../2d/attention_analysis/sample_csra_staged_front_quality.py)。输出为`checkpoints/RDD-CSRA-front1-{staged,direct_control}-{aligned,zero}-L400K-20260914/`和`eval_results/csra_staged_front_{staged,direct_control}_quality256_20260914/`。

**完成核验：** 四组均为固定1600、非smoke；完整恢复直接联合组的800步权重和Adam；两组正式暖身到800时，整个fine主干SHA仍与原权重逐位相同。四组首次联合更新的数据索引、时间和标签匹配。110项回归测试和四组真实4×64短测试通过，正式过程无OOM。暖身allocated峰值6.411GiB/卡，联合约8.978GiB；采样约3.607GiB。八份采样指标、全部1536张PNG的SHA、基础权重/初始化/父checkpoint/运行代码SHA已检查；原8/16阶段输出差为0，原RDD的256图及分类结果与此前800步实验逐项一致。1620文件归档为201,177,327字节，SHA256为`28d11bb6604c60629d426999fe2b7a174846653e96f35d612cee55f294698abd`，不含大模型权重。

**预先限定判断：** 看同1600步的完整生成、失败例和配对语义指标，不因训练loss更低就称成功；也必须看分阶段是否仍有来源优势，而不只是通用蒸馏改进。若没有整体改善，不自动延长本配置或跑大FID。只有2D质量可靠后，才测同一新实现的实际视频净延迟与质量；不能沿用旧bridge的13.9%计时充数。

##### 同1600步结果：分阶段反而更差

这里的Top-1/5是固定ResNet50-V2对生成图的目标类别识别率，不是FID或“画质百分比”。latent相对L2越低，只说明与同seed原RDD更接近，不保证真实感。

| 方法 | coarse来源 | 总更新数 | Top-1↑ | Top-5↑ | latent相对L2↓ |
|---|---|---:|---:|---:|---:|
| 原RDD，无跳层 | 不适用 | 基础400K | 68.75% | 87.11% | 0 |
| 直接联合 `direct_control` | 真实 | 1600 | **61.72%** | **77.73%** | **0.26047** |
| 分阶段 `staged` | 真实 | 1600 | 57.81% | 77.34% | 0.27080 |
| 直接联合 `direct_control` | 清零 | 1600 | 56.25% | 76.56% | 0.28086 |
| 分阶段 `staged` | 清零 | 1600 | 53.13% | 74.22% | 0.28948 |

- 有coarse时，分阶段相对直接联合的Top-1为**−3.91个百分点**，图像配对bootstrap 95%区间`[−7.42, −0.39]`；latent偏差约大3.97%。不是这轮质量改进。
- 直接联合中，真实coarse比清零高**5.47个百分点**，区间`[+0.78, +10.55]`。来源有帮助，但有来源仍比原RDD低7.03个百分点。
- 分阶段并没有额外放大来源收益：来源×训练顺序交互为−0.78个百分点，区间`[−5.86, +4.30]`。
- 单看直接联合从800到1600，有来源Top-1仅+0.78个百分点，区间`[−3.13, +4.69]`，Top-5反而−1.56个百分点。因此不能说“多训练已经解决了”。

上述区间只在固定训练seed及反复使用的256张开发图上重采样；不能估计重训随机性，也未做多重比较校正。本轮不新增大FID或视频训练。[四组完整统计与配对区间](../../eval_results/csra_staged_front_review_20260914/factorial_summary.json) · [完成核验及权重SHA](../../eval_results/csra_staged_front_review_20260914/completion_audit.json)。

##### 全部固定展示：五列放在一起比较

从左到右恒为：**原RDD → 直接联合1600＋真实coarse → 分阶段1600＋真实coarse → 直接联合1600＋清零 → 分阶段1600＋清零**。每张是对应shard最前8个固定样本，没有挑成功例；五列图合并了原来的八张三列grid，PNG内容未修改。

![同1600训练顺序与来源对照，shard0](../../eval_results/csra_staged_front_review_20260914/comparison_shard0.png)

![同1600训练顺序与来源对照，shard1](../../eval_results/csra_staged_front_review_20260914/comparison_shard1.png)

![同1600训练顺序与来源对照，shard2](../../eval_results/csra_staged_front_review_20260914/comparison_shard2.png)

![同1600训练顺序与来源对照，shard3](../../eval_results/csra_staged_front_review_20260914/comparison_shard3.png)

检查全部八张原grid后：房间、海滩、布料等大布局通常保留；但金色瓶盖/容器形状、动物脸与角、钢琴键盘、人物脸和肢体等仍改变。shard2的草地鸟在两种有来源结果中都接近消失；shard3钢琴键盘明显漂移。个别样本分阶段更接近原图，不构成整体改善。原RDD本身也有畸形或文字不清，不能把其原有失败都归因于跳层。

##### 接着做的诊断：是不是只缺区域内部细节？结果不支持这样简单归因

已完成：四卡各512个状态，共128张既有开发图×8个t×真实/清零两种**直接联合1600**模型。无训练、无新图、无权重写入，峰值allocated约2.780GiB/卡；原前向和手动后半段逐位相同，原教师hidden接原后半段也逐位复现，CUDA随机状态未变化。新增测试后共113项通过。

做法很直观：一个coarse token对应fine的2×2个token。把读取器与原教师的中间特征差，分成“这4个位置的平均差”和“各位置相对平均值的差”。前者在格子内相同，后者平均为0。分别用教师答案补回其中一部分，再运行**同一训练后后半段**。这不是理想FFT低通，也不是推理时能免费得到的答案。

| 桥接位置输入 | 有coarse：最终主头MSE↓ | 比不补回变化 | 清零模型：主头MSE↓ |
|---|---:|---:|---:|
| 训练后读取器原输出 | 0.0116684 | 基准 | 0.0125631 |
| 只补回教师的2×2区域均值 | 0.0189727 | **+62.60%（更差）** | 0.0203514 |
| 只补回教师的区域内部差异 | 0.0257116 | **+120.35%（更差）** | 0.0266387 |
| 完整换为教师中间特征 | 0.0086723 | −25.68% | 0.0097555 |

有来源时，区域均值占hidden误差能量约65.21%，尽管只占25%的自由度；所以不能说误差主要都是细纹理。更关键的是：**单独补一部分不代表下游必然变好**。后半段已适应读取器表征，混合教师与学生的一部分特征可能形成它没见过的输入；后半段的非线性使两部分效果不能相加。两个单独替换各使127/128张图的时刻平均误差变大，完整替换使113/128变小。只能说“整体接口仍有改善空间”，不能据此声称哪种频率是因果瓶颈。

[分解、逐图配对统计与限制](../../eval_results/csra_bridge_bands_20260914/summary.json) · [只读诊断代码](../../2d/attention_analysis/audit_csra_bridge_bands.py)。会话`rdd_csra_bridge_bands_0123_20260914`也已正常完成；不应继续轮询它。

##### 下一步：小型联合修正假设，而不是再换训练顺序

尚未启动新训练。下一轮优先测试：保留已有coarse全局读取，给当前fine特征加一个低通道数的邻域交互分支，并让它与读取器及后半段联合学习。它可以同时调整区域均值和区域内部差异，**不硬把某一频带换成教师答案**。现有读取器主要是fine查询coarse及逐token FFN；邻域分支要测试的是当前fine相邻位置之间的便宜交互是否有用，不是已经证实缺高频。

先用真实FastWan执行路径测这项额外开销，仍与同RDD+VSA比较、保留coarse输出和gate；若计入全部开销后没有≥10%的净延迟下降，就不投入长训练。通过后才做短的“有/无新分支×真实/清零coarse”对照，保持相同起点、更新数、数据和采样。普通卷积/局部交互本身可用于全分辨率模型，不是RDD独占创新；需要证明的是它与已计算的跨尺度信息组合，能用较少fine计算保住质量。此前所有不成功的配置保留，不继续无条件延长。

### 新补充的相关研究：有参考价值，但不是新的RDD专属贡献

**DisCa**用小型DiT预测器读取缓存，结合预测MSE和特征空间对抗训练；它提示仅优化逐点误差未必保住细节，也提醒视频缓存布局和多卡访存本身有成本。它主要是同分辨率跨时间缓存，不能算RDD独有；其步数/CFG蒸馏叠加速度也不能当成本项目同RDD基线下的增量。[论文§3.3、§8.5](https://arxiv.org/html/2602.05449v3) · [官方代码](https://github.com/Tencent-Hunyuan/DisCa)

**SeaCache**用按时间变化的频谱过滤来衡量特征变化，从而决定何时刷新缓存；它不是把低分辨率stage输出送到高分辨率。可借鉴“不要把高频噪声变化直接当内容变化”的诊断思路，但不能据此自动推导RDD最佳边界，也不作为满足专属性的新方法。[作者项目与方法说明](https://jiwoogit.github.io/SeaCache/)

**WaiT（2026年7月预印本）**也先生成低频，延迟高频的噪声日程，再联合细化；其Haar低频与平均池化有直接联系。它没有在这部分方法中提出本实验的跨stage hidden前缀替代，但提醒我们：“低频先做、高频后做”本身不能作为新增独创贡献。可以借鉴分频带评价，不能直接把它的阈值、归一化或像素FM日程搬到现有RDD latent DDPM。[论文§3.1–3.3](https://arxiv.org/html/2607.28760v1#S3)

**DiSC（2026年5月预印本）**同时利用跨时间token缓存与稀疏mask复用，并设计专用硬件数据通路。它不是利用真实低分辨率stage，也不是可直接照搬的CUDA kernel；其硬件速度用周期级模拟器、RTL综合等评估。可借鉴“索引/访存开销必须一起算、稀疏布局要匹配硬件”，但不能拿专用硬件的加速数字证明我们在同一GPU上能更快，也不算RDD专属方法。[论文§III–V](https://arxiv.org/html/2605.25798v1)

## 9. 验证顺序与停止条件

| 顺序 | 实验 | 成功标准 | 不应怎样解释 |
|---|---|---|---|
| 已完成，未通过质量检查 | 联合训练 vs 新冻结对照，800 步/组 | 单步验证改善，但完整生成偏差均值增加17.38%，见2.5 | 阶段互换指向coarse路径变化，见2.6 |
| 已完成，有小幅改善但未修复脸部 | 固定原coarse、训练fine主干 vs 对照 | 七组latent偏差均值下降5.00%，见2.7 | 两套权重有显存代价；尚无视频净收益结果 |
| 已完成，不作为有效改进 | V4逐通道输出修正，两种训练范围对照 | 相对V2偏差变化小于0.1%，脸部未修复，见2.9 | 不把新增参数当贡献，不继续堆V4预算 |
| 已完成，改善有限 | 简单V2匹配续训800→1600 | 平均latent偏差再降1.91%，脸部仍未恢复 | 不自动继续3200，见2.10 |
| 已完成，来源优势未获支持 | 无coarse来源也联合训练1600 | 平均latent偏差0.176095 vs有coarse0.177146，验证误差也相近 | 不将普通小网络压缩解释为已证明RDD特有收益 |
| 已完成 | coarse来源干预，见2.11 | 正确来源优于错位/换图/清零 | 扰动敏感性不等于充分训练的无coarse模型更差 |
| 小规模诊断 | 父格级 error 集中性、coarse 评分 vs random/oracle | 存在可预测、值得省计算的区域 | oracle 精度不是可部署精度 |
| 已有算子原型 | 小网格 coarse 计算与 fused lift-add | 等价性、缓存/setup/搬运全部计时 | 分支 2× 不等于视频 2× |
| 质量有希望后 | 相同 mask/层数的实际视频 adapter | 相对同样 RDD+VSA，净 denoiser 延迟至少降 10%，视频内容合理 | 不能改 VSA coarse 分支、offload、步数混淆贡献 |
| 进一步才做 | active fine refinement / 分解式 refinement | matched control 表明 coarse 来源有帮助 | 不能用普通剪枝结果冒充 RDD 专属效果 |
| 已完成，未得到稳定来源优势 | V5当前fine Q读取coarse记忆，各800步 | aligned/zero平均latent偏差0.181276/0.179628，人物仍失败 | 不继续扩训；新增attention不是成功证据 |
| 已完成，最早时刻收益很小 | V2/V5按时间、误差频带比较 | t299主预测误差少2.62%/3.13%，后半段基本无优势 | 不把单步误差下降当画质/加速证明 |
| 已完成，无稳定来源优势 | V6完整coarse h vs零，两组同初始fine学生 | 七组平均latent偏差0.174397/0.173949；脸部未修复 | 不继续自动增加预算 |
| 已完成，不支持直接旧KV替换 | 原模型保留fine MLP，仅替换attention K/V | 旧KV误差约同时间pooling的2.36倍，换位置/时间未解决 | 无FID/速度结论，dense逐位parity为0 |
| 三组预览已完成 | 旧KV/同时间pooling，全90步与仅前20步 | 短窗口明显更保守，但当前pooling同窗口三组均更接近dense | 这是质量诊断，完整QKV仍计算，不报加速 |
| 已完成，有信息优势 | 在昂贵fine前缀之前利用coarse完整h | hidden MSE比无来源少13.64%，主要是局部平均误差；正确来源也优于等参数错配组 | hidden误差仍大，不是画质或速度结果 |
| 只读检查已完成，有来源优势 | 线性前缀接入原fine后10层 | 1024状态oracle逐位parity；正确coarse主MSE少21.58%，但绝对误差仍大 | 不是完整采样/FID/速度；t0不占优 |
| 已完成，有来源优势但质量失败 | V7替代fine前14层，训练转换器和后10层 | 各800步、3组生成；有来源主MSE少17.12%、平均latent偏差少9.96%，人物仍严重变形 | 不报质量达标/视频加速，不盲目扩训14层 |
| 已完成，仍未达质量目标 | 新8层前缀，保留后16层，两组各1600步 | 有来源主MSE低12.02%、平均latent偏差低4.31%，仅2/3组占优；部分人脸恢复但另外人物仍失败 | 没有独立质量评估或净视频速度；不盲目扩大相同训练 |
| 已完成 | 已训练prefix8分时间/局部空间误差 | 来源优势从t299的34.44%降至t40的6.89%；归一化后两部分误差总体相近 | 不能只归因缓存老化或高频不足 |
| 已完成，未通过质量筛选 | prefix8只在前45/60 fine NFE使用 | 更保守，但2个seed仅1个来源组更好，人物仍失败 | 不配用全90步的计算节省，不报视频速度 |
| 已完成，未通过质量筛选 | 条件化全通道前缀读取+非线性修正，各1600步 | 三seed有来源均较接近dense，平均偏差少8.38%，但人物仍失败 | 没有FID或净视频速度；升级容量与结构是混合改变 |
| 已完成，未见单步误差爆炸 | 两seed匹配teacher状态/学生状态预测误差 | 学生状态上误差更低，仍累积轨迹偏离 | 不据此许诺on-policy训练一定有效或全局稳定 |
| 已完成，排除方差头直接影响 | 学生/teacher主头与方差头互换＋90时间点CPU审计 | jump采样器不用网络方差头；主头空间两分量响应不同 | 不是所有DDPM都不用方差，也不是全局质量因果分解 |
| 已完成，不继续原样加倍 | 条件化prefix8两组1600→3200 | 主MSE降低但有coarse平均latent偏差仅低0.71%；来源优势缩至4.85% | 保真度不是FID；不是新架构，不无限扩训 |
| 已完成 | 全部90时间点预测误差 | 未见中间时间点普遍倍数级失效 | 插值参照不是理论真值，不能强判原因 |
| 已完成，修正评价口径 | 旧图真实label＋已有分类器检查 | 人物并非这些样本的必需条件主体 | 8图类别分数不是画质证明 |
| 已完成，有来源优势但低于原RDD | 256个新样本/方法、三方法匹配pilot | Top-5原RDD91.41%/有coarse81.64%/无coarse75.39%；全部四张固定grid已查看 | 类别分数不是FID，不作速度结论 |
| 已完成，未稳定改善 | 正常学生与恢复前8层的学生，三seed五列 | 人物结构部分回来但纹理伪影增加，latent偏差改善不一致 | 恢复链未联合训练且更贵，不是可部署加速或单一原因证明 |
| 已完成，无实质性改善 | 前缀hidden对齐辅助监督及等预算对照 | 四组best4000、六组预览；有来源最终偏差仅低0.25%，无来源反而高0.14% | 不将hidden loss下降16.85%说成画质提升16.85% |
| 已完成，有来源优势但未达质量目标 | 条件化前缀6，保留18层，256个新样本/方法 | Top-1原RDD68.75%/有coarse59.38%/清零53.13%；来源差6.25pp，区间[1.17,11.33]pp | 不是FID；不同seed的prefix8 pilot不能直接作深度因果对比 |
| 已完成，计算可行性通过 | 完整条件bridge替代FastWan fine前6/8层 | 384次计时，1280前6层延迟下降13.92–15.25%，前8层19.21–20.93%，所有新增开销计入 | 未训练video bridge、共享GPU、不含VAE，质量与净10%尚未同时达成 |
| 已完成，有改善但未修复 | 固定prefix6两组各800→1600及三组预览 | 有来源平均latent偏差少6.61%，清零少4.44%；本轮只在2/3seed有来源占优 | 鸟头/牛头等仍失败；不把主MSE少18.70%说成画质提升18.70% |
| 已完成，续训语义改善 | 同一批256开发样本的best800→best1600匹配检查 | 有coarse Top-1升5.08pp，Top-5不变；所有原图SHA相同 | 仍有细节/主体失败，不代表达到质量等价 |
| 已完成，来源有效但质量未达标 | 固定原RDD/有coarse/清零各10K图的五指标评估 | FID11.8755/16.8040/20.3236；完整审计与所有五项结果已记录 | FID10K不能直接对比已发表50K结果；没有视频质量结论 |
| 已完成，支持有界loss对照 | 当前prefix6在teacher/学生轨迹上的全部90个fine时间点误差 | 无单步误差爆炸；后29个有效调用占约六成局部误差和；全部插桩parity为0 | 不是最终误差归因；高频占比须考虑75%子空间自由度 |
| 已完成，无可靠画质提升 | 原loss/采样器加权loss×有/无来源，1600→2400 | 有来源Top-1仅+0.78pp，区间含0；latent偏差少0.60%，所有结果已审计 | 不原样延长训练，不据此追加大FID |
| 已完成，值得做训练对照 | 先保留一层fine，再替代六层，18层预算不变 | 线性读取接回后的主MSE低31.86%；fine-only也低30.04%；正确来源仍有增益 | 不是31.86%画质/速度提升，也不是已证明RDD独占性 |
| 已完成，没有可靠整体改善 | front1条件读取器，有/无来源各800步＋同256图 | 有来源Top-1 60.94%，Top-5 79.30%；比旧位置同800的latent偏差高3.15% | 来源仍有用，不把线性探针31.86%写成画质收益 |
| 已完成定位诊断 | 新/旧800的全部90t，以及128图来源/前端/后缀干预 | 训练锚点也变差；来源清零/错位均差；新位置oracle中间hidden可显著降误差 | 反事实并非可部署模型，不作FID归因 |
| 已完成，分阶段无改善 | 分阶段与直接联合各1600，两种来源，共四组；同256图 | 有来源直接61.72%／分阶段57.81% Top-1；全部1536PNG及原始数据已下载 | 不延长失败训练；不是只差视频训练 |
| 已完成，单频带补回失败 | 128图×8时刻×两种来源，教师hidden分区补回 | 两种局部补回都增大输出误差，整体替换才改善 | 教师/学生混合输入可能分布外；不是可部署加速或因果归因 |
| 已完成，未见可靠质量改善 | coarse全局读取＋当前fine低成本邻域联合修正 | 384次未训练video计时有计算空间；四组固定2400完成，真实来源Top-1邻域开/关61.72/62.89% | 分类差异区间含0；不能称为有效新方法或训练后video质量 |
| 已完成，回放未改善质量 | 同一reader、同预算，teacher-only对50%学生快照回放 | 固定400更新、4×64；256图×四方法，1024PNG | 回放Top-1比teacher-only低5.86个百分点；不自动继续延长 |

如果联合训练仍然过拟合，先看实际 rollout 输入分布和每层容量，而不是无限增加更新数。需要扩充轨迹或改变网络时，为每项保留匹配对照；不能同时改变 rank、样本数、层数、学习率后再说是“全模型训练”的功劳。

## Sources

1. Ivan Skorokhodov 等，*Hierarchical Patch Diffusion Models for High-Resolution Video Generation*，CVPR 2024，§4.2–4.5、Appendix D。[论文](https://arxiv.org/html/2406.07792v1) · [项目](https://snap-research.github.io/hpdm/)
2. *Training-Free Efficient Video Generation via Dynamic Token Carving*（Jenga），2025，§3.1–3.2。[论文](https://arxiv.org/html/2505.16864v1)
3. Peiyuan Zhang 等，*Faster Video Diffusion with Trainable Sparse Attention*，2025，§2.2–2.4。[论文](https://arxiv.org/html/2505.13389v1) · [FastVideo 官方代码](https://github.com/hao-ai-lab/FastVideo)
4. Zekun Li 等，*SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration*，2026。[论文](https://arxiv.org/html/2602.04361v1)
5. *Improving Progressive Generation with Decomposable Flow Matching*，2025，§3.2–3.3。[论文](https://arxiv.org/html/2506.19839v1)
6. Xinle Cheng、Zhuoming Chen、Zhihao Jia，*CAT Pruning: Cluster-Aware Token Pruning For Text-to-Image Diffusion Models*，2025。[论文](https://arxiv.org/abs/2502.00433)
7. *Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration*，NeurIPS 2025。[官方代码及论文入口](https://github.com/yhlee-add/THG)
8. Triton 官方文档，*Fused Softmax*，说明减少中间张量内存读写的融合设计原则；这里没有引用其 benchmark 作为本 kernel 的速度。[文档](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
9. Zihao Wu，*InvarDiff: Cross-Scale Invariance Caching for Accelerated Diffusion Models*，2025预印本，§3.1–3.4、Figure3。[论文](https://arxiv.org/html/2512.05134v1)
10. Shenghai Yuan等，*Helios: Real Real-Time Long Video Generation Model*，2026预印本，§3.3.2。[论文](https://arxiv.org/html/2603.04379v1)

本项目实验来源：现有 CSRA condition/repair/capacity 原始记录，以及本页列出的新训练、源数据重放、算子和父格探针脚本。所有当前生成与 loss 数值均来自本地/服务器记录，不是文献数据。
