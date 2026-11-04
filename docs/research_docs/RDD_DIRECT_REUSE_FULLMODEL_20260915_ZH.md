# 不加 adapter：直接跨尺度复用＋整个 RDD 模型微调

更新日期：2026-09-15。本页与旧 adapter/reader 实验分开记录，避免混用结果。

> 后续已按要求加大复用比例，新增[第二轮：50%/75%图片复用、视频级联复用与采样＋VAE的10%目标](RDD_AGGRESSIVE_DIRECT_REUSE_20260915_ZH.html)。本页仍是第一轮200步结果，不要把两轮数字混在一起。

**第一轮已经跑完，不是待运行的计划。** 已完成两个200-step整网微调实验、64组配对图像（共256张）以及3种视频设置（共12段视频）。结论是：**整网微调能部分修复直接复用，但还没有恢复原RDD的图像质量；视频去噪延迟减少约11%–26%，含当前FP32 VAE后只减少约1%–3%。** 没有新的FID，也没有训练视频模型。这是可运行的方向验证，不是已经完成的保质加速方案。

## 1. 到底改了什么

只保留一个 RDD 模型。不再增加 adapter、reader 或第二个 fine 模型。

coarse 阶段真正算过第 l 个 block 后，保存这个 block 对 hidden 的**更新量**：

$$
\Delta_l^{\rm coarse}=h_{l+1}^{\rm coarse}-h_l^{\rm coarse}.
$$

fine 阶段遇到选定的同一层，就不再运行这个 block，而是：

$$
h_{l+1}^{\rm fine}=h_l^{\rm fine}+U(\Delta_l^{\rm coarse}).
$$

U 是 nearest 上采样，不是一个需要训练的网络。**没有把 fine hidden 整个替换为 coarse hidden。** 当前 fine hidden 的细节仍然保留，后面正常执行的 fine blocks 还能继续修改它。

```text
coarse 第 l 层：输入 ──真正计算 block── 输出
                         │
                    保存 输出−输入
                         │ 上采样
fine   第 l 层：输入 ─────加上更新量──── 输出
                  不运行这一层的 QKV / attention / MLP
```

这不是“找一个更快的 adapter”，而是直接去掉 adapter 的计算开销。也不是一个新 CUDA kernel：节省来自**整层不执行**。在视频中，被省掉的是完整 Wan block，包含 self-attention、text cross-attention 和 MLP；没有被选中的层仍完整执行原生 VSA 和 gates/coarse output。

最重要的训练变化：**同一套原模型参数同时生产 coarse 更新量、处理 fine 特征，全部允许更新；fine 的梯度会沿缓存回到 coarse 计算路径。** 缓存不是提前算好的固定 hidden，也没有在训练中 detach。

需要纠正一个历史印象：旧实验并非全部“只训练 adapter”。部分版本已经训练 fine backbone，更早的 joint 版本也试过 live coarse＋adapter。此次的实质区别是**完全取消 learned bridge**，用最直接的残差加法，并重新验证共享模型训练。

## 2. 为什么分散跳层，而不是整段覆盖

ImageNet 模型有 24 个 blocks，第一轮复用 6 个，索引从 0 开始为：`3,6,10,13,17,20`。首尾层保留，复用层之间还有正常的 fine block。

这样允许模型反复交替做两件事：继承 coarse 更新，再由当前分辨率的真实 block 修正细节和 timestep 差异。**保留 fine 层只是提供修复机会，不保证画质必然恢复。**

还提供了复用 8 层的代码配置：`2,5,8,10,13,15,18,21`。它还不能在没有训练/采样结果时被称作“更好的配置”。

训练时，middle 输入来自原模型真实采样中最后一次 16×16 调用（全局 t=300）；fine 输入来自同一张图的真实 32×32 轨迹。每个训练 batch 都由**当前 student** 重新计算 middle 更新量，再给 fine 使用。因此 source 时间和实际采样缓存时间对得上，不是把 clean low-res 图片冒充缓存。

## 3. 训练和比较配置

| 项目 | 本轮设置 |
|---|---|
| 起点 | 原生 ImageNet 的 RDD-Unified-L/2，300/300/400，400K checkpoint |
| 模型 | 24 blocks，hidden 1024，约 4.578 亿参数 |
| 新参数 | **0**；没有 adapter/reader |
| 更新范围 | 原 backbone 全部可训练，包括 coarse 更新量的生产者 |
| 正式 batch | 每个实验单卡直接 **256**，不拆 batch、不做梯度累积；0 卡复用整网微调，1 卡同预算 dense 控制，2/3 卡视频验证，合计最多四卡 |
| 显存策略 | activation checkpointing，重新计算部分激活；不拆小 batch |
| 优化器 | AdamW，lr=1e-5，betas=(0.9,0.999)，weight decay=0.01，梯度裁剪 1 |
| 精度 | FP32 参数和 Adam 状态，BF16 autocast，TF32 关闭 |
| 训练输入 | 已审计的 4096 张 teacher 采样轨迹，每张 8 个 fine 状态 |
| 验证 | 独立的 128 张 teacher 轨迹，不与训练 seeds 重叠 |
| 条件 | CFG=1 的条件蒸馏；关闭随机 label dropout；尚未验证 CFG=1.5 |
| 初轮预算 | 复用 6 层整网微调 200 updates；不复用的整网微调控制组也 200 updates |
| 生成比较 | 原 RDD、直接复用未微调、直接复用整网微调、不复用整网微调控制组 |
| 图像 sampler | DDPM，80/80/90，共 250 NFE，noise_filling，CFG=1 |

损失是“学会复用”与“别把原来的 coarse 模型改坏”两部分：

$$
L=L_{\rm fine}+\tfrac12(L_{16}+L_8),\qquad
L_r=\operatorname{MSE}(\epsilon_\theta,\epsilon_T)+0.05\operatorname{MSE}(v_\theta,v_T).
$$

这里 v 指网络的 **learned variance 输出通道**，不是 flow velocity。三个目标都来自原模型。6 个被跳过的 fine blocks 仍然在 coarse 路径实际执行，也接收 fine 损失的梯度。

**这轮是 teacher 轨迹上的全模型预测蒸馏，不是重新用真实 ImageNet 做原生 diffusion loss 训练，也不是对完整 250 步反向传播。** 输入仍来自 teacher，微调后自身 rollout 会偏离它。即使验证 MSE 降低，也必须看真正生成的图片，不能因此宣称 FID 更好。

## 4. 已通过的实现检查

- 不加任何新参数；模型 checkpoint 仍保存原 backbone 的 state_dict。
- CPU 小模型：无复用与原 forward 逐元素一致。
- CPU 梯度测试：只反传 fine 输出损失，被 fine 跳过的层仍可经 coarse 路径获得非零梯度。
- 真实 Unified-L/2：每卡 64 的 forward/backward/Adam 两步已通过，所有参数都有梯度；峰值约 **8.97 GiB**。这仅是显存 smoke test，不是训练完成。
- 每卡128测试也通过，峰值约11.04GiB。但双卡 NCCL 失败；独立于模型的最小 all-reduce 程序也复现了 illegal memory access。未改动共享环境或其他任务，改为两个独立单卡完整batch256实验。该通信错误不能当成模型算法失败。
- **单卡直接256的测试也通过。** 正式复用组峰值约15.20GiB，dense控制约15.95GiB。没有训练adapter，没有冻结coarse producer，也没有梯度累积。
- 视频测试要求无复用 controller 与原 FastWan 整条轨迹逐元素一致，并核对真实跳层次数。

## 5. 图像结果

### 5.1 整网微调修复了多少

四个版本使用相同64个seed/类别、同样的250NFE、CFG=1与noise_filling。下表由固定ResNet50分类器检查生成图片是否符合指定的ImageNet类别。**它不是FID，不衡量全部感知质量，也不能与论文的FID表比较。**

| 版本 | 正确类别Top-1 | 正确类别Top-5 | 目标类别平均概率 |
|---|---:|---:|---:|
| 原RDD，不复用 | **78.13%（50/64）** | **92.19%（59/64）** | 0.3303 |
| 直接复用6层，未微调 | 32.81%（21/64） | 53.13%（34/64） | 0.1142 |
| **直接复用6层，整网微调200步** | **51.56%（33/64）** | **71.88%（46/64）** | **0.2147** |
| 不复用，同预算整网微调200步 | 75.00%（48/64） | 92.19%（59/64） | 0.3344 |

微调相对生硬复用提高18.75个百分点，但仍比原RDD低26.56个百分点；后一个差值的64样本配对bootstrap区间为[-39.06,-15.63]个百分点。**不能把“有修复”写成“已经基本保质”。** dense控制基本保住原来的类别一致性，说明复用组的明显退化不能简单归因于“只要继续训练都会变差”。

| teacher轨迹验证MSE | 未微调复用 | 整网微调200步 | 如何理解 |
|---|---:|---:|---|
| fine epsilon误差 | 0.05444 | **0.01850** | 缓存替代后的预测更接近teacher |
| middle epsilon误差 | 0.0000511 | **0.002725** | 共享参数也改变了coarse生产者 |
| coarse epsilon误差 | 0.0000342 | **0.000532** | anchor没有完全保持原输出 |

后两行是一个值得警惕的信号：训练在修复fine的同时，也移动了coarse的预测。它**尚不是退化的因果证明**。此外只训练了200步，训练输入仍是teacher轨迹，而实际采样输入是student自身滚出来的；训练充分度、输入分布差异、coarse保持力度都尚待分别验证。

### 5.2 真实采样速度

原RDD与整网微调复用版都按相同BF16推理设置执行；batch=8，完整250NFE，先暖启动，再做4组ABBA/BAAB配对。包含缓存生成、上采样和noise_filling，**不含VAE**。

- 去噪延迟平均下降 **5.90%**，约 **1.063×**；组bootstrap区间[5.54%,6.37%]。
- 原版每批约13.7–14.0秒，复用版约12.9–13.1秒；不是每张图片的时间。
- 实际跳过 `6×90=540` 次fine block调用；并不是整个250步都跳过25%的模型。
- 2D没有新增足够重复的采样＋VAE总时间配对测试，因此这里不编造整条图像生成流程的加速。

为什么没有25%的整模型收益？只跳最后90步的部分层，coarse和middle仍要计算，缓存/采样也有开销；2D短序列下减少运算并不等比例减少延迟。**本轮没有做算子profiler，不能把剩余差距全部断言为某个kernel瓶颈。**

### 5.3 直接看图，不挑最好看的样本

每张图从左到右：**原RDD → 直接复用未微调 → 直接复用整网微调 → 不复用整网微调控制**。每个分片固定展示前8个样本，共32组；全部64组的256张PNG也已经保存在本地。第3列可以看到狗、头盔、龟等结构有所修复，但鸟翼、脸和不少物体形状仍与原版不同。

![固定分片0：四种方法对比](../../eval_results/direct_reuse_fullmodel_20260915/image_preview/shard_0/comparison.png)

![固定分片1：四种方法对比](../../eval_results/direct_reuse_fullmodel_20260915/image_preview/shard_1/comparison.png)

![固定分片2：四种方法对比](../../eval_results/direct_reuse_fullmodel_20260915/image_preview/shard_2/comparison.png)

![固定分片3：四种方法对比](../../eval_results/direct_reuse_fullmodel_20260915/image_preview/shard_3/comparison.png)

[全部结果、训练配置与文件SHA审计](../../eval_results/direct_reuse_fullmodel_20260915/summary_and_audit.json)。复用组实际训练约13.7分钟，dense控制约16.9分钟；两者都是相同200次更新和batch256，不是相同墙钟预算。

## 6. 视频怎么测

| 项目 | 设置 |
|---|---|
| 模型 | FastWan2.1-T2V-1.3B，原生 VSA blocks，约 1.49B 实际参数 |
| RDD 分辨率 | Spatial `(4,1)→(2,1)→(1,1)`；另测较温和Temporal `(1,2)→(1,2)→(1,1)` |
| 输出 | 832×448 或 1280×768，29 帧，16 fps |
| sampler | DMD 3 NFE，t=1000/757/522，noise_filling |
| baseline | 同样的 RDD＋原生在线 VSA，保留 20% |
| 候选 | 只在最后 fine NFE 直接复用 middle 的 6/8/10 个 block 更新量 |
| 视频层数 | 30；6 层 mask=`4,8,12,16,20,24`；8 层=`3,6,9,12,15,18,21,24`；10 层=`2,5,7,10,13,15,18,21,23,26` |
| 其他加速 | 不使用 C2F-Route，不使用 StepReuse-Out，不使用 SmallQ，不加 adapter |
| 公平项 | 同一驻留模型、VSA/gates、seed、prompt、输出形状、精度与 offload 策略 |
| 计时 | 所有形状先暖启动；Spatial各6组、Temporal 4组 ABBA/BAAB；每次都实际decode VAE |
| 总时间含义 | 采样＋真实 FP32 VAE；不含文本编码、模型载入、编译、MP4 写盘 |

**FastWan 权重没有经过 RDD 微调，本轮视频也没有进行训练。** 视频用于检查实现、观察直接复用带来的增量破坏，以及测量真实计算成本；不能用它证明“训练后的 RDD 视频已经有好画质”。2D 权重也不能直接搬进 Wan。

factor的两个数依次是**空间缩小倍数、时间缩小倍数**，不是宽、高；latent顺序为T×H×W，不含通道。三组都是同一个prompt：`A red fox walking through a snowy forest, cinematic natural light.`，seed=`20660913`。每次只在最后fine NFE复用上一stage的更新，scale=1。Temporal的source是上一stage的**半时间分辨率**特征，不是16×16空间特征；原metrics的`source`文字沿用了2D命名，实际执行应以`factors`和`target_stage`为准。

服务器还有其他训练任务。只用了允许的 0–3 卡，设置显存上限，不终止其他程序。因此这轮计时属于**共享 GPU 条件下的配对实验**。ABBA 可以减小漂移影响，但不能保证等同于独占 GPU 的正式论文数字。

### 视频结果

**已完成的832×448 Spatial421，6组ABBA/BAAB：**

| fine复用层数（总30层） | 去噪 baseline→候选 | 配对去噪延迟下降（95%组bootstrap区间） | 采样＋VAE baseline→候选 | 总延迟下降 |
|---|---|---|---|---|
| 6 | 0.915→0.798s | **13.16%** [12.13%,14.11%] | 8.533→8.412s | **1.43%** |
| 8 | 0.917→0.752s | **17.95%** [17.07%,18.75%] | 8.526→8.367s | **1.88%** |
| 10 | 0.910→0.706s | **22.80%** [21.98%,23.56%] | 8.515→8.318s | **2.40%** |

表中秒数为各臂中位数，百分比是每组配对下降的均值，二者统计口径不同，因此不能期待小数精确互相换算。三行分别使用各自同期的baseline。共享GPU下的区间只反映这次配对组间波动，**不排除共用硬件的系统性干扰**。

**画质结论不能乐观：这组原RDD-FastWan基线已经是严重扭曲的彩色结构；直接复用后更加平滑/模糊。它只能证明确实省下了计算，不能证明获得了可用视频。** 相对baseline的latent L2为0.391/0.405/0.445，也不是FID、VBench或感知质量评分。下方包含已完成的较温和Temporal221对照，避免只在坏基线上下结论。

[832完整原始数据](../../eval_results/direct_reuse_fullmodel_20260915/video_832/metrics.json)

#### Spatial421：未复用的 RDD＋VSA20 基线

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_832/baseline.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_832/baseline.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 视频 | 832×448，29帧，16fps；狐狸prompt；seed20660913 |
| sampler | DMD3NFE，1000/757/522，noise_filling |
| stage / latent | `(4,1)→(2,1)→(1,1)`；`8×14×26→8×28×52→8×56×104` |
| attention | 所有层原生在线VSA20，保持coarse output/gates |
| 时间参照 | 各轮去噪baseline约0.910–0.917s，总时间约8.515–8.533s |
| 状态 | **未经过RDD微调；基线自身已失真** |

#### Spatial421：直接复用6层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_832/reuse6.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_832/reuse6.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 与上方相同 | 模型、prompt、seed、shape、steps、noise_filling、精度、offload |
| 改动 | fine层`4,8,12,16,20,24`复用middle残差，scale=1；无adapter |
| 去噪 | 0.798s；约1.147×；配对延迟下降13.16% |
| 采样＋VAE | 8.412s；约1.014×；配对延迟下降1.43% |

#### Spatial421：直接复用8层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_832/reuse8.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_832/reuse8.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 与基线相同 | 模型、prompt、seed、shape、steps、noise_filling、精度、offload |
| 改动 | fine层`3,6,9,12,15,18,21,24`复用middle残差，scale=1；无adapter |
| 去噪 | 0.752s；约1.220×；配对延迟下降17.95% |
| 采样＋VAE | 8.367s；约1.019×；配对延迟下降1.88% |

#### Spatial421：直接复用10层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_832/reuse10.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_832/reuse10.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 与基线相同 | 模型、prompt、seed、shape、steps、noise_filling、精度、offload |
| 改动 | fine层`2,5,7,10,13,15,18,21,23,26`复用middle残差，scale=1；无adapter |
| 去噪 | 0.706s；约1.290×；配对延迟下降22.80% |
| 采样＋VAE | 8.318s；约1.024×；配对延迟下降2.40% |

### 1280×768：更大的Spatial421

同一方法、同一狐狸prompt、29帧，只把输出提高到1280×768。stage latent为 `8×24×40→8×48×80→8×96×160`。每个版本6组配对。

| fine复用层数 | 去噪 baseline→候选 | 配对去噪延迟下降（95%组区间） | 采样＋VAE baseline→候选 | 总延迟下降 |
|---|---|---|---|---|
| 6 | 2.724→2.301s | **15.72%** [15.47%,15.98%] | 22.748→22.322s | **1.86%** |
| 8 | 2.727→2.170s | **20.44%** [20.15%,20.69%] | 22.778→22.243s | **2.50%** |
| 10 | 2.715→2.009s | **26.00%** [25.64%,26.35%] | 22.736→21.994s | **3.18%** |

大输出下去噪收益确实更大。但这里基线本身的狐狸口鼻、眼睛已带彩色块状伪影，复用又改变了皮毛和细节。**最快一行不能被包装成“26%无损加速”。**

![1280 Spatial421固定五帧：原版、复用6、8、10层](../../eval_results/direct_reuse_fullmodel_20260915/video_1280/five_frame_comparison.jpg)

#### 1280 Spatial421：原RDD＋VSA20

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/baseline.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/baseline.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| stage / 输出 | Spatial `(4,1)→(2,1)→(1,1)`；1280×768×29帧，16fps |
| 模型 / 采样 | 未经RDD微调的FastWan；原生VSA20；DMD3NFE；noise_filling |
| 复用 | 不复用，完整30层 |
| 时间 | 同期去噪基线2.715–2.727s；采样＋VAE22.736–22.778s；1× |

#### 1280 Spatial421：直接复用6层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/reuse6.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/reuse6.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 继承上方 | 同一模型、stage、prompt、seed、输出、采样、VSA20、精度和offload |
| 唯一改动 | fine复用6/30层，mask=`4,8,12,16,20,24`，nearest，scale=1 |
| 去噪 | **2.301s**；中位数比约1.184×；配对延迟减少15.72% |
| 采样＋VAE | **22.322s**；约1.019×；配对延迟减少1.86% |

#### 1280 Spatial421：直接复用8层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/reuse8.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/reuse8.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 继承上方基线 | 同一模型、stage、prompt、seed、输出、采样、VSA20、精度和offload |
| 唯一改动 | fine复用8/30层，mask=`3,6,9,12,15,18,21,24`，nearest，scale=1 |
| 去噪 | **2.170s**；约1.257×；配对延迟减少20.44% |
| 采样＋VAE | **22.243s**；约1.024×；配对延迟减少2.50% |

#### 1280 Spatial421：直接复用10层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/reuse10.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_1280/reuse10.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 继承上方基线 | 同一模型、stage、prompt、seed、输出、采样、VSA20、精度和offload |
| 唯一改动 | fine复用10/30层，mask=`2,5,7,10,13,15,18,21,23,26`，nearest，scale=1 |
| 去噪 | **2.009s**；约1.352×；配对延迟减少26.00% |
| 采样＋VAE | **21.994s**；约1.034×；配对延迟减少3.18% |

[1280原始计时与配置](../../eval_results/direct_reuse_fullmodel_20260915/video_1280/metrics.json)

### 较温和的Temporal221：保留空间，只压缩前两步的时间维

factor是 `(1,2)→(1,2)→(1,1)`，latent为 `4×56×104→4×56×104→8×56×104`。输出仍是832×448×29帧。复用发生在最后NFE，将上一stage的时间维更新量nearest上采样；不是同形状跨步输出缓存。

| fine复用层数 | 去噪 baseline→候选 | 配对去噪延迟下降（95%组区间） | 采样＋VAE baseline→候选 | 总延迟下降 |
|---|---|---|---|---|
| 6 | 1.321→1.168s | **10.64%** [8.70%,12.80%] | 8.943→8.802s | **1.54%** |
| 8 | 1.321→1.158s | **12.75%** [10.98%,14.23%] | 8.927→8.761s | **1.92%** |
| 10 | 1.309→1.118s | **15.08%** [12.66%,17.53%] | 8.909→8.721s | **2.18%** |

这组的狐狸、雪地森林和大体动作能够辨认，复用后的主体也大致保留，但头部、皮毛会变软或改变。**比Spatial421更适合观察增量破坏，不代表视频质量已经匹配。** 只测一个prompt/seed，没有VBench/FVD或用户偏好评估。原图与复用的视频都放在下面。

![Temporal221固定五帧：原版、复用6、8、10层](../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/five_frame_comparison.jpg)

#### Temporal221：原RDD＋VSA20

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/baseline.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/baseline.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| stage / latent | `(1,2)→(1,2)→(1,1)`；`4×56×104→4×56×104→8×56×104` |
| 输出 / 采样 | 832×448×29帧，16fps；狐狸，seed20660913；DMD3NFE，noise_filling |
| attention | 原生VSA20，完整执行所有层；未经过RDD微调 |
| 时间 | 同期去噪1.309–1.321s；采样＋VAE8.909–8.943s；1× |

#### Temporal221：直接复用6层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/reuse6.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/reuse6.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 继承上方基线 | 同模型、stage、prompt、seed、输出、采样、VSA20、精度和offload |
| 唯一改动 | fine复用6/30层，mask=`4,8,12,16,20,24`，nearest，scale=1 |
| 去噪 | **1.168s**；中位数比约1.130×；配对延迟减少10.64% |
| 采样＋VAE | **8.802s**；约1.016×；配对延迟减少1.54% |

#### Temporal221：直接复用8层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/reuse8.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/reuse8.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 继承上方基线 | 同模型、stage、prompt、seed、输出、采样、VSA20、精度和offload |
| 唯一改动 | fine复用8/30层，mask=`3,6,9,12,15,18,21,24`，nearest，scale=1 |
| 去噪 | **1.158s**；约1.140×；配对延迟减少12.75% |
| 采样＋VAE | **8.761s**；约1.019×；配对延迟减少1.92% |

#### Temporal221：直接复用10层

<video controls preload="none" poster="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/reuse10.png" src="../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/reuse10.mp4" style="width:100%;max-width:720px"></video>

| 配置 | 内容 |
|---|---|
| 继承上方基线 | 同模型、stage、prompt、seed、输出、采样、VSA20、精度和offload |
| 唯一改动 | fine复用10/30层，mask=`2,5,7,10,13,15,18,21,23,26`，nearest，scale=1 |
| 去噪 | **1.118s**；约1.171×；配对延迟减少15.08% |
| 采样＋VAE | **8.721s**；约1.021×；配对延迟减少2.18% |

[Temporal221原始计时与配置](../../eval_results/direct_reuse_fullmodel_20260915/video_temporal221/metrics.json)

### 为什么含VAE后收益很小

这次没有用一个固定VAE常数拼接结果，而是**每次真实解码**。VAE仍要产生完整尺寸、完整帧数的视频，复用Transformer层不会省掉它。832的FP32 VAE约7.6秒，1280约20秒；去噪只占小部分总延迟。因此省去0.2–0.7秒去噪，在总时间上就只剩1%–3%。这不否定去噪省算，但限制了当前整条流程的收益。

把VAE换精度/实现可能减少这个瓶颈，但必须同样用于baseline与候选，且验证质量；**本轮没有靠单方面优化VAE来夸大新方法收益。** 这里的结果也不能和旧页面采用不同VAE执行路径的总时间直接横比。

## 7. 怎样判断这条路线是否值得继续

一个额外的只读小测试：保留coarse的attention/MLP分支值，只用fine当前timestep重新计算**原模型已有的gate**，仍不加adapter。32张验证图、256个teacher状态中，epsilon MSE从直接复用的0.05553变成0.05943，反而更差。这个结果不支持直接把“刷新gate”当作免费质量修复；暂不投入该变体训练。它也不证明所有经过训练的gate方案无效。

第一轮支持的结论是：**去掉adapter后可以直接训练共享模型，而且fine误差和部分图片确实得到修复；这还不是一个画质合格的最终方法。** 也没有做与旧adapter完全同预算、同数据的重新配对，不能宣布新方法已经全面超过旧方法。

下一轮我会优先保留这个简单结构，分开检查两个问题，而不是继续增加adapter：

1. **整网微调是否训练不足／coarse被改动太多？** 对同一reuse6模型延长训练，保留200步快照；单独对比更强的coarse保持损失或覆盖更多coarse timestep。检查真实rollout图片，而不是只盯teacher输入上的MSE。改善只是待验证假设。
2. **整个fine阶段一直用同一份缓存是否过强？** 独立候选只在fine前半段复用，后半段恢复真实block；训练和推理必须用同一个规则。这样会减少速度收益，不能免费继承当前15%–26%的数字。
3. 图片筛选先做固定seed对比，合格后再做足量FID、Precision/Recall。视频需要真正的RDD-Wan权重，并做全模型微调与同质量比较；当前只有计算成本和未微调质量sanity。

两组200步训练、64组图片与12段视频均已完成，本轮没有继续偷偷延长训练；旧adapter自动续训已暂停。**实现与本轮验证完成，不等于研究目标已经成功。**

## 8. 代码和日志

- [无 adapter 的训练/采样核心](../../2d/attention_analysis/direct_reuse_fullmodel.py)
- [全模型训练程序](../../2d/attention_analysis/train_direct_reuse_fullmodel.py)
- [配对图像和实际采样计时](../../2d/attention_analysis/sample_direct_reuse_fullmodel.py)
- [FastWan 直接复用计时与视频生成](../../rdd_video/benchmark_direct_reuse_fullmodel.py)
- [较温和的Temporal221视频测试](../../rdd_video/benchmark_direct_reuse_temporal221.py)
- [汇总与SHA审计程序](../../2d/attention_analysis/summarize_direct_reuse_fullmodel.py)
- [完整后台队列](../../2d/exp/run_direct_reuse_fullmodel_20260915.sh)
- [并行使用0/1卡的训练队列](../../2d/exp/run_direct_reuse_training_gpu01_20260915.sh)
- [梯度和等价性测试](../../tests/test_direct_reuse_fullmodel.py)

本轮任务已经退出，不需要再attach一个不存在的tmux。训练checkpoint与日志保存在服务器：`/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915`。其中 `train_reuse6/sample.pt` 是整网微调后的原模型state_dict；`train_reuse6/latest.pt` 还包含优化器，可用于显式续训；dense控制在 `train_dense`。旧400K权重未覆盖。

复现队列已整理为独立单卡训练，避免本机NCCL问题；已完成的输出会跳过。要用现有服务器环境复查这轮队列，可运行：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash 2d/exp/run_direct_reuse_fullmodel_20260915.sh
```

独立日志是同目录的 `video_832.log`、`video_1280.log`、`video_temporal221.log`、`train_reuse6.log`、`train_dense.log`、`preview_0.log` 至 `preview_3.log`。原始metrics、256张图片和12段视频已复制到本地，见[完整审计](../../eval_results/direct_reuse_fullmodel_20260915/summary_and_audit.json)。旧模型、旧结果、FastVideo-RDD-minimal 教学仓库均未修改。
