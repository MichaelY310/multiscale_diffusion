# FastVideo → RDD 多分辨率采样与 C2F 路由：代码说明

核对日期：2026-09-14。本文只解释已经存在的实现，不新增训练、不改采样行为，也不把原型当成已验证的新方法。

## 1. 先说结论：我们到底改了什么？

**主要是给 FastVideo 增加外部采样器、运行时 attention 控制器，以及几个可选 Triton kernel，而不是重写整个 FastVideo。**

- **RDD 采样接入**：同一个 FastWan 模型依次处理不同大小的 latent；切换时重建 clean 预测、上采样、按下一时刻加噪，再更新当前形状对应的 VSA 元数据。
- **C2F-Route**：保存上一尺度每层、每个 head 的 block 路由，将 coarse key blocks 展开为 fine child blocks，交给稀疏 attention 执行器。
- **自写 kernel 的主角**：`rdd_c2f_route_triton.py`负责展开路由；`rdd_sparse_attn_smallq_triton.py`是另一个可选的真正稀疏 QK/softmax/AV 执行器。两者不是同一件事。
- **默认 fine attention 执行器仍来自 FastVideo**。不能把官方 sparse attention 的全部代码算成我们的原创。

这些自定义 GPU kernel 用 **Triton** 写，虽然文件后缀是`.py`，`@triton.jit`函数实际编译后在 GPU 上执行。准确说法是“实现了 Triton GPU kernels”，不是“从零写了整套 CUDA C++ attention 库”。

### 本地与服务器版本核对

服务器实际引用目录为`/scr/yuheng/DevOpsBench/multiscale_diffusion/third_party/FastVideo-ref`，HEAD为`2413a57651a9720e820b64dddac9de4450a1164b`；核对时暂存区、工作区及非忽略的新文件列表均为空。该commit属于[官方 FastVideo 仓库](https://github.com/hao-ai-lab/FastVideo/commit/2413a57651a9720e820b64dddac9de4450a1164b)。因此**本次核验的接入改动主要在外部`rdd_video/`，不是当前第三方源码树中的未提交补丁**。这不等于证明历史上从未用过别的补丁或版本。

本地另有`third_party/FastVideo-kernel-ref`，HEAD为`2f3d4074064e4d86f99dc784ebbaa296e6f5925f`。它不是上述服务器版本；阅读官方实现时不要把两个版本行号混用。

四个核心接入文件的SHA256已逐一核对，本地与服务器完全相同，见末尾。

## 2. 文件地图：按这个顺序读

下面均为本项目路径。第三方原文件的作用另列在第3节。

| 文件 | 主要入口/位置 | 做了什么 | 属于哪一层 |
|---|---|---|---|
| [sample_fastwan_native_vsa.py](../../rdd_video/sample_fastwan_native_vsa.py) | `add_fastvideo_paths` L133；`sample_three_call_dmd` L153；加载与配置 L361起 | 加载真正的Wan VSA block；不同尺度采样；noise filling；元数据与计时 | Python采样/接入 |
| [fastwan_native_attention.py](../../rdd_video/attention_analysis/fastwan_native_attention.py) | 类级dispatch L54；controller L79；`_impl_forward` L1449 | 在运行时替换attention后端，管理route、输出缓存和执行器 | Python模型控制 |
| [rdd_c2f_route_triton.py](../../rdd_video/attention_analysis/rdd_c2f_route_triton.py) | `_c2f_lift_route_kernel` L22；`lift_c2f_route` L100 | coarse路由→fine紧凑索引列表 | **自定义路由GPU kernel** |
| [rdd_sparse_attn_smallq_triton.py](../../rdd_video/attention_analysis/rdd_sparse_attn_smallq_triton.py) | `_smallq_sparse_fwd` L24；wrapper L139 | 保持64-token路由块，但每个program只算16或32个query | **自定义稀疏attention GPU kernel** |
| [rdd_noise_filling_triton.py](../../rdd_video/attention_analysis/rdd_noise_filling_triton.py) | `_noise_filling_fused` L20；wrapper L78 | 融合上采样读取及加噪运算；默认不启用 | 可选边界GPU kernel |
| [benchmark_fastwan_paired_suite.py](../../rdd_video/benchmark_fastwan_paired_suite.py) | `main` L128；`run_one` L235 | 同一个模型、同prompt/seed、多配置预热与配对计时，保存视频/指标 | 实验公平性与复现 |
| [fastwan_resident_c2f_20pct_isolation_suite.json](../../rdd_video/experiments/fastwan_resident_c2f_20pct_isolation_suite.json) | 配置列表 | 之前20%/25%密度隔离实验；注意其中是`route_only` | 历史实验配置 |

`sample_fastwan_faux_rdd.py`、`fastwan_diffusers_rdd_route.py`属于较早的Diffusers接入尝试，**不要把它们当作后面原生FastVideo/VSA实验的主入口**。原生入口特意保留checkpoint中的`to_gate_compress`参数；早期用错block类型、漏掉VSA gate的结果不能当成正确官方基线。

## 3. FastVideo 原来哪些部分被保留？插入点在哪？

官方文件（以下链接固定到服务器核对的commit）：

| 官方文件 | 本项目如何使用 |
|---|---|
| [wanvideo.py](https://github.com/hao-ai-lab/FastVideo/blob/2413a57651a9720e820b64dddac9de4450a1164b/fastvideo/models/dits/wanvideo.py) | 加载`WanTransformerBlock_VSA`；沿用QKV投影、RoPE、输出投影、文字cross-attention和MLP |
| [video_sparse_attn.py](https://github.com/hao-ai-lab/FastVideo/blob/2413a57651a9720e820b64dddac9de4450a1164b/fastvideo/attention/backends/video_sparse_attn.py) | 沿用每种latent形状的tile元数据、padding和tile/untile流程；替换其底层forward |
| [block_sparse_attn.py](https://github.com/hao-ai-lab/FastVideo/blob/2413a57651a9720e820b64dddac9de4450a1164b/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py) | 默认调用`block_sparse_attn_from_indices`，接收我们生成的索引 |
| [block_sparse_attn_triton.py](https://github.com/hao-ai-lab/FastVideo/blob/2413a57651a9720e820b64dddac9de4450a1164b/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py) | 官方可移植Triton稀疏执行器，也是SmallQ设计参考 |

运行时调用关系：

```text
本项目 sample_three_call_dmd
  ├─ 决定本次 latent 大小、timestep、noise filling
  ├─ 官方 metadata builder：构建当前 tile 网格、有效token数
  └─ 官方 Wan 模型
       └─ 每一层 WanTransformerBlock_VSA
            ├─ 官方 norm / QKV / gate / RoPE / tile
            ├─ 本项目 controller._impl_forward
            │    ├─ vsa：调用原官方后端
            │    └─ C2F：生成或读取跨尺度路由
            │          ├─ 我们的 route-lift kernel
            │          └─ 官方 sparse kernel 或可选 SmallQ kernel
            └─ 官方 untile / output projection / residual / cross-attention / MLP
```

具体安装方式是`types.MethodType`绑定新的`block.attn1.attn_impl.forward`。并保存原forward，所以`attention_mode="vsa"`能回到原后端。

另有`install_projected_cache_class_patch`在模型构造前安装类级dispatch，主要服务于整层输出缓存/跳层。原因是FastVideo加载或FSDP管理期间某些权重可能只是占位tensor；构造后随意绕过原managed forward会读到不正确的权重状态。**这是Python生命周期接入，不是新CUDA kernel。**

## 4. RDD 的多分辨率采样是怎样接进去的？

### 4.1 不是把模型重新造三遍，而是给它不同形状

采样参数`stage_factors`中的每个二元组是 **(空间缩小倍数, 时间缩小倍数)**，不是(T,H)。

例如`(4,4) → (2,2) → (1,1)`、输出832×448、29帧：

| 本次调用 | latent T×H×W | patch后token网格，patch=(1,2,2) | VSA的4×4×4-token tile网格 |
|---|---|---|---|
| coarse | 2×14×26 | 2×7×13 | 1×2×4，共8块 |
| middle | 4×28×52 | 4×14×26 | 1×4×7，共28块 |
| fine | 8×56×104 | 8×28×52 | 2×7×13，共182块 |

“一个VSA block有64个token”说的是attention tile，不是64个视频像素，也不是Transformer的一层。边缘tile会padding，并记录真实token数。

### 4.2 每次调用前后做什么

现有FastWan sanity sampler使用DMD三次调用，常用t=`1000,757,522`，沿用其`shift=8.0` scheduler；sigma由scheduler查询，**不能简单把t/1000当作实际sigma**。

简化执行流程：

```python
for call, (factor, t) in enumerate(zip(stage_factors, timesteps)):
    shape = shape_for(factor)
    if call == 0:
        state = gaussian(shape)
    else:
        state = (1 - sigma_next) * upsample(clean) + sigma_next * fresh_noise
    metadata = official_vsa_metadata(shape, patch_size, sparsity)
    controller.begin_model_forward(call, shape)
    prediction = model(state, text, t, metadata)
    clean = official_prediction_to_clean(state, prediction, t, scheduler)
```

源码里`stage`在这个三调用入口实际上按**调用序号**增加；连续两次相同shape也会有两个序号，不能把它等同于“分辨率每次都变”。

### 4.3 当前 `noise_filling` 的精确定义

这条FastWan原生入口目前的`noise_filling`是：**上采样clean预测＋下一sigma的新独立高斯噪声**。`redraw`在代码里保留为同一路径的旧别名。另一种“保留coarse innovation，再补正交噪声”叫`innovation_preserving`，是单独分支，不要混用旧文档名称。

上述普通FM表达式是这个**FastWan零微调sanity sampler**的实际实现。文件头明确叫`faux-rdd`：FastWan权重没有学过RDD的drift目标，改变shape/边界不能让它自动成为训练正确的RDD模型。

真正的视频drift训练/采样原型另在[flow_matching_video.py](../../rdd_video/flow_matching_video.py)、[rdd_distillation.py](../../rdd_video/rdd_distillation.py)、[train_rdd_wan.py](../../rdd_video/train_rdd_wan.py)和[distill_rdd_wan.py](../../rdd_video/distill_rdd_wan.py)。它们不是这里三调用速度实验已经完成训练的证据，也不属于本次C2F kernel接入。

## 5. C2F-Route：详细实现

### 5.1 第一步：在低分辨率生成路由

`_capture_route`（L1050）对每层、每个head单独处理：

1. 沿用VSA的block平均Q、K，得到`Q_center`、`K_center`。
2. 算block级分数`S = Q_center @ K_center.T / sqrt(head_dim)`。
3. 每个query block保留`ceil(route_density × block_count)`个key blocks，至少1个。
4. 跨尺度缓存路由强制包含self block：若Top-K没有自己，用自己替换最后一个，不额外扩大K。
5. 保存`BlockRoute(block_grid, indices, counts, probabilities, probability_grid)`。

其中`indices`是`int32 [B, heads, Q_blocks, K_capacity]`，`counts`是每行真正有效的长度。`probabilities`是低分辨率的**block级**softmax矩阵，供后面的可选coarse-output复用；它不是完整token级N×N attention。

注意：coarse本次输出使用官方VSA top-k及coarse分支；用于后续尺度的缓存路由还加了self保护。因此两者不保证是逐项完全相同的mask。

### 5.2 第二步：把coarse路由展开成fine路由

假设coarse query块A选中了key块B和D；升分辨率后A、B、D各拆成4个孩子：

```text
coarse： A → {B, D}
fine：   A0 → {B0,B1,B2,B3,D0,D1,D2,D3}
         A1 → 同一组8个key blocks
         A2 → 同一组8个key blocks
         A3 → 同一组8个key blocks
```

这不是复制attention数值；只是说“这些fine位置允许互相看”。随后仍要用**当前fine的Q、K、V**计算它们之间的attention。

规则是每个fine query按坐标找coarse parent；读取parent选中的coarse keys，再找每个key的全部fine children。真实网格有ceil/padding，因此不是所有parent都同样多孩子。

核对代码支持的例子：`(1,2,4) → (1,4,7) → (2,7,13)`。采用按各维网格比例的整数映射；这是定义好的几何路由规则，**不保证等于真实fine attention的Top-K**，也不能把ceil后的tile映射说成所有边界都与latent逐token严格重合。

### 5.3 真正自己写的 route-lift GPU kernel

位置：[rdd_c2f_route_triton.py](../../rdd_video/attention_analysis/rdd_c2f_route_triton.py)。

它启动`B × heads × fine_query_blocks`个Triton programs。每个program处理一行：

```python
# 源码关键逻辑的节选/简化，省略指针与边界参数
row = tl.program_id(0)
q_fine = row % fine_q
bh = row // fine_q

q_t = q_fine // (fine_h * fine_w)
q_h = (q_fine // fine_w) % fine_h
q_w = q_fine % fine_w
parent_t = q_t * coarse_t // fine_t
parent_h = q_h * coarse_h // fine_h
parent_w = q_w * coarse_w // fine_w
parent_q = (parent_t * coarse_h + parent_h) * coarse_w + parent_w

# 加载parent这一行的有效key数、逐个展开children
selected_count = tl.load(coarse_counts + bh * coarse_q + parent_q)
# ... masked load coarse key id
# ... 计算child坐标范围，写fine key ids
tl.store(fine_counts + row, valid_fine_key_count)
```

这个kernel的硬件价值主要是：

- 不把route搬回CPU，不逐行`.tolist()`再拷回GPU。
- 不构造fine的稠密block×block布尔mask，更不构造token×token大矩阵。
- 一次kernel完成parent映射、child展开和compact索引写入，减少一串gather/中间tensor/launch。
- 支持每行不同有效长度，后端只读取`counts`范围内的索引。剩余capacity可能未初始化，**不能把整个缓冲区都当有效路由读**。

wrapper检查int32、CUDA、shape及fine网格不小于coarse。正常调用源自`_capture_route`，有非空/self/合法索引等内部约束；它不是任意外部坏索引都能安全输入的通用算子。Triton路径没有额外做fallback里的self修复，依赖这个内部约束。

`_lift_route`（L1098）按`(当前调用序号, layer, target_grid)`缓存展开结果，找不到Triton模块时有PyTorch几何展开fallback。后者是fallback/reference，不是另一个CUDA手写实现。

### 5.4 第三步：执行fine attention——这里通常仍用官方kernel

`_execute_sparse`（L1350）默认调用：

```python
block_sparse_attn_from_indices(
    query_h, key_h, value_h,
    route.indices, route.counts,
    variable_sizes,
)[0]
```

`query_h/key_h/value_h`为tile后的`[B, heads, padded_tokens, head_dim]`。官方kernel只遍历指定key blocks，计算当前fine的`QK → softmax → AV`，并屏蔽边缘padding。

因此，“C2F不用再算attention”是错的。准确说法是：**某些模式不用重新算用于选路的coarse QK，但选中fine blocks内部的attention仍然计算。** QKV投影、output projection、文字cross-attention、MLP也没有因只改route而消失。

### 5.5 四种名字必须分清

以下区别针对有上一尺度来源的后续调用；第一次仍需seed。

| `attention_mode` | fine路由来自哪里 | 还算当前coarse QK吗？ | coarse输出是否保留？ | 后一stage从哪里拿route？ |
|---|---|---|---|---|
| `vsa` | 当前Q/K在线打分 | 是 | 是，当前VSA coarse分支 | 不启用C2F |
| `c2f_route` | 上一stage的路由展开 | **是**，为了当前coarse分支 | 是，与原VSA同类运算 | 当前stage新算出的block scores会再capture |
| `c2f_route_only` | 上一stage的路由展开 | 否 | **后续stage删除该分支** | 当前实际使用的继承路由继续传下去 |
| `c2f_route_reuse_coarse` | 上一stage的路由展开 | 否 | 加回近似coarse输出：旧block概率×当前pool后的V，再乘当前gate | 继承route，block概率仍追溯seed网格 |

`c2f_route_reuse_coarse`仍要pool当前V、聚合到seed网格、算旧概率×当前V并broadcast；**不是完全没有coarse计算，也不是原VSA输出的数值等价替换**。

两种严格继承模式中，16阶段并没有用它刚算出的fine sparse attention概率来重新排序32阶段route。它是**把实际用过的索引继续展开**。所以不能把当前实现讲成“每一层高分辨率都重新学习/精化了下一层路由”。

`c2f_route`则不同：当前Q/K还会在线计算并capture，它只是本次使用上一stage路由；下一stage可使用这次新capture的路由。

### 5.6 为什么设20%却有时实际是25%？

coarse只有8个blocks时：`ceil(8×0.2)=2`，已经是25%。如果每个parent孩子数相同，全部展开后**密度基本不变**，不会从25%自动变成6.25%。不等孩子数、padding与self保护会让实际计数再偏移。

因此记录里要区分：

- `route_density`：请求的比例。
- `counts.mean()/fine_block_count`：实际平均block比例。
- 有效token-pair密度：边缘块大小不同，未必等于block比例。

### 5.7 这已经是“静态kernel”了吗？

不是硬编码route的专用执行器。当前只是shape/容量等参数可被Triton编译特化；**选中的key索引仍由GPU在运行时加载**。缓存route也不等于CUDA Graph或编译器已消除了所有间接索引。未来可以进一步做固定route调度，但不能把它写成已经实现的收益。

## 6. SmallQ-16/32：另外一个确实自写的attention kernel

位置：[rdd_sparse_attn_smallq_triton.py](../../rdd_video/attention_analysis/rdd_sparse_attn_smallq_triton.py)。这是参考官方VSA可移植Triton结构写的forward-only变体，而不是路由生成器。

原64-token路由块不变；SmallQ-32把其中的64个query拆成两份，各32个query交给不同program：

```text
原来：一个program处理64个query → 读取该块route → 遍历64-token key块
现在：program A处理前32个query ┐
      program B处理后32个query ┴→ 共用同一条route → 同样的key块
```

关键代码位置：`query_start = fine_q_block * block_m`；`route_q_block = query_start // block_n`；其中`block_m=16或32`，`block_n=64`。

kernel内部真正执行：

1. `tl.make_block_ptr`按tensor stride定位Q/K/V块。
2. 从紧凑route读取下一个key block，只加载选中的数据。
3. `tl.dot(query_tile, key_tile)`计算QK。
4. 用每个key block的`variable_block_sizes`屏蔽padding。
5. 维护running max、归一化sum和FP32输出累加器，实现在线softmax。
6. `tl.dot(probabilities, value_tile, accumulator)`计算AV，最后除归一化sum写回。

它不改变哪些key被保留，改变的是GPU工作如何切分。小stage能获得更多独立program，但也可能重复加载K/V、增加调度开销，因此不是所有shape都快。控制器仅在`small_stage_executor="triton_q32"/"triton_q16"`且token数低于配置阈值时调用，超过阈值回到官方kernel。没有选择它的C2F实验仍走官方executor。

数学上目标是相同稀疏attention；不能不经测试声称所有shape/硬件都逐位相同。已有小shape测试记录过BF16逐位一致，但更一般的精度要看具体测试。此实现无反向，**不能直接把它作为训练kernel使用**。

## 7. 还有哪些自定义GPU kernel？哪些只是原型？

| 文件 | 运算 | 接入状态/限制 |
|---|---|---|
| [rdd_noise_filling_triton.py](../../rdd_video/attention_analysis/rdd_noise_filling_triton.py) | 上采样parent读取＋`(1-sigma)*clean + sigma*noise`融合 | FastWan入口可选`triton_fused`；历史表现不佳，默认不用。随机数仍由PyTorch生成；代码逐步BF16舍入以匹配原表达式，不是自写高斯RNG |
| [rdd_video_route_triton.py](../../rdd_video/attention_analysis/rdd_video_route_triton.py) | 通过预生成parent/child查找表展开路由 | 较早的无checkpoint算子benchmark版本；native FastWan controller调用的是另一份`rdd_c2f_route_triton.py` |
| [rdd_video_reuse_triton.py](../../rdd_video/attention_analysis/rdd_video_reuse_triton.py) | `x + scalar_gate * cached_output`融合 | `benchmark_wan_video_methods.py`使用的微基准原型；**native FastWan的StepReuse没有调用这个kernel**，不能把它说成所有视频reuse计时的实现 |
| [csra_coarse_transport_kernel.py](../../rdd_video/attention_analysis/csra_coarse_transport_kernel.py) | 读取nearest parent的coarse向量，直接与fine向量相加 | 后期coarse transport微基准；调用点在`benchmark_csra_coarse_transport.py`。不是旧C2F route，也不是当前训练后读取器必经路径 |
| [2d/rdd_block_sparse_triton.py](../../2d/attention_analysis/rdd_block_sparse_triton.py) | 多种2D block-sparse QK/AV、存储稀疏概率、旧概率×当前V | 早期ImageNet RDD实验的自写kernel集合；不是FastWan默认backend，详细区别如下 |

### 2D版本确实写过“attention probability复用”的kernel

`2d/attention_analysis/rdd_block_sparse_triton.py`主要接口：

- `block_sparse_attention` L448：稀疏attention。
- `block_sparse_attention_store` L459：计算同时保存选中区域的softmax概率。
- `cached_probability_av` L483：**只用旧A乘当前V**，不重算QK/softmax。
- `micro_*` L741起：更小的4-token块。
- `packed_*` L1119起：按空间块打包布局；支持相应store与cached-AV入口。

最初16-token块、DiT-XL的head_dim=72等设定与视频64-token cube不同。因此需要单独的布局、维度padding与kernel，不是把视频版本直接改个函数名。

这与下面的FastWan `StepReuse-Out`不同：**2D这里缓存的是稀疏A；视频原生路径主要缓存的是已经乘V后的输出。**

## 8. 哪些重要加速改动不是自写CUDA？

### 8.1 StepReuse-Out

同一个`fastwan_native_attention.py`内：

- `reuse_scope="backend"`：缓存底层self-attention输出；命中时仍付QKV、RoPE、tile及后续输出投影等上层开销。
- `reuse_scope="projected"`：在`_block_forward_projected_cache`（L919）保存`to_out`之后的self-attention输出；命中时跳过该分支的norm/QKV/gate_compress/RoPE/tile/attention/output投影。
- 当前timestep的外层modulation/residual gate、文字cross-attention、MLP仍计算。

两者都是复用旧输出，不是`A_old @ V_current`。缓存键包括shape和layer，按该shape已调用次数及`reuse_every`决定命中。严格三次不同shape的`4→2→1`没有同shape命中；`4→4→1`才有机会。

这是Python层缓存/调度配合FastVideo既有算子；并没有把前一节的微基准scalar-gate kernel硬接成Wan的实际动态gate实现。

### 8.2 resident/offload与计时修正

采样器和配对runner显式处理`dit_cpu_offload`与`dit_layerwise_offload`两个开关；只关前者而漏掉后者会反复搬权重。让所有对照一致常驻GPU，是执行口径修正，**不是RDD独有算法或我们写的新CUDA kernel**。

采样计时把每阶段同步改为同stream的CUDA events，在完整轨迹末尾同步一次；记录prepare/model/reconstruct分项。同样属于测量/调度改进，不是减少了模型数学计算。

`boundary_executor="torch_nearest"`使用PyTorch nearest插值并保持dtype；它调用框架已有GPU算子，不是我们的Triton kernel。

## 9. 测试与已有数字：不要把局部速度当整模型速度

| 检查/实验 | 文件或记录 | 能说明什么 |
|---|---|---|
| C2F路由范围与集合一致性 | [test_c2f_route_lift_gpu.py](../../rdd_video/attention_analysis/test_c2f_route_lift_gpu.py) | 测真实`8→28→182`非均匀网格；与Python参考parent/child规则比较，检查索引和有效长度 |
| C2F路由生成微基准 | [benchmark_c2f_vs_online_route.py](../../rdd_video/attention_analysis/benchmark_c2f_vs_online_route.py)；[原始JSON](../../eval_results/rdd_fastwan_fair_accel_20260907/c2f_vs_online_route_benchmark.json) | 记录一次route-lift约0.0314ms、在线VSA route builder约0.2093ms，即该路由子流程6.66×。不含QKV/fine attention/coarse输出；且top-k取整后继承38个与在线37个不同，不是精确等密度整模型对照 |
| SmallQ与官方executor | [benchmark_small_stage_executor.py](../../rdd_video/attention_analysis/benchmark_small_stage_executor.py)；[原始JSON](../../eval_results/rdd_fastwan_efficiency_audit_20260908/small_stage_executor_smallq.json) | 512 padded tokens下native约0.1399ms、Q32约0.0359ms，样例输出差0；仅算子，不能写成视频约4× |
| 完整视频多配置配对 | [benchmark_fastwan_paired_suite.py](../../rdd_video/benchmark_fastwan_paired_suite.py)；[历史报告](RDD_FASTWAN_FAIR_ACCELERATION_SWEEP_ZH.html) | 多次预热、同模型与seed、实际视频及denoise/VAE计时；需区分offload、coarse分支和实际density |

本次只是重新核对代码/已有记录，**没有重新占GPU跑上述测试，也没有重新证明每个历史指标**。测试代码存在与某项速度在当前环境复现，是两件事。

必须保留的限制：

1. C2F route-lift节省在线选路，不自动节省QKV/MLP/fine sparse attention。
2. `c2f_route`为了保留原coarse输出仍计算coarse QK，收益自然受限。
3. `route_only`删coarse输出后较快，不能把删分支的差值全部算作公平C2F收益。
4. SmallQ是一种小stage执行优化；局部倍数不能代替端到端收益。
5. 路由几何继承可测试正确执行，但“继承路由是否保画质”需要真实样本/训练后评价；不是索引测试能证明的。

## 10. 想展示GPU编程理解，最值得读的三段

1. **route-lift L22–97**：从三维几何映射到GPU并行行任务；compact indices/counts；不均匀children；masked读写；避免稠密中间矩阵。
2. **SmallQ L24–136**：调度粒度与路由粒度分离；K/V间接索引；padding mask；在线softmax；Tensor Core矩阵乘与FP32累加。
3. **noise-filling L20–75**：融合访存、真实stride、避免上采样临时tensor；以及为什么BF16的舍入顺序会影响后续生成。这个原型默认未采用，讲清失败原因也比夸大收益更可信。

可以准确描述为：“基于FastVideo原生Wan/VSA接口实现跨尺度路由复用，编写Triton路由展开和SmallQ sparse-attention kernel，并做索引/数值一致性及端到端开销分析。”不要描述成“从头实现VSA”或“这几个kernel已经在所有视频上带来显著净加速”。

### 核心文件SHA256，本地与服务器一致

```text
sample_fastwan_native_vsa.py
d90103db5010ee3725fabc1504f15daeb7b5a1ded3c51524ca2b7c8f2e028522
fastwan_native_attention.py
decaff5d729158e4be0de284aa7af88695233afb258b7ab091a970722ac8bcc3
rdd_c2f_route_triton.py
a448ceb39d7fb4fb3e9bee491e8356decd12d32877833bf98a1f671c29ac6d96
rdd_sparse_attn_smallq_triton.py
a0f9cdf2b9dc1f7e69543e8e7f2424c7c8d72fcb112cdcbcd9692fc034d3413b
```
