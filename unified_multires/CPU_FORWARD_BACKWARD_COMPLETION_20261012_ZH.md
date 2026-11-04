# CPU forward/backward逐层对照：2026-10-12

**12种有训练/codec/蒸馏核心的方法、33份选中配置，全部完成本轮逐层对照。所有实际捕获的模块输入、输出、activation gradient、处理前parameter gradient均逐字节相同，max_abs=0。** 31份image配置使用当前完整发布网络/输入尺寸，另有完整SD式VAE及30层Wan1.3B结构。

这不是所有论文最佳结果或完整官方CLI认证：两侧共用作者backbone，实现独立目标/loss/backward组装；CPU FP32、同一非零随机权重和合成模型域输入。RMD仍使用缩小视频网格、随机text/teacher。31配置各2次更新，RMD两transition各17次更新覆盖15步warmup，完整codec2次更新。

范围：Relay、Scale Space Diffusion、PixelFlow、DFM、LapFlow、PPFlow、DCTdiff、MDM、DiMR、Diffusability、RMD及本地RDD。RMD固定仓库自称官方且metadata作者匹配，但论文页未独立链接此仓库；归属限制仍保留，不能把仓库数值parity变成作者身份认证。

## 每份完整网络配置的实际结果

| 配置 | 参数量 | micro | 输入Tensor对 | 输出Tensor对 | activation-grad对 | parameter-grad对 | 证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| dctdiff_afhq512_uvit_mid_8by8 | 130853484 | 1 | 486 | 468 | 466 | 420 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_afhq512_uvit_mid_8by8.json) |
| dctdiff_celeba64_uvit_small_2by2 | 44267544 | 1 | 374 | 360 | 358 | 324 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_celeba64_uvit_small_2by2.json) |
| dctdiff_cifar10_uvit_mid_2by2 | 129987096 | 1 | 486 | 468 | 466 | 420 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_cifar10_uvit_mid_2by2.json) |
| dctdiff_cifar10_uvit_mid_deep_2by2 | 160690200 | 1 | 598 | 576 | 574 | 516 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_cifar10_uvit_mid_deep_2by2.json) |
| dctdiff_cifar10_uvit_small_2by2 | 44169240 | 1 | 374 | 360 | 358 | 324 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_cifar10_uvit_small_2by2.json) |
| dctdiff_ffhq128_uvit_mid_4by4 | 145532214 | 1 | 542 | 522 | 520 | 468 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_ffhq128_uvit_mid_4by4.json) |
| dctdiff_ffhq128_uvit_mid_deep_4by4 | 145532214 | 1 | 542 | 522 | 520 | 468 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_ffhq128_uvit_mid_deep_4by4.json) |
| dctdiff_ffhq128_uvit_small_4by4 | 44298294 | 1 | 374 | 360 | 358 | 324 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_ffhq128_uvit_small_4by4.json) |
| dctdiff_ffhq256_uvit_mid_4by4 | 130761264 | 1 | 486 | 468 | 466 | 420 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_ffhq256_uvit_mid_4by4.json) |
| dctdiff_ffhq512_uvit_mid_8by8 | 130853484 | 1 | 486 | 468 | 466 | 420 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_ffhq512_uvit_mid_8by8.json) |
| dctdiff_imgnet64_uvit_mid_2by2 | 130903320 | 1 | 490 | 470 | 468 | 422 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_imgnet64_uvit_mid_2by2.json) |
| dctdiff_imgnet64_uvit_mid_deep_2by2 | 146254872 | 1 | 1020 | 998 | 522 | 470 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_imgnet64_uvit_mid_deep_2by2.json) |
| dctdiff_imgnet64_uvit_small_2by2 | 44780056 | 1 | 378 | 362 | 360 | 326 | [JSON](evidence/forward_backward_20261012/full_current/dctdiff_imgnet64_uvit_small_2by2.json) |
| dimr_DiMR-G-2R_imagenet256 | 1056352904 | 1 | 4458 | 5212 | 3144 | 2892 | [JSON](evidence/forward_backward_20261012/full_current/dimr_DiMR-G-2R_imagenet256.json) |
| dimr_DiMR-XL-2R_imagenet256 | 505046408 | 1 | 3132 | 3658 | 2214 | 2036 | [JSON](evidence/forward_backward_20261012/full_current/dimr_DiMR-XL-2R_imagenet256.json) |
| pixelflow_xl_c2i | 676611120 | 4 | 1344 | 1156 | 1152 | 982 | [JSON](evidence/forward_backward_20261012/full_current/pixelflow_xl_c2i.json) |
| ppflow_xl_2 | 675428128 | 8 | 1376 | 1126 | 1124 | 592 | [JSON](evidence/forward_backward_20261012/full_current/ppflow_xl_2.json) |
| ppflow_xl_3 | 675688544 | 8 | 1382 | 1134 | 1132 | 600 | [JSON](evidence/forward_backward_20261012/full_current/ppflow_xl_3.json) |
| dfm_sit_xl | 679402388 | 1 | 1272 | 1196 | 1196 | 602 | [JSON](evidence/forward_backward_20261012/full_current/dfm_sit_xl.json) |
| dfm_sit_xl_repa_localfeatures | 687533716 | 1 | 1274 | 1206 | 1206 | 614 | [JSON](evidence/forward_backward_20261012/full_current/dfm_sit_xl_repa_localfeatures.json) |
| dfm_sit_b_released_yaml | 132476180 | 1 | 600 | 556 | 556 | 282 | [JSON](evidence/forward_backward_20261012/full_current/dfm_sit_b_released_yaml.json) |
| lapflow_mot_xl_2 | 1347159200 | 2 | 1514 | 1391 | 1391 | 948 | [JSON](evidence/forward_backward_20261012/full_current/lapflow_mot_xl_2.json) |
| lapflow_imagenet_xl_2_paper | 1347159200 | 2 | 1514 | 1391 | 1391 | 948 | [JSON](evidence/forward_backward_20261012/full_current/lapflow_imagenet_xl_2_paper.json) |
| ssd_celeba64 | 92256131 | 1 | 840 | 748 | 748 | 652 | [JSON](evidence/forward_backward_20261012/full_current/ssd_celeba64.json) |
| mdm_cc12m256 | 476613958 | 3 | 1428 | 1258 | 1258 | 1778 | [JSON](evidence/forward_backward_20261012/full_current/mdm_cc12m256.json) |
| mdm_cc12m1024 | 480970057 | 1 | 1722 | 1512 | 1512 | 2126 | [JSON](evidence/forward_backward_20261012/full_current/mdm_cc12m1024.json) |
| rdd_unified_xl_300_300_400 | 674834720 | 2 | 3642 | 2784 | 2784 | 582 | [JSON](evidence/forward_backward_20261012/full_current/rdd_unified_xl_300_300_400.json) |
| relay_imagenet_low | 295900995 | 1 | 718 | 634 | 632 | 1084 | [JSON](evidence/forward_backward_20261012/full_current/relay_imagenet_low.json) |
| relay_imagenet_high | 553833475 | 1 | 776 | 680 | 678 | 1136 | [JSON](evidence/forward_backward_20261012/full_current/relay_imagenet_high.json) |
| rdd_unified_l_300_300_400 | 457840672 | 2 | 3138 | 2400 | 2400 | 502 | [JSON](evidence/forward_backward_20261012/full_current/rdd_unified_l_300_300_400.json) |
| pixelflow_xl_c2i_paper_dopri5 | 676611120 | 4 | 1344 | 1156 | 1152 | 982 | [JSON](evidence/forward_backward_20261012/full_current/pixelflow_xl_c2i_paper_dopri5.json) |
| diffusability_se24 | 83653863 | 2 | 1660 | 1772 | 1056 | 280 | [JSON](evidence/forward_backward_20261012/codec_current/diffusability.json) |
| rmd_wan13b | 1418996800 | 2 | 399840 | 314160 | 52156 | 56100 | [JSON](evidence/forward_backward_20261012/rmd/rmd_full_backbone.json) |

对数是执行次数×模块边界Tensor数，不是独立网络层数。每个JSON的 `activation.coverage`/`layer_tensor_stats`/`gradient_coverage`/`absent_gradient_outputs` 给出模块路径、重复调用、首次差异与无梯度原因；RMD/codec用activation列表。

## 实际执行与一致性定义

- original侧执行固定源码的作者forward/loss函数或AST语句，以及作者backward语句；unified侧调用我们的Method。主31配置没有共用Trainer。
- 在相同初始state和输入下分别执行两张计算图，恢复相同Torch/NumPy/Python随机流；比较原始loss及标量objective，梯度先比对再更新。非零输出head避免全零网络假通过。
- 比较shape、dtype、uint8原始字节；signed-zero/dtype/nonfinite negative controls已通过。不是“误差小于某阈值就通过”。
- 12方法各128步完整模块回归，以及12个扩展case各128步已重新执行，loss/grad/weight/EMA/采样所有检查byte exact。复用官方网络、DPM solver、CPUattention及导入shim均显式记录。
- MDM256保留作者FP32 backward先于accum除法的次序；MDM1024原FP16控制并未在CPU模拟成CUDA认证。
- DiMR checkpoint首次no_grad pass的view没有grad_fn，真实重算输出梯度已捕获；冻结encoder/teacher和未调用容器不会冒充已测梯度。functional/ATen并非全部逐操作hook。

39对临时checkpoint另外实际重载：14,948个model/EMA/Adam张量以及全部保存的parameter groups、typed scalars和容器结构相同，没有过滤冻结参数组。文件ZIP的SHA单列，未宣称序列化文件相同。见 [checkpoint状态证据](evidence/forward_backward_20261012/checkpoint_state_bytes.json)。

## 修正并重测的真实差异

Relay完整增强/augment_dim/EMA/官方loss sum与DCT FFT；PPFlow stage端点/velocity运算树；RDD DDPM effective noise、FM clean和阶段先mean后sum；LapFlow官方whole-batch F.mse_loss；DFM REPA denoising.mean()+projection.mean()*weight及DFM-SiT CLI；PixelFlow packing标量归约/ndarray API；MDM setup RNG/FP32累积/1024文本夹具；DCT float64 decode/条件label；两个reader drop_last/epoch边界；Diffusability原B×1 reconstruction与B-vector KL广播及SE加法次序。修正未修改作者源文件。

通用组件：Relay84/84、PPFlow484/484、RDD412/412 byte exact。LapFlow713/715：仅两项是FP32分解→重构对原图恒等式的2.3841858e-7舍入；**native两侧训练/采样并无这个差异**，未把恒等式强行标byte exact。

NAMI初次逐层检查还暴露了测试clone把模型/VAE共用SiLU别名打散的问题：现已一起deepcopy、保持别名图，并显式适配bridge参数/返回容器；数学和作者权重没有改。失败历史日志保留，最终证据使用修复后报告。

## 仅公开推理的方法与新发现的CAT源码

SPEED、NAMI、MrFlow的作者推理workflow、执行模块输出及量化前浮点像素检查全部通过；它们的公开路径没有作者训练loss/backward，严格标记 **backward N/A**，没有用0个梯度冒充通过。真实pretrained资产不在本次范围。

CAT的NeurIPS官方补充ZIP包含nested VAE两文件、scoring和classification。补充源码与独立组装codec在1/2/3和mixed compression、posterior mean/sample上通过：缩小六block网络2轮16case，以及classification.py原始512网格完整架构1轮8case，逐层输出和诊断adjoint均逐字节相同；但没有发布diffusion trainer或完整codec训练loss。诊断loss明确是pixel.square().mean()+作者KL，不冒称论文训练复现；不能据此把CAT加入上面的12种发布训练核心已验证方法。见 [CAT证据](evidence/forward_backward_20261012/cat_codec/cat_codec.json)、[CAT完整512配置](evidence/forward_backward_20261012/cat_codec_full/cat_codec.json)、[8个默认rate/返回tuple/显式generator API与诊断gradient case](evidence/forward_backward_20261012/cat_api/cat_api.json) 与 [发布复核](evidence/forward_backward_20261012/RELEASE_RECHECK_ZH.md)。

CAT缩小夹具的通道全部设为32，改变了作者encoder中依赖channel差异的提前退出条件，因此该夹具的homogeneous-rate输出是16/32/64；这不是完整发布配置的行为。187,619,158参数的原classification.py配置已独立实测：8case全部输出512×512且逐层/梯度一致。完整配置的证据不能由缩小夹具代替。

## 原清单的处理结果和不可混淆的结论

148项全部逐项登记了本次处理与剩余前提：[逐项记录](CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md)。它包含真正不可能仅靠当前CPU任务关闭的事项：未发布的作者trainer/私有分支，缺失的真实VAE/T5/DINO/最佳权重/数据，原CUDA Flash/AMP和8/16rank分布式，百万步正式训练及50kFID。**这148项没有全部关闭；完整官方CLI仍为0/33认证，最佳主表仍为0行正式复现。**

逐层零差可以证明本次明确路径/配置下的实现一致性，不能证明随机模型具有论文最佳画质、所有输入分支均等价或真实训练能取得同样FID。未执行层、None grad、CPU替代、精度/网格偏离均保留明细。

## 证据绑定边界

31份完整image配置绑定当前配置/harness/实现/tracer hash；12个128步方法和12个扩展绑定当前实现/harness。CAT绑定4份补充源码、adapter和harness。RMD完整报告绑定source_executions、recipe及本地架构JSON，其运行wrapper未在启动时记录自己的harness SHA；没有事后填成启动时证据。

## 重跑与产物

重跑入口：[run_cpu_layer_parity.sh](run_cpu_layer_parity.sh)。最终机器可读索引：[AUDIT.json](evidence/forward_backward_20261012/AUDIT.json)，逐配置报告、逐层统计、当前源码/配置/harness hash和日志全部保留。测试使用现有只读Python环境，不下载模型/数据、不初始化CUDA、不启动正式长训练。临时checkpoint/像素Tensor在对照结束后按用户既定要求清理，清单与hash记录于cleanup.json；它们不是可恢复的正式训练模型。
