# FM fresh FID 33：模型身份与采样核查

本次核查的问题：是否把早期有问题的 FM 模型，或者普通训练权重，误当成历史效果好的模型给了新服务器。

**结论：选定资产、训练日志和用户提供的远端张量核验均指向 XL-v2 400K EMA，没有发现选错模型。fresh 的 FID 33.2233 的原因尚未被隔离验证，不能直接归因于 DDPM/FM 的数学差别，也不能宣称这个结果已经证实“正常”。**

完整机器证据见 [FM_CHECKPOINT_IDENTITY_AUDIT.json](FM_CHECKPOINT_IDENTITY_AUDIT.json)。远端张量比较是用户提供的执行记录；本机独立核验了记录一致性、HF 资产身份、配置和 CPU 代码一致性。

## 1. 区分训练版本

| 记录 | 已找到的证据 | 本次是否选择 |
|---|---|---|
| `RDD-FM-Unified-XL`，早期原型 | `2d/train_RDD_FM_imagenet256_unified_XL.log:2693` 从此目录的 215K 恢复；没有定位到可独立核验的归档权重与正式 FID | 否 |
| `RDD-FM-Unified-XL-v2` | v2 启动脚本明确写 corrected time-axis partitioning；训练日志从 v2 的 30K 恢复，记录保存 400K；实际权重内部 metadata 同样指向 v2 | **是** |
| 后续名称含 `corrected` 的 XL/L 实验 | 另有阶段分配实验，包括 `.4/.7` 时钟和 L/2 网络；不能看到 corrected 就替换 v2 的时钟或权重 | 否 |

早期原型现存日志是在恢复时打印的，不能据此重建它前 215K 的实际代码，更不能仅凭名字断言具体 bug。
另一方面，后续文档把 v2 称作 legacy，主要涉及与 DDPM 阶段分配的比较约定；这不等于其历史低 FID 属于另一个模型。

历史正式采样 JSONL、实验账本均把以下结果归于 **同一个 `RDD-FM-Unified-XL-v2/0400000.pt`**：

| 边界 | solver / 更新数 | CFG | FID |
|---|---|---:|---:|
| noise_filling | Euler，80/80/90 | 1 | 14.6173905 |
| noise_filling | Euler，80/80/90 | 1.5 | 3.3587141 |
| noise_filling | midpoint，40/40/45 | 1.5 | 3.2908968 |
| fresh，本次用户提供的远端结果 | Euler，80/80/90 | 1 | 33.2233 |

历史账本：[RDD_COMPLETE_EXPERIMENT_LEDGER_20260918.md](../../research_docs/RDD_COMPLETE_EXPERIMENT_LEDGER_20260918.md)。历史原始采样配置已摘录到本报告 JSON；本次没有重算这些 FID。

## 2. 实际读取权重内部，而不是只看文件名

固定 HF revision 为 `848d8a1eccaab8c1807ab792c4ec403dde25288d`，文件为
`checkpoints/RDD-FM-Unified-XL-v2/0400000.pt`。

通过 HTTP Range 只读 ZIP 目录和 `data.pkl`，共读取 **299,766 bytes 元数据**；使用受限 pickle reader，不加载 tensor storage，也不执行任意 pickle globals。没有下载 10.8GB 权重。

内部字段确认：`step=400000`、`DiT-XL/2`、`resolutions=[32,16,8]`、`stage_lengths=[300,300,400]`、`s_boundaries=[0,.3,.6,1]`；训练目录是 v2，resume 指向 v2 的 30K。model/EMA 各有 291 个 state entries。

HF API 的 LFS SHA256 与本仓库锁文件一致：

```text
dbbbe1325494295ac17e0a7c33cda733058cadd06e989b2f8c0d9ce9e6e543c2
```

这是远端 LFS 身份与锁文件的比对，本机没有读取完整 tensor 文件重新计算其 SHA256。

## 3. 用户提供的正式运行记录

- 原始 checkpoint SHA256 与上面的固定资产相同；导入选择 `ema`，step 为 400K。
- 远端核验称 **291/291 个导入张量与原始 EMA 相等**，且 291 个都不同于原始 `model` 字段。
- 实际采样 run、导入记录和当前推理文件 SHA256 均为 `985b9fb14414f970feb6a9bebd7eb83f07576a36b962a1ddaa4297029bfc4e6f`。
- 实际 method 配置与本仓库 fresh 配方完全相同，仅增加导入器正常设置的 `inference_only=true`。
- VAE 权重 SHA256 与 lock 相同；另取固定 revision 的 547-byte VAE config，SHA256 也与远端记录相同。latent scale 为 `0.18215`。
- 实际运行是 FP32/允许 TF32、batch16、50K、CFG1、Euler80/80/90、fresh；不存在 `.4/.7` 时钟误配。

因此，依据这份记录，应停止把“误用了 raw model 或早期 XL”作为当前解释。

## 4. 代码与数值核验

与归档提交 `c90213b1acfef13248f721986a28f9ac6a014f0a` 比较，去掉注释/docstring 后：

- 整个 `DiT/models_multires.py` 的执行 AST 相同。
- `FlowMatchingWithDrift`、`upsample_with_noise_fm`、`noise_filling_transition_fm` 执行 AST 相同。

本次再次运行 CPU FP32 对照：随机非零小 DiT、真实 8/16/32 latent 网格、CFG1、Euler80/80/90，分别测试 fresh 和 noise_filling。**每种方法各自在旧版与 unified 之间，252 次网络调用、27,468 对模块边界张量、最终 latent 与 RNG 状态均逐字节一致。** 两种边界方法彼此输出不同；没有把 fresh 与 filling 测成同一个分支。

此测试证明所测配置的迁移一致性，不证明完整预训练 XL 的 GPU 采样质量。

新发现并补充的保护：复用 `fm_ema_inference.pt` 时，现在除了源/输出哈希，还检查导入字段必须是 EMA、step 必须为 400K。首次自动导入原本就明确选择 EMA。**远端记录已证明本次用了 EMA，这个保护缺口不是本次已发现的故障原因。** 相关资产检查、复用拒绝和采样像素测试：3 passed。

## 5. 仍不能从这些记录推断的事情

实际远端使用 `torch.compile(mode=reduce-overhead, fullgraph=True, dynamic=False)` 和四进程 round-robin 分批；之前发布的 CPU parity 没有覆盖这一执行方式。后续用户补充了真实权重采样复核和完整评估复算的说明，见第 6 节。这补足了仅有 run.json 时的证据；本机仍未收到实际源码复核包。

远端导入器源码 hash 也不同于本地；但导入后的 291 个 tensor 已与原 EMA 比对，所以源码 hash 不同本身不是权重错误的证据。正式采样源码、编译和分片仍需单独核验。

还有已知协议区别：新包固定每类 50 张、每批 `seed+batch_start`，历史采样随机抽类别并使用旧 RNG 顺序。同样的整数 seed 不保证逐图复现，不能把这个新 fresh 实验描述成历史评估仅改了一个字段。

最小定位步骤及后续证据状态：

1. 真实权重旧版 eager / unified eager / compiled 输出检查：后续说明已报告四个 batch、64 张完整采样的复核；无需把它继续列为“从未做过”，但本机没有独立读取原始记录。
2. 全局编号、类别及文件检查：后续说明报告全部 50K 检查通过；本机另验证了其分片规则的覆盖、各 rank 数量和四个抽查 batch 所属 rank。
3. **仍缺：同一新协议下仅改 fresh→noise_filling 的 50K 质量对照。** 小规模配对输出检查已经报告完成，但不等同于正式 FID 对照。

注意：现成 `fm-sample` 包内会强制设置 fresh，**只改 JSON 的 boundary 不会生效**。对照应使用支持该字段的 native 入口，或给远端实验入口显式增加 boundary 参数并确认 run.json，而不能以为改配置后已经测了 filling。

本次没有启动 GPU、正式采样或训练；也没有把用户提供的远端记录冒充本机独立执行结果。

## 6. 后续远端执行说明复核

用户随后提供了更完整的执行说明。摘要与本机算术检查存于 [FM_REMOTE_EXECUTION_REVIEW.json](FM_REMOTE_EXECUTION_REVIEW.json)。其 2.85MB 复核包位于另一服务器，未在本机直接访问；以下真实采样/评估执行结果仍以对方报告为依据。

- 四卡按全局 batch 轮转的规则正确：50K / 16 = 3125 批，rank0 分到 782 批，其余各 781 批；无须在 rank 间同步模型。抽查的 start=0/25104/37600/49968 分别属于 rank0/1/2/3，本机计算确认。
- 对方报告 50K 编号齐全、逐文件校验通过、50K 个不同 PNG 哈希、每类 50 张。硬链接只改变文件存储方式，不改变像素内容。
- 对方使用真实 EMA 对 64 张完整轨迹复核：旧版 fresh 与 unified eager 逐元素一致；compiled 重生成与正式 PNG 逐像素一致。eager/compiled PNG 的 MAE 为 0.013–0.038 个 8-bit 灰度级，PSNR 61.09–66.91dB。这不支持“编译导致大幅生成质量偏差”的猜测，也不是全 50K eager FID 复算。
- 完整 50K 重新打包的 NPZ 哈希一致；独立官方 evaluator 复算 FID 仅差 `4.77e-6`，sFID 差 `3.56e-5`，Precision/Recall 完全相同。其输入、统计量读取、IS 分组和 PR 默认值与[固定官方 evaluator](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py)符合。
- 这份说明没有显示能解释 FID 翻倍的操作错误。当前最有价值的追加实验是同协议 50K noise_filling；不应没有新证据就要求把 EMA、VAE、官方指标或 50K eager 全部再查一遍。
- 未提交的四卡/编译代码虽打包保留，但基础 Git commit 单独不足以复现运行；应把实际修改提交归档。这是可复现记录的不足，不是已发现的数值故障。
