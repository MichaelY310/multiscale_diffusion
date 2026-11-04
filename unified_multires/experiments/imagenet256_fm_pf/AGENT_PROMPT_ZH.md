# 给新服务器agent的任务

你现在位于已clone的`MichaelY310/multiscale_diffusion`仓库根目录。请实际执行实验，先完成RDD-FM fresh采样，再开始Pyramid Flow在ImageNet-256上的训练。不要只提供计划或命令；无需询问常规环境安装、指定资产下载或已授权实验启动的确认。

先完整阅读`unified_multires/experiments/imagenet256_fm_pf/README_ZH.md`、两个JSON配置、`assets.lock.json`和`VALIDATION_ZH.md`。以这些文件为准，不加载其他服务器的旧聊天或恢复旧监控。

## 目标及固定范围

1. 使用已有RDD-FM Unified-XL v2 400K **EMA**，阶段0/.3/.6/1，fresh边界、Euler80/80/90（实际252NFE）。先64张预览，再50K正式样本，主对比CFG1，seed0，默认采样batch4/解码batch1。不要把旧noise_filling的FID填进fresh结果。CFG1.5是额外实验，不替代CFG1主表。
2. PF使用本目录`pyramid_flow_train.json`：同DiT-XL/2主体、同SD-VAE-ft-ema、同ImageNet预处理；latent8→16→32；名义0/.3/.6/1；保留PF自身路径、目标、作者network clock、RoPE及补噪缩放；三阶段mean loss系数1相加；global256；FP32/允许TF32；AdamW lr1e-4、EMA0.9999；**从头训练到总400K更新**，不能用RDD训练后的权重初始化PF。PF后续Euler81/81/90同为252NFE。
3. 这是共同设置下的方法比较，非PF原论文最佳系统复现。旧RDD八卡随机流、作者原1:2:1阶段分配等差异在README中已披露，不能抹掉。不能擅自换小网络、BF16、VAE、固定latent、数据子集或阶段日程。

## 按顺序执行

1. 检查本机GPU及正在运行的作业、RAM、磁盘、驱动/Python。只用一张获准且空闲的GPU，不终止别人进程；具体编号按本机情况选择。不要沿用旧surfi1的卡号或过期时钟。本runner为单进程单GPU，不套torchrun。没有可用GPU或显存不足时准确报告，不能声称已经开始正式训练。
2. 从根目录执行`bash unified_multires/experiments/imagenet256_fm_pf/setup_environment.sh`，使用独立`.venv-fm-pf`。必要时选择Python3.11/3.12或README中的匹配CUDA wheel，不能修改其他项目共享环境。设`UMDIFF_PYTHON="$PWD/.venv-fm-pf/bin/python"`。执行README里的CPU测试；失败先修复和记录原因，不能跳过。
3. 运行`"$UMDIFF_PYTHON" unified_multires/experiments/imagenet256_fm_pf/workflow.py assets`。只下载lock指定的FM checkpoint和VAE两个文件，检查hash；不clone整个HF归档、不下载PF预训练权重。大磁盘可通过`--assets`指定并在后续命令保持一致。模型来源公开；遇到权限/网络阻塞如实报告，不获取或泄露其他账户token。
4. 设置一张允许的`CUDA_VISIBLE_DEVICES`，按README运行FM64张预览。检查PNG、非有限值、标签和日志；输出位置写入交接。随后同一采样协议生成50K CFG1图像。中断用相同命令恢复；不能改batch/seed覆盖已有结果。若batch4显存不足，先用新目录验证较小batch，再固定正式采样和PF评估均使用它。保存配置、环境、模型/VAE哈希和计时。当前无自动FID，必须区分“采样完成”和“质量评估完成”。
5. 优先寻找已有原始ImageNet-1K训练集；设置`IMAGENET_TRAIN_DIR`。若只有合法原tar，用`prepare_imagenet.py --archive`；若无数据但服务器账号已获HF ILSVRC/imagenet-1k访问权限，用README的`--hf`流式原字节导出。允许下载本实验需要的数据，不下载validation/test。不要代用户接受访问条款；缺访问权限或磁盘不足时只在必要信息处询问，先完成不依赖ImageNet的FM采样。不能改用其他数据集开始正式PF训练。
6. 用`run_pf_train.sh --preflight-only`实际执行完整XL的真实图片→VAE→forward/backward→optimizer/EMA→checkpoint。默认micro1/global256；仅在首次更新前允许选择更大的microbatch，必须整除256并记录。预检第1步就是正式训练第1步，后续直接恢复，不重置。显存不足不得静默降低模型规模或改精度。
7. 预检通过后通过本机调度系统或tmux运行`run_pf_train.sh`至总400K；记录PID/session、命令、日志、输出目录和最新step，验证进程确实在更新。长训练还在运行时如实报告“运行中”，不能把启动成功当作训练完成。出现故障从latest恢复；不要盲目并行启动两个同目录训练进程。
8. 训练完成后使用README的`pf-sample`以相同50K/CFG1/seed/batch采样。已有ADM资产时用同一评估器比较两边，否则明确列出缺失的评估资产，不编造FID。保存指标和证据。只清理本任务产生且可重建的临时tensor/中间NPZ，不删正式checkpoint、他人文件或数据集。

## 交付记录

在本机run目录维护一份简短状态文档，列出Git提交、选用GPU、环境版本、资产路径/hash、数据来源/数量、有效batch/精度、FM完成张数和样例位置、PF当前step/损失/峰值显存/进程、checkpoint路径，以及已完成或缺失的FID评估。发生配置变化必须单列，不能仍称同一实验。

开始执行；只有真实外部阻塞（数据访问权限、无可用设备、空间不足等）才需要我补信息。
