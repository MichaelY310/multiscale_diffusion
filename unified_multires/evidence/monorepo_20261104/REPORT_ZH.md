# 目录合并与CPU入口验收（2026-11-04）

## 目录与发布内容

规范仓库：`/scr/yuheng/bugswarm/rdd_workspace_20261005/multiscale_diffusion`。
新框架：该仓库下的`unified_multires/`，普通Git目录，不是submodule或嵌套Git仓库。
旧工作区的`unified_multires`和`git_checkouts/multiscale_diffusion`保留符号链接，指向规范位置。

以旧RDD已发布提交`ec68633ffb121a9a8de714351153b8fde7675927`为基础，核对1479个已有追踪文件后迁移Git元数据。旧RDD算法文件不因本次目录合并而改动。新框架调整源码路径定位、启动环境、相关工具路径和包的可选依赖；CPU-only激活继续显式拒绝CUDA。临时任务的旧主机名/过期截止日期不再嵌入通用设备选择函数，设备授权与deadline由外部启动环境管理。本次没有执行GPU操作。

发布自有框架、配置、工具、测试、说明和固定来源清单。23个第三方来源的2172个选定源码文件由`sources.lock.json`固定commit与SHA256，使用`tools/fetch_sources.py`按需获取；本地副本全部离线校验通过。第三方Git历史、环境、模型、数据、运行tensor、网站打包副本和约2GB历史证据不重新提交。本地历史资料未删除。

## 实际执行结果

| 检查 | 结果与范围 |
|---|---|
| 原有pytest | 179 passed；64.72秒；现有CPU环境 |
| 新目录/CPU策略/源码获取契约 | 4 passed；含拒绝CUDA探测、错误路径、hash不符以及禁止覆盖测试 |
| README图像闭环 | 配置生成、随机数据训练2步、采样通过；输出`2×3×16×16` |
| 训练恢复 | 2+2步与连续4步的model、EMA、optimizer和最终采样逐元素相等 |
| README视频闭环 | 配置生成、随机数据训练2步、采样通过；输出`1×4×3×16×16` |
| RDD原生DDPM搬迁 | canonical目录与任意名称临时checkout分别训练/采样；model、EMA、optimizer和样本逐元素相等 |
| RDD原生FM搬迁 | 同上；采用恢复残差的noise_filling边界，结果逐元素相等 |
| 完整实验plan | 12份main、6份controlled全部通过；没有分配完整XL网络或加载数据 |
| 作者来源清单 | 23仓库、2172份文件全部匹配本地固定hash；无模型/数据下载 |
| 语法 | 188份自有Python文件AST解析及5个启动shell语法检查通过 |

入口脚本共执行37条子命令、23项检查。测试在Python临时目录中复制必要代码，checkout名特意不使用`multiscale_diffusion`，没有依靠旧目录符号链接寻找RDD。随机checkpoint和tensor已自动清理。机器可读结果、样本hash和执行命令见[SUMMARY.json](SUMMARY.json)；脚本为[validate_monorepo.py](../../tools/validate_monorepo.py)。

## 结论的边界

此次验证的是目录搬迁与可执行入口，不是重新跑完所有论文的数值对照。RDD原生搬迁测试缩小到hidden32、depth2，保留32/16/8网格，采样每阶段3步；没有加载真实预训练权重、VAE或ImageNet，没有正式训练或FID/VBench评估。既有旧RDD/unified对照继续单列于[此前证据](../../../docs/validation/20261104/README.md)，不把新迁移测试改称最佳配置复现。

Pytest的97条弃用警告来自现有NumPy/Torch接口，未出现失败。新框架公开入口仍为单进程、单设备；普通图像VAE解码命令不等于完整文本到视频管线。正式实验使用方式和限制见[README](../../README.md)。
