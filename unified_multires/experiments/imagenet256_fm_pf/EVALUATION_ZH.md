# 正式50K指标评估

采样使用unified的`NativeRDDFM`入口，其底层调用本仓库原`diffusion/flow_matching_with_drift.py`的`sample_ode`、`upsample_with_noise_fm`。这不是generic RDDProcess替代旧FM算法。当前这套unified训练/采样入口是单进程单GPU，没有DDP训练或rank/world_size采样分片；旧RDD入口的多卡能力不自动传递到新入口。

评估使用仓库内ADM评估器，输出 **FID、sFID、Inception Score、Precision、Recall**。遵循[OpenAI官方评估协议](https://github.com/openai/guided-diffusion/blob/main/evaluations/README.md)。参考NPZ含全数据集预计算统计和用于precision/recall的参考图片，不是训练集，也不是另外一个需要训练的模型。

## 安装和下载

从Git仓库根目录运行：

```bash
EXP=unified_multires/experiments/imagenet256_fm_pf
BOOTSTRAP_PYTHON="$PWD/.venv-fm-pf/bin/python" bash "$EXP/setup_evaluation.sh"
```

创建单独`.venv-adm`：tensorflow-cpu2.17.1、numpy1.26.4、scipy1.12.0等，不改动PyTorch训练环境。只下载以下两个官方资产到`eval_results/reference/`：

- Inception GraphDef，95,673,916bytes：
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb
- ImageNet-256参考，2,037,122,530bytes：
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

2026-11-04核实两个URL均返回HTTP200。脚本校验官方Content-MD5、文件大小，保存本地SHA256。遇到不符停止，不偷偷换模型或参考集。自定义路径用`ADM_ASSET_DIR=/large/disk/adm`，后续评估也传同一个值。网络传输支持curl续传。

## 依次评估FM和PF

FM50K采样完成后：

```bash
bash "$EXP/run_adm_evaluation.sh" unified_multires/runs/fm_pf/fm_fresh_cfg1
```

PF训练400K并按README完成同协议50K采样后：

```bash
bash "$EXP/run_adm_evaluation.sh" unified_multires/runs/fm_pf/pf_400k_cfg1
```

传入的是含`run.json`、`SUMMARY.json`和`png/`的**采样run目录**，不是`png/`本身。脚本拒绝64张预览，检查50K文件名与已有批次hash，再用旧PNG→NPZ工具打包，保持原图像字节。切到Inception模型所在目录调用原评估器，避免隐式下载或找错GraphDef。

输出位于该run的`evaluation/adm.log`、`evaluation/metrics.json`；JSON记录全部5项指标、样本/模型/参考/评估器hash和环境。仅在5项指标成功保存后删除可重建的约9.83GB样本NPZ，保留PNG和日志。失败时保留中间文件供排查。

默认用CPU评估，避免与PF抢GPU；耗时可能较长。旧打包器把图像载入RAM后stack，临时RAM超过20GB，建议64GB主机RAM并预留至少15GB临时磁盘。这里没有为省时间更换成其他FID库；不同Inception实现/resize/参考统计的分数不能混进旧ADM表。

本轮仅核验官方下载地址/元数据和脚本语法，没有在此机器下载这两个评估资产，也没有执行正式50K评估；新服务器必须实际运行并保存结果。不能把采样完成或现有论文FID当作本次fresh实验的指标。
