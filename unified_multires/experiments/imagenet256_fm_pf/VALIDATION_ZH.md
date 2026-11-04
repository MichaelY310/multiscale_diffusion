# 启动包验证（2026-11-04）

CPU验证使用随机小DiT和随机小AutoencoderKL，没有下载正式模型或数据，没有使用GPU。

- 独立新建Python3.12虚拟环境，实际执行`setup_environment.sh`安装CPU版Torch2.8.0/torchvision0.23.0及固定requirements；`pip check`和模型/codec导入通过。GPU wheel及新服务器驱动仍须现场检查。
- `tests/test_fm_pf_launch.py`的4项检查通过：资产hash失败检测及公平配置核对；完整252NFE FM fresh/CFG1.5采样、真实VAE加载解码、PNG与直接原生method执行逐像素一致；续跑跳过已完成批次、配置变化与损坏文件拒绝；PF实际VAE编码/forward/backward/优化器更新，1+2步恢复与连续3步的model/EMA/Adam状态逐元素一致，并执行252NFE PF采样；数据原字节写入和防覆盖。
- 同样4项检查在原有CPU环境也通过。源码通过Python语法和启动shell语法检查。
- 正式HF资产只查询了公开元数据，确认revision、文件大小和hash。真实下载由新服务器的显式`assets`动作执行。未实际下载ImageNet；导出器不代用户接受访问条款。

本次没有运行完整XL/真实权重的采样、400K训练或50K FID。新服务器必须先跑CPU测试、真实FM预览及完整XL真实数据预检。小模型验证不能替代完整模型显存验证，也不证明训练质量。PF当前入口的CFG仅支持1，其余值明确报错。

为降低GPU恢复峰值，Trainer.load改为先把checkpoint加载到CPU，再由模型/优化器state_dict接口搬到目标设备；CPU训练恢复已验证。GPU内存收益需现场确认。

PNG量化使用旧评估器的`((clip(x,-1,1)+1)/2*255).uint8`截断。新共同采样协议使用固定顺序类别与固定batch；不能宣称与旧随机类别评估逐张相同。所有正式结果以新服务器的run.json、进度和评估记录为准。
