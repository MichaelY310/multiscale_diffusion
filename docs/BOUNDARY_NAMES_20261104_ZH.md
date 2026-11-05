# RDD 边界名称统一（2026-11-04）

按用户定义：

- `fresh`（同义 `refresh`）：丢弃旧噪声，用预测clean加整张全新IID噪声。
- `noise_filling`：从旧noisy和预测clean恢复残差，归一化上采样后保留，新增细节子空间补新噪声。
- `legacy_initial_noise`：保留阶段最初抽的噪声。它不是上述从末端恢复残差的noise filling。

旧DDPM主评估脚本中的noise_filling实际是fresh；旧FM同名函数是真正恢复残差。视频native入口中旧noise_filling实际fresh、旧innovation_preserving实际恢复残差。旧DDPM innovation_preserving则可能是阶段初始噪声，不能机械统一成恢复残差。

可执行模板已将旧DDPM/video fresh行为写成fresh，历史文件夹、日志、原始结果不改。旧DDPM JSON写noise_filling会报歧义；要选择新的恢复残差变体，显式设置`boundary_semantics: recovered_v2`。旧结果的FID仍归给其实际旧算法，不能归给新变体。

原训练目标没有因改名而变化。fresh与noise_filling是采样边界的两个不同选择，并非同一个算法的两个名字。归一化lift的白噪声结论需要理想Gaussian残差；任意模型预测残差不自动满足，不能声称已解决所有后验分布问题。

**fresh同名不代表DDPM与FM的噪声日程相同。** 旧DDPM在每阶段重新归一化累计alpha，
阶段出口接近clean；当前FM使用全局`t,1-t`，0.3/0.6出口仍有较多噪声。
同系数下两者fresh算子已在CPU逐字节核对一致；真实系数和clean恢复检查见
[专项核查](validation/20261104/FM_DDPM_FRESH_AUDIT_ZH.md)。

训练/采样证据及限制见[仓库内容说明](REPOSITORY_CONTENTS_ZH.md)与[验证报告](validation/20261104/README.md)。
新unified原生配置现已并入本仓库的[unified_multires/configs/rdd_reproduction/](../unified_multires/configs/rdd_reproduction/README_ZH.md)；目录合并不改变上述边界语义。
网站：<https://rdd-math-reference.aa3101132006.chatgpt.site/boundaries.html>。
