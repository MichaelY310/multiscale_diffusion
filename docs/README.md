# RDD 理论分析与调研文档总览

本目录汇集了 RDD (Residual Drift Diffusion) 项目的所有理论推导、调研报告和分析脚本。

新机器运行实验或新 agent 接手，请先看[操作指南](AGENT_GUIDE_ZH.md)和
[仓库内容及验证范围](REPOSITORY_CONTENTS_ZH.md)。下列理论索引保留历史研究材料，
不能替代当前代码、实验配置和数值证据。

---

## 核心文档索引

| 文件 | 内容 | 何时阅读 |
|------|------|----------|
| `theory/RDD_research_compendium.md` | **研究全书**：视频生成论文配置对比 + RDD 最优 Stage 分配理论完整推导 + 深度理论数值结果 | 了解 RDD 理论全貌 |
| `theory/RDD_related_work_survey.md` | **相关工作调研**：27 篇多尺度扩散论文的方法细节、数学对比、与 RDD 差异分析 | 写 Related Work / 找差异化 |
| `theory/RESEARCH_REPORT.md` | **自主科研报告**：13 组理论实验，4 个核心发现（线性速度场、95× 加速比、FM 梯度优势、包络函数） | 了解 RDD-FM 理论贡献 |
| `theory/posterior_upsample_derivation.md` | Posterior Upsample 数学推导（阶段衔接公式） | 理解跨分辨率转换 |
| `theory/paper.txt` | 论文正文核心内容 | 写论文时参考 |
| `theory/SNR_VISUALIZATION_GUIDE.md` | 多尺度 SNR 可视化指南 | 画 SNR 曲线图 |
| `theory/experiment_plan.md` | 实验计划（含已有 FID 数据汇总） | 决定下一步跑什么 |

## 调研文档（Related Work 辅助）

| 文件 | 内容 |
|------|------|
| `theory/related_work_comprehensive_survey.txt` | 30 篇论文全面调研（含 2D/3D/Video），每篇含方法、配置、与 RDD 关系评级 |
| `theory/related_work_code_analysis.txt` | 12 个开源 repo 的代码精读（行号级别），含 ImageNet 实验配置汇总 |
| `theory/RDD_vs_all_methods_comparison.txt` | 所有方法的对比大表 |

## 理论实验脚本

均为独立 Python 脚本，可直接 `python expN_xxx.py` 运行：

| 脚本 | 分析内容 |
|------|----------|
| `exp1_fit_power_spectrum.py` | 自然图像功率谱拟合 (1/f² 法则) |
| `exp2_snr_theory_analysis.py` | 多尺度 SNR 理论分析 |
| `exp3_drift_ablation.py` | Drift 包络函数消融 (t(1-t) vs t²(1-t)²) |
| `exp4_fm_vs_ddpm_training_signal.py` | FM vs DDPM 梯度预算对比 (197× 优势) |
| `exp5_block_noise_analysis.py` | Block noise 频谱分析 |
| `exp7_dpm_solver_compatibility.py` | RDD + DPM-Solver 兼容性验证 |
| `exp8_ot_coupling_analysis.py` | Optimal Transport coupling |
| `exp9_power_spectrum_diagnosis.py` | 频谱诊断 |
| `exp10_linear_velocity_theorem.py` | **核心**：RDD-FM 速度场分段线性定理 (R²=1.0) |
| `exp11_cpu_micro_training.py` | CPU 微型训练验证 |
| `exp12_sampling_efficiency_showcase.py` | 95.2× 理论采样加速计算 |
| `exp13_unified_theory_framework.py` | 统一理论框架 |
| `rdd_allocation_theory.py` | 最优 Stage 分配理论 |
| `rdd_allocation_deep_theory.py` | 分配理论深度版 |
| `rdd_allocation_validation.py` | 分配理论数值验证 |
| `rdd_optimal_allocation.py` | 最优分配求解器 |
| `rdd_detailed_analysis.py` | RDD 详细数值分析 |
| `rdd_sparsity_analysis.py` | 稀疏性分析 |

## 数学验证与可视化

| 脚本 | 用途 |
|------|------|
| `theory/verify_fm_math.py` | Flow Matching 公式数学验证 |
| `theory/test_fm_cpu.py` | FM 采样 CPU 端到端测试 |
| `theory/RDD_FM_derivation.py` | RDD-FM 数学推导的代码实现 |
| `theory/visualize_snr.py` | 多尺度 SNR 曲线可视化 |

## 论文笔记

| 文件 | 对应论文 |
|------|----------|
| `theory/LapFlow.txt` | LapFlow (ICLR 2026) 阅读笔记 |
| `theory/UniVid.txt` | UniVid (arXiv 2026) 阅读笔记 |

## JSON 数据

| 文件 | 内容 |
|------|------|
| `theory/deep_theory_results.json` | 深度理论实验的原始数值结果 |
| `theory/rdd_allocation_results.json` | 最优分配搜索结果 |
| `theory/rdd_optimal_allocation_results.json` | 最优分配详细输出 |
| `theory/rdd_sparsity_analysis/` | 稀疏性分析 JSON 数据 |

---

## 已调研的 RDD 相关论文清单

### A. 核心竞品（多分辨率扩散 / 渐进式生成）

| # | 论文 | 会议 | 链接 |
|---|------|------|------|
| 1 | **f-DM**: Multi-stage Diffusion via Progressive Signal Transform | ICLR 2023 | https://arxiv.org/abs/2210.04955 |
| 2 | **Relay Diffusion** | ICLR 2024 Spotlight | https://arxiv.org/abs/2309.03350 |
| 3 | **LapFlow**: Laplacian Flow Matching | NeurIPS 2025 | https://arxiv.org/abs/2506.19839 |
| 4 | **DiMR**: Multi-Resolution Diffusion Transformer | ICLR 2026 | https://arxiv.org/abs/2602.19461 |
| 5 | **Pyramid Flow**: Video Generation with Flow Matching | ICLR 2025 | https://arxiv.org/abs/2410.05954 |
| 6 | **Matryoshka Diffusion Models** | ICLR 2024 | https://arxiv.org/abs/2310.15111 |
| 7 | **simple diffusion** (Hoogeboom et al.) | ICML 2023 | https://arxiv.org/abs/2301.11093 |
| 8 | **SPEED**: Spectrally Enhanced Progressive Denoising | arXiv 2025 | https://arxiv.org/abs/2605.18736 |
| 9 | **SwD**: Scale-wise Distillation | arXiv 2025 | https://arxiv.org/abs/2503.16397 |
| 10 | **LowDiff** | arXiv 2025 | https://arxiv.org/abs/2501.13349 |
| 11 | **FreCaS**: Frequency and Cascade Self-Ensemble | NeurIPS 2024 | https://arxiv.org/abs/2406.09416 |
| 12 | **DCTdiff**: Frequency Diffusion in DCT Space | ICML 2025 | https://arxiv.org/abs/2412.15032 |
| 13 | **DFM**: Diffusion Flow Matching | arXiv 2024 | https://arxiv.org/abs/2512.21593 |
| 14 | **MrFlow** | arXiv 2025 | https://arxiv.org/abs/2509.15342 |
| 15 | **TPDiff**: Temporal Pyramid Diffusion | arXiv 2025 | https://arxiv.org/abs/2607.01642 |
| 16 | **PyramidalWan** | CVPR 2026 | https://arxiv.org/abs/2601.04792 |
| 17 | **FlowAR**: Autoregressive Flow Matching | ICML 2025 | https://arxiv.org/abs/2502.06782 |
| 18 | **PixelFlow** | arXiv 2025 | https://arxiv.org/abs/2505.13219 |
| 19 | **Blockwise Parallel Flow Matching** | arXiv 2024 | https://arxiv.org/abs/2403.03206 |

### B. 3D 多分辨率生成

| # | 论文 | 会议 | 链接 |
|---|------|------|------|
| 20 | **XCube**: Large-Scale 3D Generative Model (NVIDIA) | CVPR 2024 | https://arxiv.org/abs/2312.03806 |
| 21 | **OctFusion**: Octree-based Diffusion for 3D | CGF/SGP 2025 | https://arxiv.org/abs/2408.14732 |
| 22 | **PDD**: Pyramid Discrete Diffusion for 3D | ECCV 2024 Oral | https://arxiv.org/abs/2311.12085 |
| 23 | **Mesh Denoising Diffusion** (Inria + Adobe) | CVPR 2025 | https://arxiv.org/abs/2502.02187 |
| 24 | **Wavelet Latent Diffusion** (Autodesk) | 2024 | https://arxiv.org/abs/2411.08017 |
| 25 | **ResGen**: Residual Diffusion for Point Cloud | 2024 | https://openreview.net/forum?id=v1gaXJDCbf |

### C. 高分辨率/多尺度架构

| # | 论文 | 会议 | 链接 |
|---|------|------|------|
| 26 | **Hourglass Diffusion Transformer (HDiT)** | ICML 2024 | https://arxiv.org/abs/2401.11605 |
| 27 | **DemoFusion** | CVPR 2024 | https://arxiv.org/abs/2311.16973 |
| 28 | **ScaleCrafter** | ICLR 2024 | https://arxiv.org/abs/2310.07702 |
| 29 | **Spectral Noise Schedule** | ECCV 2024 | https://arxiv.org/abs/2402.10491 |
| 30 | **ZoomLDM** | CVPR 2025 | https://github.com/cvlab-stonybrook/ZoomLDM |
| 31 | **DiffuseHigh** | NeurIPS 2023 | https://arxiv.org/abs/2304.12526 |

### D. 基础方法参考

| # | 论文 | 会议 | 链接 |
|---|------|------|------|
| 32 | **DiT**: Scalable Diffusion Models with Transformers | ICCV 2023 | https://arxiv.org/abs/2212.09748 |
| 33 | **ADM**: Diffusion Models Beat GANs | NeurIPS 2021 | https://arxiv.org/abs/2105.05233 |
| 34 | **DPM-Solver**: Fast ODE Solver for Diffusion | NeurIPS 2022 | https://arxiv.org/abs/2206.00927 |
| 35 | **Flow Matching** (Lipman et al.) | ICLR 2023 | https://arxiv.org/abs/2210.02747 |
| 36 | **Rectified Flow** (Liu et al.) | ICLR 2023 | https://arxiv.org/abs/2209.03003 |
| 37 | **Hierarchical Rectified Flow** | ICLR 2025 | https://arxiv.org/abs/2208.01864 |
| 38 | **Frequency Diffusion** | ICLR 2025 Workshop | — |

---

## 文件清理说明

以下旧文件已被合并/删除（内容已保留在上述核心文档中）：
- `related_papers_analysis.txt` → 已合并入 `related_work_comprehensive_survey.txt`
- `existing_experiments.txt` → 已合并入 `experiment_plan.md`
