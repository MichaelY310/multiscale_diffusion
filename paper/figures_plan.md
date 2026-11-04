# 论文图表计划

## 已生成的图（可直接使用）

| 图 | 文件 | 状态 | 说明 |
|----|------|------|------|
| Fig 1 | `experiments_scripts/fig_method_overview.pdf` | ✅ 已生成 | 方法概览：时间轴+分辨率段+SNR/DNR曲线 |
| Fig 3 | `experiments_scripts/fig_effective_noise_dnr.pdf` | ✅ 已生成 | DNR vs timestep，展示两个学习regime |
| Fig 2 | `experiments_scripts/fig_convergence_curve.pdf` | ⚠️ Placeholder | 需要真实数据替换（等exp04完成） |

## 需要实验数据的图

### Fig 2: Convergence Curve (FID vs Training Steps)
- **依赖实验**: exp04_convergence_curve.sh
- **数据来源**: 各checkpoint的FID评估结果
- **脚本**: `fig02_convergence_curve.py` (模板已有，需替换placeholder数据)
- **画法**: 
  - X轴: training steps (50K-750K)
  - Y轴: FID (log scale)
  - 两条线: DiT baseline (蓝) vs RDD (红)
  - 虚线标注 iso-FLOP 点 (525K DiT = 400K RDD)
  - 水平虚线标注 DiT-7M 的 FID=2.27
- **更新方法**: 运行完exp04后，修改fig02脚本中的数据数组，重新执行

### Fig 4: PSD Analysis
- **依赖实验**: exp09_psd_analysis.py
- **数据来源**: drift d^c 的频率分析
- **脚本**: exp09_psd_analysis.py (画图集成在脚本内)
- **画法**:
  - Radial PSD曲线：3条线（original z, low-freq UP(DOWN(z)), drift d^c）
  - 标注 Nyquist/2 分界线
  - 标注高频能量占比

### Fig 5: Ablation Summary (Bar Chart)
- **依赖实验**: exp01 (no-drift), exp06 (lambda sweep)
- **数据来源**: factorial decomposition结果
- **画法**:
  - 堆叠柱状图：DiT → +MultiRes → +Drift → +DPM2 → Full RDD
  - 展示各component贡献
- **脚本**: 需新建 `fig05_factorial_bars.py`（等实验数据后）

## 可以直接从现有数据画的表

### Table: Per-Stage CFG (等exp05完成后填入)
### Table: σ_h Sweep (等exp07完成后填入)  
### Table: Drift λ Sweep (等exp06完成后填入)

## 不需要实验的图（已生成）

1. **Fig 1 (Method Overview)**: 展示RDD的核心思想——时间轴分段、分辨率递减、drift方向
   - 已用matplotlib生成，包含：时间轴分段图 + SNR曲线 + DNR曲线
   - 可能需要后续美化（用TikZ重绘用于camera-ready）

2. **Fig 3 (Effective Noise DNR)**: 展示两个学习regime的分界
   - 纯计算图（只需noise schedule参数）
   - 已生成，展示每个stage内DNR从0上升到峰值再下降的pattern
   - 注意：当前用 ||d^c||/sqrt(n) = 1 归一化；得到真实数据后需要更新

## 执行顺序

1. ✅ 直接画: fig01, fig03 (已完成)
2. ⚠️ 等exp04完成后更新: fig02
3. 🔲 等exp09完成后生成: fig04 (PSD)
4. 🔲 等exp01/06完成后生成: fig05 (factorial bars)
5. 所有结果table等对应实验完成后填入论文

## 论文中图表的最终排列

- Figure 1: Method Overview (概览图) — §1 Introduction
- Figure 2: Convergence Curve — §4.2 Main Results  
- Figure 3: DNR Curriculum — §3.3 Frequency-Decomposition Curriculum
- Figure 4: PSD Analysis — §3.3 或 Appendix
- Figure 5: Factorial Decomposition — §4.x Ablation

- Table 1: Main Results (已有)
- Table 2: Fair CFG=1.0 Comparison (已有)
- Table 3: NFE Pareto (已有)
- Table 4-7: Various ablations (已有)
- Table 8: Published Comparison (已有)
- NEW Table: Iso-FLOP comparison (exp02)
- NEW Table: Per-stage CFG (exp05)
- NEW Table: σ_h sweep (exp07)
- NEW Table: λ sweep (exp06)
- NEW Table: Factorial decomposition (exp01)
