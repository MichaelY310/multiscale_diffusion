# RDD 自主科研实验报告

**日期**: 2026-07-31  
**实验目录**: `/home/tiger/multiscale_diffusion/research_experiments/`  
**总计**: 13 组实验，生成 40+ 张分析图表

---

## 一、核心发现摘要（论文级贡献）

### 🏆 发现 1：RDD-FM 速度场的分段线性性质（Piecewise Linear Velocity）

**定理**：RDD-FM 在每个 stage 内的速度场 v(x_t, t) 是 t 的**精确线性函数**：
```
v(x_t, t) = (x₁ - ε) + (1 - 2t) · d
           = (x₁ - ε + d) - 2d · t
           = a + b·t  (其中 b = -2d 是常数)
```

**推论**：二阶 ODE 求解器（Heun/中点法）对每个 stage **仅需 1 步即精确**！

**数值验证**（exp10）：
- R² = 1.000000000000000（完美线性）
- Heun 6 NFE（3 stages × 2 NFE）误差 = 1.21e-16（机器精度）
- 对比：Euler 150 NFE 误差仍有 6.78e-03

**论文价值**：这是一个**非平凡的理论结果**——标准 FM 速度场是常数（trivially 1 步精确），但 RDD-FM 加入了 drift 后速度场变为线性而非更高阶，恰好是二阶求解器的甜点。

### 🏆 发现 2：95.2× 理论采样加速比

**计算**（exp12）：
| 配置 | 总计算量 (GFLOPs) |
|------|-------------------|
| DDPM 250 步 (32×32) | 29,750 |
| RDD-FM + Heun (6 NFE, 多分辨率) | 312.4 |
| **理论加速比** | **95.2×** |

分辨率分解：
- Stage 3 (8×8, 2 NFE): 14.9 GFLOPs（仅占 4.8%）
- Stage 2 (16×16, 2 NFE): 59.5 GFLOPs
- Stage 1 (32×32, 2 NFE): 238.0 GFLOPs

实际（σ=0.1 模型误差）预估：Heun-6 ≈ Euler-30 质量 → **实际 ~12-15× 加速**。

### 🏆 发现 3：FM 框架对 RDD 有 197× 梯度预算优势

**实验 4 核心数据**：
| 指标 | DDPM | Flow Matching | FM 优势 |
|------|------|---------------|---------|
| 中间时间步 (0.3-0.7) 梯度预算 | 0.002 | 0.400 | **197×** |
| 漂移修正幅度范围 | [0.006, 100] | [0, 1] | 有界 vs 爆炸 |
| Loss 均匀性 (CV) | 0.2652 | 0.0010 | **272×** |
| 梯度方差比 (max/min) | 13.5× | 1.0× | **13.4×** |

**物理意义**：FM 的均匀时间加权恰好在 drift 最活跃的区域（t=0.3-0.7）分配了大量梯度，而 DDPM 在此区域几乎无梯度（SNR 权重极低）。

### 🏆 发现 4：t²(1-t)² 包络优于 t(1-t)

**实验 3&11 数据**：
- 速度方差：t²(1-t)² = 731K vs t(1-t) = 19.7M（**降低 27 倍**）
- 训练 loss (5000步)：t²(1-t)² = 0.6501 vs t(1-t) = 0.8068（**降低 19.4%**）
- 满足 ∫₀¹g'(t)dt = 0 ✓（ODE 终点正确）
- 边界导数为零 → 速度场在 t=0,1 处更平滑

**建议**：将 RDD-FM 的 drift 包络从 `t(1-t)` 升级为 `t²(1-t)²`。

### 🏆 发现 5：RDD 保留 42.3% 更多低频互信息

**实验 13**：
- 在 t=0.5 时，RDD 的 I(x_t; x₁) 比标准 FM 高 42.3%
- 目标方差平均降低 18.4%，高频中间时刻最大降低 78.4%
- **物理意义**：drift 使中间状态更偏向低频，模型在中间时刻的任务更简单

---

## 二、诊断性发现

### 发现 6：SPEED 公式不适配 DiT VAE latent（exp9）

- DiT VAE latent 功率谱：A=13910, β=1.54
- SPEED 原始参数（FLUX）：A=203.6, β=1.92
- **问题**：DiT P(ω=4)=1657 >> FLUX P(ω=16)=0.99，差异 1000-6000×
- **后果**：SPEED 公式给出 t*>0.999（几乎全在第一阶段），完全无意义
- **根因**：32×32 latent 分辨率太低，功率集中在极少数频率上
- **建议**：对 RDD 使用经验验证的 stage_lengths=[300,300,400]，不依赖 SPEED

### 发现 7：OT Coupling 与 drift 完全正交（exp8）

- OT 减少传输代价 3.2%（高维空间中效果有限）
- Drift 贡献与 OT 配对方式无关（d 只依赖 x₁）
- 预估净训练加速 1.5-1.65×
- **结论**：可以安全引入 OT，作为训练效率改进

### 发现 8：RDD 加速而非延迟高频激活（exp2）

- 标准 DDPM 中 f=16 需 986 步才激活
- RDD 中 f=16 只需 742 步激活（**快 24.7%**）
- **原因**：drift 在前向过程中更快压制高频 → 反向时更早需要恢复
- **直觉修正**：RDD 不是"延迟高频生成"，而是"强制模型更早关注低频"

### 发现 9：Block Noise 的最佳选择是 Spectral Noise（exp5）

- Spectral Noise（DCT 域仅高频填充）频率相关性最高 (0.54)
- White Noise 和 Block Noise 各有局限
- Block Noise (2×2) 有效维度仅为白噪声的一半 → 降低生成难度
- **建议**：RDD 的 noise_filling 用 SPEED 的 DCT spectral expansion 替代白噪声

---

## 三、方法论贡献

### RDD 作为统一框架（exp13）

RDD 通过 drift 强度 λ 统一了多种方法：
- **λ=0**: 恢复标准 FM/DDPM（无多分辨率）
- **λ∈(0,1]**: RDD（连续多分辨率）
- **λ→∞**: 退化为独立级联（类 f-DM）
- **K→∞**: 退化为连续频率扫描（类 Frequency Diffusion）

RDD 是**唯一**同时满足以下所有性质的方法：
| 性质 | RDD-FM | Relay | Pyramid-Flow | f-DM | FlowAR |
|------|:---:|:---:|:---:|:---:|:---:|
| 连续路径 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 单一模型 | ✓ | ✗ | ✓ | ✗ | ✓ |
| DPM-Solver 兼容 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 频率自适应 SNR | ✓ | 部分 | ✗ | ✓ | ✗ |
| 闭式后验 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 线性速度场 | ✓ | N/A | ✗ | N/A | ✗ |

---

## 四、实验文件索引

| 编号 | 文件 | 内容 | 图表数 |
|------|------|------|--------|
| 1 | `exp1_fit_power_spectrum.py` | VAE latent 功率谱拟合 | 1 |
| 2 | `exp2_snr_theory_analysis.py` | 频域 SNR 理论分析 | 4 |
| 3 | `exp3_drift_ablation.py` | Drift 包络消融 | 6 |
| 4 | `exp4_fm_vs_ddpm_training_signal.py` | FM vs DDPM 训练信号对比 | 6 |
| 5 | `exp5_block_noise_analysis.py` | Block/Spectral Noise 分析 | 6 |
| 7 | `exp7_dpm_solver_compatibility.py` | DPM-Solver 兼容性 | 4 |
| 8 | `exp8_ot_coupling_analysis.py` | OT Coupling 可行性 | 5 |
| 9 | `exp9_power_spectrum_diagnosis.py` | 功率谱诊断与修正 | 4 |
| 10 | `exp10_linear_velocity_theorem.py` | 线性速度场定理证明 | 5 |
| 11 | `exp11_cpu_micro_training.py` | CPU 小型训练验证 | 6 |
| 12 | `exp12_sampling_efficiency_showcase.py` | 采样效率展示（论文图） | 6 |
| 13 | `exp13_unified_theory_framework.py` | 统一理论框架分析 | 6 |

---

## 五、论文写作建议

基于以上实验，建议论文的核心卖点排序：

1. **Theorem: Piecewise Linear Velocity → 6 NFE Exact Sampling**  
   最强卖点。标准 FM 是 constant velocity（trivial），RDD-FM 是 linear velocity（sweet spot），更高阶包络（如 sin）会是非线性（不可精确求解）。

2. **197× Gradient Budget for Drift Learning**  
   FM 在 drift 活跃区域分配 197 倍更多梯度，理论上学 drift 快得多。

3. **Unified Framework: λ-interpolation between methods**  
   RDD 通过单一参数 λ 连接了标准 FM、Relay Diffusion、和独立级联。

4. **42.3% More Information Preservation**  
   Drift 使中间状态保留更多低频信息，降低模型在每个时间步的不确定性。

5. **t²(1-t)² Envelope Improvement**  
   27× 降低速度方差，19.4% 降低训练 loss，几乎零成本改进。

---

## 六、后续 GPU 实验建议（优先级排序）

| 优先级 | 实验 | 预期结果 | 所需 GPU 时间 |
|--------|------|----------|---------------|
| P0 | RDD-FM + t²(1-t)² 训练 400K 步 | 验证 FM > DDPM 收敛 | ~2 天 (8×A100) |
| P0 | Heun 采样 FID 对比 (6/10/20 NFE) | 验证采样效率优势 | ~1 小时 |
| P1 | FA-CFG (stage-wise CFG) 采样 | 零成本 FID 改进 | ~1 小时 |
| P1 | Spectral noise_filling 替代白噪声 | 改善阶段过渡质量 | ~1 小时 |
| P2 | OT Coupling 训练 | 加速收敛 1.5-1.65× | ~2 天 |
| P2 | λ sweep (0.5, 0.75, 1.0, 1.5) | 找最优 drift 强度 | ~8 天 |
| P3 | DPM-Solver-2/3 + RDD-FM 少步 FID | 证明高阶求解器优势 | ~2 小时 |

---

*报告完毕。所有实验脚本均可独立运行，图表已生成保存。*
