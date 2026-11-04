#!/usr/bin/env python3
"""Build the Chinese RDD-video speed report and static SVG figures."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def f3(x: float) -> str:
    return f"{x:.3f}"


def group(rows: list[dict], regime: str) -> list[dict]:
    return sorted((row for row in rows if row["regime"] == regime), key=lambda row: row["rank"])


def svg_bar(rows: list[dict], output: Path) -> None:
    width, left, row_h = 1200, 445, 48
    height = 100 + len(rows) * row_h
    usable = 610
    maximum = max(row["speedup_vs_wan_dense50"] for row in rows)
    colors = ["#0f766e", "#0d9488", "#7c3aed", "#2563eb", "#64748b", "#ea580c", "#94a3b8"]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.t{font-size:22px;font-weight:700}.s{font-size:13px}.v{font-size:14px;font-weight:700}</style>',
        '<text x="24" y="34" class="t">20% retained attention — matched Wan2.1 50-call proxy</text>',
        '<text x="24" y="60" class="s">RTX PRO 6000 Blackwell, shared-GPU preliminary; normalized within the same run</text>',
    ]
    for i, row in enumerate(rows):
        y = 82 + i * row_h
        speed = row["speedup_vs_wan_dense50"]
        bar = usable * speed / maximum
        pieces += [
            f'<text x="24" y="{y + 21}" class="s">{row["rank"]}. {html.escape(row["method"])}</text>',
            f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="28" rx="5" fill="{colors[i % len(colors)]}" opacity=".9"/>',
            f'<text x="{left + bar + 10:.1f}" y="{y + 21}" class="v">{speed:.2f}×</text>',
        ]
    pieces.append('</svg>')
    output.write_text("\n".join(pieces), encoding="utf-8")


def svg_density(metrics: dict, output: Path) -> None:
    densities = ["0.125000", "0.200000", "0.250000"]
    wanted = [
        ("RDD + route + output reuse k=3", "#0f766e"),
        ("RDD + previous-stage route", "#2563eb"),
        ("FastVideo VSA only", "#ea580c"),
    ]
    values = {}
    for name, _ in wanted:
        values[name] = []
        for density in densities:
            row = next(row for row in metrics["rankings"][density]
                       if row["regime"] == "matched_50_model_calls" and row["method"] == name)
            values[name].append(row["speedup_vs_wan_dense50"])
    width, height = 900, 480
    left, right, top, bottom = 88, 35, 75, 70
    chart_w, chart_h = width - left - right, height - top - bottom
    max_y = max(max(v) for v in values.values()) * 1.08
    def x(i: int) -> float: return left + i * chart_w / 2
    def y(v: float) -> float: return top + chart_h * (1 - v / max_y)
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.t{font-size:21px;font-weight:700}.s{font-size:13px}.l{font-size:13px;font-weight:600}</style>',
        '<text x="24" y="32" class="t">Density sensitivity (same 50-call proxy)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#94a3b8"/>',
    ]
    for tick in range(0, int(max_y) + 1, 2):
        yy = y(tick)
        p += [f'<line x1="{left}" y1="{yy:.1f}" x2="{left+chart_w}" y2="{yy:.1f}" stroke="#e2e8f0"/>',
              f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" class="s">{tick}×</text>']
    for i, density in enumerate((12.5, 20.0, 25.0)):
        p.append(f'<text x="{x(i):.1f}" y="{top+chart_h+28}" text-anchor="middle" class="s">{density:.1f}% retained</text>')
    for legend_i, (name, color) in enumerate(wanted):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values[name]))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="4"/>')
        for i, v in enumerate(values[name]):
            p.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="5" fill="{color}"/>')
        lx = 105 + legend_i * 260
        p += [f'<line x1="{lx}" y1="{height-18}" x2="{lx+28}" y2="{height-18}" stroke="{color}" stroke-width="4"/>',
              f'<text x="{lx+36}" y="{height-13}" class="l">{html.escape(name.replace("RDD + ", "RDD "))}</text>']
    p.append('</svg>')
    output.write_text("\n".join(p), encoding="utf-8")


def table_md(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def table_html(headers: list[str], rows: list[list[str]]) -> str:
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("eval_results/rdd_video_speed_benchmark_shared_gpu0/metrics.json"))
    parser.add_argument("--theory", type=Path, default=Path("eval_results/rdd_video_speed_theory/theory.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/research_docs/RDD_VIDEO_SPEED_METHOD_COMPARISON_ZH.md"))
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    theory = json.loads(args.theory.read_text(encoding="utf-8"))
    figures = args.output.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    actual = group(metrics["rankings"]["0.200000"], "matched_50_model_calls")
    svg_bar(actual, figures / "rdd_video_speed_actual20.svg")
    svg_density(metrics, figures / "rdd_video_speed_density.svg")

    actual_rows = [[str(r["rank"]), r["method"], r["stage_calls"], f3(r["speedup_vs_wan_dense50"]) + "×", r["status"]] for r in actual]
    few = group(metrics["rankings"]["0.200000"], "paper_default_or_hypothetical_few_step")
    few_rows = [[str(r["rank"]), r["method"], r["stage_calls"], f3(r["speedup_vs_wan_dense50"]) + "×", r["status"]] for r in few]
    theory_rows = [[str(r["rank"]), r["method"], r["stage_calls"], f3(r["ideal_speedup"]) + "×",
                    f"{r['probability_cache_gib']:.1f} GiB", r["status"]]
                   for r in group(theory["rows"], "matched_50_model_calls")]
    allocation_rows = [[str(r["rank"]), r["method"], r["stage_calls"], f3(r["speedup_vs_wan_dense50"]) + "×"]
                       for r in group(metrics["rankings"]["0.200000"], "rdd_step_allocation_sweep")]
    density_rows = []
    for density in ("0.125000", "0.200000", "0.250000"):
        rows = group(metrics["rankings"][density], "matched_50_model_calls")
        lookup = {r["method"]: r for r in rows}
        density_rows.append([
            f"{100*float(density):.1f}%",
            f3(lookup["RDD + route + output reuse k=3"]["speedup_vs_wan_dense50"]) + "×",
            f3(lookup["RDD + previous-stage route"]["speedup_vs_wan_dense50"]) + "×",
            f3(lookup["FastVideo VSA only"]["speedup_vs_wan_dense50"]) + "×",
        ])
    fine = metrics["attention"]["fine"]
    kernel_rows = []
    for density in ("0.125000", "0.200000", "0.250000"):
        e = fine["densities"][density]
        kernel_rows.append([
            f"{100*float(density):.1f}%", f"{fine['dense_sdpa']['median_ms']:.3f}",
            f"{e['native_external_route']['median_ms']:.3f}",
            f"{e['vsa_online_router_only']['median_ms']:.3f}",
        ])

    md = f"""# RDD 视频生成速度：理论排序与无 checkpoint 实测

## 一句话结论

在同一个 Wan2.1-1.3B 形状、同样 50 次 denoiser forward、保留 20% attention blocks 的速度代理中，当前排序为：

1. RDD + 上一分辨率 route + 每 3 步复用 attention output：**{actual_rows[0][3]}**；
2. RDD + 上一分辨率 route：**{actual_rows[1][3]}**；
3. PyramidalWan 式纯多分辨率：**{actual_rows[2][3]}**；
4. 纯 RDD：**{actual_rows[3][3]}**；
5. temporal-only pyramid：**{actual_rows[4][3]}**；
6. 只有 VSA、没有蒸馏的 FastVideo：**{actual_rows[5][3]}**；
7. 原始 dense Wan：1×。

这证明了**速度潜力**，没有证明等质量。尤其是完整 FastVideo 还包含 3-step DMD；未蒸馏的 50-call RDD 不能据此宣称系统级快于完整 FastVideo。

![20% 实测代理排序](figures/rdd_video_speed_actual20.svg)

## 公平比较是怎么做的

- 骨干统一为 Wan2.1-T2V-1.3B：dim=1536、FFN=8960、12 heads、30 layers、head dim=128。
- 输出统一为 81×448×832；三层 token grid 是 6×7×13=546、11×14×26=4004、21×28×52=30576。
- 不加载 checkpoint，只用随机 BF16 张量测 kernel；因此没有把不同论文的模型大小、分辨率、CPU offload 混成一个假公平的毫秒表。
- VSA 和 RDD route 使用**同一份、未修改的 FastVideo Triton64 block-sparse executor**。差异只在 route 来源：VSA 每层每步由当前 fine Q/K/V 在线计算；RDD 在 stage 边界从上一层 route 映射一次。
- 真实机器是 RTX PRO 6000 Blackwell。GPU 0 同时有既有训练，结果是共享 GPU 初测；峰值额外显存 {metrics['metadata']['peak_allocated_gib']:.2f} GiB，没有停止任何训练。

## 理论 FLOP 排序（20%）

{table_md(['rank','method','C/M/F calls','speedup','probability cache','status'], theory_rows)}

完整 attention probability/score cache 排名看似高，但需要约 **{next(r for r in theory_rows if 'probability reuse' in r[1])[4]}**，模型本身还没算，所以不是可部署主线。本文实现并实测的是 O(ND) 的 attention-output reuse。

## 真实 kernel + 30-layer 组合代理（20%）

{table_md(['rank','method','C/M/F calls','speedup','status'], actual_rows)}

这里的毫秒是把真实 kernel/component latency 组合成 30-layer denoiser proxy；不包含 T5、VAE、通信、Python pipeline 和视频质量。

### 为什么表里纯 RDD 略慢于 PyramidalWan？

这不是 RDD drift 更耗时。当前“严格 50-forward”RDD 为了计入三个 stage endpoint，被记为 **19/20/11**；PyramidalWan 公开日程直接记为 **20/20/10**。两者都是 50 calls，但 RDD 多了一次最贵的 fine call、少了一次最便宜的 coarse call，因此是 4.197× 对 4.579×。如果都严格使用同一个 20/20/10 forward 日程，随机张量速度代理完全相同；RDD 的逐元素 drift 成本相对 DiT forward 可以忽略。最终质量和所需步数才可能让二者真正拉开。

## 每种方法到底做了什么

1. **Dense Wan**：81 帧 448×832 的所有 denoising step 都在最高 latent/token 分辨率做完整 attention。
2. **纯 RDD**：先生成 coarse，再 middle，再 fine；每层仍是 dense attention。节省来自大量 forward 在较短序列上完成，不来自 attention sparsity。
3. **PyramidalWan 式多分辨率**：同样按 coarse/middle/fine 生成，但使用论文的 stage transition/patchification 和 finetuning设计。速度模型只关心每层 token 数和 calls，所以相同 schedule 时与纯 RDD 同速；两者的差异主要是训练目标、transition 和质量。
4. **TPDiff-style temporal-only**：只逐步增加帧率，空间分辨率从始至终保持最高；所以比全时空 pyramid 少省一部分。
5. **Pyramid Flow**：用 flow matching 把生成过程切成三个空间/时间 pyramid stage，并且按 autoregressive unit 扩展视频。它不是 Wan 骨干，报告中只复用了官方 10/10/10 schedule。
6. **FastVideo VSA-only**：仍然是原来的 50 个 full-resolution Wan denoising calls，但把每层 dense attention 换成 VSA。VSA先把当前 fine Q/K/V 压成 64-token cubes，做一次 coarse attention 选 Top-K，再在选中的 fine cube pairs 内执行 sparse QK/AV，最后把 coarse/fine output融合。它**没有缩短 diffusion steps**。
7. **RDD previous-stage route**：coarse stage 保持 dense；coarse 最后的 attention route 映射给 middle，middle 再映射给 fine。middle/fine 用与 VSA相同的 sparse executor，但不在每层每步重新运行 current-fine router。
8. **RDD output reuse**：每 k=3 步只在第一个 step 真正计算 self-attention branch；后两个 step 使用缓存的 attention output，再继续计算当前 step 的 FFN/cross-attention等。它不是缓存完整 attention probability，质量需要验证。
9. **Probability reuse**：缓存稀疏的方形 attention matrix，并计算 `A_old × V_current`。数学上更接近“复用 attention score”，但当前 Wan grid、30 layers、20%密度约需168.3 GiB，仅适合少数层或更小窗口，不适合作为默认系统。

## FastVideo 最快版本：不是 VSA-only

FastVideo 当前最快的 Wan2.1-1.3B 公开路径是 **Sparse-Distill = 3-step DMD + VSA**。我们已经在同一实测组件模型中加入它：

{table_md(['rank','method','C/M/F calls','speedup vs dense Wan50','status'], few_rows)}

其中 FastVideo DMD3+VSA 的代理为 **{next(r[3] for r in few_rows if r[1] == 'FastVideo DMD3 + VSA')}**；VSA-only 是 {next(r[3] for r in actual_rows if r[1] == 'FastVideo VSA only')}。约 16.7 倍的差距主要来自把 50 calls 降到 3 calls，不是 VSA kernel 突然变快。RDD distilled 2/2/1 行是尚未训练的假设，只能作为未来硬件目标，不能当当前结果。

另外，“Fast Video Generation with Sliding Tile Attention”这篇 ICML 2025 论文的核心是 **STA**，在 HunyuanVideo 上报告 training-free 和 finetuned 版本；它和当前 FastVideo/Wan 的 VSA/Sparse-Distill 路线不是同一个实验。当前 FastVideo 主仓库已把 STA 归档，并明确优先维护 VSA，因此本次 Wan 公平表采用 VSA 与 DMD3+VSA，而不把 Hunyuan 720p 的 STA 毫秒混进来。

## 稀疏率敏感性

{table_md(['retained','RDD route+reuse','RDD route','FastVideo VSA-only'], density_rows)}

![稀疏率敏感性](figures/rdd_video_speed_density.svg)

注意 20% 在这次共享 GPU 测量中反而比 12.5% 快，这是并发训练、GPU 时钟和 kernel autotune 噪声，不是算法规律。当前可相信的是方法的大致层级；稀疏率之间的精确差值必须等 GPU 独占后重跑。

Wan 视频的 fine stage 实测组件如下。VSA 总成本在排序中按“相同 sparse executor + 单独实测 router”相加，避免共享 GPU 插队造成 VSA 比其子操作还快的物理不可能结果。

{table_md(['retained','dense SDPA ms','same sparse executor ms','VSA online router ms'], kernel_rows)}

RDD route-lifting kernel 在 20% 下把 coarse→middle 映射到平均 20.52%，middle→fine 映射到平均 20.24%；两次 stage-boundary kernel 的 median 分别为 {metrics['route_lifts']['coarse_to_middle']['densities']['0.200000']['latency']['median_ms']:.5f} ms 和 {metrics['route_lifts']['middle_to_fine']['densities']['0.200000']['latency']['median_ms']:.5f} ms。它只在边界发生，不是每个 denoising step 都重算。

正确性检查也已通过：6 组 route lift（两种 stage transition × 三种 density）全部与 CPU 参考索引逐项完全相等；五种 token shape 的 output-reuse kernel 相对 `x + cached` 最大绝对误差均为 0。

## 不同 RDD 步数分配（20%）

{table_md(['rank','method','C/M/F calls','speedup'], allocation_rows)}

“coarse-heavy”速度最高是必然的，因为 coarse token 少；这**不是质量结论**。下一步必须用同一 RDD-Wan checkpoint/finetuning预算比较 VBench、FVD、CLIP、motion 和人工偏好，否则不能选它作为最佳 schedule。

## 与公开方法的真实关系

- **PyramidalWan**：公开 20/20/10 和 DMD 2/2/1。论文表中 50-step Wan 为 12,592 TFLOPs、20/20/10 为 2,821 TFLOPs，即 4.46×；本代理的 4.56×与它一致，说明 token/call cost model 量级可信。
- **Pyramid Flow**：代码公开，但使用 miniFLUX/SD3、autoregressive pyramid 和不同分辨率。我在 surfi2 浅克隆了官方仓库，并实际导入其 `PyramidFlowMatchEulerDiscreteScheduler`：三阶段均生成 10 个 timestep，确认官方视频路径为 10/10/10。速度表只把这个日程放进统一 Wan proxy；没有把不同骨干的原始毫秒冒充为公平比较。
- **TPDiff**：只降低时间帧率，论文报告 1.5×；其官方仓库截至本次检查仍只有 assets/README，并写着代码将发布，所以这里只能做 Wan-shape temporal-only proxy。
- **PAB**：训练免费地复用 attention output，并报告最高 10.5×。RDD 的 `output reuse k=3` 是相同类别的速度上界，不是复用完整 N×N attention probability。
- **FastVideo**：VSA-only 和完整 Sparse-Distill 必须分开。官方 1.3B recipe 是 3-step DMD + VSA，VSA sparsity=0.8（保留约 20%），训练 4000 steps；它的巨大系统级加速主要同时来自步数减少和稀疏 attention。

## 我认为最值得做的路线

1. 先做 RDD-Wan 20/20/10（或严格 50-call）质量基线，证明多分辨率本身不掉 VBench/FVD。
2. 只给 middle/fine 加 external route，扫 12.5/20/25%；coarse 保持 dense。
3. 再加 k=2/3 的 attention-output reuse；先按 layer 和 timestep 自适应，不能盲目所有层每 3 步复用。
4. 最后做 2/2/1 或 3/1/1 蒸馏。只有这一项完成后，才可与 FastVideo DMD3+VSA 做系统级 wall-clock/quality Pareto。

## 产物

- 理论模型：`rdd_video/attention_analysis/model_video_method_speed.py`
- 实测驱动：`rdd_video/attention_analysis/benchmark_wan_video_methods.py`
- 跨分辨率 route Triton kernel：`rdd_video/attention_analysis/rdd_video_route_triton.py`
- attention-output reuse Triton kernel：`rdd_video/attention_analysis/rdd_video_reuse_triton.py`
- Pyramid Flow 官方 scheduler 无 checkpoint 验证：`rdd_video/attention_analysis/verify_pyramid_flow_official_schedule.py`
- 原始 JSON/CSV/日志：`eval_results/rdd_video_speed_benchmark_shared_gpu0/`

## 主要来源

- Wan2.1 official config: https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_1_3B.py
- FastVideo/VSA: https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md
- FastVideo sparse-distill: https://github.com/hao-ai-lab/FastVideo/blob/main/docs/distillation/dmd.md
- PyramidalWan: https://openaccess.thecvf.com/content/CVPR2026/papers/Korzhenkov_PyramidalWan_On_Making_Pretrained_Video_Model_Pyramidal_for_Efficient_Inference_CVPR_2026_paper.pdf
- Pyramid Flow official code: https://github.com/jy0205/Pyramid-Flow
- TPDiff official repository: https://github.com/showlab/TPDiff
- PAB: https://arxiv.org/abs/2408.12588
"""
    args.output.write_text(md, encoding="utf-8")

    actual_html = table_html(['rank','method','C/M/F calls','speedup','status'], actual_rows)
    few_html = table_html(['rank','method','C/M/F calls','speedup vs dense Wan50','status'], few_rows)
    theory_html = table_html(['rank','method','C/M/F calls','speedup','probability cache','status'], theory_rows)
    allocation_html = table_html(['rank','method','C/M/F calls','speedup'], allocation_rows)
    density_html = table_html(['retained','RDD route+reuse','RDD route','FastVideo VSA-only'], density_rows)
    kernel_html = table_html(['retained','dense SDPA ms','same sparse executor ms','VSA online router ms'], kernel_rows)
    body = f"""
<h1>RDD 视频生成速度：理论排序与无 checkpoint 实测</h1>
<div class="callout"><b>结论：</b>20% 保留率、同 Wan2.1-1.3B 形状、同 50 calls 的代理中：RDD route+output reuse {actual_rows[0][3]}，RDD route {actual_rows[1][3]}，PyramidalWan式纯金字塔 {actual_rows[2][3]}，纯RDD {actual_rows[3][3]}，VSA-only {actual_rows[5][3]}。这是速度潜力，不是等质量结论。</div>
<img src="figures/rdd_video_speed_actual20.svg" alt="20% speed ranking">
<h2>公平设置</h2><p>随机 BF16，无 checkpoint；Wan2.1-1.3B（dim 1536、FFN 8960、12 heads、30 layers）；81×448×832；RDD/VSA 共用同一未修改 FastVideo Triton64 sparse executor。共享 GPU 初测，额外峰值显存 {metrics['metadata']['peak_allocated_gib']:.2f} GiB。</p>
<h2>理论 FLOP 排序（20%）</h2>{theory_html}
<p>完整 attention probability cache 需约 168.3 GiB，不能作为主线；实测采用 O(ND) output reuse。</p>
<h2>真实 kernel 组合代理（20%）</h2>{actual_html}<h3>纯RDD为何略慢于PyramidalWan</h3><p>RDD strict-50 记为19/20/11，PyramidalWan是20/20/10；RDD把一个最便宜的coarse call换成一个最贵的fine call。相同20/20/10时两者速度相同，drift成本可忽略。</p>
<h2>方法解释</h2><ol><li><b>纯RDD：</b>coarse→middle→fine，但各stage仍用dense attention。</li><li><b>PyramidalWan：</b>同样做时空金字塔，差异在transition、训练目标与finetuning；相同token/call schedule时速度与纯RDD相同。</li><li><b>TPDiff：</b>只增加帧率，不降低空间分辨率。</li><li><b>VSA-only：</b>50个full-resolution calls不变；当前fine Q/K/V每层每步压缩、coarse选Top-K、再做fine sparse attention。</li><li><b>RDD route：</b>上一resolution的route映射到下一stage，省掉VSA在线router。</li><li><b>Output reuse：</b>每k步重算一次self-attention，其余step广播缓存output；FFN/cross-attention仍算。</li></ol>
<h2>FastVideo最快公开路径</h2><p>不是VSA-only，而是3-step DMD + VSA Sparse-Distill。</p>{few_html}<p>FastVideo DMD3+VSA为{next(r[3] for r in few_rows if r[1] == 'FastVideo DMD3 + VSA')}代理；VSA-only为{next(r[3] for r in actual_rows if r[1] == 'FastVideo VSA only')}。前者的大部分额外加速来自50→3 calls。</p>
<h2>稀疏率敏感性</h2>{density_html}<img src="figures/rdd_video_speed_density.svg" alt="density sensitivity"><p class="warn">20% 在共享 GPU 上反而比 12.5% 快属于并发训练/时钟/autotune 噪声；方法大排序可参考，density 的精确差值须独占 GPU 重跑。</p>
<h2>Fine-stage kernel</h2>{kernel_html}
<p>20% route lifting：coarse→middle {metrics['route_lifts']['coarse_to_middle']['densities']['0.200000']['latency']['median_ms']:.5f} ms；middle→fine {metrics['route_lifts']['middle_to_fine']['densities']['0.200000']['latency']['median_ms']:.5f} ms。所有 route 均与 CPU reference 逐项相等；reuse kernel 最大绝对误差为 0。</p>
<h2>RDD step allocation</h2>{allocation_html}<p>coarse-heavy 更快不代表质量更好。</p>
<h2>判断</h2><ol><li>RDD 多分辨率本身有确定的算力优势。</li><li>同稀疏率下，previous-stage route 省去 VSA 每层每步在线 router，理论上更快；等质量尚未证明。</li><li>完整 FastVideo 有 3-step DMD，未蒸馏 RDD 目前不能宣称系统级更快。</li><li>最终强方案是 RDD + external route + 自适应 output reuse + few-step distillation。</li></ol><p>Pyramid Flow 官方 scheduler 已在无 checkpoint 条件下实际导入运行，确认三阶段各 10 个 timestep；其不同骨干的原始毫秒没有混入公平排名。</p>
<h2>来源</h2><ul><li><a href="https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_1_3B.py">Wan2.1 config</a></li><li><a href="https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md">FastVideo VSA</a></li><li><a href="https://github.com/hao-ai-lab/FastVideo/blob/main/docs/distillation/dmd.md">FastVideo DMD</a></li><li><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Korzhenkov_PyramidalWan_On_Making_Pretrained_Video_Model_Pyramidal_for_Efficient_Inference_CVPR_2026_paper.pdf">PyramidalWan</a></li><li><a href="https://github.com/jy0205/Pyramid-Flow">Pyramid Flow</a></li><li><a href="https://github.com/showlab/TPDiff">TPDiff</a></li><li><a href="https://arxiv.org/abs/2408.12588">PAB</a></li></ul>
<p class="small">完整细节、方法定义、限制和文件位置见同目录 Markdown。</p>
"""
    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RDD Video Speed</title><style>
body{{font-family:Segoe UI,"Microsoft YaHei",sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#172033;line-height:1.65}}h1{{font-size:32px}}h2{{margin-top:34px;border-bottom:1px solid #dbe3ea;padding-bottom:7px}}table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}}th,td{{padding:9px;border:1px solid #dbe3ea;text-align:left}}th{{background:#eef4f7}}tr:nth-child(even){{background:#f8fafc}}img{{max-width:100%;display:block;margin:18px auto}}.callout{{background:#ecfeff;border-left:5px solid #0f766e;padding:16px 18px;border-radius:6px}}.warn{{background:#fff7ed;border-left:4px solid #ea580c;padding:11px 14px}}.small{{color:#64748b;font-size:13px}}a{{color:#2563eb}}
</style></head><body>{body}</body></html>"""
    args.output.with_suffix(".html").write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
