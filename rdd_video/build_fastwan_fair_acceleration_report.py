#!/usr/bin/env python3
"""Build the beginner-friendly HTML/Markdown report for the fair sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_LABELS = {
    "final_fixed_vsa20_pre": "固定全分辨率 VSA-20（基线）",
    "final_temporal_vsa20": "Temporal RDD + 在线 VSA-20",
    "final_temporal_c2f20": "Temporal RDD + C2F-RouteOnly-20",
    "final_temporal_c2f20_reuse2": "Temporal RDD + C2F-RouteOnly-20 + StepReuse(2)",
    "fixed_vsa20": "固定全分辨率 VSA-20（基线）",
    "temporal_vsa20": "Temporal RDD + 在线 VSA-20",
    "temporal_c2f_only20": "Temporal RDD + C2F-RouteOnly-20",
    "temporal_c2f_only20_reuse2_projected": "Temporal RDD + C2F-20 + StepReuse(2)",
    "grid_temporal_4x2x1_c2f20": "Temporal RDD 4→2→1 + C2F-20",
    "spatial_c2f_only20": "Spatial RDD 2→2→1 + C2F-20",
    "grid_spatial_4x2x1_c2f20": "Spatial RDD 4→2→1 + C2F-20",
    "spatiotemporal_c2f_only20": "Spatiotemporal RDD 2→2→1 + C2F-20",
    "spatiotemporal_3stage_c2f_only20_triton": "Spatiotemporal RDD 4→2→1 + C2F-20",
    "grid_spatiotemporal_421_c2f20_recheck": "Spatiotemporal RDD 4→2→1 + C2F-20（严格复测）",
}

QUALITY_SUFFIX_LABELS = {
    "fixed_vsa20": "固定全分辨率 VSA-20（基线）",
    "temporal_vsa20": "Temporal RDD + 在线 VSA-20",
    "temporal_c2f20": "Temporal RDD + C2F-20",
    "temporal_c2f20_reuse2": "Temporal RDD + C2F-20 + StepReuse-Out(2)",
    "temporal421_c2f20": "Temporal RDD 4→2→1 + C2F-20",
    "spatial221_c2f20": "Spatial RDD 2→2→1 + C2F-20",
    "spatial421_c2f20": "Spatial RDD 4→2→1 + C2F-20",
    "spatiotemporal221_c2f20": "Spatiotemporal RDD 2→2→1 + C2F-20",
    "spatiotemporal421_c2f20": "Spatiotemporal RDD 4→2→1 + C2F-20",
}


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def method_label(name: str) -> str:
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    for scene in ("city", "coast"):
        prefix = f"quality_{scene}_"
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            if suffix in QUALITY_SUFFIX_LABELS:
                return QUALITY_SUFFIX_LABELS[suffix]
    return name


def fmt_factors(row: dict) -> str:
    return " → ".join(f"({a},{b})" for a, b in row["stage_factors"])


def fmt_shapes(row: dict) -> str:
    return " → ".join("×".join(map(str, shape)) for shape in row["stage_latent_shapes"])


def config_table(row: dict) -> str:
    timesteps = "/".join(map(str, row.get("dmd_timesteps") or []))
    mode = {
        "vsa": "在线 VSA（每层重新计算 coarse route）",
        "c2f_route_only": "C2F-RouteOnly（上一 stage route 引导下一 stage）",
    }.get(row.get("attention_mode"), str(row.get("attention_mode")))
    reuse = (
        "不复用（k=1）"
        if row.get("reuse_every") in (None, 1)
        else f"StepReuse-Out(k={row['reuse_every']}, scope={row.get('reuse_scope')})"
    )
    speed = row.get("denoise_speedup_vs_matched_fixed_vsa")
    if speed is None:
        speed = row.get("denoise_speedup_vs_fixed_vsa20")
    if speed is None:
        speed = row.get("denoise_speedup_vs_quality_fixed_vsa")
    if speed is None and "fixed_vsa" in row["name"]:
        speed = 1.0
    speed_text = "—（本视频只作画质检查）" if speed is None else f"{speed:.2f}×"
    return f"""<table class="rdd-config">
<tr><th>项目</th><th>该视频的实际配置</th></tr>
<tr><td>Stage factor</td><td><code>{fmt_factors(row)}</code></td></tr>
<tr><td>Latent 路径</td><td><code>{fmt_shapes(row)}</code></td></tr>
<tr><td>Sampler / boundary</td><td>DMD 3 NFE，t=<code>{timesteps}</code>；<code>{row.get('transition_mode')}</code></td></tr>
<tr><td>Attention</td><td>{mode}</td></tr>
<tr><td>稀疏参数</td><td>keep={pct(row.get('keep_nominal'))}；route={pct(row.get('route_density_requested'))}；实际 C2F={pct(row.get('effective_c2f_density'))}</td></tr>
<tr><td>Attention reuse</td><td>{reuse}</td></tr>
<tr><td>实测时间</td><td>denoise <strong>{row['denoise_seconds_median']:.3f}s</strong>；sample+VAE <strong>{row['sample_plus_vae_seconds']:.3f}s</strong>；相对邻接/配对 fixed-VSA <strong>{speed_text}</strong></td></tr>
</table>"""


def video_card(row: dict, media_prefix: str) -> str:
    name = row["name"]
    label = method_label(name)
    video = f"{media_prefix}/{row['video_file']}"
    poster = row.get("poster_file")
    poster_attr = f' poster="{media_prefix}/{poster}"' if poster else ""
    return f"""<article class="rdd-card">
<video controls preload="metadata" playsinline{poster_attr} src="{video}"></video>
<h4>{label}</h4>
<p><code>{name}</code></p>
{config_table(row)}
</article>"""


def group_summary_table(group_rows: list[dict]) -> list[str]:
    lines = [
        "| 视频方法 | Stage factor | Attention / keep | Reuse | Denoiser | Sample+VAE | Fixed-VSA 基线加速 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in group_rows:
        speed = row.get("denoise_speedup_vs_matched_fixed_vsa")
        if speed is None:
            speed = row.get("denoise_speedup_vs_fixed_vsa20")
        if speed is None:
            speed = row.get("denoise_speedup_vs_quality_fixed_vsa")
        if speed is None and "fixed_vsa" in row["name"]:
            speed = 1.0
        speed_text = "—" if speed is None else f"{speed:.2f}×"
        reuse = "无" if row.get("reuse_every") in (None, 1) else f"Out({row['reuse_every']})"
        lines.append(
            f"| {method_label(row['name'])} | `{fmt_factors(row)}` | "
            f"`{row.get('attention_mode')}`, {pct(row.get('keep_nominal'))} | {reuse} | "
            f"{row['denoise_seconds_median']:.3f}s | {row['sample_plus_vae_seconds']:.3f}s | {speed_text} |"
        )
    return lines


def find(rows: dict[str, dict], name: str) -> dict | None:
    return rows.get(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    summary = json.loads((root / "fair_acceleration_summary.json").read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in summary["all_rows"]}
    baseline = summary["primary_final_baseline"]
    primary = summary["primary_final"]
    requested_grids = summary.get("requested_stage_grids", [])
    primary_by_name = {row["name"]: row for row in primary}
    requested_by_name = {row["name"]: row for row in requested_grids}
    longwarm = summary.get("longwarm_stability")
    perturbation = summary.get("same_seed_pixel_perturbation")
    media_prefix = "../../eval_results/rdd_fastwan_fair_accel_20260907"

    lines = [
        "# RDD × FastWan：公平加速实验与视频报告",
        "",
        "> 结论先说：在同一个 FastWan2.1-1.3B checkpoint、同一个 DMD 3-step sampler、同一提示词/seed、832×448×29 帧、同一 VSA 20% keep 下，最保守的单卡前后夹测得到：Temporal RDD 为 **1.39× denoiser 加速**，加 C2F 为 **1.43×**，再加 StepReuse(2) 为 **1.58×**。把约 1.15 秒 VAE 解码也计入，最终组合是 **1.20× sample+VAE 加速**。这些是零微调 plumbing/性能结论，不是最终画质结论。",
        "",
        "<style>.rdd-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px}.rdd-card{border:1px solid #d0d7de;border-radius:10px;padding:12px;background:#fff}.rdd-card video{width:100%;border-radius:7px;background:#111}.rdd-card h4{margin:.7em 0 .25em}.rdd-card p{margin:.3em 0;font-size:.92em}.rdd-config{width:100%;border-collapse:collapse;margin-top:.7em;font-size:.84em}.rdd-config th,.rdd-config td{border:1px solid #d8dee4;padding:5px 7px;vertical-align:top}.rdd-config th:first-child,.rdd-config td:first-child{width:29%;white-space:nowrap}.rdd-config th{background:#f6f8fa;text-align:left}.good{color:#137333}.warn{color:#9a6700}</style>",
        "",
        "## 1. 到底比较了什么",
        "",
        "先把四个名字翻译成人话：",
        "",
        "1. **固定全分辨率 VSA-20（基线）**：三个去噪调用都在完整 video latent 上运行。每层 VSA 都把 token 池化成 3D tiles，重新计算一次 tile-to-tile 分数，选 20% tiles，再执行 block-sparse attention；它还保留一个 coarse 全局输出分支和可学习 gate。",
        "2. **Temporal RDD + VSA-20**：前两个去噪调用只使用一半时间分辨率，第三个恢复完整时间分辨率；空间分辨率不动，边界统一使用 `noise_filling`。每次 attention 仍完整执行 VSA，因此这里只测 RDD 自己省下的 token 计算。",
        "3. **Temporal RDD + C2F-RouteOnly-20**：最低时间分辨率第一次仍正常计算 VSA route；后续直接把前一 stage 的 3D block route 映射到更细 stage，并调用与 VSA 完全相同的 block-sparse attention executor。它不再计算 fine-stage 的 dense tile-score matrix，也不再额外计算 VSA coarse output。",
        "4. **再加 StepReuse-Out(2)**：前两个调用形状相同，第二次直接复用第一次经过 output projection 的 self-attention 输出。当前 timestep 的 residual gate、cross-attention 和 MLP 仍照常计算；不是把整个 Transformer block 跳过。",
        "",
        "这里的 `(spatial factor, temporal factor)` 中，`(1,2)` 表示空间不缩、时间 token 减半；`(1,1)` 表示完整 latent。",
        "",
        "## 2. 公平性约束",
        "",
        "所有主比较固定如下：",
        "",
        "| 项目 | 固定值 |",
        "|---|---|",
        "| 模型 | `FastWan2.1-T2V-1.3B-Diffusers`，同一权重 |",
        "| 输出 | 832×448，29 帧，16 fps（约 1.81 秒视频） |",
        "| batch | 1 条视频/次 |",
        "| sampler | DMD 3 NFE，t=`1000/757/522` |",
        "| guidance | 使用 FastWan 蒸馏 checkpoint 的原生单次条件 forward；没有额外 CFG 双分支 |",
        "| prompt / seed | 同一组内完全相同 |",
        "| 边界 | 只用 `noise_filling` |",
        "| attention executor | 都调用 FastVideo 的同一个 64-token block-sparse Triton kernel |",
        "| 稀疏率 | 主表统一 keep 20%；另做 50/25/20/12.5% 同密度配对 |",
        "| 计时 | RTX PRO 6000 Blackwell；warmup 后 20 次中位数；基线在队列前后各测一次 |",
        "| 不计入 | checkpoint/T5 加载与首次 JIT 编译 |",
        "",
        f"基线前后漂移仅 **{100*baseline['drift_fraction']:.2f}%**，说明主表计时稳定。",
        "",
        "这里 `Denoiser` 是三次 Transformer 去噪和 stage boundary 的总时间；`Sample+VAE` 再加上把 latent 解码成像素视频的 VAE 时间。`1.58×` 的含义是同一工作从 0.869 秒降到约 0.550 秒，也就是延迟减少约 36.7%，不是减少 58%。模型加载、T5 文本编码和第一次 Triton/JIT 编译不在 steady-state 主表内；这是生成系统论文通常使用的编译后吞吐/延迟口径，冷启动应另表报告。",
        "",
        "## 3. 最可信的主结果",
        "",
        "| 方法 | Denoiser | Denoiser 加速 | Sample+VAE | Sample+VAE 加速 | 关键工作量 |",
        "|---|---:|---:|---:|---:|---|",
        f"| 固定全分辨率 VSA-20 | {baseline['denoise']:.3f}s | 1.00× | {baseline['sample_plus_vae']:.3f}s | 1.00× | 90 次在线 VSA layer-call |",
    ]
    workload = {
        "final_temporal_vsa20": "90 次在线 VSA layer-call",
        "final_temporal_c2f20": "30 次 seed VSA + 60 次 C2F sparse",
        "final_temporal_c2f20_reuse2": "30 次 seed VSA + 30 次 C2F sparse + 30 次 projected cache hit",
    }
    for row in primary:
        lines.append(
            f"| {METHOD_LABELS[row['name']]} | {row['denoise_seconds_median']:.3f}s | "
            f"**{row['denoise_speedup_vs_fixed_vsa20']:.2f}×** | "
            f"{row['sample_plus_vae_seconds']:.3f}s | "
            f"**{row['sample_plus_vae_speedup_vs_fixed_vsa20']:.2f}×** | {workload[row['name']]} |"
        )
    temporal_vsa, temporal_c2f, temporal_reuse = primary
    lines += [
        "",
        "这张表还能拆出三个增量贡献：",
        "",
        f"- **RDD 本身**：{baseline['denoise']:.3f}s → {temporal_vsa['denoise_seconds_median']:.3f}s，延迟减少 {100*(1-temporal_vsa['denoise_seconds_median']/baseline['denoise']):.1f}%。这是最大的一项，因为前两个 NFE 的时间 token 直接减半。",
        f"- **C2F-Route 替换在线 router**：{temporal_vsa['denoise_seconds_median']:.3f}s → {temporal_c2f['denoise_seconds_median']:.3f}s，在已经使用 RDD 后再减少 {100*(1-temporal_c2f['denoise_seconds_median']/temporal_vsa['denoise_seconds_median']):.1f}%。它省的是 route 构造和 coarse branch，不是整个 Transformer。",
        f"- **StepReuse(2)**：{temporal_c2f['denoise_seconds_median']:.3f}s → {temporal_reuse['denoise_seconds_median']:.3f}s，再减少 {100*(1-temporal_reuse['denoise_seconds_median']/temporal_c2f['denoise_seconds_median']):.1f}%。它只命中第二个低时间分辨率 NFE；最后一次 full-resolution NFE 仍真实计算。",
        "",
        "一个代表性轨迹的逐 NFE 时间更直观：固定 VSA 三步约为 `0.290/0.290/0.291s`；Temporal RDD+VSA 为 `0.169/0.168/0.290s`；C2F 为 `0.174/0.155/0.281s`；最终组合为 `0.177/0.089/0.281s`。所以加速 pattern 与设计一致：RDD 缩短前两步，Reuse 主要把第二步再砍掉一截，最后完整分辨率步没有被伪装成低分辨率计算。逐 NFE 数字只用于解释机制，正式加速仍使用多次完整轨迹中位数。",
        "",
        "为什么 denoiser 1.58×，端到端只有 1.20×？因为这是只有 3 NFE 的蒸馏模型，VAE 解码约 1.15 秒且四种方法都没有加速它。对于 20–50 NFE 的普通 video diffusion/flow 模型，denoiser 占比更高，RDD 的端到端收益理论上会更接近 denoiser 加速，但必须实际测，不能直接外推成论文数字。",
        "",
    ]
    if longwarm:
        lb = longwarm["fixed_vsa_baseline"]
        lm = longwarm["method"]
        lines += [
            "### 更长预热/重复是否推翻主表",
            "",
            "没有。另一次 **20 次预热 + 50 次正式重复** 的单卡夹测中：",
            "",
            "| 方法 | Denoiser | 加速 | Sample+VAE | 加速 |",
            "|---|---:|---:|---:|---:|",
            f"| 固定 VSA-20（前后平均） | {lb['denoise']:.3f}s | 1.00× | {lb['sample_plus_vae']:.3f}s | 1.00× |",
            f"| Temporal RDD+C2F+Reuse(2) | {lm['denoise_seconds_median']:.3f}s | **{lm['denoise_speedup_vs_fixed_vsa20']:.2f}×** | {lm['sample_plus_vae_seconds']:.3f}s | **{lm['sample_plus_vae_speedup_vs_fixed_vsa20']:.2f}×** |",
            "",
            f"长跑基线前后漂移为 {100*lb['drift_fraction']:.2f}%。组合方法单次记录仍受 GPU 动态频率影响，因此正文采用较保守的 20-run **1.58×**，把 50-run **1.63×** 当稳定性复核，而不是挑最快一次。",
            "",
        ]
    lines += [
        "### 为什么 `C2F-RouteOnly` 单独只快一点",
        "",
        "直觉里‘少算一次 coarse attention’听起来很大，但 VSA 的 coarse route 并不是对所有 fine tokens 再做一次完整 attention。它先把每个 3D tile 压成一个 centroid，只在很小的 block grid 上算分数。当前真实 transition 的 block grid 只有 `[1,7,13] → [2,7,13]`。实测在线 VSA route builder 为约 **0.209 ms/层**，C2F route lift 为 **0.031 ms/层**：route 本身虽然快了 6.66×，每层实际只省约 **0.178 ms**。即使 60 个 fine-stage layer-call 都命中，纯 route 理论节省也只有约 **10.7 ms/视频**。",
        "",
        "C2F 仍必须计算当前 fine hidden state 的 Q/K/V 投影、20% block-sparse QK/softmax/AV、输出投影、cross-attention、MLP 和 residual；VAE 也完全没变。第一次 coarse stage 还必须正常跑 VSA 才能产生 route。因此主实验里 `RDD+VSA` 到 `RDD+C2F` 只有小幅增益是符合算子分解的，不是 C2F kernel 没有工作。",
        "",
        "更激进的路径确实让 **RDD 总体**加速更明显，因为前两次 denoiser 的 token 数更少；它们不保证让 **C2F 相对在线 VSA 的增量**更大。分辨率越低，在线 VSA 的 centroid grid 也越小，省掉 router 的绝对毫秒数反而可能更少。C2F 真正有价值的场景是长视频、高分辨率或更多 NFE，此时 block grid、层调用次数和 route 重用次数都更大。",
        "",
    ]
    lines += [
        "## 4. 同密度 VSA 对照",
        "",
        "每个密度在同一 GPU 上按 `VSA前测 → RDD+VSA → RDD+C2F → RDD+C2F+Reuse → VSA后测` 顺序运行。这样不会把更稀疏的 C2F 拿去对比更密的 VSA。",
        "",
        "| Keep | 固定 VSA 基线 | RDD+VSA | 加速 | RDD+C2F | 加速 | RDD+C2F+Reuse | 加速 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["density_sweep"]:
        vsa, c2f, reuse = group["methods"]
        lines.append(
            f"| {100*group['density']:.1f}% | {group['fixed_vsa_baseline']['denoise']:.3f}s | "
            f"{vsa['denoise_seconds_median']:.3f}s | {vsa['denoise_speedup_vs_matched_fixed_vsa']:.2f}× | "
            f"{c2f['denoise_seconds_median']:.3f}s | {c2f['denoise_speedup_vs_matched_fixed_vsa']:.2f}× | "
            f"{reuse['denoise_seconds_median']:.3f}s | **{reuse['denoise_speedup_vs_matched_fixed_vsa']:.2f}×** |"
        )
    lines += [
        "",
        "读表重点：降低 keep 会让 VSA 和 C2F 的绝对时间都下降，但整个 block 还有 QKV projection、cross-attention、MLP 和 VAE，所以收益不会与被删 attention pair 的比例线性增长。50% keep 的相对加速最大，是因为同密度固定 VSA 更慢；主推荐仍是 checkpoint 原生训练的 20% keep。",
        "",
        "## 5. Route kernel 是否真的省计算",
        "",
    ]
    route = summary["route_comparison_benchmark"]
    lines += [
        f"在真实 temporal transition 的 block grid `{route['shape']['coarse_grid']} → {route['shape']['fine_grid']}` 上，12 heads、head dim 128、keep 20%：",
        "",
        f"- 在线 VSA route builder：**{route['online_vsa_route']['median_ms']:.3f} ms/layer**；它包含 Q/K block pooling、$Q_cK_c^T$、Top-K 和压缩索引。",
        f"- C2F Triton route lift：**{route['c2f_lift']['median_ms']:.3f} ms/layer**。",
        f"- 单看 route 构造，C2F 是 **{route['route_builder_speedup']:.2f}×**；30 层一次 transition 约 {30*route['c2f_lift']['median_ms']:.2f} ms。",
        "",
        "因此当前瓶颈已不是 route lift。真正剩余的大头是 QKV/输出投影、cross-attention、MLP 和 VAE；继续只抠 route kernel 不可能再获得数量级加速。",
        "",
        "## 6. 同 seed 像素差异怎么读",
        "",
        "SSIM/PSNR 在这里仅回答“改方法后，同一个随机种子的视频像素离基线多远”，**不回答视频本身是否好看**。生成模型允许合理构图改变，因此最终仍要看视频，并在微调后做 VBench/FVD/人工偏好。",
        "",
    ]
    if perturbation:
        lines += [
            "| 场景 | RDD+VSA SSIM | RDD+C2F SSIM | RDD+C2F+Reuse SSIM |",
            "|---|---:|---:|---:|",
        ]
        for key, label in (("primary_fox", "狐狸"), ("quality_city", "城市行人"), ("quality_coast", "海岸汽车")):
            group = perturbation[key]
            values = list(group.values())
            lines.append(
                f"| {label} | {values[0]['ssim_all']:.3f} | {values[1]['ssim_all']:.3f} | {values[2]['ssim_all']:.3f} |"
            )
        lines += [
            "",
            "三个场景里组合方法反而始终最接近固定 VSA 基线；这不是质量证明，但至少没有出现“加 reuse 后数值扰动必然更大”的信号。",
            "",
        ]
    lines += [
        "## 7. 狐狸场景：主方法与五条分辨率路径放在一起",
        "",
        "下面所有视频使用同一个狐狸 prompt 和 seed。前四项回答逐层加入 RDD、C2F 和 Reuse 的影响；随后五项回答不同 temporal/spatial/spatiotemporal stage path 的速度与破坏模式。factor 为 `(空间缩小倍数, 时间缩小倍数)`。",
        "",
    ]
    fox_names = [
        "final_fixed_vsa20_pre",
        "final_temporal_vsa20",
        "final_temporal_c2f20",
        "final_temporal_c2f20_reuse2",
        "grid_temporal_4x2x1_c2f20",
        "spatial_c2f_only20",
        "grid_spatial_4x2x1_c2f20",
        "spatiotemporal_c2f_only20",
        (
            "grid_spatiotemporal_421_c2f20_recheck"
            if "grid_spatiotemporal_421_c2f20_recheck" in rows
            else "spatiotemporal_3stage_c2f_only20_triton"
        ),
    ]
    fox_rows = []
    for name in fox_names:
        if name in requested_by_name:
            fox_rows.append(requested_by_name[name])
        elif name in primary_by_name:
            fox_rows.append(primary_by_name[name])
        elif name in rows:
            fox_rows.append(rows[name])
    lines += group_summary_table(fox_rows)
    lines += [
        "",
        "> 原始秒数来自不同负载时段，不能跨行硬比；带配对加速的行使用同卡前后夹测固定 VSA，才是可信的速度比较。",
        "",
    ]
    st421 = requested_by_name.get("grid_spatiotemporal_421_c2f20_recheck")
    if st421:
        matched = st421["matched_fixed_vsa_baseline"]
        lines += [
            "### 为什么原来 `(4,4)→(2,2)→(1,1)` 看起来反而慢",
            "",
            f"原来的 1.63× 来自一次没有紧邻配对基线的 10-run 历史 sweep，单次 denoise 在约 0.472–0.640 秒之间波动；它不能和后来在共享 GPU 上做前后夹测的 1.66×/1.96×直接横比。严格复测使用同一 GPU、5 次 warmup、20 次计时，并按 `固定 VSA 前测 → (4,4) 方法 → 固定 VSA 后测` 运行：前后基线分别约 1.871/1.869 秒，方法为 **{st421['denoise_seconds_median']:.3f} 秒**，所以真实配对 denoiser 加速是 **{st421['denoise_speedup_vs_matched_fixed_vsa']:.2f}×**；计入 VAE 后是 **{st421['sample_plus_vae_speedup_vs_matched_fixed_vsa']:.2f}×**。异常已经消失。",
            "",
            "从计算量也应如此：若只按 stage token 数估算，temporal 4→2→1、spatial 4→2→1、spatiotemporal 4→2→1 三条路径分别约为 full 三步的 `1.75/3`、`1.3125/3`、`1.140625/3`。因此最后一条不应在相同实现和负载下更慢。它也不会按理论 FLOPs 成比例无限加速，因为第三次 full stage 完全保留，而且极小 latent 的 GPU kernel 利用率差；requested keep=20% 还因 block 整数取整变成了约 "
            f"**{pct(st421.get('effective_c2f_density'))} 实际 pair density**。",
            "",
            "2.03× 与纯 spatial 的 1.96× 来自不同 GPU/不同时段，差值不能当成精确排名；能下的结论是 `(4,4)` 并没有反常变慢，旧行是计时口径混用造成的。",
            "",
        ]
    lines += [
        '<div class="rdd-grid">',
    ]
    for item in fox_rows:
        lines.append(video_card(item, media_prefix))
    lines += [
        "</div>",
        "",
        "人工检查：温和 temporal 路径仍保持单只狐狸、雪地和树林，主体跨帧连续；更激进 temporal 4→2→1 开始卡通化并出现彩色贴片。纯 spatial 与 spatiotemporal 路径出现明显色块、拉伸或结构崩坏。它们目前用于定位速度上限和训练需求，不应被表述成零微调同质量结果。",
        "",
        "## 8. 城市与海岸：同样把五条路径放回各自 prompt",
        "",
        "每个场景先保留原来的四个方法视频，再紧接着放五条新 stage path。原来的四方法画质矩阵分别跑在不同 GPU 上，所以表中只列原始时间、不伪造横向加速；新五条在每个场景内同 GPU 顺序运行，并在队尾补了同负载 fixed-VSA baseline，因此它们有邻接基线加速。正式论文速度仍以前面的 20-run 夹测为准。",
        "",
    ]
    quality_groups = {
        "城市行人": [
            "quality_city_fixed_vsa20",
            "quality_city_temporal_vsa20",
            "quality_city_temporal_c2f20",
            "quality_city_temporal_c2f20_reuse2",
            "quality_city_temporal421_c2f20",
            "quality_city_spatial221_c2f20",
            "quality_city_spatial421_c2f20",
            "quality_city_spatiotemporal221_c2f20",
            "quality_city_spatiotemporal421_c2f20",
        ],
        "海岸汽车": [
            "quality_coast_fixed_vsa20",
            "quality_coast_temporal_vsa20",
            "quality_coast_temporal_c2f20",
            "quality_coast_temporal_c2f20_reuse2",
            "quality_coast_temporal421_c2f20",
            "quality_coast_spatial221_c2f20",
            "quality_coast_spatial421_c2f20",
            "quality_coast_spatiotemporal221_c2f20",
            "quality_coast_spatiotemporal421_c2f20",
        ],
    }
    quality_observations = {
        "城市行人": "原来的四个温和方法都维持人物、黄色雨衣、霓虹街道和湿地反射。新补的 temporal 4→2→1 代表帧仍非常完整；spatial/spatiotemporal 2→2→1 还能辨认人物，但出现明显霓虹拖影和局部形变；两条 4→2→1 空间路径则严重块化、主体基本崩坏。",
        "海岸汽车": "原来的四个温和方法都维持蓝色老爷车、道路和夕阳。新补的 temporal 4→2→1 仍保留汽车、海岸和道路，仅有彩色贴片；spatial/spatiotemporal 2→2→1 保住汽车主体但背景和反光更扭曲；两条 4→2→1 空间路径严重色块化。这个跨 prompt pattern 与狐狸一致：只缩时间远比零微调直接缩空间稳。",
    }
    for title, names in quality_groups.items():
        scene_rows = [rows[name] for name in names if name in rows]
        lines += [f"### {title}", ""]
        lines += group_summary_table(scene_rows)
        lines += ["", '<div class="rdd-grid">']
        for item in scene_rows:
            lines.append(video_card(item, media_prefix))
        lines += ["</div>", "", quality_observations[title], ""]

    lines += [
        "## 9. 稀疏率从 mild 到 aggressive",
        "",
        "下面只看最终组合 `Temporal RDD + C2F + Reuse`，依次把 keep 从 50% 降到 12.5%。这些 checkpoint 原生是 VSA-20 训练的，因此 50/25/12.5% 都属于零微调扰动测试。",
        "",
        '<div class="rdd-grid">',
    ]
    for tag in ("50", "25", "20", "12p5"):
        name = f"pair_d{tag}_temporal_c2f_reuse2"
        if name in rows:
            lines.append(video_card(rows[name], media_prefix))
    lines += [
        "</div>",
        "",
        "## 10. 已验证的实现优化",
        "",
        "- **Projected StepReuse**：旧实现只缓存 sparse backend 的 $AV$，cache hit 仍会执行 norm/QKV/gate/RoPE/tile/output projection。新实现缓存 `to_out` 后的 self-attention tensor，hit 时把这些全部跳过，但保留当前 residual/cross-attention/MLP。与旧 backend-cache 的 latent `max_abs=0`。",
        "- **单次 seed VSA**：旧 C2F seed 先算一遍 centroids/scores 保存 route，再调用原 VSA 重算同样的 centroids/scores/sparse/coarse。新路径只算一次并同时产出当前 VSA output 与下一 stage route。优化前后 temporal latent `max_abs=0`。",
        "- **Triton C2F route lift**：一条 program 负责一个 `(batch, head, fine query block)`，直接把 coarse key block 展开为 fine children，输出 VSA sparse executor 所需的 compact indices/counts；不落 CPU、不构造 fine $Q_cK_c^T$。",
        "- **复用官方 sparse executor**：C2F 并没有换一个更有利的 attention kernel。它与 VSA 一样调用 FastVideo `block_sparse_attn_from_indices`；差别只在 route 从哪里来。",
        "",
        "## 11. 失败模式与边界",
        "",
        "- 纯 spatial RDD 的零微调视频出现明显彩色块/结构拉伸；spatiotemporal 更激进时也有伪影。它们能更快，但当前不属于可用 Pareto 点。",
        "- `C2F-RouteOnly` 移除了 checkpoint 训练时存在的 VSA coarse branch。现在能生成合理 temporal 视频是可行性证据，不代表无需微调。最终论文必须给 VSA baseline 与 RDD+C2F 相同数据、相同更新步数/LoRA rank/硬件预算的微调对照。",
        "- 当前只有 3 prompts/3 seeds 的可视 sanity check，没有足够样本支持 VBench/FVD 结论。正式实验应至少报告 VBench 16 维、FVD/CLIP 或人工偏好以及多个 seed 的置信区间。",
        "- VSA 论文报告其方法可将 Wan-2.1 attention 加速约 6×、端到端 31s 降到 18s；模型、NFE、硬件和计时范围都与本报告不同，不能直接横向除法。",
        "",
        "## 12. 目前可以和不可以写进论文的话",
        "",
        "**可以写：**在完全相同 FastWan/VSA sparse executor 和 20% keep 下，RDD 降低早期 token 数；C2F 用上一 stage route 省掉后续在线 router；StepReuse 在相同低分辨率连续步跳过一次 self-attention 分支。单卡前后夹测显示组合 denoiser 1.58×，且零微调 temporal 输出仍可辨认、结构完整。",
        "",
        "**暂时不能写：**“RDD+C2F 在同等画质下击败 VSA”。当前 checkpoint 没有接受 RDD/C2F 微调，样本数量也远不足以证明 comparable quality。",
        "",
        "## 13. 复现入口和原始数据",
        "",
        "- `rdd_video/run_fastwan_fair_worker.sh`：单实验唯一入口。",
        "- `rdd_video/launch_fastwan_fair_phase4_density.sh`：四个密度的同卡配对 sweep。",
        "- `rdd_video/launch_fastwan_fair_phase6_final_timing.sh`：最终 20-run 前后夹测。",
        "- `rdd_video/launch_fastwan_fair_phase7_quality.sh`：两提示词×四方法视频。",
        "- `rdd_video/launch_fastwan_fair_phase8_longwarm.sh`：20 warmup / 50-run 稳定性复核。",
        "- `rdd_video/attention_analysis/rdd_c2f_route_triton.py`：我们的 C2F route kernel。",
        "- `eval_results/rdd_fastwan_fair_accel_20260907/fair_acceleration_summary.json`：结构化汇总。",
        "- `eval_results/rdd_fastwan_fair_accel_20260907/fair_acceleration_all_metrics.csv`：所有逐配置指标。",
        "",
        "## 14. RDD-256 与其他多分辨率生成方法",
        "",
        "先给结论：RDD 的 **400K 早期收敛非常强**，但当前还不能说 ImageNet-256 绝对 SOTA。最接近 RDD 的渐进扩散方法里，Relay Diffusion 的标准 FID 为 1.99（class-balanced sampling 为 1.87）；扩大到多分辨率架构、scale-wise AR 和像素级联 flow，DiMR-G、FlowAR-H、PixelFlow 分别达到 1.63、1.65、1.98。RDD 当前最好 2.617，还需要约 24%–38% 的相对 FID 降幅才能超过这些最强点。",
        "",
        "### 14.1 你自己的严格 DiT 对照",
        "",
        "| 模型 | 训练 | 采样 | CFG | FID-50K | 相对同 step DiT |",
        "|---|---:|---|---:|---:|---:|",
        "| 本地 DiT-XL/2 | 400K | DDPM 250 NFE | 1.0 | 19.781 | 1.00× |",
        "| RDD Unified XL/2 | 400K | DDPM 250 NFE，90/80/80，noise filling | 1.0 | **7.062** | **低 64.3%** |",
        "| 本地 DiT-XL/2 | 500K | DDPM 250 NFE | 1.0 | 17.464 | 1.00× |",
        "| RDD Unified XL/2 | 500K | DDPM 250 NFE，90/80/80，noise filling | 1.0 | **6.461** | **低 63.0%** |",
        "| RDD Unified XL/2（当前最佳 FID） | 400K | DPM-Solver-2，250 scheduled / **494 实际 NFE** | 1.5 | **2.617** | 非同 sampler/CFG，不能填入上面严格列 |",
        "",
        "这组数据最有价值的不是 2.617 单点，而是同代码体系、同 ImageNet、同 DiT-XL/2 规模、同 DDPM250、同 CFG1.0 下的 400K/500K 曲线。即使按实测训练 step 慢 1.258× 折算，RDD 400K 的 wall-clock 约等于 DiT 503K；此时 7.065 对 17.464，仍低约 **59.5%**。不过这还不是完全 bit-for-bit 架构控制：RDD 使用可变分辨率/RoPE 与三个尺度 loss，而原始 DiT 使用固定位置编码。论文应把它写成“强 controlled convergence evidence”，不要写成已经证明任意多分辨率方法都更优。",
        "",
        "### 14.2 ImageNet-256 已发表最好数据（只作清晰参考，不冒充同协议榜单）",
        "",
        "| 方法 | 多分辨率方式 | 模型/规模 | 训练与采样配置 | 最好 FID-50K | 与 RDD 的可比性 |",
        "|---|---|---|---|---:|---|",
        "| **RDD Unified XL/2（本项目）** | 同一 DiT 沿时间轴 8→16→32 latent，residual drift | 674.8M | 400K，batch 256；DPM2，494 NFE，CFG1.5 | **2.617** | 本项目真实结果；采样 NFE 较高 |",
        "| [f-DM](https://openreview.net/forum?id=iBdwKIsg4m) | progressive signal transform；DS/VQVAE stages | lighter ADM U-Net | ImageNet 2.5M iters；250 steps；CFG scale 2 | DS 8.2；**VQVAE 6.8** | 像素/U-Net、transform 和 guidance 都不同 |",
        "| [MDM / Matryoshka Diffusion](https://arxiv.org/abs/2310.15111) | NestedUNet 同时联合多分辨率 denoising | NestedUNet | 64² 预训练后 progressive 256²；CFG1.5 | **3.51**（CFG1.0 为 8.18） | 架构和训练阶段不同 |",
        "| [Relay Diffusion](https://github.com/zai-org/RelayDiffusion) | 64→256 两模型；blurring diffusion + block noise | 两个 ADM/EDM-style 模型 | 1.25B training samples；CFG | **1.99**；class-balanced **1.87** | 绝对强，但两模型且训练量远大 |",
        "| [DiMR-XL/2R](https://github.com/qihao067/DiMR) | 网络内部多分辨率分支和 multi-scale loss | 505M | 800 epochs；250 sampling steps | **1.70** | 改架构而非分辨率随时间增长 |",
        "| [DiMR-G/2R](https://github.com/qihao067/DiMR) | 同上 | 1.06B | 800 epochs；250 sampling steps | **1.63** | 参数比 RDD 大约 57% |",
        "| [FlowAR-L](https://github.com/OliverRensu/FlowAR) | 逐尺度自回归 + 每尺度 Flow Matching | 589M | 400 epochs，global batch 2048；25 FM steps/scale setting | **1.90** | 不是单条 diffusion/flow 时间链 |",
        "| [FlowAR-H](https://github.com/OliverRensu/FlowAR) | 同上 | 1.9B | 400 epochs，global batch 2048；50-step eval | **1.65** | 参数约 RDD 2.8× |",
        "| [PixelFlow](https://github.com/ShoufaChen/PixelFlow) | pixel-space cascaded flow | 677M | pixel space；公开评估表列 320 steps | **1.98** | 无 VAE，目标与采样协议不同 |",
        "| [LapFlow](https://openreview.net/forum?id=kdrc4o6okz) | Laplacian residual + multi-scale MoT | DiT-XL/2 或 B/2 | XL/2 600K 无 CFG 为 14.38；B/2 7M + CFG1.5 | **4.12**（长训最佳） | 600K 对照最有参考性；最好值来自不同小模型长训 |",
        "| [Scale Space Diffusion](https://openaccess.thecvf.com/content/CVPR2026/papers/Mukhopadhyay_Scale_Space_Diffusion_CVPR_2026_paper.pdf) | generalized linear degradation + Flexi-UNet | pixel-space Flexi-UNet | 只报告 ImageNet-64：FID 13.08 | — | **没有 ImageNet-256 数字**，不能排进榜单 |",
        "| [DFM](https://openreview.net/forum?id=3isHlkiykj) | decomposition levels 上独立 Flow Matching | DiT-XL | ImageNet-512，500K，CFG1.25/1.5 | —（512² FID 4.28） | **没有 256²结果**，只证明路线可扩展 |",
        "| [SPEED](https://arxiv.org/abs/2605.18736) | 预训练模型 training-free spectral resolution growth | FLUX/PixelGen/Wan 等 | 报速度-感知质量，不报 ImageNet-256 FID | — | 是加速 prior，不是本表质量对手 |",
        "",
        "表里的 FID 都叫 FID-50K，但 VAE、网络、参数量、训练样本量、CFG、采样器和 NFE 并不统一。尤其 RDD 的 2.617 用 494 次真实模型调用；不能仅凭数值说它已经优于使用 250 或更少步的系统。反过来，Relay 的 1.87 是 class-balanced sampling，标准随机 class protocol 是 1.99，也应该同时报告。",
        "",
        "### 14.3 RDD 到底有多大潜力",
        "",
        "1. **很有希望超过 f-DM、MDM 和 LapFlow 的现有 256 最好点**：按当前 2.617，绝对数值已经更低；RDD 400K CFG1.0 的 7.062 也已接近 f-DM-VQVAE 的 6.8，并优于 f-DM-DS 的 8.2、MDM CFG1.0 的 8.18 和 LapFlow-XL/2 600K 的 14.38。但协议不同，论文仍应实现或复现一个同 DiT 的 prior baseline。",
        "2. **有现实机会追到 Relay/PixelFlow**：从 2.617 到 1.99/1.98 需要再降约 24%。RDD 只训练 400K，而 Relay 报告 1.25B training samples；RDD 的 400K×batch256 只有 102.4M image presentations。这个训练余量是真实的，但 RDD 每 step 有多尺度额外计算，不能把 12.2× presentations 差距直接叫 12.2× compute。",
        "3. **打败 DiMR-G/FlowAR-H 不是‘继续训一点’就能保证**：需要从 2.617 再降约 37%–38%，对 FID≈2 的区域并不小；两者分别使用 1.06B/1.9B 参数和很长训练。最诚实的判断是“潜力中高，尚未被实验证实”。",
        "4. **长训练可能收窄 RDD 对 DiT 的优势**：官方 DiT-XL/2 最终能到 2.27，而本地 400K 仍是 19.78。RDD 已证明显著加快早期收敛，但还没有证明最终 asymptotic quality 更高；必须至少补 1M 左右 checkpoint、3 seeds 和同 sampler/CFG 的 DiT/RDD 对照。",
        "5. **最强论文故事不是宣称 ImageNet SOTA**：应主张‘在受控 DiT-XL/2 setting 下，大幅改善前 400K–500K 的收敛；同一多分辨率轨迹还天然提供 video token/attention 加速接口’。这比把不同 VAE、不同 NFE 的 2.617 与 1.63 硬拼更可信。",
        "",
        "### 14.4 下一步达到可发表比较的最低实验",
        "",
        "- 在 400K/500K 对 RDD 与 DiT 同时跑 **DDPM250 CFG1.0、DDPM250 CFG1.5、相同真实 NFE 的 DPM2**，补齐同协议表。",
        "- 把 Unified XL 继续到至少 1M，并在 400K 与最终点做 3 seeds；否则无法判断早期加速还是最终质量提升。",
        "- 报 training wall-clock / FLOPs 到目标 FID（例如达到 10、7、5、3 所需成本），这才是 RDD 相对 Relay/DiMR/FlowAR 最有机会赢的轴。",
        "- 复现一个**同一 DiT backbone** 的 progressive prior（建议 LapFlow-style/Pyramidal-style control 或 drift-free multi-resolution DiT），避免审稿人把收益归因于多尺度监督本身。",
        "",
        "## 15. 参考",
        "",
        "- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)",
        "- [FastVideo VSA 文档](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md)",
        "- [FastVideo kernel inventory](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/README.md)",
        "- [DiT 官方实现与 ImageNet-256 400K/最终结果](https://github.com/facebookresearch/DiT)",
        "- [f-DM, ICLR 2023](https://openreview.net/forum?id=iBdwKIsg4m)",
        "- [MDM, ICLR 2024](https://arxiv.org/abs/2310.15111)",
        "- [Relay Diffusion, ICLR 2024 Spotlight](https://github.com/zai-org/RelayDiffusion)",
        "- [DiMR, NeurIPS 2024](https://papers.nips.cc/paper/2024/hash/f1aa53cac69ef6980ca4a911ffcf278b-Abstract-Conference.html)",
        "- [FlowAR, ICML 2025](https://github.com/OliverRensu/FlowAR)",
        "- [PixelFlow](https://github.com/ShoufaChen/PixelFlow)",
        "- [LapFlow, ICLR 2026](https://openreview.net/forum?id=kdrc4o6okz)",
        "- [Scale Space Diffusion, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Mukhopadhyay_Scale_Space_Diffusion_CVPR_2026_paper.pdf)",
        "- [DFM, NeurIPS 2025 Spotlight](https://openreview.net/forum?id=3isHlkiykj)",
        "- [SPEED, 2026](https://arxiv.org/abs/2605.18736)",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
