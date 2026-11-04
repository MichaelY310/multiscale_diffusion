"""Build the concise Chinese report for the RDD/Wan attention method matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FASTWAN_ORDER = [
    "fixed_vsa20",
    "fixed_vsa20_stepreuse2",
    "fixed_vsa20_stepreuse3",
    "rdd_spatiotemporal_vsa100",
    "rdd_spatiotemporal_vsa50",
    "rdd_spatiotemporal_vsa20",
    "rdd_spatial_only_vsa20",
    "rdd_temporal_only_vsa20",
]

ORIGINAL_ORDER = [
    "fixed_fullres_dense",
    "fixed_fullres_stepreuse2",
    "rdd_spatiotemporal_dense",
    "rdd_spatiotemporal_stepreuse2",
    "rdd_spatiotemporal_stepreuse3",
    "rdd_spatiotemporal_c2f75",
    "rdd_spatiotemporal_c2f75_stepreuse2",
    "rdd_spatiotemporal_c2f50",
    "rdd_spatiotemporal_c2f50_stepreuse2",
    "rdd_spatiotemporal_c2f25",
    "rdd_spatial_only_dense",
    "rdd_spatial_only_c2f50",
    "rdd_temporal_only_dense",
    "rdd_temporal_only_c2f50",
]

FASTWAN_WARM_ORDER = [
    "fixed_vsa20_hot",
    "fixed_vsa20_reuse2_hot",
    "rdd_temporal_2stage_hot",
    "rdd_temporal_2stage_reuse2_hot",
    "rdd_spatial_2stage_hot",
    "rdd_spatial_2stage_reuse2_hot",
    "rdd_spatiotemporal_2stage_hot",
    "rdd_spatiotemporal_2stage_reuse2_hot",
    "rdd_spatiotemporal_3stage_hot",
]


def load_metrics(directory: Path, names: list[str]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for name in names:
        path = directory / f"{name}.metrics.json"
        if not path.exists():
            rows.append((name, {}))
            continue
        rows.append((name, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def factors(data: dict[str, Any]) -> str:
    values = data.get("stage_factors_spatial_temporal")
    if values is None:
        values = [
            [item["spatial_factor"], item["temporal_factor"]]
            for item in data.get("stage_configs", [])
        ]
    return "→".join(f"({a},{b})" for a, b in values) if values else "—"


def trajectory_label(data: dict[str, Any]) -> str:
    text = factors(data)
    if text in {"(1,1)", "(1,1)→(1,1)→(1,1)"}:
        return "固定全分辨率"
    if text == "(4,4)→(2,2)→(1,1)":
        return "时空 RDD"
    if text == "(4,1)→(2,1)→(1,1)":
        return "仅空间 RDD"
    if text == "(1,4)→(1,2)→(1,1)":
        return "仅时间 RDD"
    if text == "(1,2)→(1,2)→(1,1)":
        return "仅时间两阶段 RDD"
    if text == "(2,1)→(2,1)→(1,1)":
        return "仅空间两阶段 RDD"
    if text == "(2,2)→(2,2)→(1,1)":
        return "时空两阶段 RDD"
    return text


def fastwan_method(data: dict[str, Any]) -> tuple[str, str, str]:
    attention = data.get("attention", {})
    keep = 100.0 * float(data.get("attention_pair_keep_fraction_nominal", 1.0))
    reuse = int(attention.get("reuse_every", 1))
    method = f"VSA({keep:g}%)"
    if reuse > 1:
        method += f" + StepReuse-Out({reuse})"
    hits = str(attention.get("step_reuse_hits", 0))
    return method, f"{keep:g}%", hits


def original_method(data: dict[str, Any]) -> tuple[str, str, str]:
    attention = data.get("attention", {})
    mode = attention.get("mode", "dense")
    if mode == "dense":
        return "Dense", "100%", "0"
    if mode == "reuse":
        k = int(attention.get("reuse_every", 1))
        return f"Dense + StepReuse-Out({k})", "100%", str(attention.get("reused_attention_calls", 0))
    density = 100.0 * float(attention.get("requested_route_density", 1.0))
    method = f"C2F-Route({density:g}%)"
    if mode == "route_reuse":
        method += f" + StepReuse-Out({int(attention.get('reuse_every', 1))})"
    return method, f"{density:g}%", str(attention.get("reused_attention_calls", 0))


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def recorded_total(data: dict[str, Any], family: str) -> float | None:
    if family == "fastwan":
        wrapper = data.get("sample_wrapper_seconds")
        decode = data.get("vae_decode_seconds")
        if wrapper is None:
            return None
        return float(wrapper) + float(decode or 0.0)
    value = data.get("end_to_end_inference_seconds")
    return None if value is None else float(value)


def effective_keep(data: dict[str, Any], family: str) -> str:
    if family == "fastwan":
        return "—"
    attention = data.get("attention", {})
    value = attention.get("mean_effective_sparse_density")
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def table(rows: list[tuple[str, dict[str, Any]]], family: str) -> str:
    baseline = next(
        (
            float(data["denoise_and_boundary_seconds"])
            for _, data in rows
            if data and data.get("denoise_and_boundary_seconds") is not None
        ),
        None,
    )
    baseline_total = next(
        (recorded_total(data, family) for _, data in rows if data),
        None,
    )
    output = [
        "| ID | 分辨率路径 | attention 方法 | 名义保留 | 实际保留 | Reuse 命中 | denoise | denoise 加速 | sampler+VAE | 总加速 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in rows:
        if not data:
            output.append(f"| `{name}` | — | — | — | — | — | — | — | — | 未完成 |")
            continue
        method, keep, hits = fastwan_method(data) if family == "fastwan" else original_method(data)
        denoise = data.get("denoise_and_boundary_seconds")
        speed = None if baseline is None or denoise in (None, 0) else baseline / float(denoise)
        total = recorded_total(data, family)
        total_speed = (
            None
            if baseline_total is None or total in (None, 0)
            else baseline_total / float(total)
        )
        output.append(
            f"| `{name}` | {trajectory_label(data)} `{factors(data)}` | {method} | {keep} | "
            f"{effective_keep(data, family)} | {hits} | {fmt(denoise)} s | {fmt(speed)}× | "
            f"{fmt(total)} s | {fmt(total_speed)}× |"
        )
    return "\n".join(output)


def gallery(rows: list[tuple[str, dict[str, Any]]], family: str, rel_dir: str) -> str:
    cards = ['<div class="method-gallery">']
    for name, data in rows:
        if not data:
            continue
        method, keep, hits = fastwan_method(data) if family == "fastwan" else original_method(data)
        if family == "fastwan":
            sampler = "DMD 3-step / 3 NFE"
        else:
            steps = "/".join(str(value) for value in data.get("steps_per_stage", []))
            sampler = (
                f"AB2 {steps} updates / "
                f"{data.get('actual_model_forward_calls', '—')} forwards"
            )
        cards.append(
            f'''<article class="method-card">
  <video controls preload="metadata" playsinline poster="{rel_dir}/{name}_mid.jpg" src="{rel_dir}/{name}.mp4"></video>
  <div class="method-copy"><h4><code>{name}</code></h4><ul>
    <li>路径：{trajectory_label(data)} <code>{factors(data)}</code></li>
    <li>采样：{sampler}</li>
    <li>attention：{method}；名义保留 {keep}；padding 后实际保留 {effective_keep(data, family)}</li>
    <li>StepReuse 实际命中：{hits} 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>{fmt(data.get('denoise_and_boundary_seconds'))} s</strong>；记录总时间 <strong>{fmt(recorded_total(data, family))} s</strong></li>
  </ul></div>
</article>'''
        )
    cards.append("</div>")
    return "\n".join(cards)


def build(base: Path) -> str:
    fast_dir = base / "fastwan_f29"
    warm_dir = base / "fastwan_reuse_warm_f29"
    fast = load_metrics(fast_dir, FASTWAN_ORDER)
    warm = load_metrics(warm_dir, FASTWAN_WARM_ORDER)
    fast_complete = sum(bool(data) for _, data in fast)
    warm_complete = sum(bool(data) for _, data in warm)
    return f"""# FastWan 上的 RDD、VSA 与 Attention Reuse

> 本页只展示 FastWan/FastVideo 实验。所有动态分辨率实验均使用 **noise filling**：上采样上一 stage 的 clean prediction，再按新 stage 的噪声强度抽取完整 IID Gaussian。原始 Wan 调试结果不再放在主页面。

## 1. 四个方法分别是什么

| 名称 | 唯一作用 | 不做什么 |
|---|---|---|
| **RDD** | DMD denoising 从低空间/时间分辨率走到完整分辨率 | 本身不规定 attention 必须 dense 或 sparse |
| **C2F-Route(p%)** | 上一分辨率的 block route 决定下一分辨率只计算哪些 Q-K block pairs | 不复用旧 attention 数值；不与在线 VSA router 同时使用 |
| **StepReuse-Out(k)** | 同一 resolution stage 内每 k 次 model call 重算一次 `O=A·V`，中间调用复用旧 `O` | 不减少 denoiser 调用次数；分辨率一变缓存立即失效 |
| **VSA(p%)** | 每层、每步用当前 fine Q/K/V 做 coarse router，再计算选中的 fine block pairs，并融合 learned coarse branch | 它是 FastWan checkpoint 的训练组成；不是 C2F-Route |

FastWan 的 `StepReuse-Out` 在官方 VSA backend 输出处命中：复用旧的 `O=A·V`，跳过本次 VSA coarse/sparse attention，但当前 QKV projection 和 output projection 仍运行。它是保守的 training-free 版本，不代表最终可达到的最大加速。

## 2. 哪些组合真的有意义

| 组合 | 是否运行 | 原因 |
|---|---|---|
| RDD + VSA | 是 | 使用 FastWan 原生 VSA，只改变三次 DMD call 的分辨率路径 |
| RDD 三分辨率 + StepReuse-Out | **无命中** | `(4,4)→(2,2)→(1,1)` 每个 shape 只调用一次 |
| RDD 两分辨率 + StepReuse-Out | 是 | 令前两次 DMD call 使用同一 coarse shape，第二次产生一次真实复用 |
| 固定全分辨率 + VSA + StepReuse-Out | 是 | 三次 DMD call 都共享 full-resolution shape |
| C2F-Route + 在线 VSA router | **否** | 两者都决定 sparse route；C2F 是对在线 VSA router 的替代，不是叠加 |

重要限制：目前没有 RDD-FastWan 训练权重。动态分辨率视频是 **zero-finetune 机制实验**，用于判断速度上限、哪类分辨率切换破坏最小，以及复用能否运行；不能当作最终论文质量。

计时口径：冷启动表保留首次 shape 编译成本。热启动表对每种路径先完整运行 1 次，再运行 3 次并报告 denoise 中位数；不含模型加载、T5 和 VAE。“相对首行”使用同表的固定全分辨率 VSA20 作为 1.00×。任务仍与训练共享 GPU，因此应看中位数和数量级，不把小于约 5% 的差距当成结论。

## 3. 第一批视频：冷启动与 VSA 密度

共同配置：FastWan2.1-T2V-1.3B-Diffusers，832×448，29 帧，seed 20260905，DMD timestep `[1000,757,522]`，shift 8，无额外 CFG。完成 {fast_complete}/{len(fast)}。

{table(fast, 'fastwan')}

{gallery(fast, 'fastwan', '../../eval_results/rdd_wan_method_matrix/fastwan_f29')}

## 4. FastWan 热启动：RDD × StepReuse-Out

共同配置仍是 FastWan DMD3、832×448、29 帧、相同 prompt/seed/timestep、VSA 保留 20%，只使用 noise filling。`reuse2` 表示同一 shape 的第二次 model call 在每个 transformer layer 复用第一次的 attention output。完成 {warm_complete}/{len(warm)}。

{table(warm, 'fastwan')}

{gallery(warm, 'fastwan', '../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29')}

## 5. 当前结论

- 固定全分辨率 `fixed_vsa20` 正常且清晰；此前 `StepReuse-Out(2)` 仍保留主体，`StepReuse-Out(3)` 更模糊。复用不是无损操作。
- 你指出的旧 #16 已用规范名称重跑为 `rdd_spatiotemporal_vsa100`。它确实保留了狐狸轮廓，但仍有严重块状和颜色伪影。降低到 VSA 50%/20% 后更差：**主因是未训练权重直接走空间 RDD 路径，稀疏化是叠加损伤，不是唯一原因。**
- **热启动后 RDD 明确更快。**固定全分辨率是 1.880 s；标准三分辨率 RDD 是 0.880 s，即 denoiser **2.14×**，连同约 2.2 s VAE 后仍约 **1.35×**。
- **当前“又快、又最不坏”的配置是 temporal 两阶段 + reuse2：**`(1,2)→(1,2)→(1,1)`，1.233 s，denoiser **1.52×**；sampler+VAE 约 3.48 s，相对 4.11 s 为 **1.18×**。中间帧仍是清楚、写实的狐狸；但还需查看完整运动并增加 seeds。
- 时空两阶段 + reuse2 达到 0.842 s，denoiser **2.23×**、sampler+VAE **1.35×**；狐狸和场景可辨认，但风格化与结构伪影明显强于 temporal-only。
- 空间两阶段约 2.04–2.06× denoiser 加速，但画面出现明显块状/绘画化伪影。标准三分辨率虽然有 2.14×，质量仍崩坏。因此 zero-finetune 的主要不兼容来自空间 latent 网格切换。
- StepReuse 的增量并不稳定：固定全分辨率约 +6%，temporal 两阶段约 +1%，时空两阶段约 +25%。当前 GPU 与训练共享，正式论文数字必须在独占 GPU 上增加 warmup/repeats；大于 1.5× 的 RDD 主效应可信，小幅 reuse 差异暂不应过度解读。

## 6. 读图顺序

1. `fixed_vsa20_hot` 是严格速度与画质基线。
2. 比较每对两阶段 `*_hot` 与 `*_reuse2_hot`，只看复用的增量影响。
3. 横向比较 temporal、spatial、spatiotemporal，判断减少哪一维最划算。
4. `rdd_spatiotemporal_3stage_hot` 回答此前慢是否主要由三种 shape 的首次编译造成。

<style>
.method-gallery {{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;margin:18px 0 30px}}
.method-card {{overflow:hidden;border:1px solid #d0d7de;border-radius:12px;background:#fff;box-shadow:0 2px 8px rgb(31 35 40 / 6%)}}
.method-card video {{display:block;width:100%;aspect-ratio:832/448;object-fit:contain;background:#111}}
.method-copy {{padding:12px 15px 15px}}
.method-copy h4 {{margin:0 0 7px}}
.method-copy ul {{margin:0;padding-left:20px}}
</style>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.base), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
