"""Build a standalone, cache-safe report for the FastWan warm-start matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ORDER = [
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


QUALITY_NOTES = {
    "fixed_vsa20_hot": "正常基线：写实狐狸、雪地和树木清楚，主体结构稳定。",
    "fixed_vsa20_reuse2_hot": "仍是清楚的写实狐狸；相对基线略软，但没有明显块状崩坏。",
    "rdd_temporal_2stage_hot": "当前最佳质量/速度折中：狐狸与森林清楚，zero-finetune 下仍接近正常视频。",
    "rdd_temporal_2stage_reuse2_hot": "主体和场景仍清楚；复用改变了细节与构图，但没有出现空间 RDD 那种严重块状伪影。",
    "rdd_spatial_2stage_hot": "狐狸可辨认，但明显绘画化、块状化，局部几何和颜色不稳定。",
    "rdd_spatial_2stage_reuse2_hot": "与无复用空间版相似，主体可辨认但伪影明显；复用没有修复空间切换问题。",
    "rdd_spatiotemporal_2stage_hot": "狐狸与雪地可辨认，但画面明显风格化，结构和纹理存在 transition 伪影。",
    "rdd_spatiotemporal_2stage_reuse2_hot": "仍能辨认狐狸和森林；速度最高之一，但画质明显弱于 temporal-only。",
    "rdd_spatiotemporal_3stage_hot": "严重彩色和块状崩坏，只剩局部动物/场景线索；不属于可用画质。",
}


def read_metrics(directory: Path, name: str) -> dict[str, Any]:
    path = directory / f"{name}.metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def factor_text(data: dict[str, Any]) -> str:
    return "→".join(
        f"({int(pair[0])},{int(pair[1])})"
        for pair in data["stage_factors_spatial_temporal"]
    )


def shape_text(data: dict[str, Any]) -> str:
    return "→".join(
        f"({int(shape[0])},{int(shape[1])},{int(shape[2])})"
        for shape in data["stage_latent_shapes"]
    )


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def f(value: float) -> str:
    return f"{float(value):.3f}"


def run_times(data: dict[str, Any]) -> str:
    records = data["steady_state_benchmark"]["records"]
    return "/".join(f(record["denoise_and_boundary_seconds"]) for record in records)


def stage_times(data: dict[str, Any]) -> str:
    records = data["steady_state_benchmark"]["records"]
    return "; ".join(
        "[" + ", ".join(f(value) for value in record["stage_forward_seconds"]) + "]"
        for record in records
    )


def build(directory: Path) -> str:
    rows = [(name, read_metrics(directory, name)) for name in ORDER]
    baseline_denoise = float(rows[0][1]["denoise_and_boundary_seconds"])
    baseline_total = baseline_denoise + float(rows[0][1]["vae_decode_seconds"])

    table_lines = [
        "| ID | stage factor `(spatial,temporal)` | latent `(T,H,W)` | Reuse 命中 | 3 次 denoise | median | denoise 加速 | sampler+VAE | 总加速 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    cards: list[str] = ['<div class="warm-gallery">']
    details: list[str] = []

    for name, data in rows:
        denoise = float(data["denoise_and_boundary_seconds"])
        decode = float(data["vae_decode_seconds"])
        total = float(data["sample_wrapper_seconds"]) + decode
        hits = int(data["attention"]["step_reuse_hits"])
        reuse = int(data["attention"]["reuse_every"])
        video = directory / f"{name}.mp4"
        poster = directory / f"{name}_mid.jpg"
        if not video.is_file() or not poster.is_file():
            raise FileNotFoundError(f"missing media for {name}")
        table_lines.append(
            f"| `{name}` | `{factor_text(data)}` | `{shape_text(data)}` | {hits} | "
            f"{run_times(data)} s | **{f(denoise)} s** | **{baseline_denoise / denoise:.2f}×** | "
            f"{f(total)} s | **{baseline_total / total:.2f}×** |"
        )
        media_rel = f"../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/{name}"
        cards.append(
            f'''<article class="warm-card">
  <video controls preload="metadata" playsinline poster="{media_rel}_mid.jpg" src="{media_rel}.mp4"></video>
  <div class="warm-copy"><h3><code>{name}</code></h3><ul>
    <li>stage factor：<code>{factor_text(data)}</code></li>
    <li>latent (T,H,W)：<code>{shape_text(data)}</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out({reuse})；实际命中 {hits}</li>
    <li>3 次 denoise：{run_times(data)} s；median <strong>{f(denoise)} s</strong>；相对基线 <strong>{baseline_denoise / denoise:.2f}×</strong></li>
    <li>VAE：{f(decode)} s；sampler+VAE：{f(total)} s；总加速 {baseline_total / total:.2f}×</li>
    <li>目视记录：{QUALITY_NOTES[name]}</li>
    <li>文件校验：<code>sha256:{sha256_prefix(video)}</code>；{video.stat().st_size / 1024:.1f} KiB</li>
  </ul></div>
</article>'''
        )
        details.append(
            f"### `{name}`\n\n"
            f"- 三次完整 trajectory 的 stage-forward 秒数依次为：`{stage_times(data)}`。\n"
            f"- attention 调用统计：online VSA `{data['attention']['online_vsa_calls']}`，"
            f"reuse hits `{hits}`，recompute/cache writes `{data['attention']['step_reuse_recomputes']}`。\n"
            f"- 当前进程峰值 CUDA allocation（decode 前）："
            f"`{float(data['peak_cuda_memory_before_decode_bytes']) / 2**30:.2f} GiB`。\n"
        )
    cards.append("</div>")

    return f"""# FastWan：RDD + Attention Reuse 热启动实验

> 这是一个全新的独立报告，只对应 `fastwan_reuse_warm_f29` 中刚完成的 **9 条 FastWan 视频**。不包含原始 Wan，不包含旧的冷启动视频，不使用 redraw。

## 1. 完整公共配置

| 项目 | 设置 |
|---|---|
| 模型 | `FastWan2.1-T2V-1.3B-Diffusers`，由 FastVideo native `WanTransformerBlock_VSA` 加载 |
| 输出 | `832×448`，29 帧，16 FPS（约 1.81 秒） |
| 完整 latent | `(C,T,H,W)=(16,8,56,104)` |
| Prompt | `A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.` |
| Seed | `20260905` |
| Sampler | FastWan DMD 3-step / 3 NFE |
| DMD timestep | `[1000,757,522]`，shift `8.0` |
| CFG | 无额外 CFG；使用蒸馏后的 FastWan checkpoint |
| Attention | FastWan 原生 VSA；`vsa_sparsity=0.8`，即名义保留 20% fine tile pairs |
| Transition | 只使用 `noise_filling` |
| 热启动 | 每个配置先完整运行 1 条不计时 trajectory，再运行 3 条计时 trajectory，报告中位数 |
| 硬件状态 | 物理 GPU0；与已有训练共享 GPU，因此不是最终论文级独占计时 |

### Stage factor 到底表示什么

每个 pair 是 `(spatial_factor, temporal_factor)`：

- `(1,1)`：完整空间、完整时间；latent 为 `(8,56,104)`。
- `(1,2)`：空间不变，时间减半；latent 为 `(4,56,104)`。
- `(2,1)`：时间不变，H/W 各减半；latent 为 `(8,28,52)`。
- `(2,2)`：时间、H、W 都减半；latent 为 `(4,28,52)`。
- `(4,4)`：时间、H、W 都为完整 latent 的约四分之一；本实验为 `(2,14,26)`。

### Noise filling 的精确含义

从上一 stage 得到 clean prediction 后，先把 clean latent 上采样到下一 stage 的 shape，然后重新抽取该 shape 的完整 IID Gaussian：

$$
x_{{\\sigma_{{next}}}}=(1-\\sigma_{{next}})\\,\\operatorname{{Upsample}}(\\hat x_0)+\\sigma_{{next}}\\epsilon_{{new}}.
$$

这里没有保留上一 stage 的旧噪声，也没有 redraw 这个独立实验分支。

### StepReuse-Out(2) 的精确含义

如果连续两个 DMD model call 的 latent shape 相同，每个 transformer layer 的第二次调用直接复用第一次保存的 `O=A·V`。当前实现仍计算当前 Q/K/V projection 和 output projection，只跳过 VSA attention backend。因此：

- 三分辨率 `(4,4)→(2,2)→(1,1)` 没有重复 shape，reuse 必然命中 0 次。
- 两阶段 `coarse→coarse→full` 的第二个 coarse call 会在 30 层各命中一次，共 30 次。
- 固定 `full→full→full` 在第二个 call 命中 30 次；第三个 call 重新计算。

## 2. 全部数值结果

基线是 `fixed_vsa20_hot`。`sampler+VAE` 不含 T5/model load 和视频编码，只是 sampler wrapper 中位数加本次 VAE decode。

{chr(10).join(table_lines)}

## 3. 最重要的结论

1. **热启动后 RDD 的 denoiser 确实更快。**标准三分辨率 RDD 为 0.880 s，对比固定全分辨率 1.880 s，是 2.14×。
2. **目前最可信的质量/速度折中是 temporal 两阶段 RDD。**无复用版 1.246 s，即 denoiser 1.51×、sampler+VAE 1.19×，中间帧保持写实且结构正常。
3. temporal 两阶段加入 reuse2 后是 1.233 s，只比 1.246 s 快约 1%；这不能算明确的额外加速。它的意义主要是确认 FastWan+RDD+reuse 的执行路径和缓存命中正确。
4. 时空两阶段+reuse2 达到 0.842 s，即 denoiser 2.23×、sampler+VAE 1.35×；但画质出现明显结构/风格伪影。
5. 空间降采样和三分辨率 RDD 的 zero-finetune 画质明显更差。因此当前最合理的下一步是训练/微调 RDD-FastWan，而不是继续用未适配 checkpoint 推高空间降采样比例。
6. VAE 约固定占 2.2 s，把 denoiser 的 2× 加速压缩成约 1.35× 总加速。论文若汇报端到端，还需要 VAE 优化或更长视频，让 denoiser 占比更大。

## 4. 九条视频：ID、参数与文件一一对应

{chr(10).join(cards)}

## 5. 每条实验的低层计时记录

{chr(10).join(details)}

## 6. 为什么冷启动特别慢；换 shape 会不会再次变慢

会。FastWan 的 VSA 使用 shape/block-layout 专门化的 CUDA/Triton kernel。某个 `(T,H,W)`、tile grid、dtype 或稀疏布局第一次出现时，除了真实矩阵计算，还可能发生：

1. Triton kernel 编译与 autotune；
2. cuBLAS/CUDA handle、workspace 和显存池的首次建立；
3. shape-specific VSA metadata、padding/index buffer 的首次构造；
4. lazy module/kernel 初始化。

冷启动日志直接显示了这个效应：

| 路径 | 第一次运行的三个 stage/model-call | 冷启动总计 | 热启动 median |
|---|---:|---:|---:|
| 固定 full→full→full | 2.920 / 0.588 / 0.627 s | 4.20 s | 1.880 s |
| 三分辨率 `(4,4)→(2,2)→(1,1)` | 3.482 / 2.273 / 2.244 s | 8.16 s | 0.880 s |

固定全分辨率只在第一个 call 遇到一种新 shape，后两个 call 已经复用同一 shape 的已编译 kernel；三分辨率 RDD 的三个 call 分别遇到三种新 shape，因此第一次视频连续支付三次初始化成本。这就是旧冷启动比较看起来 RDD 反而慢约两倍的主要原因。

如果先只热身 shape A，再换到此前从未出现的 shape B，B **很可能再次出现一次性变慢**。但只要进程不退出，之后在已经热身过的 A/B 之间切换，通常会复用缓存，不再重复完整编译；实际执行时间仍会随 token 数变化。

部署和正式 benchmark 应当：

- 启动时依次预热所有允许的 RDD stage shape，而不是只预热 full resolution；
- 把任意尺寸归入少数几个固定 `(T,H,W)` bucket，避免无限产生新 kernel key；
- 保持同一 worker 进程，保存 Triton cache，并为每个 stage 预建 metadata/workspace；
- 论文同时报告 cold-first-video 与 warm steady-state latency，主吞吐量表使用 warm median；
- 若服务主要是一次性 CLI 调用，冷启动是真实成本，不能只报告热启动。

## 7. 复现命令

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
GPU=0 WARMUP_RUNS=1 TIMED_RUNS=3 \\
  bash rdd_video/run_fastwan_rdd_reuse_warm_benchmark.sh 2>&1 \\
  | tee eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/driver.log
```

<style>
.warm-gallery {{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px;margin:18px 0 30px}}
.warm-card {{overflow:hidden;border:1px solid #d0d7de;border-radius:12px;background:#fff;box-shadow:0 2px 8px rgb(31 35 40 / 7%)}}
.warm-card video {{display:block;width:100%;aspect-ratio:832/448;object-fit:contain;background:#111}}
.warm-copy {{padding:12px 15px 15px}}
.warm-copy h3 {{margin:0 0 7px}}
.warm-copy ul {{margin:0;padding-left:20px}}
</style>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.input), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
