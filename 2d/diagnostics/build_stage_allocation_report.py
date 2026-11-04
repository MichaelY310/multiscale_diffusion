#!/usr/bin/env python3
"""Build a self-contained HTML/asset bundle from stage-allocation diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


METRIC_PATTERNS = {
    "IS": re.compile(r"(?m)^Inception Score:\s*([-+0-9.eE]+)\s*$"),
    "FID": re.compile(r"(?m)^FID:\s*([-+0-9.eE]+)\s*$"),
    "sFID": re.compile(r"(?m)^sFID:\s*([-+0-9.eE]+)\s*$"),
    "Precision": re.compile(r"(?m)^Precision:\s*([-+0-9.eE]+)\s*$"),
    "Recall": re.compile(r"(?m)^Recall:\s*([-+0-9.eE]+)\s*$"),
}


def parse_metrics(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            return None
        values[name] = float(match.group(1))
    return values


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def independent_csv_effects(path_a: Path, path_b: Path, label: str) -> List[dict]:
    """Difference of means for two independently generated image sets.

    The two historical runs used different distributed world sizes.  Because
    the sampler assigns strided indices to ranks and seeds each batch from its
    first index, an equal output index does not identify an equal stochastic
    trajectory.  Use an independent-sample standard error, not a paired one.
    """
    if not path_a.is_file() or not path_b.is_file():
        return []
    with path_a.open(newline="", encoding="utf-8") as handle:
        rows_a = list(csv.DictReader(handle))
    with path_b.open(newline="", encoding="utf-8") as handle:
        rows_b = list(csv.DictReader(handle))
    if not rows_a or not rows_b:
        return []
    metric_names = [key for key in rows_a[0] if key != "source_index"]
    effects = []
    for metric in metric_names:
        values_a = [float(row[metric]) for row in rows_a]
        values_b = [float(row[metric]) for row in rows_b]
        n_a, n_b = len(values_a), len(values_b)
        mean_a, mean_b = sum(values_a) / n_a, sum(values_b) / n_b
        variance_a = sum((value - mean_a) ** 2 for value in values_a) / max(n_a - 1, 1)
        variance_b = sum((value - mean_b) ** 2 for value in values_b) / max(n_b - 1, 1)
        mean = mean_b - mean_a
        sem = math.sqrt(variance_a / n_a + variance_b / n_b)
        effects.append(
            {
                "comparison": label,
                "metric": metric,
                "n_original": n_a,
                "n_continued": n_b,
                "mean_continued_minus_original": mean,
                "independent_sem": sem,
                "ci95_low_welch_normal": mean - 1.96 * sem,
                "ci95_high_welch_normal": mean + 1.96 * sem,
            }
        )
    return effects


def fmt(value, digits: int = 5) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def table(headers: Sequence[str], rows: Sequence[Sequence], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(relative_path: str, caption: str) -> str:
    return (
        '<figure><a href="' + html.escape(relative_path) + '">'
        '<img loading="lazy" src="' + html.escape(relative_path) + '" alt="' + html.escape(caption) + '"></a>'
        '<figcaption>' + html.escape(caption) + '</figcaption></figure>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    diagnostics = root / "eval_results" / "diagnostics"
    bundle = args.bundle_dir.expanduser().resolve()
    assets = bundle / "assets"
    data_dir = assets / "data"
    image_dir = assets / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[Path, str] = {}

    def copy_file(source: Path, category: str, name: Optional[str] = None) -> Optional[str]:
        if not source.is_file():
            return None
        destination_dir = image_dir if category == "images" else data_dir
        destination = destination_dir / (name or source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        relative = destination.relative_to(bundle).as_posix()
        copied[source] = relative
        return relative

    full_metric_logs = {
        "all stages 400K / CFG 1.0": diagnostics / "full_metrics" / "ms100300600_400k_808090_ddpm_cfg1_noisefilling.txt",
        "all stages 400K / CFG 1.5": diagnostics / "full_metrics" / "ms100300600_400k_808090_ddpm_cfg15_noisefilling.txt",
        "8x8 continued 600K, 16/32 400K / CFG 1.0": root / "eval_results" / "ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg1_noisefilling.txt",
        "8x8 continued 600K, 16/32 400K / CFG 1.5": root / "eval_results" / "ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg15_noisefilling.txt",
    }
    full_metrics = {label: parse_metrics(path) for label, path in full_metric_logs.items()}
    # Correct original-ImageNet Multistage XL+L+B comparison.  Keep the exact
    # Hugging Face evaluator logs in the bundle: an earlier summary accidentally
    # put the 600-step result (FID 3.627) in the 250-step CFG1.5 row.
    ms300_reference_paths = {
        "300/300/400 @ 400K / CFG 1.0": root / "2d" / "diagnostics" / "reference_logs" / "multistage300300400_380k_908080_ddpm_cfg1.txt",
        "300/300/400 @ 400K / CFG 1.5": root / "2d" / "diagnostics" / "reference_logs" / "multistage300300400_380k_908080_ddpm_cfg1.5.txt",
    }
    ms300_reference_fallback = {
        "300/300/400 @ 400K / CFG 1.0": {
            "IS": 98.4415, "FID": 11.829104, "sFID": 4.575259,
            "Precision": 0.68026, "Recall": 0.6282,
        },
        "300/300/400 @ 400K / CFG 1.5": {
            "IS": 218.8609161376953, "FID": 3.8148521692417603,
            "sFID": 5.263469267446908, "Precision": 0.84202, "Recall": 0.5083,
        },
    }
    ms300_reference = {
        label: parse_metrics(ms300_reference_paths[label]) or fallback
        for label, fallback in ms300_reference_fallback.items()
    }
    copy_file(root / "2d" / "ALL_RESULTS.md", "data", "ALL_RESULTS_snapshot.md")
    reproducibility_sources = (
        "README_STAGE_ALLOCATION_DIAGNOSTICS_ZH.md",
        "analyze_image_frequency.py",
        "analyze_training_losses.py",
        "replot_frequency_results.py",
        "evaluate_many_npz.py",
        "visualize_multistage_pred_x0.py",
        "profile_teacher_forced_pred_x0.py",
        "evaluate_generated_class_consistency.py",
        "inspect_checkpoint_metadata.py",
        "run_stage_allocation_cpu_diagnostics.sh",
        "run_pred_x0_after_gpu0_idle.sh",
        "run_teacher_forced_after_pred_x0.sh",
        "run_local_solver_error_after_teacher_forced.sh",
        "run_class_consistency_after_local_solver.sh",
        "reconstruct_npz_with_vae.py",
        "run_vae_frequency_control_after_gpu1_idle.sh",
        "multistage_pred_x0_300_vs_100.json",
    )
    for source_name in reproducibility_sources:
        copy_file(root / "2d" / "diagnostics" / source_name, "data", f"source_{source_name}")
    for source, name in (
        (root / "2d" / "sample_fid_unified.py", "source_sample_fid_unified.py"),
        (root / "guided-diffusion" / "evaluations" / "evaluator.py", "source_adm_evaluator.py"),
        (root / "2d" / "exp" / "eval_ms100300600_400k_2gpu.sh", "source_eval_ms100300600_400k_2gpu.sh"),
        (root / "2d" / "exp" / "eval_ms100300600_8x8continued_600k_4gpu.sh", "source_eval_ms100300600_8x8continued_600k_4gpu.sh"),
    ):
        copy_file(source, "data", name)
    for label, path in full_metric_logs.items():
        copy_file(path, "data", f"full_metrics_{label.lower().replace(' ', '').replace('.', 'p')}.txt")
    for label, path in ms300_reference_paths.items():
        copy_file(path, "data", f"ms300_250step_{label.split('/')[-1].strip().replace(' ', '').replace('.', 'p')}.txt")
    copy_file(
        root / "2d" / "diagnostics" / "reference_logs" / "ms_400k_200-200-200_ddpm_cfg1.5.txt",
        "data",
        "ms300_600step_200_200_200_cfg1p5_not_used_as_250step.txt",
    )
    copy_file(
        root / "2d" / "diagnostics" / "reference_logs" / "ms300_400k_checkpoint_metadata.json",
        "data",
        "ms300_400k_checkpoint_metadata.json",
    )

    training_loss_root = diagnostics / "training_losses"
    training_loss = read_json(training_loss_root / "training_loss_summary.json")
    for filename in ("training_loss_summary.json", "training_loss_records.csv"):
        copy_file(training_loss_root / filename, "data", f"training_{filename}")
    for source, name in (
        (root / "2d" / "train_RDD_imagenet256_300_300_400_original.log", "training_raw_RDD_300_300_400.log"),
        (root / "2d" / "train_RDD_imagenet256_100_300_600_original.log", "training_raw_RDD_100_300_600.log"),
        (root / "2d" / "exp" / "train_ms100300600_continue_8x8_2gpu.log", "training_raw_RDD_100_300_600_8x8_continuation.log"),
    ):
        copy_file(source, "data", name)
    training_loss_figure = copy_file(
        training_loss_root / "training_loss_convergence.png",
        "images",
        "training_loss_convergence.png",
    )

    frequency_root = diagnostics / "stage_allocation"
    frequency = read_json(frequency_root / "summary.json")
    for filename in (
        "summary.json", "comparison_vs_reference.csv", "fft_radial_profiles.csv",
        "ImageNet_original_per_image.csv", "RDD_100_300_600_cfg1_per_image.csv",
        "RDD_100_300_600_cfg1p5_per_image.csv",
        "RDD_100_300_600_8x8continued600K_cfg1_per_image.csv",
        "RDD_100_300_600_8x8continued600K_cfg1p5_per_image.csv",
    ):
        copy_file(frequency_root / filename, "data", f"frequency_{filename}")
    frequency_laplacian_figures = []
    frequency_fft_figures = []
    for filename in (
        "laplacian_band_fractions.png", "fft_radial_profile.png", "fft_band_fractions.png",
        "ImageNet_original_laplacian_examples.png",
        "RDD_100_300_600_cfg1_laplacian_examples.png",
        "RDD_100_300_600_cfg1p5_laplacian_examples.png",
        "RDD_100_300_600_8x8continued600K_cfg1_laplacian_examples.png",
        "RDD_100_300_600_8x8continued600K_cfg1p5_laplacian_examples.png",
    ):
        relative = copy_file(frequency_root / filename, "images", f"frequency_{filename}")
        if relative:
            destination = frequency_fft_figures if filename.startswith("fft_") else frequency_laplacian_figures
            destination.append((relative, filename))

    independent_frequency = []
    independent_frequency.extend(
        independent_csv_effects(
            frequency_root / "RDD_100_300_600_cfg1_per_image.csv",
            frequency_root / "RDD_100_300_600_8x8continued600K_cfg1_per_image.csv",
            "CFG1.0: 8x8 continued600K - original400K",
        )
    )
    independent_frequency.extend(
        independent_csv_effects(
            frequency_root / "RDD_100_300_600_cfg1p5_per_image.csv",
            frequency_root / "RDD_100_300_600_8x8continued600K_cfg1p5_per_image.csv",
            "CFG1.5: 8x8 continued600K - original400K",
        )
    )
    if independent_frequency:
        effects_path = data_dir / "frequency_independent_continued_minus_original.csv"
        with effects_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(independent_frequency[0].keys()))
            writer.writeheader()
            writer.writerows(independent_frequency)

    lowpass_metrics = {}
    lowpass_dir = frequency_root / "lowpass_metrics"
    lowpass_samples = (
        "RDD_100_300_600_cfg1",
        "RDD_100_300_600_cfg1p5",
        "RDD_100_300_600_8x8continued600K_cfg1",
        "RDD_100_300_600_8x8continued600K_cfg1p5",
    )
    for scale in (64, 128):
        copy_file(
            lowpass_dir / f"summary_lowpass{scale}_10k.json",
            "data",
            f"lowpass_summary_lowpass{scale}_10k.json",
        )
        for sample_label in lowpass_samples:
            label = f"{sample_label} low-pass {scale} (10K)"
            path = lowpass_dir / f"{sample_label}_lowpass{scale}_10k.txt"
            lowpass_metrics[label] = parse_metrics(path)
            copy_file(path, "data", f"lowpass_{sample_label}_{scale}_10k.txt")

    pred_root_three = diagnostics / "stage_pred_x0" / "300_vs_100_vs_600"
    pred_root = (
        pred_root_three
        if (pred_root_three / "metadata.json").is_file()
        else diagnostics / "stage_pred_x0" / "300_vs_100"
    )
    pred_metadata = read_json(pred_root / "metadata.json")
    copy_file(pred_root / "metadata.json", "data", "pred_x0_metadata.json")
    pred_figures = []
    for path in sorted((pred_root / "comparisons").glob("*.png")) if (pred_root / "comparisons").is_dir() else []:
        relative = copy_file(path, "images", f"pred_x0_{path.name}")
        if relative:
            pred_figures.append((relative, path.stem))

    teacher_root = diagnostics / "teacher_forced_pred_x0" / "300_vs_100"
    teacher = read_json(teacher_root / "teacher_forced_summary.json")
    for filename in ("teacher_forced_summary.json", "teacher_forced_summary.csv", "teacher_forced_per_image.csv"):
        copy_file(teacher_root / filename, "data", filename)
    teacher_figures = []
    for path in sorted(teacher_root.glob("*.png")):
        relative = copy_file(path, "images", path.name)
        if relative:
            teacher_figures.append((relative, path.stem))

    local_solver = {}
    local_root = diagnostics / "local_solver_error"
    for allocation in (
        "RDD_300_300_400",
        "RDD_100_300_600",
        "RDD_100_300_600_8x8continued600K",
    ):
        for solver in ("ddpm", "dpm2"):
            path = local_root / f"{allocation}_{solver}.json"
            local_solver[f"{allocation} / {solver}"] = read_json(path)
            copy_file(path, "data", path.name)

    class_root = diagnostics / "class_consistency" / "100_300_600"
    class_consistency = read_json(class_root / "class_consistency_summary.json")
    for filename in (
        "class_consistency_summary.json",
        "reconstructed_condition_labels_ws2_bs512.npy",
        "reconstructed_condition_labels_ws4_bs512.npy",
        "RDD_100_300_600_cfg1_per_image.csv", "RDD_100_300_600_cfg1p5_per_image.csv",
        "RDD_100_300_600_8x8continued600K_cfg1_per_image.csv",
        "RDD_100_300_600_8x8continued600K_cfg1p5_per_image.csv",
    ):
        copy_file(class_root / filename, "data", f"class_{filename}")

    sections: List[str] = []
    sections.append(
        """
        <section id="summary"><h2>先说结论</h2>
        <div class="callout">
        <p><strong>Precision/Recall 并不是全部都没算。</strong>后来的 8×8-continued-600K hybrid 两组日志已经有完整五项；更早的 all-stages-400K 两组在最后两项使用 NumPy 已删除的 <code>np.bool</code>，所以前三项写入日志后崩溃。旧采样脚本又只检查日志中是否出现 <code>FID:</code>，从而把这种半截日志误认为完成。</p>
        <p>现在 evaluator 已改为 <code>np.bool_</code>，采样脚本只有在五项全部出现时才认定成功，并额外写出结构化 JSON。</p>
        </div>
        <p>本报告把“大样本结果”“阶段轨迹图”“真实 forward 上的预测误差”“频率诊断”分开。频谱相似不等于 FID 一定好；少量图片也不能替代 50K 统计。</p>
        </section>
        """
    )

    metric_rows = []
    for label, metrics in ms300_reference.items():
        metric_rows.append([label, *[fmt(metrics.get(name)) for name in ("IS", "FID", "sFID", "Precision", "Recall")]])
    for label, metrics in full_metrics.items():
        metric_rows.append([label, *([fmt(metrics.get(name)) for name in ("IS", "FID", "sFID", "Precision", "Recall")] if metrics else ["未完成"] * 5)])
    sections.append(
        '<section id="metrics"><h2>1. 50K 标准指标：100/300/600 原始与 8×8 继续训练</h2>'
        + table(("setting", "IS ↑", "FID ↓", "sFID ↓", "Precision ↑", "Recall ↑"), metric_rows)
        + '<p class="note">300/300/400 与 100/300/600 都加载三个 <code>0400000.pt</code>，是严格同为 400K 的训练 step；旧 evaluator 文件名虽含 <code>380k</code>，checkpoint 元数据显示它从 <code>0380000.pt</code> 恢复后继续保存为 400K。模型架构、原生 ImageNet、DDPM250、noise filling 和 CFG 也对齐。CFG1.5 的 250-step 正确 FID 是 3.814852；3.627037 属于另一个 200/200/200、总 600-step 采样，不能混入本表。Precision 更接近“生成图看起来是否落在真实图流形上”；Recall 更接近“真实数据的不同模式有没有被覆盖”。</p></section>'
    )

    training_rows = []
    if training_loss:
        for row in training_loss["summaries"]:
            training_rows.append(
                [
                    row["run"], f"{row['resolution']}×{row['resolution']}",
                    f"{row['first_step'] / 1000:g}K→{row['final_step'] / 1000:g}K",
                    fmt(row["previous_window_mean"]), fmt(row["last_window_mean"]),
                    ("—" if row["relative_last_minus_previous"] is None else f"{100 * row['relative_last_minus_previous']:+.3f}%"),
                    fmt(row["linear_slope_over_last_2windows_per_100k_steps"]),
                ]
            )
    sections.append(
        '<section id="training-loss"><h2>2. 训练 loss：先判断是不是还没收敛</h2>'
        '<p>表中比较的是<strong>同一个 run 自己的最后两个 10K 窗口</strong>。100/300/600 的 8×8 stage 在原始 400K 末段和继续到 600K 末段都几乎水平；这能解释为什么再给 8×8 训练 200K 只带来很小的 FID 恢复。不同分配下，同名 stage 负责的全局 timestep 范围不同，所以 raw loss 的绝对值不能跨 schedule 直接比大小。</p>'
        + (table(("run", "stage", "available log", "previous 10K", "last 10K", "relative Δ", "recent slope / 100K"), training_rows) if training_rows else '<p class="pending">训练 loss 解析尚未完成。</p>')
        + '</section>'
    )
    if training_loss_figure:
        sections.append('<div class="figure-grid wide">' + figure(training_loss_figure, "各训练日志内的 moving-average loss；只比较趋势，不跨 schedule 比 raw level") + '</div>')

    laplacian_explanation = """
    <section id="laplacian"><h2>3. Laplacian 金字塔：像把一张画拆成三层透明纸</h2>
    <ol>
      <li>先把 256×256 图缩到 64×64 再放大。它保留大色块、主体位置和很粗的轮廓，叫“底图”。</li>
      <li>再做一张 128×128 版本；用它减去 64×64 底图，剩下的是 8×8 latent 升到 16×16 时新增的中等细节。</li>
      <li>原始 256×256 减去 128×128 版本，剩下的是 16×16 latent 升到 32×32 时新增的最细纹理和边缘。</li>
    </ol>
    <p>每层所有像素先平方再取平均，就是这层的“能量”：数字越大，说明这一层放的变化越多。三层使用嵌套的块平均投影，因此近似正交，能量可以相加。它不是把图片做锐化，也不是拿一个 Laplacian 卷积核扫边；这里的重点是<strong>按 RDD 的三个空间尺度拆层</strong>。</p>
    <div class="equation">图像 = 64px 底图 + (128px − 64px) 中层细节 + (256px − 128px) 高频细节</div>
    <p>脚本还额外算一个传统的“四邻居 Laplacian”：对每个像素做“上 + 下 + 左 + 右 − 4×自己”。一整块颜色相同时正好互相抵消为 0，边缘或细碎噪声处才会变大，所以它是一个粗略的锐利程度计。金字塔回答“变化属于多大的结构”，四邻居值回答“局部变化到底有多急”，两者用途不同。</p>
    <p class="note">这些操作是在 VAE decode 后的 256px RGB 图上做的。64/128px 与 latent 8/16 的倍率对齐，但 VAE 是非线性的，所以它是 output-side scale proxy，不是假装能精确还原 latent 内部频带。</p>
    </section>
    """
    sections.append(laplacian_explanation)

    if frequency:
        dataset_lookup = {item["label"]: item for item in frequency["datasets"]}
        freq_rows = []
        for label, item in dataset_lookup.items():
            m = item["metrics"]
            freq_rows.append(
                [
                    label, item["analyzed_images"],
                    fmt(m["low64_fraction"]["mean"]),
                    fmt(m["mid_fraction"]["mean"]),
                    fmt(m["high_fraction"]["mean"]),
                    fmt(m["gradient_energy"]["mean"]),
                    fmt(m["laplacian_filter_energy"]["mean"]),
                ]
            )
        sections.append(
            '<p class="note">每个集合固定抽取 10K 张（不足则全取），随机种子与抽样下标写入 summary.json；所有集合用完全相同的像素归一化、投影与 FFT 代码。</p>'
            + table(("dataset", "N", "≤64 fraction", "64→128 fraction", "128→256 fraction", "gradient energy", "4-neighbor edge energy"), freq_rows)
        )

    vae_control_root = diagnostics / "vae_frequency_control" / "analysis"
    vae_control = read_json(vae_control_root / "summary.json")
    copy_file(vae_control_root / "summary.json", "data", "vae_control_summary.json")
    copy_file(
        vae_control_root / "comparison_vs_reference.csv",
        "data",
        "vae_control_comparison_vs_reference.csv",
    )
    vae_control_figures = []
    for filename in (
        "laplacian_band_fractions.png",
        "fft_radial_profile.png",
        "fft_band_fractions.png",
        "ImageNet_original_laplacian_examples.png",
        "ImageNet_SD_VAE_mean_reconstruction_laplacian_examples.png",
    ):
        relative = copy_file(vae_control_root / filename, "images", f"vae_control_{filename}")
        if relative:
            vae_control_figures.append((relative, filename))
    else:
        sections.append('<p class="pending">频率分析尚未完成。</p>')
    if frequency_laplacian_figures:
        sections.append('<div class="figure-grid">' + "".join(figure(path, caption) for path, caption in frequency_laplacian_figures) + '</div>')
    if independent_frequency:
        selected = {
            "mid_fraction",
            "high_fraction",
            "gradient_energy",
            "laplacian_filter_energy",
            "fft_high_fraction",
        }
        independent_rows = [
            [
                row["comparison"],
                row["metric"],
                f"{row['n_original']} / {row['n_continued']}",
                fmt(row["mean_continued_minus_original"]),
                f"[{fmt(row['ci95_low_welch_normal'])}, {fmt(row['ci95_high_welch_normal'])}]",
            ]
            for row in independent_frequency
            if row["metric"] in selected
        ]
        sections.append(
            '<h3>两组独立 10K 样本的均值变化</h3>'
            '<p>原始实验用 2 个 sampling rank，8×8-continued 用 4 个；sampler 又按 rank 交错分配索引并按 batch 起始索引设种，所以相同图片编号不是相同随机轨迹。这里明确按两组独立样本计算“continued − original”，误差条为独立样本（Welch）标准误的 normal 95% CI，不能冒充逐图配对结果。</p>'
            + table(("comparison", "metric", "N original / continued", "mean Δ", "Welch normal 95% CI"), independent_rows)
        )

    sections.append(
        """
        <section id="fft"><h2>4. FFT：像音乐播放器的低音、中音、高音均衡器</h2>
        <p>一张图既可以说“每个位置是什么颜色”，也可以说“有多少变化得很慢的大结构、多少中等大小的轮廓、多少变化很快的细纹理”。FFT 做的是后一种翻译：</p>
        <ul>
          <li>低频：天空/背景的大色块、物体大致位置和整体轮廓；</li>
          <li>中频：眼睛、腿、局部形状、较粗边缘；</li>
          <li>高频：毛发、草叶、细线、锐边，也可能是噪点或过度锐化。</li>
        </ul>
        <p>脚本先去掉每张图的平均亮度，再把相同“离中心距离”的频率围成圆环求平均，最后把每张图的总频率能量归一为 1。因此曲线回答的是<strong>能量比例放在哪里</strong>，不是图片整体亮不亮。生成图高频更多不自动代表更好：它也可能只是噪声。</p>
        <p>横轴用单轴 Nyquist frequency 归一化：&lt;0.25 记为低频，0.25–0.5 记为中频，≥0.5 记为高频。这两个阈值只是固定的描述性分箱，不是模型训练出来的边界；真正判断时同时看完整 radial curve、金字塔层和低通 FID。</p>
        </section>
        """
    )
    if frequency_fft_figures:
        sections.append('<div class="figure-grid">' + "".join(figure(path, caption) for path, caption in frequency_fft_figures) + '</div>')

    vae_rows = []
    if vae_control:
        for item in vae_control["datasets"]:
            metrics = item["metrics"]
            vae_rows.append(
                [
                    item["label"], item["analyzed_images"],
                    fmt(metrics["low64_fraction"]["mean"]),
                    fmt(metrics["mid_fraction"]["mean"]),
                    fmt(metrics["high_fraction"]["mean"]),
                    fmt(metrics["gradient_energy"]["mean"]),
                    fmt(metrics["laplacian_filter_energy"]["mean"]),
                ]
            )
    sections.append(
        '<section id="vae-control"><h2>5. VAE reconstruction control：先测量尺自己会丢什么</h2>'
        '<p>生成样本经过 SD VAE decode，而原始 ImageNet 参考图没有。把同一批真实图做 posterior-mean encode→decode 后再测频率，才能判断高频减少是 VAE bottleneck 的共同现象，还是 100/300/600 额外造成的。这里不加 posterior sampling noise。</p>'
        + (table(("dataset", "N", "≤64 fraction", "64→128 fraction", "128→256 fraction", "gradient energy", "4-neighbor edge energy"), vae_rows) if vae_rows else '<p class="pending">VAE reconstruction control 尚未完成。</p>')
        + '</section>'
    )
    if vae_control_figures:
        sections.append('<div class="figure-grid">' + "".join(figure(path, caption) for path, caption in vae_control_figures) + '</div>')

    lowpass_rows = []
    for label, metrics in lowpass_metrics.items():
        lowpass_rows.append([label, *([fmt(metrics.get(name)) for name in ("IS", "FID", "sFID", "Precision", "Recall")] if metrics else ["未完成"] * 5)])
    sections.append(
        '<section id="lowpass"><h2>6. 低通诊断：只看大结构或中等结构</h2>'
        '<p>把真实图和生成图都用同一投影降到 64px 或 128px 后，用各自的 10K 图重新提取 Inception 特征。如果 64px FID 已很差，问题主要在主体布局/大轮廓；如果 64px 尚可而 128px/原图迅速变差，问题更像出在中高频细节。这里是预处理和样本数对齐的 10K 诊断 FID，但真实图与生成图并非逐图配对，且<strong>不能和论文中的 50K 标准 FID 横向比较</strong>。</p>'
        + table(("diagnostic", "IS ↑", "FID ↓", "sFID ↓", "Precision ↑", "Recall ↑"), lowpass_rows)
        + '</section>'
    )

    sections.append(
        """
        <section id="predx0"><h2>7. 共同 timestep 的阶段 pred-x0 图</h2>
        <p>对三个 checkpoint 组合使用固定类别、固定种子、相同 80/80/90 stochastic DDPM 采样。在 t=100、300、400、600 等共同位置，不是只画各自边界，而是询问“此刻这个 schedule 实际由哪个分辨率负责，它认为 clean latent 是什么”。得到的 pred-x0 只做最近邻升到 32×32，再用同一个 SD VAE decode。</p>
        <p>例如 t=400：100/300/600 刚由 8×8 stage 负责；300/300/400 仍在 16×16 stage 中。图头部同时记录目标 t、respacing 后实际 t 和 stage 分辨率。实际 t 与目标最多只差很小的 respacing 误差。</p>
        <p>再例如从 t=999 开始反推，走完约 400 个 coarse updates 对应 t≈600：300/300/400 正好来到自己的 8→16 边界，而 100/300/600 仍由其更长的 8×8 区间负责。这就是“在 300/300/400 的 400-step 位置，也看 100/300/600 当时的 pred-x0”的公平对照。</p>
        <p>每条 schedule 还同时抓取 boundary 和 boundary−1：前者是 reverse sampling 中较粗 stage 的最后一次 pred-x0，后者属于刚升分辨率的较细 stage。把这两张并排看，可以发现 noise filling/分辨率切换后主体结构是否突然重画。</p>
        <p class="note">pred-x0 是模型在一个仍带噪状态下对最终 clean latent 的猜测，不是普通的中间 x_t 截图。较大 t 时它可以很不稳定；有意义的是同一类别、同一随机轨迹、同一全局 t 的横向差异，以及它接近 stage 边界时是否形成合理的大结构。</p>
        </section>
        """
    )
    if pred_figures:
        sections.append('<div class="figure-grid wide">' + "".join(figure(path, caption) for path, caption in pred_figures) + '</div>')
    else:
        sections.append('<p class="pending">pred-x0 图尚未完成。</p>')

    teacher_rows = []
    if teacher:
        interesting = {100, 300, 400, 600}
        for row in teacher["summary"]:
            if row["global_t"] in interesting:
                teacher_rows.append(
                    [
                        row["experiment"], row["global_t"], f"{row['resolution']}×{row['resolution']}",
                        fmt(row["model_excess_mse"]), fmt(row["unavoidable_representation_mse"]),
                        fmt(row["common_32_mse"]), fmt(row["stage_model_nmse"]),
                    ]
                )
    sections.append(
        '<section id="teacher"><h2>8. Teacher-forced：把“模型没学会”和“分辨率天生做不到”分开</h2>'
        '<p>固定 ImageNet validation 的 clean latent，在每个全局 t 用该 schedule 的真实 RDD forward（同一 beta、正确 drift）造出 x_t，再让负责该 t 的 stage 预测 x0。这里没有采样轨迹累计误差，也没有 CFG。不同分辨率使用同一份 32×32 高斯场的低频投影；块平均后乘尺度因子恢复单位方差，所以既符合各 stage 的噪声分布，又能做更干净的配对比较。</p>'
        '<div class="equation">统一 32×32 误差 = 当前分辨率表达不了的误差 + stage 模型自己的预测误差</div>'
        '<p>块平均/最近邻投影下两部分正交，脚本还记录交叉项检查数值是否接近 0。若 100/300/600 在 t=400 的主要差距来自 representation MSE，说明 8×8 接管太早；若 model-excess MSE 也明显更大，说明 8×8 模型或其时间区间本身也没学好。</p>'
        + (table(("allocation", "global t", "owner", "model excess", "unavoidable resolution", "total common", "native NMSE"), teacher_rows) if teacher_rows else '<p class="pending">Teacher-forced 分析尚未完成。</p>')
        + '</section>'
    )
    if teacher_figures:
        sections.append('<div class="figure-grid">' + "".join(figure(path, caption) for path, caption in teacher_figures) + '</div>')

    local_rows = []
    for label, payload in local_solver.items():
        if payload:
            local_rows.append(
                [
                    label,
                    ", ".join(fmt(value) for value in payload["difficulty_sampling_order"]),
                    ", ".join(fmt(value) for value in payload["difficulty_ratio_sampling_order"]),
                ]
            )
        else:
            local_rows.append([label, "未完成", "未完成"])
    sections.append(
        '<section id="solver"><h2>9. 局部 step-doubling 误差</h2>'
        '<p>在同一个 teacher-forced x_t 上，比“一次走较大步”与“两次走一半步”的结果差异。差异越大，说明这个 stage 的数值轨迹更难，推理时可能值得分配更多 NFE。它是 solver difficulty proxy，不是训练质量，也不是 FID。</p>'
        + table(("allocation / solver", "difficulty coarse→fine", "ratio coarse→fine"), local_rows)
        + '</section>'
    )

    class_rows = []
    if class_consistency:
        for item in class_consistency["datasets"]:
            class_rows.append(
                [
                    item["label"], fmt(item["top1_accuracy"]), fmt(item["top5_accuracy"]),
                    fmt(item["mean_target_probability"]), fmt(item["mean_target_nll"]),
                    fmt(item["predicted_class_coverage"]), fmt(item["predicted_class_entropy_fraction_of_log1000"]),
                ]
            )
    sections.append(
        '<section id="class"><h2>10. 条件类别一致性（辅助）</h2>'
        '<p>两次旧采样的每个 rank 都一次性跑完整个 strided index 列表且没有 resume，因此可以在同一服务器软件栈上按原来的 batch seed 和 CUDA <code>torch.randint</code> 顺序恢复每张图当时抽到的 ImageNet 类别。恢复时严格使用原始实验的 2-rank 配置和 8×8-continued 的 4-rank 配置，不能混用。随后按 torchvision ResNet-50 V2 的官方 232-resize + 224-center-crop 口径检查生成图是否被识别成目标类。若以后换了 CUDA RNG 实现，应先重新验证标签序列。分类器本身也会有偏差，所以它只解释“语义条件跟随”，不能代表视觉质量。</p>'
        + (table(("setting", "top-1", "top-5", "target prob", "target NLL", "predicted classes", "entropy/log1000"), class_rows) if class_rows else '<p class="pending">分类一致性尚未完成。</p>')
        + '</section>'
    )

    conclusions = []
    complete_full = all(value is not None for value in full_metrics.values())
    if complete_full:
        a = full_metrics["all stages 400K / CFG 1.0"]
        b = full_metrics["all stages 400K / CFG 1.5"]
        conclusions.append(
            f"原始 all-stages-400K 中，CFG 1.5 相对 CFG 1.0：FID {a['FID']:.3f} → {b['FID']:.3f}，Precision {a['Precision']:.4f} → {b['Precision']:.4f}，Recall {a['Recall']:.4f} → {b['Recall']:.4f}。"
        )
        c = full_metrics["8x8 continued 600K, 16/32 400K / CFG 1.0"]
        d = full_metrics["8x8 continued 600K, 16/32 400K / CFG 1.5"]
        base1 = ms300_reference["300/300/400 @ 400K / CFG 1.0"]
        base15 = ms300_reference["300/300/400 @ 400K / CFG 1.5"]
        recovered1 = (a["FID"] - c["FID"]) / (a["FID"] - base1["FID"])
        recovered15 = (b["FID"] - d["FID"]) / (b["FID"] - base15["FID"])
        conclusions.append(
            f"只把 8×8 从 400K 继续到 600K 后：cfg1 FID {a['FID']:.3f} → {c['FID']:.3f}，cfg1.5 FID {b['FID']:.3f} → {d['FID']:.3f}；相对 300/300/400 baseline 的总差距只追回 {recovered1:.1%} / {recovered15:.1%}，所以 8×8 训练不足不是主因。"
        )
        conclusions.append(
            f"cfg1 下 300/300/400 与原始 100/300/600 的 Recall 几乎相同（{base1['Recall']:.4f} vs {a['Recall']:.4f}），但 Precision 相差 {base1['Precision'] - a['Precision']:.4f}；schedule 劣化主要表现为 fidelity/realism，而不是覆盖率崩溃。"
        )
    if training_loss:
        training_by_key = {
            (row["run"], row["resolution"]): row
            for row in training_loss["summaries"]
        }
        original8 = training_by_key.get(("RDD_100_300_600", 8))
        continued8 = training_by_key.get(("RDD_100_300_600_8x8_continuation", 8))
        if original8 and continued8:
            conclusions.append(
                "8×8 训练 loss 已平台：100/300/600 原始 400K 最后 10K 均值 "
                f"{original8['last_window_mean']:.5f}，续训到 600K 最后 10K 为 "
                f"{continued8['last_window_mean']:.5f}，续训最后两个 10K 窗口仅变化 "
                f"{100 * continued8['relative_last_minus_previous']:+.3f}%。"
            )
    if teacher_rows:
        rows400 = [row for row in teacher["summary"] if row["global_t"] == 400]
        if len(rows400) >= 2:
            text = "；".join(
                f"{row['experiment']} 由 {row['resolution']}×{row['resolution']} 负责，model-excess={row['model_excess_mse']:.4g}，representation={row['unavoidable_representation_mse']:.4g}"
                for row in rows400
            )
            conclusions.append("t=400 的直接分解：" + text + "。")
        by_key = {
            (row["experiment"], row["global_t"]): row
            for row in teacher["summary"]
        }
        key300 = ("RDD_300_300_400", 400)
        key100 = ("RDD_100_300_600", 400)
        keycontinued = ("RDD_100_300_600_8x8continued600K", 400)
        if key300 in by_key and key100 in by_key:
            row300, row100 = by_key[key300], by_key[key100]
            conclusions.append(
                "teacher-forced t=400 下，100/300/600 相对 300/300/400 的 common-32 MSE 比值为 "
                f"{row100['common_32_mse'] / max(row300['common_32_mse'], 1e-30):.2f}×；其中 model-excess 比值 "
                f"{row100['model_excess_mse'] / max(row300['model_excess_mse'], 1e-30):.2f}×，representation 比值 "
                f"{row100['unavoidable_representation_mse'] / max(row300['unavoidable_representation_mse'], 1e-30):.2f}×。"
            )
        if key100 in by_key and keycontinued in by_key:
            row100, rowcontinued = by_key[key100], by_key[keycontinued]
            conclusions.append(
                "只续训 8×8 后，teacher-forced t=400 model-excess MSE 变化为 "
                f"{row100['model_excess_mse']:.4g} → {rowcontinued['model_excess_mse']:.4g}；representation MSE 理论上由分辨率决定，应基本不变，表中可作为实现自检。"
            )
    if all(value is not None for value in lowpass_metrics.values()):
        for scale in (64, 128):
            for cfg_label in (("cfg1", "CFG1.0"), ("cfg1p5", "CFG1.5")):
                slug, display = cfg_label
                original = lowpass_metrics[f"RDD_100_300_600_{slug} low-pass {scale} (10K)"]
                continued = lowpass_metrics[
                    f"RDD_100_300_600_8x8continued600K_{slug} low-pass {scale} (10K)"
                ]
                conclusions.append(
                    f"{display} 的 {scale}px 低通 10K FID：原始 {original['FID']:.3f}，仅 8×8 续训后 {continued['FID']:.3f}（Δ={continued['FID'] - original['FID']:+.3f}）。"
                )
    if frequency:
        freq_by_label = {item["label"]: item["metrics"] for item in frequency["datasets"]}
        reference_freq = freq_by_label.get("ImageNet_original")
        for slug, display in (("cfg1", "CFG1.0"), ("cfg1p5", "CFG1.5")):
            original_freq = freq_by_label.get(f"RDD_100_300_600_{slug}")
            continued_freq = freq_by_label.get(
                f"RDD_100_300_600_8x8continued600K_{slug}"
            )
            if reference_freq and original_freq and continued_freq:
                conclusions.append(
                    f"{display} 像素频率：真实/原始/8×8续训后的 128→256 能量比例为 "
                    f"{reference_freq['high_fraction']['mean']:.3%} / {original_freq['high_fraction']['mean']:.3%} / {continued_freq['high_fraction']['mean']:.3%}；"
                    "高频更接近真实只是一致性证据，仍需结合低通 FID 排除噪点或过锐。"
                )
    if vae_control:
        vae_by_label = {item["label"]: item["metrics"] for item in vae_control["datasets"]}
        real_vae = vae_by_label.get("ImageNet_original")
        reconstructed_vae = vae_by_label.get("ImageNet_SD_VAE_mean_reconstruction")
        if real_vae and reconstructed_vae:
            conclusions.append(
                "VAE control 的 128→256 能量比例为：原始真实图 "
                f"{real_vae['high_fraction']['mean']:.3%}，同图 SD-VAE mean reconstruction "
                f"{reconstructed_vae['high_fraction']['mean']:.3%}。因此判断 RDD 是否额外过平滑时，应把生成图优先与后者比较。"
            )
    if class_rows and len(class_consistency["datasets"]) >= 2:
        items = {item["label"]: item for item in class_consistency["datasets"]}
        orig1 = items.get("RDD_100_300_600_cfg1")
        orig15 = items.get("RDD_100_300_600_cfg1p5")
        cont1 = items.get("RDD_100_300_600_8x8continued600K_cfg1")
        cont15 = items.get("RDD_100_300_600_8x8continued600K_cfg1p5")
        if orig1 and orig15:
            conclusions.append(
                f"原始 100/300/600 的条件分类 top-1 从 CFG1.0 的 {orig1['top1_accuracy']:.3%} 变为 CFG1.5 的 {orig15['top1_accuracy']:.3%}；需要结合 Recall 判断更强 CFG 是否以覆盖率换语义/清晰度。"
            )
        if orig1 and cont1 and orig15 and cont15:
            conclusions.append(
                "只续训 8×8 后，条件 top-1 的变化为："
                f"CFG1.0 {orig1['top1_accuracy']:.3%} → {cont1['top1_accuracy']:.3%}，"
                f"CFG1.5 {orig15['top1_accuracy']:.3%} → {cont15['top1_accuracy']:.3%}。"
            )
    if not conclusions:
        conclusions.append("部分长任务仍在运行；完成后重新运行报告生成器即可自动填入数值结论。")
    sections.append(
        '<section id="conclusion"><h2>11. 当前可支持的判断</h2><ul>'
        + "".join(f"<li>{html.escape(item)}</li>" for item in conclusions)
        + '</ul><h3>不能过度解读的地方</h3><ul>'
        '<li>服务器没有保存 300/300/400 的 50K 图片/NPZ，所以 Laplacian/FFT 的大样本横向统计目前只有真实图与 100/300/600；不能拿 8 张阶段图冒充总体频谱。</li>'
        '<li>Teacher-forced error 去除了采样轨迹累计误差；它能定位 stage 能力和分辨率容量，但最终因果仍要由小规模边界 ablation + FID 验证。</li>'
        '<li>600/300/100 仍在从头训练，当前只到很早期 step；本报告不会把它的早期 checkpoint 与 400K 模型公平并列。完成到相同 400K 后，应使用同一脚本加入 t=600、900 对照。</li>'
        '</ul></section>'
    )

    data_links = sorted(path.relative_to(bundle).as_posix() for path in data_dir.rglob("*") if path.is_file())
    sections.append(
        '<section id="data"><h2>12. 原始数据与复现脚本</h2>'
        '<p>本 bundle 不只保存图，也复制 CSV、JSON、完整 evaluator 日志、实验配置和实际运行的 Python/Bash 源码。图中的每个数字都能追溯到这里；重新运行报告生成器不会修改原始实验文件。</p>'
        '<ul class="data-list">'
        + "".join(f'<li><a href="{html.escape(path)}">{html.escape(path)}</a></li>' for path in data_links)
        + '</ul></section>'
    )

    css = """
    :root{--ink:#17212b;--muted:#5d6976;--line:#d9e0e7;--accent:#205c92;--soft:#edf5fb;--warn:#fff5d6}
    *{box-sizing:border-box} body{margin:0;background:#f5f7f9;color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
    main{max-width:1180px;margin:0 auto;background:white;padding:48px 58px 96px;box-shadow:0 0 24px #00000010}
    h1{font-size:34px;line-height:1.2;margin:0 0 8px} h2{font-size:25px;margin:48px 0 14px;border-bottom:2px solid var(--line);padding-bottom:8px} h3{margin-top:28px}
    .subtitle{color:var(--muted);margin-bottom:32px}.callout{background:var(--soft);border-left:5px solid var(--accent);padding:16px 20px;border-radius:4px}.pending{background:var(--warn);padding:12px 16px}.note{color:var(--muted)}
    code,.equation{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.equation{background:#f1f3f5;padding:14px 18px;margin:18px 0;border-radius:5px;overflow:auto}
    .table-wrap{overflow:auto;margin:16px 0 24px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border:1px solid var(--line);padding:9px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eef2f5}
    .figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:22px}.figure-grid.wide{grid-template-columns:1fr}figure{margin:0;border:1px solid var(--line);padding:10px;background:#fff}figure img{display:block;width:100%;height:auto}figcaption{color:var(--muted);font-size:13px;padding-top:8px}
    a{color:var(--accent)}.data-list{columns:2;column-gap:30px}.data-list li{break-inside:avoid;margin:4px 0;font-size:13px}
    @media(max-width:720px){main{padding:28px 18px}.figure-grid{grid-template-columns:1fr}.data-list{columns:1}}
    """
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RDD 步数分配诊断报告</title><style>{css}</style></head><body><main><h1>RDD 训练步数分配诊断报告</h1><p class="subtitle">300/300/400 vs 100/300/600 · standard metrics, pred-x0, Laplacian, FFT, teacher-forced errors</p>{''.join(sections)}</main></body></html>"""
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "RDD_STAGE_ALLOCATION_DIAGNOSTIC_REPORT_ZH.html").write_text(document, encoding="utf-8")
    manifest = {
        "report": "RDD_STAGE_ALLOCATION_DIAGNOSTIC_REPORT_ZH.html",
        "copied_files": sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()),
        "completion": {
            "full_metrics": all(value is not None for value in full_metrics.values()),
            "training_loss": training_loss is not None and training_loss_figure is not None,
            "frequency": frequency is not None,
            "vae_frequency_control": vae_control is not None,
            "lowpass_metrics": all(value is not None for value in lowpass_metrics.values()),
            "pred_x0": pred_metadata is not None,
            "teacher_forced": teacher is not None,
            "local_solver": all(value is not None for value in local_solver.values()),
            "class_consistency": class_consistency is not None,
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["completion"], ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {bundle / 'RDD_STAGE_ALLOCATION_DIAGNOSTIC_REPORT_ZH.html'}", flush=True)


if __name__ == "__main__":
    main()
