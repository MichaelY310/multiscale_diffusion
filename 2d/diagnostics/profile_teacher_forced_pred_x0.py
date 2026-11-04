#!/usr/bin/env python3
"""Compare RDD allocations with teacher-forced clean-prediction errors.

For fixed ImageNet validation latents and fixed global timesteps, each schedule
uses the model that owns that timestep.  The script constructs the *actual RDD
forward state* (including drift), predicts x0 without CFG, and separates:

1. stage-model estimation error at its native resolution;
2. unavoidable representation error from using 8x8/16x16 instead of 32x32;
3. common 32x32 error in the <=8, 8->16, and 16->32 latent bands.

This is a small diagnostic and is not an FID substitute.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
TWO_D_DIR = THIS_FILE.parents[1]
sys.path.insert(0, str(TWO_D_DIR))
import sample_fid_unified as fid_sampler  # noqa: E402

from diffusion import create_diffusion_with_drift  # noqa: E402


PIL_RESAMPLING = getattr(Image, "Resampling", Image)
DEFAULT_ARCHITECTURES = {8: "DiT-B/2", 16: "DiT-L/2", 32: "DiT-XL/2"}
METRICS = (
    "stage_model_mse",
    "stage_model_nmse",
    "common_32_mse",
    "unavoidable_representation_mse",
    "model_excess_mse",
    "orthogonality_cross_term",
    "common_low8_mse",
    "common_mid8to16_mse",
    "common_high16to32_mse",
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def center_crop_arr(pil_image: Image.Image, image_size: int = 256) -> Image.Image:
    pil_image = pil_image.convert("RGB")
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(value // 2 for value in pil_image.size),
            resample=PIL_RESAMPLING.BOX,
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(value * scale) for value in pil_image.size),
        resample=PIL_RESAMPLING.BICUBIC,
    )
    array = np.asarray(pil_image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class ImageNetValidation(Dataset):
    """Official ILSVRC validation images with LOC CSV labels."""

    def __init__(self, imagenet_root: Path):
        self.image_dir = imagenet_root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
        solution = imagenet_root / "LOC_val_solution.csv"
        mapping = imagenet_root / "LOC_synset_mapping.txt"
        train_dir = imagenet_root / "ILSVRC" / "Data" / "CLS-LOC" / "train"
        if not self.image_dir.is_dir() or not solution.is_file() or not mapping.is_file():
            raise FileNotFoundError(
                f"expected validation images and LOC metadata below {imagenet_root}"
            )
        synsets = [line.split(maxsplit=1)[0] for line in mapping.read_text(encoding="utf-8").splitlines() if line]
        if not train_dir.is_dir():
            raise FileNotFoundError(f"missing ImageNet training directory used for class-order audit: {train_dir}")
        imagefolder_synsets = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
        if len(synsets) != len(imagefolder_synsets) or set(synsets) != set(imagefolder_synsets):
            mismatch = next(
                (
                    (index, mapped, folder)
                    for index, (mapped, folder) in enumerate(zip(synsets, imagefolder_synsets))
                    if mapped != folder
                ),
                None,
            )
            raise ValueError(
                "LOC_synset_mapping classes do not match ImageFolder/DiT classes: "
                f"mapping={len(synsets)}, folders={len(imagefolder_synsets)}, first_mismatch={mismatch}"
            )
        if synsets != imagefolder_synsets:
            print(
                "[class-order] LOC mapping order differs from ImageFolder; remapping validation wnids "
                "to the sorted training-directory order used by the checkpoints",
                flush=True,
            )
        synset_to_class = {
            synset: index for index, synset in enumerate(imagefolder_synsets)
        }
        labels = {}
        with solution.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                synset = row["PredictionString"].split()[0]
                labels[row["ImageId"]] = synset_to_class[synset]
        self.paths = sorted(self.image_dir.glob("*.JPEG"))
        if len(self.paths) != len(labels):
            raise ValueError(f"found {len(self.paths)} images but {len(labels)} validation labels")
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Lambda(center_crop_arr),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, int(self.labels[path.stem]), index


def _project(latent: torch.Tensor, resolution: int) -> torch.Tensor:
    if latent.shape[-1] == resolution:
        return latent
    factor = latent.shape[-1] // resolution
    return F.avg_pool2d(latent, factor, factor)


def _project_unit_gaussian(noise32: torch.Tensor, resolution: int) -> torch.Tensor:
    """Project one shared 32x32 Gaussian field while preserving unit variance.

    Averaging an ``factor x factor`` block divides the variance by
    ``factor**2``.  Multiplication by ``factor`` restores variance one.  The
    result is therefore distributed exactly like the i.i.d. unit Gaussian
    expected by each native-resolution RDD forward process, while schedules
    that own the same global timestep share the same low-frequency noise.
    """
    if noise32.shape[-1] == resolution:
        return noise32
    factor = noise32.shape[-1] // resolution
    return F.avg_pool2d(noise32, factor, factor) * factor


def _upsample(latent: torch.Tensor, resolution: int = 32) -> torch.Tensor:
    if latent.shape[-1] == resolution:
        return latent
    factor = resolution // latent.shape[-1]
    return latent.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


def _per_image_mse(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=(1, 2, 3))


def _common_band_errors(error32: torch.Tensor):
    p8 = _upsample(_project(error32, 8), 32)
    p16 = _upsample(_project(error32, 16), 32)
    return _per_image_mse(p8), _per_image_mse(p16 - p8), _per_image_mse(error32 - p16)


def _group_summary(records: Sequence[dict]) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = {}
    for row in records:
        grouped.setdefault((row["experiment"], row["global_t"]), []).append(row)
    output = []
    for (experiment, global_t), rows in sorted(grouped.items()):
        item = {
            "experiment": experiment,
            "global_t": global_t,
            "resolution": rows[0]["resolution"],
            "stage_interval": rows[0]["stage_interval"],
            "n": len(rows),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            item[metric] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_sem"] = item[f"{metric}_std"] / math.sqrt(max(len(values), 1))
        output.append(item)
    return output


def _write_plots(summary: Sequence[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = list(dict.fromkeys(row["experiment"] for row in summary))
    plot_specs = (
        ("common_32_mse", "Common 32x32 clean-prediction MSE", "teacher_forced_common32_mse.png"),
        ("model_excess_mse", "Model estimation MSE (representation removed)", "teacher_forced_model_excess_mse.png"),
        ("unavoidable_representation_mse", "Unavoidable MSE from current resolution", "teacher_forced_representation_mse.png"),
        ("stage_model_nmse", "Native-stage normalized x0 MSE", "teacher_forced_stage_nmse.png"),
    )
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for experiment in experiments:
            rows = [row for row in summary if row["experiment"] == experiment]
            ax.plot(
                [row["global_t"] for row in rows],
                [max(row[metric], 1e-12) for row in rows],
                marker="o",
                linewidth=2,
                label=experiment,
            )
        ax.set_yscale("log")
        ax.set_xlabel("global diffusion timestep t")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    for experiment in experiments:
        rows = [row for row in summary if row["experiment"] == experiment]
        x = np.asarray([row["global_t"] for row in rows])
        low = np.asarray([row["common_low8_mse"] for row in rows])
        mid = np.asarray([row["common_mid8to16_mse"] for row in rows])
        high = np.asarray([row["common_high16to32_mse"] for row in rows])
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.stackplot(x, low, mid, high, labels=("<=8 latent", "8->16 detail", "16->32 detail"), alpha=0.85)
        ax.set_xlabel("global diffusion timestep t")
        ax.set_ylabel("common 32x32 x0 MSE by orthogonal band")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        fig.tight_layout()
        safe = experiment.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"teacher_forced_bands_{safe}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--imagenet-root", default="imagenet")
    parser.add_argument("--vae-path", default="checkpoints/pretrained/stabilityai/sd-vae-ft-ema")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--probe-times",
        default="0,50,99,100,150,250,299,300,350,399,400,450,550,599,600,650,750,850,950,999",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.num_images <= 0 or args.batch_size <= 0:
        parser.error("num-images and batch-size must be positive")

    experiments = json.loads(args.config.read_text(encoding="utf-8"))
    probe_times = sorted(set(int(value) for value in args.probe_times.split(",")))
    if not probe_times or probe_times[0] < 0 or probe_times[-1] > 999:
        parser.error("probe-times must lie in [0,999]")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ImageNetValidation(_resolve(args.imagenet_root))
    rng = np.random.default_rng(args.seed)
    subset_indices = np.sort(
        rng.choice(len(dataset), size=min(args.num_images, len(dataset)), replace=False)
    ).tolist()
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    vae = AutoencoderKL.from_pretrained(str(_resolve(args.vae_path)), local_files_only=True).to(device).eval()
    latent_batches = []
    print(f"[encode] {len(subset_indices)} ImageNet validation images", flush=True)
    with torch.no_grad():
        for images, labels, original_indices in loader:
            # Posterior mean removes irrelevant VAE sampling noise from this
            # model-comparison diagnostic.
            latent = vae.encode(images.to(device, non_blocking=True)).latent_dist.mean.mul(0.18215)
            latent_batches.append((latent.cpu(), labels.cpu(), original_indices.cpu()))
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    records: List[dict] = []
    for experiment in experiments:
        label = experiment["label"]
        lengths = [int(value) for value in experiment["stage_lengths"]]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        ckpt_dir = _resolve(experiment["ckpt_dir"])
        ckpt_step = str(experiment.get("ckpt_step", "0400000"))
        for finest_index, resolution in enumerate((32, 16, 8)):
            low, high = boundaries[finest_index], boundaries[finest_index + 1]
            stage_probes = [value for value in probe_times if low <= value < high]
            if not stage_probes:
                continue
            architecture = experiment.get(f"model_{resolution}", DEFAULT_ARCHITECTURES[resolution])
            checkpoint = ckpt_dir / f"checkpoints_{resolution}x{resolution}" / f"{ckpt_step}.pt"
            print(f"[load] {label} {resolution}x{resolution}: {checkpoint}", flush=True)
            model, num_classes = fid_sampler._load_dit_model(
                str(checkpoint), architecture, resolution, str(device), torch.float32
            )
            diffusion = create_diffusion_with_drift(
                "",
                diffusion_steps=1000,
                c=finest_index if finest_index < 2 else None,
                T_c=low,
                T_c_plus_1=high,
            )
            generators = {}
            for global_t in stage_probes:
                generators[global_t] = torch.Generator(device=device).manual_seed(
                    args.seed + 100_003 * global_t
                )
            with torch.no_grad():
                for clean32_cpu, labels_cpu, original_indices_cpu in latent_batches:
                    clean32 = clean32_cpu.to(device, non_blocking=True)
                    clean = _project(clean32, resolution)
                    labels = labels_cpu.to(device, non_blocking=True)
                    if labels.max().item() >= num_classes:
                        raise ValueError("validation label exceeds checkpoint class count")
                    if resolution > 8:
                        projected_lower = _upsample(_project(clean, resolution // 2), resolution)
                        drift = projected_lower - clean
                    else:
                        drift = None
                    clean_up = _upsample(clean, 32)
                    oracle_error = clean_up - clean32
                    for global_t in stage_probes:
                        local_t = global_t - low
                        timestep = torch.full(
                            (len(clean),), local_t, dtype=torch.long, device=device
                        )
                        shared_noise32 = torch.randn(
                            clean32.shape,
                            generator=generators[global_t],
                            device=device,
                            dtype=clean.dtype,
                        )
                        noise = _project_unit_gaussian(shared_noise32, resolution)
                        state = diffusion.q_sample(clean, timestep, d=drift, noise=noise)
                        prediction = diffusion.p_mean_variance(
                            model.forward,
                            state,
                            timestep,
                            clip_denoised=False,
                            model_kwargs={"y": labels},
                        )["pred_xstart"].float()
                        stage_error = prediction - clean
                        pred_up = _upsample(prediction, 32)
                        model_error_up = pred_up - clean_up
                        common_error = pred_up - clean32
                        low_error, mid_error, high_error = _common_band_errors(common_error)
                        values = {
                            "stage_model_mse": _per_image_mse(stage_error),
                            "stage_model_nmse": _per_image_mse(stage_error)
                            / torch.clamp(_per_image_mse(clean), min=1e-12),
                            "common_32_mse": _per_image_mse(common_error),
                            "unavoidable_representation_mse": _per_image_mse(oracle_error),
                            "model_excess_mse": _per_image_mse(model_error_up),
                            "orthogonality_cross_term": (
                                2.0 * (oracle_error * model_error_up).mean(dim=(1, 2, 3))
                            ),
                            "common_low8_mse": low_error,
                            "common_mid8to16_mse": mid_error,
                            "common_high16to32_mse": high_error,
                        }
                        cpu_values = {key: tensor.cpu().numpy() for key, tensor in values.items()}
                        for row_index in range(len(clean)):
                            row = {
                                "experiment": label,
                                "global_t": global_t,
                                "local_t": local_t,
                                "resolution": resolution,
                                "stage_interval": f"[{low},{high})",
                                "imagenet_val_index": int(original_indices_cpu[row_index]),
                                "class_id": int(labels_cpu[row_index]),
                            }
                            row.update({key: float(value[row_index]) for key, value in cpu_values.items()})
                            records.append(row)
            del model, diffusion
            gc.collect()
            torch.cuda.empty_cache()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "experiment", "global_t", "local_t", "resolution", "stage_interval",
        "imagenet_val_index", "class_id", *METRICS,
    )
    with (output_dir / "teacher_forced_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = _group_summary(records)
    with (output_dir / "teacher_forced_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        "status": "teacher_forced_diagnostic_not_fid",
        "dataset": "official ImageNet validation",
        "num_images": len(subset_indices),
        "subset_indices": subset_indices,
        "clean_latent": "SD-VAE posterior mean times 0.18215",
        "conditioning": "ground-truth class, no CFG",
        "forward_process": "RDD q_sample with the owning schedule's exact drift and beta interval",
        "noise_pairing": "one shared 32x32 Gaussian field per image/global-t, variance-preserving block projection at lower resolutions",
        "probe_times": probe_times,
        "experiments": experiments,
        "interpretation": {
            "model_excess_mse": "prediction error after removing resolution capacity",
            "unavoidable_representation_mse": "clean detail that current resolution cannot represent",
            "common_32_mse": "their orthogonal sum (up to numerical error)",
        },
        "summary": summary,
    }
    (output_dir / "teacher_forced_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_plots(summary, output_dir)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
