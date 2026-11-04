#!/usr/bin/env python3
"""Visualize multistage RDD clean-latent predictions at matched global times.

For every requested global timestep, this script records ``pred_xstart`` from
the stage that owns that timestep, nearest-neighbor upsamples the latent to
32x32, and decodes it with the same SD VAE used by the FID sampler.  It also
builds side-by-side grids for different training allocations using fixed class
labels and seeds.

This is a small qualitative diagnostic (default eight images), not a 50K FID
sampling job.  Sampling itself deliberately follows ``sample_fid_unified.py``:
stochastic DDPM jump steps, 80/80/90 coarse-to-fine updates, noise filling,
float32 model weights, and the historical CFG path.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


THIS_FILE = Path(__file__).resolve()
TWO_D_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(TWO_D_DIR))
import sample_fid_unified as sampler  # noqa: E402

from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


DEFAULT_CLASSES = [207, 360, 387, 954, 130, 25, 980, 933]
DEFAULT_ARCHITECTURES = {8: "DiT-B/2", 16: "DiT-L/2", 32: "DiT-XL/2"}


def _resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _cfg_tag(value: float) -> str:
    return (f"cfg{value:g}").replace(".", "p")


def _decode_latents(vae, latents: torch.Tensor, device: str, chunk: int = 4) -> np.ndarray:
    decoded = []
    upsampled = F.interpolate(latents.float(), size=(32, 32), mode="nearest")
    with torch.no_grad():
        for start in range(0, len(upsampled), chunk):
            x = upsampled[start : start + chunk].to(device)
            image = vae.decode(x / 0.18215).sample
            decoded.append(
                ((image.clamp(-1, 1) + 1.0) * 127.5)
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            del x, image
    return np.concatenate(decoded, axis=0)


def _labeled_grid(images: np.ndarray, title: str, output: Path, columns: int = 4) -> None:
    n = len(images)
    columns = min(columns, n)
    rows = (n + columns - 1) // columns
    tile = 256
    header = 42
    canvas = Image.new("RGB", (columns * tile, header + rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 12), title, fill="black")
    for i, array in enumerate(images):
        x = (i % columns) * tile
        y = header + (i // columns) * tile
        canvas.paste(Image.fromarray(array), (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _comparison_grid(
    entries: Sequence[Tuple[str, Mapping, np.ndarray]],
    class_ids: Sequence[int],
    title: str,
    output: Path,
) -> None:
    tile = 256
    left = 82
    header = 76
    rows = len(class_ids)
    cols = len(entries)
    canvas = Image.new("RGB", (left + cols * tile, header + rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for col, (label, meta, _) in enumerate(entries):
        stage = meta["resolution"]
        actual = meta["actual_global_t"]
        requested = meta["requested_global_t"]
        text = f"{label}\nstage {stage}x{stage}, actual t={actual} (target {requested})"
        draw.multiline_text((left + col * tile + 5, 32), text, fill="black", spacing=2)
    for row, class_id in enumerate(class_ids):
        draw.text((8, header + row * tile + 10), f"class\n{class_id}", fill="black")
        for col, (_, _, images) in enumerate(entries):
            canvas.paste(Image.fromarray(images[row]), (left + col * tile, header + row * tile))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _build_diffusions(stage_lengths: Sequence[int], num_steps: Sequence[int]):
    resolutions = [32, 16, 8]
    boundaries = [0]
    for length in stage_lengths:
        boundaries.append(boundaries[-1] + int(length))
    finest_first = []
    for i, resolution in enumerate(resolutions):
        sampling_position = 2 - i
        finest_first.append(
            create_diffusion_with_drift(
                str(num_steps[sampling_position]),
                c=i if i < 2 else None,
                T_c=boundaries[i],
                T_c_plus_1=boundaries[i + 1],
            )
        )
    return list(reversed(finest_first)), boundaries


def _load_models(experiment: Mapping, device: str):
    ckpt_dir = _resolve(experiment["ckpt_dir"])
    step = str(experiment.get("ckpt_step", "0400000"))
    models = []
    num_classes = None
    for resolution in (8, 16, 32):
        architecture = experiment.get(f"model_{resolution}", DEFAULT_ARCHITECTURES[resolution])
        checkpoint = ckpt_dir / f"checkpoints_{resolution}x{resolution}" / f"{step}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[load] {experiment['label']} {resolution}x{resolution}: {checkpoint}", flush=True)
        model, classes = sampler._load_dit_model(
            str(checkpoint), architecture, resolution, device, torch.float32
        )
        if num_classes is None:
            num_classes = classes
        elif num_classes != classes:
            raise ValueError(f"class-count mismatch at {resolution}: {classes} != {num_classes}")
        models.append(model)
    return models, int(num_classes)


def _capture_one_cfg(
    experiment: Mapping,
    diffusions,
    boundaries: Sequence[int],
    models,
    num_classes: int,
    probe_times: Sequence[int],
    num_steps: Sequence[int],
    class_ids: Sequence[int],
    cfg_scale: float,
    seed: int,
    device: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, dict]]:
    n = len(class_ids)
    use_cfg = cfg_scale > 1.0
    labels = torch.tensor(class_ids, dtype=torch.long, device=device)
    if use_cfg:
        null = torch.full((n,), num_classes, dtype=torch.long, device=device)
        model_kwargs = {"y": torch.cat([labels, null]), "cfg_scale": cfg_scale}
        batch_mult = 2
        forwards = [model.forward_with_cfg for model in models]
    else:
        model_kwargs = {"y": labels}
        batch_mult = 1
        forwards = [model.forward for model in models]

    torch.manual_seed(seed)
    captures: Dict[int, torch.Tensor] = {}
    metadata: Dict[int, dict] = {}
    pred_xstart = None
    resolutions_sampling = [8, 16, 32]
    for stage_index, (diffusion, model_forward, resolution, steps) in enumerate(
        zip(diffusions, forwards, resolutions_sampling, num_steps)
    ):
        finest_index = 2 - stage_index
        low = int(boundaries[finest_index])
        high = int(boundaries[finest_index + 1])
        stage_targets = [int(t) for t in probe_times if low <= int(t) < high]
        actual_to_targets: Dict[int, List[int]] = {}
        for target in stage_targets:
            actual = min(diffusion.timestep_map, key=lambda value: (abs(int(value) - target), int(value)))
            actual_to_targets.setdefault(int(actual), []).append(target)

        if stage_index == 0:
            stage_input = torch.randn(batch_mult * n, 4, resolution, resolution, device=device)
        else:
            t_bound = torch.full((n,), steps - 1, dtype=torch.long, device=device)
            stage_input = sampler._do_upsample(
                pred_xstart, t_bound, diffusion, resolution, "noise_filling"
            )
            if use_cfg:
                stage_input = torch.cat([stage_input, stage_input], dim=0)

        print(
            f"[sample] {experiment['label']} {_cfg_tag(cfg_scale)} stage={resolution} "
            f"interval=[{low},{high}) mapped={diffusion.timestep_map[0]}..{diffusion.timestep_map[-1]}",
            flush=True,
        )
        generator = diffusion.jump_step_sample_loop_progressive(
            model_forward,
            stage_input.shape,
            noise=stage_input,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            deterministic=False,
        )
        last_out = None
        for local_index, out in zip(range(diffusion.num_timesteps - 1, -1, -1), generator):
            last_out = out
            global_t = int(diffusion.timestep_map[local_index])
            if global_t not in actual_to_targets:
                continue
            clean = out["pred_xstart"].float()[:n].detach().cpu()
            for target in actual_to_targets[global_t]:
                captures[target] = clean.clone()
                metadata[target] = {
                    "experiment": experiment["label"],
                    "cfg_scale": cfg_scale,
                    "requested_global_t": target,
                    "actual_global_t": global_t,
                    "absolute_t_error": abs(global_t - target),
                    "resolution": resolution,
                    "stage_interval": [low, high],
                    "local_spaced_index": local_index,
                    "available_spaced_steps": int(diffusion.num_timesteps),
                    "visualization": "pred_xstart nearest-upsampled to latent 32x32, then SD-VAE decoded",
                }
        if last_out is None:
            raise RuntimeError(f"stage {resolution} produced no outputs")
        pred_xstart = last_out["pred_xstart"].float()[:n]

    missing = sorted(set(probe_times) - set(captures))
    if missing:
        raise RuntimeError(f"failed to capture probe times {missing}")
    return captures, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON list of multistage experiments")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae-path", default="checkpoints/pretrained/stabilityai/sd-vae-ft-ema")
    parser.add_argument("--cfg-scales", default="1.0,1.5")
    parser.add_argument("--num-steps", default="80,80,90", help="coarse,middle,fine")
    parser.add_argument("--classes", default=",".join(str(v) for v in DEFAULT_CLASSES))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    experiments = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(experiments, list) or len(experiments) < 1:
        parser.error("config must be a non-empty JSON list")
    for experiment in experiments:
        lengths = [int(v) for v in experiment["stage_lengths"]]
        if len(lengths) != 3 or sum(lengths) != 1000:
            parser.error(f"{experiment.get('label')} stage_lengths must have three values summing to 1000")
    cfg_scales = [float(value) for value in args.cfg_scales.split(",")]
    num_steps = [int(value) for value in args.num_steps.split(",")]
    class_ids = [int(value) for value in args.classes.split(",")]
    if len(num_steps) != 3 or not class_ids:
        parser.error("num-steps needs three values and classes cannot be empty")

    probe_times = {0}
    for experiment in experiments:
        running = 0
        for length in experiment["stage_lengths"][:-1]:
            running += int(length)
            probe_times.add(running)
            # Capture both sides of every reverse-sampling transition:
            # t=boundary is the coarser stage's final clean prediction, while
            # t=boundary-1 belongs to the newly upsampled finer stage.
            probe_times.add(running - 1)
    probe_times = sorted(probe_times)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probes] global timesteps={probe_times}", flush=True)

    vae_path = _resolve(args.vae_path)
    print(f"[load] VAE: {vae_path}", flush=True)
    vae = AutoencoderKL.from_pretrained(str(vae_path)).to(args.device)
    vae.eval()

    decoded_by_key: Dict[Tuple[str, float, int], np.ndarray] = {}
    metadata_by_key: Dict[Tuple[str, float, int], dict] = {}
    all_metadata = {
        "purpose": "fair qualitative comparison of clean predictions at shared global timesteps",
        "sampler": "stochastic DDPM jump-step",
        "num_steps_coarse_middle_fine": num_steps,
        "upsample_mode_between_stages": "noise_filling",
        "pred_x0_display": "nearest latent upsample to 32x32 followed by SD-VAE decode",
        "seed": args.seed,
        "class_ids": class_ids,
        "probe_times": probe_times,
        "boundary_pair_convention": {
            "boundary": "last pred-x0 owned by the coarser stage during reverse sampling",
            "boundary_minus_1": "first-side pred-x0 owned by the newly upsampled finer stage",
        },
        "experiments": experiments,
        "captures": [],
    }

    for experiment in experiments:
        label = experiment["label"]
        diffusions, boundaries = _build_diffusions(experiment["stage_lengths"], num_steps)
        models, num_classes = _load_models(experiment, args.device)
        if any(value < 0 or value >= num_classes for value in class_ids):
            raise ValueError(f"class IDs must be in [0,{num_classes}), got {class_ids}")
        for cfg_scale in cfg_scales:
            captures, capture_meta = _capture_one_cfg(
                experiment=experiment,
                diffusions=diffusions,
                boundaries=boundaries,
                models=models,
                num_classes=num_classes,
                probe_times=probe_times,
                num_steps=num_steps,
                class_ids=class_ids,
                cfg_scale=cfg_scale,
                seed=args.seed,
                device=args.device,
            )
            for target in probe_times:
                images = _decode_latents(vae, captures[target], args.device)
                meta = capture_meta[target]
                key = (label, cfg_scale, target)
                decoded_by_key[key] = images
                metadata_by_key[key] = meta
                all_metadata["captures"].append(meta)
                capture_dir = output_dir / label / _cfg_tag(cfg_scale) / f"target_t{target:04d}"
                capture_dir.mkdir(parents=True, exist_ok=True)
                for i, image in enumerate(images):
                    Image.fromarray(image).save(capture_dir / f"class_{class_ids[i]:04d}.png")
                _labeled_grid(
                    images,
                    (
                        f"{label}, cfg={cfg_scale:g}, target t={target}, "
                        f"actual t={meta['actual_global_t']}, stage={meta['resolution']}x{meta['resolution']}"
                    ),
                    capture_dir / "grid.png",
                )
                (capture_dir / "metadata.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                del captures[target]
            torch.cuda.empty_cache()
        del models, diffusions
        gc.collect()
        torch.cuda.empty_cache()

    comparison_dir = output_dir / "comparisons"
    for cfg_scale in cfg_scales:
        for target in probe_times:
            entries = []
            for experiment in experiments:
                key = (experiment["label"], cfg_scale, target)
                entries.append((experiment["label"], metadata_by_key[key], decoded_by_key[key]))
            _comparison_grid(
                entries,
                class_ids,
                f"Matched global timestep t={target}, cfg={cfg_scale:g}",
                comparison_dir / f"{_cfg_tag(cfg_scale)}_target_t{target:04d}.png",
            )

    (output_dir / "metadata.json").write_text(
        json.dumps(all_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
