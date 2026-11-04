#!/usr/bin/env python3
"""Evaluate flat or RDD SLat generation on identical ground-truth supports.

This is the controlled geometry arm: structure/topology, image condition, model,
NFE, decoder, and metric implementation are fixed.  Only the SLat flow path is
changed.  It intentionally does not claim end-to-end structure generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from trellis2 import models, trainers
from trellis2.modules import sparse as sp
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler

from trellis2_rdd.rdd_sampler import SparseRDDDetailEulerCfgSampler


def condition_tensor(render_root: Path, asset: str, image_size: int) -> torch.Tensor:
    root = render_root / asset
    metadata = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    image = Image.open(root / metadata["frames"][0]["file_path"]).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    nonzero = np.nonzero(alpha)
    if not len(nonzero[0]):
        raise ValueError(f"empty alpha mask for {asset}")
    x0, x1 = int(nonzero[1].min()), int(nonzero[1].max())
    y0, y1 = int(nonzero[0].min()), int(nonzero[0].max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    image = image.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    rgba = np.asarray(image).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4]
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def sparse_noise(coords: torch.Tensor, channels: int) -> sp.SparseTensor:
    batch = torch.zeros((len(coords), 1), dtype=torch.int32, device=coords.device)
    return sp.SparseTensor(
        feats=torch.randn((len(coords), channels), device=coords.device),
        coords=torch.cat((batch, coords.int()), dim=1),
    )


def mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    return (
        mesh.vertices.detach().cpu().numpy().astype(np.float64),
        mesh.faces.detach().cpu().numpy().astype(np.int64),
    )


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices, faces, process=False), count
        )
    finally:
        np.random.set_state(state)
    return points


def paired_metrics(reference: np.ndarray, prediction: np.ndarray, threshold: float):
    pred_to_ref = cKDTree(reference).query(prediction, workers=-1)[0]
    ref_to_pred = cKDTree(prediction).query(reference, workers=-1)[0]
    precision = float(np.mean(pred_to_ref < threshold))
    recall = float(np.mean(ref_to_pred < threshold))
    return {
        "chamfer_l2_squared": float(
            0.5 * (np.mean(pred_to_ref**2) + np.mean(ref_to_pred**2))
        ),
        "precision": precision,
        "recall": recall,
        "fscore": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def decode(decoder, coords: np.ndarray, feats: np.ndarray):
    coords_t = torch.from_numpy(coords).int().cuda()
    feats_t = torch.from_numpy(feats).float().cuda()
    tensor = sp.SparseTensor(
        feats=feats_t,
        coords=torch.cat(
            (torch.zeros((len(coords_t), 1), dtype=torch.int32, device="cuda"), coords_t),
            dim=1,
        ),
    )
    meshes = decoder(tensor)
    arrays = mesh_arrays(meshes[0])
    del tensor, meshes
    gc.collect()
    torch.cuda.empty_cache()
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rdd", "flat"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument(
        "--supports",
        type=Path,
        default=None,
        help="optional predicted-support .npz directory; default uses ground-truth latent support",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--steps", type=int, nargs=3, default=(8, 12, 20))
    parser.add_argument("--guidance-strength", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-points", type=int, default=100_000)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="stop after sampling; useful for validating a new checkpoint before 1024 decoding",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["models"]["denoiser"]
    dataset_cfg = config["dataset"]["args"]
    trainer_cfg = config["trainer"]["args"]
    fine_resolution = int(model_cfg["args"]["resolution"])
    ov_resolution = int(dataset_cfg["resolution"])
    resolutions = (fine_resolution // 4, fine_resolution // 2, fine_resolution)
    channels = int(model_cfg["args"]["in_channels"])
    sigma_min = float(trainer_cfg.get("sigma_min", 1e-5))
    normalization = dataset_cfg["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)[None]
    std = np.asarray(normalization["std"], dtype=np.float32)[None]

    model = getattr(models, model_cfg["name"])(**model_cfg["args"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.cuda().eval()
    encoder_cfg = trainer_cfg["image_cond_model"]
    encoder = getattr(trainers, encoder_cfg["name"])(**encoder_cfg["args"])
    encoder.to("cuda")

    files = sorted(args.latents.glob("*.npz"))[: args.limit]
    if not files:
        raise SystemExit(f"no latent files under {args.latents}")
    generated = []
    sampling_rows = []
    for index, path in enumerate(files):
        with np.load(path) as data:
            gt_coords_np = data["coords"].astype(np.int32)
            gt_feats = data["feats"].astype(np.float32)
        coords_np = gt_coords_np
        if args.supports is not None:
            support_path = args.supports / path.name
            if not support_path.exists():
                raise FileNotFoundError(f"predicted support is missing: {support_path}")
            with np.load(support_path) as support:
                coords_np = support["coords"].astype(np.int32)
            if coords_np.ndim == 2 and coords_np.shape[1] == 4:
                coords_np = coords_np[:, 1:]
        image = condition_tensor(args.renders, path.stem, int(dataset_cfg["image_size"]))
        # Match the trainer's AMP context.  DINOv2's xFormers attention has no
        # FP32 kernel on Blackwell (sm_120), while BF16 is supported.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cond = encoder(image.cuda())
        neg_cond = torch.zeros_like(cond)
        coords = torch.from_numpy(coords_np).long().cuda()
        torch.manual_seed(args.seed + index)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.mode == "rdd":
                sampler = SparseRDDDetailEulerCfgSampler(
                    sigma_min=sigma_min,
                    resolutions=resolutions,
                    time_fractions=(0.5, 0.3, 0.2),
                )
                sample = sampler.sample(
                    model,
                    coords,
                    cond,
                    neg_cond,
                    channels=channels,
                    steps=args.steps,
                    guidance_strength=args.guidance_strength,
                    verbose=False,
                ).samples
            else:
                sampler = FlowEulerCfgSampler(sigma_min=sigma_min)
                sample = sampler.sample(
                    model,
                    sparse_noise(coords, channels),
                    cond,
                    neg_cond,
                    steps=sum(args.steps),
                    guidance_strength=args.guidance_strength,
                    verbose=False,
                ).samples
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pred_normalized = sample.feats.detach().float().cpu().numpy()
        pred_feats = pred_normalized * std + mean
        same_support = np.array_equal(coords_np, gt_coords_np)
        latent_mse = None
        if same_support:
            gt_normalized = (gt_feats - mean) / std
            latent_mse = float(np.mean((pred_normalized - gt_normalized) ** 2))
        sampling_rows.append(
            {
                "asset": path.stem,
                "tokens": len(coords_np),
                "seconds": elapsed,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "support": "ground_truth" if same_support else "predicted",
                "normalized_latent_mse": latent_mse,
                "prediction_mean": float(pred_normalized.mean()),
                "prediction_std": float(pred_normalized.std()),
                "prediction_max_abs": float(np.max(np.abs(pred_normalized))),
                "prediction_finite": bool(np.isfinite(pred_normalized).all()),
            }
        )
        if args.save_latents:
            latent_dir = args.output.with_suffix("") / "latents"
            latent_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                latent_dir / path.name,
                coords=coords_np,
                feats=pred_feats.astype(np.float32),
                normalized_feats=pred_normalized.astype(np.float32),
            )
        generated.append((path.stem, gt_coords_np, coords_np, gt_feats, pred_feats))
        del sample, cond, neg_cond, image, coords

    del model, encoder, state
    gc.collect()
    torch.cuda.empty_cache()

    protocol = (
        "shared predicted support; geometry SLat only; original SC-VAE decode as reference"
        if args.supports is not None
        else "known ground-truth support; geometry SLat only; original SC-VAE decode as reference"
    )
    base_report = {
        "protocol": protocol,
        "mode": args.mode,
        "resolution": ov_resolution,
        "latent_resolution": fine_resolution,
        "steps_coarse_middle_fine": list(args.steps),
        "total_nfe": sum(args.steps),
        "guidance_strength": args.guidance_strength,
        "checkpoint": str(args.checkpoint),
        "sampling": sampling_rows,
    }
    if sampling_rows:
        base_report["sampling_aggregate"] = {
            key: float(np.mean([row[key] for row in sampling_rows]))
            for key in ("seconds", "peak_memory_gib")
        }
        latent_mse = [
            row["normalized_latent_mse"]
            for row in sampling_rows
            if row["normalized_latent_mse"] is not None
        ]
        if latent_mse:
            base_report["sampling_aggregate"]["normalized_latent_mse"] = float(
                np.mean(latent_mse)
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampling_output = args.output.with_name(f"{args.output.stem}_sampling.json")
    sampling_output.write_text(json.dumps(base_report, indent=2), encoding="utf-8")
    if args.skip_decode:
        args.output.write_text(json.dumps(base_report, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), **base_report["sampling_aggregate"]}, indent=2))
        return

    decoder = models.from_pretrained(dataset_cfg["pretrained_slat_dec"]).eval().cuda()
    decoder.set_resolution(ov_resolution)
    mesh_dir = args.mesh_dir or args.output.with_suffix("")
    if args.save_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, (asset, gt_coords, pred_coords, gt_feats, pred_feats) in enumerate(generated):
        try:
            reference_mesh = decode(decoder, gt_coords, gt_feats)
            prediction_mesh = decode(decoder, pred_coords, pred_feats)
            if args.save_meshes:
                trimesh.Trimesh(*reference_mesh, process=False).export(
                    mesh_dir / f"{asset}_reference.ply"
                )
                trimesh.Trimesh(*prediction_mesh, process=False).export(
                    mesh_dir / f"{asset}_{args.mode}.ply"
                )
            reference_points = sample_surface(
                *reference_mesh, args.sample_points, args.seed + 1000 + index
            )
            prediction_points = sample_surface(
                *prediction_mesh, args.sample_points, args.seed + 2000 + index
            )
            row = {"asset": asset}
            row.update(paired_metrics(reference_points, prediction_points, args.fscore_threshold))
            rows.append(row)
        except Exception as exc:
            failures.append({"asset": asset, "error": f"{type(exc).__name__}: {exc}"})

    report = {
        **base_report,
        "metrics": rows,
        "failures": failures,
    }
    if rows:
        report["aggregate"] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("chamfer_l2_squared", "precision", "recall", "fscore")
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report.get("aggregate", {})}, indent=2))


if __name__ == "__main__":
    main()
