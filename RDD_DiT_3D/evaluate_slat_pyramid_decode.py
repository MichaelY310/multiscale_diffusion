#!/usr/bin/env python3
"""Decode real TRELLIS.2 SLat projections and quantify coarse semantic retention.

The original decoded SLat is the paired reference.  P16/P8 replace the finest
features by recursively pooled, nearest-replicated features on the *same* fine
support.  This isolates whether linear averaging in the learned SLat remains a
meaningful coarse 3D representation; it does not claim ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from trellis2 import models
from trellis2.modules import sparse as sp

from trellis2_rdd.hierarchy import build_masked_latent_pyramid, expand_parent_features


def sparse_tensor(coords: torch.Tensor, feats: torch.Tensor) -> sp.SparseTensor:
    batch = torch.zeros((len(coords), 1), dtype=torch.int32, device=coords.device)
    return sp.SparseTensor(feats=feats, coords=torch.cat((batch, coords.int()), dim=1))


def projected_features(pyramid, resolution: int) -> torch.Tensor:
    if resolution == pyramid.levels[-1].resolution:
        return pyramid.levels[-1].feats
    index = [level.resolution for level in pyramid.levels].index(resolution)
    values = pyramid.levels[index].feats
    for parent, child in zip(pyramid.levels[index:-1], pyramid.levels[index + 1 :]):
        values = expand_parent_features(parent, child, values)
    return values


def mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    return (
        mesh.vertices.detach().cpu().numpy().astype(np.float64),
        mesh.faces.detach().cpu().numpy().astype(np.int64),
    )


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("decoded mesh is empty")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(trimesh.Trimesh(vertices, faces, process=False), count)
    finally:
        np.random.set_state(state)
    return points


def paired_metrics(reference: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, float]:
    pred_to_ref = cKDTree(reference).query(prediction, workers=-1)[0]
    ref_to_pred = cKDTree(prediction).query(reference, workers=-1)[0]
    precision = float(np.mean(pred_to_ref < threshold))
    recall = float(np.mean(ref_to_pred < threshold))
    fscore = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "chamfer_l2_squared": float(
            0.5 * (np.mean(pred_to_ref**2) + np.mean(ref_to_pred**2))
        ),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="directory containing encoded SLat .npz files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512, help="decoded O-Voxel resolution")
    parser.add_argument("--latent-resolution", type=int, default=32)
    parser.add_argument("--sample-points", type=int, default=100_000)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument(
        "--decoder",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    )
    args = parser.parse_args()

    files = sorted(args.input.glob("*.npz"))[: args.limit]
    if not files:
        raise SystemExit(f"no .npz files found under {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = args.mesh_dir or args.output.with_suffix("")
    mesh_dir.mkdir(parents=True, exist_ok=True)

    decoder = models.from_pretrained(args.decoder).eval().cuda()
    decoder.set_resolution(args.resolution)
    rows = []
    failures = []
    for asset_index, path in enumerate(files):
        try:
            with np.load(path) as data:
                coords = torch.from_numpy(data["coords"]).long().cuda()
                feats = torch.from_numpy(data["feats"]).float().cuda()
            resolutions = (
                args.latent_resolution // 4,
                args.latent_resolution // 2,
                args.latent_resolution,
            )
            pyramid = build_masked_latent_pyramid(
                coords,
                feats,
                fine_resolution=args.latent_resolution,
                resolutions=resolutions,
            )
            variants = {
                "original": feats,
                f"p{resolutions[1]}": projected_features(pyramid, resolutions[1]),
                f"p{resolutions[0]}": projected_features(pyramid, resolutions[0]),
            }
            decoded = {}
            for name, values in variants.items():
                meshes = decoder(sparse_tensor(coords, values))
                mesh = meshes[0]
                vertices, faces = mesh_arrays(mesh)
                decoded[name] = (vertices, faces)
                trimesh.Trimesh(vertices, faces, process=False).export(
                    mesh_dir / f"{path.stem}_{name}.ply"
                )
                # The 1024 O-Voxel decoder peaks near 94 GiB.  Do not retain the
                # GPU-side mesh while decoding the next projected variant.
                del mesh, meshes
                gc.collect()
                torch.cuda.empty_cache()

            ref_points = sample_surface(*decoded["original"], args.sample_points, 1000 + asset_index)
            for name in (f"p{resolutions[1]}", f"p{resolutions[0]}"):
                pred_points = sample_surface(
                    *decoded[name], args.sample_points, 2000 + asset_index
                )
                row = {
                    "asset": path.stem,
                    "variant": name,
                    "fine_tokens": len(coords),
                    "coarse_tokens": int(
                        pyramid.levels[1 if name == f"p{resolutions[1]}" else 0].logical_coords.shape[0]
                    ),
                    "reference_vertices": len(decoded["original"][0]),
                    "reference_faces": len(decoded["original"][1]),
                    "prediction_vertices": len(decoded[name][0]),
                    "prediction_faces": len(decoded[name][1]),
                }
                row.update(paired_metrics(ref_points, pred_points, args.fscore_threshold))
                rows.append(row)
        except Exception as exc:
            failures.append({"asset": path.stem, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            torch.cuda.empty_cache()

    aggregate = {}
    variants = (f"p{args.latent_resolution // 2}", f"p{args.latent_resolution // 4}")
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        if subset:
            aggregate[variant] = {
                "num_assets": len(subset),
                "chamfer_mean": float(np.mean([row["chamfer_l2_squared"] for row in subset])),
                "fscore_mean": float(np.mean([row["fscore"] for row in subset])),
                "token_reduction_mean": float(
                    np.mean([row["fine_tokens"] / row["coarse_tokens"] for row in subset])
                ),
            }
    report = {
        "reference": "original SC-VAE decode (not ground-truth mesh)",
        "fscore_threshold": args.fscore_threshold,
        "sample_points": args.sample_points,
        "aggregate": aggregate,
        "samples": rows,
        "failures": failures,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **aggregate, "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
