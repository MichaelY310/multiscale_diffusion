#!/usr/bin/env python3
"""Decode predicted/target TRELLIS SLats and report paired mesh metrics."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

from trellis2 import models
from trellis2.modules import sparse as sp


def load_slat(
    path: Path, *, feature_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as data:
        coords = data["coords"].astype(np.int32)
        feats = data["feats"].astype(np.float32)
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:]
    return (
        torch.from_numpy(coords).int().cuda(),
        torch.from_numpy(feats).cuda().to(feature_dtype),
    )


def sparse_tensor(coords: torch.Tensor, feats: torch.Tensor) -> sp.SparseTensor:
    batch = torch.zeros((len(coords), 1), dtype=torch.int32, device=coords.device)
    return sp.SparseTensor(feats=feats, coords=torch.cat((batch, coords), dim=1))


def decode_arrays(decoder, path: Path) -> tuple[np.ndarray, np.ndarray]:
    # This published decoder is explicitly an fp16 sparse-convolution model.
    # Some normalization parameters remain fp32, so inspecting the first model
    # parameter is not a reliable way to choose the FlexGEMM input dtype.
    coords, feats = load_slat(path, feature_dtype=torch.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mesh = decoder(sparse_tensor(coords, feats))[0]
    vertices = mesh.vertices.detach().float().cpu().numpy().astype(np.float64)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int64)
    del coords, feats, mesh
    gc.collect()
    torch.cuda.empty_cache()
    if not len(vertices) or not len(faces):
        raise ValueError("decoded mesh is empty")
    return vertices, faces


def sample_surface(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices, faces, process=False), count
        )
    finally:
        np.random.set_state(state)
    return points


def paired_metrics(
    reference: np.ndarray, prediction: np.ndarray, threshold: float
) -> dict[str, float]:
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
        "fscore": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--export-meshes", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--sample-points", type=int, default=50_000)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument(
        "--decoder",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    )
    args = parser.parse_args()

    prediction_paths = sorted(args.prediction_dir.glob("*.npz"))[: args.limit]
    if not prediction_paths:
        raise SystemExit(f"no predictions under {args.prediction_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = args.mesh_dir or args.output.with_suffix("")
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # FlexGEMM requires activation and sparse-convolution weight dtypes to be
    # identical.  The published fp16 decoder may load a few normalization
    # parameters as fp32, which can promote an intermediate sparse tensor and
    # make Triton reject a later fp16 dot.  Use a uniform fp16 inference model.
    decoder = models.from_pretrained(args.decoder).eval().half().cuda()
    decoder.set_resolution(args.resolution)
    rows = []
    failures = []
    for index, prediction_path in enumerate(prediction_paths):
        target_path = args.ground_truth_dir / prediction_path.name
        try:
            reference_mesh = decode_arrays(decoder, target_path)
            prediction_mesh = decode_arrays(decoder, prediction_path)
            if args.export_meshes:
                trimesh.Trimesh(*reference_mesh, process=False).export(
                    mesh_dir / f"{prediction_path.stem}_target.ply"
                )
                trimesh.Trimesh(*prediction_mesh, process=False).export(
                    mesh_dir / f"{prediction_path.stem}_prediction.ply"
                )
            reference_points = sample_surface(*reference_mesh, args.sample_points, 1000 + index)
            prediction_points = sample_surface(*prediction_mesh, args.sample_points, 2000 + index)
            row = {
                "asset": prediction_path.stem,
                "target_vertices": len(reference_mesh[0]),
                "target_faces": len(reference_mesh[1]),
                "prediction_vertices": len(prediction_mesh[0]),
                "prediction_faces": len(prediction_mesh[1]),
            }
            row.update(
                paired_metrics(
                    reference_points, prediction_points, args.fscore_threshold
                )
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        except Exception as exc:
            failure = {
                "asset": prediction_path.stem,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)

    aggregate = {}
    if rows:
        for key in ("chamfer_l2_squared", "precision", "recall", "fscore"):
            aggregate[f"{key}_mean"] = float(np.mean([row[key] for row in rows]))
    report = {
        "prediction_dir": str(args.prediction_dir),
        "ground_truth_dir": str(args.ground_truth_dir),
        "fscore_threshold": args.fscore_threshold,
        "sample_points": args.sample_points,
        "aggregate": aggregate,
        "samples": rows,
        "failures": failures,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **aggregate, "failures": len(failures)}))


if __name__ == "__main__":
    main()
