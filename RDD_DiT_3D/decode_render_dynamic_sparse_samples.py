#!/usr/bin/env python3
"""Decode dynamic-sparse RDD samples and render compact preview sheets."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import trimesh

from trellis2 import models
from trellis2.modules import sparse as sp
from trellis2.utils import render_utils


def first_condition(render_root: Path, asset: str, size: int) -> Image.Image:
    candidates = sorted((render_root / asset).glob("*.png"))
    if not candidates:
        return Image.new("RGB", (size, size), "white")
    image = Image.open(candidates[0]).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    return ImageOps.contain(background.convert("RGB"), (size, size))


def sparse_tensor(path: Path) -> sp.SparseTensor:
    with np.load(path) as data:
        coords = data["coords"].astype(np.int32)
        feats = data["feats"].astype(np.float32)
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:]
    coords_t = torch.from_numpy(coords).int().cuda()
    feats_t = torch.from_numpy(feats).half().cuda()
    batch = torch.zeros((len(coords_t), 1), dtype=torch.int32, device="cuda")
    return sp.SparseTensor(
        feats=feats_t,
        coords=torch.cat((batch, coords_t), dim=1),
    )


def normal_view(normal: np.ndarray, mask: np.ndarray) -> Image.Image:
    normal_image = Image.fromarray(normal).convert("RGB")
    alpha = Image.fromarray(mask[..., 0] if mask.ndim == 3 else mask).convert("L")
    canvas = Image.new("RGB", normal_image.size, "white")
    canvas.paste(normal_image, mask=alpha)
    return canvas


def preview_sheet(
    condition: Image.Image,
    views: list[Image.Image],
    *,
    asset: str,
    stage_counts: list[int] | None,
    tile: int,
) -> Image.Image:
    condition = ImageOps.pad(condition, (tile, tile), color="white")
    view_tiles = [ImageOps.fit(view, (tile // 2, tile // 2)) for view in views]
    view_grid = Image.new("RGB", (tile, tile), "white")
    for index, view in enumerate(view_tiles[:4]):
        view_grid.paste(view, ((index % 2) * tile // 2, (index // 2) * tile // 2))
    header = 54
    sheet = Image.new("RGB", (tile * 2, tile + header), "white")
    sheet.paste(condition, (0, header))
    sheet.paste(view_grid, (tile, header))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"condition | {asset[:12]}", fill="black")
    counts = "" if not stage_counts else " | retained 8/16/32: " + "/".join(map(str, stage_counts))
    draw.text((tile + 12, 8), "RDD-3D generated mesh" + counts, fill="black")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--preview-resolution", type=int, default=512)
    parser.add_argument(
        "--decoder",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.prediction_dir / "manifest.json"
    sampling_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"samples": []}
    )
    sampling_rows = {row["asset"]: row for row in sampling_manifest.get("samples", [])}

    decoder = models.from_pretrained(args.decoder).eval().half().cuda()
    decoder.set_resolution(args.resolution)
    output_rows = []
    failures = []
    for sample_path in sorted(args.prediction_dir.glob("*.npz"))[: args.limit]:
        asset = sample_path.stem
        try:
            sparse = sparse_tensor(sample_path)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mesh = decoder(sparse)[0]
            vertices = mesh.vertices.detach().float().cpu().numpy()
            faces = mesh.faces.detach().cpu().numpy()
            if not len(vertices) or not len(faces):
                raise ValueError("decoded mesh is empty")
            exported = trimesh.Trimesh(vertices, faces, process=False)
            ply_path = args.output_dir / f"{asset}_prediction.ply"
            glb_path = args.output_dir / f"{asset}_prediction.glb"
            exported.export(ply_path)
            exported.export(glb_path)

            rendered = render_utils.render_snapshot(
                mesh,
                resolution=args.preview_resolution,
                r=2,
                fov=36,
                nviews=4,
                verbose=False,
            )
            views = [
                normal_view(normal, mask)
                for normal, mask in zip(rendered["normal"], rendered["mask"])
            ]
            sample_row = sampling_rows.get(asset, {})
            stage_counts = [
                int(stage["retained_tokens"])
                for stage in sample_row.get("stages", [])
                if "retained_tokens" in stage
            ]
            sheet = preview_sheet(
                first_condition(args.render_dir, asset, args.preview_resolution),
                views,
                asset=asset,
                stage_counts=stage_counts,
                tile=args.preview_resolution,
            )
            preview_path = args.output_dir / f"{asset}_preview.png"
            sheet.save(preview_path)
            row = {
                "asset": asset,
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "stage_counts": stage_counts,
                "ply": str(ply_path),
                "glb": str(glb_path),
                "preview": str(preview_path),
            }
            output_rows.append(row)
            print(json.dumps(row), flush=True)
            del sparse, mesh, exported
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            failure = {"asset": asset, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps(failure), flush=True)

    report = {"samples": output_rows, "failures": failures}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not output_rows:
        raise SystemExit("all preview decodes failed")


if __name__ == "__main__":
    main()
