#!/usr/bin/env python3
"""Create lightweight GLB copies of exported PLY meshes for local viewing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument("--pattern", default="*_prediction.ply")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_dir.glob(args.pattern)):
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        original_faces = len(mesh.triangles)
        if original_faces > args.target_faces:
            mesh = mesh.simplify_quadric_decimation(args.target_faces)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        output = args.output_dir / f"{path.stem}_light.glb"
        trimesh.Trimesh(vertices, faces, process=False).export(output)
        row = {
            "source": str(path),
            "output": str(output),
            "original_faces": int(original_faces),
            "light_faces": int(len(faces)),
            "light_vertices": int(len(vertices)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output_dir / "light_manifest.json").write_text(
        json.dumps({"meshes": rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
