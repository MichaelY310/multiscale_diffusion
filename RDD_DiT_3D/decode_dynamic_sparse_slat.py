#!/usr/bin/env python3
"""Decode a generated sparse SLat with the unmodified TRELLIS SC-VAE decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from trellis2 import models
from trellis2.modules import sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument(
        "--trellis-config",
        type=Path,
        required=True,
        help="official generation config containing dataset.args.pretrained_slat_dec",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.trellis_config.read_text(encoding="utf-8"))
    dataset = config["dataset"]["args"]
    with np.load(args.sample) as data:
        coords = data["coords"].astype(np.int32)
        feats = data["feats"].astype(np.float32)
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    coords_t = torch.from_numpy(coords).int().cuda()
    feats_t = torch.from_numpy(feats).float().cuda()
    sparse = sp.SparseTensor(
        feats=feats_t,
        coords=torch.cat(
            (torch.zeros((len(coords_t), 1), dtype=torch.int32, device="cuda"), coords_t),
            dim=1,
        ),
    )
    decoder = models.from_pretrained(dataset["pretrained_slat_dec"]).eval().cuda()
    decoder.set_resolution(int(dataset["resolution"]))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mesh = decoder(sparse)[0]
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices, faces, process=False).export(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
