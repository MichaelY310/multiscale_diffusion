#!/usr/bin/env python3
"""Capture real dense RDD trajectories without changing the sampling path.

Teacher trajectories, not student on-policy trajectories. Fine inputs and
teacher predictions are retained at a small fixed set of fine-stage NFE.
No VAE or image dataset is needed for this latent-level distillation corpus.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from probe_rdd_vs_dit_block_reuse import manual_forward
from sample_csra_adapter_preview import RDDRollout, sample_rdd
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher


class TrajectoryCollector:
    def __init__(self, model, layers, capture_indices):
        self.model = model
        self.layers = sorted(layers)
        self.capture_indices = set(capture_indices)
        self.coarse = None
        self.source_timestep = None
        self.labels = None
        self.states = []
        self.fine_calls = 0

    def forward(self, x, t, y):
        source = x.shape[-1] == 16 and bool(torch.all(t == 300))
        output, traces = manual_forward(
            self.model, x, t, y, multires=True,
            capture_layers=set(self.layers) if source else None,
        )
        if source:
            if self.coarse is not None:
                raise RuntimeError("Middle-stage boundary was captured twice")
            # Preserve the exact dtype. Do not add lossy cache compression.
            self.coarse = {
                layer: traces[layer]["residual"].detach().cpu().clone()
                for layer in self.layers
            }
            self.source_timestep = t.detach().cpu().clone()
            self.labels = y.detach().cpu().clone()
        if x.shape[-1] == 32:
            if self.coarse is None:
                raise RuntimeError("Fine stage reached without coarse cache")
            if self.fine_calls in self.capture_indices:
                if not torch.equal(self.labels, y.cpu()):
                    raise RuntimeError("Labels changed within trajectory")
                self.states.append({
                    "nfe_index": self.fine_calls,
                    "x": x.detach().cpu().clone(),
                    "t": t.detach().cpu().clone(),
                    "target": output.detach().cpu().clone(),
                })
            self.fine_calls += 1
        return output

    def payload(self):
        if self.coarse is None or self.fine_calls != 90:
            raise RuntimeError("Incomplete 80/80/90 trajectory")
        if len(self.states) != len(self.capture_indices):
            raise RuntimeError("Missing requested fine-stage states")
        return {
            "coarse": self.coarse, "labels": self.labels,
            "source_timestep": self.source_timestep,
            "nfe_indices": torch.tensor([s["nfe_index"] for s in self.states]),
            "x": torch.stack([s["x"] for s in self.states]),
            "t": torch.stack([s["t"] for s in self.states]),
            "target": torch.stack([s["target"] for s in self.states]),
        }


def collect(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output {output}")
    if args.batch_size < 1 or args.train_batches < 1 or args.validation_batches < 1:
        raise ValueError("Batch sizes/counts must be positive")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", "rdd")
    indices = [0, 12, 25, 38, 51, 64, 77, 89]
    files = []
    parity = None
    start = time.perf_counter()
    for batch in range(args.train_batches + args.validation_batches):
        split = "train" if batch < args.train_batches else "validation"
        seed = args.seed + args.shard * 1000 + batch
        collector = TrajectoryCollector(model, args.layers, indices)
        final, _ = sample_rdd(collector, args.batch_size, [80, 80, 90], device, seed)
        if batch == 0 and args.check_parity:
            dense, _ = sample_rdd(RDDRollout(model, None, [], "dense"),
                                  args.batch_size, [80, 80, 90], device, seed)
            parity = float((final - dense).abs().max())
            if parity != 0.0:
                raise RuntimeError(f"Collector changed dense trajectory: max_abs={parity}")
            del dense
        data = collector.payload()
        data.update({"version": 1, "seed": seed, "split": split,
                     "base_checkpoint": str(Path(args.base_ckpt).resolve()),
                     "layers": args.layers, "trajectory": "dense_teacher",
                     "sampler": "DDPM", "stage_steps": [80, 80, 90],
                     "cfg_scale": 1.0, "upsample": "noise_filling"})
        path = output / f"{split}_{batch:04d}.pt"
        atomic_save(data, path)
        files.append({"name": path.name, "split": split, "seed": seed,
                      "images": args.batch_size, "states_per_image": len(indices),
                      "bytes": path.stat().st_size})
        print(json.dumps({"saved": str(path), "batch": batch + 1,
                          "elapsed_seconds": time.perf_counter() - start,
                          "parity_max_abs": parity,
                          "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
        del final, data, collector
    manifest = {"complete": True, "files": files, "shard": args.shard,
                "shards": args.shards, "layers": args.layers,
                "base_checkpoint": str(Path(args.base_ckpt).resolve()),
                "parity_checked": args.check_parity, "parity_max_abs": parity,
                "trajectory": "dense_teacher", "config": vars(args)}
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=lambda v: [int(i) for i in v.split(",")], default=[13, 15, 17, 20])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261000)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--check-parity", action="store_true")
    collect(parser.parse_args())
