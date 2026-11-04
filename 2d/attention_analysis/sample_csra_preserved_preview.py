#!/usr/bin/env python3
"""Paired preview with original coarse stages and separately trained fine model.

This explicitly retains TWO backbone weight sets. Do not label it a single
unified checkpoint or count model swapping/offloading as free in a benchmark.
"""
import argparse
import json
import re
from pathlib import Path

import torch

from csra_joint_distillation import PreservedCoarseStudent
from train_rdd_csra_adapter import CSRAAdapterBank, load_frozen_teacher
from sample_csra_adapter_preview import (
    RDDRollout, sample_rdd, manual_forward, decode, relative_l2, save_grid, to_pil,
    AutoencoderKL,
)


class PreservedRollout:
    def __init__(self, teacher, student=None):
        self.teacher, self.student = teacher, student
        self.cache = None
        self.endpoints = {}

    def forward(self, x, t, y):
        res, step = int(x.shape[-1]), int(t[0])
        if res == 16 and step == 300 and self.student is not None and self.student.coarse_source == "aligned":
            result, trace = manual_forward(self.teacher, x, t, y, multires=True,
                                          capture_layers=set(self.student.adapter.layers))
            self.cache = {layer: trace[layer][getattr(self.student.adapter.adapters[str(layer)], "coarse_feature", "residual")]
                          for layer in self.student.adapter.layers}
        elif res == 32 and self.student is not None:
            if self.cache is None:
                if self.student.coarse_source == "zero":
                    self.cache = {layer: None for layer in self.student.adapter.layers}
                else:
                    raise RuntimeError("Missing unchanged original coarse-stage cache")
            result = self.student.stage(x, t, y, cache=self.cache)[0]
        else:
            result, _ = manual_forward(self.teacher, x, t, y, multires=True)
        if (res, step) in ((8, 600), (16, 300)):
            self.endpoints[str(res)] = {"input": x.detach().cpu(), "output": result.detach().cpu()}
        return result


def main(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]*", args.checkpoint_suffix):
        raise ValueError("Invalid checkpoint suffix")
    out = Path(args.output_dir)
    if (out / "metrics.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    base = Path("checkpoints/RDD-Unified-L/0400000.pt")
    teacher = load_frozen_teacher(base, device, "DiT-L/2", "rdd")
    original = PreservedRollout(teacher)
    latents, configurations, parity = {}, {}, {}
    latents["original_dense"], _ = sample_rdd(original, 4, [80,80,90], device, args.seed)
    if getattr(args, "hidden_control", False):
        if args.checkpoint_suffix or args.source_control or args.memory_control:
            raise ValueError("Hidden comparison cannot also select another comparison")
        entries = [(f"hidden_{mode}", Path(f"checkpoints/RDD-CSRA-hidden_v6-{mode}-L400K-20260913/best.pt"))
                   for mode in ("aligned", "zero")]
    elif args.memory_control:
        if args.checkpoint_suffix or args.source_control:
            raise ValueError("Memory control cannot also select another comparison")
        entries = [(f"memory_{mode}", Path(f"checkpoints/RDD-CSRA-memory_v5-{mode}-L400K-20260913/best.pt"))
                   for mode in ("aligned", "zero")]
    elif args.source_control:
        if args.checkpoint_suffix:
            raise ValueError("Source-control comparison cannot also select a suffix")
        entries = [("coarse_joint1600", Path("checkpoints/RDD-CSRA-preserved_joint-continue1600-L400K-20260913/best.pt")),
                   ("no_coarse_joint1600", Path("checkpoints/RDD-CSRA-preserved_joint-no_coarse-L400K-20260913/best.pt"))]
    else:
        entries = [(arm, Path(f"checkpoints/RDD-CSRA-{arm}{args.checkpoint_suffix}-L400K-20260913/best.pt"))
                   for arm in ("preserved_frozen", "preserved_joint")]
    for arm, path in entries:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        source_mode = ck["args"].get("coarse_source", "aligned")
        expected_format = "csra_preserved_source_control_v1" if source_mode != "aligned" else "csra_preserved_coarse_v1"
        if ck["adapter_metadata"]["adapter_architecture"] == "hidden_memory_v6":
            expected_format = "csra_preserved_hidden_memory_v1"
        if ck["format"] != expected_format:
            raise ValueError("This sampler requires a preserved-coarse checkpoint")
        meta = ck["adapter_metadata"]
        adapter = CSRAAdapterBank(meta["layers"], meta["hidden_size"], meta["rank"], meta["adapter_architecture"]).to(device)
        fine_backbone = load_frozen_teacher(base, device, "DiT-L/2", "rdd")
        student = PreservedCoarseStudent(fine_backbone, adapter, False, False, coarse_source=source_mode)
        student.load_state_dict(ck["student"])
        student.eval().requires_grad_(False)
        wrapper = PreservedRollout(teacher, student)
        latents[arm], _ = sample_rdd(wrapper, 4, [80,80,90], device, args.seed)
        parity[arm] = max(float((v.float()-original.endpoints[r][key].float()).abs().max())
                          for r, item in wrapper.endpoints.items() for key, v in item.items())
        if parity[arm] != 0 or set(wrapper.endpoints) != {"8", "16"}:
            raise RuntimeError("Original coarse-stage input/output parity failed")
        configurations[arm] = {"path": str(path), "step": ck["step"], "args": ck["args"]}
        del wrapper, student, fine_backbone, adapter, ck
        torch.cuda.empty_cache()
    del original, teacher
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained("checkpoints/pretrained/stabilityai/sd-vae-ft-ema",
        local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    decoded = {name: decode(vae, z) for name, z in latents.items()}
    for name, images in decoded.items():
        (out / name).mkdir(exist_ok=True)
        for i, im in enumerate(images):
            to_pil(im).save(out / name / f"{i:03d}.png")
    save_grid(list(decoded.items()), out / "comparison_grid.png")
    metrics = {"seed": args.seed, "batch": 4, "steps": [80,80,90], "cfg": 1.,
        "upsample": "noise_filling", "coarse_endpoint_max_abs_vs_original": parity,
        "relative_latent_l2": {name: relative_l2(z, latents["original_dense"]) for name,z in latents.items()},
        "checkpoints": configurations,
        "warning": "Seven development seeds, not FID. Original coarse weights plus separately trained fine weights: extra resident memory. No speed claim."}
    temp = out / "metrics.json.tmp"
    temp.write_text(json.dumps(metrics, indent=2))
    temp.replace(out / "metrics.json")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-suffix", default="")
    parser.add_argument("--source-control", action="store_true")
    parser.add_argument("--memory-control", action="store_true")
    parser.add_argument("--hidden-control", action="store_true")
    main(parser.parse_args())
