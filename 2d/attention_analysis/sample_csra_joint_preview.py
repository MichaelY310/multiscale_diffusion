#!/usr/bin/env python3
"""Original teacher vs frozen and joint student: identical latent RNG and sampler."""
import argparse
import json
from pathlib import Path
import torch
from csra_joint_distillation import JointCSRAStudent
from train_rdd_csra_adapter import CSRAAdapterBank, load_frozen_teacher
from sample_csra_adapter_preview import (sample_rdd, RDDRollout, decode, relative_l2,
                                         save_grid, to_pil, AutoencoderKL)


class JointRollout:
    def __init__(self, student):
        self.student = student
        self.cache = None

    def forward(self, x, t, y):
        capture = x.shape[-1] == 16 and bool(torch.all(t == 300))
        cache = self.cache if x.shape[-1] == 32 else None
        if x.shape[-1] == 32 and cache is None:
            raise RuntimeError("Missing live source cache")
        pred, captured = self.student.stage(x, t, y, cache=cache, capture=capture)
        if capture:
            self.cache = captured
        return pred


def main(args):
    out = Path(args.output_dir)
    if (out / "metrics.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    base = Path("checkpoints/RDD-Unified-L/0400000.pt")
    teacher = load_frozen_teacher(base, device, "DiT-L/2", "rdd")
    latents, timings, configurations = {}, {}, {}
    latents["original_dense"], timings["original_dense"] = sample_rdd(
        RDDRollout(teacher, None, [], "dense"), 4, [80,80,90], device, args.seed)
    del teacher
    torch.cuda.empty_cache()
    for arm in ("frozen", "joint"):
        path = Path(f"checkpoints/RDD-CSRA-jointpilot-{arm}-L400K-20260913/best.pt")
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        meta = ck["adapter_metadata"]
        adapter = CSRAAdapterBank(meta["layers"], meta["hidden_size"], meta["rank"], meta["adapter_architecture"]).to(device)
        backbone = load_frozen_teacher(base, device, "DiT-L/2", "rdd")
        student = JointCSRAStudent(backbone, adapter, False, False)
        student.load_state_dict(ck["student"])
        student.eval().requires_grad_(False)
        latents[arm], timings[arm] = sample_rdd(JointRollout(student), 4, [80,80,90], device, args.seed)
        configurations[arm] = {"path": str(path), "step": ck["step"], "args": ck["args"]}
        del student, backbone, adapter, ck
        torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained("checkpoints/pretrained/stabilityai/sd-vae-ft-ema",
        local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    decoded = {name: decode(vae, z) for name,z in latents.items()}
    for name, images in decoded.items():
        (out / name).mkdir(exist_ok=True)
        for i, im in enumerate(images):
            to_pil(im).save(out / name / f"{i:03d}.png")
    save_grid(list(decoded.items()), out / "comparison_grid.png")
    metrics = {"seed": args.seed, "batch": 4, "steps": [80,80,90], "cfg": 1.,
        "upsample": "noise_filling", "layers": [13,15,17,20],
        "relative_latent_l2": {name: relative_l2(z, latents["original_dense"]) for name,z in latents.items()},
        "first_sample_seconds_not_benchmark": timings, "checkpoints": configurations,
        "precision": "BF16 original teacher; FP32 student master weights under BF16 autocast; FP32 adapters",
        "quality_warning": "Development previews, not FID or held-out quality benchmark"}
    temp = out / "metrics.json.tmp"
    temp.write_text(json.dumps(metrics, indent=2))
    temp.replace(out / "metrics.json")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    main(p.parse_args())
