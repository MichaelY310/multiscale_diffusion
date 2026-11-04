#!/usr/bin/env python3
"""Paired teacher-state vs student-state error, using held-out trajectories.

No training, no VAE, no speed claims. Auxiliary teacher calls must leave the
actual student rollout unchanged, verified by exact final-latent parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from csra_runtime import predict_adapter
from probe_rdd_vs_dit_block_reuse import manual_forward, upsample_tokens
from sample_csra_adapter_preview import RDDRollout, load_adapter, sample_rdd
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher


def lowpass(x):
    """Orthogonal 2x2-cell average projection on the current latent grid."""
    return F.avg_pool2d(x.float(), 2, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)


def errors(prediction, reference):
    difference = prediction.float() - reference.float()
    return {"mse": float(difference.square().mean()),
            "target_power": float(reference.float().square().mean()),
            "relative_l2": float(difference.norm() / reference.float().norm().clamp_min(1e-12))}


class RolloutShiftProbe:
    def __init__(self, model, adapter, reference):
        self.model, self.adapter, self.reference = model, adapter, reference
        self.rollout = RDDRollout(model, adapter, adapter.layers, "adapter", "fp32")
        self.indices = {int(n): i for i, n in enumerate(reference["nfe_indices"])}
        self.fine_calls = 0
        self.rows, self.states = [], []
        self.source_max_abs = None

    def forward(self, x, t, y):
        output = self.rollout.forward(x, t, y)
        if x.shape[-1] == 16 and bool(torch.all(t == 300)):
            self.source_max_abs = max(float((self.rollout.coarse[layer] - upsample_tokens(
                self.reference["coarse"][layer].to(x.device), 16)).abs().max())
                for layer in self.adapter.layers)
            if self.source_max_abs != 0:
                raise RuntimeError("Paired runs have different coarse caches")
        if x.shape[-1] == 32:
            nfe = self.fine_calls
            if nfe in self.indices:
                i = self.indices[nfe]
                ref_x = self.reference["x"][i].to(x.device)
                ref_t = self.reference["t"][i].to(t.device)
                if not torch.equal(t, ref_t) or not torch.equal(y.cpu(), self.reference["labels"]):
                    raise RuntimeError("Paired trajectory time/labels mismatch")
                if nfe == 0 and not torch.equal(x, ref_x):
                    raise RuntimeError("First fine input must match exactly before any replacement")
                target = self.reference["target"][i].to(x.device)
                teacher_ref, _ = manual_forward(self.model, ref_x, t, y, multires=True)
                if not torch.equal(teacher_ref, target):
                    raise RuntimeError("Saved teacher predictions no longer reproduce exactly")
                callbacks = {layer: (lambda hidden, layer=layer: predict_adapter(
                    self.adapter.adapters[str(layer)], self.rollout.coarse[layer], hidden, t, "fp32"))
                    for layer in self.adapter.layers}
                student_ref, _ = manual_forward(self.model, ref_x, t, y, multires=True,
                                                 replacements=callbacks)
                teacher_on, _ = manual_forward(self.model, x, t, y, multires=True)
                delta = output[:, :4].float() - teacher_on[:, :4].float()
                projected = lowpass(delta)
                row = {"nfe": nfe, "t": int(t[0]), "images": len(y),
                       "teacher_state_main": errors(student_ref[:, :4], target[:, :4]),
                       "student_state_main": errors(output[:, :4], teacher_on[:, :4]),
                       "teacher_state_variance": errors(student_ref[:, 4:], target[:, 4:]),
                       "student_state_variance": errors(output[:, 4:], teacher_on[:, 4:]),
                       "trajectory_state": errors(x, ref_x),
                       "student_main_low_error_power": float(projected.square().mean()),
                       "student_main_high_error_power": float((delta - projected).square().mean())}
                self.rows.append(row)
                self.states.append({"x": x.detach().cpu().clone(), "t": t.cpu().clone(),
                                    "target": teacher_on.cpu().clone()})
            self.fine_calls += 1
        return output


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing probe")
    root = Path(args.data_root) / f"shard_{args.shard}"
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"] or manifest["parity_max_abs"] != 0 or not manifest["parity_checked"]:
        raise ValueError("Require a completed parity-checked teacher dataset")
    if Path(manifest["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve():
        raise ValueError("Wrong teacher")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", "rdd")
    adapter, info = load_adapter(Path(args.adapter_ckpt), device)
    if adapter.layers != manifest["layers"] or Path(info["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve():
        raise ValueError("Wrong adapter layers or teacher")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    items = [item for item in manifest["files"] if item["split"] == "validation"]
    if not items:
        raise ValueError("No validation trajectories")
    for item in items:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unsafe dataset filename")
        reference = torch.load(root / item["name"], map_location="cpu", weights_only=True)
        if reference["split"] != "validation" or reference["trajectory"] != "dense_teacher":
            raise ValueError("Probe must use held-out dense teacher trajectories")
        probe = RolloutShiftProbe(model, adapter, reference)
        final, _ = sample_rdd(probe, item["images"], [80, 80, 90], device, item["seed"])
        expected, _ = sample_rdd(RDDRollout(model, adapter, adapter.layers, "adapter", "fp32"),
                                 item["images"], [80, 80, 90], device, item["seed"])
        parity = float((final - expected).abs().max())
        if parity != 0 or probe.fine_calls != 90 or len(probe.rows) != 8:
            raise RuntimeError("Probe changed or incompletely observed the actual sampling path")
        # Retain validation states for subsequent diagnostics, never relabel as train.
        data = {"coarse": reference["coarse"], "labels": reference["labels"],
                "source_timestep": reference["source_timestep"], "nfe_indices": reference["nfe_indices"],
                **{key: torch.stack([s[key] for s in probe.states]) for key in ("x", "t", "target")},
                "split": "validation", "trajectory": "frozen_student",
                "adapter_checkpoint": str(Path(args.adapter_ckpt).resolve()),
                "base_checkpoint": str(Path(args.base_ckpt).resolve()), "layers": adapter.layers,
                "seed": item["seed"]}
        atomic_save(data, output / f"student_validation_{item['seed']}.pt")
        result = {"seed": item["seed"], "images": item["images"], "rows": probe.rows,
                  "probe_parity_max_abs": parity, "coarse_cache_max_abs": probe.source_max_abs}
        results.append(result)
        print(json.dumps(result), flush=True)
    result = {"complete": True, "shard": args.shard, "adapter_train_step": info["step"],
              "config": vars(args), "results": results,
              "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
    temp = output / "metrics.json.tmp"
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(output / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    run(parser.parse_args())
