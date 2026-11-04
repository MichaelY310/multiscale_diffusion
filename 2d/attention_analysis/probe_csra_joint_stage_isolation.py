#!/usr/bin/env python3
"""Small counterfactual stage swap, NOT a deployment or unbiased quality benchmark.

Uses existing frozen/joint checkpoints without modifying either.  Tests whether
the changed coarse trajectory, changed fine function, or their interaction is
responsible for rollout regressions. Swapped internal feature spaces may be
incompatible, so this cannot establish a unique causal mechanism by itself.
"""
import argparse
import json
from pathlib import Path

import torch

from csra_joint_distillation import JointCSRAStudent
from train_rdd_csra_adapter import CSRAAdapterBank, load_frozen_teacher
from sample_csra_adapter_preview import (
    RDDRollout, sample_rdd, decode, relative_l2, save_grid, to_pil, AutoencoderKL,
)


class StageSwap:
    def __init__(self, source, fine, use_adapter=True):
        self.source, self.fine = source, fine
        self.use_adapter = use_adapter
        self.cache = None

    def forward(self, x, t, y):
        if x.shape[-1] < 32:
            capture = x.shape[-1] == 16 and bool(torch.all(t == 300))
            pred, captured = self.source.stage(x, t, y, capture=capture)
            if capture:
                self.cache = captured
            return pred
        if self.use_adapter and self.cache is None:
            raise RuntimeError("Missing 16x16 source")
        return self.fine.stage(x, t, y,
            cache=self.cache if self.use_adapter else None)[0]


class TraceRollout:
    def __init__(self, wrapped, teacher):
        self.wrapped, self.teacher = wrapped, teacher
        self.boundaries = {}
        self.teacher_checks = []
        self.counts = {}

    def forward(self, x, t, y):
        res = int(x.shape[-1])
        index = self.counts.get(res, 0)
        self.counts[res] = index + 1
        output = self.wrapped.forward(x, t, y)
        timestep = int(t[0])
        if index == 0 or timestep in (600, 300, 0):
            tag = f"r{res}_t{timestep}"
            self.boundaries[tag] = {"input": x.detach().float().cpu(),
                                    "output": output.detach().float().cpu()}
        if index % 10 == 0 or timestep in (600, 300, 0):
            # Same-state teacher call: no extra stochastic sampling or RNG use.
            target = self.teacher.forward(x, t, y)
            diff = output.float() - target.float()
            self.teacher_checks.append({"resolution": res, "nfe_in_stage": index,
                "timestep": timestep, "main_mse": diff[:, :4].square().mean().item(),
                "variance_mse": diff[:, 4:].square().mean().item()})
        return output


def load_student(arm, device):
    path = Path(f"checkpoints/RDD-CSRA-jointpilot-{arm}-L400K-20260913/best.pt")
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if ck.get("format") != "csra_joint_v1":
        raise ValueError("Unexpected checkpoint format")
    meta = ck["adapter_metadata"]
    adapter = CSRAAdapterBank(meta["layers"], meta["hidden_size"], meta["rank"],
                              meta["adapter_architecture"]).to(device)
    backbone = load_frozen_teacher(Path("checkpoints/RDD-Unified-L/0400000.pt"),
                                   device, "DiT-L/2", "rdd")
    student = JointCSRAStudent(backbone, adapter, False, False)
    student.load_state_dict(ck["student"])
    return student.eval().requires_grad_(False), {"path": str(path), "step": ck["step"]}


def main(args):
    out = Path(args.output_dir)
    if (out / "metrics.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = load_frozen_teacher(Path("checkpoints/RDD-Unified-L/0400000.pt"),
                                device, "DiT-L/2", "rdd")
    teacher = RDDRollout(model, None, [], "dense")
    frozen, fm = load_student("frozen", device)
    joint, jm = load_student("joint", device)
    modes = {
        "original_dense": teacher,
        "frozen_both": StageSwap(frozen, frozen),
        "joint_both": StageSwap(joint, joint),
        "joint_fine_only": StageSwap(frozen, joint),
        "joint_coarse_only": StageSwap(joint, frozen),
        "joint_backbone_dense": StageSwap(joint, joint, False),
        "frozen_backbone_dense": StageSwap(frozen, frozen, False),
    }
    latents, traces, results = {}, {}, {}
    for name, wrapper in modes.items():
        trace = TraceRollout(wrapper, teacher)
        latent, _ = sample_rdd(trace, 4, [80, 80, 90], device, args.seed)
        latents[name] = latent.cpu()
        traces[name] = trace.boundaries
        results[name] = {
            "relative_latent_l2": relative_l2(latent.cpu(), latents["original_dense"]),
            "teacher_same_state_checks": trace.teacher_checks,
            "stage_nfe_counts": trace.counts,
            "boundaries": {key: {
                "input_relative_l2": relative_l2(item["input"], traces["original_dense"][key]["input"]),
                "output_relative_l2": relative_l2(item["output"], traces["original_dense"][key]["output"]),
            } for key, item in trace.boundaries.items()},
        }
        print(json.dumps({"mode": name, "seed": args.seed,
                          "relative_latent_l2": results[name]["relative_latent_l2"]}), flush=True)
    # Verify controls against the original completed pipeline, if available.
    previous = Path(f"eval_results/csra_joint_pair_20260913/seed_{args.seed}/metrics.json")
    if previous.exists():
        prior = json.loads(previous.read_text())
        for arm in ("frozen", "joint"):
            if abs(results[f"{arm}_both"]["relative_latent_l2"] - prior["relative_latent_l2"][arm]) > 2e-6:
                raise RuntimeError(f"{arm} control differs from original completed preview")
    del modes, wrapper, frozen, joint, teacher, model
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained("checkpoints/pretrained/stabilityai/sd-vae-ft-ema",
        local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    decoded = {name: decode(vae, z.to(device)) for name, z in latents.items()}
    for name, images in decoded.items():
        (out / name).mkdir(exist_ok=True)
        for i, im in enumerate(images):
            to_pil(im).save(out / name / f"{i:03d}.png")
    save_grid(list(decoded.items()), out / "comparison_grid.png")
    torch.save({"latents": latents, "boundaries": traces}, out / "trace.pt")
    payload = {"seed": args.seed, "batch": 4, "steps": [80, 80, 90], "cfg": 1.,
        "upsample": "noise_filling", "results": results, "checkpoints": {"frozen": fm, "joint": jm},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "warning": "Failure-selected development seeds; feature-space swaps are counterfactual, not trained variants. Teacher checks add cost, no timing benchmark."}
    temp = out / "metrics.json.tmp"
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(out / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
