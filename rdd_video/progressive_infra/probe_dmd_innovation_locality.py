"""True trained RDD-DMD: does newborn fine noise require distant output coupling?

Preserve the input's pooled parent exactly; change only zero-mean noise in one
2x2 cell. This tests endpoint stochastic response, not instantaneous FM targets.
No sampler change, no skipped blocks, no new training, no speed claim.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


@torch.inference_mode()
def main(a):
    root = Path(a.root)
    sys.path[:0] = [str(root), str(root/"DiT"), str(root/"2d/attention_analysis")]
    from models_multires import DiTMultiRes_models
    from dmd_cascade_reuse import CascadeDMD, coupled_noises
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.mem_get_info()[0] < 16*2**30:
        raise RuntimeError("Need16GiB headroom; never evict an existing process")
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(a.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = DiTMultiRes_models["DiT-L/2"](num_classes=1000, base_size=16)
    model.load_state_dict(ck["model"], strict=True)
    step = int(ck["step"])
    del ck
    model = model.cuda().eval().requires_grad_(False)
    net = CascadeDMD(model, "dense", False)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    noises, labels = coupled_noises(4, torch.device("cuda"), gen)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
        baseline = net(noises, labels)
    x = baseline["inputs"][2]
    base = baseline["cleans"][2]
    def parent(v):
        return F.avg_pool2d(v, 2)
    rows, maps = [], []
    yy, xx = torch.meshgrid(torch.arange(32, device="cuda"), torch.arange(32, device="cuda"), indexing="ij")
    for image in range(4):
        for cell in ((8, 8), (4, 12)):
            cy, cx = cell
            delta = torch.zeros_like(x[image:image+1])
            noise = torch.randn((1, 4, 2, 2), device="cuda", generator=gen)
            noise -= noise.mean((-2, -1), keepdim=True)
            delta[..., 2*cy:2*cy+2, 2*cx:2*cx+2] = .5*noise
            changed = x[image:image+1]+delta
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                pred = net.clean_at(changed, labels[image:image+1], 2)[0]
                repeated = net.clean_at(x[image:image+1], labels[image:image+1], 2)[0]
                repeated_twice = net.clean_at(x[image:image+1], labels[image:image+1], 2)[0]
            difference = pred-repeated
            energy = difference.square().mean((0, 1))
            distance = torch.maximum((yy-(2*cy+.5)).abs(), (xx-(2*cx+.5)).abs())
            bands = {}
            for label, lo, hi in (("local_radius2", -1, 2), ("radius2to6", 2, 6), ("far_radius6plus", 6, 100)):
                mask = (distance > lo) & (distance <= hi)
                bands[label] = dict(rms=float(energy[mask].mean().sqrt()),
                    energy_fraction=float(energy[mask].sum()/energy.sum().clamp_min(1e-20)))
            row = dict(image=image, label=int(labels[image]), changed_parent_cell=cell,
                       input_parent_max_difference=float((parent(changed)-parent(x[image:image+1])).abs().max()),
                       batched_vs_single_nochange_max=float((repeated-base[image:image+1]).abs().max()),
                       identical_single_repeat_max=float((repeated-repeated_twice).abs().max()),
                       output_parent_response_rms=float(parent(difference).square().mean().sqrt()),
                       output_response_rms=float(difference.square().mean().sqrt()), bands=bands)
            rows.append(row)
            maps.append(energy.cpu())
            print(json.dumps(row), flush=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(maps), out/"response_energy_maps.pt")
    result = dict(complete=True, checkpoint=a.checkpoint, checkpoint_step=step,
                  model="real RDD Unified DiT-L/2 DMD", seed=a.seed, NFE=3,
                  timesteps=[999, 599, 299], stage_resolutions=[8, 16, 32],
                  new_training=False, precision=a.precision, records=rows,
                  warning="Local perturbations diagnose teacher endpoint coupling; eight cases are not a quality/speed benchmark.")
    (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/scr/yuheng/DevOpsBench/multiscale_diffusion")
    p.add_argument("--checkpoint", default="/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915/start/generator_start.pt")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="fp32")
    main(p.parse_args())
