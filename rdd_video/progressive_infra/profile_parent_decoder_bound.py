"""Native Wan VAE parent-decode timing: an optimistic overlap bound, NOT a new codec.

Existing Wan VAE features cannot be reused exactly across resolutions. Timings
here only screen whether an early-parent decoding research direction has enough
critical-path time to matter. No generated-quality or achieved-speed claim.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


def main(a):
    root = Path(a.root)
    sys.path.insert(0, str(root/"Wan2.1"))
    from wan.modules.vae import WanVAE
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < 40*2**30:
        raise RuntimeError("Need 40GiB free; never evict another process")
    torch.cuda.set_per_process_memory_fraction(.65)
    vae = WanVAE(z_dim=16, vae_pth=str(root/"pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
                 dtype=torch.bfloat16, device=torch.device("cuda:0"))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), seed=20261003,
                  warning="Parent decoder latency is only an optimistic overlap bound. It is not reusable mandatory work in current WanVAE.",
                  rows=[])
    gen = torch.Generator(device="cuda").manual_seed(20261003)
    for label, shape in (("832x448x29", (16, 8, 56, 104)), ("1280x768x61", (16, 16, 96, 160))):
        full = torch.randn(shape, device="cuda", generator=gen)
        parent = F.avg_pool3d(full.unsqueeze(0), (1, 2, 2))[0]
        xs = dict(full=full, parent=parent)
        def run(name):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                video = vae.decode([xs[name]])[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if not torch.isfinite(video).all():
                raise RuntimeError("Nonfinite decoder output")
            record = dict(name=name, seconds=elapsed, peak_gib=torch.cuda.max_memory_allocated()/2**30)
            del video
            return record
        for name in xs:
            for _ in range(2):
                run(name)
        rows = []
        for group in range(a.groups):
            order = ("full", "parent", "parent", "full") if group%2 == 0 else ("parent", "full", "full", "parent")
            for name in order:
                row = dict(group=group, **run(name))
                rows.append(row)
                print(json.dumps(dict(shape=label, **row)), flush=True)
        item = dict(shape=label, latent_shape=shape, records=rows,
                    full_median=statistics.median(r["seconds"] for r in rows if r["name"] == "full"),
                    parent_median=statistics.median(r["seconds"] for r in rows if r["name"] == "parent"))
        result["rows"].append(item)
        (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["complete"] = True
    (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/scr/yuheng/DevOpsBench/multiscale_diffusion")
    p.add_argument("--output", required=True)
    p.add_argument("--groups", type=int, default=4)
    main(p.parse_args())
