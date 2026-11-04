"""Screen clean cross-resolution VAE residual reuse; NOT a trained new codec.

Uses existing real RDD parent/final predictions. All serial parent capture,
transport and fine decoding costs are charged. A speed result here alone does
not prove novelty, acceptable quality, or improvement over Flash-VAED.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

import torch
import torch.nn.functional as F


def similarity(a, b):
    a, b = a.float(), b.float()
    return dict(cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12)),
                relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)))


def main(a):
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"metrics.json").exists():
        raise FileExistsError("Use a fresh output directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "2", "3"):
        raise ValueError("Exactly one of physical GPUs 0--3 is allowed")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 65*2**30:
        raise RuntimeError("Need 65GiB free; never evict another process")
    torch.cuda.set_per_process_memory_fraction(.65)
    sys.path.insert(0, str(root/"Wan2.1"))
    from wan.modules.vae import WanVAE, ResidualBlock, CausalConv3d
    vae = WanVAE(z_dim=16, vae_pth=str(root/"pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
                 dtype=torch.bfloat16, device=torch.device("cuda:0"))
    modules = {n:m for n,m in vae.model.decoder.named_modules()
               if isinstance(m, ResidualBlock) and m.in_dim == m.out_dim}
    originals = {n:m.forward for n,m in modules.items()}
    cache, counters, stats = {}, {}, []
    mode, selected = "none", set()

    def install(name, module):
        def forward(self, x, feat_cache=None, feat_idx=None):
            k = counters.get(name, 0)
            counters[name] = k+1
            key = (name, k)
            if mode == "reuse" and name in selected:
                parent = cache[key]
                if parent.shape[2] != x.shape[2]:
                    raise AssertionError("This probe only changes spatial resolution")
                lifted = F.interpolate(parent, size=x.shape[2:], mode="nearest")
                if feat_cache is not None:
                    feat_idx[0] += sum(isinstance(l, CausalConv3d) for l in self.residual)
                return x + lifted
            y = originals[name](x, feat_cache, feat_idx)
            if mode == "capture" and name in selected:
                cache[key] = (y-x).detach().clone()
            elif mode == "audit" and key in cache:
                delta = y-x
                lifted = F.interpolate(cache[key], size=delta.shape[2:], mode="nearest")
                low = F.avg_pool3d(delta.float(), (1,2,2))
                stats.append(dict(layer=name, chunk=k, shape=list(delta.shape),
                                  full=similarity(lifted, delta),
                                  low=similarity(cache[key], low)))
            return y
        module.forward = types.MethodType(forward, module)
    for name, module in modules.items():
        install(name, module)

    report = dict(complete=False, warning=__doc__, gpu=torch.cuda.get_device_name(),
                  physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"], torch=torch.__version__,
                  selected_eligible_layers=list(modules), parent_source=a.parent_source, rows=[])
    from PIL import Image
    def preview(v, label):
        v = ((v.float().clamp(-1,1)+1)*127.5).round().byte().cpu()
        frames = [v[:,i].permute(1,2,0).numpy() for i in (0, v.shape[1]//2, v.shape[1]-1)]
        import numpy as np
        Image.fromarray(np.concatenate(frames, axis=1)).save(out/f"{label}.png")

    def decode(z, stage_mode):
        nonlocal mode
        mode = stage_mode
        counters.clear()
        return vae.decode([z])[0]

    for row in range(a.rows):
        payload = torch.load(Path(a.inputs)/f"contract_row{row}.pt", map_location="cpu", weights_only=False)
        print("INPUT_KEYS", list(payload), flush=True)
        parent = payload["parent"].to("cuda").squeeze(0)
        final = payload["final"].to("cuda").squeeze(0)
        if a.parent_source == "pooled_final":
            parent = F.avg_pool3d(final.unsqueeze(0), (1,2,2))[0]
        selected = set(modules)
        cache.clear()
        with torch.inference_mode():
            pv = decode(parent, "capture")
            stats.clear()
            reference = decode(final, "audit")
            audit = list(stats)
            preview(pv, f"row{row}_parent")
            preview(reference, f"row{row}_native")
            cache_bytes = sum(v.numel()*v.element_size() for v in cache.values())
            variants = {"native": [], "last2": list(modules)[-2:],
                        "last4": list(modules)[-4:], "last6": list(modules)[-6:]}
            records, quality = [], {}
            for variant, names in variants.items():
                selected = set(names)
                def run():
                    cache.clear()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if names:
                        unused = decode(parent, "capture")
                        del unused
                    torch.cuda.synchronize()
                    boundary = time.perf_counter()
                    v = decode(final, "reuse" if names else "none")
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    return v, dict(parent_seconds=boundary-start, fine_seconds=end-boundary,
                                   serial_seconds=end-start,
                                   peak_gib=torch.cuda.max_memory_allocated()/2**30,
                                   cache_gib=sum(t.numel()*t.element_size() for t in cache.values())/2**30)
                for _ in range(2):
                    video, measurement = run()
                    del video
                video, measurement = run()
                mse = float((video-reference).square().mean())
                quality[variant] = dict(pixel_rmse=mse**.5,
                    psnr_vs_native_db=float(-10*torch.log10(torch.tensor(mse/4)).item()) if mse else None)
                preview(video, f"row{row}_{variant}")
                del video
                for repeat in range(a.repeats):
                    video, measurement = run()
                    records.append(dict(variant=variant, repeat=repeat, **measurement))
                    del video
                print(json.dumps(dict(row=row, variant=variant, quality=quality[variant], last=measurement)), flush=True)
            result = dict(row=row, parent_shape=list(parent.shape), final_shape=list(final.shape),
                          full_cache_gib=cache_bytes/2**30, audit=audit, quality=quality,
                          variants=variants, measurements=records,
                          medians={v:statistics.median(r["serial_seconds"] for r in records if r["variant"]==v)
                                   for v in variants})
            report["rows"].append(result)
            (out/"progress.json").write_text(json.dumps(report, indent=2))
        del parent, final, pv, reference
        cache.clear()
        torch.cuda.empty_cache()
    report["complete"] = True
    (out/"metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/scr/yuheng/DevOpsBench/multiscale_diffusion")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--parent-source", choices=("progressive", "pooled_final"), default="progressive")
    main(p.parse_args())
