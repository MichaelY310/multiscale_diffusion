#!/usr/bin/env python3
"""Matched operator probe, NOT a video-model speed/quality benchmark.

Same synthetic features, H/r, FiLM/SILU/weights/precision; 100 ABBA samples.
Preprojection, parent-map setup and repeated fine calls reported separately.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csra_coarse_transport_kernel import lift_add, nearest_parent_map


def elapsed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), value


def stats(values):
    values = sorted(values)
    return {"median_ms": statistics.median(values), "p10_ms": values[len(values)//10],
            "p90_ms": values[9*len(values)//10], "repeats": len(values)}


@torch.inference_mode()
def main(args):
    torch.manual_seed(20279000)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    device = torch.device("cuda:0")
    records = []
    # Wan-1.3B width; token grids AFTER patch embedding, not pixel dimensions.
    for fine_grid in ((8,28,52), (8,56,104), (20,45,80)):
        for factors in ((1,2,2),(2,2,2)):
            coarse_grid = tuple(math.ceil(f/c) for f,c in zip(fine_grid,factors))
            nf,nc,h,r = math.prod(fine_grid),math.prod(coarse_grid),1536,64
            parent = nearest_parent_map(coarse_grid,fine_grid,device)
            coarse = torch.randn(1,nc,h,device=device)
            fine = torch.randn(1,nf,h,device=device)
            wd = torch.randn(r,h,device=device)/math.sqrt(h)
            wu = torch.randn(h,r,device=device)/math.sqrt(r)
            bias = torch.randn(h,device=device)
            scale, shift = torch.randn(1,1,r,device=device)*.1, torch.randn(1,1,r,device=device)*.1
            cold_setup_ms, z = elapsed(lambda: F.linear(coarse,wd))
            expanded_z = z.index_select(1,parent)
            def projected_low(z):
                return F.linear(F.silu(z*(1+scale)+shift),wu,bias)
            def old():
                expanded = coarse.index_select(1,parent)
                return projected_low(F.linear(expanded,wd)) + fine
            def hoisted_uncached():
                low = projected_low(F.linear(coarse,wd))
                return lift_add(low,fine,parent)
            def hoisted_cached():
                return lift_add(projected_low(z),fine,parent)
            def expanded_cached():
                return projected_low(expanded_z) + fine
            def hoisted_torch():
                return projected_low(z).index_select(1,parent) + fine
            methods = {"expanded_reference": old, "coarse_uncached_fused": hoisted_uncached,
                       "coarse_preprojected_fused": hoisted_cached,
                       "expanded_preprojected": expanded_cached,
                       "coarse_preprojected_torch": hoisted_torch}
            ref = old()
            errors = {}
            for name, fn in methods.items():
                output = fn()
                diff = output-ref
                errors[name] = {"max_abs": float(diff.abs().max()),
                    "relative_l2": float(diff.norm()/ref.norm())}
                if args.tf32 == "off":
                    torch.testing.assert_close(output,ref,atol=3e-5,rtol=3e-5)
                elif errors[name]["relative_l2"] > .002:
                    raise RuntimeError(f"Excessive TF32 error: {errors[name]}")
            # Fused gather/add itself must be exactly equal in FP32.
            low = projected_low(z)
            torch.testing.assert_close(lift_add(low,fine,parent), low.index_select(1,parent)+fine,
                                       rtol=0,atol=0)
            for _ in range(10):
                for fn in methods.values(): fn()
            # Exclude BLAS initialization from steady-state setup amortization;
            # retain the actual cold setup separately. Measure BOTH methods'
            # preprojection cost, so caching is not an unfair advantage.
            setup_ms = statistics.median(elapsed(lambda: F.linear(coarse,wd))[0] for _ in range(30))
            expanded_setup_ms = statistics.median(elapsed(lambda: F.linear(coarse.index_select(1,parent),wd))[0] for _ in range(30))
            times = {name: [] for name in methods}
            names = list(methods)
            for i in range(args.repeats):
                order = names if i%2 == 0 else names[::-1]
                for name in order:
                    ms, result = elapsed(methods[name])
                    times[name].append(ms)
            measured = {k: stats(v) for k,v in times.items()}
            base_ms = measured["expanded_reference"]["median_ms"]
            optimized_ms = measured["coarse_preprojected_fused"]["median_ms"]
            row = {"fine_token_grid": fine_grid, "coarse_token_grid": coarse_grid,
                   "fine_tokens": nf,"coarse_tokens": nc,"width":h,"rank":r,"batch":1,
                   "dtype":"float32", "timings":measured,"parity":errors,
                   "initial_projection_ms":setup_ms,
                   "cold_projection_ms":cold_setup_ms,
                   "expanded_projection_ms":expanded_setup_ms,
                   "branch_speedup_including_setup_once": {str(n): n*base_ms/(setup_ms+n*optimized_ms) for n in (1,3,90)},
                   "branch_speedup_vs_expanded_cache_including_setup": {str(n):
                       (expanded_setup_ms+n*measured["expanded_preprojected"]["median_ms"])/(setup_ms+n*optimized_ms)
                       for n in (1,3,90)},
                   "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
            records.append(row)
            print(json.dumps(row),flush=True)
            del coarse,fine,wd,wu,bias,z,expanded_z,ref,low,result,output,diff
            torch.cuda.empty_cache()
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"complete":True,"scope":"coarse branch only; not model latency",
       "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"tf32":args.tf32=="on",
       "shared_gpu_warning":True,"records":records},indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--repeats",type=int,default=100)
    p.add_argument("--tf32",choices=["on","off"],default="on")
    main(p.parse_args())
