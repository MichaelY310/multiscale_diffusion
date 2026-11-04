#!/usr/bin/env python3
"""Can source parent cells identify where fine residual repair is needed?

Read-only LOCAL residual diagnostic on original teacher inputs. Oracle uses
the answer; NOT a deployable algorithm, end-to-end repair, or speed benchmark.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from csra_runtime import predict_adapter
from probe_rdd_vs_dit_block_reuse import manual_forward, upsample_tokens
from sample_csra_adapter_preview import load_adapter
from train_rdd_csra_adapter import load_frozen_teacher


@torch.inference_mode()
def main(args):
    device=torch.device("cuda:0")
    base=Path("checkpoints/RDD-Unified-L/0400000.pt")
    teacher=load_frozen_teacher(base,device,"DiT-L/2","rdd")
    adapter,_=load_adapter(Path("checkpoints/RDD-CSRA-condition-v2-L400K-20260912/best.pt"),device)
    rows=[]
    for shard in (0,1):
        folder=Path("eval_results/csra_rollout_data4k_20260912")/f"shard_{shard}"
        manifest=json.loads((folder/"manifest.json").read_text())
        if not manifest["complete"]: raise ValueError("Incomplete source")
        for item in manifest["files"]:
            if item["split"]!="validation": continue
            data=torch.load(folder/item["name"],weights_only=True)
            y=data["labels"].to(device)
            for index in range(8):
                x,t,target=(data[k][index].to(device) for k in ("x","t","target"))
                output,traces=manual_forward(teacher,x,t,y,multires=True,capture_layers=set(adapter.layers))
                if not torch.equal(output,target): raise RuntimeError("Teacher replay changed")
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    condition=teacher.t_embedder(t)+teacher.y_embedder(y,False)
                for layer in adapter.layers:
                    coarse=data["coarse"][layer].to(device)
                    trace=traces[layer]
                    predicted=predict_adapter(adapter.adapters[str(layer)],upsample_tokens(coarse,16),
                        trace["input"],t,"fp32",condition=condition).to(trace["input"].dtype)
                    err=(predicted.float()-trace["residual"].float()).square().mean(-1)
                    # 16x16 fine tokens -> 8x8 parents, four children per parent.
                    parent_error=err.reshape(len(y),8,2,8,2).mean((2,4)).flatten(1)
                    energy=coarse.float().square().mean(-1)
                    for fraction in (.25,.5):
                        k=round(64*fraction)
                        oracle=parent_error.topk(k,dim=1).indices
                        selected=energy.topk(k,dim=1).indices
                        denominator=parent_error.sum(1).clamp_min(1e-12)
                        rows.append({"seed":item["seed"],"layer":layer,"t":int(t[0]),
                            "fraction":fraction,"images":len(y),
                            "coarse_energy_captured_error":float((parent_error.gather(1,selected).sum(1)/denominator).mean()),
                            "oracle_captured_error":float((parent_error.gather(1,oracle).sum(1)/denominator).mean()),
                            "random_expected_captured_error":fraction})
                print(json.dumps({"seed":item["seed"],"t":int(t[0]),"rows":len(rows)}),flush=True)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    summary={str(f):{key:sum(r[key]*r["images"] for r in rows if r["fraction"]==f)/
        sum(r["images"] for r in rows if r["fraction"]==f) for key in
        ("coarse_energy_captured_error","oracle_captured_error","random_expected_captured_error")}
        for f in (.25,.5)}
    out.write_text(json.dumps({"complete":True,"images":64,"states":512,"layers":adapter.layers,
        "scope":"local residual on teacher inputs, not final image; oracle requires fine teacher",
        "peak_gib":torch.cuda.max_memory_allocated()/2**30,"summary":summary,"rows":rows},indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    main(p.parse_args())
