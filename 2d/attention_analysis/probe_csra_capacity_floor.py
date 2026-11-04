#!/usr/bin/env python3
"""Measure the output-subspace floor of the three rank-r residual branches.

FP32 continuous-output bound; BF16 rounding is excluded. No fitting/training.
An optional fixed coarse-residual offset is an oracle diagnostic, not a model.
"""
import argparse
import json
from pathlib import Path

import torch

from csra_runtime import predict_adapter
from probe_csra_layer_repair import replacement_callbacks
from probe_rdd_vs_dit_block_reuse import manual_forward, upsample_tokens
from sample_csra_adapter_preview import load_adapter
from train_rdd_csra_adapter import load_frozen_teacher


def output_basis(module):
    if module.architecture not in {"fine_context_v2","teacher_condition_v3"}:
        raise ValueError("Only the three-branch linear-output adapter is supported")
    matrices = [module.low_up.weight, module.high_up.weight, module.fine_up.weight]
    # The inference loader enables TF32. Analyze geometry in FP64 so TF32
    # projection error is not mistaken for a broken mathematical subspace.
    matrix = torch.cat(matrices, dim=1).double()
    u,s,_ = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((s > s.max()*1e-6).sum())
    bias = module.low_up.bias + module.high_up.bias + module.fine_up.bias
    return u[:,:rank], bias.float(), rank


def projection_floor(target, basis, bias, offset=None):
    residual = target.double() - bias.double()
    basis = basis.double()
    if offset is not None:
        residual = residual - offset.double()
    orthogonal = residual - (residual @ basis) @ basis.T
    return float(orthogonal.square().mean())


@torch.inference_mode()
def run(args):
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Do not overwrite probes")
    root=Path(args.data_root)/f"shard_{args.shard}"
    manifest=json.loads((root/"manifest.json").read_text())
    if not manifest["complete"] or not manifest["parity_checked"] or manifest["parity_max_abs"]!=0:
        raise ValueError("Invalid corpus")
    if Path(manifest["base_checkpoint"]).resolve()!=Path(args.base_ckpt).resolve():
        raise ValueError("Wrong teacher")
    device=torch.device("cuda:0");torch.cuda.set_device(device)
    model=load_frozen_teacher(Path(args.base_ckpt),device,"DiT-L/2","rdd")
    adapter,info=load_adapter(Path(args.adapter_ckpt),device)
    if adapter.layers!=manifest["layers"] or Path(info["base_checkpoint"]).resolve()!=Path(args.base_ckpt).resolve():
        raise ValueError("Wrong adapter")
    bases={l:output_basis(adapter.adapters[str(l)]) for l in adapter.layers}
    out.mkdir(parents=True)
    rows=[];images=0;parity=0.
    for item in manifest["files"]:
        if item["split"]!="validation":continue
        if Path(item["name"]).name!=item["name"]:raise ValueError("Unsafe filename")
        data=torch.load(root/item["name"],map_location="cpu",weights_only=True)
        if data["split"]!="validation" or data["trajectory"]!="dense_teacher":
            raise ValueError("Only validation teacher states")
        y=data["labels"].to(device);images+=len(y)
        coarse={l:upsample_tokens(v.to(device),16) for l,v in data["coarse"].items()}
        for i,nfe in enumerate(data["nfe_indices"].tolist()):
            x,t,target=(data[k][i].to(device) for k in ("x","t","target"))
            dense,traces=manual_forward(model,x,t,y,multires=True,capture_layers=set(adapter.layers))
            if not torch.equal(dense,target):raise RuntimeError("Dense replay changed")
            _,student_traces=manual_forward(model,x,t,y,multires=True,capture_layers=set(adapter.layers),
                replacements=replacement_callbacks(adapter,coarse,t),conditioned_replacements=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                condition=model.t_embedder(t)+model.y_embedder(y,False)
                cos,sin=model.get_rope(16,16,device)
            for layer in adapter.layers:
                basis,bias,rank=bases[layer];module=adapter.adapters[str(layer)]
                for kind,trace in [("teacher_input",traces[layer]),("student_input",student_traces[layer])]:
                    h=trace["input"]
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        dense_h=model.blocks[layer](h,condition,cos,sin)
                    exact=(dense_h-h).float()
                    if kind=="teacher_input" and not torch.equal(exact,trace["residual"].float()):
                        raise RuntimeError("Local block replay changed")
                    predicted=predict_adapter(module,coarse[layer],h,t,"fp32",condition=condition)
                    actual_mse=float((predicted-exact).square().mean())
                    floor=projection_floor(exact,basis,bias)
                    # The actual network output must lie in its stated affine subspace.
                    output_floor=projection_floor(predicted,basis,bias)
                    scale=max(1.,float(predicted.square().mean()))
                    # Actual FP32 inference matmuls use TF32: their output is
                    # only approximately in the full-precision weight basis.
                    # A separate numeric probe measured relative MSE ~1e-8.
                    if output_floor > 1e-6*scale or floor > actual_mse+1e-5*max(1.,actual_mse):
                        raise RuntimeError(f"Invalid output-subspace bound: {output_floor=}, {scale=}, {floor=}, {actual_mse=}")
                    parity=max(parity,output_floor/scale)
                    rows.append({"seed":item["seed"],"images":len(y),"t":int(t[0]),"nfe":nfe,
                        "layer":layer,"input_kind":kind,"output_basis_rank":rank,
                        "target_power":float(exact.square().mean()),"actual_mse":actual_mse,
                        "subspace_floor_mse":floor,"coarse_identity_floor_mse":projection_floor(exact,basis,bias,coarse[layer])})
            print(json.dumps({"seed":item["seed"],"nfe":nfe,"rows":len(rows)}),flush=True)
    if not images:raise ValueError("Empty validation")
    result={"complete":True,"config":vars(args),"images":images,"states":images*8,"rows":rows,
        "max_output_subspace_relative_mse":parity,"max_memory_gib":torch.cuda.max_memory_allocated()/2**30,
        "caveat":"Fixed learned output-basis oracle in FP64; TF32/BF16 rounding excluded. Not best retrainable basis or quality/FID."}
    temp=out/"metrics.json.tmp";temp.write_text(json.dumps(result,indent=2));temp.replace(out/"metrics.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("base-ckpt","adapter-ckpt","data-root","output-dir"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--shard",type=int,choices=range(4),required=True)
    run(parser.parse_args())
