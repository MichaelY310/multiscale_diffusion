#!/usr/bin/env python3
"""Disposable actual L/2 per-GPU64 full-backbone backward/Adam test.

One batch only, no checkpoint write. Does not need completed sidecar corpus.
"""
import json
from pathlib import Path
import torch
from csra_joint_distillation import JointCSRAStudent
from csra_output_distillation import prediction_loss
from sample_csra_adapter_preview import load_adapter
from train_rdd_csra_adapter import load_frozen_teacher


torch.manual_seed(20279000)
device=torch.device("cuda:0")
teacher=load_frozen_teacher(Path("checkpoints/RDD-Unified-L/0400000.pt"),device,"DiT-L/2","rdd")
adapter,_=load_adapter(Path("checkpoints/RDD-CSRA-condition-v2-L400K-20260912/best.pt"),device)
net=JointCSRAStudent(teacher,adapter,True)
data=torch.load("eval_results/csra_rollout_data4k_20260912/shard_0/train_0000.pt",weights_only=True)
source=torch.load("eval_results/csra_joint_sources_20260913/shard_0/train_0000.pt",weights_only=True)
batch={"x":data["x"].flatten(0,1).to(device),"t":data["t"].flatten().to(device),
       "target":data["target"].flatten(0,1).to(device),"labels":data["labels"].repeat(8).to(device)}
for r in (8,16):
    for k in ("x","t","target"):
        v=source["sources"][str(r)][k]
        batch[f"source{r}_{k}"]=v.repeat((8,)+(1,)*(v.ndim-1)).to(device)
optimizer=torch.optim.AdamW(net.parameters(),lr=1e-5)
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    fine,middle,coarse=net(batch)
    loss=prediction_loss(fine,batch["target"])[0]+.5*(
        prediction_loss(middle,batch["source16_target"])[0]+
        prediction_loss(coarse,batch["source8_target"])[0])
    loss.backward()
    missing=[n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]
    if missing: raise RuntimeError(missing)
    grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({"step":step+1,"batch":len(batch["t"]),"loss":float(loss),
        "grad_norm":float(grad),"trainable":sum(p.numel() for p in net.parameters()),
        "allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "reserved_gib":torch.cuda.max_memory_reserved()/2**30}),flush=True)
