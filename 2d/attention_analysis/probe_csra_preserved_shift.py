#!/usr/bin/env python3
"""Matched teacher-input/student-input errors for preserved-coarse checkpoints.

Validation only. No training or quality/FID claim. Auxiliary calls must not
change RNG or the real student rollout; exact final-latent parity is checked.
"""
import argparse
import json
from pathlib import Path

import torch

from csra_joint_distillation import PreservedCoarseStudent
from sample_csra_preserved_preview import PreservedRollout
from sample_csra_adapter_preview import sample_rdd, manual_forward
from train_rdd_csra_adapter import CSRAAdapterBank, load_frozen_teacher
from probe_csra_rollout_shift import errors, lowpass


class PreservedShiftProbe:
    def __init__(self, teacher, student, reference):
        self.teacher, self.student, self.reference = teacher, student, reference
        self.wrapper = PreservedRollout(teacher, student)
        self.indices = {int(n): i for i, n in enumerate(reference["nfe_indices"])}
        self.fine_calls, self.rows = 0, []
        self.cache_error = None

    def forward(self, x, t, y):
        output = self.wrapper.forward(x,t,y)
        if x.shape[-1] == 16 and bool(torch.all(t == 300)):
            self.cache_error = max(float((self.wrapper.cache[l].float() -
                self.reference["coarse"][l].to(x.device).float()).abs().max())
                for l in self.student.adapter.layers)
            if self.cache_error != 0:
                raise RuntimeError("Original native coarse cache changed")
        if x.shape[-1] == 32:
            nfe = self.fine_calls
            if nfe in self.indices:
                idx = self.indices[nfe]
                rx = self.reference["x"][idx].to(x.device)
                rt = self.reference["t"][idx].to(t.device)
                if not torch.equal(t,rt) or not torch.equal(y.cpu(), self.reference["labels"]):
                    raise RuntimeError("Reference time or class mismatch")
                if nfe == 0 and not torch.equal(x,rx):
                    raise RuntimeError("The first fine input differs before replacement")
                target = self.reference["target"][idx].to(x.device)
                ref_teacher, _ = manual_forward(self.teacher, rx,t,y,multires=True)
                if not torch.equal(ref_teacher,target):
                    raise RuntimeError("Saved original teacher output not reproduced exactly")
                rng_before = torch.cuda.get_rng_state(x.device)
                student_ref = self.student.stage(rx,t,y,cache=self.wrapper.cache)[0]
                teacher_on, _ = manual_forward(self.teacher,x,t,y,multires=True)
                if not torch.equal(rng_before,torch.cuda.get_rng_state(x.device)):
                    raise RuntimeError("Auxiliary model calls consumed sampling RNG")
                difference = output[:,:4].float()-teacher_on[:,:4].float()
                lp = lowpass(difference)
                self.rows.append({"nfe":nfe,"t":int(t[0]),"images":len(y),
                    "teacher_state_main":errors(student_ref[:,:4],target[:,:4]),
                    "student_state_main":errors(output[:,:4],teacher_on[:,:4]),
                    "teacher_state_variance":errors(student_ref[:,4:],target[:,4:]),
                    "student_state_variance":errors(output[:,4:],teacher_on[:,4:]),
                    "trajectory_state":errors(x,rx),
                    "student_main_low_error_power":float(lp.square().mean()),
                    "student_main_high_error_power":float((difference-lp).square().mean())})
            self.fine_calls += 1
        return output


def load_student(teacher_path, arm, device):
    path=Path(f"checkpoints/RDD-CSRA-{arm}-L400K-20260913/best.pt")
    ck=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    if ck["format"]!="csra_preserved_coarse_v1":
        raise ValueError("Wrong checkpoint format")
    meta=ck["adapter_metadata"]
    adapter=CSRAAdapterBank(meta["layers"],meta["hidden_size"],meta["rank"],meta["adapter_architecture"]).to(device)
    backbone=load_frozen_teacher(teacher_path,device,"DiT-L/2","rdd")
    student=PreservedCoarseStudent(backbone,adapter,False,False)
    student.load_state_dict(ck["student"])
    return student.eval().requires_grad_(False),{"path":str(path),"step":ck["step"]}


def main(args):
    out=Path(args.output_dir)
    if (out/"metrics.json").exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    root=Path(args.data_root)/f"shard_{args.shard}"
    manifest=json.loads((root/"manifest.json").read_text())
    if not manifest["complete"] or not manifest["parity_checked"] or manifest["parity_max_abs"]!=0:
        raise ValueError("Incomplete or non-parity-checked dataset")
    base=Path("checkpoints/RDD-Unified-L/0400000.pt")
    if Path(manifest["base_checkpoint"]).resolve()!=base.resolve():
        raise ValueError("Wrong dataset teacher")
    items=[i for i in manifest["files"] if i["split"]=="validation"][:args.max_files]
    if not items:
        raise ValueError("No validation examples")
    device=torch.device("cuda:0")
    teacher=load_frozen_teacher(base,device,"DiT-L/2","rdd")
    arms={}
    for arm in ("preserved_frozen","preserved_joint"):
        student,info=load_student(base,arm,device)
        results=[]
        for item in items:
            if Path(item["name"]).name!=item["name"]:
                raise ValueError("Unsafe dataset filename")
            reference=torch.load(root/item["name"],map_location="cpu",weights_only=True)
            if reference["split"]!="validation" or reference["trajectory"]!="dense_teacher" or reference["layers"]!=student.adapter.layers:
                raise ValueError("Invalid validation reference")
            probe=PreservedShiftProbe(teacher,student,reference)
            final,_=sample_rdd(probe,item["images"],[80,80,90],device,item["seed"])
            expected,_=sample_rdd(PreservedRollout(teacher,student),item["images"],[80,80,90],device,item["seed"])
            parity=float((final-expected).abs().max())
            if parity!=0 or probe.fine_calls!=90 or len(probe.rows)!=8:
                raise RuntimeError("Probe changed or incompletely observed the rollout")
            results.append({"seed":item["seed"],"images":item["images"],"rows":probe.rows,
                "probe_parity_max_abs":parity,"coarse_cache_max_abs":probe.cache_error})
            print(json.dumps({"arm":arm,"seed":item["seed"],"parity":parity,
                "teacher_state_mean_mse":sum(r["teacher_state_main"]["mse"] for r in probe.rows)/8,
                "student_state_mean_mse":sum(r["student_state_main"]["mse"] for r in probe.rows)/8}),flush=True)
        arms[arm]={"checkpoint":info,"results":results}
        del probe,student,final,expected
        torch.cuda.empty_cache()
    payload={"complete":True,"shard":args.shard,"config":vars(args),"arms":arms,
        "max_memory_gib":torch.cuda.max_memory_allocated()/2**30,
        "warning":"Validation trajectories already used for model selection; no training, FID, VAE or speed claim. Auxiliary teacher calls are diagnostic only."}
    temp=out/"metrics.json.tmp"
    temp.write_text(json.dumps(payload,indent=2))
    temp.replace(out/"metrics.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",default="eval_results/csra_rollout_data4k_20260912")
    parser.add_argument("--shard",type=int,required=True)
    parser.add_argument("--max-files",type=int,default=1)
    parser.add_argument("--output-dir",required=True)
    main(parser.parse_args())
