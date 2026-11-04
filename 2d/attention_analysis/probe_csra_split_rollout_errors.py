"""Read-only16-pair diagnostic per arm. No new training or checkpoint edits.

Compare student error on original vs own rollout states; test memory spatial
alignment and the native-coarse attention branch without changing sampled path.
All errors reference ORIGINAL RDD predictions, not unknowable true epsilon.
"""
import argparse
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from sample_csra_split_quality256 import load as load800
from sample_csra_split1600_quality256 import load as load1600
from sample_csra_adapter_preview import manual_forward,sample_rdd
from csra_skip_schedule import ScheduledPrefixRollout
from sample_csra_skip_schedules import sha,tensor_sha,ROOT
from train_rdd_csra_adapter import load_frozen_teacher
from probe_csra_prefix_information import BASE,write_json

ARMS=('split800','split1600','front800','front1600')
CHECKS=set(range(0,90,10))|{89}


def errors(a,b):
    error=(a[:,:4].float()-b[:,:4].float())
    low=F.interpolate(F.avg_pool2d(error,2),scale_factor=2,mode='nearest')
    high=error-low
    mse=lambda x:x.square().flatten(1).mean(1).cpu().tolist()
    return dict(epsilon_mse=mse(error),low_band_mse=mse(low),high_band_mse=mse(high),
        teacher_epsilon_power=mse(b[:,:4].float()),
        variance_head_mse=mse(a[:,4:].float()-b[:,4:].float()))


class OriginalCapture:
    def __init__(self,teacher):self.teacher=teacher;self.records=[];self.cache=None
    def forward(self,x,t,y):
        if x.shape[-1]==16 and int(t[0])==300:
            out,trace=manual_forward(self.teacher,x,t,y,multires=True,capture_layers={13})
            self.cache={13:trace[13]['output'].detach().clone()}
        else:out,_=manual_forward(self.teacher,x,t,y,multires=True)
        if x.shape[-1]==32:
            self.records.append(dict(x=x.detach().cpu(),t=t.detach().cpu(),y=y.detach().cpu(),
                prediction=out.detach().cpu()))
        return out


def perturbations(student,batch,teacher_prediction):
    memory=batch['coarse'][13]
    altered={'wrong_image':memory.roll(1,0),'mean_cell':memory.mean(1,keepdim=True).expand_as(memory),
        'shift_cells':memory.reshape(memory.shape[0],8,8,-1).roll((1,1),(1,2)).reshape_as(memory)}
    results={}
    for name,value in altered.items():
        results[name]=errors(student({**batch,'coarse':{13:value}}),teacher_prediction)
    hooks=[reader.read_out.register_forward_hook(lambda module,args,out:torch.zeros_like(out)) for reader in student.readers]
    try:results['zero_attention_branch']=errors(student(batch),teacher_prediction)
    finally:
        for hook in hooks:hook.remove()
    return results


class OwnStateProbe(ScheduledPrefixRollout):
    def __init__(self,teacher,student,dense):
        super().__init__(teacher,student,range(90));self.dense=dense;self.rows=[]
    def forward(self,x,t,y):
        out=super().forward(x,t,y)
        if x.shape[-1]==32:
            index=self.fine_calls-1;original=self.dense.records[index]
            if int(t[0])!=int(original['t'][0]):raise ValueError('Timestep mapping mismatch')
            teacher,_=manual_forward(self.teacher,x,t,y,multires=True)
            row=dict(index=index,t=int(t[0]),on_student_state=errors(out,teacher),
                state_difference_mse=(x.float()-original['x'].to(x.device).float()).square().flatten(1).mean(1).cpu().tolist())
            if index in CHECKS:
                batch=dict(x=x,t=t,labels=y,coarse=self.cache)
                row['perturbations']=perturbations(self.student,batch,teacher)
            self.rows.append(row)
        return out


@torch.inference_mode()
def main(a):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1','2','3'):raise ValueError('Single physical0-3 only')
    dev=torch.device('cuda:0');torch.set_num_threads(2)
    if torch.cuda.mem_get_info(dev)[0]<20*2**30:raise RuntimeError('Need20GiB free; no displacement')
    torch.cuda.set_per_process_memory_fraction(.20,dev);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.root)/'split_rollout_errors'/a.arm;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='csra_split_readonly_rollout_diagnostic_v1',arm=a.arm,script_sha256=sha(__file__),
        images=16,seeds=[85660915,85670915],batch=8,steps=[80,80,90],cfg=1.,boundary='noise_filling',
        notes='Teacher prediction error, NOT true epsilon/quality; low band=2x2 latent block-average projection, not semantic frequency.',
        checked_fine_indices=sorted(CHECKS))
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Different existing protocol')
    if (out/'metrics.json').exists():print('Exact probe complete');return
    write_json(out/'protocol.json',protocol)
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    name='split3_3' if a.arm.startswith('split') else 'front0_6'
    student,config=(load1600 if a.arm.endswith('1600') else load800)(a.root,name,'aligned',dev)
    output=[]
    for seed in protocol['seeds']:
        dense=OriginalCapture(teacher);z0,_=sample_rdd(dense,8,[80,80,90],dev,seed)
        if len(dense.records)!=90 or dense.cache is None:raise ValueError('Missing original capture')
        teacher_rows=[]
        for i,r in enumerate(dense.records):
            batch=dict(x=r['x'].to(dev),t=r['t'].to(dev),labels=r['y'].to(dev),coarse=dense.cache)
            reference=r['prediction'].to(dev)
            row=dict(index=i,t=int(r['t'][0]),on_original_state=errors(student(batch),reference))
            if i in CHECKS:row['perturbations']=perturbations(student,batch,reference)
            teacher_rows.append(row)
        own=OwnStateProbe(teacher,student,dense);z1,_=sample_rdd(own,8,[80,80,90],dev,seed)
        torch.testing.assert_close(own.cache[13],dense.cache[13],atol=0,rtol=0)
        # Extra teacher/perturbation calls must never affect actual RNG/path.
        standard=ScheduledPrefixRollout(teacher,student,range(90));zcheck,_=sample_rdd(standard,8,[80,80,90],dev,seed)
        torch.testing.assert_close(z1,zcheck,atol=0,rtol=0)
        output.append(dict(seed=seed,teacher_path=teacher_rows,student_path=own.rows,
            final_latent_mse=(z1.float()-z0.float()).square().flatten(1).mean(1).cpu().tolist(),
            original_latent_sha256=tensor_sha(z0),diagnostic_vs_standard_max_abs=0.,coarse_parity_max_abs=0.))
        write_json(out/'progress.json',dict(protocol=protocol,config=config,records=output))
        print(json.dumps(dict(arm=a.arm,seed=seed,complete=True)),flush=True)
    write_json(out/'metrics.json',dict(complete=True,protocol=protocol,config=config,records=output,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--arm',choices=ARMS,required=True)
    main(p.parse_args())
