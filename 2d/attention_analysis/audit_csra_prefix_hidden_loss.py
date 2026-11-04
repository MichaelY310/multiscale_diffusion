"""Audit frozen3200 checkpoints and calibrate one TRAIN-only auxiliary weight."""
import argparse
import json
from pathlib import Path
import torch
from csra_prefix_hidden_loss import teacher_prefix_hidden,hidden_loss_per_image
from csra_prefix_probe_math import split_error
from sample_csra_prefix_quality_pilot import load_student
from probe_csra_prefix_information import BASE,TIMES,records,load_record,write_json
from train_rdd_csra_adapter import load_frozen_teacher


@torch.inference_mode()
def run(rank):
    root=Path('eval_results/csra_prefix8_hidden_audit_20260913')
    if (root/f'shard_{rank}.json').exists():raise FileExistsError('Already complete')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0]<18*2**30:raise RuntimeError('Need18GiB free')
    entries,fingerprints,_=records(Path('eval_results/csra_prefix8_information_20260913'))
    entries=[e for e in entries if e[0].parent.name==f'shard_{rank}']
    chosen=[]
    for role in ('fit','validation'):
        chosen.extend([e for e in entries if e[1]['role']==role][:2])
    if len(chosen)!=4:raise ValueError('Expected16train+16dev images/shard')
    teacher=load_frozen_teacher(BASE,device,'DiT-L/2','rdd');rows=[];configs={}
    for arm in ('aligned','zero'):
        student,configs[arm]=load_student(arm,device)
        for entry in chosen:
            rec=load_record(entry)
            src=Path('eval_results/csra_rollout_data4k_20260912')/entry[0].parent.name/entry[0].name
            original=torch.load(src,map_location='cpu',weights_only=True)
            torch.testing.assert_close(original['labels'],rec['labels'],atol=0,rtol=0)
            for ti,t in enumerate(TIMES):
                batch=dict(x=original['x'][ti].to(device),t=original['t'][ti].to(device),
                    labels=rec['labels'].to(device),coarse={13:rec['coarse'].to(device)})
                rng=torch.cuda.get_rng_state(device)
                pred,hidden=student(batch,return_hidden=True)
                torch.testing.assert_close(pred,student(batch),atol=0,rtol=0)
                target=teacher_prefix_hidden(teacher,batch)
                torch.testing.assert_close(target,rec['target'][ti].to(device),atol=0,rtol=0)
                if not torch.equal(rng,torch.cuda.get_rng_state(device)):raise RuntimeError('Changed RNG')
                err=hidden.float()-target.float();low,high=split_error(err)
                nmse=hidden_loss_per_image(hidden,target)
                cosine=torch.nn.functional.cosine_similarity(hidden.float().flatten(1),target.float().flatten(1),dim=1)
                main=(pred[:,:4].float()-original['target'][ti,:,:4].to(device).float()).square().mean((1,2,3))
                for i in range(8):
                    rows.append(dict(arm=arm,role=entry[1]['role'],seed=rec['seed'],image=i,t=t,
                        hidden_nmse=float(nmse[i]),hidden_cosine=float(cosine[i]),main_mse=float(main[i]),
                        hidden_low_mse=float(low[i].square().mean()),hidden_high_mse=float(high[i].square().mean())))
            print(json.dumps(dict(rank=rank,arm=arm,role=entry[1]['role'],seed=rec['seed'])),flush=True)
        del student;torch.cuda.empty_cache()
    if len(rows)!=512:raise ValueError('Incomplete records')
    write_json(root/f'shard_{rank}.json',dict(complete=True,rank=rank,rows=rows,configs=configs,
        fingerprints=fingerprints,teacher_feature_parity_max_abs=0.,forward_parity_max_abs=0.,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))


def aggregate():
    root=Path('eval_results/csra_prefix8_hidden_audit_20260913');shards=[];rows=[]
    for rank in range(4):
        m=json.loads((root/f'shard_{rank}.json').read_text())
        if not m['complete'] or m['rank']!=rank or len(m['rows'])!=512:raise ValueError('Incomplete shard')
        if shards and any(m[k]!=shards[0][k] for k in ('configs','fingerprints')):raise ValueError('Mixed checkpoints/data')
        shards.append(m);rows.extend(m['rows'])
    if len({(r['arm'],r['role'],r['seed'],r['image'],r['t']) for r in rows})!=2048:raise ValueError('Duplicates')
    summaries={}
    for role in ('fit','validation'):
        summaries[role]={}
        for arm in ('aligned','zero'):
            select=[r for r in rows if r['arm']==arm and r['role']==role]
            summaries[role][arm]={k:sum(r[k] for r in select)/len(select) for k in
                ('main_mse','hidden_nmse','hidden_cosine','hidden_low_mse','hidden_high_mse')}
            summaries[role][arm]['states']=len(select)
    train=[r for r in rows if r['role']=='fit']
    coefficient=.05*sum(r['main_mse'] for r in train)/sum(r['hidden_nmse'] for r in train)
    if not 0<coefficient<1:raise ValueError('Suspicious calibration; inspect manually')
    write_json(root/'metrics.json',dict(complete=True,summaries=summaries,coefficient=coefficient,rows=rows,
        configs=shards[0]['configs'],fingerprints=shards[0]['fingerprints'],calibration_role='fit_only',
        calibration_initial_fraction=.05,peak_memory_gib=max(s['peak_memory_gib'] for s in shards),
        warning='64train and64development trajectoriesx8timesx2arms. Coefficient uses only64training trajectories, pooledarms. Student/teacher hidden coordinates need not remain identical after fine-tuning: discrepancy is a candidate training regularizer, NOT proven error cause. Same-input original frozen teacher first8 blocks are training-only. No new training, images, FID or speed in this audit.'))
    print(json.dumps(dict(summaries=summaries,coefficient=coefficient),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=range(4));p.add_argument('--aggregate',action='store_true')
    a=p.parse_args()
    if a.aggregate:aggregate()
    elif a.rank is not None:run(a.rank)
    else:p.error('Need --rank or --aggregate')
