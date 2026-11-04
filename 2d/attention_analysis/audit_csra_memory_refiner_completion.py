"""Audit fixed400 refiner/control comparison and bundle non-weight artifacts."""
import json
from pathlib import Path
import tarfile

import torch
from PIL import Image

from train_csra_memory_refiner import ROOT, PROTOCOL, CHECKPOINT, source_hashes, dataset_hashes
from csra_state_replay_data import ROOT as OLD_ROOT, PARENT, PARENT_SHA, BASE_SHA, QUALITY_SEED, SEED
from probe_csra_front_keep import BASE, file_sha, write_json


def main():
    if file_sha(BASE)!=BASE_SHA or file_sha(PARENT)!=PARENT_SHA: raise ValueError('Changed base/parent')
    ck=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True)
    done=json.loads((CHECKPOINT.parent/'complete.json').read_text())
    exp=json.loads((CHECKPOINT.parent/'experiment.json').read_text())
    control_folder=Path('checkpoints/RDD-CSRA-state-replay-teacher_only-L400K-20260914')
    control=json.loads((control_folder/'experiment.json').read_text())
    code,data=source_hashes(),dataset_hashes()
    if (ck['protocol']!=PROTOCOL or ck['updates']!=400 or ck['smoke'] or ck['arm']!='teacher_only'
        or not done['complete'] or done['smoke'] or done['updates']!=400
        or ck['parent_sha256']!=PARENT_SHA or ck['base_sha256']!=BASE_SHA
        or ck['code_sha256']!=code or ck['data_manifests_sha256']!=data
        or ck['validation']!=done['final_validation'] or len(ck['rank_rng_states'])!=4
        or not ck['optimizer']['state']): raise ValueError('Training/checkpoint contract mismatch')
    for k in ('global_batch','per_gpu_batch','accumulation','updates_planned','seed','optimizer','initial_validation'):
        if exp[k]!=control[k]: raise ValueError(f'Unmatched control:{k}')
    if exp['global_batch']!=256 or exp['per_gpu_batch']!=64 or exp['seed']!=SEED:
        raise ValueError('Wrong batch/seed')
    if exp['parent']['zero_init_actual_batch64_max_abs']!=0: raise ValueError('Missing actual64 parity')
    for kind,new_path,old_path in [('formal',CHECKPOINT.parent,control_folder),
        ('smoke',ROOT/'train_smoke',OLD_ROOT/'smoke_teacher_only')]:
        global_indices={0:[],1:[]}
        for rank in range(4):
            a=json.loads((new_path/f'trace_rank{rank}.json').read_text())
            b=json.loads((old_path/f'trace_rank{rank}.json').read_text())
            if a!=b: raise ValueError('Data indices/labels/t differ')
            for r in a['trace']:
                if len(r['indices'])!=64 or any(r['state_role']): raise ValueError('Not actual64 teacher-only')
                global_indices[r['update']].extend(r['indices'])
        if any(len(set(ids))!=256 for ids in global_indices.values()): raise ValueError('Repeated global samples')
    w=ck['student']['memory_refiner.output.weight'].float()
    learned=dict(final_correction_weight_rms=float(w.square().mean().sqrt()),
                 final_correction_weight_max_abs=float(w.abs().max()))
    if learned['final_correction_weight_max_abs']==0: raise ValueError('Refiner remained identically zero')
    digest=file_sha(CHECKPOINT);control_digest=file_sha(control_folder/'latest.pt')
    del ck
    seen=set();classifier=None
    for shard in range(4):
        p=ROOT/'quality'/f'shard_{shard}'
        d=json.loads((p/'metrics.json').read_text())
        old=json.loads((OLD_ROOT/'quality'/f'shard_{shard}'/'metrics.json').read_text())
        if (not d['complete'] or d['protocol']!=PROTOCOL or d['seed']!=QUALITY_SEED+shard*10000
            or d['steps']!=[80,80,90] or d['cfg']!=1. or d['boundary']!='noise_filling'
            or d['updates']!=400 or d['coarse_endpoint_max_abs']!=0 or not d['original_png_parity']
            or d['config']['checkpoint_sha256']!=digest or d['control']['checkpoint_sha256']!=control_digest
            or d['labels']!=old['labels'] or d['control_metrics_sha256']!=file_sha(OLD_ROOT/'quality'/f'shard_{shard}'/'metrics.json')):
            raise ValueError('Unmatched generation contract')
        if classifier is None: classifier=d['classifier_sha256']
        if d['classifier_sha256']!=classifier or classifier!=old['classifier_sha256']: raise ValueError('Different classifier')
        original_rows={(r['image'],r['variant']):r for r in old['rows']}
        if len(d['rows'])!=192 or not (p/'comparison_grid.png').is_file(): raise ValueError('Missing quality images/grid')
        for row in d['rows']:
            key=(row['image'],row['variant'])
            if key in seen or row['variant'] not in ('original','teacher_only','refined'): raise ValueError('Bad row')
            seen.add(key)
            if row['variant']!='refined' and row!=original_rows[key]: raise ValueError('Changed control row')
            image=p/row['variant']/f"{row['local_image']:03d}.png"
            if file_sha(image)!=row['image_sha256']: raise ValueError('PNG hash changed')
            with Image.open(image) as im:
                if im.size!=(256,256): raise ValueError('Wrong image size')
                im.verify()
    if len(seen)!=768: raise ValueError('Missing paired768 samples')
    result=dict(complete=True,protocol=PROTOCOL,images_sha_verified=768,
        checkpoint_sha256=digest,control_checkpoint_sha256=control_digest,code_sha256=code,
        data_manifests_sha256=data,paired_traces_and_initial_validation=True,
        original_coarse_endpoint_and_png_parity=True,learned=learned,
        warning='Audit, not a quality acceptance or RDD-exclusive novelty proof. Dev set reused.')
    write_json(ROOT/'completion_audit.json',result)
    files={p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.png')}
    files.update(CHECKPOINT.parent.glob('*.json'))
    files.update(Path('eval_results')/name for name in ('csra_memory_refiner_cost_launcher_20260914.log',
                                                     'csra_memory_refiner_train_launcher_20260914.log'))
    archive=Path('eval_results/csra_memory_refiner_review_20260914.tar.gz')
    if archive.exists(): raise FileExistsError('Preserve completed bundle')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files): tar.add(p,arcname=p.as_posix(),recursive=False)
    receipt=dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive),files=len(files))
    write_json(ROOT/'bundle_receipt.json',receipt)
    print(json.dumps(dict(audit=result,receipt=receipt),indent=2))


if __name__=='__main__': main()
