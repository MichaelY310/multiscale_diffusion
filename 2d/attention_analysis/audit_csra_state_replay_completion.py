"""Read-only evidence audit plus compact bundle; never includes weights/data.pt."""
import json
from pathlib import Path
import tarfile

import torch
from PIL import Image

from csra_state_replay_data import ROOT, PROTOCOL, PARENT, PARENT_SHA, BASE_SHA, SEED, QUALITY_SEED
from train_csra_state_replay import source_hashes, dataset_hashes
from probe_csra_front_keep import BASE, file_sha, write_json


def main():
    if file_sha(BASE)!=BASE_SHA or file_sha(PARENT)!=PARENT_SHA:
        raise ValueError('Base/parent mutated')
    summary=json.loads((ROOT/'quality_summary.json').read_text())
    if not summary['complete'] or summary['images_sha_verified']!=1024 or summary['protocol']!=PROTOCOL:
        raise ValueError('Require complete1024-image summary')
    code,data=source_hashes(),dataset_hashes()
    checkpoints={};trace_by_kind={};initial=[]
    files=set()
    for arm in ('teacher_only','mixed50'):
        folder=Path(f'checkpoints/RDD-CSRA-state-replay-{arm}-L400K-20260914')
        done=json.loads((folder/'complete.json').read_text())
        exp=json.loads((folder/'experiment.json').read_text())
        path=folder/'latest.pt'
        ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        if (not done['complete'] or done['smoke'] or done['updates']!=400 or done['arm']!=arm
            or ck['protocol']!=PROTOCOL or ck['updates']!=400 or ck['smoke'] or ck['arm']!=arm
            or ck['parent_sha256']!=PARENT_SHA or ck['base_sha256']!=BASE_SHA
            or ck['code_sha256']!=code or ck['data_manifests_sha256']!=data
            or exp['seed']!=SEED or exp['global_batch']!=256 or exp['per_gpu_batch']!=64
            or exp['accumulation']!=1 or exp['updates_planned']!=400
            or len(ck['rank_rng_states'])!=4 or not ck['optimizer']['state']
            or ck['validation']!=done['final_validation']):
            raise ValueError('Incomplete/changed fixed-budget training contract')
        digest=file_sha(path)
        checkpoints[arm]=dict(path=str(path),sha256=digest,updates=400,
            validation=ck['validation'],code_sha256=code,data_manifests_sha256=data,
            peak_allocated_gib=done['peak_allocated_gib'])
        initial.append(exp['initial_validation'])
        del ck
        smoke=ROOT/f'smoke_{arm}'
        s=json.loads((smoke/'complete.json').read_text())
        if not s['complete'] or not s['smoke'] or s['updates']!=2:
            raise ValueError('Missing actual4x64 smoke')
        for kind,prefix in (('formal',folder),('smoke',smoke)):
            for rank in range(4):
                trace=json.loads((prefix/f'trace_rank{rank}.json').read_text())['trace']
                if [r['update'] for r in trace]!=[0,1]:raise ValueError('Incomplete trace')
                for row in trace:
                    expected=[i%2 for i in row['indices']] if arm=='mixed50' else [0]*64
                    if len(row['indices'])!=64 or row['state_role']!=expected:
                        raise ValueError('Wrong state selection')
                canonical=[{k:r[k] for k in ('update','indices','labels','t')} for r in trace]
                if trace_by_kind.setdefault((kind,rank),canonical)!=canonical:
                    raise ValueError('Arms do not use same indices/labels/t')
        files.update(folder.glob('*.json'))
    if initial[0]!=initial[1]:raise ValueError('Different initial validation')
    for kind in ('formal','smoke'):
        for update in (0,1):
            indices=sum([trace_by_kind[kind,rank][update]['indices'] for rank in range(4)],[])
            if len(set(indices))!=256:raise ValueError('Duplicated samples in global batch')
    variants={'original','parent','teacher_only','mixed50'}
    classifier=None;seen=set()
    for shard in range(4):
        folder=ROOT/'quality'/f'shard_{shard}'
        m=json.loads((folder/'metrics.json').read_text())
        if (not m['complete'] or m['protocol']!=PROTOCOL or m['seed']!=QUALITY_SEED+shard*10000
            or m['steps']!=[80,80,90] or m['cfg']!=1. or m['boundary']!='noise_filling'
            or m['coarse_endpoint_max_abs']!={v:0. for v in variants-{'original'}}):
            raise ValueError('Quality trajectory contract mismatch')
        if classifier is None:classifier=m['classifier_sha256']
        if classifier!=m['classifier_sha256']:raise ValueError('Classifier changed')
        for arm in ('teacher_only','mixed50'):
            if m['configs'][arm]['checkpoint_sha256']!=checkpoints[arm]['sha256']:
                raise ValueError('Sampling used wrong checkpoint')
        if m['configs']['parent']['checkpoint_sha256']!=PARENT_SHA:
            raise ValueError('Wrong parent reference')
        if len(list(folder.glob('*/*.png')))!=256 or not (folder/'comparison_grid.png').is_file():
            raise ValueError('Missing images/grid')
        for row in m['rows']:
            key=(row['image'],row['variant'])
            if key in seen or row['variant'] not in variants:raise ValueError('Duplicate/wrong row')
            seen.add(key)
            if row['target']!=m['labels'][row['local_image']]:raise ValueError('Unpaired labels')
            image=folder/row['variant']/f"{row['local_image']:03d}.png"
            if file_sha(image)!=row['image_sha256']:raise ValueError('Image modified')
            with Image.open(image) as im:
                if im.size!=(256,256):raise ValueError('Wrong output size')
                im.verify()
    if len(seen)!=1024:raise ValueError('Missing paired quality images')
    write_json(ROOT/'completion_audit.json',dict(complete=True,protocol=PROTOCOL,checkpoints=checkpoints,
        base_sha256=BASE_SHA,parent_sha256=PARENT_SHA,images_sha_verified=1024,
        paired_formal_and_smoke_traces=True,initial_validation_equal=True,
        warning='Contract audit only; fixed snapshot replay, not live on-policy or quality acceptance.'))
    files.update(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.png'))
    files.update(Path('eval_results')/n for n in ('csra_state_replay_launcher_20260914.log',
                                                'csra_state_replay_train_launcher_20260914.log'))
    archive=Path('eval_results/csra_state_replay_review_20260914.tar.gz')
    if archive.exists():raise FileExistsError('Preserve existing bundle')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files):
            if p.suffix not in ('.json','.log','.png'):raise ValueError('Weights/data must stay remote')
            tar.add(p,arcname=p.as_posix(),recursive=False)
    receipt=dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,
                 sha256=file_sha(archive),files=len(files))
    write_json(ROOT/'bundle_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
