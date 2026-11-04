"""Audit and bundle completed four-arm fixed1600 experiment; never include weights."""
import hashlib
import json
from pathlib import Path
import tarfile
import torch
from csra_staged_protocol import PROTOCOL
from probe_csra_front_keep import BASE, BASE_SHA, file_sha, write_json

ROOT = Path('eval_results/csra_staged_front_review_20260914')
MODES = ('staged', 'direct_control')
ARMS = ('aligned', 'zero')
PARENT = {'aligned':'24483c2f912db24bcbda519add80aeba24644b307585a70eb594499a9a55dff1',
          'zero':'afc899af245edb7b454e16023fbd8cf955f06c34aa305c1fe85917686a9265f2'}
INIT_SHA = 'f43775e51506f8f6a15c43bd0947ccc43b558579cf42dc79ab0d022aa92412ca'


def audit():
    if file_sha(BASE) != BASE_SHA: raise ValueError('Teacher changed')
    checked, files, references, traces, classifier = {}, set(), {}, {}, set()
    png_count = 0
    for mode in MODES:
        quality = Path(f'eval_results/csra_staged_front_{mode}_quality256_20260914')
        for arm in ARMS:
            folder = Path(f'checkpoints/RDD-CSRA-front1-{mode}-{arm}-L400K-20260914')
            d = json.loads((folder/'complete.json').read_text())
            cp = folder/'latest.pt'
            ck = torch.load(cp,map_location='cpu',mmap=True,weights_only=True)
            if (not d['complete'] or d['smoke'] or d['step'] != 1600 or ck['step'] != 1600
                    or ck['training_protocol'] != PROTOCOL or ck['format'] != 'csra_front1_skip6_conditional_v1'
                    or ck['base_sha256'] != BASE_SHA or ck['initializer_sha256'] != INIT_SHA
                    or ck['args']['mode'] != mode or ck['args']['coarse_source'] != arm
                    or ck['args']['seed'] != 20279000 or ck['front_keep'] != 1 or ck['prefix_depth'] != 6
                    or ck['source_layer'] != 13):
                raise ValueError('Training/checkpoint contract mismatch')
            if mode == 'staged':
                if not d['warmup_backbone_verified'] or not ck['warmup_backbone_verified']:
                    raise ValueError('Missing original-backbone warmup parity')
                log = Path(f'logs/csra_staged_front_{mode}_{arm}_20260914.log').read_text()
                events = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
                gate = [e for e in events if e.get('event') == 'warmup_backbone_exact' and e['step'] == 800]
                if len(gate) != 1 or gate[0]['sha256'] != 'e784e143d599cb758710c037612ee6e6a02f183080cc577246a2077fe24b118e':
                    raise ValueError('Warmup weights not exactly original')
            elif ck['parent']['sha256'] != PARENT[arm] or ck['parent']['step'] != 800:
                raise ValueError('Wrong direct-control optimizer parent')
            if file_sha(ck['args']['initializer']) != INIT_SHA: raise ValueError('Initializer changed')
            exp = json.loads((folder/'experiment.json').read_text())
            if exp['actual_global_batch'] != 256 or exp['per_gpu_batch'] != 64 or exp['accumulation'] != 1:
                raise ValueError('Batch mismatch')
            for name, digest in ck['code_sha256'].items():
                if file_sha(Path('2d/attention_analysis')/name) != digest: raise ValueError('Training code changed')
            checked[f'{mode}_{arm}'] = dict(checkpoint=str(cp),sha256=file_sha(cp),parent=ck['parent'],
                final_validation=d['final_validation'],step=1600,warmup_verified=d['warmup_backbone_verified'],
                peak_allocated_gib=d['peak_allocated_gib'],code_sha256=ck['code_sha256'])
            del ck
            for rank in range(4):
                trace = json.loads((folder/f'start_trace_rank{rank}.json').read_text())['trace']
                joint = [r for r in trace if r['step'] in (800,801)]
                if len(joint) != 2 or traces.setdefault(rank,joint) != joint:
                    raise ValueError('Unpaired joint update inputs')
            files.update(folder.glob('*.json'))
            smoke = Path(f'eval_results/csra_staged_front_{mode}_{arm}_smoke_20260914')
            sd = json.loads((smoke/'complete.json').read_text())
            if not sd['complete'] or not sd['smoke'] or sd['step'] != (2 if mode == 'staged' else 802):
                raise ValueError('Missing real4x64 smoke')
            files.update(smoke.glob('*.json'))
        if len(list(quality.glob('shard_*/metrics.json'))) != 4: raise ValueError('Missing quality shard')
        for rank in range(4):
            folder = quality/f'shard_{rank}'
            m = json.loads((folder/'metrics.json').read_text())
            if (not m['complete'] or m['rank'] != rank or m['checkpoint_step'] != 1600
                    or m['steps'] != [80,80,90] or m['cfg'] != 1 or m['upsample'] != 'noise_filling'
                    or m['coarse_endpoint_max_abs'] != {'aligned':0.,'zero':0.}):
                raise ValueError('Quality parity/config mismatch')
            classifier.add(m['classifier_sha256'])
            for arm in ARMS:
                config = m['configs'][arm]
                if (config['checkpoint_sha256'] != checked[f'{mode}_{arm}']['sha256']
                        or config['training_protocol'] != PROTOCOL or config['training_mode'] != mode):
                    raise ValueError('Sampling used other weights')
            previous = json.loads((Path('eval_results/csra_front1_skip6_quality256_20260914')/f'shard_{rank}'/'metrics.json').read_text())
            old = [r for r in previous['rows'] if r['variant'] == 'original_dense']
            original = [r for r in m['rows'] if r['variant'] == 'original_dense']
            if old != original or references.setdefault(rank,original) != original:
                raise ValueError('Original teacher images/classification changed')
            for r in m['rows']:
                file = folder/r['variant']/f"{r['local_image']:03d}.png"
                if file_sha(file) != r['image_sha256']: raise ValueError('PNG hash mismatch')
                png_count += 1
            if len(list(folder.glob('*/*.png'))) != 192 or not (folder/'comparison_grid.png').exists():
                raise ValueError('Unexpected sample/grid count')
        files.update(p for p in quality.rglob('*') if p.is_file())
    if len(classifier) != 1 or png_count != 1536: raise ValueError('Incomplete image/classifier audit')
    files.update(Path('logs').glob('csra_staged_front*_20260914.log'))
    report = dict(complete=True,checkpoints=checked,base_sha256=BASE_SHA,initializer_sha256=INIT_SHA,
                  images_sha_verified=png_count,baseline256_byte_equal=True,joint_data_traces_equal=True,
                  classifier_sha256=next(iter(classifier)),warning='Audit only. Fixed1600 offline training, not quality/speed success.')
    write_json(ROOT/'completion_audit.json', report)
    files.add(ROOT/'completion_audit.json')
    archive = Path('eval_results/csra_staged_front_review_20260914.tar.gz')
    if archive.exists(): raise FileExistsError('Do not replace an existing verified bundle')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files):
            if p.suffix not in ('.json','.log','.png'): raise ValueError(f'Unexpected bundle member:{p}')
            tar.add(p,arcname=p.as_posix(),recursive=False)
    print(json.dumps(dict(complete=True,files=len(files),archive=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive)),indent=2))


if __name__ == '__main__': audit()
