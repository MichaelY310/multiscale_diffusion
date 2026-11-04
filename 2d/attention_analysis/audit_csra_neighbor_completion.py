"""Audit all four fixed2400 checkpoints and bundle evidence, never the weights."""
import json
from pathlib import Path
import tarfile
import torch

from csra_neighbor_protocol import check_new, source_paths, PROTOCOL, PARENT_SHA, OPTIMIZER_POLICY
from probe_csra_front_keep import BASE, BASE_SHA, file_sha, write_json

ROOT = Path('eval_results/csra_neighbor_2x2_20260914')
INIT_SHA = 'f43775e51506f8f6a15c43bd0947ccc43b558579cf42dc79ab0d022aa92412ca'


def main():
    if file_sha(BASE) != BASE_SHA: raise ValueError('Original checkpoint changed')
    summary = json.loads((ROOT/'factorial_summary.json').read_text())
    if not summary['complete'] or summary['images_sha_verified'] != 1536:
        raise ValueError('All fixed quality comparisons must finish before bundling')
    code = {name:file_sha(Path(name)) for name in source_paths()}
    files, configs, traces = set(), {}, {}
    for neighbor in ('off','on'):
        quality = Path(f'eval_results/csra_neighbor_{neighbor}_quality256_20260914')
        for source in ('aligned','zero'):
            folder = Path(f'checkpoints/RDD-CSRA-neighbor-{neighbor}-{source}-L400K-20260914')
            complete = json.loads((folder/'complete.json').read_text())
            experiment = json.loads((folder/'experiment.json').read_text())
            path = folder/'latest.pt'
            ck = torch.load(path,map_location='cpu',mmap=True,weights_only=True)
            check_new(ck,ck['args'],code,BASE_SHA,INIT_SHA,require_final=True)
            if (ck['args']['neighbor'] != neighbor or ck['args']['coarse_source'] != source
                    or not complete['complete'] or complete['smoke'] or complete['step'] != 2400
                    or complete['protocol'] != PROTOCOL or complete['initial_branch_max_abs'] != 0
                    or experiment['actual_global_batch'] != 256 or experiment['per_gpu_batch'] != 64
                    or experiment['accumulation'] != 1 or experiment['optimizer_policy'] != OPTIMIZER_POLICY
                    or len(ck['rank_rng_states']) != 4 or not ck['optimizer']['state']
                    or ck['parent']['sha256'] != PARENT_SHA[source]):
                raise ValueError('Wrong actual training configuration or missing optimizer/RNG')
            if file_sha(Path(ck['parent']['path'])) != PARENT_SHA[source]:
                raise ValueError('Parent checkpoint changed during this pilot')
            digest = file_sha(path)
            key = f'{neighbor}_{source}'
            configs[key] = dict(path=str(path),sha256=digest,step=ck['step'],parent=ck['parent'],
                code_sha256=code,final_validation=complete['final_validation'],
                peak_allocated_gib=complete['peak_allocated_gib'])
            if summary['configurations'][key]['checkpoint_sha256'] != digest:
                raise ValueError('Quality used a different checkpoint')
            del ck
            smoke = ROOT/f'smoke_{neighbor}_{source}'
            sd = json.loads((smoke/'complete.json').read_text())
            if not sd['complete'] or not sd['smoke'] or sd['step'] != 1602 or sd['initial_branch_max_abs'] != 0:
                raise ValueError('Real4x64 smoke missing')
            for rank in range(4):
                for kind, directory in (('formal',folder),('smoke',smoke)):
                    trace = json.loads((directory/f'start_trace_rank{rank}.json').read_text())['trace']
                    if [r['step'] for r in trace] != [1600,1601] or traces.setdefault((kind,rank),trace) != trace:
                        raise ValueError('Unpaired actual initial update indices/t/labels')
            files.update(folder.glob('*.json'))
        for rank in range(4):
            shard = quality/f'shard_{rank}'
            m = json.loads((shard/'metrics.json').read_text())
            if m['coarse_endpoint_max_abs'] != {'aligned':0.,'zero':0.}:
                raise ValueError('Original8/16 trajectory changed')
            if len(list(shard.glob('*/*.png'))) != 192 or not (shard/'comparison_grid.png').exists():
                raise ValueError('Incomplete image set')
            for row in m['rows']:
                image = shard/row['variant']/f"{row['local_image']:03d}.png"
                if file_sha(image) != row['image_sha256']: raise ValueError('PNG changed after summary')
        files.update(p for p in quality.rglob('*') if p.is_file())
    report = dict(complete=True,protocol=PROTOCOL,checkpoints=configs,base_sha256=BASE_SHA,
        initializer_sha256=INIT_SHA,paired_smoke_and_formal_traces=True,images_sha_verified=1536,
        warning='Evidence audit only, not a quality/acceptance claim.')
    write_json(ROOT/'completion_audit.json',report)
    files.update(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.png'))
    files.add(Path('eval_results/csra_neighbor_2x2_launcher_20260914.log'))
    archive = Path('eval_results/csra_neighbor_2x2_review_20260914.tar.gz')
    if archive.exists(): raise FileExistsError('Keep existing verified bundle; inspect before rerunning')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(files):
            if path.suffix not in ('.json','.log','.png'): raise ValueError(f'Unexpected file:{path}')
            tar.add(path,arcname=path.as_posix(),recursive=False)
    receipt = dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,
                   sha256=file_sha(archive),files=len(files))
    write_json(ROOT/'bundle_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':main()
