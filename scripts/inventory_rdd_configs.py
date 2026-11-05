#!/usr/bin/env python3
"""Inventory executable RDD allocations separately from completed result rows."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rdd_identity import describe, legacy_identity, display

OUT=ROOT/'docs/experiment_registry'
records=[]

def add(path,pointer,config,identity,origin):
    records.append(dict(source_path=path,source_pointer=pointer,origin=origin,
        status='configuration_only_not_proof_of_completed_run',
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        identity=identity,training_display=display(identity['training_stages'],'duration'),
        sampling_display=display(identity['sampling_stages']),config=config))

for base in ('2d','configs','eval_results'):
    for p in sorted((ROOT/base).rglob('*.jsonl')):
        if p.is_symlink():continue
        for n,line in enumerate(p.read_text().splitlines(),1):
            if not line.strip():continue
            try:d=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(d,dict) and all(k in d for k in ('model_type','stage_lengths','resolutions','num_steps')):
                add(str(p.relative_to(ROOT)),f'line:{n}',d,legacy_identity(d),'local_legacy')
for row in json.loads((ROOT/'docs/validation/20261104/HF_FM_SAMPLING_CONFIGS.json').read_text()):
    for n,d in enumerate(row['records'],1):
        add(row['path'],f'line:{n}',d,legacy_identity(d),'pinned_HF_FM')

for p in sorted((ROOT/'unified_multires/configs').rglob('*.json')):
    if p.is_symlink():continue
    try:full=json.loads(p.read_text())
    except json.JSONDecodeError:continue
    if not isinstance(full,dict):continue
    d=full.get('method',full)
    if not isinstance(d,dict):continue
    s=d.get('sampler',{})
    if isinstance(s,dict) and s.get('name') in ('rdd_native_stages','rdd_native_multistage'):
        a=s['args']; ident=describe('ddpm',a['resolutions'],a['stage_lengths'],a['steps'],a.get('solver','ddpm'))
        ident.update(boundary_label=a.get('boundary'),cfg=a.get('guidance_scale'))
        add(str(p.relative_to(ROOT)),'method.sampler' if 'method' in full else 'sampler',full,ident,'unified_native_config')
    elif d.get('task') in ('rdd_fm_native','rdd_fm_multistage_native'):
        a=d['process'];b=a['bounds'];lengths=[round((v-u)*1000) for u,v in zip(b,b[1:])]
        ident=describe('fm',a['resolutions'],lengths,a['steps'],a['solver'])
        ident.update(boundary_label=a.get('boundary'),cfg=a.get('cfg_scale'))
        add(str(p.relative_to(ROOT)),'process',full,ident,'unified_native_config')

OUT.mkdir(exist_ok=True)
(OUT/'configs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
with (OUT/'configs.csv').open('w',encoding='utf-8-sig',newline='') as out:
    keys=['source_path','source_pointer','origin','status','training_display','sampling_display','solver','expected_native_nfe','config_sha256']
    w=csv.DictWriter(out,keys);w.writeheader()
    for r in records:w.writerow({k:r.get(k,r['identity'].get(k)) for k in keys})
print(json.dumps({'configurations':len(records),'origins':{s:sum(r['origin']==s for r in records) for s in {r['origin'] for r in records}}}))
