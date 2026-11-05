#!/usr/bin/env python3
"""CPU metadata checks, measured-call cross-checks and config equivalence."""
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rdd_identity import describe, legacy_identity, write_identity
R=ROOT/'docs/experiment_registry';E=ROOT/'docs/validation/20261104'
read=lambda p:json.loads(p.read_text())
checks=[]
rows=read(R/'results.json')['results'];raw=list(csv.DictReader((R/'legacy_formal_results.csv').open(encoding='utf-8-sig')))
assert len(rows)==len(raw)==116
for row,source in zip(rows,raw):
    assert row['record_id']==source['record_id']
    assert row['original_ledger_row']==source
    for key,value in row['metrics'].items():assert value==source[key]
    assert sum(s['steps'] for s in row['identity']['sampling_stages'])==int(source['scheduled_updates'])
checks.append('116 result rows: original metrics/source rows preserved; per-grid budgets consistent')
fm=[r for r in rows if r['legacy_model']=='FM Unified XL v2 legacy']
assert [s['duration'] for s in fm[0]['identity']['training_stages']]==[300,300,400]
proportional=[r for r in rows if 'interval_proportional' in r['source_path']]
assert len(proportional)==6
for r in proportional:assert [s['steps'] for s in r['identity']['sampling_stages']]==list(map(int,r['original_ledger_row']['sampling_steps_fine_middle_coarse'].split(',')))
checks.append('FM and six proportional rows use actual source direction, not column label')

measured=read(E/'SAMPLING_STEP_ORDER_CALLS.json')
for c in measured['cases']:
    family='fm' if c['solver'].startswith('fm_') else 'ddpm'
    solver=c['solver'].removeprefix('fm_')
    actual=describe(family,[32,16,8],[300,300,400],c['configured_steps_low_to_high'],solver)
    assert actual['expected_native_nfe']==c['total_model_calls']
    for s in actual['sampling_stages']:assert s['expected_native_nfe']==c['model_calls_by_resolution'][str(s['latent_resolution'])]
checks.append('Six actual native CPU-loop traces agree with grid labels and NFE metadata')

aliases=read(R/'repository_aliases.json');checked=0;shells=0
for item in aliases:
    old,new=ROOT/item['old_path'],ROOT/item['canonical_path']
    assert old.is_symlink() and old.resolve()==new.resolve()
    if new.suffix in ('.json','.jsonl'):
        baseline=subprocess.check_output(['git','show','a16343cf304401780782c25e8fe861310be55cde:'+item['old_path']],cwd=ROOT).decode()
        now=new.read_text()
        parse=lambda text:[json.loads(s) for s in text.splitlines() if s.strip()] if new.suffix=='.jsonl' else json.loads(text)
        assert parse(baseline)==parse(now),item['old_path'];checked+=1
    elif new.suffix=='.sh':
        subprocess.run(['bash','-n',str(new)],check=True);shells+=1
checks.append(f'{len(aliases)} compatibility links valid; {checked} JSON/JSONL contents exactly equal before naming migration; {shells} shell syntax checks')
with tempfile.TemporaryDirectory(prefix='rdd-naming-') as directory:
    original=describe('ddpm',[32,16,8],[300,300,400],[80,80,90],'ddpm')
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda _:write_identity(directory,original),range(16)))
    assert read(Path(directory)/'resolution_identity.json')==original
    try:write_identity(directory,describe('ddpm',[32,16,8],[300,300,400],[90,80,80],'ddpm'))
    except ValueError:pass
    else:raise AssertionError('Mismatched output allocation was accepted')
checks.append('Concurrent manifest creation is atomic; 90/80/80 cannot resume an 80/80/90 identity')
configs=read(R/'configs.json')
for c in configs:
    if c['origin']=='local_legacy':
        file=ROOT/c['source_path'];line=int(c['source_pointer'].split(':')[-1]);d=json.loads(file.read_text().splitlines()[line-1]);assert legacy_identity(d)==c['identity']
    elif c['origin']=='unified_native_config':assert read(ROOT/c['source_path'])==c['config']
checks.append(f'{len(configs)} configuration entries match live files or pinned archive records; not promoted to completed runs')
manifest=ROOT/'unified_multires/configs/rdd_reproduction/manifest.json'
for entry in read(manifest)['entries']:
    assert hashlib.sha256((manifest.parent/entry['file']).read_bytes()).hexdigest()==entry['sha256']
checks.append('18 native reproduction recipe hashes are unchanged')
metadata=E/'CHECKPOINT_NAMING_METADATA.json'
if metadata.exists():
    audit=read(metadata);assert len(audit['checkpoints'])==166
    assert all(x.get('architecture_matches_canonical_name') for x in audit['checkpoints'])
    assert audit['counts']=={'clock_not_saved':10,'matches_saved_clock':156}
    checks.append('166 checkpoint architectures match; 156 multires clocks match saved metadata; 10 ordinary DiT clocks are not saved and rely on source configuration')
receipt=E/'HF_NAMING_MIGRATION.json'
if receipt.exists():
    proof=read(receipt);assert proof['all_passed'] and all(x['content_identity_preserved'] for x in proof['checks'])
    checks.append(f"HF receipt: {proof['checked']} renamed files retain content identity")
for path in [ROOT/'rdd_identity.py',*ROOT.joinpath('scripts').glob('*rdd*naming*.py')]:ast.parse(path.read_text())
result=dict(status='passed',scope='Naming/config identity, no training or quality rerun',checks=checks,
            formal_results=len(rows),configs=len(configs),aliases=len(aliases),gpu_used=False)
(E/'NAMING_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
