#!/usr/bin/env python3
"""Install explicit canonical names, with symlinks for existing launch commands.

Only filenames and metadata change. The old numerical dictionaries are checked
before/after. Historical metric paths and checkpoint internals remain provenance.
"""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rdd_identity import tag

registry=json.loads((ROOT/'docs/experiment_registry/configs.json').read_text())
mapfile=ROOT/'docs/experiment_registry/repository_aliases.json'
mapping=json.loads(mapfile.read_text()) if mapfile.exists() else []
seen=set()
for row in registry:
    if row['origin']!='unified_native_config':continue
    old=ROOT/row['source_path']
    if old.is_symlink() or row['source_path'] in {m['canonical_path'] for m in mapping}:continue
    d=json.loads(old.read_text());method=d.get('method',d);ident=row['identity']
    prefix=re.sub(r'_(?:300_300_400|400_300_300|100_300_600|600_300_100)', '', old.stem)
    name=prefix+'__'+tag(ident['training_stages'],'duration','train')+'__'+tag(ident['sampling_stages'])+'.json'
    new=old.with_name(name)
    assert new!=old and not new.exists(),new
    before=old.read_bytes()
    old.rename(new);old.symlink_to(new.name)
    assert old.read_bytes()==before==new.read_bytes()
    mapping.append(dict(old_path=str(old.relative_to(ROOT)),canonical_path=str(new.relative_to(ROOT)),
                        compatibility='legacy symlink; identical numeric config',identity=ident))

# Legacy DDPM launch filenames: these exact entry points have been checked
# against their JSONL/native stage conventions. Raw metric logs are not renamed.
for old in sorted((ROOT/'2d/exp').glob('*')):
    if old.is_symlink() or old.suffix not in ('.jsonl','.sh'):continue
    name=old.name
    for code in ('300300400','400300300','100300600','600300100'):
        if code in name:
            a,b,c=code[:3],code[3:6],code[6:]
            name=name.replace(code,f'train_r8-{c}_r16-{b}_r32-{a}')
    name=name.replace('908080step','sample_r8-80_r16-80_r32-90')
    if name==old.name:continue
    new=old.with_name(name)
    assert not new.exists(),new
    before=old.read_bytes();old.rename(new);old.symlink_to(new.name)
    assert old.read_bytes()==before
    mapping.append(dict(old_path=str(old.relative_to(ROOT)),canonical_path=str(new.relative_to(ROOT)),
                        compatibility='legacy symlink; identical source'))

for relative in [
    '2d/train_RDD_FM_unified_XL_corrected_300_300_400.sh',
    '2d/train_RDD_FM_unified_XL_corrected_300_300_400_8gpu_sf7.sh',
    '2d/train_RDD_imagenet256_100_300_600_original.sh',
    '2d/train_RDD_imagenet256_300_300_400_original.sh',
    '2d/train_RDD_unified_L_100_300_600_4gpu.sh',
    'scripts/download_rdd_multistage_100_300_600_400k.sh',
    'scripts/rdd3d/run_dynamic_sparse_fm_unified_xl_300300400_8gpu.sh',
]:
    old=ROOT/relative
    if old.is_symlink():continue
    name=old.name.replace('100_300_600','train_r8-600_r16-300_r32-100').replace('300_300_400','train_r8-400_r16-300_r32-300').replace('300300400','train_r8cubed-400_r16cubed-300_r32cubed-300')
    new=old.with_name(name);assert not new.exists()
    before=old.read_bytes();old.rename(new);old.symlink_to(new.name);assert new.read_bytes()==before
    mapping.append(dict(old_path=relative,canonical_path=str(new.relative_to(ROOT)),
                        compatibility='legacy symlink; identical source',hf_source_kept_as_archive=True))

mapfile.write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'renamed_entries':len(mapping),'old_paths_retained_as_symlinks':True,'numerical_configs_unchanged':True}))
