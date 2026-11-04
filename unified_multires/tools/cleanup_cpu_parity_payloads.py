#!/usr/bin/env python3
"""Delete only report-referenced temporary paired checkpoints and local payloads.

Run after all producers finish. JSON/Markdown/log/source/config files remain.
Default is inventory only; --apply follows the user's retained-report policy.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def refs(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in ('locally_trained_checkpoint_pair','checkpoint_pair') and isinstance(v,str):yield v
            else:yield from refs(v)
    elif isinstance(obj,list):
        for v in obj:yield from refs(v)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();base=a.evidence.resolve()
    assert base.is_relative_to(ROOT/'evidence') and base!=ROOT/'evidence'
    files=set();directories=set()
    for report in base.rglob('*.json'):
        if report.name=='cleanup.json':continue
        try:obj=json.loads(report.read_text())
        except (UnicodeError,json.JSONDecodeError):continue
        for rel in refs(obj):
            directory=(ROOT/rel).resolve()
            assert directory.is_relative_to(ROOT/'runs'),directory
            directories.add(directory)
            for name in ('native.pt','unified.pt'):
                candidate=directory/name
                if candidate.is_file():files.add(candidate)
    for file in base.rglob('*'):
        if file.is_file() and file.suffix in ('.pt','.pth','.safetensors','.png','.npy','.npz'):
            assert file.resolve().is_relative_to(base)
            files.add(file)
    records=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)]
    if a.apply:
        for file in files:file.unlink()
        for directory in sorted(directories,key=lambda x:len(x.parts),reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):directory.rmdir()
    report=dict(applied=a.apply,deleted_files=len(records) if a.apply else 0,bytes=sum(x['bytes'] for x in records),files=records,scope='Only native.pt/unified.pt referenced by this run reports and payload extensions below its evidence directory; source/config/log/JSON/Markdown retained. No pretrained/user training models deleted.')
    (base/'cleanup.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('applied','deleted_files','bytes')}))
if __name__=='__main__':main()
