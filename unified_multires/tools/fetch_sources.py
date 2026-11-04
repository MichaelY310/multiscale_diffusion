#!/usr/bin/env python3
"""Fetch only pinned, hash-checked author source files; never models or datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 4 * 1024 * 1024

def validate_entry(name, entry, destination):
    if not re.fullmatch(r'[a-z0-9_]+', name):raise ValueError('Invalid source name')
    if not re.fullmatch(r'https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',entry['repo']):raise ValueError('Only pinned GitHub source repositories are supported')
    if not re.fullmatch(r'[a-f0-9]{40}',entry['commit']):raise ValueError('A full source commit is required')
    root=(destination/name).resolve()
    for relative,digest in entry['selected_files'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or '.git' in p.parts:raise ValueError('Unsafe source path')
        if not (root/p).resolve().is_relative_to(root):raise ValueError('Source path escapes checkout')
        if p.suffix.lower() in {'.pt','.pth','.bin','.safetensors','.ckpt','.npz','.npy','.png','.jpg','.mp4','.zip','.tar','.gz'}:raise ValueError('Binary/model/data payload is not a source file')
        if not re.fullmatch(r'[a-f0-9]{64}',digest):raise ValueError('A SHA256 is required for every source file')

def synchronize(name, entry, destination, verify=False):
    validate_entry(name,entry,destination)
    def one(item):
        relative,expected=item;path=destination/name/relative
        if path.exists():
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'{name}/{relative}: existing file differs; preserve/review it before replacing')
            return 'verified'
        if verify:raise FileNotFoundError(f'{name}/{relative}: run fetch_sources.py --repo {name}')
        repo=entry['repo'].removeprefix('https://github.com/')
        url=f"https://raw.githubusercontent.com/{repo}/{entry['commit']}/{urllib.parse.quote(relative,safe='/')}"
        with urllib.request.urlopen(url,timeout=45) as response:data=response.read(MAX_BYTES+1)
        if len(data)>MAX_BYTES:raise ValueError(f'{name}/{relative}: source exceeds size limit')
        if data.startswith(b'version https://git-lfs.github.com/spec/'):raise ValueError('LFS pointers are not downloaded')
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'{name}/{relative}: downloaded SHA256 mismatch')
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as temp:
            temp.write(data);temporary=Path(temp.name)
        try:temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
        return 'fetched'
    with ThreadPoolExecutor(max_workers=4) as pool:states=list(pool.map(one,entry['selected_files'].items()))
    return {'repo':name,'commit':entry['commit'],'files':len(states),'fetched':states.count('fetched'),'verified':states.count('verified')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',action='append',help='Repeat for multiple source names')
    p.add_argument('--all',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--list',action='store_true');p.add_argument('--destination',type=Path,default=ROOT/'sources')
    args=p.parse_args();lock=json.loads((ROOT/'sources.lock.json').read_text());repos=lock['repositories']
    if args.list:
        print(json.dumps({k:{'repo':v['repo'],'commit':v['commit'],'files':len(v['selected_files'])} for k,v in repos.items()},indent=2));return
    if bool(args.repo)==bool(args.all):p.error('Choose --repo NAME (repeatable) or --all')
    names=sorted(repos) if args.all else list(dict.fromkeys(args.repo))
    if set(names)-set(repos):p.error('Unknown repository name; use --list')
    for name in names:print(json.dumps(synchronize(name,repos[name],args.destination.resolve(),args.verify)),flush=True)

if __name__=='__main__':main()
