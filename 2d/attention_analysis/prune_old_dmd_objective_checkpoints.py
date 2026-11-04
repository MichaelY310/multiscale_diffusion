"""User-authorized latest-only cleanup of TWO exact old runs. Dry-run default.

Never recurses, never touches latest.pt, source checkpoints, logs or validation
latents. New files above the verified saved step and temporary files are kept.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import stat
import time
import torch

OLD_ROOT=Path('/home/yuheng/rdd_experiments/dmd_reuse_objective_20260915')
NAMES=('control','dmd_lpips')


def candidates(directory, saved_step, now=None):
    directory=directory.resolve(strict=True)
    now=time.time() if now is None else now
    found=[]
    for path in sorted(directory.iterdir()):
        match=re.fullmatch(r'step_(\d{7})\.pt',path.name)
        if not match or int(match[1])>saved_step: continue
        if path.is_symlink(): raise ValueError(f'Refusing symlink: {path}')
        info=path.stat()
        if not stat.S_ISREG(info.st_mode): continue
        if path.resolve().parent!=directory: raise ValueError('Target escapes exact checkpoint directory')
        if now-info.st_mtime<60: continue
        if hasattr(os,'getuid') and info.st_uid!=os.getuid(): raise ValueError('Not our file')
        found.append(dict(path=str(path),bytes=info.st_size,mtime_ns=info.st_mtime_ns,inode=info.st_ino))
    return found


def latest_metadata(folder):
    path=folder/'latest.pt'
    if path.is_symlink() or path.resolve(strict=True).parent!=folder.resolve(strict=True):
        raise ValueError('Refusing redirected latest checkpoint')
    before=path.stat()
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    required={'model','optimizer','rng','step','family','objective','protocol'}
    if not required.issubset(saved) or saved['family']!='high_heavy':
        raise ValueError('Latest is not the expected recoverable full checkpoint')
    if not saved['optimizer'].get('state') or not saved['rng']:
        raise ValueError('Latest lacks optimizer/RNG state')
    if saved['objective']=='dmd_lpips' and not {'critic','critic_optimizer'}.issubset(saved):
        raise ValueError('Latest lacks critic state')
    if not torch.isfinite(saved['model']['final_layer.linear.weight']).all():
        raise ValueError('Nonfinite latest model head')
    step=int(saved['step']); del saved; gc.collect()
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError('Latest changed during verification; retry next check')
    return dict(path=str(path),step=step,bytes=after.st_size,inode=after.st_ino,mtime_ns=after.st_mtime_ns)


def cleanup(apply=False):
    if OLD_ROOT.resolve(strict=True)!=OLD_ROOT:
        raise ValueError('Old experiment root is redirected')
    plan={'time':datetime.now(timezone.utc).isoformat(),'apply':apply,'retained':[],'candidates':[]}
    for name in NAMES:
        folder=OLD_ROOT/name/'checkpoints'
        if folder.resolve(strict=True)!=folder: raise ValueError('Checkpoint directory is redirected')
        latest=latest_metadata(folder);plan['retained'].append(latest)
        plan['candidates'].extend(candidates(folder,latest['step']))
    plan['candidate_bytes']=sum(item['bytes'] for item in plan['candidates'])
    if apply:
        audit=OLD_ROOT/'latest_only_cleanup.jsonl'
        with audit.open('a') as f:
            f.write(json.dumps(dict(event='planned',**plan))+'\n');f.flush();os.fsync(f.fileno())
        deleted=[]
        for item in plan['candidates']:
            path=Path(item['path']);before=path.stat()
            if path.is_symlink() or (before.st_ino,before.st_mtime_ns,before.st_size)!=(item['inode'],item['mtime_ns'],item['bytes']):
                raise RuntimeError('Snapshot changed; refusing deletion')
            # Exactly one named, verified historical file; no recursive deletion.
            path.unlink();deleted.append(item)
            with audit.open('a') as f:f.write(json.dumps(dict(event='deleted',**item))+'\n')
        plan['deleted_count']=len(deleted);plan['deleted_bytes']=sum(item['bytes'] for item in deleted)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true')
    print(json.dumps(cleanup(p.parse_args().apply),indent=2))
