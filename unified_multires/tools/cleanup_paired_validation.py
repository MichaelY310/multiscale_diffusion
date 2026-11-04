#!/usr/bin/env python3
"""Delete ONLY this suite's completed temporary payloads; keep all reports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/paired_experiments_20261010'

def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    if summary['status']!='passed_scoped_checks':raise ValueError('Finish numerical checks before deleting payloads')
    suffixes={'.pt','.pth','.png','.mp4','.npz','.safetensors'}
    files={p for p in OUT.rglob('*') if p.is_file() and p.suffix in suffixes}
    # Exact directories created by the specialized validators in this suite.
    case_dirs=[ROOT/'runs/diffusability/native_pairs/diffusability_random24_VGG_frozen',
      ROOT/'runs/auxiliary/rmd/rmd_released_flags/rmd_transition1',
      ROOT/'runs/auxiliary/rmd/rmd_released_flags/rmd_transition2']
    for directory in case_dirs:
        if directory.exists():files.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in suffixes)
    protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.json','.log','.md'}}
    rows=[]
    for p in sorted(files):
        resolved=p.resolve()
        if not resolved.is_relative_to(ROOT):raise ValueError('Refuse outside-suite symlink: '+str(p))
        rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for row in rows:(ROOT/row['path']).unlink()
    if any(not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha for p,sha in protected.items()):raise AssertionError('A report changed during cleanup')
    previous=OUT/'ARTIFACT_CLEANUP.json'
    if previous.exists():rows=json.loads(previous.read_text())['removed']+rows
    report=dict(status='completed',removed=rows,removed_bytes=sum(r['bytes'] for r in rows),
      protected_reports=len(protected),policy='Completed temporary validation payloads only; reports/configs/logs/statistics/hashes retained; no user dataset/formal weights deleted.')
    previous.write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'PAYLOAD_REMOVED_NOTICE.txt').write_text('Temporary test tensor pairs/checkpoints/media were deleted after verification. Reports, numerical statistics and SHA256 remain. These validation checkpoints cannot be resumed. See ARTIFACT_CLEANUP.json.\n')
    print(json.dumps({k:report[k] for k in ('status','removed_bytes','protected_reports')}))

if __name__=='__main__':main()
