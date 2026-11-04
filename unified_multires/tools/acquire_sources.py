"""Fetch public source blobs only, never LFS weights, datasets, or media."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import argparse

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    'ralu': 'ignoww/RALU',
    'jit_spatial': 'Wenhao-Sun77/Just-in-Time',
    'inverse_heat': 'AaltoML/generative-inverse-heat-dissipation',
    'relay': 'zai-org/RelayDiffusion',
    'ssd': 'prateksha/ScaleSpaceDiffusion',
    'speed': 'howardhx/speed',
    'pixelflow': 'ShoufaChen/PixelFlow',
    'dfm': 'snap-research/dfm',
    'lapflow': 'sjtuytc/ICLR26_LapFlow_Official',
    'ppflow': 'fudan-generative-vision/PPFlow',
    'nami': '360CVGroup/NAMI-T2I',
    'mrflow': 'Xingyu-Zheng/MrFlow',
    'dctdiff': 'forever208/DCTdiff',
    'fresco': 'DarrenZheng303/Fresco',
    'avits': 'QHR69/AViTS',
    'diffusability': 'snap-research/diffusability',
    'mdm': 'apple/ml-mdm',
    'dimr': 'qihao067/DiMR',
    'rmd': 'Skylar-aigc/RMD-Inference-Acceleration',
}
SUFFIXES = {'.py', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.sh', '.md',
            '.rst', '.txt', '.json', '.cu', '.cuh', '.h', '.hpp', '.cpp', '.c',
            '.gitignore', '.gitattributes', '.gitmodules'}
env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')

def git(dest, *args, **kw):
    return subprocess.run(['git', '-C', str(dest), *args], env=env,
                          check=True, capture_output=True, **kw).stdout

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=REPOS);args=parser.parse_args()
    root = ROOT / 'sources'
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'evidence' / 'sources.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key, repo in REPOS.items():
        if args.only and key!=args.only:continue
        dest = root / key
        try:
            if not (dest / '.git').exists():
                subprocess.run(['git', 'clone', '--depth=1', '--filter=blob:none',
                                '--no-checkout', 'https://github.com/' + repo + '.git',
                                str(dest)], env=env, check=True, capture_output=True)
            commit = git(dest, 'rev-parse', 'HEAD', text=True).strip()
            # -l queries blob sizes and can hydrate EVERY omitted blob in a
            # partial clone. Never use it: classify paths without opening blobs.
            tree = git(dest, 'ls-tree', '-rz', 'HEAD').split(b'\0')
            selected, excluded = [], []
            for record in tree:
                if not record:
                    continue
                metadata, filename = record.split(b'\t', 1)
                path = filename.decode()
                mode, kind, obj = metadata.decode().split()
                p = Path(path)
                # No git submodules (may contain binary assets), symlinks, large text,
                # pretrained tokenizer vocabularies or asset dumps.
                allow = (kind == 'blob' and mode != '120000'
                         and (p.suffix.lower() in SUFFIXES or p.name.lower() in
                              {'license', 'licence', 'makefile', 'dockerfile', 'agents.md'})
                         and not any(t in p.name.lower() for t in
                                     ('tokenizer.json', 'vocab.json', 'merges.txt'))
                         and not (p.suffix.lower() in ('.txt','.json','.jsonl') and any(t in p.name.lower() for t in ('prompt','annotation','dataset'))))
                (selected if allow else excluded).append(path)
            git(dest, 'sparse-checkout', 'init', '--no-cone')
            # Literal root-relative patterns; no directory pattern downloads.
            patterns = ''.join('/' + p + '\n' for p in selected)
            git(dest, 'sparse-checkout', 'set', '--no-cone', '--stdin',
                input=patterns, text=True)
            git(dest, 'checkout', 'HEAD')
            hashes = {p: hashlib.sha256((dest / p).read_bytes()).hexdigest()
                      for p in selected}
            manifest[key] = dict(repo='https://github.com/' + repo, commit=commit,
                                 fetched_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 selected_files=hashes, excluded_files=excluded,
                                 source_bytes=sum((dest / p).stat().st_size for p in selected),
                                 status='source_only_downloaded')
            print(key, commit, len(selected), 'files', flush=True)
        except Exception as exc:
            detail = getattr(exc, 'stderr', str(exc))
            if isinstance(detail, bytes):
                detail = detail.decode(errors='replace')
            manifest[key] = dict(repo='https://github.com/' + repo, status='acquisition_failed',
                                 error=detail)
            print(key, 'FAILED', detail, flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
