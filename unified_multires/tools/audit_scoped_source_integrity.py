#!/usr/bin/env python3
"""Read-only closing audit of the pinned upstream sources and supported contracts."""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT/'evidence/unified_scope_20261012')
    args = parser.parse_args()
    base = args.evidence.resolve()
    start = json.loads((base/'source_integrity.json').read_text())
    pins_path = ROOT/'evidence/sources.json'
    failures = []
    if sha(pins_path) != start['manifest_sha256']:
        failures.append('upstream manifest changed')
    rows = []
    for name, repo in json.loads(pins_path.read_text()).items():
        files = repo.get('selected_files', {})
        bad = [f for f, h in files.items() if sha(ROOT/'sources'/name/f) != h]
        rows.append(dict(repo=name, commit=repo.get('commit'), files=len(files), mismatches=bad))
        failures.extend(name+'/'+f for f in bad)
    contracts_path = ROOT/'SUPPORTED_REPRODUCTIONS.json'
    contracts = json.loads(contracts_path.read_text())
    contract_files = {}
    for method in contracts['methods']:
        for f, h in method['source_file_sha256'].items():
            actual = sha(ROOT/f)
            contract_files[f] = actual
            if actual != h:
                failures.append('contract source: '+f)
    baseline = json.loads((base/'local_rdd_source_baseline.json').read_text())
    rdd = {f: sha(ROOT.parent/f.removeprefix('multiscale_diffusion/')) for f in baseline['sha256']}
    failures.extend('local RDD: '+f for f, h in baseline['sha256'].items() if rdd[f] != h)
    report = dict(status='passed' if not failures else 'failed', failures=failures,
                  repositories=rows, selected_files=sum(row['files'] for row in rows),
                  manifest_sha256=sha(pins_path), contract_source_sha256=contract_files,
                  local_rdd_source_sha256=rdd, local_rdd_capture_scope=baseline['capture_scope'],
                  scope='File hashes, not a new numerical test; supplements the per-job frozen implementation snapshots.',
                  harness_sha256=sha(Path(__file__)))
    (base/'postflight_source_integrity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], selected_files=report['selected_files'],
                          local_rdd_files=len(rdd), contract_files=len(contract_files), failures=failures)))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
