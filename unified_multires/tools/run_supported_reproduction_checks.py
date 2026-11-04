#!/usr/bin/env python3
"""CPU acceptance plus report/readback/cleanup, with explicit source pinning."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def require_finished_suite(base):
    from run_scoped_validation import jobs, snapshot
    from audit_snapshot_compatibility import compatibility
    expected = {row['name'] for row in jobs(base, True)}
    current, latest = snapshot(), {}
    for attempt in sorted((base/'suite_attempts').iterdir()):
        summary = attempt/'SUMMARY.json'
        if not summary.exists():
            continue
        recorded = json.loads((attempt/'source_snapshot.json').read_text())
        report = json.loads(summary.read_text())
        if report['status'] == 'running' and (recorded == current or any(compatibility(base, recorded, current, name) for name in expected)):
            raise RuntimeError('The suite is still running; do not finalize or clean its payloads yet.')
        for row in report['jobs']:
            if compatibility(base, recorded, current, row['name']):
                latest[row['name']] = row['status']
    unfinished = sorted(name for name in expected if latest.get(name) != 'passed')
    if unfinished:
        raise RuntimeError('Current-source acceptance is not finished: '+repr(unfinished))


def commands(base, workers, finalize_only):
    def command(script, *args):
        return [sys.executable, str(ROOT/'tools'/script), *map(str, args)]
    result = []
    if not finalize_only:
        result.append(command('run_scoped_validation.py', '--full', '--workers', workers, '--output', base))
    result.extend([
        command('audit_dopri5_sampling.py', '--output', base/'sampling_dopri5'),
        command('audit_rdd_distributed_sampler.py', '--output', base/'rdd_distributed_sampler.json'),
        command('audit_supported_profile_configs.py', '--output', base),
        command('audit_author_reader_contracts.py', '--output', base/'author_reader_contracts.json'),
        command('audit_comparator_negative_controls.py', '--output', base/'comparator_additional_negative_controls.json'),
        command('verify_cpu_checkpoint_states.py', '--evidence', base),
        command('audit_scoped_trace_reports.py', '--evidence', base),
        command('audit_scoped_source_integrity.py', '--evidence', base),
        command('build_scoped_reproduction_report.py', '--evidence', base),
        command('cleanup_cpu_parity_payloads.py', '--evidence', base, '--apply'),
        command('build_scoped_reproduction_report.py', '--evidence', base),
    ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/unified_scope_20261012')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--finalize-only', action='store_true', help='Finish an existing successful fixed-source run before its payloads are deleted.')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    base = args.output.resolve()
    plan = commands(base, args.workers, args.finalize_only)
    if args.plan:
        print(json.dumps(dict(commands=plan, gpu_used=False, downloads=False, deletes_only_temporary_verified_payloads=True), indent=2))
        return 0
    if not args.finalize_only:
        base.mkdir(parents=True, exist_ok=True)
        paths = sorted((ROOT.parent).rglob('*.py'))
        baseline = dict(capture_scope='Recorded before the fixed-source suite starts.',
                        sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        (base/'local_rdd_source_baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    else:
        require_finished_suite(base)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'PYTHONDONTWRITEBYTECODE':'1'}
    for command in plan:
        print(json.dumps(dict(command=command)), flush=True)
        subprocess.run(command, cwd=ROOT.parent, env=env, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
