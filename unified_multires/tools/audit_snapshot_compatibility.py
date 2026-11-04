"""Accept a narrowly recorded harness-only repair without discarding valid runs.

Production/config changes are never allowed here. The one permitted repair fixes
the independent gap regression runner; its job and pytest must be rerun. Original
snapshots, failed results and the old harness are retained for inspection.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = 'tools/validate_cpu_gap_fixes.py'
AFFECTED_JOBS = ('gap_regressions', 'pytest')


def compatibility(base, recorded, current, job):
    if recorded == current:
        return {'kind': 'exact_full_snapshot'}
    changed = {key for key in recorded.keys() | current.keys()
               if recorded.get(key) != current.get(key)}
    if changed != {HARNESS} or job in AFFECTED_JOBS:
        return None
    proof_path = base/'harness_repair.json'
    if not proof_path.exists():
        return None
    proof = json.loads(proof_path.read_text())
    if (proof.get('status') != 'passed'
            or proof.get('changed_file') != HARNESS
            or proof.get('affected_jobs') != list(AFFECTED_JOBS)
            or proof.get('before_sha256') != recorded[HARNESS]
            or proof.get('after_sha256') != current[HARNESS]):
        return None
    original = (base/proof['retained_original_harness']).resolve()
    if not original.is_relative_to(base.resolve()):
        return None
    if hashlib.sha256(original.read_bytes()).hexdigest() != recorded[HARNESS]:
        return None
    return {'kind': 'unchanged_production_and_executed_harness',
            'unused_harness_difference': HARNESS,
            'repair_record': 'harness_repair.json',
            'explanation': 'Only the separate gap-regression script changed. '
                           'The production code, configs, this job runner and all '
                           'other snapshotted harnesses are identical.'}
