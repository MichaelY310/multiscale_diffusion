#!/usr/bin/env python3
"""Independently read recorded module statistics and reject tolerance-only traces.

This audits already executed comparisons; it does not run another neural network.
Intentional source-bug/identity negative controls are outside the trace directories.
"""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('forward_input', 'forward_output', 'activation_gradient', 'parameter_gradient')
DIRECTORIES = ('full_current', 'reduced_current', 'long_release', 'extensions_current',
               'codec_current', 'rmd', 'inference_layers', 'inference_new', 'cat_codec',
               'cat_codec_full', 'rdd_optional', 'native_ssd', 'native_ppflow')


def traces(value, path=''):
    if isinstance(value, dict):
        totals = value.get('totals')
        if isinstance(totals, dict) and 'forward_output' in totals:
            yield path, value
        for key, child in value.items():
            yield from traces(child, path+'/'+str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from traces(child, path+'/'+str(index))


def numeric_checks(value, path=''):
    if isinstance(value, dict):
        if 'bitwise' in value and 'max_abs' in value:
            yield path, value
        for key, child in value.items():
            yield from numeric_checks(child, path+'/'+str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_checks(child, path+'/'+str(index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT/'evidence/unified_scope_20261012')
    args = parser.parse_args()
    base = args.evidence.resolve()
    rows, failures, missing = [], [], []
    for name in DIRECTORIES:
        directory = base/name
        if not directory.is_dir():
            missing.append(name)
            continue
        found = 0
        for file in sorted(directory.rglob('*.json')):
            if file.name == 'SUMMARY.json':
                continue  # summaries may embed the same traces; inspect their leaf reports
            report = json.loads(file.read_text())
            for path, trace in traces(report):
                found += 1
                location = str(file.relative_to(base))+':'+path
                totals = trace['totals']
                row_failures = []
                for kind in KINDS:
                    stats = totals.get(kind)
                    if stats is not None and (stats.get('byte_exact') != stats.get('pairs') or stats.get('max_abs') != 0.):
                        row_failures.append(kind)
                if totals['forward_output']['pairs'] <= 0:
                    row_failures.append('no observed forward outputs')
                if trace.get('first_differences') or trace.get('structural_mismatches'):
                    row_failures.append('recorded structural/numerical differences')
                for tensor, stats in trace.get('by_module_tensor', {}).items():
                    if stats.get('byte_exact') != stats.get('pairs') or stats.get('max_abs') != 0.:
                        row_failures.append(tensor)
                failures.extend(location+':'+item for item in row_failures)
                rows.append(dict(location=location, name=trace.get('name'), totals=totals,
                                 status='failed' if row_failures else 'passed',
                                 report_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        if not found:
            missing.append(name+' (no leaf traces)')
    checks_seen, deliberate_diagnostics = 0, []
    for file in sorted(base.rglob('*.json')):
        if any(part in ('suite_attempts','attempts','rdd_optional_preflight') for part in file.parts):
            continue
        if file.name in ('AUDIT.json','SUMMARY.json','trace_report_readback.json','comparator_additional_negative_controls.json'):
            continue
        report = json.loads(file.read_text())
        for path, check in numeric_checks(report):
            checks_seen += 1
            if check['bitwise']:
                continue
            location = str(file.relative_to(base))+':'+path
            identity = (str(file.relative_to(base)) == 'components/lapflow.json'
                        and check.get('name') == 'lapflow/reconstruction'
                        and check.get('passed') is True and check['max_abs'] <= 3e-7)
            if identity:
                deliberate_diagnostics.append(dict(location=location,name=check['name'],max_abs=check['max_abs'],
                    reason='FP32 decomposition/reconstruction identity; not an original-versus-unified tensor comparison.'))
            else:
                failures.append('Unexpected non-byte-exact numeric comparison: '+location)
    result = dict(status='passed' if rows and not failures and not missing else 'failed',
                  traces=len(rows), rows=rows, failures=failures, missing=missing,
                  numeric_checks=checks_seen, deliberate_nonexact_identity_diagnostics=deliberate_diagnostics,
                  scope='Readback of recorded module tensor and aggregate byte comparisons; excludes unexecuted modules, functional interiors, and intentional negative controls. No new forward/backward execution.',
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (base/'trace_report_readback.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'traces', 'failures', 'missing')}))
    return result['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
