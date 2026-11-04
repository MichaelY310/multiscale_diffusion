#!/usr/bin/env python3
"""Re-audit saved numeric evidence and configuration provenance, CPU files only."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/strict_original_unified_audit_20261010'
sys.path.insert(0, str(ROOT / 'tools'))
from paired_experiments import load_manifest, compile_plan
from prepare_gpu_checks import source_integrity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def config_delta(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(a.keys() | b.keys()):
            child = path + '.' + key
            if key not in a or key not in b:
                rows.append(dict(path=child, old=a.get(key), current=b.get(key)))
            else:
                rows += config_delta(a[key], b[key], child)
        return rows
    return [] if a == b else [dict(path=path, old=a, current=b)]


def activation_stats(a):
    if isinstance(a, list):
        return [activation_stats(row) for row in a]
    return dict(totals=a['totals'], structural_mismatches=a['structural_mismatches'],
        unexecuted_module_types=dict(collections.Counter(v['type'] for v in a['coverage'].values() if not v['native_calls'])),
        unexecuted_modules=[key for key, value in a['coverage'].items() if not value['native_calls']],
        gradient_coverage=a['gradient_coverage'], excluded_modules=a['excluded_module_paths'],
        all_captured_byte_equal=not a['structural_mismatches'] and all(v['pairs'] == v['byte_exact'] for v in a['totals'].values()))


def numeric_record(row):
    path = ROOT / row['evidence']; data = read(path)
    out = dict(**row, sha256=sha(path), gpu_used=data['gpu_used'])
    if data['status'] != row['status']:
        raise ValueError('Report/index status mismatch: ' + str(path))
    if 'activation' in data:
        out['activation'] = activation_stats(data['activation'])
    if 'trajectory' in data:
        out['trajectory'] = data['trajectory']
        for step in data['trajectory']:
            if not step.get('loss_exact', True) or not step.get('rng_exact', True):
                raise ValueError('Indexed training report not exact: ' + str(path))
            for key in ('model', 'optimizer', 'ema'):
                if step[key]['tensors'] != step[key]['exact']:
                    raise ValueError('Indexed state comparison not exact: ' + str(path))
    if isinstance(data.get('checks'), list):
        checks = data['checks']; unequal = [v for v in checks if not v.get('bitwise')]
        out['numeric_checks'] = len(checks)
        out['non_value_equal_checks'] = unequal
        out['checks'] = checks
    else:
        out['state_checks'] = data.get('checks')
    out['scope'] = data.get('scope', data.get('quality_claim'))
    return out


def main():
    manifest = load_manifest()
    baseline = read(ROOT / 'evidence/paired_experiments_20261010/SUMMARY.json')
    changed = [file for file, expected in baseline['code_sha256'].items() if sha(ROOT / file) != expected]
    if changed:
        raise ValueError('Baseline implementation changed during audit: ' + repr(changed))
    profiles = []
    for entry in manifest['profiles']:
        plan = compile_plan(manifest, entry, 'paper')
        profiles.append(dict(id=entry['id'], paper=entry['paper'], config_sha256=entry['config_sha256'],
            contract_sha256=plan['contract_sha256'], native_cli_certified=plan['production_entrypoints']['certified'],
            native_cli_blockers=plan['production_entrypoints']['blocked_reasons'],
            matched_core_blockers=plan['blocked_reasons'], required_assets=plan['required_assets'],
            author_contract=entry['paper_contract'], compiled_controls=plan['contract']['controls'],
            production_references=plan['production_entrypoints'],
            template_notes=plan['contract']['method_config']['evidence']['notes']))
    train = {mode: [numeric_record(row) for row in rows] for mode, rows in baseline['training'].items()}
    other = [numeric_record(row) for row in baseline['other_checks']]
    latest_keys = ['relay', 'mdm_rng', 'comparator', 'native_cli', 'saved_sampling', 'data_order', 'lap_cli', 'labels', 'cli_inventory']
    latest_keys += ['component_' + name for name in ('relay', 'ppflow', 'lapflow', 'rdd')]
    probes = {name: dict(evidence=str((OUT / (name + '.json')).relative_to(ROOT)),
                        sha256=sha(OUT / (name + '.json')), **read(OUT / (name + '.json')))
              for name in latest_keys}
    code_sha = sha(ROOT / 'tools/probe_parity_claims.py')
    if any(row['probe_code_sha256'] != code_sha for row in probes.values()):
        raise ValueError('Rerun probes after changing the probe source so all reports bind the final code')
    full = []
    directory = ROOT / 'evidence/current_recipe_parity_20261009'
    for path in sorted(directory.glob('*.json')):
        data = read(path)
        if not data.get('profile') or 'activation' not in data:
            continue
        full.append(dict(evidence=str(path.relative_to(ROOT)), sha256=sha(path), profile=data['profile'],
            status=data['status'], steps=data.get('steps'), activation=activation_stats(data['activation']),
            recorded_harness_sha256=data.get('harness_sha256'),
            stored_harness_matches_current=(data.get('harness_sha256') == sha(ROOT / 'tools/compare_released_profiles.py')),
            template_sha256_matches_current=sha(ROOT / data['template']) == data['template_sha256'],
            stored_config_differences=config_delta(data['config'], read(ROOT / data['template'])),
            original_best_config_fully_executed=data['original_best_config_fully_executed'],
            precision_deviation=data['precision_deviation'], deviations=data['deviations']))
    totals = {}
    for mode, records in train.items():
        totals[mode] = {}
        for kind in ('forward_input', 'forward_output', 'activation_gradient', 'parameter_gradient'):
            totals[mode][kind] = {key: sum(row['activation']['totals'][kind][key] for row in records)
                                for key in ('pairs', 'byte_exact', 'nonzero_native')}
    cleanup = []
    payloads = OUT / 'component_payloads'
    if payloads.exists():
        for path in sorted(payloads.rglob('*')):
            if path.is_file() and path.suffix in ('.pt', '.png'):
                if path.is_symlink() or not path.resolve().is_relative_to(payloads.resolve()):
                    raise ValueError('Refuse to clean a symlink/outside payload')
                cleanup.append(dict(file=str(path.relative_to(ROOT)), size=path.stat().st_size, sha256=sha(path)))
        cleanup_record = OUT / 'ARTIFACT_CLEANUP.json'
        if cleanup or not cleanup_record.exists():
            cleanup_record.write_text(json.dumps(dict(temporary_only=True, files=cleanup), indent=2) + '\n')
        for item in cleanup:(ROOT / item['file']).unlink()
    result = dict(status='incomplete_with_confirmed_differences', gpu_used=False,
        formal_training_started=False, models_or_data_downloaded=False,
        answers=dict(original_is_identical_to_all_papers=False, all_best_results_reproduced=False,
                     every_unified_detail_identical_to_native_cli=False,
                     all_framework_checks_byte_exact=False,
                     captured_final_native_profile_module_values_and_gradients_equal=True),
        source_integrity=source_integrity(), baseline_implementation_files_verified=len(baseline['code_sha256']),
        confirmed_static_relay_differences=dict(
            source_default_augment_probability=0.12, source_augment_dim=9,
            template_augment_dim=0, template_augment_pipe=False,
            source_high_stage_default_ema_half_life_images=500000,
            template_high_stage_ema_half_life_images=500000000,
            ema_difference_factor=1000,
            source_paths=['sources/relay/README.md', 'sources/relay/train.py'],
            config_path='configs/paper_recipes/relay_imagenet_high.json'),
        methods=manifest['papers'], profiles=profiles, mapping_only_unexecuted=manifest['unreleased'],
        latest_training=train, totals=totals, latest_other=other, probes=probes, historical_full_profiles=full,
        old_paper_suite_status=read(ROOT / 'evidence/paired_experiments_20261010/validation/paper/SUMMARY.json')['status'],
        mdm_note='Old whole-suite failure is retained; final root index uses corrected mdm_final reports. This is not a pending MDM shape failure.',
        pending_cpu_work=['Relay augmentation/augment_dim and native backward normalization.',
                          'Correct/certify all original CLI overlays, including LapFlow dataset/model/stage/mask parameters.',
                          'Require real labels for conditional DCTdiff and Relay; never fabricate labels for real data.',
                          'Remove extra MDM setup construction RNG and compare real initialization/text/data lifecycles.',
                          'Match original split/shuffle/drop_last/workers, codec posterior draws and callback RNG.',
                          'Strengthen numeric comparators to check dtype/bytes and distinguish raw gradients from clipped/repaired gradients.',
                          'Bind source-specific pretrained assets/hashes, exact full training budgets and evaluation rows.'],
        temporary_component_payloads_removed=len(cleanup),
        audit_code_sha256=sha(Path(__file__)), report='REPORT_ZH.md')
    (OUT / 'AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], profiles=len(profiles),
        paper_short_reports=len(train['paper']), fair_short_reports=len(train['fair-rdd']),
        probes=len(probes), saved_raw_sampling_pairs=len(probes['saved_sampling']['pairs']),
        original_cli_certified=sum(row['native_cli_certified'] for row in profiles),
        gpu_used=False, source_files=result['source_integrity']['selected_files'])))


if __name__ == '__main__':
    raise SystemExit(main())
