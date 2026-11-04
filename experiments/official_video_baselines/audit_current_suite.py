"""Current readiness audit; historical blocked/failed reports remain immutable."""
from typing import Any
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket

WORK = Path(__file__).resolve().parents[3]
G = WORK / 'gpu_checks_20261006'


def audit() -> Any:
    evidence = [
        'PF_FULL_MODEL_NATIVE_FASTVIDEO_PARITY.json', 'PF_FULL_MODEL_DDP2_NATIVE_FASTVIDEO_PARITY.json',
        'PF_81_FRAME_NATIVE_FASTVIDEO_DECODE_PARITY.json', 'PW_NATIVE_TEACHER_CURRENT_FRAMEWORK_PARITY.json',
        'pw_packed_full_fp32_v3.json', 'WAN_NATIVE_ADAPTER_FULL_PARITY.json', 'WAN_ORIGINAL_VAE_ALL_WEIGHTS.json',
        'WAN_ORIGINAL_FULL_PIPELINE_PARITY.json', 'CURRENT_SOLVER_ACTUAL_DECODE_MATRIX.json',
        'ADM_ACTUAL_EVALUATOR_SELF_CHECK.json', 'V5_PUBLIC_CLI_FULL_MODEL_DDP2_ATOMIC_RESUME.json',
        'DEADLINE_BOUNDARY_SAVE_RESUME.json', 'OFFICIAL_VBENCH_ALL16_REAL_INTEGRATION.json',
        'PREPROCESSING_REAL_CACHE_CPU_REGRESSION.json', 'QUALITY_CHUNK_CURRENT_ENTRYPOINTS.json',
        'REAL_DATA_FINAL_HASH_AUDIT.json', 'FINAL_MEDIA_DECODE_AUDIT.json', 'CURRENT_REAL_LATENCY_CI_CHECK.json',
        'CURRENT_TRAINED_HELDOUT_AND_TIMING.json', 'OFFICIAL_SCORE_PROVENANCE_CPU_CHECK.json',
        'FINAL_ENTRYPOINT_VALIDATION.json'
    ]
    checks = {}
    files = [G / name for name in evidence] + [
        WORK / name
        for name in ('SOLVER_MATH_CPU_20261006.json', 'PROJECTION_ALGEBRA_CPU_20261006.json',
                     'PW_PAPER_MATH_CPU_20261006.json', 'PUBLIC_CLI_CPU_CONTRACT.json', 'PRODUCTION_PARITY_STATUS.json',
                     'FINAL_SOURCE_FORMAT_CHECK.json', 'OFFICIAL_EVALUATOR_ASSET_LEDGER_20261006.json',
                     'FINAL_SHARED_FASTVIDEO_SAFETY_CHECK.json')
    ]
    for file in files:
        name = str(file.relative_to(WORK))
        checks[name] = dict(exists=file.is_file(),
                            status=json.loads(file.read_text()).get('status') if file.is_file() else 'pending',
                            sha256=hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None)
    data = {}
    for family in ['pf', 'wan']:
        path = WORK / 'fair_data_20261006' / ('fixture14_' + family) / 'cache_audit.json'
        value = json.loads(path.read_text()) if path.is_file() else {}
        cached_files = [(p, h) for row in value.get('committed_clips', []) for p, h in row['file_sha256'].items()]
        data[family] = dict(complete=value.get('complete_manifest_processed', False),
                            counts=value.get('counts'),
                            committed_files=len(cached_files),
                            files_exist=all(Path(p).is_file() for p, h in cached_files),
                            original_vae_precision=[c['vae_weight_precision'] for c in value.get('contracts', [])])
    checkpoints = {}
    for family in ('native_rdd', 'native_pyramid_flow', 'paper_pyramidal_wan'):
        folder = G / ('v5_cli_' + family + '_continuous/checkpoint-3')
        metadata = json.loads((folder / 'metadata.json').read_text())
        contract = metadata['training_contract']
        checksums = contract['implementation']
        matches = {
            name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected
            for name, expected in checksums.items()
        }
        checkpoints[family] = dict(path=str(folder),
                                   committed=json.loads((folder / 'complete.json').read_text()).get('complete') is True,
                                   current_implementation_matches=all(matches.values()),
                                   source_files=len(matches),
                                   changed_files=[name for name, ok in matches.items() if not ok],
                                   recipe_steps=contract['configured_max_train_steps'],
                                   trained_updates=metadata['step'])
    full_info = json.loads((WORK / 'paper_sources_20261006/VBench_full_info.json').read_text())
    vbench_asset = WORK / 'assets_20261006/vbench/preparation.json'
    format_value = json.loads((WORK / 'FINAL_SOURCE_FORMAT_CHECK.json').read_text())
    current_sources = {
        name: (WORK / name).is_file() and hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected
        for name, expected in format_value['source_sha256'].items()
    }
    asset_value = json.loads((WORK / 'OFFICIAL_EVALUATOR_ASSET_LEDGER_20261006.json').read_text())
    evaluator_files = {
        name: (WORK / name).is_file() and (WORK / name).stat().st_size == expected['bytes']
        for name, expected in asset_value['files'].items()
    }
    report = dict(
        status='passed' if all(x['status'] == 'passed' for x in checks.values())
        and all(x['complete'] and x['files_exist']
                for x in data.values()) and all(x['committed'] and x['current_implementation_matches']
                                                for x in checkpoints.values()) and all(current_sources.values())
        and all(evaluator_files.values()) else 'incomplete',
        host=socket.gethostname(),
        gpu_policy=json.loads((WORK / 'CURRENT_RUN_POLICY_20261006.json').read_text()),
        disk_free_gib=shutil.disk_usage(WORK).free / 2**30,
        checks=checks,
        data=data,
        exact_resume_checkpoints=checkpoints,
        current_validated_source_sha256_matches=current_sources,
        evaluator_assets_and_locks_exist_with_recorded_sizes=evaluator_files,
        vbench=dict(official_records=len(full_info),
                    distinct_prompts=len({x['prompt_en']
                                          for x in full_info}),
                    required_samples=5 * len({x['prompt_en']
                                              for x in full_info}),
                    assets_preparation=json.loads(vbench_asset.read_text())['status']),
        scopes=dict(
            native_pf='Original author modules/runner, source-pinned; native Flux/CLIP/T5/CausalVideoVAE',
            native_wan=
            'Official HF Diffusers Wan modules, original FP32 VAE; replicatedDP SP=TP1; not Alibaba source-runner parity',
            pyramidal_wan='Strict published paper implementation; no author-source parity claim; private80K absent',
            controlled=
            'Five spatial-only methods on identical original Wan architecture/init/data/runtime; not native PF/PW'),
        quality_training_complete=False,
        standard_vbench_results_complete=False,
        paper_matrix='PAPER_EXPERIMENT_MATRIX_20261006.json; prepared commands, not executed formal results',
        external_assets_required=[
            'Formal captioned training/validation dataset',
            'Authors private80K if claiming original PW training corpus',
            'Existing ImageNet Unified-L CFG1.5 NPZ and 50K reference for exp25',
            'Native ImageNet DiT/RDD checkpoints for exp12'
        ],
        gpu_used=False)
    report['engineering_preparation_complete'] = report['status'] == 'passed'
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=WORK / 'CURRENT_READINESS_20261006.json')
    args = p.parse_args()
    report = audit()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
