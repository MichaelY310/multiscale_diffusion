#!/usr/bin/env python3
"""Exercise PixelFlow's separately selected paper Dopri5 recipe and pin provenance."""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from run_scoped_validation import snapshot


def worker(output, config):
    from unittest.mock import patch
    import torch
    import validate_profile_inference as audit
    from umdiff.native_models import activate
    activate('pixelflow', ['pixelflow'])
    from pixelflow.solver_ode_wrapper import ODE
    from umdiff.engine import Method
    audit.OUT, audit.CONFIG_OVERRIDE = output, config
    torch.set_num_threads(2)
    audit.training.seed(20261009)
    original, original_sample, calls = ODE.sample, Method.sample, []
    def matched_input(self, *args, model_kwargs=None, **kwargs):
        # Match the author's explicit torch.int32 input; this is fixture input
        # preparation, not a change to the unified sampling implementation.
        condition = dict(model_kwargs or {})
        labels = condition['class_labels']
        converted = labels.to(torch.int32)
        assert torch.equal(converted.to(labels.dtype), labels)
        condition['class_labels'] = converted
        return original_sample(self, *args, model_kwargs=condition, **kwargs)
    def sample(self, x, model, start, end, **kwargs):
        # Observe without modifying the solver, the model or its random draws.
        initial = x.detach().clone()
        result = original(self, x, model, start, end, **kwargs)
        calls.append(dict(initial=initial, interval=[start,end], kwargs=kwargs,
                          trajectory=result.detach().clone()))
        return result
    started = time.time()
    report = dict(profile='pixelflow_dopri5', gpu_used=False)
    try:
        with patch.object(ODE, 'sample', sample), patch.object(Method, 'sample', matched_input):
            details = audit.run('pixelflow')
        assert len(calls) == 8, 'Expected four native then four unified stages'
        for index in range(4):
            a, b = calls[index], calls[index+4]
            for key in ('initial','interval','kwargs','trajectory'):
                audit.CHECK(f'dopri5/stage{index}/{key}', b[key], a[key], atol=0, rtol=0,
                            scope='unclipped_native_ODE_state_and_boundary')
        report.update(status='passed', details=details, stages=4,
                      raw_stage_input_and_output_trajectories_compared=True,
                      input_alignment='The author pipeline explicitly constructs int32 class indices. The fixture supplies identical int32 indices to the unchanged Method.sample API. The earlier int64 fixture already had identical pixels/RNG but different raw index bytes; retained in attempts/.',
                      solver_reuse='Same published ODE.sample backend; separate pipeline vs unified stage/CFG/noise assembly.')
    except Exception:
        report.update(status='failed', failure=traceback.format_exc())
    report.update(seconds=time.time()-started, checks=audit.conformance.CHECKS,
                  source_executions=audit.native.AUDIT)
    (output/'pixelflow.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])))
    return report['status'] != 'passed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = ROOT/'configs/paired_experiments/pixelflow_xl_c2i_paper_dopri5.json'
    if args.worker:
        return worker(output, config)
    path = output/'pixelflow.json'
    if path.exists():
        history = output/'attempts'
        history.mkdir(exist_ok=True)
        path.rename(history/(str(time.time_ns())+'.json'))
    before = snapshot()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'PYTHONDONTWRITEBYTECODE':'1'}
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(output)]
    with (output/'run.log').open('w') as log:
        result = subprocess.run(command, env=env, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads(path.read_text())
    report['source_unchanged_during_test'] = snapshot() == before
    report['source_snapshot'] = before
    report['selected_config'] = str(config.relative_to(ROOT))
    report['selected_config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
    report['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if report.get('details'):
        report['details']['harness_default_template_label'] = report['details']['template']
        report['details']['template'] = config.stem
    report['all_recorded_comparisons_byte_exact'] = bool(report.get('checks')) and all(x.get('bitwise') is True and x.get('max_abs') == 0. for x in report['checks'])
    good = result.returncode == 0 and report['status'] == 'passed' and report['source_unchanged_during_test'] and report['all_recorded_comparisons_byte_exact']
    report['status'] = 'passed' if good else 'failed'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], checks=len(report.get('checks', [])), seconds=report.get('seconds'), config=report['selected_config'])))
    return not good


if __name__ == '__main__':
    raise SystemExit(main())
