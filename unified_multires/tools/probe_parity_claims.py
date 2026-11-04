#!/usr/bin/env python3
"""Small CPU probes of gaps in original/unified equivalence claims.

No official training job, pretrained assets, downloads or CUDA use. Each probe
runs in a separate process to keep author package namespaces isolated.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '_deps'), str(ROOT / 'tools')]
OUT = ROOT / 'evidence/strict_original_unified_audit_20261010'
COMPONENT = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relay():
    import torch
    from paired_experiments import load_manifest, compile_plan
    from umdiff.training import Trainer
    from umdiff.engine import TrainingResult
    path = ROOT / 'sources/relay/training/training_loop.py'
    tree = ast.parse(path.read_text())
    scale = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 't' for t in n.targets)
                 and isinstance(n.value, ast.IfExp))
    backward = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                    and ast.unparse(n).startswith('loss.sum().mul(loss_scaling / t).backward()'))
    code = compile(ast.Module(body=[scale, backward], type_ignores=[]), str(path), 'exec')

    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.value = torch.nn.Parameter(torch.ones(()))
    class ScalarLoss:
        config = {}
        def training(self, net, *a, **kw):
            return TrainingResult(net.value, None, None, None, {})

    manifest = load_manifest(); rows = []
    for stage, resolution, loss_scaling in [('low', 64, 25), ('high', 256, 1)]:
        entry = next(e for e in manifest['profiles'] if e['id'] == 'relay_imagenet_' + stage)
        plan = compile_plan(manifest, entry, 'paper')
        recipe = plan['contract']['method_config']['training']['optimizer']
        net = ScalarModel(); trainer = Trainer(ScalarLoss(), net, **recipe)
        # Retain the compiled global4096 / micro1 / accumulation4096 split.
        trainer.step(torch.zeros(4096, 1, 1, 1), None)
        actual = float(net.value.grad)
        references = []
        for world in (1, 8):
            p = torch.ones((), requires_grad=True)
            local = 4096 // world
            # Constant per-pixel objective makes splitting into microbatches
            # immaterial. For8 ranks, identical local gradients also make the
            # mathematical DDP average identical; no DDP process is started.
            env = dict(images=torch.empty(1, 3, resolution, resolution),
                       loss=p.expand(local, 3, resolution, resolution),
                       loss_scaling=loss_scaling)
            exec(code, env)
            expected = float(p.grad)
            references.append(dict(world_size=world, local_examples=local,
                source_formula_gradient=expected, compiled_core_gradient=actual,
                compiled_over_source=actual / expected,
                exact=actual == expected))
        rows.append(dict(stage=stage, resolution=resolution,
                         compiled_contract_sha256=plan['contract_sha256'],
                         compiled_loss_multiplier=recipe['loss_multiplier'], references=references))
    assert all(not ref['exact'] for row in rows for ref in row['references'])
    return dict(status='confirmed_gradient_normalization_difference', cases=rows,
        source=str(path.relative_to(ROOT)), source_sha256=sha(path),
        source_lines=[scale.lineno, backward.lineno],
        scope='Source backward normalization AST vs actual compiled Trainer on a scalar fixture; no Relay network, data, DDP or quality test.')


def mdm_rng():
    import torch
    from umdiff.source_objectives import setup
    config = json.loads((ROOT / 'configs/paired_experiments/mdm_cc12m256.json').read_text())
    images = torch.zeros(3, 3, 64, 64); labels = torch.arange(3)
    conditions = dict(lm_outputs=torch.zeros(3, 8, 32), lm_mask=torch.ones(3, 8))
    torch.manual_seed(20261010)
    prior = torch.get_rng_state().clone()
    setup('mdm', config, images=images, labels=labels, input_conditions=conditions)
    after = torch.get_rng_state()
    changed = not torch.equal(prior, after)
    assert changed
    return dict(status='confirmed_extra_setup_rng_consumption', rng_changed=changed,
        before_sha256=hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
        after_sha256=hashlib.sha256(after.numpy().tobytes()).hexdigest(),
        all_inputs_supplied=True,
        scope='Binding the source-core objective constructs/discards a tiny native MDM before either objective; neither forward nor backward executed.')


def comparator():
    import torch
    import conformance
    expected = torch.tensor([1, 2], dtype=torch.int64)
    actual = torch.tensor([1, 2], dtype=torch.float32)
    conformance.compare('strict_audit/dtype_blindspot', actual, expected, atol=0, rtol=0)
    check = conformance.CHECKS[-1]
    assert check['passed'] and check['bitwise'] and actual.dtype != expected.dtype
    return dict(status='confirmed_comparator_dtype_blindspot', recorded_check=check,
        actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
        actual_bytes=actual.numpy().tobytes().hex(), expected_bytes=expected.numpy().tobytes().hex(),
        scope='A comparator regression probe, not an observed dtype mismatch in any paper output. conformance.bitwise uses torch.equal, not byte/dtype comparison.')


def native_cli():
    import subprocess
    rows = []
    with tempfile.TemporaryDirectory(prefix='umdiff_strict_native_cli_') as directory:
        for backend in ('original', 'unified'):
            result = subprocess.run([sys.executable, str(ROOT / 'tools/paired_experiments.py'),
                '--backend', backend, '--profile', 'rdd_unified_xl_300_300_400',
                '--mode', 'fair-rdd', '--action', 'run', '--entrypoint', 'native-cli',
                '--output', str(Path(directory) / backend)], text=True, capture_output=True, timeout=30)
            rejected = result.returncode == 2 and 'not certified' in result.stderr
            assert rejected
            rows.append(dict(backend=backend, exit_code=result.returncode, rejection=result.stderr.strip(), rejected=rejected))
    return dict(status='confirmed_native_cli_run_blocked', cases=rows,
        scope='Negative launcher test; the requested action is rejected before model import/training. Temporary plan files removed.')


def saved_sampling():
    import torch
    rows = []
    directory = ROOT / 'evidence/current_recipe_parity_20261009/sampling_profiles'
    for path in sorted(directory.glob('*.pt')):
        # These fixed-path artifacts were written by our earlier sampler tests,
        # not supplied/downloaded models. PixelFlow's native output is NumPy,
        # which the restricted weights loader cannot deserialize.
        value = torch.load(path, map_location='cpu', weights_only=False)
        native_type, unified_type = type(value['native']).__name__, type(value['unified']).__name__
        a, b = torch.as_tensor(value['native']), torch.as_tensor(value['unified'])
        exact = a.shape == b.shape and a.dtype == b.dtype and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
        rows.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
            shape=list(a.shape), native_dtype=str(a.dtype), unified_dtype=str(b.dtype),
            native_type=native_type, unified_type=unified_type, byte_exact=exact))
    assert rows
    return dict(status='existing_saved_sampling_pairs_byte_exact' if all(row['byte_exact'] for row in rows) else 'existing_saved_sampling_pairs_different', pairs=rows,
        scope='Re-read existing Oct9 reduced-model sampling payloads on CPU; no new sampling, pretrained assets or backward test.')


def data_order():
    import torch
    from paired_experiments import execute
    tree = ast.parse(Path(sys.modules[execute.__module__].__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    nodes = [next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))
             for key in ('absolute', 'indices')]
    code = compile(ast.Module(body=nodes, type_ignores=[]), 'paired_experiments.execute/data_order', 'exec')
    dataset = list(range(10)); compiled = []
    for step in range(3):
        env = dict(step=step, batch=4, dataset=dataset, recipe={'shuffle': False})
        exec(code, env); compiled.append(env['indices'])
    native = []
    for epoch in range(2):
        native.extend(batch.tolist() for batch in torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True))
    assert compiled[2] != native[2]
    return dict(status='confirmed_epoch_tail_policy_difference', compiled_first_three_batches=compiled,
        native_drop_last_first_three_batches=native[:3], dataset_size=10, global_batch=4,
        scope='Actual compiled index expressions vs source-recipe drop_last policy on a deterministic range fixture; no model or formal training.')


def lap_cli():
    from types import SimpleNamespace
    from paired_experiments import compile_plan, load_manifest
    manifest = load_manifest(); entry = next(e for e in manifest['profiles'] if e['id'] == 'lapflow_imagenet_xl_2_paper')
    plan = compile_plan(manifest, entry, 'paper')
    argv = plan['production_entrypoints']['original']['argv']
    name = argv[argv.index('--dataset') + 1]
    path = ROOT / 'sources/lapflow/datasets_prep/__init__.py'
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'get_dataset')
    env = {}; exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    try:
        env['get_dataset'](SimpleNamespace(dataset=name))
    except UnboundLocalError as error:
        failure = str(error)
    else:
        raise AssertionError('Wrong dataset reference unexpectedly accepted')
    defaults = {}
    for node in ast.walk(ast.parse((ROOT / 'sources/lapflow/feb12_train_ours.py').read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for keyword in node.keywords:
                if keyword.arg == 'default' and node.args and isinstance(node.args[0], ast.Constant):
                    try: defaults[node.args[0].value] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError): pass
    names = ['--model_type', '--image_size', '--f', '--num_levels', '--num_in_channels', '--num_out_channels', '--use_mask', '--time_seg_small', '--label_dropout', '--num_classes', '--num_epoch']
    return dict(status='confirmed_invalid_and_incomplete_lapflow_reference', dataset_argument=name,
        supported_imagenet_name='imagenet_256', error=failure,
        missing_reference_flags={k:defaults.get(k, 'not_literal_or_unknown') for k in names if k not in argv},
        scope='Source get_dataset rejects the actual prepared reference name; static parser defaults show missing overlays. No dataset/model is loaded.')


def labels():
    import torch
    from umdiff.source_objectives import setup
    rows = []
    for name, template, size in [('dctdiff', 'dctdiff_imgnet64_uvit_small_2by2', 64),
                                 ('relay_low', 'relay_imagenet_low', 64)]:
        config = json.loads((ROOT / 'configs/paired_experiments' / (template + '.json')).read_text())
        _, original, _, _, _ = setup(name, config, images=torch.zeros(2, 3, size, size))
        closure = dict(zip(original.__code__.co_freevars, (cell.cell_contents for cell in original.__closure__)))
        kwargs = closure['kwargs']
        value = kwargs.get('y') if 'y' in kwargs else kwargs['class_labels'].argmax(1)
        assert value.tolist() == [0, 1]
        rows.append(dict(profile=template, labels_supplied=False, fabricated_classes=value.tolist()))
    return dict(status='confirmed_fabricated_class_labels_in_source_core_binding', cases=rows,
        scope='No model execution: real-image binding with labels omitted fabricates arange; execute required-label guard excludes conditional DCTdiff and Relay.')


def cli_inventory():
    import contextlib
    import io
    from paired_experiments import load_manifest, compile_plan
    manifest = load_manifest(); records = []
    for entry in manifest['profiles']:
        plan = compile_plan(manifest, entry, 'paper')
        command = plan['production_entrypoints']['original']
        cwd = Path(command['cwd']); argv = command['argv']
        scripts = [cwd / token for token in argv if token.endswith('.py')]
        flags = []
        for script in scripts:
            if script.exists():
                for node in ast.walk(ast.parse(script.read_text())):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                        flags += [value.value for value in node.args if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith('--')]
        passed = [token.split('=')[0] for token in argv if token.startswith('--') and token not in ('--standalone', '--nproc_per_node=1')]
        records.append(dict(profile=entry['id'], original_reference=command,
            python_scripts=[dict(path=str(path), exists=path.exists()) for path in scripts],
            argparse_options_found=sorted(set(flags)),
            unknown_argparse_options=sorted(set(passed) - set(flags)) if flags else None,
            unified_reference=plan['production_entrypoints']['unified'],
            compiled_controls=plan['contract']['controls'],
            compiled_asset_overlay_written=False, native_cli_certified=plan['production_entrypoints']['certified']))
    path = ROOT / 'sources/dfm/train.py'
    tree = ast.parse(path.read_text())
    names = {'build_parser', 'parse_args', '_normalize_key'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    selected += [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'DEFAULTS' for target in node.targets)]
    registry = {}
    for filename in ('sit.py', 'sit_dfm.py', 'sit_elit.py'):
        for node in ast.walk(ast.parse((ROOT / 'sources/dfm/models' / filename).read_text())):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SiT_models' for target in node.targets):
                registry.update({key.value: None for key in node.value.keys})
    env = dict(argparse=argparse, ALL_MODELS=registry)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    row = next(row for row in records if row['profile'] == 'dfm_sit_xl')
    args = row['original_reference']['argv'][2:]
    failures = []
    for extra in ([], ['--exp-name', 'strict_cpu_probe']):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:env['parse_args'](args + extra)
            except SystemExit as error:
                failures.append(dict(added_flags=extra, exit_code=error.code, error=stderr.getvalue().splitlines()[-1]))
    assert len(failures) == 2 and all(row['exit_code'] == 2 for row in failures)
    return dict(status='confirmed_incomplete_native_cli_references', profiles=records,
        dfm_parser_failures=failures,
        scope='All33 reference commands inspected. DFM original parser/default AST actually executed without model imports. Non-argparse/Hydra dynamic defaults are not certified by this inventory; no native training started.')


def component():
    import conformance
    from umdiff import native
    conformance.ARTIFACT_ROOT = OUT / 'component_payloads'
    conformance.RUNNERS[COMPONENT]()
    differences = [row for row in conformance.CHECKS if row['bitwise'] is not True]
    return dict(status='tolerance_pass_with_nonbitwise_checks' if differences else 'all_value_equal',
        component=COMPONENT, checks=conformance.CHECKS, source_executions=native.AUDIT,
        nonbitwise_checks=len(differences), maximum_nonbitwise_abs=max((x['max_abs'] for x in differences), default=0.),
        scope='Current generic component conformance including small shared-backbone objectives; this is distinct from the 33 native profile templates. No full training or pretrained image quality.')


def main():
    functions = dict(relay=relay, mdm_rng=mdm_rng, comparator=comparator,
                     native_cli=native_cli, saved_sampling=saved_sampling,
                     data_order=data_order, lap_cli=lap_cli, labels=labels,
                     cli_inventory=cli_inventory, component=component)
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('probe', choices=functions)
    p.add_argument('--component', choices=['relay', 'ppflow', 'lapflow', 'rdd'])
    a = p.parse_args()
    if a.probe == 'component' and not a.component:p.error('component probe requires --component')
    global COMPONENT
    COMPONENT = a.component
    if a.probe != 'native_cli':
        import torch
        torch.set_num_threads(2)
    result = functions[a.probe]()
    result.update(gpu_used=False, formal_training_started=False,
                  probe=a.probe, probe_code_sha256=sha(Path(__file__)))
    OUT.mkdir(parents=True, exist_ok=True)
    name = a.probe + ('_' + a.component if a.probe == 'component' else '')
    (OUT / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(probe=name, status=result['status'], gpu_used=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
