"""Read-only module-boundary tracing for native/unified CPU conformance.

Tensor hooks return None and never install backward module hooks (which can
introduce views). Snapshots are detached copies, bounded to one training segment.
Functional operations inside forward are outside this module-boundary audit.
"""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import inspect
import torch


def tensors(value, path='$'):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from tensors(value[key], path + '[' + repr(key) + ']')
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from tensors(child, path + '[' + str(i) + ']')


def stats():
    return dict(pairs=0, exact=0, byte_exact=0, within_tolerance=0,
                max_abs=0., nonzero_native=0, nonzero_unified=0)


def byte_view(tensor):
    flat = tensor.contiguous().reshape(-1)
    # PyTorch regards a length-one view as contiguous even with stride > 1.
    # dtype reinterpretation nevertheless requires a unit last stride.
    if flat.stride(-1) != 1:
        flat = torch.empty(flat.numel(), dtype=flat.dtype, device=flat.device).copy_(flat)
    return flat.view(torch.uint8)


class ActivationAudit:
    def __init__(self, name, pairs, directory, excluded_modules=()):
        self.name = name
        self.directory = Path(directory) / name.replace('/', '_')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active = None
        self.execution = 'forward'
        self.handles = []
        self.events = {'native': {}, 'unified': {}}
        self.calls = {'native': Counter(), 'unified': Counter()}
        self.coverage = {}
        self.aggregates = {}
        self.totals = {k: stats() for k in ('forward_input', 'forward_output',
                                          'activation_gradient', 'parameter_gradient')}
        self.structural_mismatches = []
        self.differences = []
        self.saved = 0
        self.segments = 0
        self.steps = set()
        self.gradient_coverage = Counter()
        self.absent_gradient_outputs = {}
        self.excluded_modules = list(excluded_modules)
        for role, native, unified in pairs:
            for side, model in [('native', native), ('unified', unified)]:
                for path, module in model.named_modules():
                    key = role + '/' + (path or '<root>')
                    self.coverage.setdefault(key, dict(type=type(module).__module__ + '.' + type(module).__name__,
                                                       native_calls=0, unified_calls=0,
                                                       native_backward_recompute_calls=0, unified_backward_recompute_calls=0,
                                                       excluded=path in excluded_modules))
                    if path in excluded_modules:
                        continue
                    signature=inspect.signature(module.forward)
                    def pre(mod, args, kwargs, side=side, key=key, signature=signature):
                        if self.active != side:
                            return
                        index = self.calls[side][key]
                        self.calls[side][key] += 1
                        self.coverage[key][side + '_calls'] += 1
                        if self.execution=='backward_recompute':
                            self.coverage[key][side+'_backward_recompute_calls'] += 1
                        # Stack supports recursive/repeated calls of one module.
                        self.stacks[side].setdefault(key, []).append(index)
                        # Positional vs keyword use is an API spelling difference.
                        bound=signature.bind(*args, **kwargs)
                        bound.apply_defaults()
                        self.capture(side, key, index, 'input', bound.arguments)
                    def post(mod, args, kwargs, output, side=side, key=key):
                        if self.active != side:
                            return
                        index = self.stacks[side][key].pop()
                        self.capture(side, key, index, 'output', output)
                    self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
                    self.handles.append(module.register_forward_hook(post, with_kwargs=True))

    def begin(self, step, segment='training'):
        self.step, self.segment = step, segment
        self.steps.add(step)
        self.events = {'native': {}, 'unified': {}}
        self.calls = {'native': Counter(), 'unified': Counter()}
        self.stacks = {'native': {}, 'unified': {}}
        self.active = None

    def side(self, side, execution='forward'):
        self.active = side
        self.execution = execution

    @contextmanager
    def suspended(self):
        old = self.active
        self.active = None
        try:
            yield
        finally:
            self.active = old

    def capture(self, side, module, index, direction, value):
        for path, tensor in tensors(value):
            key = (module, index, direction, path)
            record = dict(value=tensor.detach().cpu().clone(), requires_grad=tensor.requires_grad,
                          gradient=None, backward_calls=0, execution=self.execution,
                          grad_enabled=torch.is_grad_enabled(),grad_fn=type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None)
            self.events[side][key] = record
            # Only output gradients are activation gradients in this report.
            # Input snapshots still locate preprocessing/time/layout differences.
            if direction == 'output' and tensor.requires_grad:
                def hook(gradient, record=record):
                    copy = gradient.detach().cpu().clone()
                    record['gradient'] = copy if record['gradient'] is None else record['gradient'] + copy
                    record['backward_calls'] += 1
                tensor.register_hook(hook)

    def structural(self, detail):
        if len(self.structural_mismatches) < 100:
            self.structural_mismatches.append(dict(step=self.step, segment=self.segment, **detail))

    def compare_tensor(self, kind, key, expected, actual):
        label = self.segment + '/' + key
        aggregate = self.aggregates.setdefault(kind + '/' + label, stats())
        total = self.totals[kind]
        for s in (aggregate, total):
            s['pairs'] += 1
        compatible = expected.shape == actual.shape and expected.dtype == actual.dtype
        exact = compatible and torch.equal(expected, actual)
        byte_exact = compatible and torch.equal(byte_view(expected), byte_view(actual))
        if compatible:
            # Identical +/-inf attention-mask entries are equal, not NaN errors.
            promoted=torch.complex128 if actual.is_complex() else torch.float64
            left=actual.to(promoted);right=expected.to(promoted)
            diff = torch.where(actual == expected, 0., (left - right).abs())
            diff = torch.nan_to_num(diff, nan=float('inf'), posinf=float('inf'))
            max_abs = diff.max().item() if diff.numel() else 0.
            within = bool(torch.isclose(left, right, atol=1e-6, rtol=1e-5).all())
        else:
            max_abs, within = float('inf'), False
        for s in (aggregate, total):
            s['exact'] += int(exact)
            s['byte_exact'] += int(byte_exact)
            s['within_tolerance'] += int(within)
            s['max_abs'] = max(s['max_abs'], max_abs)
            s['nonzero_native'] += int(bool(torch.any(expected != 0)))
            s['nonzero_unified'] += int(bool(torch.any(actual != 0)))
        if not byte_exact and len(self.differences) < 100:
            record = dict(step=self.step, segment=self.segment, kind=kind, key=key,
                          native_shape=list(expected.shape), unified_shape=list(actual.shape),
                          native_dtype=str(expected.dtype), unified_dtype=str(actual.dtype),
                          max_abs=max_abs, within_tolerance=within, values_exact=exact, byte_exact=byte_exact)
            if self.saved < 8:
                file = self.directory / ('difference_' + str(self.saved) + '.pt')
                torch.save(dict(native=expected, unified=actual, metadata=record), file)
                record['tensor_pair'] = str(file)
                self.saved += 1
            self.differences.append(record)

    def parameters(self, native, unified, role='network'):
        a, b = dict(native.named_parameters()), dict(unified.named_parameters())
        if a.keys() != b.keys():
            self.structural(dict(kind='parameter_names'))
        for key in a.keys() & b.keys():
            ga, gb = a[key].grad, b[key].grad
            if (ga is None) != (gb is None):
                self.structural(dict(kind='parameter_gradient_presence', key=role + '/' + key))
            elif ga is not None:
                self.compare_tensor('parameter_gradient', role + '/' + key, ga, gb)
            else:
                self.gradient_coverage['parameter_gradient_absent_pairs'] += 1

    def finish_segment(self):
        self.active = None
        self.segments += 1
        a, b = self.events['native'], self.events['unified']
        if a.keys() != b.keys():
            self.structural(dict(kind='module_tensor_paths', native_only=[str(k) for k in a.keys()-b.keys()][:30],
                                 unified_only=[str(k) for k in b.keys()-a.keys()][:30]))
        if self.calls['native'] != self.calls['unified']:
            self.structural(dict(kind='module_call_counts', native=dict(self.calls['native']), unified=dict(self.calls['unified'])))
        for key in sorted(a.keys() & b.keys()):
            ea, eb = a[key], b[key]
            module, index, direction, path = key
            label = module + '/call' + str(index) + '/' + path
            self.compare_tensor('forward_' + direction, label, ea['value'], eb['value'])
            if ea['requires_grad'] != eb['requires_grad']:
                self.structural(dict(kind='requires_grad', key=label, native=ea['requires_grad'], unified=eb['requires_grad']))
            if ea['execution'] != eb['execution']:
                self.structural(dict(kind='execution_phase', key=label, native=ea['execution'], unified=eb['execution']))
            if direction != 'output':
                continue
            ga, gb = ea['gradient'], eb['gradient']
            if (ga is None) != (gb is None) or ea['backward_calls'] != eb['backward_calls']:
                self.structural(dict(kind='activation_gradient_presence', key=label,
                                     native_calls=ea['backward_calls'], unified_calls=eb['backward_calls']))
            elif ga is not None:
                self.compare_tensor('activation_gradient', label, ga, gb)
                self.gradient_coverage['compared_activation_gradient_pairs'] += 1
            elif ea['requires_grad']:
                self.gradient_coverage['requires_grad_but_unused_output_pairs'] += 1
                cause='View created with grad disabled (e.g. reentrant checkpoint initial pass); gradients belong to the backward recomputation.' if not ea['grad_enabled'] and ea['grad_fn'] is None else 'Both output gradients absent from the executed loss graph.'
                self.absent_gradient_outputs.setdefault(self.segment+'/'+label,dict(native=None,unified=None,grad_enabled=ea['grad_enabled'],grad_fn=ea['grad_fn'],execution=ea['execution'],reason=cause+' Not counted as measured activation gradients.'))
            else:
                self.gradient_coverage['no_grad_output_pairs'] += 1
        self.events = {'native': {}, 'unified': {}}

    def close(self):
        self.active = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return dict(name=self.name, steps=len(self.steps), segments=self.segments, totals=self.totals,
                    structural_mismatches=self.structural_mismatches, first_differences=self.differences,
                    coverage=self.coverage, by_module_tensor=self.aggregates,
                    excluded_module_paths=self.excluded_modules,
                    gradient_coverage=dict(self.gradient_coverage),
                    absent_gradient_outputs=self.absent_gradient_outputs,
                    definition='All tensor leaves at executed nn.Module input/output boundaries, matched by qualified module path, invocation index and tensor path. Output tensor hooks measure dL/dactivation. Unexecuted modules and internal functional operators are not certified.',
                    tolerance=dict(atol=1e-6, rtol=1e-5),
                    exact_definition='torch.equal plus shape/dtype; byte_exact additionally compares contiguous uint8 representations')
