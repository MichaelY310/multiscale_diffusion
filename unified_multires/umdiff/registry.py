"""Small explicit component registry; custom modules use the same contracts."""
from copy import deepcopy
import importlib
import inspect

REGISTRY = {}

def register(group, name):
    def decorator(factory):
        key = (group, name)
        if key in REGISTRY:
            raise KeyError(f'Duplicate component: {key}')
        REGISTRY[key] = factory
        return factory
    return decorator

def build(group, spec, **injected):
    if not isinstance(spec, dict) or 'name' not in spec:
        raise TypeError(f'{group} requires {{"name":..., "args":...}}')
    unknown = set(spec) - {'name', 'args', 'module'}
    if unknown:
        raise ValueError(f'Unknown {group} configuration fields: {unknown}')
    if 'module' in spec:
        importlib.import_module(spec['module'])
    key = (group, spec['name'])
    if key not in REGISTRY:
        raise KeyError(f'Unknown component {key}; available: {sorted(k for k in REGISTRY if k[0]==group)}')
    args = deepcopy(spec.get('args', {}))
    if set(args) & set(injected):
        raise ValueError('Cannot override injected runtime objects')
    args.update(injected)
    # Bind before calling: typos must not silently change an experiment.
    inspect.signature(REGISTRY[key]).bind(**args)
    return REGISTRY[key](**args)
