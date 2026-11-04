"""Opt-in, narrowly scoped fixes; never overwrite pinned author files."""
import ast
import copy
import hashlib
from pathlib import Path
from .. import native
from ..registry import register


@register('model','native_ppflow_no_sigma_compat')
def ppflow_no_sigma(**kwargs):
    from ..native_models import ppflow
    kwargs=dict(kwargs)
    if kwargs.get('learn_sigma',False):raise ValueError('This compatibility adapter is only for learn_sigma=False')
    kwargs['learn_sigma']=False
    model=ppflow(**kwargs)
    path=native.ROOT/'sources/ppflow/models.py'
    tree=ast.parse(path.read_text());count=0
    forward=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='PPFlow'
                 for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    forward=copy.deepcopy(forward)
    class Fix(ast.NodeTransformer):
        def visit_IfExp(self,node):
            nonlocal count
            if ast.unparse(node.test)=='self.learn_sigma' and isinstance(node.orelse,ast.Call) and ast.unparse(node.orelse.func)=='self.unpatchify':
                if ast.unparse(node.orelse.args[1])!='channels * 2':raise ValueError('Author source changed; re-audit compatibility patch')
                node.orelse.args[1]=ast.Name(id='channels',ctx=ast.Load());count+=1
            return self.generic_visit(node)
    forward=Fix().visit(forward);ast.fix_missing_locations(forward)
    if count!=1:raise ValueError('Expected exactly one author no-sigma channel defect')
    env=dict(model.forward.__func__.__globals__)
    exec(compile(ast.Module(body=[forward],type_ignores=[]),str(path),'exec'),env)
    import types
    model.forward=types.MethodType(env['forward'],model)
    unpatchify=copy.deepcopy(next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='PPFlow'
        for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='unpatchify'))
    geometry_changes=0
    class Geometry(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal geometry_changes
            if isinstance(node.func,ast.Name) and node.func.id=='is_power_of_four':
                if ast.unparse(node.args[0])!='c // 2':raise ValueError('Author unpatchify geometry changed')
                node.args[0]=ast.Name(id='c',ctx=ast.Load());geometry_changes+=1
            return self.generic_visit(node)
    unpatchify=Geometry().visit(unpatchify);ast.fix_missing_locations(unpatchify)
    if geometry_changes!=1:raise ValueError('Expected exactly one no-sigma channel geometry defect')
    exec(compile(ast.Module(body=[unpatchify],type_ignores=[]),str(path),'exec'),env)
    model.unpatchify=types.MethodType(env['unpatchify'],model)
    model.compatibility_evidence=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        patch='No-sigma forward unpatchify channels*2→channels; no-sigma unpatchify square-geometry test c//2→c',changes=2)
    return model


def stop_only_loop(loop_class,max_updates):
    """Bound SSD's original while loop without changing its LR schedule.

    Only the while predicate changes; run_step/optimizer/LR/EMA/save statements
    are the original functions. No changes are made to checkpoint semantics.
    """
    if not isinstance(max_updates,int) or max_updates<1:raise ValueError('Positive update cap required')
    path=native.ROOT/'sources/ssd/guided_diffusion/train_util.py'
    tree=ast.parse(path.read_text())
    fn=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='TrainLoop'
            for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_loop')
    fn=copy.deepcopy(fn);loops=[n for n in ast.walk(fn) if isinstance(n,ast.While)]
    if len(loops)!=1:raise ValueError('Author loop changed; re-audit stop-only adapter')
    loops[0].test=ast.BoolOp(op=ast.And(),values=[loops[0].test,ast.parse('self.step < self.independent_update_cap',mode='eval').body])
    ast.fix_missing_locations(fn);env=dict(loop_class.run_loop.__globals__)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    class Bounded(loop_class):
        independent_update_cap=max_updates
        run_loop=env['run_loop']
    Bounded.compatibility_evidence=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        patch='Original while predicate AND self.step < independent_update_cap; LR budget unchanged')
    return Bounded
