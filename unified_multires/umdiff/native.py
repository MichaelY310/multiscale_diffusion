"""Execute pinned author sources without running their download/CLI entrypoints.

AST extraction retains selected function/class bodies verbatim. This is useful
when a module imports unrelated distributed training or pretrained encoders.
Every extraction is recorded in the conformance evidence.
"""
import ast,hashlib,importlib.util,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUDIT=[]

def method(relative,name,namespace=None):
    path=ROOT/'sources'/relative;raw=path.read_bytes();tree=ast.parse(raw,filename=str(path))
    matches=[node for cls in tree.body if isinstance(cls,ast.ClassDef) for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(matches)!=1:raise ValueError(f'Expected exactly one native method {name}')
    env={'__name__':'native_oracle','__file__':str(path)};env.update(namespace or {})
    exec(compile(ast.Module(body=matches,type_ignores=[]),str(path),'exec'),env)
    AUDIT.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'symbols':[name],'execution':'verbatim class method body, supplied dependency namespace'})
    return env[name]

def extract(relative,names,namespace=None):
    path=ROOT/'sources'/relative
    raw=path.read_bytes();tree=ast.parse(raw,filename=str(path))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names]
    found={node.name for node in selected}
    if found!=set(names):raise ValueError(f'Native symbols absent: {set(names)-found}')
    module=ast.Module(body=selected,type_ignores=[])
    env={'__name__':'native_oracle','__file__':str(path)};env.update(namespace or {})
    exec(compile(module,str(path),'exec'),env)
    AUDIT.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'symbols':list(names),'execution':'verbatim AST bodies, supplied dependency namespace'})
    return env

def load(relative,name=None):
    path=ROOT/'sources'/relative;name=name or 'native_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    AUDIT.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'execution':'full module import'})
    return module

def statements(relative,start_line,end_line,namespace):
    """Run unchanged statement bodies from an author's training loop.

    This does not rewrite equations or optimizer calls. Dependencies and local
    fixture variables must be supplied explicitly. Hash + line span are logged.
    """
    path=ROOT/'sources'/relative;raw=path.read_bytes();tree=ast.parse(raw,filename=str(path));selected=[]
    def visit(node):
        if isinstance(node,ast.stmt) and start_line<=node.lineno<=end_line and node.end_lineno>end_line:
            raise ValueError(f'Native statement is cut by requested span: {node.lineno}..{node.end_lineno}')
        if isinstance(node,ast.stmt) and node.lineno>=start_line and node.end_lineno<=end_line:
            selected.append(node);return
        for child in ast.iter_child_nodes(node):visit(child)
    visit(tree)
    if not selected:raise ValueError('Native statement span is empty')
    env=dict(namespace);exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
    AUDIT.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'line_span':[start_line,end_line],'execution':'unchanged training-loop statement bodies'})
    return env
