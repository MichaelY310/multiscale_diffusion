import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest
from umdiff import locations
from umdiff.cli import choose_device

def test_monorepo_and_historical_source_paths():
    root=locations.rdd_root()
    assert root==locations.UNIFIED_ROOT.parent
    assert locations.workspace_path('multiscale_diffusion/DiT/models.py')==root/'DiT/models.py'
    assert locations.workspace_path('unified_multires/umdiff/cli.py')==locations.UNIFIED_ROOT/'umdiff/cli.py'
    with pytest.raises(ValueError):locations.workspace_path('../escape')

def test_explicit_root_override_fails_closed(monkeypatch,tmp_path):
    monkeypatch.setenv('UMDIFF_RDD_ROOT',str(tmp_path))
    with pytest.raises(FileNotFoundError):locations.rdd_root()

def test_cpu_policy_does_not_probe_or_initialize_cuda(monkeypatch):
    monkeypatch.setenv('UMDIFF_CPU_ONLY','1')
    with patch('torch.cuda.is_available',side_effect=AssertionError('Unexpected CUDA probe')):
        assert str(choose_device('cpu'))=='cpu'
        with pytest.raises(RuntimeError,match='CPU-only'):choose_device('cuda')

def test_source_fetch_is_pinned_and_does_not_overwrite(tmp_path):
    file=locations.UNIFIED_ROOT/'tools/fetch_sources.py'
    spec=importlib.util.spec_from_file_location('fetch_sources_test',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    body=b'# source fixture\n';digest=hashlib.sha256(body).hexdigest()
    entry={'repo':'https://github.com/example/project','commit':'a'*40,'selected_files':{'model.py':digest}}
    path=tmp_path/'fixture/model.py';path.parent.mkdir();path.write_bytes(body)
    with patch.object(module.urllib.request,'urlopen',side_effect=AssertionError('Network disabled')):
        assert module.synchronize('fixture',entry,tmp_path,verify=True)['verified']==1
        path.write_text('local modification')
        with pytest.raises(ValueError,match='existing file differs'):module.synchronize('fixture',entry,tmp_path)
    for bad in ('../outside.py','weights.pt','/absolute.py'):
        with pytest.raises(ValueError):module.validate_entry('fixture',{**entry,'selected_files':{bad:digest}},tmp_path)
