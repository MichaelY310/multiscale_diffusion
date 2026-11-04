"""CPU CLI regression for paths crossing the pinned worker working directory."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def main() -> None:
    work=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('public_experiment_cli',work/'rdd_experiments.py')
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checks={}
    with tempfile.TemporaryDirectory(prefix='rdd-cli-contract-') as temporary:
        for method in ['controlled_rdd','native_rdd','native_wan','native_pyramid_flow','paper_pyramidal_wan']:
            captured=[]
            checkpoint='gpu_checks_20261006/relative_test_checkpoint'
            argv=['rdd_experiments.py','sample',method,'--checkpoint',checkpoint,
                  '--data','fair_data_20261006/vbench_full_pf_prompts' if method=='native_pyramid_flow' else 'fair_data_20261006/vbench_full_wan_prompts',
                  '--full-info','paper_sources_20261006/VBench_full_info.json','--output',str(Path(temporary)/method)]
            with patch.object(sys,'argv',argv),patch.object(module.subprocess,'run',side_effect=lambda command,**kw:captured.append(command)),contextlib.redirect_stdout(io.StringIO()):
                module.main()
            command=captured[0]
            assert command[command.index('--checkpoint')+1]==str((work/checkpoint).resolve())
            flag='--validation' if method in ('controlled_rdd','native_rdd') else '--prompts'
            assert Path(command[command.index(flag)+1]).is_absolute()
            checks[method+'_relative_paths_resolved']=True
        captured=[]
        argv=['rdd_experiments.py','score','--data','gpu_checks_20261006/vbench_native_full_resolution_v2',
              '--full-info','paper_sources_20261006/VBench_integration_subset.json','--allow-subset',
              '--output',str(Path(temporary)/'score')]
        with patch.object(sys,'argv',argv),patch.object(module.subprocess,'run',side_effect=lambda command,**kw:captured.append(command)),contextlib.redirect_stdout(io.StringIO()):
            module.main()
        assert captured[0][captured[0].index('--full-info')+1]==str(work/'paper_sources_20261006/VBench_integration_subset.json')
        checks['score_full_info_relative_path_resolved']=True
        for name,extra in {'duplicate_seeds':['--seeds','0,0'], 'zero_batch':['--batch-size','0'],
                           'zero_nfe':['--nfe','0'],'wrong_native_pf_shape':['--height','448']}.items():
            argv=['rdd_experiments.py','sample','native_pyramid_flow','--output',str(Path(temporary)/name),*extra]
            with patch.object(sys,'argv',argv),patch.object(module.subprocess,'run',side_effect=AssertionError('GPU dispatch must not occur')),contextlib.redirect_stderr(io.StringIO()):
                try:module.main()
                except SystemExit as error:assert error.code==2
                else:raise AssertionError('Invalid CLI accepted: '+name)
            checks[name+'_rejected_before_gpu']=True
    (work/'PUBLIC_CLI_CPU_CONTRACT.json').write_text(json.dumps(dict(status='passed',checks=checks,
        gpu_dispatched=False,scope='real CLI with only GPU dispatch mocked; file/config/path validation executes normally'),indent=2))


if __name__=='__main__':
    main()
