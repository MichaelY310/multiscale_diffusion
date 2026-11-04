"""Offline CUDA compile and CPU RNG KATs only; no CUDA context or launch."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from test_stage_birth_cpu import KATS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ccbin')
    p.add_argument('--output',default='eval_results/progressive_infra_20261003')
    args=p.parse_args()
    nvcc=shutil.which('nvcc')
    if not nvcc: raise SystemExit('nvcc not found; no compilation claim')
    source=Path(__file__).resolve().parent
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    base=[nvcc,'--std=c++17']
    if args.ccbin: base+=['-ccbin',args.ccbin]
    def run(cmd):
        r=subprocess.run(cmd,capture_output=True,text=True)
        print(r.stdout,end=''); print(r.stderr,end='',file=sys.stderr)
        if r.returncode: raise SystemExit(r.returncode)
        return r.stdout+r.stderr
    version=run([nvcc,'--version'])
    for flag,suffix in [('--ptx','.ptx'),('--cubin','.cubin')]:
        log=run(base+[flag,'-arch=sm_120a','--fmad=false','-Xptxas=-v',
                      str(source/'stage_birth_patch.cu'),'-o',str(out/('stage_birth_patch'+suffix))])
    exe=out/('stage_birth_rng_host_check.exe' if sys.platform=='win32' else 'stage_birth_rng_host_check')
    run(base+['--cudart=none',str(source/'stage_birth_rng_host_check.cpp'),'-o',str(exe)])
    results=run([str(exe)]).strip().splitlines()
    assert len(results)==len(KATS)
    for row,(_,_,answer) in zip(results,KATS):
        assert tuple(int(v,16) for v in row.split())==answer
    report=dict(compiled=True,architecture='sm_120a',nvcc=version,resources=log,
                cpu_header_rng_known_answers_passed=True,gpu_run=False,
                distributed_runtime_implemented=False,measured_speedup=None,
                limitations=['GPU float math and indexing not executed',
                             'No NCCL reshard or GPU patch-projection adapter; CPU Conv3d adapter checked separately',
                             'No real stage/SP timing profiles'])
    (out/'stage_birth_compile_checks.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
