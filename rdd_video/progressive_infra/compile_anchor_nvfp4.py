"""Compile the CUDA producer and run ONLY a CPU conversion-header executable.

No CUDA context, device allocation or kernel launch is performed. This does
not validate GPU memory correctness, native GEMM integration or speed.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ccbin')
    p.add_argument('--output',default='eval_results/progressive_infra_20261003')
    args=p.parse_args()
    nvcc=shutil.which('nvcc')
    if not nvcc: raise SystemExit('nvcc not found; no compilation claim')
    source=Path(__file__).resolve().parent
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    common=[nvcc,'--std=c++17']
    if args.ccbin: common+=['-ccbin',args.ccbin]
    def run(command):
        r=subprocess.run(command,capture_output=True,text=True)
        print(r.stdout,end=''); print(r.stderr,end='',file=sys.stderr)
        if r.returncode: raise SystemExit(r.returncode)
        return r.stdout+r.stderr
    version=run([nvcc,'--version'])
    cuda=source/'anchor_nvfp4_epilogue.cu'
    ptx=output/'anchor_nvfp4_epilogue.ptx'
    run(common+['--ptx','-arch=sm_120a','--fmad=false',str(cuda),'-o',str(ptx)])
    cubin=output/'anchor_nvfp4_epilogue.cubin'
    resources=run(common+['--cubin','-arch=sm_120a','--fmad=false','-Xptxas=-v',str(cuda),'-o',str(cubin)])
    exe=output/('anchor_fp4_host_check.exe' if sys.platform=='win32' else 'anchor_fp4_host_check')
    run(common+['--cudart=none',str(source/'anchor_fp4_host_check.cpp'),'-o',str(exe)])
    conversion=run([str(exe)]).splitlines()
    expected_codes=[0,8,1,9,2,3,4,5,6,7,0,2,2,4,4,6,6,15,7]
    expected_scales=[0,1,56,126]
    assert list(map(int,conversion[0].split()))==expected_codes
    assert list(map(int,conversion[1].split()))==expected_scales
    report=dict(compiled=True,architecture='sm_120a',nvcc=version,
                cpu_cuda_header_conversions_passed=True,resources=resources,
                cuda_kernel_run=False,gemm_implemented=False,
                scale_layout='Logical row-major; requires SM120 GEMM layout adapter',
                scope='Compiler/codebook checks only. No GPU correctness or timing.')
    (output/'anchor_nvfp4_compile_checks.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
