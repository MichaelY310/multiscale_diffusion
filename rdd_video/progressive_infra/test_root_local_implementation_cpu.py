"""No CUDA initialization: indexing, arithmetic emulation and host guards.

Does not compile/execute Triton. Passing does NOT verify a GPU binary.
"""
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch
from root_local_plan import AttentionPlan,KERNEL_CONFIGS,spatial_pool_projected
from root_local_attention_reference import joint_dense


def bf16_stream(q,k,v,rk,rv,mapping,local,bias,chunk):
    """Emulate kernel exp2 and BF16 probabilities with FP32 accumulators."""
    result=torch.empty_like(q)
    for start in range(0,q.shape[2],local):
        ix=mapping[start:start+local]
        query=q[:,:,ix].float()
        maximum=torch.full((*query.shape[:-1],1),-float('inf'))
        total=torch.zeros_like(maximum)
        acc=torch.zeros_like(query)
        for keys,values,offset in ((k[:,:,ix],v[:,:,ix],0.),(rk,rv,bias)):
            for j in range(0,keys.shape[2],chunk):
                logits=(query@keys[:,:,j:j+chunk].float().transpose(-1,-2)/math.sqrt(q.shape[-1])+offset)*math.log2(math.e)
                new=torch.maximum(maximum,logits.amax(-1,keepdim=True))
                p=torch.exp2(logits-new)
                correction=torch.exp2(maximum-new)
                total=total*correction+p.sum(-1,keepdim=True)
                acc=acc*correction+p.to(torch.bfloat16).float()@values[:,:,j:j+chunk].float()
                maximum=new
        result[:,:,ix]=(acc/total).to(torch.bfloat16)
    return result


def load_host_without_triton():
    """Stub ONLY decorator/types, never invoke kernel; CPU guard tests."""
    triton=types.ModuleType('triton');language=types.ModuleType('triton.language')
    language.constexpr=object
    triton.language=language
    def forbidden(*args,**kwargs):raise AssertionError('No GPU/kernel launch allowed in CPU tests')
    class NeverLaunch:
        def __getitem__(self,key):return forbidden
    triton.jit=lambda fn:NeverLaunch()
    path=Path(__file__).with_name('root_local_triton.py')
    spec=importlib.util.spec_from_file_location('root_local_host_test',path)
    mod=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'triton':triton,'triton.language':language}):spec.loader.exec_module(mod)
    return mod


class Checks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(937)

    def test_actual_plans(self):
        for shape,win,r,local,keep in (((8,28,52),4,728,128,37),((8,28,52),4,182,128,37),
                                     ((16,48,80),8,3840,1024,192),((16,48,80),8,960,1024,192)):
            p=AttentionPlan(shape,win,r)
            self.assertEqual(p.local,local);self.assertEqual(p.keep,keep)
            for mapping in (p.window_map(),p.cube_map()):
                torch.testing.assert_close(mapping.sort().values,torch.arange(p.tokens),rtol=0,atol=0)
            self.assertLess(p.bias,0)

    def test_window_geometry_and_query_tails(self):
        p=AttentionPlan((8,8,12),4,48)
        mapping=p.window_map();t,h,w=p.shape
        for ix in mapping.split(p.local):
            tt=ix//(h*w);yy=ix//w%h;xx=ix%w
            self.assertEqual(len(tt.unique()),t)
            self.assertEqual(len((yy//4).unique()),1)
            self.assertEqual(len((xx//4).unique()),1)
        # Kernel L=48 with Q64/Q128: padding must never write another window.
        m=torch.randperm(192)
        for block in (32,64,128):
            written=[]
            for window in range(4):
                for first in range(0,48,block):
                    rows=torch.arange(first,first+block);rows=rows[rows<48]
                    written.extend(m[window*48+rows].tolist())
            self.assertEqual(sorted(written),list(range(192)))

    def test_cube_layout_native(self):
        p=AttentionPlan((8,8,12),4,48)
        cube=p.cube_map().reshape(-1,64)
        for ix in cube:
            z=ix//96;y=ix//12%8;x=ix%12
            self.assertEqual(len((z//4).unique()),1)
            self.assertEqual(len((y//4).unique()),1)
            self.assertEqual(len((x//4).unique()),1)
        torch.testing.assert_close(cube[0],torch.tensor([z*96+y*12+x for z in range(4) for y in range(4) for x in range(4)]))

    def test_current_pool_order_and_mean(self):
        shape=(4,8,12)
        x=torch.arange(math.prod(shape)*3,dtype=torch.float32).reshape(1,1,-1,3)
        pooled=spatial_pool_projected(x,shape)
        original=x.reshape(4,8,12,3)
        expected=torch.stack([original[t,y:y+4,z:z+4].mean((0,1)) for t in range(4) for y in range(0,8,4) for z in range(0,12,4)])
        torch.testing.assert_close(pooled[0,0],expected,rtol=0,atol=0)

    def test_bf16_math_and_bank_tails(self):
        mapping=torch.randperm(192);tiles=tuple(mapping.split(48))
        for d in (64,128):
            xs=[torch.randn(2,2,192,d).to(torch.bfloat16) for _ in range(3)]
            roots=[torch.randn(2,2,91,d).to(torch.bfloat16) for _ in range(2)]
            expected=joint_dense(*[x.double() for x in xs+roots],tiles,bias=-.7)
            for chunk in (32,64,128):
                actual=bf16_stream(*xs,*roots,mapping,48,-.7,chunk).double()
                rel=float((actual-expected).norm()/expected.norm())
                self.assertLess(rel,.009)
                self.assertTrue(torch.isfinite(actual).all())

    def test_host_guards_no_launch(self):
        host=load_host_without_triton()
        q=torch.zeros(1,1,64,64,dtype=torch.bfloat16)
        mapping=torch.arange(64)
        for local in (0,-1,3,64):
            with self.assertRaises(ValueError):host.root_local_attention(q,q,q,q,q,mapping,local)
        with self.assertRaises(ValueError):AttentionPlan((8,28,52),8,728)
        with self.assertRaises(ValueError):AttentionPlan((8,28,52),4,12)

    def test_benchmark_cpu_help_and_baseline_ast(self):
        # Importing the runner must not initialize CUDA or require local Triton.
        spec=importlib.util.spec_from_file_location('root_local_runner_test',Path(__file__).with_name('benchmark_root_local_gpu.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.object(torch.cuda,'set_device',side_effect=AssertionError('CUDA initialization forbidden')):
            spec.loader.exec_module(module)
        tree=ast.parse(Path(__file__).with_name('benchmark_root_local_gpu.py').read_text())
        vsa=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='vsa')
        calls=[n.func.id for n in ast.walk(vsa) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(calls.count('video_sparse_attn'),1)
        self.assertNotIn('native_coarse',calls) # Do not accidentally double baseline coarse cost.
        self.assertEqual(len(KERNEL_CONFIGS),4)

    def test_dryrun_and_paired_statistics(self):
        from benchmark_root_local_gpu import dry_run,paired_reduction
        with patch.object(torch.cuda,'set_device',side_effect=AssertionError('No CUDA')):
            report=dry_run()
        self.assertFalse(report['gpu_run']);self.assertEqual(len(report['cases']),4)
        records=[dict(group=g,variant=v,ms=ms) for g in range(5) for v,ms in (('vsa20',10.),('root_local',8.))]
        result=paired_reduction(records,'vsa20')
        self.assertAlmostEqual(result['mean_percent'],20.)
        self.assertAlmostEqual(result['lower95_percent'],20.)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path('eval_results/progressive_infra_20261003');out.mkdir(parents=True,exist_ok=True)
    report=dict(tests=result.testsRun,passed=result.wasSuccessful(),gpu_run=False,triton_compiled=False,
        scope='CPU mapping/FP32 accumulator + BF16 probability emulation/host guards; not GPU binary verification',
        cases=[dict(shape=s,window=w,root=r) for s,w,r in (((8,28,52),4,728),((8,28,52),4,182),((16,48,80),8,3840),((16,48,80),8,960))],
        deferred=['First GPU kernel correctness/compilation','Real speed','Full Wan/RoPE integration','SP2/broadcast','Backward/finetune/quality'])
    (out/'root_local_implementation_cpu_checks.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
