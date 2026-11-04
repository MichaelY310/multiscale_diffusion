"""CPU-only geometry/RNG/planner tests; synthetic schedules are NOT timings."""
import json
from pathlib import Path
import unittest
import numpy as np
from stage_birth_reference import *

KATS = [
    ((0,0,0,0), (0,0), (0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8)),
    ((MASK,)*4, (MASK,)*2, (0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd)),
    ((0x243f6a88,0x85a308d3,0x13198a2e,0x03707344), (0xa4093822,0x299f31d0),
     (0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1))]


class Checks(unittest.TestCase):
    def test_philox_known_answers(self):
        for ctr,key,expected in KATS:
            self.assertEqual(philox(ctr,key), expected)

    def test_partition_invariance(self):
        rng=np.random.default_rng(19)
        clean=rng.normal(size=(2,3,2,4,6)).astype(np.float32)
        for fine in [(2,8,12),(4,4,6),(4,8,12),(2,4,6)]:
            full=birth_patch(clean,packet_plan((2,4,6),fine,1,1,0),.4,73,2)
            for degree in (2,3,4):
                parts=[birth_patch(clean,packet_plan((2,4,6),fine,3,degree,r),.4,73,2)
                       for r in range(degree)]
                np.testing.assert_array_equal(np.concatenate(parts,axis=1),full)

    def test_repeat_and_patch_order(self):
        clean=np.arange(2*4*6,dtype=np.float32).reshape(1,1,2,4,6)
        fine=(4,8,12)
        result=birth_patch(clean,packet_plan((2,4,6),fine,1,1,0),1.,7,1)
        up=clean.repeat(2,axis=2).repeat(2,axis=3).repeat(2,axis=4)
        patches=up.reshape(1,1,4,4,2,6,2).transpose(0,2,3,5,1,4,6).reshape(1,-1,1,4)
        np.testing.assert_array_equal(result,patches)

    def test_alpha_rounding_and_bf16_storage(self):
        clean=np.full((1,1,2,4,6),1.0001,dtype=np.float32)
        plan=packet_plan((2,4,6),(4,8,12),1,1,0)
        near_one=birth_patch(clean,plan,1-1e-8,73,2)
        exactly_one=birth_patch(clean,plan,1.,73,2)
        np.testing.assert_array_equal(near_one,exactly_one)
        bf=birth_patch(clean,plan,1.,73,2,emulate_bf16=True)
        np.testing.assert_array_equal(bf,np.ones_like(bf))
        with self.assertRaises(ValueError): bf16_rne(np.array([np.finfo(np.float32).max]))

    def test_bf16_arithmetic_modes(self):
        import torch
        clean=bf16_rne(np.random.default_rng(5).normal(size=(1,2,2,4,6)).astype(np.float32))
        p=packet_plan((2,4,6),(4,8,12),1,1,0)
        up=birth_patch(clean,p,1.,73,2)
        noise=birth_patch(np.zeros_like(clean),p,0.,73,2)
        c=torch.from_numpy(up).to(torch.bfloat16)
        e=torch.from_numpy(noise).to(torch.bfloat16)
        sigma=torch.tensor(np.float32(1)-np.float32(.4),dtype=torch.bfloat16)
        expected=((1-sigma)*c+sigma*e).float().numpy()
        result=birth_patch(clean,p,.4,73,2,arithmetic='bf16_sigma')
        np.testing.assert_array_equal(result,expected)
        # Original tiny scheduler sigma survives even when FP32 alpha -> 1.
        zeros=np.zeros_like(clean)
        result=birth_patch(zeros,p,1-1e-8,73,2,arithmetic='bf16_sigma',sigma=1e-8)
        expected=(torch.tensor(1e-8,dtype=torch.bfloat16)*e).float().numpy()
        np.testing.assert_array_equal(result,expected)
        expected=(.4*c+.6*e).float().numpy()
        result=birth_patch(clean,p,.4,73,2,arithmetic='bf16_scalar')
        np.testing.assert_array_equal(result,expected)

    def test_original_projection_adapter(self):
        import torch
        rng=np.random.default_rng(12)
        clean=rng.normal(size=(2,3,2,4,6)).astype(np.float32)
        p=packet_plan((2,4,6),(4,8,12),1,1,0)
        patches=birth_patch(clean,p,.4,73,2)
        weight=rng.normal(size=(7,3,1,2,2)).astype(np.float32)
        bias=rng.normal(size=(7,)).astype(np.float32)
        b,l,c,_=patches.shape
        x=patches.reshape(b,4,4,6,c,2,2).transpose(0,4,1,2,5,3,6).reshape(b,c,4,8,12)
        expected=torch.nn.functional.conv3d(torch.from_numpy(x),torch.from_numpy(weight),
                 torch.from_numpy(bias),stride=(1,2,2)).permute(0,2,3,4,1).reshape(b,l,7).numpy()
        np.testing.assert_allclose(patch_project_reference(patches,weight,bias),expected,rtol=2e-5,atol=3e-6)

    def test_noise_streams_statistics(self):
        # Small deterministic sanity check, not a PRNG statistical battery.
        noise=np.array([normal_pair(i,74,1) for i in range(6000)]).flatten()
        self.assertLess(abs(float(noise.mean())),.04)
        self.assertLess(abs(float(noise.var())-1),.05)
        a=np.array([normal_pair(i,74,2) for i in range(6000)]).flatten()
        self.assertLess(abs(float(np.corrcoef(noise,a)[0,1])),.04)
        self.assertNotEqual(normal_pair(0,74,1),normal_pair(0,74,1,1))

    def test_source_ownership_and_remote_count(self):
        parent=(2,6,10); fine=(4,12,20)
        for r in range(4):
            p=packet_plan(parent,fine,3,4,r)
            for voxel,owner in zip(p.voxel_ids,p.source_owners):
                t=voxel//60; y=voxel//10%6; x=voxel%10
                token=(t*3+y//2)*5+x//2
                start,end=shard_range(30,3,int(owner))
                self.assertTrue(start <= token < end)
            counts=p.byte_counts(2,16)
            self.assertEqual(counts['remote_value_bytes'],2*16*2*sum(p.source_owners!=r))
            self.assertLessEqual(counts['remote_value_bytes'],counts['value_bytes'])

    def test_nonprefix_physical_rank_mapping(self):
        for r in range(4):
            p=packet_plan((2,6,10),(4,12,20),2,4,r,source_ranks=(2,3),destination_ranks=(0,1,2,3))
            owners=np.array([2,3])[p.source_owners]
            self.assertEqual(p.byte_counts(1,16)['remote_value_bytes'],32*sum(owners!=r))
        with self.assertRaises(ValueError):
            packet_plan((2,6,10),(4,12,20),2,4,0,source_ranks=(2,2))

    def test_elastic_planner_against_best_fixed(self):
        # Invented values solely test exhaustive scheduling and overhead accounting.
        times=[{1:1.,2:2.,4:3.},{1:3.,2:1.,4:2.},{1:5.,2:3.,4:1.}]
        edges={(i,a,b):(.1 if a!=b else .05) for i,a,b in product(range(2),(1,2,4),(1,2,4))}
        best=best_sp_schedule(times,edges,2.)
        self.assertEqual(best['schedule'],[1,2,4])
        self.assertEqual(best['fixed_schedule'],[2,2,2])
        self.assertAlmostEqual(best['seconds'],5.2)
        self.assertAlmostEqual(best['fixed_seconds'],8.1)
        costly={key:(20. if key[1]!=key[2] else .05) for key in edges}
        result=best_sp_schedule(times,costly,2.)
        self.assertEqual(result['schedule'],[2,2,2])
        self.assertAlmostEqual(result['reduction_percent'],0.)
        with self.assertRaises(KeyError): best_sp_schedule(times,{})

    def test_planner_startup_final_physical_groups(self):
        groups=((0,),(2,3),(0,1,2,3))
        times=[dict(zip(groups,values)) for values in ([1.,2.,3.],[3.,1.,2.],[5.,3.,1.])]
        edges={(i,a,b):(.1 if a!=b else .05) for i,a,b in product(range(2),groups,groups)}
        start={g:0. for g in groups}; start[(0,)]=5.
        final={g:0. for g in groups}; final[(0,1,2,3)]=4.
        result=best_sp_schedule(times,edges,2.,start,final)
        self.assertEqual(result['schedule'],[(2,3)]*3)
        self.assertAlmostEqual(result['reduction_percent'],0.)
        with self.assertRaises(ValueError): best_sp_schedule(times,edges,max_gpus=2)

    def test_reject_invalid_inputs(self):
        for parent,fine in [((2,3,4),(4,6,8)),((2,4,6),(2,6,6)),((2,4,6),(1,4,6))]:
            with self.assertRaises(ValueError): packet_plan(parent,fine,1,1,0)
        with self.assertRaises(ValueError): shard_range(2,4,0)
        with self.assertRaises(ValueError): normal_pair(2**64,1,0)
        p=packet_plan((2,4,6),(4,8,12),1,1,0)
        with self.assertRaises(ValueError): birth_patch(np.ones((1,1,2,4,6)),p,1.1,1,0)
        with self.assertRaises(ValueError): birth_patch(np.full((1,1,2,4,6),np.nan),p,.4,1,0)
        with np.errstate(over='ignore'):
            with self.assertRaises(ValueError): birth_patch(np.full((1,1,2,4,6),1e100),p,0.,1,0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    rows=[]
    for parent,fine,p,q in [((4,24,40),(8,48,80),1,2),((8,48,80),(16,96,160),2,4)]:
        plans=[packet_plan(parent,fine,p,q,r) for r in range(q)]
        counts=[x.byte_counts(1,16) for x in plans]
        rows.append(dict(parent=parent,fine=fine,sp=f'{p}->{q}',
                         **{key:sum(c[key] for c in counts) for key in counts[0]},
                         hidden_tensor_bytes_reference=int(np.prod(fine))*1536*2//4))
    report=dict(tests=result.testsRun,passed=result.wasSuccessful(),gpu_run=False,
                compiled_kernel_run=False,distributed_runtime_implemented=False,
                measured_speedup=None,training_required=False,
                rng_scope='Philox KATs and small CPU sanity tests; new sequence, not torch.randn; GPU math unchecked',
                arithmetic_scope='Three modes: FP32 fused; BF16 sigma coefficients/products; BF16 scalar/products. CPU latter two match torch arithmetic with supplied common noise. GPU math/new RNG trajectory still unchecked.',
                projection_adapter='CPU FP32 patch-GEMM checked against original Conv3d weights; no CUDA integration or BF16 GEMM validation',
                packet_geometry=rows,
                timing_scope='No measured SP profiles; synthetic planner tests are not timing evidence')
    report['byte_scope']='Hidden byte count is a dimensional illustration, NOT what an optimized baseline transfers. Both baselines must reshard only C=16 latent.'
    out=Path('eval_results/progressive_infra_20261003'); out.mkdir(parents=True,exist_ok=True)
    (out/'stage_birth_cpu_checks.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
