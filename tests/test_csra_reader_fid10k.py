import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from sample_csra_reader_fid10k import batches,protocol,ARMS,assemble,sha,write_json,check_meta
from sample_csra_prefix_fid_pilot import batches as old_batches


class ReaderFidTests(unittest.TestCase):
    def test_counts_tail_seed_disjointness(self):
        seeds=set();old=set();smoke=set()
        for rank in range(4):
            plan=batches(rank)
            self.assertEqual(sum(n for _,n,_ in plan),2500)
            self.assertEqual(len(plan),40)
            self.assertEqual(plan[-1][1],4)
            seeds.update(s for _,_,s in plan)
            old.update(s for _,_,s in old_batches(rank))
            smoke.update(s for _,_,s in batches(rank,True))
        self.assertEqual(len(seeds),160)
        self.assertFalse(seeds&old);self.assertFalse(seeds&smoke)

    def test_protocol_fixed_without_refiner(self):
        p=protocol(False)
        self.assertEqual(ARMS,('original_rdd','reader'))
        self.assertEqual(p['count'],10000);self.assertEqual(protocol(True)['count'],16)
        self.assertFalse(p['memory_refiner']);self.assertFalse(p['neighbor'])
        self.assertEqual(p['reader_additional_updates'],400)
        self.assertEqual(len(p['executed_fine_blocks']),18)
        self.assertEqual(p['steps'],[80,80,90]);self.assertEqual(p['cfg'],1.)
        with self.assertRaises(ValueError):batches(4)

    def test_npz_assembly_resume_and_corruption_guard(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provenance={'protocol':protocol(True),'sampler_sha256':'unit-test'}
            for rank in range(4):
                shard=root/f'shard_{rank}';shard.mkdir()
                write_json(shard/'complete.json',dict(complete=True,**provenance))
                for arm in ARMS:
                    (shard/arm).mkdir()
                    i,n,seed=batches(rank,True)[0]
                    dest=shard/arm/f'chunk_{i:03d}.npz'
                    labels=list(range(rank*n,(rank+1)*n))
                    np.savez_compressed(dest,arr_0=np.full((n,256,256,3),rank,dtype=np.uint8),arr_1=labels)
                    write_json(dest.with_suffix('.json'),dict(complete=True,index=i,count=n,seed=seed,
                        provenance=provenance,labels=labels,endpoints={'8':'paired','16':'paired'},sha256=sha(dest)))
            with patch('sample_csra_reader_fid10k.shutil.disk_usage',return_value=SimpleNamespace(free=100*2**30)):
                args=SimpleNamespace(root=root,smoke=True)
                assemble(args);assemble(args)
            for arm in ARMS:
                with np.load(root/f'{arm}.npz',allow_pickle=False) as data:
                    self.assertEqual(data['arr_0'].shape,(16,256,256,3))
                    self.assertEqual(data['arr_1'].tolist(),list(range(16)))
            dest=root/'shard_0/reader/chunk_000.npz'
            meta=json.loads(dest.with_suffix('.json').read_text())
            meta['count']=999
            with self.assertRaises(ValueError):check_meta(meta,dest,*batches(0,True)[0],provenance)


if __name__=='__main__':unittest.main()
