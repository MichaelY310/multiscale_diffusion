import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from csra_state_replay_data import PROTOCOL, PARENT_SHA, BASE_SHA, TIMES, role_for, check_record


class ReplayTests(unittest.TestCase):
    def record(self):
        def tensors():
            return dict(x=torch.zeros(8, 2, 4, 32, 32), target=torch.zeros(8, 2, 8, 32, 32),
                        t=torch.tensor(TIMES)[:, None].expand(-1, 2))
        return dict(protocol=PROTOCOL, parent_sha256=PARENT_SHA, base_sha256=BASE_SHA,
                    split='train', seed=99, fine_calls=90, rng_unchanged=True, coarse_endpoint_max_abs=0.,
                    labels=torch.tensor([1,2]), coarse=torch.zeros(2,64,1024),
                    teacher=tensors(), student=tensors())

    def test_exact_half_and_epoch_swap(self):
        for epoch in range(3):
            self.assertEqual(sum(role_for(i,epoch,'mixed50')=='student' for i in range(128)),64)
        self.assertTrue(all(role_for(i,0,'teacher_only')=='teacher' for i in range(128)))
        self.assertTrue(all(role_for(i,0,'mixed50')!=role_for(i,1,'mixed50') for i in range(128)))
        with self.assertRaises(ValueError): role_for(1,1,'bogus')

    def test_record_contract(self):
        d=self.record(); check_record(d,'train',99,2)
        d['parent_sha256']='wrong'
        with self.assertRaises(ValueError): check_record(d,'train',99,2)

    def test_reject_mismatched_first_state_or_nonfinite(self):
        d=self.record();d['student']['x'][0,0,0,0,0]=1
        with self.assertRaises(ValueError): check_record(d,'train',99,2)
        d=self.record();d['student']['target'][3,0,0,0,0]=float('nan')
        with self.assertRaises(ValueError): check_record(d,'train',99,2)


if __name__=='__main__': unittest.main()
