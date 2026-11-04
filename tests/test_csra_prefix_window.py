import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from sample_csra_prefix_preview import PrefixRollout


class PrefixWindowTests(unittest.TestCase):
    def test_exact_window_and_original_remainder(self):
        class Student:
            coarse_source='zero'
            def __call__(self,b): return b['x']+2
        wrapper=PrefixRollout(SimpleNamespace(),Student(),reuse_fine_steps=2)
        x=torch.zeros(1,4,32,32); y=torch.tensor([1])
        with patch('sample_csra_prefix_preview.manual_forward',return_value=(x+1,{})):
            for i in range(4):
                result=wrapper.forward(x,torch.tensor([299-i]),y)
                self.assertEqual(float(result.mean()),2. if i<2 else 1.)
        self.assertEqual(wrapper.fine_calls,4)
        self.assertEqual(wrapper.reused_fine_calls,2)

    def test_default_preserves_90(self):
        self.assertEqual(PrefixRollout(None).reuse_fine_steps,90)
        with self.assertRaises(ValueError): PrefixRollout(None,reuse_fine_steps=91)
        with self.assertRaises(ValueError): PrefixRollout(None,reuse_fine_steps=0)


if __name__=='__main__': unittest.main()
