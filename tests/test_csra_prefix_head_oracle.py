import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from probe_csra_prefix_output_heads import mix_heads


class PrefixHeadOracleTests(unittest.TestCase):
    def test_both_hybrids_use_correct_channels(self):
        student=torch.randn(2,8,4,4);teacher=torch.randn_like(student)
        a=mix_heads(student,teacher,'student_main_teacher_var')
        b=mix_heads(student,teacher,'teacher_main_student_var')
        torch.testing.assert_close(a[:,:4],student[:,:4],atol=0,rtol=0)
        torch.testing.assert_close(a[:,4:],teacher[:,4:],atol=0,rtol=0)
        torch.testing.assert_close(b[:,:4],teacher[:,:4],atol=0,rtol=0)
        torch.testing.assert_close(b[:,4:],student[:,4:],atol=0,rtol=0)

    def test_bad_shapes_or_modes_rejected(self):
        x=torch.zeros(1,8,4,4)
        with self.assertRaises(ValueError): mix_heads(x,x,'unknown')
        with self.assertRaises(ValueError): mix_heads(x,x[:,:4],'student_main_teacher_var')
        with self.assertRaises(ValueError): mix_heads(x[:,:4],x[:,:4],'teacher_main_student_var')


if __name__=='__main__': unittest.main()
