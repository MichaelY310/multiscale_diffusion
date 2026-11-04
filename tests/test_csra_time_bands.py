import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from probe_csra_time_source_advantage import band_errors


class TimeBandTests(unittest.TestCase):
    def test_power_decomposition_and_constant_error(self):
        torch.manual_seed(12)
        x=torch.randn(3,8,8,8)
        rows=band_errors(x,torch.zeros_like(x))
        torch.testing.assert_close(rows['low_error_mse']+rows['high_error_mse'],rows['main_mse'])
        constant=band_errors(torch.ones_like(x),torch.zeros_like(x))
        torch.testing.assert_close(constant['high_error_mse'],torch.zeros(3),atol=0,rtol=0)

    def test_odd_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            band_errors(torch.zeros(1,8,7,8),torch.zeros(1,8,7,8))


if __name__=='__main__':
    unittest.main()
