import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from diagnose_csra_prefix6_error_budget import local_jump_error
from diagnose_csra_prefix_rollout_shift import StateProbe, TIMES


class ErrorBudgetTests(unittest.TestCase):
    def test_default_times_unchanged(self):
        self.assertEqual(StateProbe(None,None,'teacher').times,set(TIMES))

    def test_explicit_times_are_private(self):
        times=[3,2,1,0];p=StateProbe(None,None,'student',times)
        times.append(99)
        self.assertEqual(p.times,{0,1,2,3})
        self.assertEqual(StateProbe(None,None,'teacher').times,set(TIMES))

    def test_squared_band_response(self):
        self.assertEqual(local_jump_error({'low_mse':2.,'high_mse':3.},
                         {'low_gain':-2.,'high_gain':1.}),11.)
        self.assertEqual(local_jump_error({'low_mse':2.,'high_mse':3.},
                         {'low_gain':0.,'high_gain':0.}),0.)


if __name__=='__main__':unittest.main()
