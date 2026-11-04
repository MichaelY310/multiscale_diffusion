import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from diagnose_csra_prefix_all_times import bracket_reference


class TimeInterpolationProbeTests(unittest.TestCase):
    def test_measured_anchors_and_intervals(self):
        anchors={0:2.,40:6.,84:8.}
        self.assertEqual(bracket_reference(40,anchors),6.)
        self.assertEqual(bracket_reference(20,anchors),4.)
        self.assertEqual(bracket_reference(62,anchors),7.)

    def test_extrapolation_rejected(self):
        for t in (-1,85):
            with self.assertRaises(ValueError): bracket_reference(t,{0:2.,84:8.})


if __name__=='__main__': unittest.main()
