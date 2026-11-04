import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from prune_old_dmd_objective_checkpoints import candidates


class RetentionTests(unittest.TestCase):
    def test_only_completed_numbered_files_at_or_below_latest(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            names=['latest.pt','latest.tmp','teacher.pt','step_0000001.pt','step_0000500.pt',
                   'step_0000501.pt','step_0000500.pt.tmp','step_0000400.pt']
            for name in names:
                path=folder/name;path.write_bytes(b'test');os.utime(path,(time.time()-120,)*2)
            os.utime(folder/'step_0000400.pt',None)  # too recent to prune
            got=candidates(folder,500)
            self.assertEqual([Path(x['path']).name for x in got],['step_0000001.pt','step_0000500.pt'])
            self.assertTrue(all((folder/name).exists() for name in names))  # dry plan did not delete


if __name__=='__main__':unittest.main()
