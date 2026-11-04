"""Regression: immutable dataset and new model must use DISTINCT protocols."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
import train_csra_memory_refiner as pilot


class RefinerProtocolTests(unittest.TestCase):
    def test_old_dataset_keeps_old_protocol_after_factory_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i in range(4):
                folder=root/'data'/f'shard_{i}';folder.mkdir(parents=True)
                (folder/'record.pt').write_bytes(b'unit-test-bytes')
                (folder/'manifest.json').write_text(json.dumps(dict(complete=True,smoke=False,
                    protocol=pilot.DATA_PROTOCOL,files=[dict(name='record.pt',sha256=pilot.file_sha(folder/'record.pt'))])))
            # Rebinding the training format must not reinterpret the dataset.
            with patch.object(pilot,'DATA_ROOT',root), patch.object(pilot.shared,'PROTOCOL',pilot.PROTOCOL):
                self.assertNotEqual(pilot.PROTOCOL,pilot.DATA_PROTOCOL)
                self.assertEqual(len(pilot.dataset_hashes()),4)
                path=root/'data/shard_0/manifest.json'
                d=json.loads(path.read_text());d['protocol']=pilot.PROTOCOL
                path.write_text(json.dumps(d))
                with self.assertRaises(ValueError): pilot.dataset_hashes()


if __name__=='__main__': unittest.main()
