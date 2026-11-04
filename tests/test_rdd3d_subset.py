from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rdd3d" / "create_objaversexl_subset.py"


class ObjaverseSubsetTest(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        path = root / "source.csv"
        pd.DataFrame(
            {
                "sha256": [f"{index:064x}" for index in range(20)],
                "file_identifier": [f"https://example.test/{index}" for index in range(20)],
                "aesthetic_score": [4.0 + index / 10 for index in range(20)],
                "captions": [""] * 20,
            }
        ).to_csv(path, index=False)
        return path

    def _create(self, source: Path, output: Path, split: str, exclude: Path | None = None):
        command = [
            sys.executable,
            str(SCRIPT),
            "--source-csv",
            str(source),
            "--output-root",
            str(output),
            "--count",
            "4",
            "--seed",
            "7",
            "--split",
            split,
            "--min-aesthetic-score",
            "5.0",
        ]
        if exclude is not None:
            command += ["--exclude", str(exclude)]
        subprocess.run(command, check=True, capture_output=True, text=True)

    def test_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            train_a, train_b, validation = root / "train_a", root / "train_b", root / "val"
            self._create(source, train_a, "train")
            self._create(source, train_b, "train")
            self._create(source, validation, "validation", train_a / "metadata.csv")

            a = pd.read_csv(train_a / "metadata.csv")
            b = pd.read_csv(train_b / "metadata.csv")
            val = pd.read_csv(validation / "metadata.csv")
            self.assertEqual(a["sha256"].tolist(), b["sha256"].tolist())
            self.assertFalse(set(a["sha256"]) & set(val["sha256"]))
            self.assertTrue((a["aesthetic_score"] >= 5.0).all())
            manifest = json.loads(
                (train_a / "rdd3d_subset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["count"], 4)
            self.assertEqual(manifest["split"], "train")


if __name__ == "__main__":
    unittest.main()
