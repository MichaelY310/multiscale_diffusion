#!/usr/bin/env python3
"""Register the out-of-tree RDD trainer, then invoke upstream TRELLIS.2 train.py."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = Path(os.environ.get("TRELLIS2_ROOT", ROOT / "third_party" / "TRELLIS.2"))
if not (TRELLIS_ROOT / "train.py").is_file():
    raise SystemExit(
        f"TRELLIS.2 not found at {TRELLIS_ROOT}; run scripts/rdd3d/setup_trellis2_pilot.sh"
    )
sys.path.insert(0, str(TRELLIS_ROOT))
sys.path.insert(0, str(ROOT / "RDD_DiT_3D"))

from trellis2 import trainers  # noqa: E402
from trellis2_rdd.rdd_sparse_flow_matching import (  # noqa: E402
    ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer,
)

trainers.ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer = (
    ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer
)
runpy.run_path(str(TRELLIS_ROOT / "train.py"), run_name="__main__")
