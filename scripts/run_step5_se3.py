#!/usr/bin/env python
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.pipeline import se3_trajectory
from lfv.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs/pipeline/picknplace.yaml")
    se3_trajectory.run(cfg)
