#!/usr/bin/env python3
"""Export one Zarr episode frame to the static deployment input contract."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from lfv.deployment.episode_reader import read_episode_frame

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--episode",required=True); p.add_argument("--frame",type=int,default=0); p.add_argument("--output",required=True); a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); item=read_episode_frame(a.episode,a.frame); Image.fromarray(item.camera.rgb).save(out/"rgb.png"); np.save(out/"depth.npy",item.camera.depth_m); k=item.camera.intrinsic_cv; (out/"intrinsics.json").write_text(json.dumps({"intrinsic_cv":k.tolist(),"depth_scale":1.0,"frame_index":a.frame},indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
