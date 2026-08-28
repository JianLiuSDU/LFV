#!/usr/bin/env python3
"""Reproducible real-episode smoke/benchmark runner for the camera pipeline."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from lfv.deployment.episode_reader import read_episode_frame
from lfv.deployment.camera_plan_pipeline import CameraToPlanPipeline
from lfv.perception.backends import PrecomputedMaskBackend
from lfv.deployment.transfer_backend import PrecomputedHeatBackend
from lfv.geometry.sam3d_completion import VisibleOnlyCompletionBackend
from lfv.deployment.grasp_backend import NPZGraspBackend
from lfv.deployment.model_backend import LegacyPouringBackend

def prepare(episode: Path, mode: str, frame: int, input_dir: Path) -> None:
    item = read_episode_frame(episode, frame); input_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(item.camera.rgb).save(input_dir/'rgb.png'); np.save(input_dir/'depth.npy', item.camera.depth_m); (input_dir/'intrinsics.json').write_text(json.dumps({'intrinsic_cv':item.camera.intrinsic_cv.tolist(),'depth_scale':1.0,'frame_index':frame},indent=2))
    cup = episode/'sam_mask'/'affordance_mask.npy'; bowl = episode/'target_sam_mask'/'target_mask.npy'
    if not cup.exists() or not bowl.exists(): raise FileNotFoundError('Existing episode SAM masks are required for this reproducible test')
    shutil.copy2(cup, input_dir/'cup_mask.npy'); shutil.copy2(bowl, input_dir/'bowl_mask.npy')
    if mode == 'stage1':
        heat = np.load(episode/'contact_heatmap'/'contact_heatmap.npz',allow_pickle=False)['heatmap_2d']
        grasp = np.load(episode/'graspnet_contact_roi_verify'/'filtered_candidates.npy',allow_pickle=False)
        np.savez_compressed(input_dir/'grasps.npz',grasps=grasp)
    else:
        heat = np.load(cup,allow_pickle=False).astype(np.float32)
    np.savez_compressed(input_dir/'target_heatmap.npz',heatmap=np.asarray(heat,dtype=np.float32))

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['stage1','stage2'],required=True); p.add_argument('--episode',required=True); p.add_argument('--frame',type=int,default=0); p.add_argument('--output',required=True); p.add_argument('--python-executable',default='python'); p.add_argument('--device',default='cpu'); p.add_argument('--run-model',action='store_true'); a=p.parse_args(); episode=Path(a.episode).expanduser().resolve(); out=Path(a.output).expanduser().resolve(); inp=out/'input'; prepare(episode,a.mode,a.frame,inp)
    if a.mode == 'stage1':
        pipeline=CameraToPlanPipeline(perception=PrecomputedMaskBackend(),transfer=PrecomputedHeatBackend(inp/'target_heatmap.npz'),completion=VisibleOnlyCompletionBackend(),grasp=NPZGraspBackend(inp/'grasps.npz'))
    else:
        motion=None
        if a.run_model:
            motion=LegacyPouringBackend(model_repo='/home/users1/ljian/object_centric_diffusion',goal_checkpoint='/home/users1/ljian/object_centric_diffusion/data/outputs_local_goal_pose/pouring_seed42/20260428_171300/checkpoints/epoch=0700-val_sample_goal_pos_err_cm=3.086.ckpt',trajectory_checkpoint='/home/users1/ljian/object_centric_diffusion/data/outputs_goal_full64/pouring_seed42/20260429_210924/checkpoints/epoch=1500.ckpt',language_embedding='/media/ljian/lj/data_3d/pouring/lang_emb.npy',python_executable=a.python_executable,device=a.device)
        pipeline=CameraToPlanPipeline(perception=PrecomputedMaskBackend(),transfer=PrecomputedHeatBackend(inp/'target_heatmap.npz'),completion=VisibleOnlyCompletionBackend(),motion=motion,allow_fallback_grasp=True)
    result=pipeline.run(inp,out/'result'); print(json.dumps(result.report,indent=2,ensure_ascii=False,default=str)); return 0
if __name__ == '__main__': raise SystemExit(main())
