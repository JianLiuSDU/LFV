#!/usr/bin/env python3
"""Render an Open3D snapshot in a child process (headless crashes are isolated)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--width", type=int, default=960); p.add_argument("--height", type=int, default=720); a = p.parse_args()
    import open3d as o3d
    data = np.load(a.input, allow_pickle=False); points = data["points"]; heat = data["heat"]; cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); color = np.zeros((len(points),3)); color[:,0]=heat; color[:,2]=1-heat; cloud.colors=o3d.utility.Vector3dVector(color); geometries=[cloud]
    if "grasp" in data:
        frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=.06); frame.transform(data["grasp"]); geometries.append(frame)
    if "trajectory" in data and len(data["trajectory"])>1:
        traj=data["trajectory"]; lines=np.stack((np.arange(len(traj)-1),np.arange(1,len(traj))),-1); line=o3d.geometry.LineSet(o3d.utility.Vector3dVector(traj[:, :3, 3]),o3d.utility.Vector2iVector(lines)); line.colors=o3d.utility.Vector3dVector(np.tile([[1.,.55,0.]],(len(lines),1))); geometries.append(line)
    vis=o3d.visualization.Visualizer(); vis.create_window(visible=False,width=a.width,height=a.height)
    for g in geometries: vis.add_geometry(g)
    vis.poll_events(); vis.update_renderer(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); vis.capture_screen_image(a.output,do_render=True); vis.destroy_window(); return 0
if __name__ == "__main__": raise SystemExit(main())
