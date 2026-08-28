#!/usr/bin/env python3
"""Run the single-view red-cup grasp fallback and oracle evaluation.

Generation consumes only ``rgb/depth/intrinsic + visible points + transferred
heatmap``.  The simulator's full point cloud is read only when available to
measure whether the virtual opposite contact is geometrically plausible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lfv.deployment.partial_grasp import build_contact_pair_hypotheses, evaluate_contact_pair_against_full_cloud
from lfv.visualization.contact_pair import save_partial_grasp_overlay


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, default=Path("/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/red_mug_a3b_seed_0/snapshot/pouring_snapshot.npz"))
    p.add_argument("--transfer", type=Path, default=Path("/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/red_mug_a3b_seed_0/transfer/transfer_result.npz"))
    p.add_argument("--output", type=Path, default=Path("/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/red_mug_a3b_seed_0/partial_grasp"))
    p.add_argument("--top-k", type=int, default=8)
    return p


def main() -> None:
    args = _parser().parse_args()
    out = args.output.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    snap = np.load(args.snapshot, allow_pickle=False)
    rgb = np.asarray(snap["rgb"], dtype=np.uint8)
    depth = np.asarray(snap["depth_m"], dtype=np.float32)
    intrinsic = np.asarray(snap["intrinsic_cv"], dtype=np.float32)
    cup_mask = np.asarray(snap["cup_mask"]).astype(bool)
    points = np.asarray(snap["visible_points_camera"], dtype=np.float32)
    uv = np.asarray(snap["visible_pixels_uv"], dtype=np.int64)
    if args.transfer.exists():
        transfer = np.load(args.transfer, allow_pickle=False)
        heatmap = np.asarray(transfer["target_heatmap"], dtype=np.float32)
        transfer_name = str(args.transfer)
    else:
        heatmap = cup_mask.astype(np.float32)
        transfer_name = "cup_mask_fallback"
    heat_at_points = heatmap[np.clip(uv[:, 1], 0, heatmap.shape[0] - 1), np.clip(uv[:, 0], 0, heatmap.shape[1] - 1)]
    hypotheses = build_contact_pair_hypotheses(points, heat_at_points, top_k=args.top_k)
    if not hypotheses:
        raise RuntimeError("No contact-pair hypothesis was generated")
    full = np.asarray(snap["full_points_camera"], dtype=np.float32) if "full_points_camera" in snap.files else None
    quality = evaluate_contact_pair_against_full_cloud(hypotheses, full) if full is not None and len(full) else []
    # Select the best candidate with a supported first endpoint, if the oracle
    # is available; deployment itself never uses this selection signal.
    selected_index = 0
    if quality:
        supported = [q for q in quality if q["first_supported"]]
        if supported:
            selected_index = int(max(supported, key=lambda q: q["score"])["rank"])
    selected = hypotheses[selected_index]
    np.savez_compressed(out / "contact_pair_hypotheses.npz", first_contact_camera=np.stack([h.first_contact_camera for h in hypotheses]), second_contact_camera=np.stack([h.second_contact_camera for h in hypotheses]), tcp_camera=np.stack([h.tcp_camera for h in hypotheses]), closing_axis_camera=np.stack([h.closing_axis_camera for h in hypotheses]), approach_axis_camera=np.stack([h.approach_axis_camera for h in hypotheses]), width_m=np.asarray([h.width_m for h in hypotheses]), score=np.asarray([h.score for h in hypotheses]))
    np.savez_compressed(out / "selected_grasp_partial.npz", tcp_camera=selected.tcp_camera, first_contact_camera=selected.first_contact_camera, second_contact_camera=selected.second_contact_camera, closing_axis_camera=selected.closing_axis_camera, approach_axis_camera=selected.approach_axis_camera, width_m=np.asarray(selected.width_m), score=np.asarray(selected.score), heat_at_visible_points=heat_at_points, visible_points_camera=points)
    save_partial_grasp_overlay(rgb, intrinsic, heatmap, cup_mask, selected.first_contact_camera, selected.second_contact_camera, selected.tcp_camera, out / "partial_grasp_overlay.png")
    report = {"snapshot": str(args.snapshot), "transfer": transfer_name, "generation_input": "visible_points_camera + transferred 2D heatmap sampled at visible_pixels_uv", "full_cloud_used_for_generation": False, "full_cloud_used_for_evaluation": full is not None, "visible_point_count": int(len(points)), "full_point_count": 0 if full is None else int(len(full)), "selected_index": int(selected_index), "selected": selected.as_dict(), "quality": quality, "limitations": ["SAM3D is optional and not required by this fallback", "the second contact is a virtual thickness/symmetry hypothesis; use a real partial-cloud GraspNet runner when its checkpoint/command is configured"]}
    (out / "partial_grasp_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "selected_index": selected_index, "quality": quality[:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
