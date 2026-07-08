if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import glob
import os

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy_3d.dataset.lfv_dataset import (
    load_episode_camera_params,
    unproject_2d_to_3d,
)


GOAL_CKPT_PATH = os.environ.get(
    "LFV_GOAL_CKPT",
    "data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest/checkpoints/latest.ckpt",
)
SCENE_ROOT = os.environ.get("LFV_SCENE_ROOT", "data/env_data/pickNplace_lfv")
OUTPUT_SUBDIR = "model_inference_goal_pose"
MAX_SCENES = 5
NUM_PTS = 256
INTRINSICS_SOURCE = "depth_intrinsics_original"

AFFORDANCE_POINTS_CANDIDATES = [
    os.path.join("affordance_sample_points", "sampled_2d_uniform.npy"),
    os.path.join("affordance_sample_points", "sample_points.npy"),
    os.path.join("sample_points", "sampled_2d_uniform.npy"),
]

TARGET_POINTS_CANDIDATES = [
    os.path.join("target_sample_points", "target_sampled_2d_uniform.npy"),
    os.path.join("target_sample_points", "sampled_2d_uniform.npy"),
    os.path.join("target_sample_points", "sample_points.npy"),
]


def load_depth(scene_dir: str) -> np.ndarray:
    candidates = [
        os.path.join(scene_dir, "depth", "0000.npy"),
        os.path.join(scene_dir, "depth", "0.npy"),
        os.path.join(scene_dir, "depth", "0000.npz"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = "depth" if "depth" in arr else arr.files[0]
            arr = arr[key]
        return np.asarray(arr, dtype=np.float32)
    raise FileNotFoundError(f"Missing depth file under {scene_dir}; tried {candidates}")


def load_points_2d(scene_dir: str, candidates):
    tried = []
    for rel_path in candidates:
        path = os.path.join(scene_dir, rel_path)
        tried.append(path)
        if not os.path.exists(path):
            continue
        arr = np.load(path, allow_pickle=True)
        try:
            obj = arr.item()
            if isinstance(obj, dict) and "query_points_2d" in obj:
                return np.asarray(obj["query_points_2d"], dtype=np.float32), path
        except Exception:
            pass
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float32), path
    raise FileNotFoundError(f"Missing 2D sample points; tried {tried}")


def resize_points(points: np.ndarray, num_pts: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == num_pts:
        return points
    if points.shape[0] > num_pts:
        idx = np.linspace(0, points.shape[0] - 1, num_pts).astype(np.int64)
        return points[idx].astype(np.float32)
    if points.shape[0] == 0:
        return np.zeros((num_pts, 3), dtype=np.float32)
    pad_idx = np.arange(num_pts - points.shape[0]) % points.shape[0]
    return np.concatenate([points, points[pad_idx]], axis=0).astype(np.float32)


def normalize_lang_emb_shape(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]
    if emb.shape[-1] != 1024:
        raise ValueError(f"lang_emb last dim should be 1024, got {emb.shape}")
    return emb.astype(np.float32)


def load_lang_emb(scene_root: str, task_data_dirs, require_lang: bool):
    if not require_lang:
        return None, "disabled"
    candidates = [os.path.join(scene_root, "lang_emb.npy")]
    candidates.extend(os.path.join(str(d), "lang_emb.npy") for d in task_data_dirs)
    for path in candidates:
        if os.path.exists(path):
            return normalize_lang_emb_shape(np.load(path)), path
    print("[Warn] Missing lang_emb.npy; using zero language embedding [1,1024].")
    return np.zeros((1, 1024), dtype=np.float32), "zero_fallback"


def load_policy(device):
    payload = torch.load(GOAL_CKPT_PATH, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload.get("state_dicts", {})
    if "ema_model" in state_dicts:
        policy.load_state_dict(state_dicts["ema_model"], strict=True)
        print("[*] Loaded ema_model")
    else:
        policy.load_state_dict(state_dicts["model"], strict=True)
        print("[*] Loaded model")

    if "normalizer" in payload:
        policy.normalizer.load_state_dict(payload["normalizer"])
    else:
        train_dataset = hydra.utils.instantiate(cfg.task.dataset)
        policy.set_normalizer(train_dataset.get_normalizer())

    policy.eval().to(device)
    return policy, cfg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy(device)
    task_data_dirs = OmegaConf.to_container(cfg.task.dataset.data_dirs, resolve=True)
    if isinstance(task_data_dirs, str):
        task_data_dirs = [task_data_dirs]
    lang_emb, lang_source = load_lang_emb(SCENE_ROOT, task_data_dirs, bool(getattr(policy, "use_lang_emb", False)))

    scene_dirs = sorted(d for d in glob.glob(os.path.join(SCENE_ROOT, "scene_*")) if os.path.isdir(d))
    scene_dirs = scene_dirs[:MAX_SCENES]
    print(f"[*] scene_root={SCENE_ROOT}, scenes={len(scene_dirs)}, lang={lang_source}")

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        intrinsics, depth_scale = load_episode_camera_params(scene_dir, INTRINSICS_SOURCE)
        depth = load_depth(scene_dir) * depth_scale
        affordance_2d, affordance_path = load_points_2d(scene_dir, AFFORDANCE_POINTS_CANDIDATES)
        target_2d, target_path = load_points_2d(scene_dir, TARGET_POINTS_CANDIDATES)

        pc_man_0 = resize_points(unproject_2d_to_3d(affordance_2d, depth, intrinsics, NUM_PTS), NUM_PTS)
        pc_tgt_0 = resize_points(unproject_2d_to_3d(target_2d, depth, intrinsics, NUM_PTS), NUM_PTS)
        centroid_0 = pc_man_0.mean(axis=0).astype(np.float32)

        obs = {
            "pc_manipulated": torch.from_numpy(pc_man_0 - centroid_0).float().unsqueeze(0).to(device),
            "pc_target": torch.from_numpy(pc_tgt_0 - centroid_0).float().unsqueeze(0).to(device),
            "agent_pos": torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, device=device),
        }
        if lang_emb is not None:
            obs["lang_token_embs"] = torch.from_numpy(lang_emb).float().unsqueeze(0).to(device)

        with torch.no_grad():
            result = policy.predict_action(obs)

        pred_pose7d = result["goal_pose7d"][0].detach().cpu().numpy().astype(np.float32)
        pred_pose9d = result["goal_pose9d"][0].detach().cpu().numpy().astype(np.float32)

        out_dir = os.path.join(scene_dir, OUTPUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "pred_goal_pose7d.npy"), pred_pose7d)
        np.save(os.path.join(out_dir, "pred_goal_pose9d.npy"), pred_pose9d)
        np.savez(
            os.path.join(out_dir, "debug_inputs.npz"),
            pc_man_0=pc_man_0,
            pc_tgt_0=pc_tgt_0,
            centroid_0=centroid_0,
            affordance_points_path=affordance_path,
            target_points_path=target_path,
        )
        print(f"[*] {scene_name}: saved {os.path.join(out_dir, 'pred_goal_pose7d.npy')}")


if __name__ == "__main__":
    main()
