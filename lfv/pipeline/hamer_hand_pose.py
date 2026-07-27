from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import zarr

from lfv.data_processing.episode_io import find_rgb_path, iter_processed_episodes
from lfv.utils.config import get_nested
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()

ROOT = Path(__file__).resolve().parents[2]
RUN_HAMER_SCRIPT = ROOT / "scripts" / "run_hamer_demo_env.sh"


def choose_frames(timing: dict, window_size: int) -> list[int]:
    frames = [int(f) for f in timing.get("contact_frames", [])]
    if frames:
        return frames[:window_size]
    start = timing.get("contact_start")
    if start is None:
        raise ValueError("contact_timing has no contact_start/contact_frames.")
    return [int(start) + i for i in range(window_size)]


def hamer_dirs(ep_path: Path, cfg) -> tuple[Path, Path, Path]:
    out_dir = ep_path / str(get_nested(cfg, "hamer.output_dir", "hamer_grasp_pseudo_label"))
    input_dir = out_dir / str(get_nested(cfg, "hamer.input_dirname", "hamer_input"))
    output_dir = out_dir / str(get_nested(cfg, "hamer.output_dirname", "hamer_output"))
    return out_dir, input_dir, output_dir


def skeleton_dir(output_dir: Path) -> Path:
    # demo.py always writes keypoints under <out_folder>/skeleton2d.
    return output_dir / "skeleton2d"


def frame_has_keypoints(skel_dir: Path, frame: int) -> bool:
    if not skel_dir.exists():
        return False
    return any(skel_dir.glob(f"frame_{frame:06d}_person*_hand_2d.npy"))


def _export_frames(ep_path: Path, frames: list[int], input_dir: Path) -> None:
    rgb = zarr.open(str(find_rgb_path(ep_path)), mode="r")
    for frame in frames:
        out_path = input_dir / f"frame_{frame:06d}.png"
        if out_path.exists():
            continue
        if frame < 0 or frame >= len(rgb):
            raise ValueError(f"frame {frame} out of rgb range (len={len(rgb)})")
        img = np.asarray(rgb[frame], dtype=np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _load_attempted(output_dir: Path) -> set[int]:
    meta_path = output_dir / "hamer_run_meta.json"
    if not meta_path.exists():
        return set()
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return {int(f) for f in json.load(f).get("frames_attempted", [])}
    except Exception:
        return set()


def _distribute_staging_outputs(staging_out: Path, episodes: list[dict]) -> int:
    """Copy per-image HaMeR outputs from the shared staging folder back to each episode.

    Staging image stems are "<episode_name>__frame_XXXXXX"; the episode prefix is
    stripped so per-episode skeleton2d files keep the "frame_XXXXXX_*" layout.
    """
    skel = skeleton_dir(staging_out)
    if not skel.exists():
        return 0
    copied = 0
    for ep in episodes:
        prefix = ep["path"].name + "__"
        dst_dir = skeleton_dir(ep["output_dir"])
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in skel.glob(f"{prefix}frame_*"):
            dst = dst_dir / src.name[len(prefix):]
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    return copied


def _write_episode_meta(ep: dict, staged_frames: set[int]) -> None:
    skel = skeleton_dir(ep["output_dir"])
    attempted = _load_attempted(ep["output_dir"]) | staged_frames
    with_kp = [f for f in ep["frames"] if frame_has_keypoints(skel, f)]
    meta = {
        "episode": ep["path"].name,
        "frames_requested": [int(f) for f in ep["frames"]],
        "frames_attempted": sorted(int(f) for f in attempted),
        "frames_with_keypoints": [int(f) for f in with_kp],
        "skeleton_dir": str(skel),
    }
    with (ep["output_dir"] / "hamer_run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run(cfg) -> None:
    processed_root = Path(cfg.paths.processed_root)
    window_size = int(get_nested(cfg, "hamer.window_size", 4))
    batch_size = int(get_nested(cfg, "hamer.batch_size", 8))
    timing_rel = str(get_nested(cfg, "hamer.timing_path", "contact_timing/contact_timing.json"))
    overwrite = bool(get_nested(cfg, "runtime.overwrite", False))
    staging_root = processed_root / str(get_nested(cfg, "hamer.staging_dirname", "_hamer_batch_staging"))

    episodes: list[dict] = []
    failed: list[tuple[str, str]] = []
    for ep_path in iter_processed_episodes(processed_root, cfg.runtime.episodes):
        try:
            timing_path = ep_path / timing_rel
            if not timing_path.exists():
                raise FileNotFoundError(f"Missing contact timing file: {timing_path}")
            with timing_path.open("r", encoding="utf-8") as f:
                timing = json.load(f)
            frames = choose_frames(timing, window_size)
            _out_dir, input_dir, output_dir = hamer_dirs(ep_path, cfg)
            input_dir.mkdir(parents=True, exist_ok=True)
            skeleton_dir(output_dir).mkdir(parents=True, exist_ok=True)
            _export_frames(ep_path, frames, input_dir)
            episodes.append({"path": ep_path, "frames": frames, "input_dir": input_dir, "output_dir": output_dir})
        except Exception as exc:
            print(f"[hamer] failed to prepare {ep_path.name}: {exc}")
            failed.append((ep_path.name, str(exc)))

    # Pick up partial outputs from a previously interrupted batch run first.
    resumed = _distribute_staging_outputs(staging_root / "output", episodes)
    if resumed:
        print(f"[hamer] recovered {resumed} files from previous staging output")

    staged_per_episode: dict[str, set[int]] = {}
    for ep in episodes:
        attempted = _load_attempted(ep["output_dir"])
        skel = skeleton_dir(ep["output_dir"])
        for frame in ep["frames"]:
            done = frame_has_keypoints(skel, frame) or (frame in attempted and not overwrite)
            if not done:
                staged_per_episode.setdefault(ep["path"].name, set()).add(int(frame))

    total_missing = sum(len(v) for v in staged_per_episode.values())
    print(f"[hamer] episodes prepared: {len(episodes)}, frames needing HaMeR: {total_missing}")

    attempted_from_run: dict[str, set[int]] = {}
    if total_missing:
        staging_in = staging_root / "input"
        staging_out = staging_root / "output"
        if staging_in.exists():
            shutil.rmtree(staging_in)
        staging_in.mkdir(parents=True)
        staging_out.mkdir(parents=True, exist_ok=True)
        for ep in episodes:
            frames = staged_per_episode.get(ep["path"].name)
            if not frames:
                continue
            for frame in sorted(frames):
                src = ep["input_dir"] / f"frame_{frame:06d}.png"
                dst = staging_in / f"{ep['path'].name}__frame_{frame:06d}.png"
                dst.symlink_to(src)

        cmd = [
            "bash",
            str(RUN_HAMER_SCRIPT),
            "--img_folder",
            str(staging_in),
            "--out_folder",
            str(staging_out),
            "--batch_size",
            str(batch_size),
            "--save_skeleton2d",
            "--full_frame",
            "--file_type",
            "*.png",
        ]
        print("[hamer] run:", " ".join(cmd))
        log_path = staging_root / "hamer_demo.log"
        # demo.py prints "[Info] Processing image: <path>" for every image it reaches
        # and writes nothing for images without a detected person, so these log lines
        # are the only reliable record of "attempted" frames.
        processing_re = re.compile(r"\[Info\] Processing image: .*?/([^/]+)\.png")
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
                match = processing_re.search(line)
                if match:
                    stem = match.group(1)
                    ep_name, sep, frame_part = stem.partition("__frame_")
                    if sep:
                        try:
                            attempted_from_run.setdefault(ep_name, set()).add(int(frame_part))
                        except ValueError:
                            pass
        returncode = proc.wait()
        if returncode != 0:
            failed.append(("(batch)", f"hamer demo exited with code {returncode}; processed frames were still recorded"))
            print(f"[hamer] demo exited with code {returncode}; distributing whatever outputs exist")

    copied = _distribute_staging_outputs(staging_root / "output", episodes)
    print(f"[hamer] distributed {copied} keypoint/skeleton files to episode dirs")

    for ep in episodes:
        _write_episode_meta(ep, attempted_from_run.get(ep["path"].name, set()))

    if failed:
        log_path = processed_root / "hamer_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[hamer] failed {len(failed)} episodes; wrote {log_path}")
