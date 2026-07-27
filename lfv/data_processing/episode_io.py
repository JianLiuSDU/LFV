from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import zarr

from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


@dataclass(frozen=True)
class EpisodePaths:
    name: str
    raw_path: Path
    processed_path: Path


def iter_episode_dirs(root: str | Path, episodes: Iterable[str | int] | None = None) -> list[Path]:
    root = Path(root)
    if episodes is not None:
        result = []
        for ep in episodes:
            ep_name = f"episode_{ep}" if isinstance(ep, int) or str(ep).isdigit() else str(ep)
            path = root / ep_name
            if path.is_dir():
                result.append(path)
        return sorted(result, key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name)

    matches = [Path(p) for p in glob.glob(str(root / "episode_*")) if Path(p).is_dir()]
    return sorted(matches, key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name)


def find_rgb_path(ep_path: str | Path) -> Path:
    ep_path = Path(ep_path)
    candidates = [ep_path / "rgb", ep_path / "camera_0" / "rgb"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No rgb zarr found under {ep_path}")


def find_depth_path(ep_path: str | Path) -> Path:
    ep_path = Path(ep_path)
    candidates = [ep_path / "depth", ep_path / "camera_0" / "depth"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No depth zarr found under {ep_path}")


def first_rgb_frame(ep_path: str | Path):
    rgb = zarr.open(str(find_rgb_path(ep_path)), mode="r")
    return rgb[0]


def safe_symlink(src: str | Path, dst: str | Path, overwrite: bool = False) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"[WARN] skip missing source: {src}")
        return
    if dst.is_symlink() or dst.exists():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(f"Refusing to overwrite real directory: {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def prepare_processed_episode(raw_ep: str | Path, processed_root: str | Path, overwrite: bool = False) -> EpisodePaths:
    raw_ep = Path(raw_ep)
    processed_ep = Path(processed_root) / raw_ep.name
    processed_ep.mkdir(parents=True, exist_ok=True)

    safe_symlink(find_rgb_path(raw_ep), processed_ep / "rgb", overwrite=overwrite)
    safe_symlink(find_depth_path(raw_ep), processed_ep / "depth", overwrite=overwrite)

    for filename in ("camera_0.mp4", "meta.json", "timestamps.npy"):
        src = raw_ep / filename
        if src.exists():
            safe_symlink(src, processed_ep / filename, overwrite=overwrite)

    return EpisodePaths(name=raw_ep.name, raw_path=raw_ep, processed_path=processed_ep)


def prepare_processed_dataset(raw_root: str | Path, processed_root: str | Path, episodes=None, overwrite: bool = False) -> list[EpisodePaths]:
    paths = []
    Path(processed_root).mkdir(parents=True, exist_ok=True)
    for raw_ep in iter_episode_dirs(raw_root, episodes):
        paths.append(prepare_processed_episode(raw_ep, processed_root, overwrite=overwrite))
    return paths


def iter_processed_episodes(processed_root: str | Path, episodes=None) -> list[Path]:
    return iter_episode_dirs(processed_root, episodes)

