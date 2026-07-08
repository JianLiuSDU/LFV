from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lfv.data.episode_io import first_rgb_frame, iter_processed_episodes
from lfv.pipeline.object_specs import ObjectSpec, iter_object_specs


def uniform_grid_sampling(mask, bbox, num_samples: int):
    x_min, y_min, x_max, y_max = map(int, bbox)
    area = max(1, (x_max - x_min) * (y_max - y_min))
    grid_size = int(np.sqrt(area / (num_samples * 2)))
    grid_size = max(1, grid_size)

    xs = np.arange(x_min, x_max, grid_size)
    ys = np.arange(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    valid_points = []
    for x, y in points:
        x = int(x)
        y = int(y)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            valid_points.append([x, y])

    valid_points = np.array(valid_points)
    if len(valid_points) == 0:
        raise ValueError("No valid grid points inside mask")

    if len(valid_points) > num_samples:
        indices = np.random.choice(len(valid_points), num_samples, replace=False)
        valid_points = valid_points[indices]
    elif len(valid_points) < num_samples:
        pad_size = num_samples - len(valid_points)
        pad_indices = np.random.choice(len(valid_points), pad_size, replace=True)
        valid_points = np.vstack([valid_points, valid_points[pad_indices]])
    return valid_points


def process_object(ep_path: Path, cfg, spec: ObjectSpec, initial_frame) -> bool:
    bbox_path = ep_path / spec.bbox_dir / spec.bbox_file
    mask_path = ep_path / spec.mask_dir / spec.mask_file
    if not bbox_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing bbox or mask for {ep_path.name}/{spec.name}")

    sample_dir = ep_path / spec.sample_dir
    viz_dir = ep_path / "viz"
    sample_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / spec.sample_file
    if out_path.exists() and not bool(cfg.runtime.overwrite):
        print(f"[sample] skip existing {ep_path.name}/{spec.name}")
        return True

    bbox = np.load(bbox_path)
    mask = np.load(mask_path)
    points_2d = uniform_grid_sampling(mask, bbox, int(cfg.sampling.num_points))
    np.save(out_path, {"query_points_2d": points_2d})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(initial_frame)
    ax.contour(mask, colors="red", linewidths=1.0, alpha=0.6)
    ax.scatter(points_2d[:, 0], points_2d[:, 1], c="lime", s=15, marker="o", edgecolors="black", linewidths=0.5)
    ax.set_title(f"{ep_path.name} | {spec.name} Uniform Sampling (N={len(points_2d)})")
    ax.axis("off")
    fig.savefig(viz_dir / f"{spec.viz_prefix}_sampling_2d_uniform.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[sample] {ep_path.name}/{spec.name}: {len(points_2d)} points")
    return True


def process_episode(ep_path: str | Path, cfg, specs: list[ObjectSpec]) -> bool:
    ep_path = Path(ep_path)
    initial_frame = first_rgb_frame(ep_path)
    for spec in specs:
        process_object(ep_path, cfg, spec, initial_frame)
    return True


def run(cfg) -> None:
    specs = iter_object_specs(cfg)
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, specs)
        except Exception as exc:
            print(f"[sample] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        print(f"[sample] failed {len(failed)} episodes")
