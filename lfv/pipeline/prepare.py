from __future__ import annotations

from lfv.data.episode_io import prepare_processed_dataset


def run(cfg) -> None:
    episodes = cfg.runtime.episodes
    prepared = prepare_processed_dataset(
        raw_root=cfg.paths.raw_root,
        processed_root=cfg.paths.processed_root,
        episodes=episodes,
        overwrite=bool(cfg.runtime.overwrite),
    )
    print(f"[prepare] prepared {len(prepared)} episodes under {cfg.paths.processed_root}")

