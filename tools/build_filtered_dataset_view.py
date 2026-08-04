#!/usr/bin/env python3
"""Build a symlink-only training view from a processed-data quality manifest.

The source recordings and processed episode directories are never modified.
Only links and a small provenance file are created under the requested view.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _safe_link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--language-embedding", default=None)
    args = parser.parse_args()

    processed_root = Path(args.processed_root).expanduser().resolve(strict=True)
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted = list(report.get("accepted_episodes", []))
    if not accepted:
        raise RuntimeError("Manifest contains no accepted episodes")

    output_root.mkdir(parents=True, exist_ok=True)
    expected = set(accepted)
    for stale in output_root.glob("episode_*"):
        if stale.name not in expected:
            if not stale.is_symlink():
                raise RuntimeError(f"Refusing to remove non-symlink: {stale}")
            stale.unlink()

    for episode_name in accepted:
        _safe_link(processed_root / episode_name, output_root / episode_name)

    if args.language_embedding:
        _safe_link(Path(args.language_embedding), output_root / "lang_emb.npy")

    provenance = {
        "schema_version": 1,
        "view_type": "symlink_only",
        "processed_root": str(processed_root),
        "manifest": str(manifest_path),
        "episode_count": len(accepted),
        "episodes": accepted,
        "language_embedding": args.language_embedding,
    }
    (output_root / "dataset_view.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "episode_count": len(accepted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
