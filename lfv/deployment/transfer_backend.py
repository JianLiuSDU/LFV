from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import numpy as np


class LocalAffCorrsBackend:
    """Run the repository's frozen-DINO Soft AffCorrs transfer in-process."""
    def __init__(self, source_rgb: str | Path, source_mask: str | Path, source_heatmap: str | Path, dino_weights: str | Path, *, dino_model: str = "vit_small_patch14_dinov2", device: str = "cuda", config: dict | None = None):
        self.source_rgb = Path(source_rgb).expanduser(); self.source_mask = Path(source_mask).expanduser(); self.source_heatmap = Path(source_heatmap).expanduser(); self.dino_weights = Path(dino_weights).expanduser(); self.dino_model = dino_model; self.device = device; self.config = config or {}

    def transfer(self, *, image: np.ndarray, perception: object, **_: object) -> np.ndarray:
        from PIL import Image
        from lfv.affordance_transfer.pipeline import SoftHeatmapAffCorrsPipeline
        from lfv.affordance_transfer.schema import SourceContactExample, TargetObservation
        from lfv.features.dinov2_dense import DinoV2DenseExtractor
        source_rgb = np.asarray(Image.open(self.source_rgb).convert("RGB"), dtype=np.uint8)
        source_mask = np.asarray(Image.open(self.source_mask).convert("L")) > 0
        if self.source_heatmap.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            source_heat = np.asarray(Image.open(self.source_heatmap).convert("L"), dtype=np.float32) / 255.0
        else:
            heat_payload = np.load(self.source_heatmap, allow_pickle=False)
            source_heat = np.asarray(heat_payload["heatmap"] if "heatmap" in heat_payload else heat_payload["source_heatmap"] if "source_heatmap" in heat_payload else heat_payload[heat_payload.files[0]], dtype=np.float32)
        target_mask = np.asarray(getattr(perception, "cup_mask"), dtype=bool)
        extractor = DinoV2DenseExtractor(model_name=self.dino_model, weights_path=self.dino_weights, device=self.device)
        result = SoftHeatmapAffCorrsPipeline(extractor, self.config).transfer(SourceContactExample(source_rgb, source_mask, source_heat, "source"), TargetObservation(np.asarray(image, dtype=np.uint8), target_mask, "target"))
        if not result.accepted:
            raise RuntimeError(f"AffCorrs transfer rejected: {result.rejection_reasons}")
        return np.asarray(result.target_heatmap, dtype=np.float32)


class PrecomputedHeatBackend:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def transfer(self, *, image: np.ndarray, workdir: str | Path, **_: object) -> np.ndarray:
        payload = np.load(self.path, allow_pickle=False)
        key = "target_heatmap" if "target_heatmap" in payload else "heatmap"
        heat = np.asarray(payload[key], dtype=np.float32)
        if heat.shape != image.shape[:2]:
            raise ValueError(f"Transferred heatmap {heat.shape} does not match image {image.shape[:2]}")
        return np.clip(heat, 0.0, 1.0)


class ExternalHeatBackend:
    """Run the existing AffCorrs/FGW implementation outside this process."""
    def __init__(self, command: str, output_name: str = "transfer_result.npz"):
        self.command, self.output_name = command, output_name

    def transfer(self, *, image: np.ndarray, workdir: str | Path, **_: object) -> np.ndarray:
        workdir = Path(workdir)
        subprocess.run(self.command.format(workdir=shlex.quote(str(workdir))), shell=True, check=True, cwd=str(workdir))
        payload = np.load(workdir / self.output_name, allow_pickle=False)
        key = "target_heatmap" if "target_heatmap" in payload else "heatmap"
        heat = np.asarray(payload[key], dtype=np.float32)
        if heat.shape != image.shape[:2]:
            raise ValueError(f"Transferred heatmap {heat.shape} does not match image {image.shape[:2]}")
        return np.clip(heat, 0.0, 1.0)
