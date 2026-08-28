from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskPerceptionResult:
    """Masks returned by a detector/segmenter in the input image frame."""

    cup_mask: np.ndarray
    bowl_mask: np.ndarray
    scores: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"

    def validate(self, image_shape: tuple[int, int]) -> None:
        for name, mask in (("cup_mask", self.cup_mask), ("bowl_mask", self.bowl_mask)):
            value = np.asarray(mask)
            if value.shape != image_shape or value.dtype != bool:
                raise ValueError(f"{name} must be bool [{image_shape}], got {value.shape} {value.dtype}")
            if not value.any():
                raise ValueError(f"{name} is empty")


class PerceptionBackend(Protocol):
    def predict(self, image: np.ndarray, input_root: Path) -> MaskPerceptionResult: ...


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.asarray(Image.open(path))
    mask = np.asarray(values).squeeze().astype(bool)
    if mask.shape != shape:
        raise ValueError(f"Mask {path} has shape {mask.shape}, expected {shape}")
    return mask


class PrecomputedMaskBackend:
    """Use masks produced offline by SAM/SAM2/Grounding-DINO or manual QA."""

    def __init__(self, cup_path: str | Path | None = None, bowl_path: str | Path | None = None):
        self.cup_path = Path(cup_path) if cup_path else None
        self.bowl_path = Path(bowl_path) if bowl_path else None

    def predict(self, image: np.ndarray, input_root: Path) -> MaskPerceptionResult:
        shape = tuple(image.shape[:2])
        cup = self.cup_path or next((p for p in (input_root / "cup_mask.png", input_root / "cup_mask.npy") if p.exists()), None)
        bowl = self.bowl_path or next((p for p in (input_root / "bowl_mask.png", input_root / "bowl_mask.npy") if p.exists()), None)
        if cup is None or bowl is None:
            raise FileNotFoundError("Precomputed perception requires cup_mask.png/.npy and bowl_mask.png/.npy")
        result = MaskPerceptionResult(_read_mask(cup, shape), _read_mask(bowl, shape), source="precomputed")
        result.validate(shape)
        return result


class ExternalMaskBackend:
    """Run an external detector/segmenter and read ``masks.npz``.

    The command receives ``{input_root}`` and must write ``masks.npz`` with
    boolean ``cup_mask`` and ``bowl_mask`` arrays.  This keeps GPU-specific
    SAM/SAM2/Grounding-DINO code outside the LFV runtime.
    """

    def __init__(self, command: str, output_name: str = "masks.npz"):
        self.command = command
        self.output_name = output_name

    def predict(self, image: np.ndarray, input_root: Path) -> MaskPerceptionResult:
        command = self.command.format(input_root=shlex.quote(str(input_root)))
        subprocess.run(command, shell=True, check=True, cwd=str(input_root))
        output = input_root / self.output_name
        if not output.exists():
            raise FileNotFoundError(f"Perception command did not produce {output}")
        payload = np.load(output, allow_pickle=False)
        shape = tuple(image.shape[:2])
        result = MaskPerceptionResult(
            np.asarray(payload["cup_mask"]).astype(bool),
            np.asarray(payload["bowl_mask"]).astype(bool),
            source="external",
        )
        result.validate(shape)
        return result


def save_masks(result: MaskPerceptionResult, output: str | Path) -> None:
    """Persist masks and scores for reproducible handoff/debugging."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, cup_mask=result.cup_mask, bowl_mask=result.bowl_mask)
    (output.with_suffix(".json")).write_text(
        json.dumps({"source": result.source, "scores": result.scores}, indent=2), encoding="utf-8"
    )
