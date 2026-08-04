from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class DinoV2DenseExtractor:
    """Local-weight DINOv2 patch-token extractor.

    The model is deliberately frozen and is never downloaded implicitly.  This
    keeps quick experiments reproducible and usable on machines without network
    access.
    """

    def __init__(
        self,
        *,
        model_name: str = "vit_small_patch14_dinov2",
        weights_path: str | Path,
        device: str = "cuda",
    ) -> None:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency error
            raise ImportError("DINOv2 extraction requires timm.") from exc

        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.weights_path = weights_path
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            dynamic_img_size=True,
            num_classes=0,
        )
        checkpoint: Any = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "teacher"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported DINOv2 checkpoint type: {type(checkpoint)!r}")
        checkpoint = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in checkpoint.items()
        }
        incompatible = self.model.load_state_dict(checkpoint, strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("head.", "fc_norm."))
        ]
        if missing:
            raise RuntimeError(f"DINOv2 checkpoint is missing model keys: {missing[:8]}")
        self.model.eval().requires_grad_(False).to(self.device)
        patch = self.model.patch_embed.patch_size
        self._patch_size = int(patch[0] if isinstance(patch, tuple) else patch)

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @torch.inference_mode()
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 [H,W,3], got {rgb.shape} {rgb.dtype}")
        height, width = rgb.shape[:2]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch size "
                f"{self.patch_size}."
            )

        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)

        output = self.model.forward_features(tensor)
        if isinstance(output, dict):
            tokens = output.get("x_norm_patchtokens")
            if tokens is None:
                tokens = output.get("x_prenorm")
        else:
            tokens = output
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError("DINOv2 did not return a [B,L,D] token tensor.")

        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        expected = grid_h * grid_w
        if tokens.shape[1] != expected:
            prefix = tokens.shape[1] - expected
            if prefix < 0:
                raise RuntimeError(
                    f"DINOv2 returned {tokens.shape[1]} tokens, expected at least {expected}."
                )
            tokens = tokens[:, prefix:]
        features = F.normalize(tokens.float(), dim=-1)
        return (
            features[0]
            .reshape(grid_h, grid_w, features.shape[-1])
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
