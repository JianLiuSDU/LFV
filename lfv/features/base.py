from __future__ import annotations

from typing import Protocol

import numpy as np


class DenseFeatureExtractor(Protocol):
    """Interface for frozen image encoders returning an HxWxD patch grid."""

    @property
    def patch_size(self) -> int:
        ...

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalized dense descriptors from an RGB uint8 image."""
