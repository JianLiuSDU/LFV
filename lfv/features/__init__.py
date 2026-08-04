"""Frozen dense visual feature extractors used by LFV pipelines."""

from .base import DenseFeatureExtractor
from .dinov2_dense import DinoV2DenseExtractor

__all__ = ["DenseFeatureExtractor", "DinoV2DenseExtractor"]
