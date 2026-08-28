"""Pluggable RGB-D object detection and segmentation backends."""

from .backends import (
    ExternalMaskBackend,
    MaskPerceptionResult,
    PrecomputedMaskBackend,
)

__all__ = ["ExternalMaskBackend", "MaskPerceptionResult", "PrecomputedMaskBackend"]
