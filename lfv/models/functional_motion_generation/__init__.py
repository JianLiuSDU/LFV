"""Functional Motion Generation Network package."""

from .registry import MODEL_REGISTRY, build_model, register_model
from .system import ThreeTokenHierarchicalDiffusion
from .v7_system import V7FunctionalAlignmentDiffusion
from .loading import load_stage2_checkpoint
from .canonical_alignment import CanonicalFieldMemory

__all__ = [
    "MODEL_REGISTRY",
    "ThreeTokenHierarchicalDiffusion",
    "V7FunctionalAlignmentDiffusion",
    "CanonicalFieldMemory",
    "build_model",
    "load_stage2_checkpoint",
    "register_model",
]
