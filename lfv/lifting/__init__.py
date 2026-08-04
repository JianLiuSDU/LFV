"""Lift image-space affordance predictions into executable 3D geometry."""

from .image_heat_to_surface import LiftedImageHeat, lift_image_heat_to_camera

__all__ = ["LiftedImageHeat", "lift_image_heat_to_camera"]
