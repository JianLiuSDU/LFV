"""One-shot, image-space affordance transfer."""

from .pipeline import SoftHeatmapAffCorrsPipeline
from .schema import RGBDPart, SourceContactExample, TargetObservation, TransferResult
from .fgw_contact_transfer import AffCorrsFGWContactTransferPipeline

__all__ = [
    "AffCorrsFGWContactTransferPipeline",
    "RGBDPart",
    "SoftHeatmapAffCorrsPipeline",
    "SourceContactExample",
    "TargetObservation",
    "TransferResult",
]
