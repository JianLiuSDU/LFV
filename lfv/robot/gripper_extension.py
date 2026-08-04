"""Geometry contract for the LFV Panda long-finger gripper extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LongFingerExtensionSpec:
    """UMI/Fin-Ray-inspired rigid proxy used for the first simulation pass.

    Dimensions are expressed in each Panda finger link frame.  The plate is
    centred on the stock TCP contact height, so changing grippers does not
    silently change the grasp-pose convention.
    """

    contact_width_m: float = 0.030
    contact_length_m: float = 0.070
    thickness_m: float = 0.008
    center_z_m: float = 0.0455
    static_friction: float = 5.0
    dynamic_friction: float = 5.0
    density_kg_m3: float = 300.0

    stock_contact_width_m: float = 0.0175
    stock_contact_length_m: float = 0.0185

    def __post_init__(self) -> None:
        for name in (
            "contact_width_m",
            "contact_length_m",
            "thickness_m",
            "center_z_m",
            "static_friction",
            "dynamic_friction",
            "density_kg_m3",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def half_size_m(self) -> tuple[float, float, float]:
        return (
            self.contact_width_m / 2,
            self.thickness_m / 2,
            self.contact_length_m / 2,
        )

    @property
    def contact_area_m2(self) -> float:
        return self.contact_width_m * self.contact_length_m

    @property
    def stock_contact_area_m2(self) -> float:
        return self.stock_contact_width_m * self.stock_contact_length_m

    @property
    def contact_area_ratio(self) -> float:
        return self.contact_area_m2 / self.stock_contact_area_m2

    def center_for_side(self, side: str) -> tuple[float, float, float]:
        """Return plate centre while keeping each inner surface at local y=0."""

        if side == "left":
            sign = 1.0
        elif side == "right":
            sign = -1.0
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return (0.0, sign * self.thickness_m / 2, self.center_z_m)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["contact_area_m2"] = self.contact_area_m2
        result["stock_contact_area_m2"] = self.stock_contact_area_m2
        result["contact_area_ratio"] = self.contact_area_ratio
        result["tcp_convention"] = "stock panda_hand_tcp remains at plate centre height"
        return result


DEFAULT_LONG_FINGER_SPEC = LongFingerExtensionSpec()

# The cup extension is intentionally broad.  A drawer pull handle has a much
# tighter rear clearance and short end supports, so retain the long, high-
# friction contact surface while narrowing the plate and reducing thickness.
DRAWER_LONG_FINGER_SPEC = LongFingerExtensionSpec(
    contact_width_m=0.016,
    contact_length_m=0.070,
    thickness_m=0.0045,
    # Shift the plate 30 mm farther along the stock finger axis.  With the
    # top-down TCP held above the cabinet this puts the lower 35 mm section
    # around the recessed handle instead of leaving it above the handle.
    center_z_m=0.0755,
    static_friction=5.0,
    dynamic_friction=5.0,
    density_kg_m3=300.0,
)
