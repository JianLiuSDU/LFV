import pytest

from lfv.robot.gripper_extension import (
    DEFAULT_LONG_FINGER_SPEC,
    DRAWER_LONG_FINGER_SPEC,
    LongFingerExtensionSpec,
)


def test_long_finger_contact_area_is_substantially_larger_than_stock_pad():
    spec = LongFingerExtensionSpec()
    assert spec.contact_area_ratio == pytest.approx(6.4864864865)
    assert spec.contact_area_ratio > 6.0


def test_long_finger_inner_surfaces_remain_on_joint_center_planes():
    spec = LongFingerExtensionSpec()
    left = spec.center_for_side("left")
    right = spec.center_for_side("right")
    assert left[1] - spec.thickness_m / 2 == pytest.approx(0.0)
    assert right[1] + spec.thickness_m / 2 == pytest.approx(0.0)
    assert left[2] == pytest.approx(right[2])


def test_long_finger_rejects_invalid_side_and_dimensions():
    with pytest.raises(ValueError):
        LongFingerExtensionSpec(contact_length_m=0.0)
    with pytest.raises(ValueError):
        LongFingerExtensionSpec().center_for_side("middle")


def test_drawer_finger_is_narrower_thinner_and_still_enlarges_contact_area():
    drawer = DRAWER_LONG_FINGER_SPEC
    cup = DEFAULT_LONG_FINGER_SPEC
    assert drawer.contact_width_m < cup.contact_width_m
    assert drawer.thickness_m < cup.thickness_m
    assert drawer.center_z_m > cup.center_z_m
    assert drawer.contact_length_m == pytest.approx(cup.contact_length_m)
    assert drawer.contact_area_ratio > 3.0
