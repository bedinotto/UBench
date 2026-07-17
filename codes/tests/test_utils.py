"""Thermal utility tests (UB-15, T2.6)."""

from __future__ import annotations

import numpy as np

from codes.utils import normalize_thermal


def test_normalize_thermal_flat_image_returns_zeros() -> None:
    """A flat image (min == max) normalizes to zeros, not the raw values (UB-15)."""
    flat = np.full((4, 4), 29315, dtype=np.uint16)   # constant thermal frame
    out = normalize_thermal(flat)
    assert np.array_equal(out, np.zeros((4, 4), dtype=np.float32))
    assert out.max() == 0.0 and out.min() == 0.0


def test_normalize_thermal_scales_to_unit_range() -> None:
    """A varying image is min-max scaled into [0, 1]."""
    img = np.array([[0, 5], [10, 20]], dtype=np.float32)
    out = normalize_thermal(img)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert np.isclose(out[1, 0], 0.5)   # 10 is halfway between 0 and 20
