"""Physical thermal augmentation (T3.4, UB-19, M7).

Augmentation acts on the Celsius array before normalization (design i). The
intensity model is a physical additive sensor drift + Gaussian noise (replacing
multiplicative brightness/contrast); geometry is flip + Affine. Reproducibility
is pinned via the ``A.Compose(seed=...)`` stream (albumentations owns its RNG).
"""

from __future__ import annotations

import albumentations as A
import numpy as np

from codes.augmentation import ThermalSensorNoise, build_thermal_transform
from codes.config_schema import AugmentationConfig


def _celsius(seed=1):
    return np.random.default_rng(seed).uniform(20.0, 40.0, (16, 16)).astype(np.float32)


def test_drift_is_additive_not_multiplicative():
    # sigma=0 -> only the per-image constant offset. Additive => output-input is
    # the same scalar everywhere (a multiplicative op would scale with the input).
    compose = A.Compose([ThermalSensorNoise(0.5, 0.0, p=1.0)], seed=0)
    img = np.linspace(20.0, 40.0, 64, dtype=np.float32).reshape(8, 8)
    delta = compose(image=img, mask=np.zeros_like(img, np.int64))["image"] - img
    # Additive: delta is a single constant everywhere (only float32 rounding,
    # ~1e-5, varies it). A multiplicative op would make delta scale with the
    # input (ptp ~1.0 across a 20->40 ramp).
    assert np.ptp(delta) < 1e-2
    assert abs(float(delta.mean())) > 0  # something was actually added


def test_intensity_op_leaves_mask_untouched():
    compose = A.Compose([ThermalSensorNoise(0.5, 0.1, p=1.0)], seed=0)
    cel = _celsius()
    mask = np.random.default_rng(2).integers(0, 10, (16, 16)).astype(np.int64)
    out = compose(image=cel, mask=mask)
    assert np.array_equal(out["mask"], mask)          # intensity op: mask intact
    assert not np.array_equal(out["image"], cel)      # image changed


def test_same_seed_identical_augmentation():
    aug = AugmentationConfig()
    cel, mask = _celsius(), np.zeros((16, 16), np.int64)
    a = build_thermal_transform(aug, seed=42)(image=cel.copy(), mask=mask.copy())["image"]
    b = build_thermal_transform(aug, seed=42)(image=cel.copy(), mask=mask.copy())["image"]
    c = build_thermal_transform(aug, seed=99)(image=cel.copy(), mask=mask.copy())["image"]
    assert np.array_equal(a, b)          # same seed -> identical (M6)
    assert not np.array_equal(a, c)      # different seed -> different


def test_augmentation_varies_per_sample():
    # The Compose RNG advances across calls, so successive samples differ.
    tr = build_thermal_transform(AugmentationConfig(), seed=7)
    cel, mask = _celsius(), np.zeros((16, 16), np.int64)
    first = tr(image=cel.copy(), mask=mask.copy())["image"]
    second = tr(image=cel.copy(), mask=mask.copy())["image"]
    assert not np.array_equal(first, second)


def test_no_multiplicative_brightness_contrast():
    # The physically-dubious op is gone (M7): the pipeline has no
    # RandomBrightnessContrast.
    tr = build_thermal_transform(AugmentationConfig(), seed=0)
    names = {type(t).__name__ for t in tr.transforms}
    assert "RandomBrightnessContrast" not in names
    assert "ThermalSensorNoise" in names
    assert "Affine" in names  # migrated from the deprecated ShiftScaleRotate
