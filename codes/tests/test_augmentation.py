"""Physical thermal augmentation (T3.4, UB-19, M7).

Augmentation acts on the Celsius array before normalization (design i). The
intensity model is a physical additive sensor drift + Gaussian noise (replacing
multiplicative brightness/contrast); geometry is flip + Affine. Reproducibility
is pinned via the ``A.Compose(seed=...)`` stream (albumentations owns its RNG).
"""

from __future__ import annotations

import albumentations as A
import numpy as np

import pytest

from codes.augmentation import (
    LateralAwareHorizontalFlip,
    ThermalSensorNoise,
    build_thermal_transform,
    lateral_index_pairs,
)
from codes.config_schema import AugmentationConfig

# Ordered class list of the shipped config (codes/config.yaml `regions:`).
REGIONS = [
    "background", "Contorno inferior do Rosto", "Sombrancelha esquerda",
    "Sombrancelha direita", "Nariz", "Olho esquerdo", "Olho direito",
    "Boca", "Labios", "Testa",
]


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
    a = build_thermal_transform(aug, REGIONS, seed=42)(image=cel.copy(), mask=mask.copy())["image"]
    b = build_thermal_transform(aug, REGIONS, seed=42)(image=cel.copy(), mask=mask.copy())["image"]
    c = build_thermal_transform(aug, REGIONS, seed=99)(image=cel.copy(), mask=mask.copy())["image"]
    assert np.array_equal(a, b)          # same seed -> identical (M6)
    assert not np.array_equal(a, c)      # different seed -> different


def test_augmentation_varies_per_sample():
    # The Compose RNG advances across calls, so successive samples differ.
    tr = build_thermal_transform(AugmentationConfig(), REGIONS, seed=7)
    cel, mask = _celsius(), np.zeros((16, 16), np.int64)
    first = tr(image=cel.copy(), mask=mask.copy())["image"]
    second = tr(image=cel.copy(), mask=mask.copy())["image"]
    assert not np.array_equal(first, second)


def test_no_multiplicative_brightness_contrast():
    # The physically-dubious op is gone (M7): the pipeline has no
    # RandomBrightnessContrast.
    tr = build_thermal_transform(AugmentationConfig(), REGIONS, seed=0)
    names = {type(t).__name__ for t in tr.transforms}
    assert "RandomBrightnessContrast" not in names
    assert "ThermalSensorNoise" in names
    assert "Affine" in names  # migrated from the deprecated ShiftScaleRotate


# ---------------------------------------------------------------- UB-30 -----
# A horizontal flip moves a lateral structure across the face's midline. The
# dataset's naming is image-side (UB-29), so the label must move with it. Left
# unswapped, the training set holds both "eye on the image-left = class 5" and
# "eye on the image-right = class 5", which caps those classes near chance.

SOBR_E, SOBR_D, OLHO_E, OLHO_D = 2, 3, 5, 6


def _lateral_mask(size=64):
    """Mask with the four lateral classes on their convention-correct sides."""
    m = np.zeros((size, size), np.int64)
    m[10:15, 5:15] = SOBR_E      # image-left  -> "esquerda"
    m[10:15, 49:59] = SOBR_D     # image-right -> "direita"
    m[20:30, 5:15] = OLHO_E
    m[20:30, 49:59] = OLHO_D
    m[35:45, 28:36] = 4          # Nariz: side-agnostic, must NOT be remapped
    return m


def _centre_x(mask, cls):
    xs = np.where(mask == cls)[1]
    return float(xs.mean()) if xs.size else float("nan")


def test_lateral_pairs_derived_from_region_list():
    assert lateral_index_pairs(REGIONS) == ((OLHO_E, OLHO_D), (SOBR_E, SOBR_D))


def test_unpaired_lateral_region_raises():
    # A left-named class with no right-named twin cannot be swapped; silently
    # ignoring it is the UB-30 state, so it must raise (R4).
    with pytest.raises(ValueError, match="counterpart"):
        lateral_index_pairs(["background", "Olho esquerdo", "Nariz"])


def test_flip_swaps_lateral_labels_keeping_the_convention():
    """The class on the image-left stays the image-left class after a flip."""
    mask = _lateral_mask()
    flip = LateralAwareHorizontalFlip(lateral_index_pairs(REGIONS), p=1.0)
    out = A.Compose([flip], seed=0)(image=np.zeros_like(mask, np.float32),
                                    mask=mask)["mask"]
    midline = mask.shape[1] / 2
    # Convention holds on BOTH sides of the flip (this is what fails without
    # the remap: class 5 lands on the right and keeps its label).
    for left_cls, right_cls in ((SOBR_E, SOBR_D), (OLHO_E, OLHO_D)):
        assert _centre_x(mask, left_cls) < midline < _centre_x(mask, right_cls)
        assert _centre_x(out, left_cls) < midline < _centre_x(out, right_cls)


def test_flip_moves_pixels_but_relabels_them():
    """The geometry really is mirrored; only the labels follow the convention."""
    mask = _lateral_mask()
    flip = LateralAwareHorizontalFlip(lateral_index_pairs(REGIONS), p=1.0)
    out = A.Compose([flip], seed=0)(image=np.zeros_like(mask, np.float32),
                                    mask=mask)["mask"]
    # Pixel-wise, the mask is the plain mirror with the pair permuted: the
    # structure at the far right of the flipped frame is the one that was at
    # the far left, and it now carries the *right* class.
    mirrored = mask[:, ::-1]
    expected = mirrored.copy()
    for a, b in lateral_index_pairs(REGIONS):
        expected[mirrored == a] = b
        expected[mirrored == b] = a
    assert np.array_equal(out, expected)
    # Side-agnostic classes are mirrored but never relabelled.
    assert (out == 4).sum() == (mask == 4).sum()


def test_flip_and_relabel_are_atomic():
    """No flip means no relabel — the two can never desynchronise."""
    mask = _lateral_mask()
    never = LateralAwareHorizontalFlip(lateral_index_pairs(REGIONS), p=0.0)
    out = A.Compose([never], seed=0)(image=np.zeros_like(mask, np.float32),
                                     mask=mask)["mask"]
    assert np.array_equal(out, mask)


def test_lateral_pairs_match_annotator():
    """One convention, not two (R5).

    ``generate_boxes_polygons`` names the lateral pairs for annotation; this
    module derives their indices for augmentation. They must agree, or a class
    could be paired here and unpaired there.
    """
    from codes.generate_boxes_polygons import _LATERAL_PAIRS

    by_name = {(REGIONS.index(left), REGIONS.index(right))
               for left, right in _LATERAL_PAIRS}
    assert set(lateral_index_pairs(REGIONS)) == by_name
