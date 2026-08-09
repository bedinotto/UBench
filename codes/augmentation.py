"""Physically-plausible thermal augmentation (T3.4/M7).

Applied in **Celsius**, before normalization (design (i)). Replaces the
physically-dubious multiplicative ``RandomBrightnessContrast`` (a temperature
field has no brightness/contrast sensor analogue) with an **additive sensor
drift + Gaussian noise** model, and migrates the deprecated ``ShiftScaleRotate``
to ``A.Affine`` (within the pinned ``albumentations<2.0``; the 2.x unpin is
T3.5). Geometry (flip/affine) is kept; intensity ops are physical.
"""

from __future__ import annotations

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# The dataset's lateral naming is **image-side**, not the subject's anatomy:
# "Olho esquerdo" holds the smaller x in 99.9% of the corpus's 4225 frontal
# rows (UB-29, proven by data/scripts/a13_landmark_side_evidence.py). A
# horizontal flip therefore moves a lateral structure across the face's
# midline, and its label must move with it — see LateralAwareHorizontalFlip.
# Same convention as `generate_boxes_polygons._LATERAL_PAIRS`; the two are
# pinned to agree by `test_augmentation.test_lateral_pairs_match_annotator`.
_LEFT_TOKEN = "esquerd"
_RIGHT_TOKEN = "direit"


class ThermalSensorNoise(ImageOnlyTransform):
    """Additive per-image sensor drift + per-pixel Gaussian noise, in °C.

    Sensor drift is a single constant offset drawn in ``[-drift, +drift]`` °C
    (an honest additive shift, not a multiplicative contrast change); Gaussian
    noise (``sigma`` °C) models per-pixel sensor noise. Both act on the Celsius
    array before normalization, so their magnitudes carry physical meaning.
    """

    def __init__(self, drift_celsius: float = 0.5, noise_sigma_celsius: float = 0.1,
                 always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.drift_celsius = float(drift_celsius)
        self.noise_sigma_celsius = float(noise_sigma_celsius)

    def apply(self, img, **params):
        # Use the transform's seeded generator (set by the Compose seed), NOT
        # global np.random — that is what albumentations controls for
        # reproducibility (M6).
        rng = self.random_generator
        offset = rng.uniform(-self.drift_celsius, self.drift_celsius)
        noise = (rng.normal(0.0, self.noise_sigma_celsius, size=img.shape)
                 if self.noise_sigma_celsius > 0 else 0.0)
        return (img.astype(np.float32) + offset + noise).astype(np.float32)

    def get_transform_init_args_names(self):
        return ("drift_celsius", "noise_sigma_celsius")


def lateral_index_pairs(region_names) -> tuple[tuple[int, int], ...]:
    """Class-index pairs whose semantics swap under a horizontal flip.

    Derived from the region list itself (R5 — `config.regions` is the single
    ordered authority), by matching the dataset's own left/right tokens. A
    left-named region with no right-named counterpart, or vice versa, raises:
    a silently unpaired lateral class is exactly the state that produced UB-30
    (labels kept while the pixels crossed the midline).
    """
    left, right = {}, {}
    for idx, name in enumerate(region_names):
        low = name.lower()
        if _LEFT_TOKEN in low:
            left[low.replace(_LEFT_TOKEN, "")] = idx
        elif _RIGHT_TOKEN in low:
            right[low.replace(_RIGHT_TOKEN, "")] = idx

    unpaired = set(left) ^ set(right)
    if unpaired:
        raise ValueError(
            f"Lateral regions without a counterpart: {sorted(unpaired)}. Every "
            "left-named class needs its right-named twin so a horizontal flip "
            "can swap the two (UB-30)."
        )
    return tuple((left[stem], right[stem]) for stem in sorted(left))


class LateralAwareHorizontalFlip(A.HorizontalFlip):
    """Horizontal flip that also swaps the labels of lateral class pairs.

    Mirroring image and mask with the *same* geometric transform is correct
    only for side-agnostic classes. For a lateral pair the label is defined by
    which side of the image the structure occupies, so the flip must permute
    the pair as well, or the training set ends up containing both "eye on the
    image-left = class 5" and "eye on the image-right = class 5", making the
    two classes indistinguishable and capping them near chance (UB-30).

    The swap lives in ``apply_to_mask`` rather than in a separate transform on
    purpose: albumentations calls it **only when this transform fires**, so the
    flip and the permutation are atomic. A second, independently-sampled
    transform would desynchronise from the flip half the time.
    """

    def __init__(self, lateral_pairs, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.lateral_pairs = tuple((int(a), int(b)) for a, b in lateral_pairs)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        flipped = super().apply(mask, **params)
        if not self.lateral_pairs:
            return flipped
        out = flipped.copy()
        for a, b in self.lateral_pairs:
            out[flipped == a] = b
            out[flipped == b] = a
        return out

    def get_transform_init_args_names(self):
        return ("lateral_pairs",)


def build_thermal_transform(aug, region_names, seed: int = None) -> A.Compose:
    """Build the training augmentation pipeline from an ``AugmentationConfig``.

    Single augmentation authority (R5). Geometry first (flip, affine on the
    Celsius array with constant fill), then physical intensity (drift + noise).

    ``region_names`` is the ordered class list (``config.regions``), required
    rather than defaulted: it is what tells the flip which class pairs to
    permute, and a default of "no pairs" would silently reinstate UB-30.

    ``seed`` is passed to ``A.Compose`` so the augmentation stream is
    **reproducible** (albumentations manages its own RNG — the global
    ``np.random`` seed does not control it, so this is what pins same-seed →
    identical augmentation, M6). It still varies per sample and per epoch (the
    Compose RNG advances across calls).
    """
    return A.Compose(
        [
            LateralAwareHorizontalFlip(
                lateral_pairs=lateral_index_pairs(region_names),
                p=aug.flip_prob,
            ),
            A.Affine(
                translate_percent=aug.translate_frac,
                scale=(1.0 - aug.scale_limit, 1.0 + aug.scale_limit),
                rotate=(-aug.rotate_limit, aug.rotate_limit),
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0.0,
                p=aug.affine_prob,
            ),
            ThermalSensorNoise(
                drift_celsius=aug.drift_celsius,
                noise_sigma_celsius=aug.noise_sigma_celsius,
                p=aug.noise_prob,
            ),
        ],
        seed=seed,
    )
