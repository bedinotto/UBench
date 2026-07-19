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


def build_thermal_transform(aug, seed: int = None) -> A.Compose:
    """Build the training augmentation pipeline from an ``AugmentationConfig``.

    Single augmentation authority (R5). Geometry first (flip, affine on the
    Celsius array with constant fill), then physical intensity (drift + noise).

    ``seed`` is passed to ``A.Compose`` so the augmentation stream is
    **reproducible** (albumentations manages its own RNG — the global
    ``np.random`` seed does not control it, so this is what pins same-seed →
    identical augmentation, M6). It still varies per sample and per epoch (the
    Compose RNG advances across calls).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=aug.flip_prob),
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
