"""
Utilities for Data Processing
=============================
Shared functions for normalization, resizing, and bounding boxes.
"""

from typing import Tuple

import cv2
import numpy as np


def normalize_thermal(thermal_img: np.ndarray) -> np.ndarray:
    """Per-image min–max normalization to [0, 1].

    Legacy mode (`preprocessing.normalization: per_image_minmax`). Destroys
    absolute temperature and does **not** commute with resize. A flat image
    (min == max) normalizes to all-zeros — returning the raw image would leak
    out-of-[0, 1] magnitudes (UB-15).
    """
    min_val = thermal_img.min()
    max_val = thermal_img.max()
    if max_val - min_val > 0:
        return ((thermal_img - min_val) / (max_val - min_val)).astype(np.float32)
    return np.zeros_like(thermal_img, dtype=np.float32)


def normalize_fixed_range(thermal_celsius: np.ndarray,
                          lo: float, hi: float) -> np.ndarray:
    """Map a Celsius image linearly from ``[lo, hi]`` °C to [0, 1] (clip outside).

    Preserves absolute temperature — the modality's core signal (M7) — so two
    frames at different absolute temperatures map differently. Commutes with
    linear resize (unlike per-image min–max).
    """
    if hi <= lo:
        raise ValueError(f"fixed_range requires hi > lo, got [{lo}, {hi}]")
    scaled = (thermal_celsius.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def apply_normalization(thermal_img: np.ndarray, mode: str,
                        fixed_range_celsius: Tuple[float, float]) -> np.ndarray:
    """Single normalization authority (R5) — dispatch by config mode (T3.4/M7).

    Used by both the training Dataset and the inference path so they normalize
    identically. ``fixed_range`` expects a Celsius image; ``per_image_minmax``
    is unit-agnostic.
    """
    if mode == "fixed_range":
        lo, hi = fixed_range_celsius
        return normalize_fixed_range(thermal_img, lo, hi)
    if mode == "per_image_minmax":
        return normalize_thermal(thermal_img)
    raise ValueError(
        f"unknown normalization mode '{mode}'; expected 'fixed_range' or "
        f"'per_image_minmax'"
    )


def preprocess_thermal_image(thermal_img: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize a thermal image to ``target_size`` (bilinear).

    T3.4/design (i): normalization is **no longer** applied here — the offline
    ``.npy`` stores resized **Celsius** (absolute temperature), and the mode is
    applied at load time via :func:`apply_normalization`. This keeps the mode
    switch runtime-pure and lets augmentation act in physical units.
    """
    return cv2.resize(thermal_img, target_size, interpolation=cv2.INTER_LINEAR)

def preprocess_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize mask image using nearest neighbor interpolation.
    """
    resized_mask = cv2.resize(
        mask, target_size,
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask
