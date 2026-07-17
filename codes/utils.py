"""
Utilities for Data Processing
=============================
Shared functions for normalization, resizing, and bounding boxes.
"""

import cv2
import numpy as np

def normalize_thermal(thermal_img: np.ndarray) -> np.ndarray:
    """Normalize thermal image to [0, 1] range using min-max scaling.

    A flat image (min == max) normalizes to all-zeros — returning the raw image
    would leak out-of-[0, 1] magnitudes into preprocessing (UB-15).
    """
    min_val = thermal_img.min()
    max_val = thermal_img.max()
    if max_val - min_val > 0:
        return (thermal_img - min_val) / (max_val - min_val)
    return np.zeros_like(thermal_img, dtype=np.float32)

def preprocess_thermal_image(thermal_img: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Apply standard preprocessing: normalization and resizing.
    """
    norm_img = normalize_thermal(thermal_img)
    resized_img = cv2.resize(
        norm_img, target_size,
        interpolation=cv2.INTER_LINEAR
    )
    return resized_img

def preprocess_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize mask image using nearest neighbor interpolation.
    """
    resized_mask = cv2.resize(
        mask, target_size,
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask
