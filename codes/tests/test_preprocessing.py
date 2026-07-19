"""Thermal normalization + processed-data manifest guard (T3.4, UB-19, M7).

Design (i): the ``.npy`` store resized **Celsius**; normalization is applied at
load time. This file pins the two normalization modes, the load-time staleness
guard, and — as an honest replacement for the (impossible) "bit-identical
refactor" proof — documents that per-image min–max does **not** commute with
resize while ``fixed_range`` does (which is *why* the per_image_minmax numbers
move under design (i)).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import time

from codes.preprocess_manifest import (
    PREPROCESS_VERSION,
    verify_preprocess_manifest,
    write_preprocess_manifest,
)
from codes.unified_data import _raw_to_celsius, raw_to_celsius
from codes.utils import apply_normalization, normalize_fixed_range, normalize_thermal


# --------------------------------------------------------------------------- #
# Normalization modes.
# --------------------------------------------------------------------------- #
def test_fixed_range_linear_mapping_and_clip():
    x = np.array([[20.0, 30.0, 40.0], [10.0, 50.0, 25.0]], np.float32)
    y = normalize_fixed_range(x, 20.0, 40.0)
    assert y[0, 0] == 0.0 and y[0, 2] == 1.0
    assert abs(y[0, 1] - 0.5) < 1e-6
    assert y[1, 0] == 0.0 and y[1, 1] == 1.0  # clipped outside [20, 40]


def test_fixed_range_preserves_absolute_temperature():
    hot = np.full((4, 4), 39.0, np.float32)
    cold = np.full((4, 4), 21.0, np.float32)
    # fixed_range: two frames at different absolute temps map differently (M7).
    assert not np.allclose(
        apply_normalization(hot, "fixed_range", (20.0, 40.0)),
        apply_normalization(cold, "fixed_range", (20.0, 40.0)),
    )
    # per_image_minmax: both flat -> all-zeros; absolute temperature is lost.
    assert np.allclose(
        apply_normalization(hot, "per_image_minmax", (20.0, 40.0)),
        apply_normalization(cold, "per_image_minmax", (20.0, 40.0)),
    )


def test_apply_normalization_unknown_mode_raises():
    with pytest.raises(ValueError):
        apply_normalization(np.zeros((2, 2), np.float32), "bogus", (20.0, 40.0))


# --------------------------------------------------------------------------- #
# Commutation with resize — the honest disclosure (T3.4).
# --------------------------------------------------------------------------- #
def test_fixed_range_commutes_with_resize():
    rng = np.random.default_rng(0)
    cel = rng.uniform(22.0, 38.0, size=(32, 32)).astype(np.float32)  # no clipping
    size = (64, 64)
    before = cv2.resize(normalize_fixed_range(cel, 20.0, 40.0), size,
                        interpolation=cv2.INTER_LINEAR)
    after = normalize_fixed_range(cv2.resize(cel, size, interpolation=cv2.INTER_LINEAR),
                                  20.0, 40.0)
    assert np.allclose(before, after, atol=1e-6)


def test_per_image_minmax_does_not_commute_with_resize():
    """Why design (i) moves per_image_minmax numbers: min/max change under resize."""
    rng = np.random.default_rng(0)
    cel = rng.uniform(20.0, 40.0, size=(64, 64)).astype(np.float32)
    size = (256, 256)
    before = cv2.resize(normalize_thermal(cel), size, interpolation=cv2.INTER_LINEAR)
    after = normalize_thermal(cv2.resize(cel, size, interpolation=cv2.INTER_LINEAR))
    assert not np.allclose(before, after)
    assert np.max(np.abs(before - after)) > 1e-3  # ~0.02 on this data


# --------------------------------------------------------------------------- #
# Processed-data manifest staleness guard (R4/M7).
# --------------------------------------------------------------------------- #
def test_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="force-preprocess"):
        verify_preprocess_manifest(tmp_path)


def test_manifest_version_mismatch_raises(tmp_path):
    (tmp_path / "preprocess_manifest.json").write_text(
        json.dumps({"preprocess_version": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="force-preprocess"):
        verify_preprocess_manifest(tmp_path)


def test_manifest_current_version_ok(tmp_path):
    write_preprocess_manifest(tmp_path, image_size=(256, 256),
                              normalization="fixed_range",
                              fixed_range_celsius=(20.0, 40.0), num_samples=5)
    manifest = verify_preprocess_manifest(tmp_path)
    assert manifest["preprocess_version"] == PREPROCESS_VERSION
    assert manifest["stored_unit"] == "celsius"


# --------------------------------------------------------------------------- #
# Vectorized raw -> Celsius (T3.4): same result as the scalar path, >=100x fast.
# --------------------------------------------------------------------------- #
def test_raw_to_celsius_matches_scalar():
    raw = np.arange(0, 65535, 137, dtype=np.uint16)
    scalar = np.array([_raw_to_celsius(v) for v in raw], dtype=np.float32)
    assert np.allclose(raw_to_celsius(raw), scalar, atol=1e-4)


def test_raw_to_celsius_at_least_100x_faster_than_vectorize():
    raw = np.random.default_rng(0).integers(29315, 31316, size=(480, 640),
                                             dtype=np.uint16)
    old = np.vectorize(_raw_to_celsius)  # the per-pixel Python loop we replaced

    t0 = time.perf_counter()
    old(raw)
    slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):        # average the fast path (sub-ms) over repeats
        raw_to_celsius(raw)
    fast = (time.perf_counter() - t0) / 10

    assert fast > 0
    assert slow / fast >= 100.0, f"only {slow / fast:.1f}x faster"
