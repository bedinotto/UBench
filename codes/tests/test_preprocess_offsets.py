"""
UB-08: polygon/mask offsets must use the *clamped* crop origin.

``crop_to_bbox`` clamps its crop origin with ``max(0, bbox.min - padding)``,
but ``preprocess_data.py`` recomputed the polygon offset as ``bbox.min - 10``
unclamped.  For bounding boxes closer than ``padding`` px to the image border
the mask was rasterized up to 10 px away from the pixels it labels — silent
label corruption (UB-08, fixed T1.5).  The fix makes ``crop_to_bbox`` return
``(img, origin)`` — the origin it actually used — and preprocessing consume
that returned origin instead of recomputing it.

These tests build a fresh 2-frame dataset per test (fresh tmp dir: stale
``data/processed/`` can never mask a failure):

* ``R1001`` — bbox ``min_x=3`` (< padding): the border case that shifted.
* ``R1002`` — bbox ``min_x=10``: control; old and new offsets agree here,
  proving the fix leaves the non-border path untouched.

Both bboxes share ``min_y=15`` (no y-clamping), so a border failure isolates
to the x-axis clamp.  Ground truth is constructed without the code under
test: rasterize the polygon shifted by the clamped origin into a crop-shaped
canvas, then nearest-neighbour resize exactly as ``preprocess_mask`` does.
AC (§9 T1.5): mask foreground centroid within 1 px of expectation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from codes.unified_data import Config, MultiDirectoryDataLoader

_IMG = 64                       # synthetic frame side (matches §7.2 fixture)
_PAD = 10                       # crop_to_bbox default padding
_MIN_Y = 15                     # both bboxes: origin_y = 5, no clamping
_BORDER_MIN_X = 3               # < _PAD → origin_x clamps to 0 (was -7)
_CONTROL_MIN_X = 10             # = _PAD → origin_x = 0 without clamping

# Clamped crop origin and crop shape shared by both frames:
# x: max(0, min_x - 10) = 0 … min(64, 54 + 10) = 64;  y: 5 … 64.
_ORIGIN = (0, _MIN_Y - _PAD)
_CROP_HW = (_IMG - _ORIGIN[1], _IMG)

# Single-region polygon rectangle in original (uncropped) frame coordinates.
_REGION = "Nariz"               # Config.REGION_NAMES[4]
_REGION_IDX = 4
_POLY_RECT = (10, 20, 20, 30)   # x0, y0, x1, y1


def _rect_polygon(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    """Return 4-corner polygon [[x, y], ...] for a rectangle."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@pytest.fixture()
def offset_dataset(tmp_path: Path) -> Path:
    """Fresh 2-frame single-subject dataset: border bbox + control bbox."""
    data_dir = tmp_path / "data"
    (data_dir / "S1").mkdir(parents=True)
    rng = np.random.default_rng(8)

    polygons: dict[str, dict] = {}
    for frame_id in ("R1001", "R1002"):
        raw = rng.integers(29315, 31316, size=(_IMG, _IMG), dtype=np.uint16)
        cv2.imwrite(str(data_dir / "S1" / f"{frame_id}.tiff"), raw)
        polygons[frame_id] = {_REGION: _rect_polygon(*_POLY_RECT)}

    with open(data_dir / "S1_polygonal_masks.json", "w", encoding="utf-8") as fh:
        json.dump(polygons, fh)

    pd.DataFrame([
        {"ID": "R1001", "min_x": _BORDER_MIN_X,
         "min_y": _MIN_Y, "max_x": 54, "max_y": 54},
        {"ID": "R1002", "min_x": _CONTROL_MIN_X,
         "min_y": _MIN_Y, "max_x": 54, "max_y": 54},
    ]).to_csv(data_dir / "S1_bounding_boxes.csv", index=False)

    return tmp_path


def _expected_mask() -> np.ndarray:
    """Ground-truth mask: polygon shifted by the clamped origin, resized as
    preprocessing does.  Built with cv2 directly, not the code under test."""
    canvas = np.zeros(_CROP_HW, dtype=np.uint8)
    x0, y0, x1, y1 = _POLY_RECT
    ox, oy = _ORIGIN
    pts = np.array(_rect_polygon(x0 - ox, y0 - oy, x1 - ox, y1 - oy), np.int32)
    cv2.fillPoly(canvas, [pts], _REGION_IDX)
    return cv2.resize(canvas, Config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    assert xs.size, "mask has no foreground pixels"
    return float(xs.mean()), float(ys.mean())


def _run_preprocessing(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the real offline preprocessing in ``root``; return the masks dir."""
    monkeypatch.chdir(root)
    from codes.preprocess_data import preprocess_all_data
    preprocess_all_data()
    return root / "data" / "processed" / "masks"


def _assert_aligned(mask_path: Path) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    assert mask is not None, f"mask was not produced: {mask_path}"
    ex, ey = _centroid(_expected_mask())
    cx, cy = _centroid(mask)
    assert abs(cx - ex) <= 1.0 and abs(cy - ey) <= 1.0, (
        f"mask centroid ({cx:.1f}, {cy:.1f}) is off expected "
        f"({ex:.1f}, {ey:.1f}) by more than 1 px"
    )


def test_crop_to_bbox_returns_clamped_origin(offset_dataset, monkeypatch):
    """crop_to_bbox reports the origin it actually used — clamped, never
    negative — and (0, 0) with the full image when no bbox exists."""
    monkeypatch.chdir(offset_dataset)
    loader = MultiDirectoryDataLoader(Config())
    loader.load_annotations()
    img = np.zeros((_IMG, _IMG), dtype=np.float32)

    cropped, origin = loader.crop_to_bbox(img, "S1/R1001")
    assert origin == _ORIGIN            # x clamped 3-10 → 0, not -7
    assert cropped.shape == _CROP_HW

    _, origin = loader.crop_to_bbox(img, "S1/R1002")
    assert origin == _ORIGIN            # x = 10-10 = 0 without clamping

    uncropped, origin = loader.crop_to_bbox(img, "S1/NO_SUCH_ID")
    assert origin == (0, 0)
    assert uncropped.shape == img.shape


def test_border_bbox_mask_alignment(offset_dataset, monkeypatch):
    """min_x=3 (< 10 px padding): the UB-08 case — mask must not shift."""
    masks_dir = _run_preprocessing(offset_dataset, monkeypatch)
    _assert_aligned(masks_dir / "S1_R1001.png")


def test_control_bbox_mask_alignment(offset_dataset, monkeypatch):
    """min_x=10 control: alignment identical — the normal path didn't move."""
    masks_dir = _run_preprocessing(offset_dataset, monkeypatch)
    _assert_aligned(masks_dir / "S1_R1002.png")
