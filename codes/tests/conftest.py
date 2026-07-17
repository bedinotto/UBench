"""
Shared pytest fixtures for UBench test suite.

Key fixture: ``synthetic_dataset`` — a session-scoped miniature dataset that is
structurally faithful to the real data layout so the full pipeline can run in
seconds without Git-LFS data.  See CLAUDE.md §7.2 for the specification.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import pytest


# Region names must match codes/config.yaml `regions:` list exactly.
# REGION_NAMES[0] = "background" (index 0 is never in polygon files).
_REGION_NAMES: list[str] = [
    "background",
    "Contorno inferior do Rosto",
    "Sombrancelha esquerda",
    "Sombrancelha direita",
    "Nariz",
    "Olho esquerdo",
    "Olho direito",
    "Boca",
    "Labios",
    "Testa",
]

# Fixed non-overlapping rectangles [x0, y0, x1, y1] per region on a 64×64 canvas.
# These stay inside the bbox crop region (x: 0-63, y: 0-63) and avoid the
# border so that even the UB-08 tripwire sample (min_x=3) has polygons in range.
_REGION_BOXES: dict[str, tuple[int, int, int, int]] = {
    "Contorno inferior do Rosto": (5,  50, 58, 60),
    "Sombrancelha esquerda":      (5,  15, 20, 20),
    "Sombrancelha direita":       (43, 15, 58, 20),
    "Nariz":                      (27, 25, 37, 38),
    "Olho esquerdo":              (8,  20, 20, 27),
    "Olho direito":               (43, 20, 55, 27),
    "Boca":                       (22, 40, 42, 48),
    "Labios":                     (25, 41, 39, 47),
    "Testa":                      (10,  5, 54, 14),
}


def _rect_polygon(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    """Return 4-corner polygon [[x,y], ...] for a rectangle."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _build_subject(data_dir, subject_n: int, rng: np.random.Generator) -> None:
    """Create all synthetic files for one subject S{subject_n}."""
    name = f"S{subject_n}"
    subj_dir = data_dir / name
    subj_dir.mkdir(parents=True, exist_ok=True)

    polygons: dict[str, dict] = {}
    bbox_rows: list[dict] = []

    for frame_i in range(1, 5):  # 4 frames per subject
        frame_id = f"R{subject_n}00{frame_i}"   # e.g. R1001, R2004

        # --- TIFF: 64×64 uint16, values ∈ [29315, 31315] → 20–40 °C ---
        raw = rng.integers(29315, 31316, size=(64, 64), dtype=np.uint16)
        tiff_path = subj_dir / f"{frame_id}.tiff"
        cv2.imwrite(str(tiff_path), raw)

        # --- Polygon entries (bare key, loader will prefix "S{n}/") ---
        polygons[frame_id] = {
            region: _rect_polygon(*box)
            for region, box in _REGION_BOXES.items()
        }

        # --- BBox row: first frame of S1 has min_x=3 (UB-08 tripwire) ---
        min_x = 3 if (subject_n == 1 and frame_i == 1) else 10
        bbox_rows.append({
            "ID":    frame_id,
            "min_x": min_x,
            "min_y": 10,
            "max_x": 54,
            "max_y": 54,
        })

    # Write root-level files (recommended layout, §4.4)
    poly_path = data_dir / f"{name}_polygonal_masks.json"
    with open(poly_path, "w", encoding="utf-8") as fh:
        json.dump(polygons, fh)

    bbox_path = data_dir / f"{name}_bounding_boxes.csv"
    pd.DataFrame(bbox_rows).to_csv(bbox_path, index=False)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped synthetic dataset root (see CLAUDE.md §7.2).

    Layout::

        <root>/
          data/
            S1/ … S5/          # one directory per subject (= one GroupKFold group)
              R{n}00{i}.tiff   # 64×64 uint16 thermal frames
            S{n}_polygonal_masks.json
            S{n}_bounding_boxes.csv
          # data/processed/ is intentionally absent (UB-01 tripwire)

    Subjects: S1–S5 (5 groups → K_FOLDS=2 splits cleanly).
    Frames per subject: 4 → 20 total samples.
    """
    rng = np.random.default_rng(42)
    random.seed(42)

    root = tmp_path_factory.mktemp("ubench_synth")
    data_dir = root / "data"
    data_dir.mkdir()

    for n in range(1, 6):   # S1 … S5
        _build_subject(data_dir, n, rng)

    return root
