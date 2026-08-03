"""UB-27: the landmark scheme must be chosen per IMAGE, not per FILE.

Charlotte-ThermalFace annotates frontal views with 73 landmarks and profile
views with 43. ``generate_polygonal_masks`` selects between
``LANDMARK_MAPPINGS_73`` and ``LANDMARK_MAPPINGS_43`` from ``num_points``, which
it derived by counting ``x{i}`` columns in the **file header**::

    num_points = sum(1 for col in header if col.startswith('x') and col[1:].isdigit())

Every ``data/S{n}.csv`` in this dataset declares all 146 x/y columns (73 pairs)
for *every* row — profile rows simply leave indices 43..72 empty. So
``num_points`` was always 73, ``LANDMARK_MAPPINGS_43`` was **dead code that
never executed**, and profile rows were rasterized through the 73-point mapping.

Because the 73-point mapping indexes landmarks that a profile row does not have,
the consequences are severe and *asymmetric* (measured on the real data):

* ``Boca`` (48-59), ``Labios`` (60-67), ``Testa`` (68-72) — every index > 42, so
  **no points at all**: the class is absent from the reference mask in ~50% of
  the corpus (measured presence 0.5% / 0.0% / 0.0% in profile frames).
* ``Olho direito`` (42-47) — only index 42 exists, leaving a **1-point polygon**
  (20 px/img vs 416 in frontal frames).
* ``Olho esquerdo`` (36-41) — six points do exist, but under the 43-point scheme
  those indices belong to ``Labios``/``Testa``, so the polygon is a large,
  anatomically displaced blob (6092 px/img vs 433 in frontal frames).

That asymmetry — not the shared-index overwrite hypothesised before this test
existed — is what produces the lateral IoU asymmetry reported in the study.

The fix resolves the scheme from the landmarks each ROW actually fills. These
tests drive ``generate_polygonal_masks`` on a hand-built raw CSV (the
``synthetic_dataset`` fixture writes the *derived* polygon JSON directly and
never a raw ``S{n}.csv``, so it cannot exercise this path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from codes.generate_boxes_polygons import (
    LANDMARK_MAPPINGS_73,
    generate_polygonal_masks,
)

_SCHEMA_POINTS = 73          # columns every real S{n}.csv declares
_PROFILE_POINTS = 43         # landmarks a profile row actually fills
_FRONTAL_ID = "R0001"
_PROFILE_ID = "R0002"


def _row(sample_id: str, n_filled: int) -> dict:
    """One CSV row: all 73 x/y columns declared, only *n_filled* populated.

    Coordinates are spread apart so every region yields a non-degenerate
    polygon; the exact geometry is irrelevant to what these tests assert.
    """
    row: dict[str, object] = {"ID": sample_id}
    for i in range(_SCHEMA_POINTS):
        filled = i < n_filled
        row[f"x{i}"] = 10 + (i % 12) * 4 if filled else None
        row[f"y{i}"] = 10 + (i // 12) * 4 if filled else None
    return row


@pytest.fixture()
def raw_csv(tmp_path: Path) -> Path:
    """A raw S{n}.csv with one frontal row (73 filled) and one profile (43)."""
    csv_path = tmp_path / "S9.csv"
    pd.DataFrame([
        _row(_FRONTAL_ID, _SCHEMA_POINTS),
        _row(_PROFILE_ID, _PROFILE_POINTS),
    ]).to_csv(csv_path, index=False)
    return csv_path


def _polygons(csv_path: Path, tmp_path: Path) -> dict:
    out = tmp_path / "polygons.json"
    generate_polygonal_masks(str(csv_path), str(out))
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


def test_profile_row_is_not_read_through_the_73_point_mapping(
    raw_csv: Path, tmp_path: Path,
) -> None:
    """UB-27: a 43-landmark row must never be rasterized as if it had 73.

    Red before the fix: the profile entry was emitted silently, missing
    Boca/Labios/Testa entirely and holding a 1-point ``Olho direito``. After
    the fix the row resolves to the 43-point scheme, which is itself rejected
    (UB-29, asserted below) — either way the silent-corruption path is gone.
    """
    with pytest.raises(ValueError) as excinfo:
        _polygons(raw_csv, tmp_path)

    message = str(excinfo.value)
    assert _PROFILE_ID in message, "the failing image must be named"
    assert "43" in message, "the resolved scheme must be named"


def test_profile_row_rejects_double_booked_lateral_ranges(
    raw_csv: Path, tmp_path: Path,
) -> None:
    """UB-29: identical index ranges for a left/right pair must raise.

    ``LANDMARK_MAPPINGS_43`` gives both eyebrows indices 12-15 and both eyes
    22-26. All 43 indices are consumed with no gaps, so the real scheme
    evidently annotates only the *visible* side of a profile face — which side
    is not encoded. Emitting both names from one range would make
    ``create_segmentation_mask`` (which paints in ``REGION_NAMES`` order) erase
    the left side in every profile frame: the exact silent corruption UB-27
    already produced, merely relocated. So it raises, naming what is missing.
    """
    with pytest.raises(ValueError, match="same landmark indices") as excinfo:
        _polygons(raw_csv, tmp_path)

    message = str(excinfo.value)
    assert "Sombrancelha" in message or "Olho" in message, (
        "the error must name the colliding regions"
    )
    assert "LANDMARK_MAPPINGS_43" in message, (
        "the error must name what to supply to unblock profile annotation"
    )


def test_scheme_is_resolved_per_row_not_per_file(tmp_path: Path) -> None:
    """The scheme comes from the row's filled landmarks, not the header width.

    This is the heart of UB-27. The file below declares all 73 x/y columns —
    exactly like every real ``S{n}.csv`` — but its single row fills only 43.
    Header-based selection saw "73"; row-based selection sees 43.
    """
    csv_path = tmp_path / "S7.csv"
    pd.DataFrame([_row(_PROFILE_ID, _PROFILE_POINTS)]).to_csv(csv_path, index=False)

    header = pd.read_csv(csv_path, nrows=0).columns
    declared = sum(1 for c in header if c.startswith("x") and c[1:].isdigit())
    assert declared == _SCHEMA_POINTS, "fixture must mimic the real 73-column header"

    with pytest.raises(ValueError, match="43"):
        generate_polygonal_masks(str(csv_path), str(tmp_path / "out.json"))


def test_frontal_only_file_is_unchanged(tmp_path: Path) -> None:
    """Regression guard: the 73-landmark path keeps its exact previous output.

    The fix must not disturb frontal frames — they were always resolved
    correctly, and the reported run's frontal masks must stay reproducible.
    """
    csv_path = tmp_path / "S6.csv"
    pd.DataFrame([_row(_FRONTAL_ID, _SCHEMA_POINTS)]).to_csv(csv_path, index=False)

    frontal = _polygons(csv_path, tmp_path)[_FRONTAL_ID]

    assert set(frontal) == set(LANDMARK_MAPPINGS_73)
    for region, indices in LANDMARK_MAPPINGS_73.items():
        assert len(frontal[region]) == len(indices), (
            f"frontal {region!r} changed: {len(frontal[region])} points, "
            f"expected {len(indices)}"
        )


def test_unknown_landmark_count_raises(tmp_path: Path) -> None:
    """An unrecognised landmark count must raise, not silently fall back (R4).

    The old code printed a warning and used the 73-point mapping anyway —
    exactly the silent-default pattern that turns a defect into a lie.
    """
    csv_path = tmp_path / "S8.csv"
    pd.DataFrame([_row("R0003", 55)]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="55"):
        generate_polygonal_masks(str(csv_path), str(tmp_path / "out.json"))
