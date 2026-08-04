"""UB-27/UB-29: the landmark scheme, and the visible side, are per-IMAGE facts.

Charlotte-ThermalFace annotates frontal views with 73 landmarks and profile
views with 43. Two distinct defects lived here.

**UB-27 — the scheme was chosen per FILE.** ``generate_polygonal_masks`` picked
between ``LANDMARK_MAPPINGS_73`` and ``LANDMARK_MAPPINGS_43`` from
``num_points``, derived by counting ``x{i}`` columns in the **file header**::

    num_points = sum(1 for col in header if col.startswith('x') and col[1:].isdigit())

Every ``data/S{n}.csv`` declares all 146 x/y columns (73 pairs) for *every* row —
profile rows simply leave indices 43..72 empty. So ``num_points`` was always 73,
``LANDMARK_MAPPINGS_43`` was **dead code that never executed**, and profile rows
were rasterized through the 73-point mapping. Measured consequences: Boca
(48-59), Labios (60-67) and Testa (68-72) got no points at all (class absent
from ~50% of the corpus); Olho direito (42-47) got a single point (20 px/img vs
416); Olho esquerdo (36-41) got six points belonging to Labios/Testa (6092
px/img vs 433).

**UB-29 — the 43-point mapping double-books the lateral pairs.** It gives both
eyebrows indices 12-15 and both eyes 22-26, and all 43 indices are consumed with
no gaps. The reason is anatomical: in a profile view only **one** side of the
face is visible, so the scheme annotates one eye and one eyebrow — and which
side that is depends on which way the head is turned, which the CSV does not
state in any column.

The side is nonetheless *derivable from the landmark geometry*, and the
derivation was validated on the real corpus by two independent, orthogonal
signals measured over 4225 frontal (73-point) rows, both monotonic across five
head-rotation bins:

* **eye width** — as the nose moves toward image-left, ``Olho esquerdo``
  narrows relative to ``Olho direito`` (ratio 0.813 / 0.920 / 1.021 / 1.123 /
  1.292 from nose-far-left to nose-far-right);
* **eye-to-nose distance** — the eye on the receding side collapses toward the
  nose (nose far left: 0.140 vs 0.269; nose far right: 0.275 vs 0.144).

Both agree: the eye that *survives* into a profile view is the one on the side
**opposite** the nose's horizontal displacement. Combined with the dataset's
own naming convention — verified as image-side, since ``Olho esquerdo`` has the
smaller x in 99.9% of the corpus's 4225 frontal rows — this yields the rule
these tests pin:

    nose displaced toward image-LEFT   -> visible side is "direito"/"direita"
    nose displaced toward image-RIGHT  -> visible side is "esquerdo"/"esquerda"

The occluded side is **omitted** from the polygon entry rather than duplicated,
because it genuinely carries no annotation; the metric authority already
excludes absent classes from the macro average (``codes/metrics.py``).

This rule is *derived and validated*, not the dataset's published
specification — see ``data/a13_landmark_side_evidence.py`` for the reproducible
evidence. Should the Charlotte-ThermalFace authors publish a per-side index
convention that differs, theirs governs.

These tests drive ``generate_polygonal_masks`` on hand-built raw CSVs: the
``synthetic_dataset`` fixture writes the *derived* polygon JSON directly and
never a raw ``S{n}.csv``, so it cannot exercise this path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from codes.generate_boxes_polygons import (
    LANDMARK_MAPPINGS_43,
    LANDMARK_MAPPINGS_73,
    generate_polygonal_masks,
)

_SCHEMA_POINTS = 73          # columns every real S{n}.csv declares
_PROFILE_POINTS = 43         # landmarks a profile row actually fills
_FRONTAL_ID = "R0001"
_PROFILE_ID = "R0002"

# Index blocks of the 43-point (profile) scheme, mirroring LANDMARK_MAPPINGS_43.
_P_CONTOUR = range(0, 12)
_P_BROW = range(12, 16)
_P_NOSE = range(16, 22)
_P_EYE = range(22, 27)

_LEFT_NAMES = ("Sombrancelha esquerda", "Olho esquerdo")
_RIGHT_NAMES = ("Sombrancelha direita", "Olho direito")


def _frontal_row(sample_id: str) -> dict:
    """A row filling all 73 landmarks; geometry irrelevant to the assertions."""
    row: dict[str, object] = {"ID": sample_id}
    for i in range(_SCHEMA_POINTS):
        row[f"x{i}"] = 10 + (i % 12) * 4
        row[f"y{i}"] = 10 + (i // 12) * 4
    return row


def _profile_row(sample_id: str, nose_x: float) -> dict:
    """A row filling exactly 43 landmarks, with the nose at a chosen x.

    The face contour spans x in [0, 100], so ``nose_x`` places the nose on the
    left half, the right half, or ambiguously near the middle. Indices 43..72
    are left empty, exactly as a real profile row leaves them.
    """
    row: dict[str, object] = {"ID": sample_id}
    for i in range(_SCHEMA_POINTS):
        row[f"x{i}"] = None
        row[f"y{i}"] = None

    for k, i in enumerate(_P_CONTOUR):          # spans the full face width
        row[f"x{i}"] = k * (100 / (len(_P_CONTOUR) - 1))
        row[f"y{i}"] = 60 + (k % 3) * 5
    for k, i in enumerate(_P_NOSE):             # clustered at nose_x
        row[f"x{i}"] = nose_x + k
        row[f"y{i}"] = 40 + k
    for k, i in enumerate(_P_BROW):             # placed away from the nose
        row[f"x{i}"] = (100 - nose_x) + k
        row[f"y{i}"] = 15 + k
    for k, i in enumerate(_P_EYE):
        row[f"x{i}"] = (100 - nose_x) + k
        row[f"y{i}"] = 25 + k
    for k, i in enumerate(range(27, 43)):       # boca/labios/testa
        row[f"x{i}"] = 30 + (k % 8) * 5
        row[f"y{i}"] = 75 + (k // 8) * 6
    return row


def _write(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    csv_path = tmp_path / name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _polygons(csv_path: Path, tmp_path: Path, max_skip_fraction: float = 0.05) -> dict:
    """Run the generator and return its polygon dict.

    These fixtures hold a handful of rows, so a single intentionally-bad row is
    a large *fraction* of the file. Tests that exercise the skip path therefore
    pass an explicit ``max_skip_fraction``; the production default stays strict.
    """
    out = tmp_path / f"{csv_path.stem}_polygons.json"
    generate_polygonal_masks(str(csv_path), str(out), max_skip_fraction)
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


def test_scheme_is_resolved_per_row_not_per_file(tmp_path: Path) -> None:
    """UB-27's core: the scheme comes from the row, not the header width.

    The file declares all 73 x/y columns — exactly like every real
    ``S{n}.csv`` — but its single row fills 43. Header-based selection saw
    "73"; row-based selection sees 43 and emits the profile regions.
    """
    csv_path = _write(tmp_path, "S7.csv", [_profile_row(_PROFILE_ID, nose_x=10)])

    header = pd.read_csv(csv_path, nrows=0).columns
    declared = sum(1 for c in header if c.startswith("x") and c[1:].isdigit())
    assert declared == _SCHEMA_POINTS, "fixture must mimic the real 73-column header"

    entry = _polygons(csv_path, tmp_path)[_PROFILE_ID]

    # Under the 73-point mapping these three get no points at all (UB-27).
    for region in ("Boca", "Labios", "Testa"):
        assert region in entry, f"{region!r} missing: row read as 73 points (UB-27)"
        assert len(entry[region]) >= 3, f"{region!r} degenerate: {entry[region]}"


def test_profile_facing_image_left_labels_the_right_side(tmp_path: Path) -> None:
    """UB-29: nose toward image-left => the visible eye/brow is the RIGHT one.

    Derived from the eye-width and eye-to-nose signals (module docstring): the
    eye that survives into profile is the one opposite the nose's displacement.
    """
    csv_path = _write(tmp_path, "S7.csv", [_profile_row(_PROFILE_ID, nose_x=10)])
    entry = _polygons(csv_path, tmp_path)[_PROFILE_ID]

    for region in _RIGHT_NAMES:
        assert region in entry, f"{region!r} should be the visible side"
        assert len(entry[region]) >= 3, f"{region!r} degenerate: {entry[region]}"
    for region in _LEFT_NAMES:
        assert region not in entry, (
            f"{region!r} is on the occluded side and carries no annotation; "
            f"emitting it would duplicate the visible side's coordinates and let "
            f"one class erase the other during rasterization (UB-29)"
        )


def test_profile_facing_image_right_labels_the_left_side(tmp_path: Path) -> None:
    """The mirror case: nose toward image-right => visible side is the LEFT one."""
    csv_path = _write(tmp_path, "S7.csv", [_profile_row(_PROFILE_ID, nose_x=90)])
    entry = _polygons(csv_path, tmp_path)[_PROFILE_ID]

    for region in _LEFT_NAMES:
        assert region in entry, f"{region!r} should be the visible side"
        assert len(entry[region]) >= 3, f"{region!r} degenerate: {entry[region]}"
    for region in _RIGHT_NAMES:
        assert region not in entry, f"{region!r} is on the occluded side"


def test_lateral_pair_never_shares_coordinates(tmp_path: Path) -> None:
    """The failure mode UB-29 guards against must be impossible in any profile.

    ``create_segmentation_mask`` paints in ``REGION_NAMES`` order, so two
    regions holding identical coordinates means the later one erases the
    earlier and a class is absent from every profile frame.
    """
    rows = [_profile_row(f"L{i}", nose_x=x) for i, x in enumerate((5, 10, 25))]
    rows += [_profile_row(f"R{i}", nose_x=x) for i, x in enumerate((75, 90, 95))]
    entries = _polygons(_write(tmp_path, "S7.csv", rows), tmp_path)

    assert len(entries) == 6
    for image_id, entry in entries.items():
        for left, right in zip(_LEFT_NAMES, _RIGHT_NAMES):
            both = left in entry and right in entry
            assert not both, (
                f"{image_id}: both {left!r} and {right!r} present in a profile "
                f"frame; only one lateral side is visible"
            )


def test_ambiguous_facing_direction_is_skipped_not_guessed(tmp_path: Path) -> None:
    """A nose too near the face's centre yields no mask, rather than a guess.

    Measured on the real corpus, 23 of 4170 profile rows (0.55%) fall in this
    band. Assigning them a side would put a lateral class on the wrong half of
    the face — the very corruption UB-29 exists to prevent — so the row is
    dropped and counted instead.
    """
    rows = [_profile_row("GOOD1", nose_x=10), _profile_row("GOOD2", nose_x=90),
            _profile_row(_PROFILE_ID, nose_x=50)]
    csv_path = _write(tmp_path, "S7.csv", rows)

    entries = _polygons(csv_path, tmp_path, max_skip_fraction=0.5)

    assert _PROFILE_ID not in entries, "ambiguous row must not be emitted"
    assert {"GOOD1", "GOOD2"} == set(entries), "unambiguous rows must survive"


def test_skips_are_counted_and_reported(tmp_path: Path) -> None:
    """Skipped rows are surfaced, never silently dropped (R4)."""
    from codes.generate_boxes_polygons import generate_polygonal_masks

    rows = [_profile_row(f"G{i}", nose_x=10) for i in range(9)]
    rows.append(_profile_row("AMBIG", nose_x=50))
    csv_path = _write(tmp_path, "S7.csv", rows)

    report = generate_polygonal_masks(
        str(csv_path), str(tmp_path / "out.json"), max_skip_fraction=0.5
    )

    assert report["written"] == 9
    assert report["skipped"] == 1
    assert "facing direction ambiguous" in report["reasons"]


def test_systemic_skip_rate_raises(tmp_path: Path) -> None:
    """A wholesale annotation failure must raise, not write a thin file (R4).

    A handful of damaged rows is data hygiene; losing most of a subject is a
    defect, and a silently shrunken mask file would hide it.
    """
    rows = [_profile_row(f"A{i}", nose_x=50) for i in range(8)]
    rows += [_profile_row(f"G{i}", nose_x=10) for i in range(2)]
    csv_path = _write(tmp_path, "S7.csv", rows)

    with pytest.raises(ValueError, match="systemic"):
        _polygons(csv_path, tmp_path)


def test_frontal_only_file_is_unchanged(tmp_path: Path) -> None:
    """Regression guard: the 73-landmark path keeps its exact previous output.

    Frontal frames were always resolved correctly, and the reported run's
    frontal masks must stay reproducible.
    """
    csv_path = _write(tmp_path, "S6.csv", [_frontal_row(_FRONTAL_ID)])
    frontal = _polygons(csv_path, tmp_path)[_FRONTAL_ID]

    assert set(frontal) == set(LANDMARK_MAPPINGS_73)
    for region, indices in LANDMARK_MAPPINGS_73.items():
        assert len(frontal[region]) == len(indices), (
            f"frontal {region!r} changed: {len(frontal[region])} points, "
            f"expected {len(indices)}"
        )


def test_profile_emits_every_non_lateral_region(tmp_path: Path) -> None:
    """All regions except the occluded lateral pair must be present."""
    csv_path = _write(tmp_path, "S7.csv", [_profile_row(_PROFILE_ID, nose_x=10)])
    entry = _polygons(csv_path, tmp_path)[_PROFILE_ID]

    non_lateral = set(LANDMARK_MAPPINGS_43) - set(_LEFT_NAMES) - set(_RIGHT_NAMES)
    assert non_lateral <= set(entry), (
        f"missing non-lateral regions: {sorted(non_lateral - set(entry))}"
    )
    assert len(entry) == len(non_lateral) + len(_RIGHT_NAMES)


def test_unknown_landmark_count_raises(tmp_path: Path) -> None:
    """An unrecognised landmark count must raise, not silently fall back (R4).

    The old code printed a warning and used the 73-point mapping anyway —
    exactly the silent-default pattern that turns a defect into a lie.
    """
    row = _frontal_row("R0003")
    for i in range(55, _SCHEMA_POINTS):
        row[f"x{i}"] = None
        row[f"y{i}"] = None
    csv_path = _write(tmp_path, "S8.csv", [row])

    with pytest.raises(ValueError, match="55"):
        generate_polygonal_masks(str(csv_path), str(tmp_path / "out.json"))
