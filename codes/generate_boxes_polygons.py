import json
from collections import Counter

import pandas as pd

def generate_bounding_boxes(input_csv, output_csv):
    """
    Parses the Sx.csv files to generate bounding boxes for each image ID.

    Args:
        input_csv (str): Path to the S1.csv file.
        output_csv (str): Path to the output CSV file for bounding boxes.
    """
    df = pd.read_csv(input_csv)
    bounding_boxes = []

    for index, row in df.iterrows():
        image_id = row["ID"]
        x_coords = []
        y_coords = []
        for i in range(73):
            if f"x{i}" in row and pd.notna(row[f"x{i}"]):
                x_coords.append(row[f"x{i}"])
            if f"y{i}" in row and pd.notna(row[f"y{i}"]):
                y_coords.append(row[f"y{i}"])

        if x_coords and y_coords:
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            bounding_boxes.append([image_id, min_x, min_y, max_x, max_y])

    bounding_boxes_df = pd.DataFrame(bounding_boxes, columns=["ID", "min_x", "min_y", "max_x", "max_y"])
    bounding_boxes_df.to_csv(output_csv, index=False)
    print(f"Bounding boxes saved to {output_csv}")

LANDMARK_MAPPINGS_73 = {
    "Contorno inferior do Rosto": list(range(0, 17)),
    "Sombrancelha esquerda": list(range(17, 22)),
    "Sombrancelha direita": list(range(22, 27)),
    "Nariz": list(range(27, 36)),
    "Olho esquerdo": list(range(36, 42)),
    "Olho direito": list(range(42, 48)),
    "Boca": list(range(48, 60)),
    "Labios": list(range(60, 68)),
    "Testa": list(range(68, 73)),
}

# The 43-point (profile) scheme. Its two lateral pairs share an index range on
# purpose: a profile view shows only ONE eye and ONE eyebrow, so the scheme
# annotates a single lateral set and leaves the occluded side unannotated
# (UB-29). Which side that is depends on which way the head is turned, and no
# CSV column states it — `_resolve_visible_side` derives it from the geometry
# and `_resolve_landmark_mapping` drops the occluded name, so the two never
# rasterize onto the same pixels.
LANDMARK_MAPPINGS_43 = {
    "Contorno inferior do Rosto": list(range(0, 12)),
    "Sombrancelha esquerda": list(range(12, 16)),
    "Sombrancelha direita": list(range(12, 16)),   # same range: only one is visible
    "Nariz": list(range(16, 22)),
    "Olho esquerdo": list(range(22, 27)),
    "Olho direito": list(range(22, 27)),           # same range: only one is visible
    "Boca": list(range(27, 34)),
    "Labios": list(range(34, 39)),
    "Testa": list(range(39, 43)),
}

_LANDMARK_MAPPINGS = {73: LANDMARK_MAPPINGS_73, 43: LANDMARK_MAPPINGS_43}

# Lateral region names, paired left→right. The dataset's naming convention is
# **image-side**, not the subject's anatomy: verified as "Olho esquerdo" holding
# the smaller x in 99.9% of the real corpus's 4225 frontal rows.
_LATERAL_PAIRS = (
    ("Sombrancelha esquerda", "Sombrancelha direita"),
    ("Olho esquerdo", "Olho direito"),
)

# A profile row whose nose sits within this fraction of the face's horizontal
# midpoint has no reliable facing direction, so it raises rather than guess
# (R4). Measured on the real corpus: 23 of 4170 profile rows (0.55%) fall here.
_AMBIGUOUS_NOSE_BAND = 0.15


def _region_centre_x(row, indices) -> float | None:
    """Mean x of a region's populated landmarks, or ``None`` if it has none."""
    xs = [row[f"x{i}"] for i in indices if pd.notna(row.get(f"x{i}"))]
    return sum(xs) / len(xs) if xs else None


def _resolve_visible_side(row, mapping: dict, source: str, image_id) -> str:
    """Return ``"esquerd"`` or ``"direit"`` — the lateral side actually visible.

    A profile view occludes one side of the face, so the 43-point scheme carries
    a single eye/eyebrow set. The visible side is the one **opposite** the
    nose's horizontal displacement: as the head turns, the receding side's eye
    both narrows and collapses toward the nose, while the approaching side's eye
    stays wide and separated.

    That relationship was measured on the real corpus over 4225 frontal rows,
    binned by head rotation, using two independent signals — relative eye width
    (0.813 / 0.920 / 1.021 / 1.123 / 1.292, nose-far-left to nose-far-right) and
    eye-to-nose distance (0.140 vs 0.269 at far left; 0.275 vs 0.144 at far
    right). Both are monotonic and agree. Combined with the dataset's image-side
    naming convention, this gives:

        nose left of centre  -> visible side is "direit"  (Olho direito)
        nose right of centre -> visible side is "esquerd" (Olho esquerdo)

    This is a *derived and validated* rule, not the dataset's published
    specification; ``data/a13_landmark_side_evidence.py`` reproduces the
    evidence. A published per-side index convention would supersede it.

    Args:
        row: the CSV row being converted.
        mapping: the resolved 43-point mapping.
        source: CSV path, for error messages.
        image_id: image identifier, for error messages.

    Returns:
        ``"esquerd"`` or ``"direit"`` — the stem shared by the visible side's
        eyebrow and eye region names.

    Raises:
        ValueError: if the face has no horizontal extent, if the nose carries no
            landmarks, or if the nose sits too near the face's centre for the
            facing direction to be read reliably (R4).
    """
    xs = [row[f"x{i}"] for i in range(43) if pd.notna(row.get(f"x{i}"))]
    nose_x = _region_centre_x(row, mapping["Nariz"])
    if not xs or nose_x is None:
        raise ValueError(
            f"{source} image {image_id!r}: cannot resolve the visible facial "
            f"side — the row has no nose landmarks or no horizontal extent, so "
            f"the 43-point scheme's single eye/eyebrow set cannot be attributed "
            f"to a side (UB-29)."
        )

    face_min, face_max = min(xs), max(xs)
    width = face_max - face_min
    if width <= 0:
        raise ValueError(
            f"{source} image {image_id!r}: all landmarks share one x coordinate; "
            f"the facing direction is undefined (UB-29)."
        )

    offset = (nose_x - face_min) / width - 0.5
    if abs(offset) < _AMBIGUOUS_NOSE_BAND:
        raise ValueError(
            f"{source} image {image_id!r}: the facing direction is ambiguous — "
            f"the nose sits {offset:+.3f} from the face's horizontal centre, "
            f"inside the +/-{_AMBIGUOUS_NOSE_BAND} band where the visible side "
            f"cannot be read reliably from geometry. A 43-point row annotates "
            f"only the visible eye/eyebrow, so guessing the side would assign "
            f"a lateral class to the wrong half of the face (UB-29). Exclude "
            f"this image or annotate its side explicitly."
        )
    return "direit" if offset < 0 else "esquerd"


def _drop_occluded_side(mapping: dict, visible: str) -> dict:
    """Return *mapping* without the lateral regions on the occluded side.

    The occluded side carries no annotation at all, so omitting it is the honest
    representation: the metric authority already excludes absent classes from
    the macro average. Emitting both names from one index range is what would
    let the later-painted class erase the earlier one (UB-29).
    """
    keep_stems = ("sombrancelha " + visible, "olho " + visible)
    drop = {
        name
        for pair in _LATERAL_PAIRS
        for name in pair
        if not name.lower().startswith(keep_stems)
    }
    return {region: idx for region, idx in mapping.items() if region not in drop}


def _declared_landmark_columns(columns) -> int:
    """Count the ``x{i}`` landmark columns a CSV *declares* in its header."""
    return sum(1 for col in columns if col.startswith("x") and col[1:].isdigit())


def _count_filled_landmarks(row, declared: int) -> int:
    """Count the ``(x{i}, y{i})`` pairs a single row actually populates.

    This — not the header width — is what identifies the annotation scheme of
    an individual image (UB-27). Every ``S{n}.csv`` declares all 73 columns for
    every row; profile rows simply leave indices 43..72 empty.
    """
    return sum(
        1 for i in range(declared)
        if pd.notna(row.get(f"x{i}")) and pd.notna(row.get(f"y{i}"))
    )


def _resolve_landmark_mapping(row, n_filled: int, source: str, image_id) -> dict:
    """Return the region→indices mapping to use for one row.

    For the 73-point (frontal) scheme this is the mapping unchanged. For the
    43-point (profile) scheme the visible facial side is resolved first and the
    occluded side's lateral regions are dropped, so no two regions ever share an
    index range — which is what would let one class erase the other during
    rasterization (UB-29).

    Args:
        row: the CSV row being converted.
        n_filled: landmark pairs actually populated in this row.
        source: CSV path, for error messages.
        image_id: image identifier, for error messages.

    Returns:
        The region→indices mapping for this row, with no duplicated ranges.

    Raises:
        ValueError: if the landmark count is not a known scheme, if a duplicated
            index range survives side resolution, or if the facing direction is
            unreadable. All are silent-corruption paths, so they raise (R4).
    """
    mapping = _LANDMARK_MAPPINGS.get(n_filled)
    if mapping is None:
        raise ValueError(
            f"{source} image {image_id!r}: {n_filled} landmark pairs is not a "
            f"known annotation scheme (expected one of "
            f"{sorted(_LANDMARK_MAPPINGS)}). Falling back to the 73-point "
            f"mapping would rasterize regions from indices the row does not "
            f"have (UB-27); fix the annotation or add the scheme explicitly."
        )

    if n_filled == 43:
        visible = _resolve_visible_side(row, mapping, source, image_id)
        mapping = _drop_occluded_side(mapping, visible)

    by_range: dict[tuple, str] = {}
    for region, indices in mapping.items():
        key = tuple(indices)
        if key in by_range:
            raise ValueError(
                f"{source} image {image_id!r}: the {n_filled}-point mapping "
                f"still assigns the same landmark indices {list(key)} to both "
                f"{by_range[key]!r} and {region!r} after resolving the visible "
                f"side. Rasterization paints regions in REGION_NAMES order, so "
                f"the later class would silently erase the earlier one (UB-29)."
            )
        by_range[key] = region
    return mapping

def generate_polygonal_masks(input_csv, output_json, max_skip_fraction: float = 0.05):
    """
    Generates polygonal masks for different facial regions based on landmark coordinates.

    The annotation scheme is resolved per IMAGE, not per file (UB-27): every
    ``S{n}.csv`` declares all 73 x/y columns, and profile rows simply leave
    indices 43..72 empty. Selecting from the header made the 43-point mapping
    unreachable and rasterized profile rows through the 73-point one, so
    Boca/Labios/Testa got no points at all, Olho direito got a single point and
    Olho esquerdo got six points belonging to Labios/Testa.

    Rows the schemes cannot describe are **skipped and counted**, never coerced
    into a scheme they do not match. Two cases occur in the real corpus: a
    landmark count matching no known scheme (59 of 8454 rows, damaged or partial
    annotations) and a profile row whose facing direction is unreadable (23
    rows, nose too near the face's centre — see :func:`_resolve_visible_side`).
    Both are reported per file, and a skip rate above *max_skip_fraction* raises
    instead: a handful of damaged rows is data hygiene, but a systemic failure
    must not pass silently as a thin output file (R4).

    Args:
        input_csv (str): Path to the input CSV file (e.g., S1.csv).
        output_json (str): Path to the output JSON file for polygonal masks.
        max_skip_fraction (float): Fraction of rows that may be skipped before
            this raises. Defaults to 0.05 (measured rate is ~0.01).

    Returns:
        dict: ``{"written": int, "skipped": int, "reasons": Counter}``.

    Raises:
        ValueError: if the skipped fraction exceeds *max_skip_fraction*.
    """
    df = pd.read_csv(input_csv)
    polygons = {}
    skipped: Counter = Counter()

    declared = _declared_landmark_columns(df.columns)

    for _index, row in df.iterrows():
        image_id = row["ID"]
        n_filled = _count_filled_landmarks(row, declared)
        try:
            current_landmark_mappings = _resolve_landmark_mapping(
                row, n_filled, input_csv, image_id
            )
        except ValueError as exc:
            # Skip only rows no scheme can describe; the reason is retained and
            # reported, and an excessive rate raises below.
            kind = ("facing direction ambiguous" if "ambiguous" in str(exc)
                    else f"unsupported landmark count ({n_filled})")
            skipped[kind] += 1
            continue

        polygons[image_id] = {}
        for region, indices in current_landmark_mappings.items():
            region_points = []
            for i in indices:
                x_col = f"x{i}"
                y_col = f"y{i}"
                if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    region_points.append([int(row[x_col]), int(row[y_col])])
            if region_points:
                polygons[image_id][region] = region_points

    n_total = len(df)
    n_skipped = sum(skipped.values())
    if n_total and n_skipped / n_total > max_skip_fraction:
        raise ValueError(
            f"{input_csv}: skipped {n_skipped} of {n_total} rows "
            f"({n_skipped / n_total:.1%}), above the {max_skip_fraction:.0%} "
            f"limit — this is a systemic annotation problem, not a few damaged "
            f"rows. Reasons: {dict(skipped)}. Writing a thin mask file would "
            f"silently shrink the dataset (R4)."
        )

    with open(output_json, "w") as f:
        json.dump(polygons, f, indent=4)
    print(f"Polygonal masks saved to {output_json} "
          f"({len(polygons)} images, {n_skipped} skipped)")
    if skipped:
        for kind, n in skipped.most_common():
            print(f"    skipped {n}x: {kind}")
    return {"written": len(polygons), "skipped": n_skipped, "reasons": skipped}


def generate_all(data_dir: str):
    """
    Generate bounding boxes and polygonal masks for all Sx.csv files
    in the given data directory.

    Args:
        data_dir: Path to the data directory containing Sx.csv files.
    """
    from pathlib import Path

    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("S*.csv"))
    valid_files = [
        f for f in csv_files
        if f.stem.startswith("S") and f.stem[1:].isdigit()
    ]
    valid_files.sort(key=lambda x: int(x.stem[1:]))

    if not valid_files:
        print(f"No Sx.csv files found in {data_dir}")
        return

    for csv_path in valid_files:
        dataset_name = csv_path.stem
        bbox_path = data_path / f"{dataset_name}_bounding_boxes.csv"
        polygon_path = data_path / f"{dataset_name}_polygonal_masks.json"

        print(f"Processing {dataset_name}...")
        generate_bounding_boxes(str(csv_path), str(bbox_path))
        generate_polygonal_masks(str(csv_path), str(polygon_path))


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Resolve project root and data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    generate_all(str(data_dir))