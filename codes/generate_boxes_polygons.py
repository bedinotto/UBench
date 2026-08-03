import pandas as pd
import json

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

# WARNING (UB-27/UB-29): the two lateral pairs below are double-booked — both
# eyebrows claim indices 12-15 and both eyes claim 22-26. All 43 indices are
# already consumed with no gaps, which means the real Charlotte-ThermalFace
# 43-point (profile) scheme annotates only the *visible* side of the face, and
# which side that is is not encoded in the data. Rasterizing both names from one
# index range makes the later-painted side (right, per REGION_NAMES order in
# create_segmentation_mask) erase the earlier one, so one class would be absent
# from every profile frame. `_resolve_landmark_mapping` therefore REFUSES this
# mapping instead of emitting knowingly-corrupt masks (R4). Supplying the real
# per-side index ranges is what unblocks profile annotation.
LANDMARK_MAPPINGS_43 = {
    "Contorno inferior do Rosto": list(range(0, 12)),
    "Sombrancelha esquerda": list(range(12, 16)),
    "Sombrancelha direita": list(range(12, 16)), # Assuming same range as left eyebrow for 43 points if not distinct
    "Nariz": list(range(16, 22)),
    "Olho esquerdo": list(range(22, 27)),
    "Olho direito": list(range(22, 27)), # Assuming same range as left eye for 43 points if not distinct
    "Boca": list(range(27, 34)),
    "Labios": list(range(34, 39)),
    "Testa": list(range(39, 43)),
}

_LANDMARK_MAPPINGS = {73: LANDMARK_MAPPINGS_73, 43: LANDMARK_MAPPINGS_43}


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


def _resolve_landmark_mapping(n_filled: int, source: str, image_id) -> dict:
    """Return the region→indices mapping for a row with ``n_filled`` landmarks.

    Args:
        n_filled: landmark pairs actually populated in this row.
        source: CSV path, for error messages.
        image_id: image identifier, for error messages.

    Returns:
        The mapping for this row's annotation scheme.

    Raises:
        ValueError: if the landmark count is not a known scheme, or if the
            resolved mapping assigns the same index range to two different
            regions — both are silent-corruption paths, so they raise (R4).
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

    by_range: dict[tuple, str] = {}
    for region, indices in mapping.items():
        key = tuple(indices)
        if key in by_range:
            raise ValueError(
                f"{source} image {image_id!r}: the {n_filled}-point mapping "
                f"assigns the same landmark indices {list(key)} to both "
                f"{by_range[key]!r} and {region!r}. Rasterization paints "
                f"regions in REGION_NAMES order, so the later class would "
                f"silently erase the earlier one and one class would be absent "
                f"from every {n_filled}-point frame (UB-29). Supply the real "
                f"per-side index ranges for the {n_filled}-point scheme in "
                f"LANDMARK_MAPPINGS_{n_filled} before generating these masks."
            )
        by_range[key] = region
    return mapping

def generate_polygonal_masks(input_csv, output_json):
    """
    Generates polygonal masks for different facial regions based on landmark coordinates.

    Args:
        input_csv (str): Path to the input CSV file (e.g., S1.csv).
        output_json (str): Path to the output JSON file for polygonal masks.
    """
    df = pd.read_csv(input_csv)
    polygons = {}

    # The annotation scheme is a property of each IMAGE, not of the file
    # (UB-27): every S{n}.csv declares all 73 x/y columns, and profile rows
    # simply leave indices 43..72 empty. Selecting from the header made the
    # 43-point mapping unreachable and rasterized profile rows through the
    # 73-point one, so Boca/Labios/Testa got no points at all (class absent),
    # Olho direito got a single point (degenerate polygon) and Olho esquerdo
    # got six points belonging to Labios/Testa (displaced blob).
    declared = _declared_landmark_columns(df.columns)

    for index, row in df.iterrows():
        image_id = row["ID"]
        n_filled = _count_filled_landmarks(row, declared)
        current_landmark_mappings = _resolve_landmark_mapping(
            n_filled, input_csv, image_id
        )
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

    with open(output_json, "w") as f:
        json.dump(polygons, f, indent=4)
    print(f"Polygonal masks saved to {output_json}")


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