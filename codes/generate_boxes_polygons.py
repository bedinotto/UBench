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

def generate_polygonal_masks(input_csv, output_json):
    """
    Generates polygonal masks for different facial regions based on landmark coordinates.

    Args:
        input_csv (str): Path to the input CSV file (e.g., S1.csv).
        output_json (str): Path to the output JSON file for polygonal masks.
    """
    df = pd.read_csv(input_csv)
    polygons = {}

    # Read the header separately to determine the number of landmark points
    with open(input_csv, 'r') as f:
        header = f.readline().strip().split(',')

    # Calculate the number of landmark points (e.g., 43 or 73)
    num_points = sum(1 for col in header if col.startswith('x') and col[1:].isdigit())

    if num_points == 73:
        current_landmark_mappings = LANDMARK_MAPPINGS_73
    elif num_points == 43:
        current_landmark_mappings = LANDMARK_MAPPINGS_43
    else:
        print(f"Warning: Unexpected number of landmark points ({num_points}). Using 73-point mapping as default.")
        current_landmark_mappings = LANDMARK_MAPPINGS_73 # Fallback to 73-point if unexpected

    for index, row in df.iterrows():
        image_id = row["ID"]
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