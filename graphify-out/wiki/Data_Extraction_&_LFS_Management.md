# Data Extraction & LFS Management

> 26 nodes · cohesion 0.13

## Key Concepts

- **extract_data.py** (13 connections) — `codes/extract_data.py`
- **extract_all_data()** (11 connections) — `codes/extract_data.py`
- **_discover_zip_files()** (5 connections) — `codes/extract_data.py`
- **_extract_zip()** (5 connections) — `codes/extract_data.py`
- **_generate_annotations()** (5 connections) — `codes/extract_data.py`
- **_is_lfs_pointer()** (5 connections) — `codes/extract_data.py`
- **_process_extracted_contents()** (5 connections) — `codes/extract_data.py`
- **generate_bounding_boxes()** (5 connections) — `codes/generate_boxes_polygons.py`
- **generate_polygonal_masks()** (5 connections) — `codes/generate_boxes_polygons.py`
- **Path** (4 connections)
- **generate_all()** (4 connections) — `codes/generate_boxes_polygons.py`
- **check_data_status()** (3 connections) — `codes/extract_data.py`
- **_is_data_already_extracted()** (3 connections) — `codes/extract_data.py`
- **generate_boxes_polygons.py** (3 connections) — `codes/generate_boxes_polygons.py`
- **Data Extraction Utility ======================= Automatically extracts all ZIP f** (1 connections) — `codes/extract_data.py`
- **Process the extracted contents: find CSVs and TIFF directories,     move them to** (1 connections) — `codes/extract_data.py`
- **Generate bounding boxes and polygonal masks from the CSV files     using generat** (1 connections) — `codes/extract_data.py`
- **Main function: extract all ZIP data from requirements/ into data/.      Args:** (1 connections) — `codes/extract_data.py`
- **Print a summary of the current data status without extracting.** (1 connections) — `codes/extract_data.py`
- **Check if data/ already contains at least one Sx directory with TIFF files.     R** (1 connections) — `codes/extract_data.py`
- **Check if a file is a Git LFS pointer instead of the actual data.** (1 connections) — `codes/extract_data.py`
- **Discover all .zip files in requirements/.** (1 connections) — `codes/extract_data.py`
- **Extract a single ZIP file to the destination directory.     Returns True on succ** (1 connections) — `codes/extract_data.py`
- **Generate bounding boxes and polygonal masks for all Sx.csv files     in the give** (1 connections) — `codes/generate_boxes_polygons.py`
- **Parses the Sx.csv files to generate bounding boxes for each image ID.      Args:** (1 connections) — `codes/generate_boxes_polygons.py`
- *... and 1 more nodes in this community*

## Relationships

- [Logging Infrastructure (TeeLogger)](Logging_Infrastructure_%28TeeLogger%29.md) (3 shared connections)
- [Environment Setup & CUDA Management](Environment_Setup_%26_CUDA_Management.md) (1 shared connections)

## Source Files

- `codes/extract_data.py`
- `codes/generate_boxes_polygons.py`

## Audit Trail

- EXTRACTED: 88 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*