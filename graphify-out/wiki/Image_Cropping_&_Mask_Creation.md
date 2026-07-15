# Image Cropping & Mask Creation

> 11 nodes · cohesion 0.20

## Key Concepts

- **.load_thermal_image()** (5 connections) — `codes/unified_data.py`
- **.load_thermal_image_from_tiff()** (4 connections) — `codes/unified_data.py`
- **ndarray** (4 connections)
- **.create_segmentation_mask()** (3 connections) — `codes/unified_data.py`
- **.crop_to_bbox()** (3 connections) — `codes/unified_data.py`
- **.get_tiff_path()** (3 connections) — `codes/unified_data.py`
- **Load thermal image from TIFF file** (1 connections) — `codes/unified_data.py`
- **Get the actual TIFF path for a given sample ID, handling variations** (1 connections) — `codes/unified_data.py`
- **Load thermal image for a given sample ID                  Args:             samp** (1 connections) — `codes/unified_data.py`
- **Crop thermal image to bounding box region** (1 connections) — `codes/unified_data.py`
- **Create segmentation mask from polygon annotations** (1 connections) — `codes/unified_data.py`

## Relationships

- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (5 shared connections)

## Source Files

- `codes/unified_data.py`

## Audit Trail

- EXTRACTED: 27 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*