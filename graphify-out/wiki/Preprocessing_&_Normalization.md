# Preprocessing & Normalization

> 12 nodes · cohesion 0.27

## Key Concepts

- **preprocess_data.py** (8 connections) — `codes/preprocess_data.py`
- **preprocess_thermal_image()** (6 connections) — `codes/utils.py`
- **preprocess_all_data()** (5 connections) — `codes/preprocess_data.py`
- **utils.py** (5 connections) — `codes/utils.py`
- **preprocess_mask()** (5 connections) — `codes/utils.py`
- **normalize_thermal()** (4 connections) — `codes/utils.py`
- **ndarray** (3 connections)
- **Offline Data Preprocessing Script ================================= Pre-computes** (1 connections) — `codes/preprocess_data.py`
- **Utilities for Data Processing ============================= Shared functions for** (1 connections) — `codes/utils.py`
- **Normalize thermal image to [0, 1] range using min-max scaling** (1 connections) — `codes/utils.py`
- **Apply standard preprocessing: normalization and resizing.** (1 connections) — `codes/utils.py`
- **Resize mask image using nearest neighbor interpolation.** (1 connections) — `codes/utils.py`

## Relationships

- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (4 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (1 shared connections)

## Source Files

- `codes/preprocess_data.py`
- `codes/utils.py`

## Audit Trail

- EXTRACTED: 41 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*