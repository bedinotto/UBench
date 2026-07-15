# MultiDirectoryDataLoader

> God node · 18 connections · `codes/unified_data.py`

**Community:** [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md)

## Connections by Relation

### calls
- preprocess_all_data() `EXTRACTED`
- test_discovery_and_loading() `EXTRACTED`

### contains
- unified_data.py `EXTRACTED`

### imports
- main_pipeline.py `EXTRACTED`
- test_suite.py `EXTRACTED`
- preprocess_data.py `EXTRACTED`

### method
- .load_thermal_image() `EXTRACTED`
- .load_annotations() `EXTRACTED`
- .load_thermal_image_from_tiff() `EXTRACTED`
- .create_segmentation_mask() `EXTRACTED`
- .crop_to_bbox() `EXTRACTED`
- .discover_datasets() `EXTRACTED`
- .get_tiff_path() `EXTRACTED`
- .load_dataset_annotations() `EXTRACTED`
- .__init__() `EXTRACTED`

### rationale_for
- Data loader that automatically discovers and loads from multiple dataset directo `EXTRACTED`

### references
- create_kfold_data_loaders() `EXTRACTED`

### uses
- [Pipeline](Pipeline.md) `INFERRED`

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*