# Data Loading & K-Fold Splits

> 21 nodes · cohesion 0.12

## Key Concepts

- **Config** (30 connections) — `codes/unified_data.py`
- **MultiDirectoryDataLoader** (18 connections) — `codes/unified_data.py`
- **create_kfold_data_loaders()** (10 connections) — `codes/unified_data.py`
- **.__init__()** (4 connections) — `codes/unified_data.py`
- **.load_annotations()** (4 connections) — `codes/unified_data.py`
- **test_discovery_and_loading()** (3 connections) — `codes/tests/test_suite.py`
- **._create_output_dirs()** (3 connections) — `codes/unified_data.py`
- **._validate_paths()** (3 connections) — `codes/unified_data.py`
- **.discover_datasets()** (3 connections) — `codes/unified_data.py`
- **.load_dataset_annotations()** (3 connections) — `codes/unified_data.py`
- **mock_config()** (2 connections) — `codes/tests/test_suite.py`
- **.__init__()** (2 connections) — `codes/unified_data.py`
- **Validate required data paths exist** (1 connections) — `codes/unified_data.py`
- **Create output directories if they don't exist** (1 connections) — `codes/unified_data.py`
- **Data loader that automatically discovers and loads from multiple dataset directo** (1 connections) — `codes/unified_data.py`
- **Discover all Sx directories in data folder                  Returns:** (1 connections) — `codes/unified_data.py`
- **Load annotations for a specific dataset directory                  Args:** (1 connections) — `codes/unified_data.py`
- **Load annotations from all discovered dataset directories** (1 connections) — `codes/unified_data.py`
- **Create K-Fold training and validation data loaders from preprocessed offline arr** (1 connections) — `codes/unified_data.py`
- **Unified configuration for all models** (1 connections) — `codes/unified_data.py`
- **Initialize and validate paths                  Args:             output_dir: Ove** (1 connections) — `codes/unified_data.py`

## Relationships

- [Inference & Model Registry](Inference_%26_Model_Registry.md) (9 shared connections)
- [Swin-UNet++ & Checkpoint Recovery](Swin-UNet%2B%2B_%26_Checkpoint_Recovery.md) (7 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (7 shared connections)
- [Benchmarking & Comparison](Benchmarking_%26_Comparison.md) (6 shared connections)
- [Image Cropping & Mask Creation](Image_Cropping_%26_Mask_Creation.md) (5 shared connections)
- [Preprocessing & Normalization](Preprocessing_%26_Normalization.md) (4 shared connections)
- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (2 shared connections)
- [Reproducibility Seeding](Reproducibility_Seeding.md) (1 shared connections)
- [Thermal Face Detector (Inference)](Thermal_Face_Detector_%28Inference%29.md) (1 shared connections)

## Source Files

- `codes/tests/test_suite.py`
- `codes/unified_data.py`

## Audit Trail

- EXTRACTED: 87 (93%)
- INFERRED: 7 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*