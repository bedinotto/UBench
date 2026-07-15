# Thermal Face Detector (Inference)

> 14 nodes · cohesion 0.19

## Key Concepts

- **ThermalFaceDetector** (11 connections) — `codes/unified_training.py`
- **ndarray** (4 connections)
- **.normalize_thermal()** (4 connections) — `codes/unified_training.py`
- **.predict()** (4 connections) — `codes/unified_training.py`
- **.get_stats_info()** (3 connections) — `codes/unified_training.py`
- **.visualize_predictions()** (3 connections) — `codes/unified_training.py`
- **test_detector_initialization_and_prediction()** (2 connections) — `codes/tests/test_suite.py`
- **.print_stats_report()** (2 connections) — `codes/unified_training.py`
- **Inference class for detecting facial regions in thermal images** (1 connections) — `codes/unified_training.py`
- **Normalize thermal image to [0, 1] range** (1 connections) — `codes/unified_training.py`
- **Predict facial regions in a thermal image          Args:             thermal_ima** (1 connections) — `codes/unified_training.py`
- **Calculate statistics information for each region using original thermal data in** (1 connections) — `codes/unified_training.py`
- **Visualize predicted regions** (1 connections) — `codes/unified_training.py`
- **Print a formatted report of thermal statistics for all regions** (1 connections) — `codes/unified_training.py`

## Relationships

- [Inference & Model Registry](Inference_%26_Model_Registry.md) (2 shared connections)
- [Swin-UNet++ & Checkpoint Recovery](Swin-UNet%2B%2B_%26_Checkpoint_Recovery.md) (2 shared connections)
- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (1 shared connections)

## Source Files

- `codes/tests/test_suite.py`
- `codes/unified_training.py`

## Audit Trail

- EXTRACTED: 38 (97%)
- INFERRED: 1 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*