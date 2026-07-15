# Unified Data Loading Module

> 17 nodes · cohesion 0.13

## Key Concepts

- **unified_data.py** (16 connections) — `codes/unified_data.py`
- **ThermalFaceDataset** (8 connections) — `codes/unified_data.py`
- **create_single_fold_loader()** (7 connections) — `codes/unified_data.py`
- **shutdown_data_loaders()** (7 connections) — `codes/unified_data.py`
- **DataLoader** (3 connections)
- **.__init__()** (3 connections) — `codes/unified_data.py`
- **_raw_to_celsius()** (2 connections) — `codes/unified_data.py`
- **DataFrame** (1 connections)
- **Unified Data Loading Module - Multi-Directory Support ==========================** (1 connections) — `codes/unified_data.py`
- **Convert raw thermal sensor value to degrees Celsius.      Defined at module leve** (1 connections) — `codes/unified_data.py`
- **Unified PyTorch Dataset reading from offline preprocessed arrays** (1 connections) — `codes/unified_data.py`
- **# NOTE: cudnn.deterministic is intentionally NOT set here.** (1 connections) — `codes/unified_data.py`
- **Create DataLoaders for a **single** K-Fold split.      Unlike ``create_kfold_dat** (1 connections) — `codes/unified_data.py`
- **Explicitly shut down DataLoader worker processes and free OS resources.      On** (1 connections) — `codes/unified_data.py`
- **.__getitem__()** (1 connections) — `codes/unified_data.py`
- **.__len__()** (1 connections) — `codes/unified_data.py`
- **Dataset** (1 connections)

## Relationships

- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (7 shared connections)
- [Inference & Model Registry](Inference_%26_Model_Registry.md) (5 shared connections)
- [Benchmarking & Comparison](Benchmarking_%26_Comparison.md) (3 shared connections)
- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (2 shared connections)
- [Preprocessing & Normalization](Preprocessing_%26_Normalization.md) (1 shared connections)
- [Reproducibility Seeding](Reproducibility_Seeding.md) (1 shared connections)
- [Swin-UNet++ & Checkpoint Recovery](Swin-UNet%2B%2B_%26_Checkpoint_Recovery.md) (1 shared connections)

## Source Files

- `codes/unified_data.py`

## Audit Trail

- EXTRACTED: 56 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*