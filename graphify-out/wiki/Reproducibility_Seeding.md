# Reproducibility Seeding

> 4 nodes · cohesion 0.50

## Key Concepts

- **.__init__()** (6 connections) — `codes/main_pipeline.py`
- **seed_everything()** (4 connections) — `codes/unified_data.py`
- **Initialize pipeline                  Args:             models_to_train: List of** (1 connections) — `codes/main_pipeline.py`
- **Set global seeds for reproducibility.** (1 connections) — `codes/unified_data.py`

## Relationships

- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (1 shared connections)
- [Hardware Detection & Optimization](Hardware_Detection_%26_Optimization.md) (1 shared connections)
- [Logging Infrastructure (TeeLogger)](Logging_Infrastructure_%28TeeLogger%29.md) (1 shared connections)
- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (1 shared connections)
- [Inference & Model Registry](Inference_%26_Model_Registry.md) (1 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (1 shared connections)

## Source Files

- `codes/main_pipeline.py`
- `codes/unified_data.py`

## Audit Trail

- EXTRACTED: 12 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*