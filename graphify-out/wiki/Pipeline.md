# Pipeline

> God node · 17 connections · `codes/main_pipeline.py`

**Community:** [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md)

## Connections by Relation

### calls
- main() `EXTRACTED`

### contains
- main_pipeline.py `EXTRACTED`

### method
- .run() `EXTRACTED`
- .train_model() `EXTRACTED`
- .__init__() `EXTRACTED`
- ._get_fold_loaders() `EXTRACTED`
- .run_benchmark() `EXTRACTED`
- ._cleanup_fold_resources() `EXTRACTED`
- .train_all_models() `EXTRACTED`
- .load_shared_data() `EXTRACTED`
- .print_summary() `EXTRACTED`

### rationale_for
- Main training pipeline orchestrator `EXTRACTED`

### uses
- [Config](Config.md) `INFERRED`
- [MultiDirectoryDataLoader](MultiDirectoryDataLoader.md) `INFERRED`
- [UnifiedTrainer](UnifiedTrainer.md) `INFERRED`
- [HardwareProfile](HardwareProfile.md) `INFERRED`
- [TeeLogger](TeeLogger.md) `INFERRED`

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*