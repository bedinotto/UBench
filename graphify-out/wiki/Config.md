# Config

> God node · 30 connections · `codes/unified_data.py`

**Community:** [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md)

## Connections by Relation

### calls
- main() `EXTRACTED`
- .__init__() `EXTRACTED`
- preprocess_all_data() `EXTRACTED`
- test_discovery_and_loading() `EXTRACTED`
- mock_config() `EXTRACTED`

### contains
- unified_data.py `EXTRACTED`

### imports
- main_pipeline.py `EXTRACTED`
- test_suite.py `EXTRACTED`
- unified_training.py `EXTRACTED`
- inference_comparison.py `EXTRACTED`
- benchmark_models.py `EXTRACTED`
- preprocess_data.py `EXTRACTED`

### method
- .__init__() `EXTRACTED`
- ._create_output_dirs() `EXTRACTED`
- ._validate_paths() `EXTRACTED`

### rationale_for
- Unified configuration for all models `EXTRACTED`

### references
- [run_benchmark()](run_benchmark%28%29.md) `EXTRACTED`
- create_kfold_data_loaders() `EXTRACTED`
- create_single_fold_loader() `EXTRACTED`
- .__init__() `EXTRACTED`
- .__init__() `EXTRACTED`
- .__init__() `EXTRACTED`
- .__init__() `EXTRACTED`
- .__init__() `EXTRACTED`

### uses
- [Pipeline](Pipeline.md) `INFERRED`
- [UnifiedTrainer](UnifiedTrainer.md) `INFERRED`
- ThermalFaceDetector `INFERRED`
- ModelBenchmark `INFERRED`
- CombinedLoss `INFERRED`
- DiceLoss `INFERRED`

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*