# Training Pipeline Orchestrator

> 20 nodes · cohesion 0.15

## Key Concepts

- **Pipeline** (17 connections) — `codes/main_pipeline.py`
- **.run()** (7 connections) — `codes/main_pipeline.py`
- **.train_model()** (7 connections) — `codes/main_pipeline.py`
- **main()** (5 connections) — `codes/main_pipeline.py`
- **._get_fold_loaders()** (5 connections) — `codes/main_pipeline.py`
- **.run_benchmark()** (5 connections) — `codes/main_pipeline.py`
- **._cleanup_fold_resources()** (4 connections) — `codes/main_pipeline.py`
- **.train_all_models()** (4 connections) — `codes/main_pipeline.py`
- **.load_shared_data()** (3 connections) — `codes/main_pipeline.py`
- **.print_summary()** (3 connections) — `codes/main_pipeline.py`
- **Load annotations once (lightweight) — DataLoaders are created lazily.** (1 connections) — `codes/main_pipeline.py`
- **Create DataLoaders for a single model + fold (lazy, on-demand).          Uses ``** (1 connections) — `codes/main_pipeline.py`
- **Shut down DataLoader workers and free GPU memory between runs.** (1 connections) — `codes/main_pipeline.py`
- **Train a dynamic model from the registry on a specific fold** (1 connections) — `codes/main_pipeline.py`
- **Train all selected models sequentially over all folds** (1 connections) — `codes/main_pipeline.py`
- **Run comprehensive benchmark on all trained models, aggregating across all folds** (1 connections) — `codes/main_pipeline.py`
- **Execute the complete pipeline** (1 connections) — `codes/main_pipeline.py`
- **Print training summary averaged across folds** (1 connections) — `codes/main_pipeline.py`
- **Main entry point with global error catcher for crash-safe operation.** (1 connections) — `codes/main_pipeline.py`
- **Main training pipeline orchestrator** (1 connections) — `codes/main_pipeline.py`

## Relationships

- [Inference & Model Registry](Inference_%26_Model_Registry.md) (4 shared connections)
- [Hardware Detection & Optimization](Hardware_Detection_%26_Optimization.md) (2 shared connections)
- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (2 shared connections)
- [Benchmarking & Comparison](Benchmarking_%26_Comparison.md) (2 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (2 shared connections)
- [Logging Infrastructure (TeeLogger)](Logging_Infrastructure_%28TeeLogger%29.md) (1 shared connections)
- [Reproducibility Seeding](Reproducibility_Seeding.md) (1 shared connections)

## Source Files

- `codes/main_pipeline.py`

## Audit Trail

- EXTRACTED: 65 (93%)
- INFERRED: 5 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*