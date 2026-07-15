# Benchmarking & Comparison

> 46 nodes · cohesion 0.07

## Key Concepts

- **UnifiedTrainer** (17 connections) — `codes/unified_training.py`
- **run_benchmark()** (12 connections) — `codes/benchmark_models.py`
- **benchmark_models.py** (11 connections) — `codes/benchmark_models.py`
- **_safe_filename()** (11 connections) — `codes/unified_training.py`
- **ModelBenchmark** (10 connections) — `codes/benchmark_models.py`
- **calculate_dice_score()** (8 connections) — `codes/unified_training.py`
- **.train()** (8 connections) — `codes/unified_training.py`
- **.benchmark_model()** (6 connections) — `codes/benchmark_models.py`
- **.compare_models()** (6 connections) — `codes/benchmark_models.py`
- **Path** (6 connections)
- **.save_checkpoint()** (6 connections) — `codes/unified_training.py`
- **.load_model()** (5 connections) — `codes/benchmark_models.py`
- **._checkpoint_path()** (5 connections) — `codes/unified_training.py`
- **._find_latest_checkpoint()** (5 connections) — `codes/unified_training.py`
- **._create_comparison_plots()** (4 connections) — `codes/benchmark_models.py`
- **._generate_report()** (4 connections) — `codes/benchmark_models.py`
- **DataFrame** (4 connections)
- **.load_checkpoint()** (4 connections) — `codes/unified_training.py`
- **.plot_training_history()** (4 connections) — `codes/unified_training.py`
- **.save_metrics()** (4 connections) — `codes/unified_training.py`
- **.validate()** (4 connections) — `codes/unified_training.py`
- **Module** (3 connections)
- **.train_epoch()** (3 connections) — `codes/unified_training.py`
- **.__init__()** (2 connections) — `codes/benchmark_models.py`
- **DataLoader** (2 connections)
- *... and 21 more nodes in this community*

## Relationships

- [Swin-UNet++ & Checkpoint Recovery](Swin-UNet%2B%2B_%26_Checkpoint_Recovery.md) (8 shared connections)
- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (6 shared connections)
- [Inference & Model Registry](Inference_%26_Model_Registry.md) (4 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (3 shared connections)
- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (2 shared connections)

## Source Files

- `codes/benchmark_models.py`
- `codes/unified_training.py`

## Audit Trail

- EXTRACTED: 172 (98%)
- INFERRED: 3 (2%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*