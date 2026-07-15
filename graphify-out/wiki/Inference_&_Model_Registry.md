# Inference & Model Registry

> 23 nodes · cohesion 0.15

## Key Concepts

- **main_pipeline.py** (24 connections) — `codes/main_pipeline.py`
- **test_suite.py** (16 connections) — `codes/tests/test_suite.py`
- **swin_unet_plus_plus.py** (14 connections) — `codes/swin_unet_plus_plus.py`
- **unet_v2.py** (14 connections) — `codes/unet_v2.py`
- **inference_comparison.py** (13 connections) — `codes/inference_comparison.py`
- **transunet.py** (12 connections) — `codes/transunet.py`
- **model_registry.py** (9 connections) — `codes/model_registry.py`
- **create_model()** (9 connections) — `codes/model_registry.py`
- **register_model()** (5 connections) — `codes/model_registry.py`
- **get_registered_models()** (2 connections) — `codes/model_registry.py`
- **mock_model_path()** (2 connections) — `codes/tests/test_suite.py`
- **test_model_shapes()** (2 connections) — `codes/tests/test_suite.py`
- **Inference & Model Comparison Script =================================== Loads th** (1 connections) — `codes/inference_comparison.py`
- **Main Training Pipeline Orchestrator =================================== Manages** (1 connections) — `codes/main_pipeline.py`
- **Module** (1 connections)
- **Model Registry ============== Dynamically load and register model architectures.** (1 connections) — `codes/model_registry.py`
- **Decorator to register a model class** (1 connections) — `codes/model_registry.py`
- **Return a list of registered model names** (1 connections) — `codes/model_registry.py`
- **Instantiate a model by name** (1 connections) — `codes/model_registry.py`
- **Thermal Facial Region Detection System - Swin-UNet++ ===========================** (1 connections) — `codes/swin_unet_plus_plus.py`
- **mock_data_dir()** (1 connections) — `codes/tests/test_suite.py`
- **Thermal Facial Region Detection System - TransUNet =============================** (1 connections) — `codes/transunet.py`
- **Thermal Facial Region Detection System - U-Net Model ===========================** (1 connections) — `codes/unet_v2.py`

## Relationships

- [Swin-UNet++ & Checkpoint Recovery](Swin-UNet%2B%2B_%26_Checkpoint_Recovery.md) (13 shared connections)
- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (9 shared connections)
- [Transformer Patch Embedding Blocks](Transformer_Patch_Embedding_Blocks.md) (7 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (5 shared connections)
- [CNN + Attention Encoder Blocks](CNN_%2B_Attention_Encoder_Blocks.md) (5 shared connections)
- [Benchmarking & Comparison](Benchmarking_%26_Comparison.md) (4 shared connections)
- [Hardware Detection & Optimization](Hardware_Detection_%26_Optimization.md) (4 shared connections)
- [Training Pipeline Orchestrator](Training_Pipeline_Orchestrator.md) (4 shared connections)
- [NaN Debug Scripts (Ad-hoc)](NaN_Debug_Scripts_%28Ad-hoc%29.md) (3 shared connections)
- [Logging Infrastructure (TeeLogger)](Logging_Infrastructure_%28TeeLogger%29.md) (2 shared connections)
- [Thermal Face Detector (Inference)](Thermal_Face_Detector_%28Inference%29.md) (2 shared connections)
- [Reproducibility Seeding](Reproducibility_Seeding.md) (1 shared connections)

## Source Files

- `codes/inference_comparison.py`
- `codes/main_pipeline.py`
- `codes/model_registry.py`
- `codes/swin_unet_plus_plus.py`
- `codes/tests/test_suite.py`
- `codes/transunet.py`
- `codes/unet_v2.py`

## Audit Trail

- EXTRACTED: 133 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*