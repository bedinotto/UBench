# Swin-UNet++ & Checkpoint Recovery

> 40 nodes · cohesion 0.06

## Key Concepts

- **unified_training.py** (16 connections) — `codes/unified_training.py`
- **main()** (10 connections) — `codes/inference_comparison.py`
- **UNet** (9 connections) — `codes/unet_v2.py`
- **CombinedLoss** (8 connections) — `codes/unified_training.py`
- **.__init__()** (7 connections) — `codes/unified_training.py`
- **DiceLoss** (6 connections) — `codes/unified_training.py`
- **SwinUNetPlusPlus** (5 connections) — `codes/swin_unet_plus_plus.py`
- **TransUNet** (5 connections) — `codes/transunet.py`
- **calculate_iou()** (5 connections) — `codes/unified_training.py`
- **.__init__()** (5 connections) — `codes/unified_training.py`
- **get_latest_checkpoint()** (4 connections) — `codes/inference_comparison.py`
- **test_unet_nan.py** (4 connections) — `codes/tests/test_unet_nan.py`
- **test_unet_nan2.py** (4 connections) — `codes/tests/test_unet_nan2.py`
- **.__init__()** (4 connections) — `codes/unified_training.py`
- **._build_scaler()** (4 connections) — `codes/unified_training.py`
- **test_unet_nan3.py** (3 connections) — `codes/tests/test_unet_nan3.py`
- **test_unet_nan6.py** (3 connections) — `codes/tests/test_unet_nan6.py`
- **Tensor** (3 connections)
- **Path** (2 connections)
- **.__init__()** (2 connections) — `codes/unified_training.py`
- **Module** (2 connections)
- **Search for a model checkpoint across standard locations.** (1 connections) — `codes/inference_comparison.py`
- **Swin-UNet++ architecture with nested dense skip connections** (1 connections) — `codes/swin_unet_plus_plus.py`
- **.forward()** (1 connections) — `codes/swin_unet_plus_plus.py`
- **check()** (1 connections) — `codes/tests/test_unet_nan3.py`
- *... and 15 more nodes in this community*

## Relationships

- [Inference & Model Registry](Inference_%26_Model_Registry.md) (13 shared connections)
- [Benchmarking & Comparison](Benchmarking_%26_Comparison.md) (8 shared connections)
- [Data Loading & K-Fold Splits](Data_Loading_%26_K-Fold_Splits.md) (7 shared connections)
- [Thermal Face Detector (Inference)](Thermal_Face_Detector_%28Inference%29.md) (2 shared connections)
- [Transformer Patch Embedding Blocks](Transformer_Patch_Embedding_Blocks.md) (1 shared connections)
- [CNN + Attention Encoder Blocks](CNN_%2B_Attention_Encoder_Blocks.md) (1 shared connections)
- [NaN Debug Scripts (Ad-hoc)](NaN_Debug_Scripts_%28Ad-hoc%29.md) (1 shared connections)
- [Unified Data Loading Module](Unified_Data_Loading_Module.md) (1 shared connections)

## Source Files

- `codes/inference_comparison.py`
- `codes/swin_unet_plus_plus.py`
- `codes/tests/test_unet_nan.py`
- `codes/tests/test_unet_nan2.py`
- `codes/tests/test_unet_nan3.py`
- `codes/tests/test_unet_nan6.py`
- `codes/transunet.py`
- `codes/unet_v2.py`
- `codes/unified_training.py`

## Audit Trail

- EXTRACTED: 128 (98%)
- INFERRED: 2 (2%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*