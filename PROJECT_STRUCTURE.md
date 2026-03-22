# Project Structure Overview

## Directory Layout

```
thermal-face-detection/
│
├── run.sh                           # ← Linux/Mac entry point
├── run.bat                          # ← Windows entry point  
├── requirements.txt                 # ← Python dependencies
├── config.yaml                      # ← User configuration
├── README.md                        # ← Documentation
│
├── codes/                           # ← ALL PYTHON CODE HERE
│   ├── setup.py                     # Environment setup script
│   ├── main_pipeline.py             # Main orchestrator
│   ├── hardware_detector.py         # GPU detection & optimization
│   ├── unified_data.py              # Multi-directory data loader
│   ├── unified_training.py          # Unified training loops
│   ├── benchmark_models.py          # Comprehensive benchmarking
│   │
│   ├── unet_v2.py                   # U-Net architecture
│   ├── transunet.py                 # TransUNet architecture
│   └── swin_unet_plus_plus.py       # Swin-UNet++ architecture
│
├── data/                            # ← INPUT DATA (user provides)
│   │
│   ├── S1/                          # Dataset 1
│   │   ├── R11104.tiff
│   │   ├── R11105.tiff
│   │   └── ... (all TIFF files)
│   ├── S1_polygonal_masks.json      # Polygon annotations for S1
│   ├── S1_bounding_boxes.csv        # Bounding boxes for S1
│   ├── S1.csv                       # Optional annotations
│   │
│   ├── S2/                          # Dataset 2
│   │   └── ... (TIFF files)
│   ├── S2_polygonal_masks.json
│   ├── S2_bounding_boxes.csv
│   │
│   ├── S3/                          # Dataset 3
│   ├── S3_polygonal_masks.json
│   ├── S3_bounding_boxes.csv
│   │
│   └── ... (up to S10)
│
├── outputs/                         # ← RESULTS (auto-generated)
│   ├── models/
│   │   ├── best_u_net_model.pth
│   │   ├── best_transunet_model.pth
│   │   └── best_swin_unet_plusplus_model.pth
│   │
│   ├── plots/
│   │   ├── u_net_training_history.png
│   │   ├── transunet_training_history.png
│   │   ├── swin_unet_plusplus_training_history.png
│   │   ├── accuracy_comparison.png
│   │   ├── speed_comparison.png
│   │   ├── complexity_comparison.png
│   │   └── per_class_iou_heatmap.png
│   │
│   ├── predictions/                 # Example predictions
│   └── benchmark_comparison.csv     # Results table
│
└── log/                             # ← LOGS (auto-generated)
    ├── hardware_profile.json
    ├── u_net_metrics.json
    ├── transunet_metrics.json
    ├── swin_unet_plusplus_metrics.json
    ├── u_net_benchmark.json
    ├── transunet_benchmark.json
    ├── swin_unet_plusplus_benchmark.json
    └── benchmark_report.txt
```

## Data Organization

### Supported Naming Conventions

The pipeline supports two naming conventions for annotation files:

#### Convention 1: Files in data/ root (Recommended)
```
data/
├── S1/
│   └── *.tiff
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
└── S1.csv
```

#### Convention 2: Files inside dataset directory
```
data/
└── S1/
    ├── *.tiff
    ├── polygonal_masks.json
    ├── bounding_boxes.csv
    └── S1.csv
```

Both conventions work! The pipeline checks both locations.

## Multi-Dataset Support

### Automatic Discovery

When you run the pipeline:

1. **Scans data/ directory** for S1, S2, ..., S10 folders
2. **Loads annotations** for each discovered dataset
3. **Combines all samples** with unique IDs (e.g., S1/R11104, S2/R21001)
4. **Splits combined data** into train/validation sets
5. **Reports distribution** across datasets

### Example Output

```
DISCOVERING AND LOADING DATASETS
======================================================================

✅ Found 3 dataset(s): S1, S2, S3

Loading S1...
  ✅ Loaded 734 polygonal masks
  ✅ Loaded 734 bounding boxes

Loading S2...
  ✅ Loaded 500 polygonal masks
  ✅ Loaded 500 bounding boxes

Loading S3...
  ✅ Loaded 612 polygonal masks
  ✅ Loaded 612 bounding boxes

======================================================================
✅ TOTAL: 1846 samples across 3 dataset(s)
======================================================================

Data Split:
  Training samples:   1477
  Validation samples: 369

Samples per dataset:
  S1: 734 total (587 train, 147 val)
  S2: 500 total (400 train, 100 val)
  S3: 612 total (490 train, 122 val)
```

## Execution Flow

```
run.sh / run.bat
    │
    ├──> codes/setup.py
    │    ├─ Check Python version
    │    ├─ Check pip
    │    ├─ Install dependencies
    │    ├─ Check CUDA
    │    └─ Create directories
    │
    └──> codes/main_pipeline.py
         │
         ├──> codes/hardware_detector.py
         │    ├─ Detect GPU
         │    ├─ Validate requirements
         │    └─ Optimize batch sizes
         │
         ├──> codes/unified_data.py
         │    ├─ Discover S1-S10 directories
         │    ├─ Load all annotations
         │    ├─ Combine datasets
         │    └─ Create data loaders
         │
         ├──> codes/unified_training.py
         │    └─ Train each model
         │         ├─ codes/unet_v2.py
         │         ├─ codes/transunet.py
         │         └─ codes/swin_unet_plus_plus.py
         │
         └──> codes/benchmark_models.py
              ├─ Load trained models
              ├─ Calculate metrics
              ├─ Generate plots
              └─ Create reports
```

## Clean Root Directory

Only essential files in root:
- ✅ `run.sh` / `run.bat` - Entry points
- ✅ `requirements.txt` - Dependencies
- ✅ `config.yaml` - Configuration
- ✅ `README.md` - Documentation

All Python code organized in `/codes` directory!

## Key Design Decisions

1. **Clean Root**: All code in `/codes`, keeps root organized
2. **Auto-Discovery**: No manual dataset configuration needed
3. **Flexible Naming**: Supports multiple annotation file locations
4. **Unique IDs**: Prefixes samples with dataset name (S1/R11104)
5. **Combined Training**: All datasets merged for better generalization
6. **Per-Dataset Stats**: Reports training distribution per dataset
