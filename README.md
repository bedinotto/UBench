# Thermal Face Detection - Automated Training Pipeline

A complete, production-ready automated training pipeline for thermal facial region detection using deep learning. Trains and compares three state-of-the-art architectures: **U-Net**, **TransUNet**, and **Swin-UNet++**.

## 🎯 Key Features

- **Multi-Dataset Support**: Automatically discovers and trains on S1-S10 directories
- **Cross-Platform**: Works on Windows and Linux/Mac
- **Automatic Hardware Optimization**: Detects GPU capabilities and scales hyperparameters
- **Hardware Validation**: Ensures minimum GTX 1660 Ti (6GB VRAM)
- **Unified Training Pipeline**: Consistent processing across all models
- **Comprehensive Benchmarking**: Automated comparison of accuracy, speed, and efficiency

## 📋 Requirements

### Minimum Hardware
- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM) or better
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Software
- **Python**: 3.8 or higher
- **CUDA**: Compatible CUDA toolkit for your GPU
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)

## 📁 Project Structure

```
project/
├── run.sh                       # Linux/Mac entry point
├── run.bat                      # Windows entry point
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration file
├── README.md                    # This file
│
├── codes/                       # All Python code (clean root)
│   ├── setup.py                 # Environment setup
│   ├── main_pipeline.py         # Main orchestrator
│   ├── hardware_detector.py     # Hardware detection
│   ├── unified_data.py          # Multi-directory data loader
│   ├── unified_training.py      # Training module
│   ├── benchmark_models.py      # Benchmarking suite
│   ├── unet_v2.py              # U-Net architecture
│   ├── transunet.py            # TransUNet architecture
│   └── swin_unet_plus_plus.py  # Swin-UNet++ architecture
│
├── data/                        # INPUT DATA (user provides)
│   ├── S1/                      # Dataset 1
│   │   ├── R11104.tiff
│   │   ├── R11105.tiff
│   │   └── ...
│   ├── S1_polygonal_masks.json  # Masks for S1
│   ├── S1_bounding_boxes.csv    # Bboxes for S1
│   ├── S1.csv                   # Annotations for S1 (optional)
│   │
│   ├── S2/                      # Dataset 2
│   │   └── ...
│   ├── S2_polygonal_masks.json
│   ├── S2_bounding_boxes.csv
│   │
│   ├── S3/                      # Dataset 3 (and so on...)
│   └── ...                      # Up to S10
│
├── outputs/                     # RESULTS (auto-generated)
│   ├── models/                  # Trained model weights
│   ├── plots/                   # Visualizations
│   └── benchmark_comparison.csv
│
└── log/                         # LOGS (auto-generated)
    ├── hardware_profile.json
    ├── *_metrics.json
    └── benchmark_report.txt
```

## 🚀 Quick Start

### 1. Prepare Your Data

The pipeline **automatically discovers** all dataset directories (S1, S2, ..., S10).

For each dataset, you need:
- **Required**: TIFF thermal images in `data/Sx/` directory
- **Required**: `data/Sx_polygonal_masks.json` (or inside `data/Sx/polygonal_masks.json`)
- **Required**: `data/Sx_bounding_boxes.csv` (or inside `data/Sx/bounding_boxes.csv`)
- **Optional**: `data/Sx.csv` (annotations)

Example for two datasets:
```
data/
├── S1/
│   ├── R11104.tiff
│   └── ...
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
│
├── S2/
│   ├── R21001.tiff
│   └── ...
├── S2_polygonal_masks.json
└── S2_bounding_boxes.csv
```

### 2. Run the Pipeline

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```cmd
run.bat
```

The pipeline will automatically discover and combine all datasets!

### 3. View Results in outputs/ and log/ directories

## ⚙️ Advanced Usage

Train specific models:
```bash
./run.sh --models unet
./run.sh --skip-setup --epochs 50
```

See full documentation in README sections below.
