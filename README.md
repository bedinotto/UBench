# UBench — Thermal Face Detection Benchmark Pipeline

A comprehensive, production-ready, and fully automated computer vision pipeline for **Thermal Facial Region Segmentation**. The project evaluates, trains, and benchmarks three state-of-the-art segmentation architectures — **U-Net**, **TransUNet**, and **Swin-UNet++** — under identical conditions for a fair comparison.

---

## 📁 Table of Contents
1. [🎯 Key Features](#-key-features)
2. [📋 Prerequisites & Requirements](#-prerequisites--requirements)
3. [🚀 Quickstart Guide](#-quickstart-guide)
4. [📁 Project Structure & Data Layout](#-project-structure--data-layout)
5. [⚙️ How the Pipeline Works](#%EF%B8%8F-how-the-pipeline-works)
6. [🎛️ Configuration & CLI Options](#%EF%B8%8F-configuration--cli-options)
7. [🔍 Verification & Troubleshooting](#-verification--troubleshooting)
8. [🎉 Version Changelog (v2.0)](#-version-changelog-v20)

---

## 🎯 Key Features

- **End-to-End Automation**: From extracting zip files to training and benchmarking, everything runs end-to-end with a single script (`run.sh` or `run.bat`).
- **Multi-Dataset Support**: Automatically discovers and combines all dataset directories (`S1`–`S10`) placed in `data/`, with no manual configuration required.
- **Dynamic Data Extraction & Annotation**: Automatically extracts nested zip archives placed in `requirements/`, discovers multi-dataset directories, and generates polygon and bounding-box annotations from CSV files.
- **Intelligent Data Mapping**: Uses advanced regex to map string IDs from annotation files directly to `.tiff` image files, automatically filtering corrupted images.
- **Hardware Auto-Optimization**: Detects GPU capabilities (minimum 6 GB VRAM, e.g., GTX 1660 Ti) and dynamically scales batch sizes, workers, and mixed-precision strategies to avoid out-of-memory errors.
- **Uniform Training Interface**: All three models are trained with standard augmentations, a combined Cross-Entropy + Dice loss, and plateau learning-rate schedulers for a fair comparison.
- **Comprehensive Benchmarking**: Automatically compares model parameters, training time, best validation loss/mIoU/Dice, inference speed, and VRAM utilization — outputting detailed reports and comparison plots.
- **Timestamped Execution Tracking**: All outputs, weights, metric histories, and console logs are saved under timestamped subdirectories in `outputs/` and `logs/`.
- **Fault Tolerance & Resilience**:
  - **Aggressive Checkpointing**: Model weights and optimizer states are saved at the end of every epoch.
  - **Auto-Resume**: Automatically resume training from the last checkpoint if training was interrupted.
  - **Global Error Catching**: Catches runtime errors (e.g. matrix mismatches or out-of-memory) and dumps them into a log file for troubleshooting.

---

## 📋 Prerequisites & Requirements

### ⚠️ Python Version (Critical)
PyTorch CUDA wheels are only published for **Python 3.8 – 3.12**. Python 3.13+ is **not supported** — pip will silently install the CPU-only build, causing a "CUDA not available" failure at runtime.
> [!IMPORTANT]
> **Recommended: Python 3.10 or 3.11**
> Download from: https://www.python.org/downloads/

### 📦 Git LFS (Large File Storage)
This project tracks large dataset files (such as `.zip` files in `requirements/`) using Git LFS. You must have Git LFS installed and initialized in your repository to download the actual files instead of small text pointer files.
- **To Install**: Follow instructions at [git-lfs.com](https://git-lfs.com/).
- **To Initialize**: Run `git lfs install` and then `git lfs pull` in the repository root.

### Hardware Requirements
| Component | Minimum | Notes |
|-----------|---------|-------|
| **GPU** | NVIDIA GTX 1660 Ti (6 GB VRAM) | Pipeline auto-scales for larger GPUs |
| **RAM** | 8 GB | 16 GB recommended |
| **Storage** | ~10 GB free space | For datasets, outputs, and model weights |

### Software Requirements
| Dependency | Version | Notes |
|------------|---------|-------|
| **Python** | 3.8 – 3.12 | 3.13+ is **not supported** |
| **Git LFS** | 3.0+ recommended | Required to retrieve large ZIP archives and dataset files (e.g. in `requirements/`) |
| **NVIDIA Driver** | Latest recommended | Required for CUDA. Install from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) |
| **CUDA** | 11.8 or 12.1 | Installed automatically via PyTorch CUDA wheel |
| **OS** | Windows 10/11, Linux, or macOS | `run.bat` for Windows, `run.sh` for Linux/Mac |

### Python Packages (auto-installed by setup)
PyTorch with CUDA support is automatically installed by `codes/setup.py` using the correct index URL for your GPU driver. All other packages come from `requirements/requirements.txt`:
```
opencv-python, Pillow, numpy, pandas, scipy,
scikit-learn, tqdm, matplotlib, seaborn,
PyYAML, psutil, GPUtil, nvidia-ml-py
```
> [!NOTE]
> `torch`, `torchvision`, and `torchaudio` are **not** in `requirements.txt`. They are installed separately with CUDA support by the setup script to avoid the CPU-only default from plain `pip install torch`.

---

## 🚀 Quickstart Guide

Get started in 3 steps:

### Step 1: Organize Your Data
Place your dataset archives (nested zip files) directly in the `requirements/` directory:
```text
UBench/
├── requirements/
│   ├── Charlotte-ThermalFace.zip   ← your archive here
│   └── requirements.txt
└── run.bat / run.sh
```
> [!TIP]
> The dataset ZIP files are tracked by Git LFS. If you cloned the repository and notice that the `.zip` file is only a few bytes in size or corrupted, make sure Git LFS is installed and run `git lfs pull` to fetch the actual file content.

Alternatively, you can manually extract your thermal imaging datasets and place them in the `data/` directory:
```text
data/
├── S1/
│   ├── R11104.tiff
│   ├── R11105.tiff
│   └── ... (more TIFF files)
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
├── S2/
│   └── ... (TIFF files)
├── S2_polygonal_masks.json
├── S2_bounding_boxes.csv
└── ... (S3, S4, ... up to S10)
```
**The pipeline automatically discovers all S1-S10 directories!**

### Step 2: Run the Pipeline
**On Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**On Windows:**
```cmd
run.bat
```

### Step 3: Check Results
After training completes, find your results in:
```text
outputs/
├── models/              # Trained weights (.pth files)
├── plots/               # Training curves and comparisons
└── benchmark_comparison.csv

log/
├── hardware_profile.json
├── *_metrics.json
└── benchmark_report.txt
```

### 🎯 Expected Console Output
After starting a successful run, you will see output like this:
```text
DISCOVERING AND LOADING DATASETS
======================================================================
✅ Found 3 dataset(s): S1, S2, S3
[...]
======================================================================
✅ TOTAL: 1846 samples across 3 dataset(s)
======================================================================

TRAINING: U-Net
======================================================================
[Training progress bars...]
✅ Training completed

TRAINING: TransUNet
======================================================================
[Training progress bars...]
✅ Training completed

TRAINING: Swin-UNet++
======================================================================
[Training progress bars...]
✅ Training completed

BENCHMARKING
======================================================================
[Comparison results...]
✅ Benchmark completed

✅✅✅ PIPELINE COMPLETED SUCCESSFULLY ✅✅✅
```

### 💡 Execution Tips
1. **Start Small**: Test with 1 dataset and 10 epochs first:
   ```bash
   ./run.sh --epochs 10
   ```
2. **Monitor GPU**: Open another terminal and run:
   ```bash
   watch -n 1 nvidia-smi
   ```
3. **Check Logs**: If training fails, check `logs/<timestamp>/pipeline.log`.
4. **Free Memory**: Close other GPU-intensive applications before training.

---

## 📁 Project Structure & Data Layout

### Complete Directory Layout
```text
UBench/
├── run.sh                         # Linux/Mac pipeline entry point
├── run.bat                        # Windows pipeline entry point
├── config.yaml                    # Global training/hardware configuration
├── README.md                      # Consolidated project documentation
│
├── requirements/
│   ├── requirements.txt           # Python dependencies (excl. PyTorch)
│   └── <YourDataset>.zip          # Place dataset archives here
│
├── codes/                         # All Python source code
│   ├── main_pipeline.py           # Primary orchestrator
│   ├── setup.py                   # Dependency builder & environment validator
│   ├── extract_data.py            # ZIP extractor & annotation generator
│   ├── hardware_detector.py       # GPU profiling & batch-size optimizer
│   ├── unified_data.py            # Multi-dataset loader & regex matcher
│   ├── unified_training.py        # PyTorch training loop (shared across models)
│   ├── benchmark_models.py        # Speed & metric comparator
│   ├── unet_v2.py                 # U-Net architecture
│   ├── transunet.py               # TransUNet architecture
│   ├── swin_unet_plus_plus.py     # Swin-UNet++ architecture
│   └── generate_boxes_polygons.py # Annotation generation helper
│
├── data/                          # Extracted datasets (auto-generated)
│   ├── S1/                        # Dataset 1 — TIFF thermal images
│   │   └── R11104.tiff
│   ├── S1.csv                     # Source annotations
│   ├── S1_bounding_boxes.csv      # Auto-generated bounding-box labels
│   ├── S1_polygonal_masks.json    # Auto-generated polygon masks
│   └── S2/, S2.csv, ...           # Additional datasets (up to S10)
│
├── outputs/                       # Timestamped training outputs
│   └── <run_timestamp>/
│       ├── models/                # Trained .pth checkpoints
│       └── plots/                 # IoU/loss curves & overlay visualizations
│
└── logs/                          # Timestamped run logs
    └── <run_timestamp>/
        ├── extract.log            # Data extraction traces
        ├── hardware_profile.json  # Detected hardware specifications
        ├── *_metrics.json         # Per-epoch training metrics per model
        └── benchmark_report.txt   # Final model comparison summary
```

### Data Annotation Conventions
The pipeline supports two annotation layouts — both work automatically:

#### Option A — Files in `data/` root (Recommended):
```text
data/
├── S1/            *.tiff files
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
└── S1.csv
```

#### Option B — Files inside dataset directory:
```text
data/
└── S1/
    ├── *.tiff
    ├── polygonal_masks.json
    └── bounding_boxes.csv
```

### Key Design Decisions
1. **Clean Root**: All logic resides in `/codes` to keep the root directory uncluttered.
2. **Auto-Discovery**: No dataset configurations are hardcoded.
3. **Flexible Naming**: Accepts multiple annotation layouts automatically.
4. **Unique ID Prefixing**: Namespaces sample keys by dataset source (e.g. `S1/R11104`) to prevent conflict during combined multi-dataset training.
5. **Stratified Splitting**: Merged datasets are split stratifically to represent each source dataset equally in the training/validation subsets.

---

## ⚙️ How the Pipeline Works

### 1. What Happens Automatically
When you run the pipeline:
- **Environment Setup**: Python versions, CUDA presence, and pip are checked. Dependencies are installed.
- **Hardware Detection**: Detects GPU capabilities and dynamically scales batch sizes and mixed-precision strategies.
- **Dataset Discovery**: Extracts ZIPs from `requirements/` and automatically maps files using regex patterns, skipping corrupted files.
- **Model Training**: Runs uniform training with `CombinedLoss` and `ReduceLROnPlateau`.
- **Benchmarking**: Compares speeds, parameters, losses, and VRAM footprints.

### 2. Execution Flow Diagram
```text
run.sh / run.bat
    │
    ├──> codes/setup.py
    │    ├─ Check Python version
    │    ├─ Check pip & Git LFS
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

### 3. Multi-Dataset Combination Details
When training, multiple dataset splits are merged:
- **Sample IDs**: Prefixed (e.g., `S1/R11104`, `S2/R21001`).
- **Data Loaders**: Multi-dataset loaders feed combined tensors to the models.
- **Stratification**: Train/Val folds split samples proportionally per dataset.

---

## 🎛️ Configuration & CLI Options

### Advanced CLI Flags
Both `run.bat` and `run.sh` accept the following flags:
| Flag | Description |
|------|-------------|
| `--skip-extract` | Skip ZIP extraction. Use if `data/` is already populated. |
| `--skip-setup` | Skip environment setup and package installation. Use after the first run. |
| `--models <name> [<name>...]` | Train only specific models. Choices: `unet`, `transunet`, `swin`. |
| `--skip-benchmark` | Skip the benchmarking step after training. |
| `--epochs N` | Override `config.yaml` and train for `N` epochs. |
| `-h`, `--help` | Show help. |

**Examples:**
```bash
# Fast validation test
./run.sh --skip-extract --skip-setup --models unet --epochs 10

# Train two specific models
./run.sh --skip-extract --models transunet swin
```

### Global Configuration (`config.yaml`)
Modify `config.yaml` in the root directory to fine-tune operations:
| Section | Description |
|---------|-------------|
| `data` | Directories, image dimensions (`256×256`), and classes (`10`). |
| `training` | Epochs, batch size optimization limits, learning rates, split k-folds. |
| `loss` | Weight factors for CE + Dice losses, dice smoothing constants. |
| `optimizer` | Adam optimizer settings (beta values, weight decay, eps). |
| `scheduler` | ReduceLROnPlateau lr patience, decay factors, and mins. |
| `augmentation` | Horizontal flips, brightness adjustments, and rotations. |
| `benchmark` | Enabling benchmarks, inference test sample sizes. |
| `visualization` | Colors, heatmaps, resolution DPI, and comparison curves. |
| `hardware` | Constraints on VRAM, system RAM, and CPU core counts. |
| `advanced` | AMP config, cuDNN optimizations, memory layouts, gradient clipping. |

---

## 🔍 Verification & Troubleshooting

### 1. Data Verification Checklist
Before running the pipeline, check that your local environment is configured:
- [ ] At least one `Sx` directory exists in `data/`
- [ ] Each dataset directory contains TIFF images
- [ ] Annotations files (`polygonal_masks.json` and `bounding_boxes.csv`) are located inside the directory or in `data/` root
- [ ] Git LFS is installed and all zip files have been fully downloaded (size > 400 MB)

You can run these commands to inspect data status manually:
```bash
# Verify datasets exist
ls -d data/S*

# Check number of TIFF files
ls data/S1/*.tiff | wc -l

# Validate zip files are downloaded properly
ls -lh requirements/*.zip
```
Or run the automated checker script:
```bash
python3 codes/extract_data.py --check
```

### 2. Common Issues & Troubleshooting
**"No dataset directories found"**
- Check that you have at least one directory named `S1`, `S2`, etc.
- Names are **case-sensitive** on Linux.

**"No polygonal masks found"**
- Ensure `S1_polygonal_masks.json` exists in `data/` or `polygonal_masks.json` exists in `data/S1/`.
- Ensure JSON is well-formed.

**"CUDA not available" / Setup stops at hardware checks**
- This usually means PyTorch was installed without CUDA support (CPU-only build).
- Run `nvidia-smi` to check if NVIDIA drivers are present. If missing, install drivers from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx).
- Run the setup utility to install the correct CUDA-supported PyTorch wheel:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

**"Unsupported Python version"**
- You are running Python 3.13+. PyTorch does not publish CUDA wheels for Python 3.13+. Please download and install Python 3.10 or 3.11.

**"Git LFS pointer detected" or "Not a valid ZIP file"**
- The repository ZIP files are placeholder files because Git LFS was not initialized.
- Install Git LFS, then initialize and pull:
  ```bash
  git lfs install
  git lfs pull
  ```

---

## 🎉 Version Changelog (v2.0)

Version 2.0 introduces multi-dataset loading, streamlined cross-platform setup, and code refactoring.

### 1. Major Updates Implemented
- **Multi-Dataset Support (S1-S10)**: Seamlessly discovers and trains on arbitrary combinations of datasets.
- **Clean Code Organization**: Relocated all Python files from the root to `/codes`.
- **Integrated Dependency Checking**: Automated validation of CUDA, Python versions, and Git LFS availability directly in the pipeline step.

### 2. Code changes details
- **`unified_data.py`**: Added `MultiDirectoryDataLoader` and automatic mapping patterns to prefix IDs and avoid data clashes.
- **`setup.py`**: Integrated modular requirements validation checklist (pip, CUDA, PyTorch, Git LFS).

### 3. Benefits
- **Automated Discovery**: Dropping new directories into `data/` triggers auto-loading immediately.
- **Generalization**: Model variance is lowered by pooling samples from different datasets.
- **Per-Dataset Stats**: Tracks and logs data distributions and accuracy metrics individually per fold and dataset.
