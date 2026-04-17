# UBench — Thermal Face Detection Benchmark Pipeline

A comprehensive, production-ready, and fully automated computer vision pipeline for **Thermal Facial Region Segmentation**. The project evaluates, trains, and benchmarks three state-of-the-art segmentation architectures — **U-Net**, **TransUNet**, and **Swin-UNet++** — under identical conditions for a fair comparison.

## 🎯 Key Features

- **End-to-End Automation**: From extracting zip files to training and benchmarking, everything runs end-to-end with a single script (`run.sh` or `run.bat`).
- **Multi-Dataset Support**: Automatically discovers and combines all dataset directories (`S1`–`S10`) placed in `data/`, with no manual configuration required.
- **Dynamic Data Extraction & Annotation**: Automatically extracts nested zip archives placed in `requirements/`, discovers multi-dataset directories, and generates polygon and bounding-box annotations from CSV files.
- **Intelligent Data Mapping**: Uses advanced regex to map string IDs from annotation files directly to `.tiff` image files, automatically filtering corrupted images.
- **Hardware Auto-Optimization**: Detects GPU capabilities (minimum 6 GB VRAM, e.g., GTX 1660 Ti) and dynamically scales batch sizes, workers, and mixed-precision strategies to avoid out-of-memory errors.
- **Uniform Training Interface**: All three models are trained with standard augmentations, a combined Cross-Entropy + Dice loss, and plateau learning-rate schedulers for a fair comparison.
- **Comprehensive Benchmarking**: Automatically compares model parameters, training time, best validation loss/mIoU/Dice, inference speed, and VRAM utilization — outputting detailed reports and comparison plots.
- **Timestamped Execution Tracking**: All outputs, weights, metric histories, and console logs are saved under timestamped subdirectories in `outputs/` and `logs/`.

---

## 📋 Prerequisites

### ⚠️ Python Version (Critical)

PyTorch CUDA wheels are only published for **Python 3.8 – 3.12**. Python 3.13+ is **not supported** — pip will silently install the CPU-only build, causing a "CUDA not available" failure at runtime.

> **Recommended: Python 3.10 or 3.11**
> Download from: https://www.python.org/downloads/

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

> **Note**: `torch`, `torchvision`, and `torchaudio` are **not** in `requirements.txt`. They are installed separately with CUDA support by the setup script to avoid the CPU-only default from plain `pip install torch`.

---

## 🚀 Quickstart

### 1. Place Your Dataset Archives

Put your thermal image dataset `.zip` file(s) directly inside the `requirements/` directory:

```text
UBench/
├── requirements/
│   ├── Charlotte-ThermalFace.zip   ← your archive here
│   └── requirements.txt
└── run.bat / run.sh
```

The extraction step will unpack the archives into `data/`, automatically discovering `S1`–`S10` subdirectories.

### 2. Run the Pipeline

**On Windows:**
```cmd
run.bat
```

**On Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**That's it!** The script will:
1. ✅ **Extract** datasets from `requirements/` into `data/`
2. ✅ **Setup** the Python environment and install CUDA-enabled PyTorch + dependencies
3. ✅ **Train** U-Net, TransUNet, and Swin-UNet++ sequentially
4. ✅ **Benchmark** and generate comparison plots and reports

---

## ⚙️ Advanced CLI Options

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
# Quick test: train only U-Net for 10 epochs (skip extraction and setup after first run)
./run.sh --skip-extract --skip-setup --models unet --epochs 10

# Train TransUNet and Swin-UNet++ without re-extracting data
./run.sh --skip-extract --models transunet swin

# Full pipeline with 50 epochs
./run.sh --epochs 50

# Windows equivalents (use run.bat with identical flags)
run.bat --skip-setup --models unet --epochs 10
```

---

## 📁 Project Structure

```text
UBench/
├── run.sh                         # Linux/Mac pipeline entry point
├── run.bat                        # Windows pipeline entry point
├── config.yaml                    # Global training/hardware configuration
├── README.md                      # This file
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
│   ├── generate_boxes_polygons.py # Annotation generation helper
│   └── inspect_ids.py / verify_regex.py / ...   # Utility/debug scripts
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
        └── benchmark_report.txt  # Final model comparison summary
```

### Data Annotation Conventions

The pipeline supports two annotation layouts — both work automatically:

**Option A — Files in `data/` root (recommended):**
```text
data/
├── S1/            *.tiff files
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
└── S1.csv
```

**Option B — Files inside dataset directory:**
```text
data/
└── S1/
    ├── *.tiff
    ├── polygonal_masks.json
    └── bounding_boxes.csv
```

---

## 🎛️ Configuration (`config.yaml`)

Modify `config.yaml` to tune the pipeline. Key sections:

| Section | What it controls |
|---------|-----------------|
| `data` | Input/output directories, image size (`256×256`), number of classes (`10`) |
| `training` | Learning rate, epoch count, batch sizes, mixed precision, validation split |
| `models` | Enable/disable individual models; add descriptions |
| `loss` | Combined CE + Dice weights and smoothing |
| `optimizer` | Adam betas, epsilon, weight decay |
| `scheduler` | ReduceLROnPlateau patience, factor, minimum LR |
| `augmentation` | Flip probability, rotation range, brightness range |
| `benchmark` | Enable/disable; metrics to compute; number of example predictions |
| `visualization` | Training curves, comparison plots, per-class IoU heatmaps, DPI |
| `hardware` | Minimum GPU memory, RAM, and CPU core requirements |
| `logging` | Checkpoint saving, verbose output, log frequency |
| `advanced` | CUDA device selection, cuDNN settings, memory pool config, grad norm clipping |

> **Tip**: To do a quick smoke-test, set `training.num_epochs: 10` and `--models unet`. For memory-constrained GPUs, do **not** disable `training.mixed_precision`.

---

## 🔍 Troubleshooting

### `CUDA not available` / Pipeline stops at hardware detection

This almost always means PyTorch was installed without CUDA support (`pip install torch` defaults to CPU-only).

1. Verify your NVIDIA driver is installed: run `nvidia-smi` in a terminal.
   - If not found → install from https://www.nvidia.com/download/index.aspx, then **restart**.
2. Reinstall PyTorch with the correct CUDA wheel:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   (Use `cu118` if `nvidia-smi` reports a CUDA version below 12.x.)
3. Re-run skipping setup if packages are already installed:
   ```cmd
   run.bat --skip-extract --skip-setup
   ```

### `Unsupported Python version` error

You are running Python 3.13+. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ and ensure it comes first in your `PATH`.

### `No dataset directories found`

- Ensure at least one `S1`, `S2`, … directory exists in `data/`.
- Directory names are **case-sensitive** on Linux/Mac.
- Run extraction step: `run.bat` (without `--skip-extract`).

### `No polygonal masks found`

- Check for `data/S1_polygonal_masks.json` **or** `data/S1/polygonal_masks.json`.
- Verify the JSON is well-formed.

### `Minimum hardware requirements not met`

You need at least a GTX 1660 Ti with 6 GB VRAM. Check your GPU with `nvidia-smi`.

---

## 📖 Additional Documentation

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | 3-step startup guide with expected console output |
| `PROJECT_STRUCTURE.md` | Detailed directory layout and execution flow diagram |
| `config.yaml` | Fully commented configuration reference |
| `UPDATE_SUMMARY.md` | Changelog and version history |
