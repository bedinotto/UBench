# Thermal Face Detection - Automated Training Pipeline (UBench)

A comprehensive, production-ready, and fully automated computer vision pipeline for **Thermal Facial Region Detection**. The project evaluates, trains, and comprehensively benchmarks three state-of-the-art segmentation architectures: **U-Net**, **TransUNet**, and **Swin-UNet++**.

## 🎯 Key Features

- **End-to-End Automation**: From extracting zip files to training and benchmarking, everything runs seamlessly with a single script (`run.sh` or `run.bat`).
- **Dynamic Data Extraction & Annotation**: The system automatically extracts nested zip files placed in the `requirements/` directory, discovers multi-dataset directories (S1-10), and generates required polygon and bounding-box annotations from basic CSV files.
- **Intelligent Data Mapping**: Uses advanced regex to seamlessly map string IDs from annotation files directly to the underlying `.tiff` files, filtering corrupted images automatically.
- **Hardware Profile Auto-Optimization**: Detects GPU capabilities (minimum 6GB VRAM like GTX 1660 Ti) and dynamically scales batch sizes, workers, and mixed precision strategies to avoid out-of-memory errors!
- **Uniform Training Interface**: Models are trained using standard augmentations, loss functions (Combined Cross-Entropy + Dice), and plateau learning-rate schedulers ensuring a fair comparison.
- **Comprehensive Benchmarking Suite**: Automatically compares Model Parameters, Training Time, Best Validation Loss/mIoU/Dice, Inference Speed, and VRAM utilization, outputting detailed reports and comparison plots.
- **Timestamped Execution Tracking**: All outputs, weights, performance histories, and console logs are automatically saved under timestamped subdirectories inside `outputs/` and `logs/`.

## 📋 Requirements & Prerequisites

### Minimum Hardware
- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM) or anything better. Pipeline scales to larger VRAMs.
- **RAM**: Minimum 8GB (16GB recommended).
- **Storage**: ~10GB of free space.

### Software
- **Python**: 3.8+ (preferably <= 3.11 for PyTorch compatibility)
- **CUDA**: Optional but highly recommended compatible CUDA Toolkit for PyTorch GPU acceleration.
- **OS**: Windows 10/11 or Linux / macOS

---

## 🚀 Quickstart & Usage

The easiest way to get started is using the unified execution scripts. 

### 1. Place your Data Archives
Zip your thermal image data and place those `.zip` files directly inside the `requirements/` directory.

Example structure before running:
```text
project/
├── requirements/
│   ├── Charlotte-ThermalFace.zip
│   ├── Some_Other_Thermal_Dataset.zip
└── run.sh
```

### 2. Run the Pipeline!

**On Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**On Windows:**
```cmd
run.bat
```

**That's it!** The script will:
1. **Extract** datasets from `requirements/` into `data/`.
2. **Setup** the python virtual environment.
3. **Train** U-Net, TransUNet, and Swin-UNet++ architectures.
4. **Benchmark** and plot the comparisons.

---

## ⚙️ Advanced Pipeline Options

You can modularize how you want the run scripts to execute using the following CLI arguments:

| Option | Description |
|--------|-------------|
| `--skip-setup` | Skips installing python requirements and environment setup. Use if you already ran the script once. |
| `--skip-extract` | Skips extracting the ZIP datasets from the `requirements/` directory. Use if your `data/` folder is already populated. |
| `--models <model_name>` | Choose which specific models to train. Allowed values: `unet`, `transunet`, `swin`. (e.g. `--models unet transunet`) |
| `--skip-benchmark` | Skips the model comparison benchmark step that happens post-training. |
| `--epochs N` | Overrides the configuration file to run the training over `N` epochs. |
| `-h, --help` | Shows the available command-line arguments. |

**Example Uses:**
```bash
# Skip package installs and data extraction, and train only Swin-UNet++ for 50 epochs:
./run.sh --skip-setup --skip-extract --models swin --epochs 50

# Train Unet and TransUNet without benchmarking:
./run.sh --models unet transunet --skip-benchmark
```

---

## 📁 Project Structure

```text
project/
├── run.sh                       # Linux/Mac pipeline execution
├── run.bat                      # Windows pipeline execution
├── requirements/                # Place dataset .ZIP archives here
├── README.md                    # This file
├── config.yaml                  # Global training/hardware configuration
│
├── codes/                       # Core python modules
│   ├── extract_data.py          # ZIP extractor & annotation generator
│   ├── setup.py                 # Dependency builder
│   ├── main_pipeline.py         # Primary orchestrator
│   ├── hardware_detector.py     # OS/GPU memory profile optimizer
│   ├── unified_data.py          # Data loaders & Regex matchers
│   ├── unified_training.py      # Core modular PyTorch training loop
│   ├── benchmark_models.py      # Speed & Metric comparator
│   ├── unet_v2.py               # U-Net Architecture
│   ├── transunet.py             # TransUNet Architecture
│   ├── swin_unet_plus_plus.py   # Swin-UNet++ Architecture
│   └── test_edge_cases.py / ... # Various utility / inspection scripts
│
├── data/                        # GENERATED/EXTRACTED DATA
│   ├── S1/                      # Raw Multi-Dataset Directories
│   │   └── R11104.tiff
│   ├── S1.csv                   # Basic Annotations
│   ├── S1_bounding_boxes.csv    # Auto-generated bbox labels
│   └── S1_polygonal_masks.json  # Auto-generated polygon masks
│
├── outputs/                     # TIMESTAMPED OUTPUTS
│   └── <run_timestamp>/
│       ├── models/              # Trained .pth Checkpoints
│       └── plots/               # IoU/Loss curves & Output Overlays
│
└── logs/                        # TIMESTAMPED LOGS
    └── <run_timestamp>/
        ├── extract.log          # Extraction traces
        ├── hardware_profile.json# Hardware specifications
        ├── *_metrics.json       # Epoch-over-epoch metrics
        └── benchmark_report.txt # Final comparison output
```

## 🎛️ Configuration File (`config.yaml`)

To granularly dictate everything from validation splits to augmentation techniques, modify `config.yaml`.
**Key Sections:**
- **Training Configurations:** Control `learning_rate`, `batch_size_*` limitations (gets auto-adjusted anyway relative to `hardware` limits), and `mixed_precision` status.
- **Model specific Configurations**: Enable or disable definitions, description injections.
- **Loss Operations**: Smoothness and weight distribution of combined Cross-Entropy & Dice scoring.
- **Augmentation**: Rotation probabilities, flip ratios, brightness scaling values.
- **Advanced Environment Flags**: Empty-cache frequencies, gradients-norm-clips, visible devices. 
