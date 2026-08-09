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
8. [🧾 Reproducing the reported run](#-reproducing-the-reported-run)
9. [🎉 Version Changelog (v2.0)](#-version-changelog-v20)

---

## 🧾 Reproducing the reported run

**Archived snapshot:** [10.5281/zenodo.21864556](https://doi.org/10.5281/zenodo.21864556)
(release `v1.0.0`). That is a *version* DOI — it points permanently at that snapshot, not
at whatever `main` becomes. The archive holds the versioned tree but **not** the Git LFS
objects, so the 15 trained checkpoints and the dataset archive are not in it; get those
from this repository.

> **`main` does not reproduce the dissertation's numbers, by construction.** It carries
> corrections (UB-27, UB-28, UB-29, UB-30) that change labels and measurements. To
> reproduce the reported results, use the tag below.

The run behind the dissertation is `run_id` **`2026-07-25_10-55-17`** (3 architectures ×
5 folds, 186.9 h of GPU). Its code is tagged:

```bash
git checkout run-2026-07-25          # commit 5a5b4b8
git apply run_sh_asrun.patch         # the working-tree delta, see below
uv venv --python 3.11 && source .venv/bin/activate
uv pip sync requirements/requirements.cuda.lock --torch-backend=cu121
```

**Two artifacts, not one.** The training was launched from an unconfirmed working tree
(`run_metadata.json` records `git_dirty: true`), so the tagged commit alone is not the
full story. The delta is versioned as **`run_sh_asrun.patch`** (committed in `11b64d5`),
and `running_sha.txt` independently records the same nominal commit.

**The patch does not touch the pipeline.** It changes exactly three files: the editor's
plugin list (`.claude/settings.json`), the `requires-python` upper bound in
`pyproject.toml` (metadata, selects no package), and the environment bootstrap in
`run.sh`. The Git tree object of `codes/` is `885fc55` both at the tagged commit and at
the commit carrying the patch, and the patch touches no file under `codes/` — so every
line of data, training, metric, model and configuration code is bit-identical to what is
published here.

**Environment and configuration** are pinned independently: `lockfile_hash` in
`logs/2026-07-25_10-55-17/run_metadata.json` is the sha256 of the CUDA lock, and the same
file records the effective config (seed 42, deterministic, K=5, 100 epochs, `fixed_range`
normalization, empty `test_subjects`).

**Caveat, stated rather than omitted.** The patch was captured on 2026-07-28, three days
after the run. `HEAD` did not move in that window, which supports but does not prove that
the tree was identical at launch. The run is therefore reproducible from *a commit plus a
recorded, inspectable patch*, not from a bare checkout: the guarantee over the pipeline
code is cryptographic, the one over the invocation script is documentary.

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
  - **Resumable Runs**: `--resume <run_id>` reuses a previous run's `outputs/<run_id>` and `logs/<run_id>` directories and continues training from its epoch checkpoints, keeping the metric history intact. `outputs/latest` always points at the newest run.
  - **Global Error Catching**: Catches runtime errors (e.g. matrix mismatches or out-of-memory) and dumps them into a log file for troubleshooting.

---

## 📋 Prerequisites & Requirements

### ⚠️ Python Version
The default install is **CPU** and works on **Python 3.10–3.13** — the committed lockfile is resolved for the 3.10 floor and CI runs on 3.11. The **CUDA/GPU** path is narrower: PyTorch publishes CUDA wheels only for **3.8–3.12**, so a GPU **training box** must use **Python 3.10 or 3.11** (`codes/setup.py` hard-errors the CUDA path on 3.13+).
> [!IMPORTANT]
> **CPU (dev/CI): Python 3.10–3.13. GPU training box: Python 3.10 or 3.11.**
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
| **Python** | 3.10–3.13 (CPU) · 3.10/3.11 (CUDA) | CUDA wheels exist only for 3.8–3.12, so the GPU path needs 3.10/3.11 |
| **Git LFS** | 3.0+ recommended | Required to retrieve large ZIP archives and dataset files (e.g. in `requirements/`) |
| **NVIDIA Driver** | Latest recommended | Required for CUDA. Install from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) |
| **CUDA** | 11.8 or 12.1 | On the GPU box, `setup.py` installs the matching CUDA torch (`cu118`/`cu121`) from the lock |
| **OS** | Windows 10/11, Linux, or macOS | `run.bat` for Windows, `run.sh` for Linux/Mac |

### Python Packages (installed from a lockfile)
All dependencies — **including `torch`/`torchvision`** — are declared in **one** place, the `[project]` table of `pyproject.toml`, and installed from a **generated lockfile** with [`uv`](https://docs.astral.sh/uv/), never from loose version ranges (UB-20b):
```
torch, torchvision, torchmetrics, timm,
opencv-python, Pillow, albumentations, numpy, pandas, scipy,
scikit-learn, tqdm, matplotlib, seaborn,
PyYAML, pydantic, psutil, GPUtil, nvidia-ml-py
```
> [!NOTE]
> `requirements/requirements.txt` is now only a pointer — the dependency source of truth is `pyproject.toml`, and the pinned closure lives in `requirements/requirements.cpu.lock` (CPU, committed; a CUDA lock is generated on the GPU box). `codes/setup.py` installs with `uv pip sync <lock>` (backend-aware; no `--force-reinstall --no-deps`, and it refuses to install into system Python).

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

### Step 2: Set Up the Environment & Run the Pipeline
First create a virtual environment and install the pinned dependencies with [`uv`](https://docs.astral.sh/uv/) (CPU — dev/CI):
```bash
uv venv && source .venv/bin/activate
uv pip sync requirements/requirements.cpu.lock --torch-backend=cpu
```
> On a **GPU training box**, install the CUDA lock instead — see
> `docs/phase1_realdata_checklist.md` (or let `codes/setup.py`, invoked by
> `run.sh`, generate and install it for you).

Then run the pipeline:

**On Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**On Windows:**
```cmd
run.bat
```

> **Offline preprocessing is automatic.** Training reads pre-computed
> arrays from `data/processed/` (created by `codes/preprocess_data.py`).
> The pipeline runs this step for you whenever
> `data/processed/metadata.csv` is missing. To rebuild the preprocessed
> data unconditionally (e.g. after changing the raw data or annotations):
>
> ```bash
> ./run.sh --force-preprocess
> ```

### Step 3: Check Results
After training completes, find your results in:
```text
outputs/<run_id>/                 # one timestamped dir per ./run.sh invocation
├── models/                       # best_<model>_fold_<n>_model.pth (selected by val mIoU)
├── plots/                        # training curves and comparison plots
├── checkpoints/                  # resume checkpoints
└── benchmark_comparison.csv      # CV + TEST metrics for all models
outputs/latest -> outputs/<run_id>   # symlink to the newest run

logs/<run_id>/
├── pipeline.log
├── run_metadata.json             # git SHA, torch/CUDA, seed, lockfile hash, effective config
├── hardware_profile.json         # detected GPU/CPU profile + chosen batch sizes
├── *_metrics.json
└── benchmark_report.txt
```

### 🎯 Expected Console Output
After starting a successful run, you will see output like this (illustrative — labels and structure, not specific metric values):
```text
DISCOVERING AND LOADING DATASETS
======================================================================
✅ Found 3 dataset(s): S1, S2, S3
✅ TOTAL: <N> samples across 3 dataset(s)
   Normalization: fixed_range [20.0, 40.0] °C
   Leave-subjects-out CV: K folds over whole-subject groups
   (any held-out test_subjects are excluded from every fold)
======================================================================

TRAINING: U-Net            [recipe: Adam + ReduceLROnPlateau]
======================================================================
[fold 1/K ... best-val-mIoU checkpoint saved]
✅ Training completed

TRAINING: TransUNet        [recipe: AdamW + warmup→cosine]
======================================================================
[Training progress...]
✅ Training completed

TRAINING: Swin-UNet++      [recipe: AdamW + warmup→cosine]
======================================================================
[Training progress...]
✅ Training completed

BENCHMARKING
======================================================================
[CV section:   per-model mean ± std across folds]
[TEST section: fold-models scored on held-out test_subjects, if set]
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
├── README.md                      # Consolidated project documentation
├── pyproject.toml                 # dependency source of truth + tooling config
│
├── requirements/
│   ├── requirements.txt           # pointer — deps now live in pyproject.toml
│   ├── requirements.cpu.lock      # pinned CPU dependency lock (uv pip sync)
│   └── <YourDataset>.zip          # Place dataset archives here
│
├── codes/                         # All Python source code
│   ├── config.yaml                # THE config (single validated authority)
│   ├── config_schema.py           # Typed pydantic schema (rejects unknown keys)
│   ├── main_pipeline.py           # Primary orchestrator
│   ├── setup.py                   # Dependency builder & environment validator
│   ├── extract_data.py            # ZIP extractor & annotation generator
│   ├── hardware_detector.py       # GPU profiling & batch-size optimizer
│   ├── unified_data.py            # Multi-dataset loader & regex matcher
│   ├── unified_training.py        # PyTorch training loop (shared across models)
│   ├── benchmark_models.py        # Speed & metric comparator
│   ├── unet_v2.py                 # U-Net architecture (from scratch)
│   ├── transunet.py               # TransUNet architecture (from scratch)
│   ├── swin_unet_plus_plus.py     # Swin-UNet++ (from scratch; defective attn, UB-17 baseline)
│   ├── swin_pretrained.py         # SwinV2 (timm, ImageNet-pretrained) + conv decoder
│   ├── transunet_pretrained.py    # R50+ViT-B/16 hybrid (timm, ImageNet-pretrained)
│   ├── pretrained_stem.py         # 1-channel stem adaptation (sum RGB kernels, M5)
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
├── outputs/                       # Timestamped training outputs (+ outputs/latest symlink)
│   └── <run_timestamp>/
│       ├── models/                # Trained .pth checkpoints (best by val mIoU)
│       ├── plots/                 # IoU/loss curves & overlay visualizations
│       └── benchmark_comparison.csv
│
└── logs/                          # Timestamped run logs
    └── <run_timestamp>/
        ├── pipeline.log           # Full pipeline console log
        ├── run_metadata.json      # git SHA, torch/CUDA, seed, lockfile hash, effective config
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
5. **Leave-Subjects-Out Cross-Validation**: `GroupKFold(groups=dataset)` holds out **whole subject directories** per fold — no subject ever appears in both train and validation of the same fold (frames of one person are near-duplicates). When fewer subject datasets than `k_folds` are present, the fold count is automatically reduced to the subject count with a warning; fewer than 2 subjects is an error.
6. **Held-Out Test Subjects** *(opt-in)*: set `training.test_subjects` in `codes/config.yaml` (or the `TEST_SUBJECTS` env var, comma-separated) to reserve whole subjects that are excluded from **all** CV folds and scored separately. Each fold-model is evaluated on the held-out set and the benchmark report gains a **CV** and a **TEST** section (mean ± std). The default is empty (CV only). A test subject absent from the data, or a holdout leaving fewer than 2 CV subjects, is a hard error.

---

## ⚙️ How the Pipeline Works

### 1. What Happens Automatically
When you run the pipeline:
- **Environment Setup**: Python/CUDA/Git-LFS are checked; dependencies are installed from the committed lockfile via `uv pip sync` (see § Python Packages).
- **Hardware Detection**: Detects GPU capabilities and dynamically scales batch sizes and mixed-precision strategies.
- **Dataset Discovery**: Extracts ZIPs from `requirements/` and automatically maps files using regex patterns, skipping corrupted files.
- **Preprocessing**: Converts raw TIFFs to Celsius `.npy` crops + `metadata.csv` (auto-run when `data/processed/metadata.csv` is missing; `--force-preprocess` rebuilds).
- **Model Training**: Shared `CombinedLoss` (CE+Dice) loop with **per-family recipes** — U-Net uses Adam + `ReduceLROnPlateau`; the transformer family uses AdamW + linear-warmup→cosine. The best checkpoint is selected by **val mIoU**.
- **Benchmarking**: Compares speeds, parameters, losses, and VRAM footprints (cross-validation + held-out TEST).

### 2. Execution Flow Diagram
```text
run.sh / run.bat
    │
    ├──> codes/setup.py
    │    ├─ Check Python version
    │    ├─ Check pip & Git LFS
    │    ├─ Install deps from the lockfile (uv pip sync)
    │    ├─ Check CUDA (GPU path)
    │    └─ Create directories
    │
    └──> codes/main_pipeline.py
         │
         ├──> codes/hardware_detector.py
         │    ├─ Detect GPU
         │    ├─ Validate requirements
         │    └─ Optimize batch sizes
         │
         ├──> codes/preprocess_data.py        # auto-run when data/processed/metadata.csv is absent (UB-01)
         │    └─ Raw TIFF → Celsius .npy crops + metadata.csv
         │
         ├──> codes/unified_data.py
         │    ├─ Discover S1-S10 directories
         │    ├─ Load all annotations
         │    ├─ Leave-subjects-out GroupKFold loaders
         │    └─ Hold out test_subjects (opt-in)
         │
         ├──> codes/unified_training.py
         │    └─ Train each model (per-family recipe; best by val mIoU)
         │         ├─ codes/unet_v2.py                 (unet)
         │         ├─ codes/transunet.py               (transunet)
         │         ├─ codes/swin_unet_plus_plus.py     (swin)
         │         ├─ codes/swin_pretrained.py         (swin_pretrained, opt-in)
         │         └─ codes/transunet_pretrained.py    (transunet_pretrained, opt-in)
         │
         └──> codes/benchmark_models.py
              ├─ Load trained models
              ├─ CV + TEST metrics (mean ± std)
              ├─ Generate plots
              └─ Create reports
```

### 3. Multi-Dataset Combination Details
When training, multiple dataset splits are merged:
- **Sample IDs**: Prefixed (e.g., `S1/R11104`, `S2/R21001`).
- **Data Loaders**: Multi-dataset loaders feed combined tensors to the models.
- **Splitting**: Leave-subjects-out CV — each fold's validation set is one or more whole subject datasets. Samples are *not* split proportionally per dataset.

---

## 🎛️ Configuration & CLI Options

### Advanced CLI Flags
Both `run.bat` and `run.sh` accept the following flags:
| Flag | Description |
|------|-------------|
| `--skip-extract` | Skip ZIP extraction. Use if `data/` is already populated. |
| `--skip-setup` | Skip environment setup and package installation. Use after the first run. |
| `--models <name> [<name>...]` | Train only specific models. Choices: `unet`, `transunet`, `swin`, `swin_pretrained`, `transunet_pretrained`. |
| `--skip-benchmark` | Skip the benchmarking step after training. |
| `--epochs N` | Override `codes/config.yaml` and train for `N` epochs. |
| `--resume RUN_ID` | Reuse `outputs/<RUN_ID>` and `logs/<RUN_ID>` from a previous run and continue from its checkpoints. |
| `-h`, `--help` | Show help. |

**Examples:**
```bash
# Fast validation test
./run.sh --skip-extract --skip-setup --models unet --epochs 10

# Train two specific models
./run.sh --skip-extract --models transunet swin

# Train the ImageNet-pretrained encoders (downloads timm weights on first use)
./run.sh --skip-extract --models swin_pretrained transunet_pretrained
```

#### Model choices

| Key | Architecture | Weights |
|-----|--------------|---------|
| `unet` | U-Net (CNN) | from scratch |
| `transunet` | TransUNet (ViT-B from scratch) | from scratch — small-data baseline (UB-16) |
| `swin` | Swin-UNet++ | from scratch — defective-attention baseline (UB-17) |
| `swin_pretrained` | SwinV2-tiny (timm) + conv decoder | **ImageNet-pretrained** |
| `transunet_pretrained` | R50+ViT-B/16 hybrid (timm) | **ImageNet-pretrained** |

The two pretrained encoders download ImageNet weights from the timm/HuggingFace hub on first use, adapting the 3-channel stem to 1-channel thermal input by **summing the pretrained RGB kernels** (M5). This needs network access; set `UBENCH_PRETRAINED=0` to build the identical architecture with **random** weights instead (this is what the test suite and CPU smoke use, so CI never downloads). The from-scratch trio is kept so the benchmark can compare pretrained-vs-scratch.

### Global Configuration (`codes/config.yaml`)
`codes/config.yaml` is the **single** configuration file. It is validated by a typed schema (`codes/config_schema.py`) when the pipeline starts, so an **unknown key or wrong-typed value raises immediately** instead of being silently ignored — every key below is actually consumed.

| Section | Keys | Description |
|---------|------|-------------|
| `paths` | `data_dir`, `processed_dir`, `output_dir`, `log_dir` | Filesystem roots (strict layout). |
| `model` | `image_size` (`256×256`), `num_classes` (`10`) | Model-shape parameters. |
| `training` | `learning_rate`, `num_epochs`, `k_folds`, `random_seed`, `deterministic`, `test_subjects` | Training / cross-validation and reproducibility. |
| `loss` | `ce_weight`, `dice_weight`, `class_weights` | Combined CE+Dice weights; `class_weights` = `null` (uniform, default) / `"balanced"` (inverse train-fold frequency, opt-in) / explicit per-class list. |
| `optimizer` | `name` (`adam`\|`adamw`), `weight_decay`, `betas`, `grad_clip_norm` | Global (default) optimizer recipe. |
| `scheduler` | `name` (`reduce_on_plateau`\|`warmup_cosine`), `patience`, `factor`, `warmup_frac` | Global (default) LR scheduler. |
| `recipes` | `model_families`, `families` | **Per-family overrides** (M4): each model key maps to a family (CNN or transformer); each family overrides the optimizer/scheduler. Ships with CNN=Adam/plateau, transformer=AdamW+warmup→cosine. |
| `preprocessing` | `normalization` (`fixed_range`\|`per_image_minmax`), `fixed_range_celsius`, `augmentation` | **Thermal preprocessing** (M7): `.npy` store Celsius, normalization applied at load (runtime-pure switch); `fixed_range` (default) preserves absolute temperature. `augmentation` = physical additive drift+noise + flip/affine. |
| `regions` | list of 10 class names | Portuguese facial-region labels (index 0 = background). |

**Not in config (by design):** batch size, workers, device, and AMP are chosen automatically by `hardware_detector` (there is no batch-size override key — that authority is deliberately single). Env-var overrides remain: `NUM_EPOCHS`, `K_FOLDS`, `TEST_SUBJECTS`, `UBENCH_DETERMINISTIC`, `UBENCH_PRETRAINED`.

> **Note (T3.4):** processed data now stores **Celsius** (`.npy`) with a `data/processed/preprocess_manifest.json` schema version. Data from before T3.4 (or a version mismatch) is rejected at load with an actionable error — rebuild with `./run.sh --force-preprocess`.

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
python3 -m codes.extract_data --check
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
- Re-run setup pinned to the CUDA backend (installs from the CUDA lock):
  ```bash
  UBENCH_TORCH_BACKEND=cu121 python -m codes.setup   # or cu118 for older drivers
  ```

**"Unsupported Python version" (CUDA path)**
- CUDA torch wheels are published only for Python 3.8–3.12, so a **GPU** box needs Python 3.10 or 3.11. The **CPU** install works on Python 3.13 — use it if you don't need CUDA:
  ```bash
  uv pip sync requirements/requirements.cpu.lock --torch-backend=cpu
  ```

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
