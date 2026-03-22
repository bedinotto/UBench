# 🎉 UPDATED PIPELINE - VERSION 2.0

## ✅ Major Updates Implemented

### 1. **Multi-Dataset Support (S1-S10)**

The pipeline now **automatically discovers and trains on multiple datasets**:

- ✅ Scans `data/` directory for S1, S2, S3, ..., S10 folders
- ✅ Loads annotations from each dataset
- ✅ Combines all samples with unique IDs (e.g., S1/R11104, S2/R21001)
- ✅ Reports per-dataset distribution
- ✅ Supports flexible file naming conventions

**Example Output:**
```
DISCOVERING AND LOADING DATASETS
======================================================================
✅ Found 3 dataset(s): S1, S2, S3

Loading S1...
  ✅ Loaded 734 polygonal masks
  
Loading S2...
  ✅ Loaded 500 polygonal masks
  
Loading S3...
  ✅ Loaded 612 polygonal masks

======================================================================
✅ TOTAL: 1846 samples across 3 dataset(s)
======================================================================

Samples per dataset:
  S1: 734 total (587 train, 147 val)
  S2: 500 total (400 train, 100 val)
  S3: 612 total (490 train, 122 val)
```

### 2. **Clean Code Organization**

All Python code moved to `/codes` directory for a clean root:

**New Structure:**
```
project/
├── run.sh                    # Entry point
├── run.bat                   # Entry point
├── requirements.txt          # Dependencies
├── config.yaml               # Config
├── README.md                 # Docs
│
├── codes/                    # ← ALL CODE HERE
│   ├── setup.py
│   ├── main_pipeline.py
│   ├── hardware_detector.py
│   ├── unified_data.py       # ← UPDATED: Multi-dataset support
│   ├── unified_training.py
│   ├── benchmark_models.py
│   ├── unet_v2.py
│   ├── transunet.py
│   └── swin_unet_plus_plus.py
│
├── data/                     # ← INPUT
└── outputs/                  # ← RESULTS
```

**Root directory now only contains:**
- Entry scripts (run.sh, run.bat)
- Configuration (config.yaml, requirements.txt)
- Documentation (README.md, QUICKSTART.md)

## 📁 Data Organization

### Supported Naming Conventions

**Option 1: Files in data/ root (Recommended)**
```
data/
├── S1/
│   └── *.tiff
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
├── S1.csv                    # Optional
│
├── S2/
│   └── *.tiff
├── S2_polygonal_masks.json
└── S2_bounding_boxes.csv
```

**Option 2: Files inside dataset directory**
```
data/
├── S1/
│   ├── *.tiff
│   ├── polygonal_masks.json
│   └── bounding_boxes.csv
│
└── S2/
    ├── *.tiff
    ├── polygonal_masks.json
    └── bounding_boxes.csv
```

**Both work! The pipeline checks both locations automatically.**

## 🔄 Key Changes in Code

### unified_data.py
- ✅ New `MultiDirectoryDataLoader` class
- ✅ `discover_datasets()` method - finds S1-S10 automatically
- ✅ `load_dataset_annotations()` - loads per-dataset annotations
- ✅ Combines all polygons with unique prefixed IDs
- ✅ Supports both naming conventions
- ✅ Reports comprehensive dataset statistics

### Import Updates
All imports updated to work from `/codes` directory:
```python
# Before:
from unified_data import Config

# After:
from codes.unified_data import Config
```

### Run Scripts
Updated to execute code from `/codes`:
```bash
# Linux/Mac
$PYTHON_CMD codes/main_pipeline.py

# Windows
%PYTHON_CMD% codes\main_pipeline.py
```

## 🚀 Usage Examples

### Basic Usage (All Datasets, All Models)
```bash
./run.sh
```

### Train on Specific Datasets
The pipeline automatically uses **all discovered datasets**. If you only want specific datasets, simply remove unwanted folders from `data/`:

```bash
# Train only on S1 and S2
# Just ensure only S1/ and S2/ exist in data/
./run.sh
```

### Train Specific Models
```bash
# Train only U-Net
./run.sh --models unet

# Train TransUNet and Swin-UNet++
./run.sh --models transunet swin
```

### Quick Test (10 epochs)
```bash
./run.sh --epochs 10
```

### Skip Setup
```bash
./run.sh --skip-setup
```

## 📊 What Gets Combined

When training with multiple datasets:

1. **Sample IDs** are prefixed: `S1/R11104`, `S2/R21001`, etc.
2. **All polygons** combined into single dictionary
3. **All bounding boxes** combined with dataset column
4. **Train/val split** stratified across all datasets
5. **Metrics** reported per-dataset

## 🎯 Benefits

### 1. Automatic Discovery
- No manual configuration needed
- Add new dataset? Just drop it in `data/`
- Pipeline finds and uses it automatically

### 2. Larger Training Set
- Combines all available data
- Better model generalization
- More robust validation

### 3. Per-Dataset Tracking
- See sample distribution
- Track performance per dataset
- Identify dataset-specific issues

### 4. Clean Organization
- Root directory uncluttered
- All code in `/codes`
- Easy to navigate

## 🔍 Verification Checklist

Before running, verify:

- [ ] At least one Sx directory exists in `data/`
- [ ] Each dataset has TIFF files
- [ ] Each dataset has `Sx_polygonal_masks.json` (or inside directory)
- [ ] Each dataset has `Sx_bounding_boxes.csv` (or inside directory)
- [ ] All files in `/codes` directory
- [ ] `run.sh` or `run.bat` in root

Quick check:
```bash
# Check for datasets
ls -d data/S*

# Check for TIFF files
ls data/S1/*.tiff | wc -l

# Check for annotations
ls data/*_polygonal_masks.json
ls data/*_bounding_boxes.csv

# Check code organization
ls codes/*.py
```

## 📖 Documentation

Three comprehensive guides provided:

1. **QUICKSTART.md** - Get started in 3 steps
2. **README.md** - Complete user guide
3. **PROJECT_STRUCTURE.md** - Directory organization details

## 🐛 Troubleshooting

### "No dataset directories found"
- Ensure at least one S1-S10 directory exists
- Directory names must be exactly `S1`, `S2`, etc. (case-sensitive)

### "No polygonal masks found for S1"
- Check `data/S1_polygonal_masks.json`
- OR check `data/S1/polygonal_masks.json`
- Ensure JSON is properly formatted

### Import errors
- Ensure all .py files are in `/codes` directory
- Run from project root, not from `/codes`

## 🎉 Summary

**Version 2.0 delivers:**
- ✅ Multi-dataset support (S1-S10)
- ✅ Automatic dataset discovery
- ✅ Clean code organization
- ✅ Flexible file naming
- ✅ Comprehensive documentation
- ✅ Per-dataset statistics

**All while maintaining:**
- ✅ Hardware optimization
- ✅ Cross-platform support
- ✅ Unified training pipeline
- ✅ Comprehensive benchmarking

---

**Ready to use! Run: `./run.sh`**
