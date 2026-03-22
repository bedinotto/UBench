# QUICK START GUIDE

## 🚀 Get Started in 3 Steps

### Step 1: Organize Your Data

Place your thermal imaging datasets in the `data/` directory:

```
data/
├── S1/
│   ├── R11104.tiff
│   ├── R11105.tiff
│   └── ... (more TIFF files)
├── S1_polygonal_masks.json
├── S1_bounding_boxes.csv
│
├── S2/
│   └── ... (TIFF files)
├── S2_polygonal_masks.json
├── S2_bounding_boxes.csv
│
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

```
outputs/
├── models/              # Trained weights (.pth files)
├── plots/               # Training curves and comparisons
└── benchmark_comparison.csv

log/
├── hardware_profile.json
├── *_metrics.json
└── benchmark_report.txt
```

## 📝 What Happens Automatically

1. ✅ **Hardware Detection**
   - Detects your GPU
   - Validates minimum GTX 1660 Ti
   - Optimizes batch sizes for your hardware

2. ✅ **Dataset Discovery**
   - Scans `data/` for S1, S2, ..., S10
   - Loads annotations for each dataset
   - Combines all samples

3. ✅ **Environment Setup**
   - Installs all dependencies
   - Validates CUDA
   - Creates output directories

4. ✅ **Model Training**
   - Trains U-Net (~50 min on GTX 1660 Ti)
   - Trains TransUNet (~85 min on GTX 1660 Ti)
   - Trains Swin-UNet++ (~95 min on GTX 1660 Ti)

5. ✅ **Benchmarking**
   - Compares all models
   - Generates comparison plots
   - Creates detailed reports

**Total Time: ~4 hours on GTX 1660 Ti for 100 epochs**

## ⚙️ Common Options

Train only one model:
```bash
./run.sh --models unet
```

Quick test run (10 epochs):
```bash
./run.sh --epochs 10
```

Skip setup if already installed:
```bash
./run.sh --skip-setup
```

Skip benchmarking:
```bash
./run.sh --skip-benchmark
```

Combine options:
```bash
./run.sh --skip-setup --models unet transunet --epochs 50
```

## 🔍 Checking Your Data

Before running, verify your data structure:

**Check for dataset directories:**
```bash
ls -d data/S*
# Should show: data/S1  data/S2  data/S3  etc.
```

**Check for TIFF files:**
```bash
ls data/S1/*.tiff | wc -l
# Shows number of TIFF files in S1
```

**Check for annotation files:**
```bash
ls data/*_polygonal_masks.json
ls data/*_bounding_boxes.csv
# Should show files for each dataset
```

## 🐛 Quick Troubleshooting

**"No dataset directories found"**
- Check that you have at least one S1, S2, etc. directory
- Directory names must be exactly `S1`, `S2` (case-sensitive)

**"No polygonal masks found"**
- Check for `S1_polygonal_masks.json` in data/
- OR check for `polygonal_masks.json` inside data/S1/

**"CUDA not available"**
- Install/update NVIDIA drivers
- Install CUDA toolkit
- Reinstall PyTorch with CUDA

**"Minimum hardware requirements not met"**
- You need GTX 1660 Ti (6GB) or better
- Check GPU: `nvidia-smi`

## 📖 Full Documentation

For complete documentation, see:
- `README.md` - Full user guide
- `PROJECT_STRUCTURE.md` - Directory organization
- `config.yaml` - Configuration options

## 🎯 Expected Output

After successful run, you'll see:

```
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

## 💡 Tips

1. **Start Small**: Test with 1 dataset and 10 epochs first
   ```bash
   ./run.sh --epochs 10
   ```

2. **Monitor GPU**: Open another terminal and run
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check Logs**: If training fails, check
   ```bash
   cat log/*_metrics.json
   ```

4. **Free Memory**: Close other applications before training

---

**Ready to start? Run: `./run.sh`**
