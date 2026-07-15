# Environment Setup & CUDA Management

> 34 nodes · cohesion 0.09

## Key Concepts

- **SetupManager** (20 connections) — `codes/setup.py`
- **.run()** (17 connections) — `codes/setup.py`
- **.install_pytorch_cuda()** (6 connections) — `codes/setup.py`
- **._detect_cuda_driver_version()** (5 connections) — `codes/setup.py`
- **.check_cuda()** (4 connections) — `codes/setup.py`
- **.extract_data()** (4 connections) — `codes/setup.py`
- **.install_dependencies()** (4 connections) — `codes/setup.py`
- **.install_other_dependencies()** (4 connections) — `codes/setup.py`
- **._pytorch_already_has_cuda()** (4 connections) — `codes/setup.py`
- **.check_data_files()** (3 connections) — `codes/setup.py`
- **.check_git_lfs()** (3 connections) — `codes/setup.py`
- **.check_pip()** (3 connections) — `codes/setup.py`
- **.check_python_version()** (3 connections) — `codes/setup.py`
- **.create_directories()** (3 connections) — `codes/setup.py`
- **.print_next_steps()** (3 connections) — `codes/setup.py`
- **.upgrade_pip()** (3 connections) — `codes/setup.py`
- **.print_header()** (2 connections) — `codes/setup.py`
- **Upgrade pip to latest version** (1 connections) — `codes/setup.py`
- **Detect the CUDA driver version using nvidia-smi.         Returns a version tuple** (1 connections) — `codes/setup.py`
- **Return True if an already-installed PyTorch build has CUDA support.         Uses** (1 connections) — `codes/setup.py`
- **Install the correct CUDA-enabled PyTorch build.         Plain `pip install torch** (1 connections) — `codes/setup.py`
- **Manages cross-platform setup and dependency installation** (1 connections) — `codes/setup.py`
- **Install non-PyTorch dependencies from requirements.txt** (1 connections) — `codes/setup.py`
- **Install all dependencies (PyTorch CUDA first, then the rest)** (1 connections) — `codes/setup.py`
- **Check CUDA availability using a subprocess.          Using a subprocess (rather** (1 connections) — `codes/setup.py`
- *... and 9 more nodes in this community*

## Relationships

- [Logging Infrastructure (TeeLogger)](Logging_Infrastructure_%28TeeLogger%29.md) (3 shared connections)
- [Data Extraction & LFS Management](Data_Extraction_%26_LFS_Management.md) (1 shared connections)

## Source Files

- `codes/setup.py`

## Audit Trail

- EXTRACTED: 108 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*