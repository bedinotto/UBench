"""
Cross-Platform Setup Script
===========================
Automatically installs dependencies and validates environment
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

try:
    from codes.logger import start_from_env as _start_log
except ImportError:
    _start_log = None  # graceful fallback if logger isn't available yet


class SetupManager:
    """Manages cross-platform setup and dependency installation"""
    
    def __init__(self):
        self.os_type = platform.system()  # 'Windows', 'Linux', 'Darwin'
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.parent
        # Dependencies are declared in pyproject.toml and installed from a
        # generated lock (UB-20b / T3.5) — never from loose ">=" floors.
        self.pyproject = self.project_root / "pyproject.toml"
        self.cpu_lock = self.project_root / "requirements" / "requirements.cpu.lock"
        self.cuda_lock = self.project_root / "requirements" / "requirements.cuda.lock"
        # Resolved torch wheel backend ("cpu" | "cu121" | "cu118"), set by
        # install_dependencies(); run() uses it to decide whether CUDA is
        # required after install.
        self.backend = None
        
    def print_header(self):
        """Print setup header"""
        print("\n" + "="*80)
        print("AUTOMATED SETUP - THERMAL FACE DETECTION PIPELINE")
        print("="*80)
        print(f"Operating System: {self.os_type}")
        print(f"Python Version:   {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print("="*80 + "\n")
    
    def check_python_version(self):
        """Verify Python version meets the >=3.8 floor.

        The CUDA-wheel ceiling (torch publishes CUDA wheels only for
        3.8–3.12) is NOT enforced here: the default install path is CPU,
        and CPU wheels exist for 3.13+.  ``_cuda_supports_python`` guards
        the CUDA path only (UB-20b / T3.5).
        """
        print("Checking Python version...")

        major = self.python_version.major
        minor = self.python_version.minor

        if major < 3 or (major == 3 and minor < 8):
            print("❌ ERROR: Python 3.8 or higher is required")
            print(f"   Current version: {sys.version}")
            return False

        print(f"✅ Python version OK: {sys.version.split()[0]}")
        return True

    def _cuda_supports_python(self):
        """CUDA torch wheels are published only for Python 3.8–3.12.

        With 3.13+, pip/uv would silently fall back to the CPU wheel from the
        CUDA index (or fail to resolve), yielding 'CUDA not available' at
        runtime.  Callers on the CUDA path must stop rather than mislead.
        """
        major, minor = self.python_version.major, self.python_version.minor
        if major == 3 and minor >= 13:
            print(f"❌ ERROR: Python {major}.{minor} has no CUDA-enabled PyTorch wheels.")
            print("   PyTorch CUDA wheels are published only for Python 3.8 – 3.12.")
            print("   Install Python 3.10 or 3.11 for a GPU training box:")
            print("     https://www.python.org/downloads/")
            return False
        return True

    
    def check_pip(self):
        """Verify pip is available"""
        print("\nChecking pip...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ pip available: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError:
            print("❌ ERROR: pip is not available")
            print("   Please install pip: https://pip.pypa.io/en/stable/installation/")
            return False

    def check_git_lfs(self):
        """Verify Git LFS is installed and large files are downloaded"""
        print("\nChecking Git LFS...")
        
        # 1. Check if git command is available
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Warning: git command not found. Cannot verify Git LFS status.")
            return True

        # 2. Check if git lfs command is available
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Git LFS available: {result.stdout.strip().splitlines()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Warning: Git LFS command not found. Please ensure Git LFS is installed.")

        # 3. Check if any zip file in requirements/ is a Git LFS pointer file
        requirements_dir = self.project_root / "requirements"
        if requirements_dir.exists():
            zip_files = list(requirements_dir.glob("*.zip"))
            for zip_file in zip_files:
                if zip_file.is_file() and zip_file.stat().st_size < 1024:
                    try:
                        with open(zip_file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(100)
                            if "version https://git-lfs" in content:
                                print(f"❌ ERROR: Git LFS pointer detected for {zip_file.name}!")
                                print("   The actual file has not been downloaded because Git LFS is not initialized.")
                                print("   Please run the following commands to pull the actual files:")
                                print("     git lfs install")
                                print("     git lfs pull")
                                print("")
                                return False
                    except Exception:
                        pass
        
        print("✅ Git LFS validation completed")
        return True
    
    def upgrade_pip(self):
        """Upgrade pip to latest version"""
        print("\nUpgrading pip...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True
            )
            print("✅ pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not upgrade pip: {e}")
            return False
    
    def _detect_cuda_driver_version(self):
        """
        Detect the CUDA driver version using nvidia-smi.
        Returns a version tuple like (12, 1) or None if not available.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                driver_ver = result.stdout.strip().split("\n")[0].strip()
                # Driver version looks like "531.61" or "552.22"
                # Map Windows driver version to max CUDA version supported
                major = int(driver_ver.split(".")[0])
                # Windows driver >= 528 supports CUDA 12.x
                # Windows driver >= 452 supports CUDA 11.x
                if major >= 528:
                    return (12, 1)
                elif major >= 452:
                    return (11, 8)
                else:
                    return None
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # Also try parsing nvidia-smi plain output for "CUDA Version"
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=10
            )
            import re
            match = re.search(r"CUDA Version: (\d+)\.(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    @staticmethod
    def _in_venv():
        """True when running inside a virtual environment (not system Python).

        setup.py refuses to install into system Python (§3.1): it can only
        install into the interpreter already active, so it requires a venv
        rather than silently creating one — creating a nested venv from inside
        run.sh's active shell would orphan the running interpreter (UB-20b).
        """
        return sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ

    def _ensure_uv(self):
        """Return True if the ``uv`` resolver is available on PATH."""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❌ ERROR: 'uv' is required to install from the lockfile but was not found.")
            print("   Install it (https://docs.astral.sh/uv/):")
            print("     curl -LsSf https://astral.sh/uv/install.sh | sh")
            print("   or:  pip install uv")
            return False

    def _resolve_backend(self):
        """Resolve the torch wheel backend: 'cpu' (default) or a CUDA tag.

        Honors an explicit ``UBENCH_TORCH_BACKEND`` override
        (cpu|cu121|cu118|auto); otherwise probes the NVIDIA driver. There is no
        aging hardcoded CUDA default — CPU is the default when no CUDA driver is
        present (UB-20b).
        """
        override = os.environ.get("UBENCH_TORCH_BACKEND", "auto").strip().lower()
        if override in ("cpu", "cu121", "cu118"):
            return override
        if override not in ("auto", ""):
            print(f"⚠️  Unknown UBENCH_TORCH_BACKEND='{override}' — falling back to auto-detect.")

        cuda_ver = self._detect_cuda_driver_version()
        if cuda_ver is None:
            return "cpu"
        if cuda_ver >= (12, 0):
            return "cu121"
        if cuda_ver >= (11, 0):
            return "cu118"
        print(f"⚠️  Detected CUDA {cuda_ver[0]}.{cuda_ver[1]} is below the 11.8 minimum — using CPU wheels.")
        return "cpu"

    def _lock_for_backend(self, backend):
        """Return the lockfile Path for *backend*, generating the CUDA lock if
        absent.

        The CUDA lock is not committed (its wheel index is unreachable from
        CI/dev, and one lock cannot pin both CPU and CUDA torch — R10): the GPU
        box materializes it from pyproject.toml on first setup.
        """
        if backend == "cpu":
            return self.cpu_lock
        if not self.cuda_lock.exists():
            print(f"   No {self.cuda_lock.name} yet — compiling it from pyproject.toml ({backend})...")
            subprocess.run(
                ["uv", "pip", "compile", str(self.pyproject),
                 "--extra", "dev", "--torch-backend", backend,
                 "-o", str(self.cuda_lock)],
                check=True,
            )
        return self.cuda_lock

    def install_dependencies(self):
        """Install the full environment from the generated lockfile via uv.

        Reproducible install (UB-20b): syncs the exact pinned closure — no loose
        ">=" floors, no ``--force-reinstall --no-deps``, no system-Python
        ``--break-system-packages``. CPU is the default path; a CUDA driver
        selects the matching wheel backend and lock. Sets ``self.backend``.
        """
        print("\nInstalling dependencies from the lockfile...")

        if not self._in_venv():
            print("❌ ERROR: not running inside a virtual environment.")
            print("   setup.py will not install into system Python (§3.1).")
            print("   Create and activate one, then re-run:")
            print("     uv venv && source .venv/bin/activate     # Windows: .venv\\Scripts\\activate")
            print("     ./run.sh   # or: python codes/setup.py")
            return False

        if not self._ensure_uv():
            return False

        self.backend = self._resolve_backend()
        if self.backend != "cpu" and not self._cuda_supports_python():
            return False

        try:
            lock = self._lock_for_backend(self.backend)
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to compile the {self.backend} lockfile: {e}")
            return False

        if not lock.exists():
            print(f"❌ ERROR: lockfile not found: {lock}")
            return False

        print(f"   Backend: {self.backend}   Lockfile: {lock.name}")
        if self.backend != "cpu":
            print("   (CUDA wheels are ~2 GB — this may take several minutes)")

        try:
            subprocess.run(
                ["uv", "pip", "sync", str(lock),
                 "--torch-backend", self.backend,
                 "--python", sys.executable],
                check=True,
            )
            print("✅ Dependencies installed from the lockfile")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to install from {lock.name}: {e}")
            return False

    
    def check_cuda(self):
        """Check CUDA availability using a subprocess.

        Using a subprocess (rather than in-process import) is essential because
        Python cannot reliably re-import a package that was installed by a child
        process after the current process has already started.  A fresh Python
        interpreter always sees the latest installed state.

        Returns:
            True  – CUDA is available
            False – PyTorch is installed but CUDA is not available
            None  – PyTorch is not installed at all
        """
        print("\nChecking CUDA availability...")

        # ---------- probe via a fresh Python subprocess ----------
        probe = subprocess.run(
            [sys.executable, "-c",
             "import torch; "
             "print('INSTALLED'); "
             "print('CUDA_OK' if torch.cuda.is_available() else 'NO_CUDA'); "
             "print(torch.__version__); "
             "print(torch.version.cuda if torch.cuda.is_available() else ''); "
             "print(torch.cuda.device_count() if torch.cuda.is_available() else '0'); "
             "[print(torch.cuda.get_device_name(i), "
             "      torch.cuda.get_device_properties(i).total_memory) "
             " for i in range(torch.cuda.device_count())]"
            ],
            capture_output=True, text=True
        )

        if probe.returncode != 0 or "INSTALLED" not in probe.stdout:
            # torch not importable → not yet installed
            print("⚠️  PyTorch not yet installed — will install with CUDA support next")
            return None

        lines = probe.stdout.strip().splitlines()
        # lines[0] = 'INSTALLED', lines[1] = 'CUDA_OK'|'NO_CUDA'
        # lines[2] = torch version, lines[3] = cuda version, lines[4] = device count
        # lines[5+] = 'GPU_name bytes' per device
        cuda_ok   = len(lines) > 1 and lines[1] == "CUDA_OK"
        torch_ver = lines[2] if len(lines) > 2 else "unknown"
        cuda_ver_str = lines[3] if len(lines) > 3 else ""
        dev_count = int(lines[4]) if len(lines) > 4 and lines[4].isdigit() else 0

        if cuda_ok:
            print(f"✅ CUDA available")
            print(f"   CUDA Version:    {cuda_ver_str}")
            print(f"   PyTorch Version: {torch_ver}")
            print(f"   GPU Count:       {dev_count}")
            for line in lines[5:]:
                parts = line.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    mem_gb = int(parts[1]) / (1024**3)
                    print(f"   GPU: {parts[0]} ({mem_gb:.1f} GB)")
            return True
        else:
            print("❌ ERROR: PyTorch is installed but CUDA is NOT available.")
            print("   This almost always means you have the CPU-only PyTorch wheel.")
            print("")
            cuda_driver = self._detect_cuda_driver_version()
            if cuda_driver is None:
                print("   nvidia-smi was not found. Check that:")
                print("     1. An NVIDIA GPU is present in your system")
                print("     2. NVIDIA drivers are installed: https://www.nvidia.com/drivers")
                print("     3. After installing drivers, re-run this setup")
            else:
                tag = "cu121" if cuda_driver >= (12, 0) else "cu118"
                print(f"   Your driver supports CUDA {cuda_driver[0]}.{cuda_driver[1]}.")
                print(f"   Re-run setup pinned to the CUDA backend (installs from the CUDA lock):")
                print(f"")
                print(f"     UBENCH_TORCH_BACKEND={tag} python codes/setup.py")
                print(f"")
                print(f"   Or visit: https://pytorch.org/get-started/locally/")
            return False

    
    def _verify_torch(self):
        """Confirm torch imports in the target interpreter (CPU path).

        The CPU install does not need CUDA — it only needs torch to load — so
        this replaces the CUDA assertion for the default path (UB-20b).
        """
        probe = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            print(f"✅ PyTorch {probe.stdout.strip()} installed (CPU build)")
            return True
        print("❌ ERROR: PyTorch failed to import after install.")
        print(probe.stderr.strip()[:500])
        return False

    def create_directories(self):
        """Create required directory structure"""
        print("\nCreating directory structure...")
        
        required_dirs = [
            "data",
            "outputs",
            "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {dir_path}")
            else:
                print(f"  Exists:  {dir_path}")
        
        print("✅ Directory structure ready")
        return True
    
    def check_data_files(self):
        """Check if required data files exist for all datasets"""
        print("\nChecking for data files...")
        
        data_dir = Path("data")
        if not data_dir.exists():
            print("  ❌ Missing 'data' directory")
            return False
            
        # Dynamically find dataset directories (S1, S2, etc.)
        dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("S") and d.name.replace("S", "", 1).isdigit()]
        
        if not dataset_dirs:
            print(f"  ❌ No dataset directories found in {data_dir}. Expected directories like S1, S2, etc.")
            return False
            
        dataset_dirs.sort(key=lambda x: int(x.name.replace("S", "", 1)))
        print(f"  ✅ Discovered {len(dataset_dirs)} dataset directories: {', '.join([d.name for d in dataset_dirs])}")
        
        missing_files = []
        for d_dir in dataset_dirs:
            dataset_name = d_dir.name
            
            # Expected file names in data/
            csv_file = data_dir / f"{dataset_name}.csv"
            polygons_file = data_dir / f"{dataset_name}_polygonal_masks.json"
            bboxes_file = data_dir / f"{dataset_name}_bounding_boxes.csv"
            
            # Alternative naming inside data/SX/
            alt_polygons_file = d_dir / "polygonal_masks.json"
            alt_bboxes_file = d_dir / "bounding_boxes.csv"
            
            # Check CSV
            if not csv_file.exists():
                missing_files.append(str(csv_file))
                print(f"  ❌ Missing: {csv_file}")
                
            # Check Polygons
            if not polygons_file.exists() and not alt_polygons_file.exists():
                missing_files.append(str(polygons_file))
                print(f"  ❌ Missing: {polygons_file} (or {alt_polygons_file})")
                
            # Check BBoxes
            if not bboxes_file.exists() and not alt_bboxes_file.exists():
                missing_files.append(str(bboxes_file))
                print(f"  ❌ Missing: {bboxes_file} (or {alt_bboxes_file})")
                
            # Check TIFF instances
            tiff_files = list(d_dir.glob("*.tiff"))
            if len(tiff_files) == 0:
                print(f"  ⚠️  WARNING: No TIFF files found in {d_dir}")
            else:
                print(f"  ✅ {dataset_name}: Found {len(tiff_files)} TIFF files")

        if missing_files:
            print("\n⚠️  WARNING: Some required data files are missing:")
            for file in missing_files:
                print(f"     - {file}")
            print("\n   The pipeline will fail without these files.")
            return False
        
        print("✅ All required data files present")
        return True
    
    def print_next_steps(self):
        """Print instructions for next steps"""
        print("\n" + "="*80)
        print("SETUP COMPLETE")
        print("="*80)
        print("\nNext Steps:")
        print("1. Ensure all TIFF thermal images are in their respective 'data/SX/' directories (e.g., data/S1, data/S2)")
        print("2. Verify metadata files are present:")
        print("   - data/SX.csv (e.g., S1.csv)")
        print("   - data/SX_polygonal_masks.json (or inside SX/)")
        print("   - data/SX_bounding_boxes.csv (or inside SX/)")
        print("\n3. Run the training pipeline:")
        print("   python main_pipeline.py")
        print("\n4. Optional: Train specific models only:")
        print("   python main_pipeline.py --models unet")
        print("   python main_pipeline.py --models transunet swin")
        print("\n5. Optional: Skip benchmarking:")
        print("   python main_pipeline.py --skip-benchmark")
        print("\n6. Optional: Custom epoch count:")
        print("   python main_pipeline.py --epochs 50")
        print("="*80 + "\n")
    
    def extract_data(self):
        """Extract training data from ZIP files in requirements/"""
        print("\nExtracting training data...")
        
        try:
            from codes.extract_data import extract_all_data
            return extract_all_data()
        except ImportError:
            print("⚠️  Could not import extract_data module — "
                  "run: python -m codes.extract_data")
            return False
    
    def run(self):
        """Execute complete setup process"""
        self.print_header()
        
        # Step 1: Check Python
        if not self.check_python_version():
            return False
        
        # Step 2: Check pip
        if not self.check_pip():
            return False
        
        # Step 2.5: Check Git LFS
        if not self.check_git_lfs():
            return False
        
        # Step 3: Upgrade pip
        self.upgrade_pip()
        
        # Step 4: Install the full environment from the lockfile. CPU is the
        # default path; a CUDA driver selects the matching wheel backend + lock.
        if not self.install_dependencies():
            print("\n❌ CRITICAL: Could not install dependencies from the lockfile.")
            return False

        # Step 5: Verify the install. CUDA is required only on the CUDA path;
        # a CPU install just needs torch to import.
        if self.backend != "cpu":
            cuda_after = self.check_cuda()
            if cuda_after is not True:
                print("\n❌ CRITICAL: CUDA still not available after installation.")
                print("   Please follow the manual install steps shown above.")
                return False
        elif not self._verify_torch():
            return False

        # Step 6: Create directories
        if not self.create_directories():
            return False

        # Step 7: Extract data from ZIP files
        self.extract_data()

        # Step 8: Check data files
        data_ready = self.check_data_files()
        
        # Print next steps
        self.print_next_steps()
        
        if not data_ready:
            print("⚠️  Setup completed with warnings - please add missing data files")
            return False
        
        print("✅ Setup completed successfully!")
        return True


def main():
    """Main entry point"""
    # Mirror all output to logs/<timestamp>/setup.log when called from the pipeline
    _logger = _start_log("setup.log") if _start_log else None

    setup = SetupManager()
    success = setup.run()

    if _logger:
        _logger.stop()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
