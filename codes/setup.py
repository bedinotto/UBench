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

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class SetupManager:
    """Manages cross-platform setup and dependency installation"""
    
    def __init__(self):
        self.os_type = platform.system()  # 'Windows', 'Linux', 'Darwin'
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements" / "requirements.txt"
        
    def print_header(self):
        """Print setup header"""
        print("\n" + "="*80)
        print("AUTOMATED SETUP - THERMAL FACE DETECTION PIPELINE")
        print("="*80)
        print(f"Operating System: {self.os_type}")
        print(f"Python Version:   {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print("="*80 + "\n")
    
    def check_python_version(self):
        """Verify Python version meets requirements"""
        print("Checking Python version...")

        major = self.python_version.major
        minor = self.python_version.minor

        if major < 3 or (major == 3 and minor < 8):
            print("❌ ERROR: Python 3.8 or higher is required")
            print(f"   Current version: {sys.version}")
            return False

        # PyTorch CUDA wheels are only published for Python 3.8 – 3.12.
        # Python 3.13+ has no CUDA-enabled wheels on download.pytorch.org;
        # pip will silently install the CPU-only build instead.
        if major == 3 and minor >= 13:
            print(f"❌ ERROR: Python {major}.{minor} is not supported for CUDA PyTorch.")
            print("   PyTorch CUDA wheels are published only for Python 3.8 – 3.12.")
            print("   With Python 3.13+, pip will silently install the CPU-only build,")
            print("   which will cause 'CUDA not available' errors at runtime.")
            print("")
            print("   Please install a supported Python version (3.10 or 3.11 recommended):")
            print("     https://www.python.org/downloads/")
            print("")
            print("   After installing, make sure the new Python is first in your PATH")
            print("   and re-run run.bat.")
            return False

        print(f"✅ Python version OK: {sys.version.split()[0]}")
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

    def _pytorch_already_has_cuda(self):
        """Return True if an already-installed PyTorch build has CUDA support.
        Uses a subprocess so that freshly-installed wheels are always visible.
        """
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() == "True"

    def install_pytorch_cuda(self):
        """
        Install the correct CUDA-enabled PyTorch build.
        Plain `pip install torch` always downloads the CPU-only wheel;
        CUDA builds require a special --index-url from pytorch.org.
        """
        print("\nInstalling PyTorch with CUDA support...")

        # Skip if CUDA torch is already present
        if self._pytorch_already_has_cuda():
            import torch
            print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda} already installed")
            return True

        # Detect which CUDA SDK the GPU driver supports
        cuda_ver = self._detect_cuda_driver_version()

        if cuda_ver is None:
            print("⚠️  nvidia-smi not found or NVIDIA driver not installed.")
            print("   Cannot auto-detect CUDA version.")
            print("   Defaulting to CUDA 12.1 wheel — install NVIDIA drivers first if this fails.")
            cuda_ver = (12, 1)

        # Choose the best matching PyTorch CUDA build
        if cuda_ver >= (12, 0):
            cuda_tag = "cu121"
            cuda_label = "CUDA 12.1"
        elif cuda_ver >= (11, 0):
            cuda_tag = "cu118"
            cuda_label = "CUDA 11.8"
        else:
            print(f"❌ ERROR: CUDA {cuda_ver[0]}.{cuda_ver[1]} is below the minimum CUDA 11.8.")
            print("   Please update your NVIDIA drivers.")
            print("   Visit: https://www.nvidia.com/drivers")
            return False

        index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        print(f"   Detected driver supports {cuda_label} → installing PyTorch {cuda_tag} build")
        print(f"   Index URL: {index_url}")
        print("   (This may take several minutes — PyTorch wheels are ~2 GB)")
        print()

        # --force-reinstall --no-deps ensures we replace any existing CPU-only
        # torch wheel. Without this, pip says "Requirement already satisfied"
        # and skips the download, leaving the CPU build in place.
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-deps",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url,
            "--no-cache-dir",
        ]

        try:
            subprocess.run(cmd, check=True)
            print("✅ PyTorch (CUDA) installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to install PyTorch with CUDA: {e}")
            print("   Please install manually:")
            print(f"   pip install torch torchvision torchaudio --index-url {index_url}")
            print("   Or visit: https://pytorch.org/get-started/locally/")
            return False


    def install_other_dependencies(self):
        """Install non-PyTorch dependencies from requirements.txt"""
        print("\nInstalling other dependencies from requirements.txt...")

        if not self.requirements_file.exists():
            print(f"❌ ERROR: requirements.txt not found at {self.requirements_file}")
            return False

        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(self.requirements_file),
            "--no-cache-dir",
        ]

        if self.os_type == "Linux":
            try:
                subprocess.run(cmd + ["--break-system-packages"], check=True)
                print("✅ Other dependencies installed successfully")
                return True
            except subprocess.CalledProcessError:
                pass  # Fall through to standard install

        try:
            subprocess.run(cmd, check=True)
            print("✅ Other dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to install dependencies: {e}")
            return False

    # Legacy alias kept for backward compatibility
    def install_dependencies(self):
        """Install all dependencies (PyTorch CUDA first, then the rest)"""
        if not self.install_pytorch_cuda():
            return False
        return self.install_other_dependencies()

    
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
                print(f"   Run the following command to fix this:")
                print(f"")
                print(f"     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{tag}")
                print(f"")
                print(f"   Or visit: https://pytorch.org/get-started/locally/")
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
            try:
                from extract_data import extract_all_data
                return extract_all_data()
            except ImportError:
                print("⚠️  Could not import extract_data module — "
                      "please run codes/extract_data.py manually")
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
        
        # Step 3: Upgrade pip
        self.upgrade_pip()
        
        # Step 4: Check CUDA (before installation)
        cuda_before = self.check_cuda()
        
        # Step 5: Install PyTorch with CUDA, then other dependencies
        if not self.install_pytorch_cuda():
            print("\n❌ CRITICAL: Could not install CUDA-enabled PyTorch.")
            print("   The pipeline requires a CUDA-capable NVIDIA GPU and matching PyTorch.")
            print("   See https://pytorch.org/get-started/locally/ for manual install instructions.")
            return False

        if not self.install_other_dependencies():
            return False

        # Step 6: Verify CUDA is now available
        # Note: check_cuda() returns True (CUDA ok), False (torch present, no CUDA),
        # or None (torch not importable). All non-True outcomes are failures here.
        cuda_after = self.check_cuda()
        if cuda_after is not True:
            print("\n❌ CRITICAL: CUDA still not available after installation.")
            print("   Please follow the manual install steps shown above.")
            return False

        
        # Step 7: Create directories
        if not self.create_directories():
            return False
        
        # Step 8: Extract data from ZIP files
        self.extract_data()
        
        # Step 9: Check data files
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
    setup = SetupManager()
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
