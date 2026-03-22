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


class SetupManager:
    """Manages cross-platform setup and dependency installation"""
    
    def __init__(self):
        self.os_type = platform.system()  # 'Windows', 'Linux', 'Darwin'
        self.python_version = sys.version_info
        self.requirements_file = Path(__file__).parent.parent / "requirements.txt"
        
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
        
        if self.python_version.major < 3 or (
            self.python_version.major == 3 and self.python_version.minor < 8
        ):
            print("❌ ERROR: Python 3.8 or higher is required")
            print(f"   Current version: {sys.version}")
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
    
    def install_dependencies(self):
        """Install all required dependencies"""
        print("\nInstalling dependencies from requirements.txt...")
        
        if not self.requirements_file.exists():
            print(f"❌ ERROR: requirements.txt not found at {self.requirements_file}")
            return False
        
        try:
            # Use --no-cache-dir to avoid cache issues
            # Use --break-system-packages for system Python (Linux)
            cmd = [
                sys.executable, "-m", "pip", "install",
                "-r", str(self.requirements_file),
                "--no-cache-dir"
            ]
            
            # On Linux, might need --break-system-packages
            if self.os_type == "Linux":
                # Try with --break-system-packages first
                try:
                    cmd_with_break = cmd + ["--break-system-packages"]
                    subprocess.run(cmd_with_break, check=True)
                    print("✅ Dependencies installed successfully")
                    return True
                except subprocess.CalledProcessError:
                    # Try without --break-system-packages
                    pass
            
            # Standard installation
            subprocess.run(cmd, check=True)
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to install dependencies: {e}")
            return False
    
    def check_cuda(self):
        """Check CUDA availability"""
        print("\nChecking CUDA availability...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                print(f"✅ CUDA available")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   PyTorch Version: {torch.__version__}")
                print(f"   GPU Count: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                
                return True
            else:
                print("⚠️  WARNING: CUDA is not available")
                print("   The pipeline will not work without CUDA")
                print("   Please install CUDA-enabled PyTorch:")
                print("   Visit: https://pytorch.org/get-started/locally/")
                return False
                
        except ImportError:
            print("⚠️  PyTorch not yet installed, will check CUDA after installation")
            return None
    
    def create_directories(self):
        """Create required directory structure"""
        print("\nCreating directory structure...")
        
        required_dirs = [
            "data",
            "outputs",
            "outputs/models",
            "outputs/plots",
            "outputs/predictions",
            "log"
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
        
        # Step 5: Install dependencies
        if not self.install_dependencies():
            return False
        
        # Step 6: Check CUDA (after installation)
        if cuda_before is None:
            cuda_after = self.check_cuda()
            if not cuda_after:
                print("\n⚠️  CRITICAL: CUDA is not available after PyTorch installation")
                print("   Please reinstall PyTorch with CUDA support")
                print("   Visit: https://pytorch.org/get-started/locally/")
        
        # Step 7: Create directories
        if not self.create_directories():
            return False
        
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
    setup = SetupManager()
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
