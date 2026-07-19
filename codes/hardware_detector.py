"""
Hardware Detection and Optimization Module
==========================================
Detects system capabilities and optimizes training parameters
"""

import platform
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

try:
    import torch
    import psutil
    import GPUtil
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install from the lockfile (see CLAUDE.md §3.1):")
    print("  uv pip sync requirements/requirements.cpu.lock --torch-backend=cpu")
    sys.exit(1)


class HardwareProfile:
    """Hardware profile with optimization parameters"""
    
    def __init__(self, gpu_name: str, gpu_memory_gb: float,
                 cpu_count: int, ram_gb: float, os_type: str = "",
                 device: str = "cuda"):
        self.gpu_name = gpu_name
        self.gpu_memory_gb = gpu_memory_gb
        self.cpu_count = cpu_count
        self.ram_gb = ram_gb
        self.os_type = os_type or platform.system()  # fallback to live detection
        self.device = device
        self.batch_sizes = self._calculate_batch_sizes()
        self.num_workers = self._calculate_workers()

    def _calculate_batch_sizes(self) -> Dict[str, int]:
        """Calculate optimal batch sizes based on GPU memory.

        Keys are the canonical model-registry names (R5/UB-05); consumers
        do hard ``[key]`` lookups, so every registered model needs an
        entry in every tier.
        """
        # The two pretrained encoders (T3.2) are heavier than the from-scratch
        # variants — SwinV2-tiny (~35M) and especially the R50+ViT-B/16 hybrid
        # (~99M, the heaviest model in the repo) — so their batch sizes are
        # sized conservatively (transunet_pretrained lowest in every tier).
        if self.device == "cpu":
            # CPU mode (UBENCH_ALLOW_CPU=1) exists for tests/CI, not
            # throughput.
            return {"unet": 4, "transunet": 2, "swin_unet_plus_plus": 2,
                    "swin_pretrained": 2, "transunet_pretrained": 2}
        if self.gpu_memory_gb < 5.5:
            print("[WARNING] GPU memory below 6GB - may encounter issues")
            return {"unet": 4, "transunet": 3, "swin_unet_plus_plus": 3,
                    "swin_pretrained": 2, "transunet_pretrained": 1}
        elif self.gpu_memory_gb < 7:
            # GTX 1660 Ti baseline (6GB)
            return {"unet": 8, "transunet": 6, "swin_unet_plus_plus": 6,
                    "swin_pretrained": 4, "transunet_pretrained": 2}
        elif self.gpu_memory_gb < 11:
            # 8GB cards (RTX 2070, RTX 3060)
            return {"unet": 12, "transunet": 8, "swin_unet_plus_plus": 8,
                    "swin_pretrained": 6, "transunet_pretrained": 4}
        elif self.gpu_memory_gb < 16:
            # 12GB cards (RTX 3060 12GB, RTX 4070)
            return {"unet": 16, "transunet": 12, "swin_unet_plus_plus": 10,
                    "swin_pretrained": 8, "transunet_pretrained": 6}
        elif self.gpu_memory_gb < 24:
            # 16GB cards (RTX 4060 Ti 16GB)
            return {"unet": 24, "transunet": 16, "swin_unet_plus_plus": 14,
                    "swin_pretrained": 12, "transunet_pretrained": 8}
        else:
            # 24GB+ cards (RTX 3090, RTX 4090, A5000)
            return {"unet": 32, "transunet": 24, "swin_unet_plus_plus": 20,
                    "swin_pretrained": 16, "transunet_pretrained": 12}
    
    def _calculate_workers(self) -> int:
        """
        Calculate the optimal number of DataLoader prefetch workers.

        DataLoader workers run on the CPU and preprocess the *next* batch
        (read TIFF → crop → resize → augment → tensor) while the GPU is
        busy training the *current* batch.  Using workers > 0 overlaps CPU
        data loading with GPU computation, which reduces idle time on both.

        Windows vs Linux difference
        ---------------------------
        Python's multiprocessing uses 'fork' on Linux (cheap — copies the
        parent process memory instantly) and 'spawn' on Windows (expensive
        — starts a brand-new Python interpreter for every worker, imports
        all modules fresh, then receives pickled data from the parent).

        Because of the higher per-worker startup and IPC cost on Windows,
        using many workers gives diminishing returns faster.  2 workers is
        the sweet spot: enough to keep the GPU fed without burning extra
        RAM and CPU cycles on worker management overhead.

        Prerequisites for Windows workers (all satisfied):
          1. if __name__ == '__main__' guard in main_pipeline.py  ✓
          2. All Dataset / Config objects are picklable              ✓
          3. DataLoader uses multiprocessing_context='spawn'        ✓
        """
        if self.device == "cpu":
            # CPU mode: workers compete with training for the same cores
            # and add IPC overhead; tests also require NUM_WORKERS=0.
            return 0
        if self.os_type == 'Windows':
            # On Windows, DataLoader uses the 'spawn' start method which
            # creates a brand-new Python interpreter per worker.  Each
            # worker holds shared-memory file mappings that count against
            # a Windows kernel limit.  With multiple models × folds, the
            # total worker count across live DataLoaders accumulates fast.
            # 2 workers is the safe sweet spot: enough to overlap I/O
            # with GPU compute, without exhausting OS resources.
            return min(2, max(1, self.cpu_count // 4))
        # Linux / Mac: when CUDA is active PyTorch forces the 'spawn'
        # multiprocessing context (each worker starts a full Python
        # interpreter), so the cost is closer to Windows than to a cheap
        # fork().  4 workers is sufficient to keep the GPU saturated for
        # image-loading workloads without excessive semaphore / IPC
        # pressure.
        optimal = min(self.cpu_count - 2, 4)
        return max(2, optimal)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_count": self.cpu_count,
            "ram_gb": self.ram_gb,
            "device": self.device,
            "batch_sizes": self.batch_sizes,
            "num_workers": self.num_workers
        }

    def __str__(self) -> str:
        return (f"Hardware Profile:\n"
                f"  Device: {self.device}\n"
                f"  GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)\n"
                f"  CPU Cores: {self.cpu_count}\n"
                f"  RAM: {self.ram_gb:.1f} GB\n"
                f"  Batch Sizes: "
                + ", ".join(f"{k}={v}" for k, v in self.batch_sizes.items())
                + f"\n  Workers: {self.num_workers}")


class HardwareDetector:
    """Detect and validate hardware capabilities"""
    
    # Minimum requirements
    MIN_GPU_MEMORY_GB = 5.5  # GTX 1660 Ti has 6GB
    MIN_RAM_GB = 8
    MIN_GPU_NAME = "GTX 1660 Ti"
    
    # Known GPU compute capabilities
    GPU_COMPUTE_CAPS = {
        "GTX 1660 Ti": 7.5,
        "RTX 2060": 7.5,
        "RTX 2070": 7.5,
        "RTX 2080": 7.5,
        "RTX 3060": 8.6,
        "RTX 3070": 8.6,
        "RTX 3080": 8.6,
        "RTX 3090": 8.6,
        "RTX 4060": 8.9,
        "RTX 4070": 8.9,
        "RTX 4080": 8.9,
        "RTX 4090": 8.9,
    }
    
    def __init__(self):
        self.os_type = platform.system()  # 'Linux', 'Windows', 'Darwin'
        self.profile: Optional[HardwareProfile] = None
        
    def detect(self) -> HardwareProfile:
        """Detect hardware and create profile"""
        print("=" * 70)
        print("HARDWARE DETECTION")
        print("=" * 70)
        
        # Detect OS
        print(f"Operating System: {self.os_type}")
        
        # Detect CUDA availability
        if not torch.cuda.is_available():
            if os.environ.get("UBENCH_ALLOW_CPU") == "1":
                return self._detect_cpu()
            print("[ERROR] CUDA is not available!")
            print("   Please install CUDA-enabled PyTorch")
            print("   Visit: https://pytorch.org/get-started/locally/")
            print("   (For tests/CI only: set UBENCH_ALLOW_CPU=1 to run on CPU.)")
            sys.exit(1)
        
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"[OK] CUDA Version: {cuda_version}")
        
        # Detect GPU
        gpu_info = self._detect_gpu()
        if gpu_info is None:
            print("[ERROR] No compatible NVIDIA GPU detected!")
            sys.exit(1)
        
        gpu_name, gpu_memory_gb = gpu_info
        
        # Validate minimum requirements
        if not self._validate_gpu(gpu_name, gpu_memory_gb):
            print(f"\n[ERROR] Minimum hardware requirements not met!")
            print(f"   Minimum GPU: {self.MIN_GPU_NAME} ({self.MIN_GPU_MEMORY_GB} GB)")
            print(f"   Detected: {gpu_name} ({gpu_memory_gb:.1f} GB)")
            sys.exit(1)
        
        # Detect CPU
        cpu_count = os.cpu_count() or 4
        print(f"CPU Cores: {cpu_count}")
        
        # Detect RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System RAM: {ram_gb:.1f} GB")
        
        if ram_gb < self.MIN_RAM_GB:
            print(f"[WARNING] RAM below recommended {self.MIN_RAM_GB} GB")
        
        # Create profile
        self.profile = HardwareProfile(gpu_name, gpu_memory_gb, cpu_count, ram_gb,
                                        os_type=self.os_type)
        
        print("\n" + "=" * 70)
        print(str(self.profile))
        print("=" * 70)
        
        return self.profile

    def _detect_cpu(self) -> HardwareProfile:
        """Build a CPU-only profile (explicit opt-in via UBENCH_ALLOW_CPU=1).

        For tests/CI and functional verification only.  AMP stays off
        (UnifiedTrainer._build_scaler returns None on CPU), pin_memory is
        False (unified_data follows torch.cuda.is_available()), workers=0
        and batch sizes are tiny.  Numbers produced in this mode are smoke
        artifacts, never benchmarks (R10).
        """
        print("!" * 70)
        print("[WARNING] UBENCH_ALLOW_CPU=1 — running WITHOUT a GPU.")
        print("          CPU mode is for testing/CI only.  Timing, memory and")
        print("          training results from this run are NOT benchmarks.")
        print("!" * 70)

        cpu_count = os.cpu_count() or 4
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"CPU Cores: {cpu_count}")
        print(f"System RAM: {ram_gb:.1f} GB")

        self.profile = HardwareProfile(
            gpu_name="CPU (no CUDA device)",
            gpu_memory_gb=0.0,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            os_type=self.os_type,
            device="cpu",
        )

        print("\n" + "=" * 70)
        print(str(self.profile))
        print("=" * 70)

        return self.profile

    def _detect_gpu(self) -> Optional[Tuple[str, float]]:
        """Detect GPU name and memory"""
        try:
            # Try using GPUtil first
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_name = gpu.name
                gpu_memory_gb = gpu.memoryTotal / 1024  # Convert MB to GB
                print(f"[OK] GPU Detected: {gpu_name}")
                print(f"   Memory: {gpu_memory_gb:.1f} GB")
                return gpu_name, gpu_memory_gb
        except Exception as e:
            print(f"GPUtil detection failed: {e}")
        
        # Fallback to PyTorch
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                # Get memory in bytes, convert to GB
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[OK] GPU Detected: {gpu_name}")
                print(f"   Memory: {gpu_memory_gb:.1f} GB")
                return gpu_name, gpu_memory_gb
        except Exception as e:
            print(f"PyTorch GPU detection failed: {e}")
        
        return None
    
    def _validate_gpu(self, gpu_name: str, gpu_memory_gb: float) -> bool:
        """Validate GPU meets minimum requirements"""
        # Check memory
        if gpu_memory_gb < self.MIN_GPU_MEMORY_GB:
            return False
        
        # Check if it's a known GPU that's too weak
        weak_gpus = [
            "GTX 1650", "GTX 1050", "GTX 1060", "GTX 960", "GTX 950",
            "MX", "Quadro K", "GT 1030"
        ]
        
        for weak in weak_gpus:
            if weak in gpu_name:
                return False
        
        return True
    
    def save_profile(self, filepath):
        """Save hardware profile to JSON"""
        if self.profile:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.profile.to_dict(), f, indent=2)
            print(f"[OK] Hardware profile saved to: {filepath}")
    
    def get_cuda_env_vars(self) -> Dict[str, str]:
        """Get recommended CUDA environment variables"""
        env_vars = {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        }
        
        # Add more aggressive memory management for lower-end GPUs
        if self.profile and self.profile.gpu_memory_gb < 8:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        
        return env_vars


def detect_and_optimize(log_dir: str = "logs") -> 'HardwareProfile':
    """Main function to detect hardware and return optimized profile

    Args:
        log_dir: Directory to save the hardware profile to.
    """
    detector = HardwareDetector()
    profile = detector.detect()

    # Set CUDA environment variables (meaningless in CPU mode)
    if profile.device != "cpu":
        env_vars = detector.get_cuda_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")

    # cuDNN determinism/benchmark is owned solely by seed_everything, driven by
    # the `deterministic` config flag (UB-20a/M6). Setting it here too would
    # reintroduce the previous contradiction, so it is intentionally not touched.

    # Save profile using pathlib (cross-platform)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    detector.save_profile(log_path / "hardware_profile.json")

    return profile


if __name__ == "__main__":
    profile = detect_and_optimize()
    print("\n✅ Hardware detection completed successfully!")
