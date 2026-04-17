"""
Hardware Detection and Optimization Module
==========================================
Detects system capabilities and optimizes training parameters
"""

import platform
import subprocess
import sys
import os
from typing import Dict, Tuple, Optional
import json

try:
    import torch
    import psutil
    import GPUtil
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


class HardwareProfile:
    """Hardware profile with optimization parameters"""
    
    def __init__(self, gpu_name: str, gpu_memory_gb: float,
                 cpu_count: int, ram_gb: float, os_type: str = ""):
        self.gpu_name = gpu_name
        self.gpu_memory_gb = gpu_memory_gb
        self.cpu_count = cpu_count
        self.ram_gb = ram_gb
        self.os_type = os_type or platform.system()  # fallback to live detection
        self.batch_sizes = self._calculate_batch_sizes()
        self.num_workers = self._calculate_workers()
        
    def _calculate_batch_sizes(self) -> Dict[str, int]:
        """Calculate optimal batch sizes based on GPU memory"""
        if self.gpu_memory_gb < 5.5:
            print("[WARNING] GPU memory below 6GB - may encounter issues")
            return {"unet": 4, "transunet": 3, "swin": 3}
        elif self.gpu_memory_gb < 7:
            # GTX 1660 Ti baseline (6GB)
            return {"unet": 8, "transunet": 6, "swin": 6}
        elif self.gpu_memory_gb < 11:
            # 8GB cards (RTX 2070, RTX 3060)
            return {"unet": 12, "transunet": 8, "swin": 8}
        elif self.gpu_memory_gb < 16:
            # 12GB cards (RTX 3060 12GB, RTX 4070)
            return {"unet": 16, "transunet": 12, "swin": 10}
        elif self.gpu_memory_gb < 24:
            # 16GB cards (RTX 4060 Ti 16GB)
            return {"unet": 24, "transunet": 16, "swin": 14}
        else:
            # 24GB+ cards (RTX 3090, RTX 4090, A5000)
            return {"unet": 32, "transunet": 24, "swin": 20}
    
    def _calculate_workers(self) -> int:
        """Calculate optimal number of data loading workers"""
        # Windows does not support multiprocessing with DataLoader in spawn context
        if self.os_type == 'Windows':
            return 0
        # Use at most 4 workers per GPU, capped by CPU count
        optimal = min(self.cpu_count - 2, 8)
        return max(2, optimal)  # At least 2 workers
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_count": self.cpu_count,
            "ram_gb": self.ram_gb,
            "batch_sizes": self.batch_sizes,
            "num_workers": self.num_workers
        }
    
    def __str__(self) -> str:
        return (f"Hardware Profile:\n"
                f"  GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)\n"
                f"  CPU Cores: {self.cpu_count}\n"
                f"  RAM: {self.ram_gb:.1f} GB\n"
                f"  Batch Sizes: UNet={self.batch_sizes['unet']}, "
                f"TransUNet={self.batch_sizes['transunet']}, "
                f"Swin={self.batch_sizes['swin']}\n"
                f"  Workers: {self.num_workers}")


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
            print("[ERROR] CUDA is not available!")
            print("   Please install CUDA-enabled PyTorch")
            print("   Visit: https://pytorch.org/get-started/locally/")
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
    
    def save_profile(self, filepath: str):
        """Save hardware profile to JSON"""
        if self.profile:
            with open(filepath, 'w') as f:
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
    
    # Set CUDA environment variables
    env_vars = detector.get_cuda_env_vars()
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # Save profile
    os.makedirs(log_dir, exist_ok=True)
    detector.save_profile(os.path.join(log_dir, "hardware_profile.json"))
    
    return profile


if __name__ == "__main__":
    profile = detect_and_optimize()
    print("\n✅ Hardware detection completed successfully!")
