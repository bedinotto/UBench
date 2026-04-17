"""
Unified Data Loading Module - Multi-Directory Support
=====================================================
Shared data loading and preprocessing for all models
Supports multiple dataset directories (S1, S2, ..., S10)
"""

import os
import platform
import numpy as np
import pandas as pd
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def _raw_to_celsius(raw):
    """Convert raw thermal sensor value to degrees Celsius.

    Defined at module level (not as a lambda) so that it is picklable
    by the 'spawn' multiprocessing context used by DataLoader workers
    on Windows and when CUDA is active on Linux/Mac.
    """
    return (raw / 100) - 273.15


class Config:
    """Unified configuration for all models"""
    
    # Data paths (STRICT structure)
    DATA_DIR = Path("data")
    
    # Output paths (STRICT structure)
    OUTPUT_DIR = Path("outputs")
    LOG_DIR = Path("logs")
    
    # Model parameters
    IMAGE_SIZE = (256, 256)
    NUM_CLASSES = 10
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    
    # Thermal conversion  (uses a named function – lambdas are not picklable)
    RAW_TO_CELSIUS = np.vectorize(_raw_to_celsius)
    
    # Region names
    REGION_NAMES = [
        "background",
        "Contorno inferior do Rosto",
        "Sombrancelha esquerda",
        "Sombrancelha direita",
        "Nariz",
        "Olho esquerdo",
        "Olho direito",
        "Boca",
        "Labios",
        "Testa"
    ]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, output_dir: str = None, log_dir: str = None):
        """Initialize and validate paths
        
        Args:
            output_dir: Override for output directory (e.g., timestamped subdir)
            log_dir: Override for log directory (e.g., timestamped subdir)
        """
        if output_dir:
            self.OUTPUT_DIR = Path(output_dir)
        if log_dir:
            self.LOG_DIR = Path(log_dir)
        self._validate_paths()
        self._create_output_dirs()
        if 'NUM_EPOCHS' in os.environ:
            self.NUM_EPOCHS = int(os.environ['NUM_EPOCHS'])
    
    def _validate_paths(self):
        """Validate required data paths exist"""
        if not self.DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.DATA_DIR}\n"
                f"Please create 'data/' directory and place dataset folders inside"
            )
    
    def _create_output_dirs(self):
        """Create output directories if they don't exist"""
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "models").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "predictions").mkdir(exist_ok=True)


class MultiDirectoryDataLoader:
    """
    Data loader that automatically discovers and loads from multiple dataset directories
    Supports S1, S2, S3, ..., S10 directory structure
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.datasets = {}  # Dictionary to store data from each directory
        self.all_annotations = None
        self.all_polygons = {}
        self.all_bboxes = None
        
    def discover_datasets(self) -> List[str]:
        """
        Discover all Sx directories in data folder
        
        Returns:
            List of dataset directory names (e.g., ['S1', 'S2', 'S3'])
        """
        discovered = []
        
        # Look for S* directories
        if self.config.DATA_DIR.exists():
            for dir_path in self.config.DATA_DIR.iterdir():
                if dir_path.is_dir() and dir_path.name.startswith("S") and dir_path.name.replace("S", "", 1).isdigit():
                    discovered.append(dir_path.name)
            
            # Sort numerically
            discovered.sort(key=lambda x: int(x.replace("S", "", 1)))
        
        if not discovered:
            raise FileNotFoundError(
                f"No dataset directories found in {self.config.DATA_DIR}\n"
                f"Expected directories: S1, S2, S3, ..., S10"
            )
        
        return sorted(discovered)
    
    def load_dataset_annotations(self, dataset_name: str):
        """
        Load annotations for a specific dataset directory
        
        Args:
            dataset_name: Name of dataset directory (e.g., 'S1')
        
        Returns:
            Dictionary with annotations, polygons, and bboxes
        """
        dataset_path = self.config.DATA_DIR / dataset_name
        
        # Expected file names
        csv_file = self.config.DATA_DIR / f"{dataset_name}.csv"
        polygons_file = self.config.DATA_DIR / f"{dataset_name}_polygonal_masks.json"
        bboxes_file = self.config.DATA_DIR / f"{dataset_name}_bounding_boxes.csv"
        
        # Alternative naming (inside directory)
        alt_polygons_file = dataset_path / "polygonal_masks.json"
        alt_bboxes_file = dataset_path / "bounding_boxes.csv"
        
        dataset_info = {
            'name': dataset_name,
            'path': dataset_path,
            'annotations': None,
            'polygons': None,
            'bboxes': None
        }
        
        # Load CSV annotations (optional)
        if csv_file.exists():
            dataset_info['annotations'] = pd.read_csv(csv_file)
        
        # Load polygonal masks (required)
        polygons_path = polygons_file if polygons_file.exists() else alt_polygons_file
        if polygons_path.exists():
            with open(polygons_path, 'r') as f:
                dataset_info['polygons'] = json.load(f)
        else:
            print(f"⚠️  Warning: No polygonal masks found for {dataset_name}")
            print(f"    Checked: {polygons_file}")
            print(f"    Checked: {alt_polygons_file}")
            dataset_info['polygons'] = {}
        
        # Load bounding boxes (optional)
        bboxes_path = bboxes_file if bboxes_file.exists() else alt_bboxes_file
        if bboxes_path.exists():
            dataset_info['bboxes'] = pd.read_csv(bboxes_path)
        
        return dataset_info
    
    def load_annotations(self):
        """Load annotations from all discovered dataset directories"""
        print("="*70)
        print("DISCOVERING AND LOADING DATASETS")
        print("="*70)
        
        # Discover datasets
        dataset_names = self.discover_datasets()
        print(f"\n✅ Found {len(dataset_names)} dataset(s): {', '.join(dataset_names)}")
        
        # Load each dataset
        all_annotations_list = []
        all_bboxes_list = []
        total_samples = 0
        
        for dataset_name in dataset_names:
            print(f"\nLoading {dataset_name}...")
            
            dataset_info = self.load_dataset_annotations(dataset_name)
            self.datasets[dataset_name] = dataset_info
            
            # Combine polygons (prefixing sample IDs with dataset name)
            if dataset_info['polygons']:
                for sample_id, regions in dataset_info['polygons'].items():
                    # Create unique key: S1/R11104, S2/R21234, etc.
                    unique_id = f"{dataset_name}/{sample_id}"
                    self.all_polygons[unique_id] = regions
                
                print(f"  ✅ Loaded {len(dataset_info['polygons'])} polygonal masks")
                total_samples += len(dataset_info['polygons'])
            
            # Combine annotations
            if dataset_info['annotations'] is not None:
                # Add dataset column
                df = dataset_info['annotations'].copy()
                df['dataset'] = dataset_name
                all_annotations_list.append(df)
                print(f"  ✅ Loaded {len(df)} annotations")
            
            # Combine bounding boxes
            if dataset_info['bboxes'] is not None:
                df = dataset_info['bboxes'].copy()
                df['dataset'] = dataset_name
                # Prefix IDs with dataset name
                df['ID'] = dataset_name + '/' + df['ID'].astype(str)
                all_bboxes_list.append(df)
                print(f"  ✅ Loaded {len(df)} bounding boxes")
        
        # Combine all annotations
        if all_annotations_list:
            self.all_annotations = pd.concat(all_annotations_list, ignore_index=True)
        
        if all_bboxes_list:
            self.all_bboxes = pd.concat(all_bboxes_list, ignore_index=True)
        
        print("\n" + "="*70)
        print(f"✅ TOTAL: {total_samples} samples across {len(dataset_names)} dataset(s)")
        print("="*70 + "\n")
        
        return self
    
    def load_thermal_image_from_tiff(self, tiff_path: str) -> np.ndarray:
        """Load thermal image from TIFF file"""
        thermal_raw = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if thermal_raw is None:
            raise FileNotFoundError(f"Could not load TIFF file: {tiff_path}")
        thermal_celsius = self.config.RAW_TO_CELSIUS(thermal_raw)
        return thermal_celsius.astype(np.float32)
    
    def get_tiff_path(self, sample_id: str):
        """Get the actual TIFF path for a given sample ID, handling variations"""
        import re
        # Parse dataset and ID
        if '/' in sample_id:
            dataset_name, img_id = sample_id.split('/', 1)
        else:
            dataset_name = 'S1'
            img_id = sample_id
            
        # Strategy 1: Extract digits (most reliable across all datasets)
        digits = "".join(re.findall(r'\d+', img_id))
        if digits:
            expected_tiff = f"R{digits}.tiff"
            tiff_path = self.config.DATA_DIR / dataset_name / expected_tiff
            if tiff_path.exists():
                return tiff_path
                
        # Strategy 2: Direct approach (fallback)
        direct_path = self.config.DATA_DIR / dataset_name / f"{img_id}.tiff"
        if direct_path.exists():
            return direct_path
            
        return None

    def load_thermal_image(self, sample_id: str) -> np.ndarray:
        """
        Load thermal image for a given sample ID
        
        Args:
            sample_id: Sample identifier with dataset prefix (e.g., 'S1/R11104')
        
        Returns:
            Thermal image as numpy array in Celsius
        """
        tiff_path = self.get_tiff_path(sample_id)
        if tiff_path:
            return self.load_thermal_image_from_tiff(str(tiff_path))
        raise FileNotFoundError(f"Thermal image not found for ID: {sample_id}")
    
    def crop_to_bbox(self, thermal_img: np.ndarray,
                     sample_id: str, padding: int = 10) -> np.ndarray:
        """Crop thermal image to bounding box region"""
        if self.all_bboxes is None or sample_id not in self.all_bboxes['ID'].values:
            return thermal_img
        
        bbox = self.all_bboxes[self.all_bboxes['ID'] == sample_id].iloc[0]
        
        min_x = max(0, int(bbox['min_x']) - padding)
        min_y = max(0, int(bbox['min_y']) - padding)
        max_x = min(thermal_img.shape[1], int(bbox['max_x']) + padding)
        max_y = min(thermal_img.shape[0], int(bbox['max_y']) + padding)
        
        return thermal_img[min_y:max_y, min_x:max_x]
    
    def create_segmentation_mask(self, sample_id: str,
                                 img_shape: Tuple[int, int],
                                 offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """Create segmentation mask from polygon annotations"""
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        if sample_id not in self.all_polygons:
            return mask
        
        regions = self.all_polygons[sample_id]
        offset_x, offset_y = offset
        
        for region_idx, region_name in enumerate(self.config.REGION_NAMES[1:], 1):
            if region_name in regions:
                polygon = np.array(regions[region_name])
                polygon[:, 0] -= offset_x
                polygon[:, 1] -= offset_y
                cv2.fillPoly(mask, [polygon.astype(np.int32)], region_idx)
        
        return mask


class ThermalFaceDataset(Dataset):
    """Unified PyTorch Dataset for thermal facial images"""
    
    def __init__(self, sample_ids: List[str], data_loader: MultiDirectoryDataLoader,
                 config: Config, augment: bool = False):
        self.sample_ids = sample_ids
        self.data_loader = data_loader
        self.config = config
        self.augment = augment
    
    def __len__(self):
        return len(self.sample_ids)
    
    def normalize_thermal(self, thermal_img: np.ndarray) -> np.ndarray:
        """Normalize thermal image to [0, 1] range"""
        min_val = thermal_img.min()
        max_val = thermal_img.max()
        if max_val - min_val > 0:
            return (thermal_img - min_val) / (max_val - min_val)
        return thermal_img
    
    def augment_data(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random rotation (-10 to 10 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
        
        return image, mask
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load thermal image
        thermal_img = self.data_loader.load_thermal_image(sample_id)
        
        # Get bounding box info for offset
        offset_x = 0
        offset_y = 0
        if (self.data_loader.all_bboxes is not None and 
            sample_id in self.data_loader.all_bboxes['ID'].values):
            bbox = self.data_loader.all_bboxes[
                self.data_loader.all_bboxes['ID'] == sample_id
            ].iloc[0]
            offset_x = int(bbox['min_x']) - 10
            offset_y = int(bbox['min_y']) - 10
        
        # Crop to region of interest
        thermal_img = self.data_loader.crop_to_bbox(thermal_img, sample_id)
        
        # Create segmentation mask
        mask = self.data_loader.create_segmentation_mask(
            sample_id, thermal_img.shape, (offset_x, offset_y)
        )
        
        # Normalize thermal image
        thermal_img = self.normalize_thermal(thermal_img)
        
        # Apply augmentation if training
        if self.augment:
            thermal_img, mask = self.augment_data(thermal_img, mask)
        
        # Resize to target size
        thermal_img = cv2.resize(
            thermal_img, self.config.IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, self.config.IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert to tensors
        thermal_img = torch.from_numpy(thermal_img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        
        return thermal_img, mask, sample_id


def create_data_loaders(config: Config, batch_size: int, num_workers: int,
                       test_size: float = 0.2, random_state: int = 42,
                       shared_data_loader: MultiDirectoryDataLoader = None):
    """
    Create training and validation data loaders from all available datasets
    
    Args:
        config: Configuration object
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        shared_data_loader: Optional preloaded data loader to skip discovering datasets again
    
    Returns:
        train_loader, val_loader, train_ids, val_ids, data_loader
    """
    # Load data from all directories if not provided
    if shared_data_loader is None:
        data_loader = MultiDirectoryDataLoader(config)
        data_loader.load_annotations()
    else:
        data_loader = shared_data_loader
    
    # Get all sample IDs and filter out missing
    raw_sample_ids = list(data_loader.all_polygons.keys())
    sample_ids = []
    missing_count = 0
    
    for sid in raw_sample_ids:
        if data_loader.get_tiff_path(sid) is not None:
            sample_ids.append(sid)
        else:
            missing_count += 1
            
    if shared_data_loader is None:
        if missing_count > 0:
            print(f"\n⚠️  Filtered out {missing_count} samples with missing thermal images.")
            
    if not sample_ids:
        raise ValueError("No samples found in any dataset directory!")
    
    # Split data
    train_ids, val_ids = train_test_split(
        sample_ids, test_size=test_size, random_state=random_state
    )
    
    if shared_data_loader is None:    
        print(f"\nData Split:")
        print(f"  Training samples:   {len(train_ids)}")
        print(f"  Validation samples: {len(val_ids)}")
        
        # Count samples per dataset
        print(f"\nSamples per dataset:")
        for dataset_name in sorted(data_loader.datasets.keys()):
            train_count = sum(1 for sid in train_ids if sid.startswith(dataset_name + '/'))
            val_count = sum(1 for sid in val_ids if sid.startswith(dataset_name + '/'))
            total_count = train_count + val_count
            print(f"  {dataset_name}: {total_count} total ({train_count} train, {val_count} val)")
    
    # Create datasets
    train_dataset = ThermalFaceDataset(
        train_ids, data_loader, config, augment=True
    )
    val_dataset = ThermalFaceDataset(
        val_ids, data_loader, config, augment=False
    )
    
    # -------------------------------------------------------------------------
    # DataLoader worker configuration
    # -------------------------------------------------------------------------
    # num_workers is already OS-aware: hardware_detector returns 2 for Windows
    # and up to 8 for Linux/Mac (see HardwareProfile._calculate_workers).
    #
    # On all platforms with workers > 0 we explicitly request 'spawn' instead
    # of letting Python choose the default start method:
    #   - Windows default is already 'spawn', so this is a no-op there.
    #   - Linux default is 'fork', which copies the CUDA context into every
    #     worker and can cause silent corruption or hard deadlocks. 'spawn'
    #     starts each worker with a fresh interpreter, avoiding this entirely.
    #
    # Prerequisites for workers > 0 on Windows (all satisfied):
    #   1. if __name__ == '__main__' guard in main_pipeline.py       ✓
    #   2. Dataset / Config / _raw_to_celsius are all picklable      ✓
    #   3. multiprocessing_context='spawn' set below                 ✓
    if num_workers > 0:
        mp_context = 'spawn'
        print(f"   DataLoader: {num_workers} worker(s), multiprocessing_context='spawn'")
    else:
        mp_context = None
        print("   DataLoader: single-process mode (num_workers=0)")

    # pin_memory speeds up host-to-GPU transfer; skip it when no GPU present.
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        multiprocessing_context=mp_context,
    )

    return train_loader, val_loader, train_ids, val_ids, data_loader


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # This guard is MANDATORY when num_workers > 0 on Windows.
    # Without it, every spawned DataLoader worker re-imports this module
    # and tries to spawn its own workers, causing a RuntimeError.
    # -----------------------------------------------------------------
    _test_workers = 0 if platform.system() == 'Windows' else 4
    config = Config()
    train_loader, val_loader, train_ids, val_ids, _ = create_data_loaders(
        config, batch_size=8, num_workers=_test_workers
    )
    print("\n✅ Multi-directory data loading test successful!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
