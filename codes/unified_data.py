"""
Unified Data Loading Module - Multi-Directory Support
=====================================================
Shared data loading and preprocessing for all models
Supports multiple dataset directories (S1, S2, ..., S10)
"""

import os
import platform
import warnings
import numpy as np
import pandas as pd
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold
import random

try:
    from codes.config_schema import load_config
    from codes.utils import apply_normalization
    from codes.preprocess_manifest import verify_preprocess_manifest
    from codes.augmentation import build_thermal_transform
except ImportError:
    from config_schema import load_config
    from utils import apply_normalization
    from preprocess_manifest import verify_preprocess_manifest
    from augmentation import build_thermal_transform


def _raw_to_celsius(raw):
    """Convert raw thermal sensor value to degrees Celsius.

    Defined at module level (not as a lambda) so that it is picklable
    by the 'spawn' multiprocessing context used by DataLoader workers
    on Windows and when CUDA is active on Linux/Mac.
    """
    return (raw / 100) - 273.15


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Set global seeds and own the cuDNN determinism decision (M6).

    This is the **single owner** of the cuDNN flags (UB-20a): the previous code
    had ``hardware_detector`` set ``cudnn.benchmark = True`` while this function
    set it ``False`` — a contradiction whose outcome depended on call order.

    Args:
        seed: base seed for python ``random``, numpy, and torch (CPU + CUDA).
        deterministic: when True (default), request reproducible cuDNN
            (``cudnn.deterministic = True``, ``cudnn.benchmark = False``); when
            False, allow ``cudnn.benchmark = True`` for speed (non-reproducible).
            ``torch.use_deterministic_algorithms`` is deliberately **not** called
            — it raises on ops without deterministic kernels and would
            destabilise the pipeline. This branch is GPU-only; on CPU the cuDNN
            flags are inert.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int):
    """DataLoader ``worker_init_fn``: seed numpy and python ``random`` per worker.

    Without this, spawned workers start with unseeded numpy/``random`` state, so
    albumentations augmentations are non-reproducible across runs (UB-20a/M6).
    The seed derives from torch's per-worker base seed (set deterministically
    from the DataLoader's seeded ``generator``) plus ``worker_id``, so it is
    reproducible across runs with the same ``RANDOM_SEED`` yet distinct per
    worker.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Config:
    """Unified configuration for all models.

    Loads ``codes/config.yaml`` through the typed pydantic schema in
    ``codes/config_schema.py`` (UB-12/T3.1): unknown keys and wrong-typed
    values raise a ``ValueError`` at import time instead of being silently
    ignored by the old ``dict.get(key, default)`` lookups.
    """

    _yaml_path = Path(__file__).parent / "config.yaml"
    _validated = load_config(_yaml_path)

    # Data paths (STRICT structure)
    DATA_DIR = Path(_validated.paths.data_dir)
    PROCESSED_DIR = Path(_validated.paths.processed_dir)

    # Output paths (STRICT structure)
    OUTPUT_DIR = Path(_validated.paths.output_dir)
    LOG_DIR = Path(_validated.paths.log_dir)

    # Model parameters
    IMAGE_SIZE = tuple(_validated.model.image_size)
    NUM_CLASSES = _validated.model.num_classes
    LEARNING_RATE = float(_validated.training.learning_rate)
    NUM_EPOCHS = _validated.training.num_epochs
    K_FOLDS = _validated.training.k_folds
    RANDOM_SEED = _validated.training.random_seed
    # Reproducibility flag (M6): the single owner of the cuDNN determinism/
    # benchmark decision. Overridable via UBENCH_DETERMINISTIC.
    DETERMINISTIC = _validated.training.deterministic
    # Subjects held out from all CV folds and evaluated as the TEST set (M1).
    # Default empty (CV only); overridable via the TEST_SUBJECTS env var.
    TEST_SUBJECTS = list(_validated.training.test_subjects)

    # Loss / optimizer / scheduler recipes (validated sub-configs, T3.1). The
    # trainer reads these instead of hardcoded literals; defaults reproduce the
    # previous behavior exactly (UB-12).
    LOSS = _validated.loss
    OPTIMIZER = _validated.optimizer
    SCHEDULER = _validated.scheduler
    # Per-family optimizer/scheduler recipe overrides (T3.3/M4).
    RECIPES = _validated.recipes
    # Thermal-domain preprocessing (T3.4/M7): load-time normalization mode+range.
    PREPROCESSING = _validated.preprocessing

    # Thermal conversion
    RAW_TO_CELSIUS = np.vectorize(_raw_to_celsius)

    # Region names
    REGION_NAMES = list(_validated.regions)

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, output_dir: str = None, log_dir: str = None):
        """Initialize and validate paths
        
        Args:
            output_dir: Override for output directory
            log_dir: Override for log directory
        """
        if output_dir:
            self.OUTPUT_DIR = Path(output_dir)
        if log_dir:
            self.LOG_DIR = Path(log_dir)
        self._validate_paths()
        self._create_output_dirs()
        if 'NUM_EPOCHS' in os.environ:
            self.NUM_EPOCHS = int(os.environ['NUM_EPOCHS'])
        if 'K_FOLDS' in os.environ:
            self.K_FOLDS = int(os.environ['K_FOLDS'])
        if 'TEST_SUBJECTS' in os.environ:
            raw = os.environ['TEST_SUBJECTS'].strip()
            self.TEST_SUBJECTS = [s.strip() for s in raw.split(',') if s.strip()]
        if 'UBENCH_DETERMINISTIC' in os.environ:
            self.DETERMINISTIC = os.environ['UBENCH_DETERMINISTIC'].strip() not in ('0', 'false', 'False', '')
    
    def _validate_paths(self):
        """Validate required data paths exist"""
        if not self.DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.DATA_DIR}\n"
                f"Please create 'data/' directory and place dataset folders inside"
            )
    
    def _create_output_dirs(self):
        """Create the run's output/log roots only.

        The ``models``/``plots``/``predictions`` subdirs are created on demand by
        their writers (checkpoint save, plot export) under ``outputs/<run_id>/``.
        Pre-creating them here left stray top-level ``outputs/{models,plots,
        predictions}`` dirs whenever a default ``Config()`` was built (e.g. in
        ``preprocess_all_data``) — clutter beside the real run dirs (UB-24).
        """
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


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
                     sample_id: str, padding: int = 10
                     ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop thermal image to bounding box region.

        Returns:
            ``(cropped_img, (origin_x, origin_y))`` where the origin is the
            clamped top-left corner actually used for the crop.  Polygon
            offsets must consume this returned origin: recomputing
            ``bbox.min - padding`` without clamping shifts masks for
            border-adjacent boxes (UB-08).
        """
        if self.all_bboxes is None or sample_id not in self.all_bboxes['ID'].values:
            return thermal_img, (0, 0)

        bbox = self.all_bboxes[self.all_bboxes['ID'] == sample_id].iloc[0]

        min_x = max(0, int(bbox['min_x']) - padding)
        min_y = max(0, int(bbox['min_y']) - padding)
        max_x = min(thermal_img.shape[1], int(bbox['max_x']) + padding)
        max_y = min(thermal_img.shape[0], int(bbox['max_y']) + padding)

        return thermal_img[min_y:max_y, min_x:max_x], (min_x, min_y)
    
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
    """Unified PyTorch Dataset reading from offline preprocessed arrays"""
    
    def __init__(self, metadata_df: pd.DataFrame, config: Config, augment: bool = False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.config = config
        self.augment = augment
        
        # Physical thermal augmentation, built from config (T3.4/M7): flip +
        # Affine + additive sensor drift/noise, applied in Celsius before
        # normalization. Single augmentation authority (R5).
        if self.augment:
            self.transform = build_thermal_transform(
                config.PREPROCESSING.augmentation, seed=config.RANDOM_SEED)
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row['sample_id']
        
        img_path = str(self.config.DATA_DIR.parent / row['image_path'])
        mask_path = str(self.config.DATA_DIR.parent / row['mask_path'])
        
        # Load precomputed arrays. The .npy stores resized **Celsius** (T3.4/
        # design i). Augmentation acts on the Celsius array (physical units),
        # then normalization to [0,1] is applied via the single authority (R5).
        thermal_celsius = np.load(img_path).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=thermal_celsius, mask=mask)
            thermal_celsius = augmented['image']
            mask = augmented['mask']

        thermal_img = apply_normalization(
            thermal_celsius,
            self.config.PREPROCESSING.normalization,
            self.config.PREPROCESSING.fixed_range_celsius,
        )

        # Convert to tensors
        thermal_img = torch.from_numpy(thermal_img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        
        return thermal_img, mask, sample_id


def _read_all_metadata(config: Config) -> pd.DataFrame:
    """Read every processed metadata row (test subjects included).

    Applies the ``LIMIT_SAMPLES`` truncation here so that every consumer
    (fold-count resolution, loader creation, held-out test set) sees the same
    rows before any test-subject partitioning.
    """
    metadata_path = config.PROCESSED_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {metadata_path}. "
            f"Please run 'python codes/preprocess_data.py' first."
        )
    # T3.4/M7: the .npy store Celsius and are normalized at load. Reject
    # data whose schema manifest is missing or a version mismatch (legacy
    # baked-normalized data would be silently misread as Celsius).
    verify_preprocess_manifest(config.PROCESSED_DIR)
    df = pd.read_csv(metadata_path)
    if 'LIMIT_SAMPLES' in os.environ:
        df = df.head(int(os.environ['LIMIT_SAMPLES']))
    return df


def _resolve_test_subjects(config: Config, df: pd.DataFrame) -> list:
    """Return the configured held-out test subjects, validated against the data.

    Holding out a subject that is not present would silently reserve *nothing*
    — exactly the failure M1 exists to prevent — so an unknown id is a hard
    error (R4), not a silent no-op.
    """
    test_subjects = list(getattr(config, 'TEST_SUBJECTS', []) or [])
    if not test_subjects:
        return []
    discovered = set(pd.unique(df['dataset']))
    missing = [s for s in test_subjects if s not in discovered]
    if missing:
        raise ValueError(
            f"Configured test_subjects {missing} are not present in the "
            f"discovered data (subjects: {sorted(discovered)}). Holding out a "
            f"nonexistent subject would reserve nothing (M1). Fix test_subjects "
            f"in codes/config.yaml or the TEST_SUBJECTS env var."
        )
    return test_subjects


def load_split_metadata(config: Config) -> pd.DataFrame:
    """Read the metadata rows that feed the CV split (held-out test excluded).

    Held-out ``test_subjects`` (M1) are removed here so that **every** CV
    consumer — the fold-count guard and both loader factories — operates on the
    same training pool and no fold can ever see a test subject.
    """
    df = _read_all_metadata(config)
    test_subjects = _resolve_test_subjects(config, df)
    if test_subjects:
        df = df[~df['dataset'].isin(test_subjects)].reset_index(drop=True)
    return df


def load_test_metadata(config: Config) -> pd.DataFrame:
    """Read exactly the held-out test-subject rows (empty when none configured)."""
    df = _read_all_metadata(config)
    test_subjects = _resolve_test_subjects(config, df)
    return df[df['dataset'].isin(test_subjects)].reset_index(drop=True)


def resolve_fold_count(requested_k: int, groups: pd.Series) -> int:
    """Resolve the effective number of leave-subjects-out CV folds (UB-03).

    ``GroupKFold(groups=df['dataset'])`` holds out whole subject datasets
    per fold, so it can never produce more splits than there are distinct
    subjects.

    Args:
        requested_k: Configured ``K_FOLDS`` value.
        groups: Per-sample group labels (the metadata ``dataset`` column).

    Returns:
        ``min(requested_k, n_groups)``, warning loudly when reduced.

    Raises:
        ValueError: When fewer than 2 subject datasets are present.
    """
    group_names = sorted(pd.unique(groups))
    n_groups = len(group_names)
    if n_groups < 2:
        raise ValueError(
            f"Leave-subjects-out CV requires >=2 subject datasets; found "
            f"{n_groups}: {group_names}. Add more S* subject directories "
            f"under data/ (or raise LIMIT_SAMPLES so more subjects survive "
            f"truncation)."
        )
    if requested_k > n_groups:
        message = (
            f"K_FOLDS={requested_k} exceeds the {n_groups} available subject "
            f"datasets ({group_names}); leave-subjects-out CV holds out whole "
            f"subjects, so the fold count is reduced to {n_groups}."
        )
        # print for the pipeline's stdout logs; warnings.warn for callers/tests
        print(f"[WARNING] {message}")
        warnings.warn(message, stacklevel=2)
        return n_groups
    return requested_k


def create_kfold_data_loaders(config: Config, batch_size: int, num_workers: int,
                              shared_data_loader: MultiDirectoryDataLoader = None):
    """
    Create K-Fold training and validation data loaders from preprocessed offline arrays
    """
    df = load_split_metadata(config)

    # Leave-subjects-out CV: whole subject datasets held out per fold (UB-03/04)
    effective_k = resolve_fold_count(config.K_FOLDS, df['dataset'])
    gkf = GroupKFold(n_splits=effective_k)
    split_generator = gkf.split(df, groups=df['dataset'])
    
    folds_data = []
    
    # Worker config
    if num_workers > 0:
        mp_context = 'spawn'
        print(f"   DataLoader: {num_workers} worker(s), multiprocessing_context='spawn'")
    else:
        mp_context = None
        print("   DataLoader: single-process mode (num_workers=0)")

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0 and platform.system() != 'Windows'
    
    print(f"\nData Split (leave-subjects-out GroupKFold: {effective_k}):")
    
    for fold_idx, (train_idx, val_idx) in enumerate(split_generator):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_ids = train_df['sample_id'].tolist()
        val_ids = val_df['sample_id'].tolist()
        
        if fold_idx == 0:
            print(f"  Fold 1 Training samples:   {len(train_ids)}")
            print(f"  Fold 1 Validation samples: {len(val_ids)}")
            print(f"\nSamples per dataset (Fold 1):")
            for dataset_name in sorted(df['dataset'].unique()):
                train_count = sum(train_df['dataset'] == dataset_name)
                val_count = sum(val_df['dataset'] == dataset_name)
                total_count = train_count + val_count
                print(f"  {dataset_name}: {total_count} total ({train_count} train, {val_count} val)")
        
        # Create datasets
        train_dataset = ThermalFaceDataset(
            train_df, config, augment=True
        )
        val_dataset = ThermalFaceDataset(
            val_df, config, augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            multiprocessing_context=mp_context,
            generator=torch.Generator().manual_seed(config.RANDOM_SEED),
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            multiprocessing_context=mp_context,
            generator=torch.Generator().manual_seed(config.RANDOM_SEED),
            worker_init_fn=seed_worker,
        )

        folds_data.append({
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_ids': train_ids,
            'val_ids': val_ids
        })
    return folds_data, None


def create_single_fold_loader(config: Config, fold_idx: int, batch_size: int,
                              num_workers: int):
    """Create DataLoaders for a **single** K-Fold split.

    Unlike ``create_kfold_data_loaders`` (which builds all K folds and
    returns a list), this function only materialises the one fold that
    is actually needed.  This avoids spawning ``(K-1) * 2 * num_workers``
    worker processes that would be immediately discarded, preventing
    POSIX semaphore leaks on Linux and shared-memory exhaustion on
    Windows.

    Parameters
    ----------
    config : Config
        Project configuration object.
    fold_idx : int
        Zero-based fold index (0 … K-1).
    batch_size : int
        Batch size for the DataLoaders.
    num_workers : int
        Number of prefetch worker processes.

    Returns
    -------
    dict
        ``{'train_loader': DataLoader, 'val_loader': DataLoader,
        'train_ids': list, 'val_ids': list}``
    """
    df = load_split_metadata(config)

    # Leave-subjects-out CV: whole subject datasets held out per fold (UB-03/04)
    effective_k = resolve_fold_count(config.K_FOLDS, df['dataset'])
    gkf = GroupKFold(n_splits=effective_k)
    split_generator = gkf.split(df, groups=df['dataset'])

    # Advance the generator to the requested fold
    for i, (train_idx, val_idx) in enumerate(split_generator):
        if i == fold_idx:
            break
    else:
        raise IndexError(
            f"fold_idx {fold_idx} is out of range for effective K={effective_k}"
        )

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Worker config
    if num_workers > 0:
        mp_context = 'spawn'
    else:
        mp_context = None

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0 and platform.system() != 'Windows'

    train_dataset = ThermalFaceDataset(train_df, config, augment=True)
    val_dataset = ThermalFaceDataset(val_df, config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        multiprocessing_context=mp_context,
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        multiprocessing_context=mp_context,
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
        worker_init_fn=seed_worker,
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_ids': train_df['sample_id'].tolist(),
        'val_ids': val_df['sample_id'].tolist(),
    }


def create_test_loader(config: Config, batch_size: int, num_workers: int):
    """Create a DataLoader over exactly the held-out test subjects (M1).

    Returns ``None`` when no ``test_subjects`` are configured, so callers get a
    CV-only run unchanged. The loader is evaluation-style (no augmentation, no
    shuffle) — it is scored, never trained on.
    """
    test_df = load_test_metadata(config)
    if test_df.empty:
        return None

    mp_context = 'spawn' if num_workers > 0 else None
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0 and platform.system() != 'Windows'

    test_dataset = ThermalFaceDataset(test_df, config, augment=False)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        multiprocessing_context=mp_context,
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
        worker_init_fn=seed_worker,
    )


def shutdown_data_loaders(*loaders: DataLoader):
    """Explicitly shut down DataLoader worker processes and free OS resources.

    On Windows, each DataLoader with num_workers>0 holds open shared-memory
    file mappings.  Calling this function between folds / models ensures those
    handles are released before new loaders are created.

    On Linux with the 'spawn' multiprocessing context, each worker holds a
    POSIX named semaphore.  Failing to shut workers down cleanly causes
    semaphore leaks that Python's resource_tracker warns about at exit.
    """
    import gc
    for loader in loaders:
        if loader is None:
            continue
        # Save the iterator reference *before* clearing it so we can
        # call _shutdown_workers() on the live object.
        it = getattr(loader, '_iterator', None)
        if it is not None:
            if hasattr(it, '_shutdown_workers'):
                try:
                    it._shutdown_workers()
                except Exception:
                    pass
            # Clear the reference so the iterator can be garbage-collected
            try:
                loader._iterator = None
            except Exception:
                pass
    gc.collect()


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # This guard is MANDATORY when num_workers > 0 on Windows.
    # Without it, every spawned DataLoader worker re-imports this module
    # and tries to spawn its own workers, causing a RuntimeError.
    # -----------------------------------------------------------------
    _test_workers = 0 if platform.system() == 'Windows' else 4
    config = Config()
    folds_data, _ = create_kfold_data_loaders(
        config, batch_size=8, num_workers=_test_workers
    )
    print("\n✅ Multi-directory data loading test successful!")
    print(f"K-Folds created: {len(folds_data)}")
    print(f"Fold 1 Train batches: {len(folds_data[0]['train_loader'])}")
    print(f"Fold 1 Val batches:   {len(folds_data[0]['val_loader'])}")
