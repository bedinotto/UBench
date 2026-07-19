"""
Offline Data Preprocessing Script
=================================
Pre-computes crops, normalizations, and mask rasterizations to save CPU cycles during training.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codes.unified_data import Config, MultiDirectoryDataLoader
from codes.utils import preprocess_thermal_image, preprocess_mask
from codes.preprocess_manifest import write_preprocess_manifest

def preprocess_all_data():
    config = Config()
    
    # Setup directories
    processed_dir = config.PROCESSED_DIR
    images_dir = processed_dir / "images"
    masks_dir = processed_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {config.DATA_DIR}...")
    data_loader = MultiDirectoryDataLoader(config)
    data_loader.load_annotations()
    
    sample_ids = list(data_loader.all_polygons.keys())
    print(f"Found {len(sample_ids)} total samples with polygons.")
    
    processed_count = 0
    missing_count = 0
    
    metadata = []
    
    print("\nStarting offline preprocessing...")
    for sample_id in tqdm(sample_ids, desc="Processing Images"):
        try:
            # Check if tiff exists
            tiff_path = data_loader.get_tiff_path(sample_id)
            if not tiff_path:
                missing_count += 1
                continue
                
            # Parse dataset and name
            dataset_name, img_id = sample_id.split('/', 1)
            safe_id = f"{dataset_name}_{img_id}"
            
            # Load raw
            thermal_img = data_loader.load_thermal_image(sample_id)

            # Crop; the returned origin is the clamped top-left corner the
            # crop actually used — the polygon offset must match it (UB-08)
            thermal_img, (offset_x, offset_y) = data_loader.crop_to_bbox(
                thermal_img, sample_id
            )

            # Create mask
            mask = data_loader.create_segmentation_mask(
                sample_id, thermal_img.shape, (offset_x, offset_y)
            )
            
            # Resize and normalize
            processed_img = preprocess_thermal_image(thermal_img, config.IMAGE_SIZE)
            processed_mask = preprocess_mask(mask, config.IMAGE_SIZE)
            
            # Save
            img_path = images_dir / f"{safe_id}.npy"
            mask_path = masks_dir / f"{safe_id}.png"
            
            np.save(str(img_path), processed_img)
            cv2.imwrite(str(mask_path), processed_mask)
            
            metadata.append({
                'sample_id': sample_id,
                'dataset': dataset_name,
                'img_id': img_id,
                'image_path': str(img_path.relative_to(config.DATA_DIR.parent)),
                'mask_path': str(mask_path.relative_to(config.DATA_DIR.parent))
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(processed_dir / "metadata.csv", index=False)

    # Record the data-schema manifest (T3.4): the .npy now store Celsius, and
    # the load-time guard rejects data without a current-version manifest.
    write_preprocess_manifest(
        processed_dir,
        image_size=config.IMAGE_SIZE,
        normalization=config.PREPROCESSING.normalization,
        fixed_range_celsius=config.PREPROCESSING.fixed_range_celsius,
        num_samples=len(df),
    )

    print("\n" + "="*70)
    print(f"✅ Preprocessing completed!")
    print(f"Successfully processed: {processed_count}")
    print(f"Missing images:         {missing_count}")
    print(f"Outputs saved to:       {processed_dir}")
    print("="*70)

if __name__ == "__main__":
    preprocess_all_data()
