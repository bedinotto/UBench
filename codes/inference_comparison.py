"""
Inference & Model Comparison Script
===================================
Loads the trained weights of U-Net, TransUNet, and Swin-UNet++,
selects random images from the validation dataset, generates predictions
from each model, and saves/prints a detailed comparison.
"""

import sys
import os
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# Add parent directory to sys.path to enable proper imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import config and loaders
try:
    from codes.unified_data import Config, create_kfold_data_loaders
    from codes.unified_training import calculate_iou, calculate_dice_score, _safe_filename
    import codes.unet_v2 as unet_v2
    import codes.transunet as transunet
    import codes.swin_unet_plus_plus as swin_unet_plus_plus
except ImportError:
    from unified_data import Config, create_kfold_data_loaders
    from unified_training import calculate_iou, calculate_dice_score, _safe_filename
    import unet_v2 as unet_v2
    import transunet as transunet
    import swin_unet_plus_plus as swin_unet_plus_plus


def get_latest_checkpoint(model_key: str) -> Path:
    """
    Search for a model checkpoint across standard locations.
    """
    search_dirs = [
        Path("outputs/models"),
        Path("outputs"),
        Path("."),
    ]
    
    # Also search inside any timestamped folders inside outputs/
    if Path("outputs").exists():
        for item in Path("outputs").iterdir():
            if item.is_dir() and (item / "models").exists():
                search_dirs.append(item / "models")
                
    # Possible filenames for each model key
    filenames = {
        'unet': [
            'best_u-net_model.pth',
            'best_u_net_model.pth',
            'best_unet_model.pth',
            'best_thermal_face_model.pth'
        ],
        'transunet': [
            'best_transunet_model.pth',
            'best_trans_unet_model.pth',
        ],
        'swin': [
            'best_swin-unet++_model.pth',
            'best_swin_unetplusplus_model.pth',
            'best_swin_unet_plusplus_model.pth',
        ]
    }
    
    candidates = filenames.get(model_key, [])
    for d in search_dirs:
        if d.exists():
            for c in candidates:
                p = d / c
                if p.exists():
                    return p.resolve()
                    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Inference and visual comparison of UBench deep learning models."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of random validation images to evaluate (default: 5)"
    )
    parser.add_argument(
        "--unet-path",
        type=str,
        default=None,
        help="Path to U-Net .pth weights (auto-detects if omitted)"
    )
    parser.add_argument(
        "--transunet-path",
        type=str,
        default=None,
        help="Path to TransUNet .pth weights (auto-detects if omitted)"
    )
    parser.add_argument(
        "--swin-path",
        type=str,
        default=None,
        help="Path to Swin-UNet++ .pth weights (auto-detects if omitted)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for image selection reproducibility"
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific sample IDs to load (e.g. S1/R11104). Overrides --num-images."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda' or 'cpu'). Auto-detected by default."
    )
    
    args = parser.parse_args()
    
    # Set seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"📌 Set random seed to {args.seed} for reproducible selection.")

    # Initialize configuration
    config = Config()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = device
    print(f"🖥️ Using Device: {device}")

    # Discover and detect weights
    unet_path = Path(args.unet_path) if args.unet_path else get_latest_checkpoint('unet')
    transunet_path = Path(args.transunet_path) if args.transunet_path else get_latest_checkpoint('transunet')
    swin_path = Path(args.swin_path) if args.swin_path else get_latest_checkpoint('swin')
    
    print("\n🔍 Detecting Model Weights:")
    print(f"  U-Net:      {unet_path if unet_path else '❌ NOT FOUND'}")
    print(f"  TransUNet:  {transunet_path if transunet_path else '❌ NOT FOUND'}")
    print(f"  SwinUNet++: {swin_path if swin_path else '❌ NOT FOUND'}")
    
    # Check if at least one model is available
    if not (unet_path or transunet_path or swin_path):
        print("\n❌ Error: Could not find any model weights. Please train the models first or specify path(s) using --unet-path, --transunet-path, or --swin-path.")
        sys.exit(1)
        
    # Instantiate models
    models = {}
    
    if unet_path:
        print("\n🏗️ Instantiating U-Net...")
        model = unet_v2.UNet(in_channels=1, num_classes=config.NUM_CLASSES)
        try:
            model.load_state_dict(torch.load(unet_path, map_location=device))
            model = model.to(device).eval()
            models['U-Net'] = model
            print("  ✅ U-Net weights loaded successfully.")
        except Exception as e:
            print(f"  ❌ Failed to load U-Net weights: {e}")
            
    if transunet_path:
        print("🏗️ Instantiating TransUNet...")
        model = transunet.TransUNet(
            img_size=config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=config.NUM_CLASSES
        )
        try:
            model.load_state_dict(torch.load(transunet_path, map_location=device))
            model = model.to(device).eval()
            models['TransUNet'] = model
            print("  ✅ TransUNet weights loaded successfully.")
        except Exception as e:
            print(f"  ❌ Failed to load TransUNet weights: {e}")
            
    if swin_path:
        print("🏗️ Instantiating Swin-UNet++...")
        model = swin_unet_plus_plus.SwinUNetPlusPlus(
            img_size=config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=config.NUM_CLASSES
        )
        try:
            model.load_state_dict(torch.load(swin_path, map_location=device))
            model = model.to(device).eval()
            models['Swin-UNet++'] = model
            print("  ✅ Swin-UNet++ weights loaded successfully.")
        except Exception as e:
            print(f"  ❌ Failed to load Swin-UNet++ weights: {e}")
            
    if not models:
        print("\n❌ Error: No models loaded successfully. Exiting.")
        sys.exit(1)

    # Load dataset validation images
    print("\n📦 Initializing Data Loaders...")
    # Single worker since it's just prediction
    folds_data, data_loader = create_kfold_data_loaders(
        config, batch_size=1, num_workers=0
    )
    # Use validation data from the first fold
    fold = folds_data[0]
    val_loader = fold['val_loader']
    val_ids = fold['val_ids']
    val_dataset = val_loader.dataset
    
    # Select images to compare
    selected_indices = []
    if args.sample_ids:
        print(f"🎯 Loading requested sample IDs: {args.sample_ids}")
        for sid in args.sample_ids:
            found = False
            for idx, item in enumerate(val_dataset.sample_ids):
                if item == sid or item.endswith(f"/{sid}"):
                    selected_indices.append(idx)
                    found = True
                    break
            if not found:
                print(f"  ⚠️ Warning: Sample ID '{sid}' not found in validation set.")
    else:
        num_to_select = min(args.num_images, len(val_dataset))
        selected_indices = random.sample(range(len(val_dataset)), num_to_select)
        
    if not selected_indices:
        print("❌ Error: No valid samples selected/found. Exiting.")
        sys.exit(1)
        
    print(f"🎯 Evaluating on {len(selected_indices)} samples:")
    for idx in selected_indices:
        print(f"  - {val_dataset.sample_ids[idx]}")
        
    # Colormap and labels for segmentation regions
    # Custom colored map for 10 classes using modern warning-free API
    cmap = plt.colormaps['tab10']
    # Background is black, other regions are standard distinct colors
    colors = [cmap(i) for i in range(config.NUM_CLASSES)]
    colors[0] = (0.0, 0.0, 0.0, 1.0) # Black background
    custom_cmap = mcolors.ListedColormap(colors)
    
    # Prepare plotting
    num_cols = 2 + len(models) # Thermal, GT, and each loaded model
    fig, axes = plt.subplots(
        len(selected_indices), num_cols,
        figsize=(4 * num_cols, 4 * len(selected_indices)),
        squeeze=False
    )
    
    # Store numerical stats
    stats_table = []
    
    for row_idx, sample_dataset_idx in enumerate(selected_indices):
        image_tensor, mask_tensor, sample_id = val_dataset[sample_dataset_idx]
        
        # Add batch dim
        input_batch = image_tensor.unsqueeze(0).to(device)
        mask_np = mask_tensor.numpy()
        
        # Plot Thermal Image
        thermal_np = image_tensor.squeeze().numpy()
        axes[row_idx, 0].imshow(thermal_np, cmap='inferno')
        axes[row_idx, 0].set_title(f"Thermal ({sample_id})", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Plot Ground Truth
        axes[row_idx, 1].imshow(mask_np, cmap=custom_cmap, vmin=0, vmax=config.NUM_CLASSES - 1)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        col_offset = 2
        for model_name, model in models.items():
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_batch)
                inference_time_ms = (time.time() - start_time) * 1000
                
                # Metrics
                preds = torch.argmax(outputs, dim=1).squeeze(0)
                
                # Calculate metrics for this specific image
                # Dice score expects logits with batch dim
                dice = calculate_dice_score(outputs, mask_tensor.unsqueeze(0).to(device), config.NUM_CLASSES)
                # IoU takes pred and mask flat or 2D
                ious = calculate_iou(preds, mask_tensor.to(device), config.NUM_CLASSES)
                mean_iou = np.nanmean(ious)
                
            pred_np = preds.cpu().numpy()
            
            # Record statistics
            stats_table.append({
                'Sample ID': sample_id,
                'Model': model_name,
                'mIoU': mean_iou,
                'Dice': dice,
                'Inference (ms)': inference_time_ms
            })
            
            # Plot Prediction
            axes[row_idx, col_offset].imshow(pred_np, cmap=custom_cmap, vmin=0, vmax=config.NUM_CLASSES - 1)
            axes[row_idx, col_offset].set_title(
                f"{model_name}\nmIoU: {mean_iou:.4f} | Dice: {dice:.4f}",
                fontsize=10
            )
            axes[row_idx, col_offset].axis('off')
            col_offset += 1
            
    # Add a global title and legend
    plt.suptitle("Thermal Face Region Detection: Model Comparison", fontsize=16, fontweight='bold', y=0.99)
    
    # Legend for classes
    patches = [
        plt.plot([], [], color=colors[i], marker="s", ls="", label=f"{i}: {config.REGION_NAMES[i]}")[0]
        for i in range(config.NUM_CLASSES)
    ]
    fig.legend(
        handles=patches, 
        loc="lower center", 
        ncol=5, 
        bbox_to_anchor=(0.5, 0.0), 
        fontsize=9,
        frameon=True,
        shadow=True
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Save visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pred_dir = config.OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)
    save_path = pred_dir / f"model_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    # Print comparison table
    print("\n" + "="*90)
    print(f"📊 COMPARISON METRICS FOR SELECTED RANDOM IMAGES")
    print("="*90)
    print(f"{'Sample ID':<15} | {'Model Name':<15} | {'mIoU':<10} | {'Dice Score':<12} | {'Latency (ms)':<12}")
    print("-"*90)
    for row in stats_table:
        print(f"{row['Sample ID']:<15} | {row['Model']:<15} | {row['mIoU']:.4f}     | {row['Dice']:.4f}     | {row['Inference (ms)']:.2f} ms")
    print("="*90)
    
    # Print average stats
    print("\n🏆 AVERAGE SUMMARY METRICS:")
    print("-" * 50)
    for model_name in models.keys():
        model_rows = [r for r in stats_table if r['Model'] == model_name]
        avg_miou = np.mean([r['mIoU'] for r in model_rows])
        avg_dice = np.mean([r['Dice'] for r in model_rows])
        avg_lat = np.mean([r['Inference (ms)'] for r in model_rows])
        print(f"🔥 {model_name:12s} -> Avg mIoU: {avg_miou:.4f} | Avg Dice Score: {avg_dice:.4f} | Avg Latency: {avg_lat:.2f} ms")
    print("-" * 50)
    
    print(f"\n🎉 Visual comparison plot saved successfully to:")
    print(f"   👉 {save_path.resolve()}\n")


if __name__ == "__main__":
    main()
