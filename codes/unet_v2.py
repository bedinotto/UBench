"""
Thermal Facial Region Detection System
=======================================
AI model for detecting facial regions in thermal sensor data
Handles full thermal matrix data from TIFF files
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration for the thermal face detection system"""


# /home/doga/Documents/UNet/S1/R143426.tiff
    # Data paths
    DATA_DIR = Path(
        "")
    # Directory with TIFF files
    THERMAL_DIR = Path(
        "S1/")
    ANNOTATIONS_FILE = "S1.csv"
    POLYGONS_FILE = "polygonal_masks.json"
    BBOXES_FILE = "bounding_boxes.csv"

    # Model parameters
    IMAGE_SIZE = (256, 256)  # Resize thermal images to this size
    NUM_CLASSES = 10  # Background + 9 facial regions
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100

    # Thermal conversion
    RAW_TO_CELSIUS = np.vectorize(lambda raw: (raw / 100) - 273.15)

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


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class ThermalDataLoader:
    """Load and preprocess thermal image data"""

    def __init__(self, config: Config):
        self.config = config
        self.annotations = None
        self.polygons = None
        self.bboxes = None

    def load_annotations(self):
        """Load all annotation files"""
        # Load landmark points
        self.annotations = pd.read_csv(
            self.config.DATA_DIR / self.config.ANNOTATIONS_FILE
        )

        # Load polygonal masks
        with open(self.config.DATA_DIR / self.config.POLYGONS_FILE, 'r') as f:
            self.polygons = json.load(f)

        # Load bounding boxes
        self.bboxes = pd.read_csv(
            self.config.DATA_DIR / self.config.BBOXES_FILE
        )

        print(f"Loaded {len(self.annotations)} annotated samples")
        print(f"Loaded {len(self.polygons)} polygon annotations")
        return self

    # def load_thermal_matrix_from_csv(self, csv_path: str) -> np.ndarray:
    #     """
    #     Load thermal matrix from CSV file

    #     Args:
    #         csv_path: Path to CSV file containing thermal matrix

    #     Returns:
    #         Thermal image as numpy array in Celsius
    #     """
    #     # Read CSV as matrix (no headers)
    #     thermal_raw = pd.read_csv(csv_path, header=None).values

    #     # Convert to Celsius
    #     thermal_celsius = self.config.RAW_TO_CELSIUS(thermal_raw)

    #     return thermal_celsius.astype(np.float32)

    def load_thermal_image_from_tiff(self, tiff_path: str) -> np.ndarray:
        """
        Load thermal image from TIFF file

        Args:
            tiff_path: Path to TIFF file

        Returns:
            Thermal image as numpy array in Celsius
        """
        # Read TIFF file
        thermal_raw = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        # print("Aberto:" + tiff_path)

        # Convert to Celsius element-wise
        thermal_celsius = self.config.RAW_TO_CELSIUS(thermal_raw)

        return thermal_celsius.astype(np.float32)

    def load_thermal_image(self, sample_id: str) -> np.ndarray:
        """
        Load thermal image for a given sample ID

        Args:
            sample_id: Sample identifier (e.g., 'R11104')

        Returns:
            Thermal image as numpy array in Celsius
        """
        # print("########################### Tentando abrir:" + sample_id)
        # Try TIFF first
        tiff_path = self.config.THERMAL_DIR / f"{sample_id}.tiff"
        if tiff_path.exists():
            # print("########################### Aberto com sucesso:" + sample_id)
            return self.load_thermal_image_from_tiff(str(tiff_path))

            # # Try CSV
            # csv_path = self.config.THERMAL_DIR / f"{sample_id}.csv"
            # if csv_path.exists():
            #     return self.load_thermal_matrix_from_csv(str(csv_path))

        raise FileNotFoundError(f"Thermal image not found for {sample_id}")

    def crop_to_bbox(self, thermal_img: np.ndarray,
                     sample_id: str, padding: int = 10) -> np.ndarray:
        """
        Crop thermal image to bounding box region

        Args:
            thermal_img: Full thermal image
            sample_id: Sample identifier
            padding: Padding around bounding box

        Returns:
            Cropped thermal image
        """
        if self.bboxes is None or sample_id not in self.bboxes['ID'].values:
            return thermal_img

        bbox = self.bboxes[self.bboxes['ID'] == sample_id].iloc[0]

        min_x = max(0, int(bbox['min_x']) - padding)
        min_y = max(0, int(bbox['min_y']) - padding)
        max_x = min(thermal_img.shape[1], int(bbox['max_x']) + padding)
        max_y = min(thermal_img.shape[0], int(bbox['max_y']) + padding)

        return thermal_img[min_y:max_y, min_x:max_x]

    def create_segmentation_mask(self, sample_id: str,
                                 img_shape: Tuple[int, int],
                                 offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Create segmentation mask from polygon annotations

        Args:
            sample_id: Sample identifier
            img_shape: Shape of the target image (height, width)
            offset: (x_offset, y_offset) to adjust polygon coordinates

        Returns:
            Segmentation mask with class labels
        """
        mask = np.zeros(img_shape, dtype=np.uint8)

        if self.polygons is None or sample_id not in self.polygons:
            return mask

        regions = self.polygons[sample_id]
        offset_x, offset_y = offset

        # Draw each region
        for region_idx, region_name in enumerate(self.config.REGION_NAMES[1:], 1):
            if region_name in regions:
                polygon = np.array(regions[region_name])
                # Apply offset
                polygon[:, 0] -= offset_x
                polygon[:, 1] -= offset_y

                # Fill polygon with class index
                cv2.fillPoly(mask, [polygon.astype(np.int32)], region_idx)

        return mask


# ============================================================================
# DATASET CLASS
# ============================================================================

class ThermalFaceDataset(Dataset):
    """PyTorch Dataset for thermal facial images"""

    def __init__(self, sample_ids: List[str], data_loader: ThermalDataLoader,
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
        if self.data_loader.bboxes is not None and sample_id in self.data_loader.bboxes['ID'].values:
            bbox = self.data_loader.bboxes[
                self.data_loader.bboxes['ID'] == sample_id
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


# ============================================================================
# MODEL ARCHITECTURE - U-Net
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2)

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss"""

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class Trainer:
    """Training pipeline for thermal face segmentation"""

    def __init__(self, model, train_loader, val_loader, config: Config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for images, masks, _ in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.config.DEVICE)
            masks = masks.to(self.config.DEVICE)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        """Full training loop"""
        print(f"Training on {self.config.DEVICE}")

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    'best_thermal_face_model.pth'
                )
                print("✓ Saved best model")

    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig('training_history.png')
        plt.close()


# ============================================================================
# INFERENCE
# ============================================================================

class ThermalFaceDetector:
    """Inference class for detecting facial regions in thermal images"""

    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model = UNet(in_channels=1, num_classes=config.NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(config.DEVICE)
        self.model.eval()

    def normalize_thermal(self, thermal_img: np.ndarray) -> np.ndarray:
        """Normalize thermal image"""
        min_val = thermal_img.min()
        max_val = thermal_img.max()
        if max_val - min_val > 0:
            return (thermal_img - min_val) / (max_val - min_val)
        return thermal_img

    def predict(self, thermal_image: np.ndarray):
        """
        Predict facial regions in a thermal image

        Args:
            thermal_image: Input thermal image (H, W) in Celsius

        Returns:
            Dictionary with region names and their masks, and the full prediction mask
        """
        # Store original shape
        original_shape = thermal_image.shape

        # Normalize
        thermal_image_norm = self.normalize_thermal(thermal_image)

        # Resize
        thermal_image_resized = cv2.resize(
            thermal_image_norm, self.config.IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to tensor
        image_tensor = torch.from_numpy(
            thermal_image_resized).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.float().to(self.config.DEVICE)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Resize back to original size
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Extract individual regions
        regions = {}
        for idx, region_name in enumerate(self.config.REGION_NAMES):
            region_mask = (pred_mask == idx).astype(np.uint8)
            regions[region_name] = region_mask

        return regions, pred_mask

    def get_stats_info(self, thermal_image: np.ndarray, regions: Dict[str, np.ndarray]):
        """
        Calculate statistics information for each region using original thermal data in Celsius
        
        Args:
            thermal_image: Original thermal image in Celsius (H, W)
            regions: Dictionary with region names and their binary masks
            
        Returns:
            Dictionary with statistics for each region
        """
        stats_info = {}

        for region_name, mask in regions.items():
            # Extract temperature values only where mask is 1
            region_temps = thermal_image[mask == 1]

            # Check if region has any pixels
            if len(region_temps) == 0:
                stats_info[region_name] = {
                    'mean': None,
                    'median': None,
                    'mode': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'pixel_count': 0
                }
                continue

            # Calculate statistics
            mean_temp = np.mean(region_temps)
            median_temp = np.median(region_temps)
            std_temp = np.std(region_temps)
            min_temp = np.min(region_temps)
            max_temp = np.max(region_temps)

            # Calculate mode (most frequent temperature value)
            # For continuous data, we bin the temperatures
            # Use 0.1°C bins for more meaningful mode
            bins = np.arange(region_temps.min(), region_temps.max() + 0.1, 0.1)
            hist, bin_edges = np.histogram(region_temps, bins=bins)
            mode_idx = np.argmax(hist)
            mode_temp = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

            # Alternative: use scipy.stats.mode if you prefer
            # from scipy import stats
            # mode_result = stats.mode(np.round(region_temps, 1), keepdims=True)
            # mode_temp = mode_result.mode[0]

            stats_info[region_name] = {
                'mean': float(mean_temp),
                'median': float(median_temp),
                'mode': float(mode_temp),
                'std': float(std_temp),
                'min': float(min_temp),
                'max': float(max_temp),
                'pixel_count': int(len(region_temps))
            }

        return stats_info

    def visualize_predictions(self, thermal_image: np.ndarray,
                              regions: Dict[str, np.ndarray],
                              save_path: Optional[str] = None):
        """Visualize predicted regions"""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        # Show original image
        axes[0].imshow(thermal_image, cmap='hot')
        axes[0].set_title('Original Thermal Image')
        axes[0].axis('off')

        # Show each region
        for idx, (region_name, mask) in enumerate(regions.items(), 1):
            if idx < len(axes):
                axes[idx].imshow(mask, cmap='gray')
                axes[idx].set_title(region_name, fontsize=8)
                axes[idx].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def print_stats_report(self, stats_info: Dict[str, Dict]):
        """
        Print a formatted report of thermal statistics for all regions
        
        Args:
            stats_info: Dictionary with statistics for each region
        """
        print("\n" + "="*80)
        print("THERMAL STATISTICS REPORT (°C)")
        print("="*80)

        for region_name, stats in stats_info.items():
            print(f"\n{region_name}:")
            print("-" * 80)

            if stats['pixel_count'] == 0:
                print("  No pixels detected in this region")
                continue

            print(f"  Mean:        {stats['mean']:.2f} °C")
            print(f"  Median:      {stats['median']:.2f} °C")
            print(f"  Mode:        {stats['mode']:.2f} °C")
            print(f"  Std Dev:     {stats['std']:.2f} °C")
            print(f"  Min:         {stats['min']:.2f} °C")
            print(f"  Max:         {stats['max']:.2f} °C")
            print(f"  Pixel Count: {stats['pixel_count']}")

        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    # Initialize configuration
    config = Config()

    # Load data
    print("Loading annotations...")
    data_loader = ThermalDataLoader(config)
    data_loader.load_annotations()

    # Get list of annotated sample IDs
    if data_loader.polygons is None:
        print("Error: Polygons not loaded. Check if annotation files exist.")
        return
    sample_ids = list(data_loader.polygons.keys())

    # Split data
    train_ids, val_ids = train_test_split(
        sample_ids, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")

    # Create datasets
    train_dataset = ThermalFaceDataset(
        train_ids, data_loader, config, augment=True)
    val_dataset = ThermalFaceDataset(
        val_ids, data_loader, config, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model
    print("\nInitializing U-Net model...")
    model = UNet(in_channels=1, num_classes=config.NUM_CLASSES)

    # Train model
    print("\nStarting training...")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    trainer.plot_training_history()

    # Test inference
    print("\nTesting inference...")
    detector = ThermalFaceDetector('best_thermal_face_model.pth', config)

    # Load a test image
    test_id = val_ids[0]
    test_image = data_loader.load_thermal_image(test_id)
    test_image = data_loader.crop_to_bbox(test_image, test_id)

    # Predict
    regions, pred_mask = detector.predict(test_image)
    detector.visualize_predictions(
        test_image, regions,
        save_path='prediction_example.png'
    )

    print("\n✓ Training completed successfully!")
    print(f"Best model saved to: best_thermal_face_model.pth")
    print(f"\nTo use on new thermal images:")
    print("  thermal_img = load_thermal_matrix_from_csv('new_image.csv')")
    print("  regions, mask = detector.predict(thermal_img)")


def execution():
    # Initialize configuration
    config = Config()

    # Load data
    print("Loading annotations...")
    data_loader = ThermalDataLoader(config)
    data_loader.load_annotations()

    # Get list of annotated sample IDs
    if data_loader.polygons is None:
        print("Error: Polygons not loaded. Check if annotation files exist.")
        return
    sample_ids = list(data_loader.polygons.keys())

    # Split data
    train_ids, val_ids = train_test_split(
        sample_ids, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")

    # Test inference
    print("\nTesting inference...")
    detector = ThermalFaceDetector('best_thermal_face_model.pth', config)

    # Load a test image
    test_id = val_ids[1]
    test_image = data_loader.load_thermal_image(test_id)
    test_image = data_loader.crop_to_bbox(test_image, test_id)

    # Print test_image data to see the min max values
    print("Min and Max values of test_image")
    print(test_image.min(), test_image.max())

    # Predict
    regions, pred_mask = detector.predict(test_image)
    # The statistics are now correctly calculated from Celsius values
    stats_info = detector.get_stats_info(test_image, regions)
    detector.print_stats_report(stats_info)

    # Access individual region stats
    background_mean = stats_info['background']['mean']
    nose_temp_std = stats_info['Nariz']['std']


if __name__ == "__main__":
    execution()
