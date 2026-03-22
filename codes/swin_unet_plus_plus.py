"""
Thermal Facial Region Detection System - Swin-UNet++
====================================================
Swin Transformer-based U-Net++ architecture for thermal face segmentation
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
import time

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration for the thermal face detection system"""

    # Data paths
    DATA_DIR = Path("")
    THERMAL_DIR = Path("S1/")
    ANNOTATIONS_FILE = "S1.csv"
    POLYGONS_FILE = "polygonal_masks.json"
    BBOXES_FILE = "bounding_boxes.csv"

    # Model parameters
    IMAGE_SIZE = (256, 256)
    NUM_CLASSES = 10
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
        self.annotations = pd.read_csv(
            self.config.DATA_DIR / self.config.ANNOTATIONS_FILE
        )

        with open(self.config.DATA_DIR / self.config.POLYGONS_FILE, 'r') as f:
            self.polygons = json.load(f)

        self.bboxes = pd.read_csv(
            self.config.DATA_DIR / self.config.BBOXES_FILE
        )

        print(f"Loaded {len(self.annotations)} annotated samples")
        print(f"Loaded {len(self.polygons)} polygon annotations")
        return self

    def load_thermal_image_from_tiff(self, tiff_path: str) -> np.ndarray:
        """Load thermal image from TIFF file"""
        thermal_raw = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        thermal_celsius = self.config.RAW_TO_CELSIUS(thermal_raw)
        return thermal_celsius.astype(np.float32)

    def load_thermal_image(self, sample_id: str) -> np.ndarray:
        """Load thermal image for a given sample ID"""
        tiff_path = self.config.THERMAL_DIR / f"{sample_id}.tiff"
        if tiff_path.exists():
            return self.load_thermal_image_from_tiff(str(tiff_path))
        raise FileNotFoundError(f"Thermal image not found for {sample_id}")

    def crop_to_bbox(self, thermal_img: np.ndarray,
                     sample_id: str, padding: int = 10) -> np.ndarray:
        """Crop thermal image to bounding box region"""
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
        """Create segmentation mask from polygon annotations"""
        mask = np.zeros(img_shape, dtype=np.uint8)

        if self.polygons is None or sample_id not in self.polygons:
            return mask

        regions = self.polygons[sample_id]
        offset_x, offset_y = offset

        for region_idx, region_name in enumerate(self.config.REGION_NAMES[1:], 1):
            if region_name in regions:
                polygon = np.array(regions[region_name])
                polygon[:, 0] -= offset_x
                polygon[:, 1] -= offset_y
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
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        return image, mask

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        thermal_img = self.data_loader.load_thermal_image(sample_id)

        offset_x = 0
        offset_y = 0
        if self.data_loader.bboxes is not None and sample_id in self.data_loader.bboxes['ID'].values:
            bbox = self.data_loader.bboxes[
                self.data_loader.bboxes['ID'] == sample_id
            ].iloc[0]
            offset_x = int(bbox['min_x']) - 10
            offset_y = int(bbox['min_y']) - 10

        thermal_img = self.data_loader.crop_to_bbox(thermal_img, sample_id)

        mask = self.data_loader.create_segmentation_mask(
            sample_id, thermal_img.shape, (offset_x, offset_y)
        )

        thermal_img = self.normalize_thermal(thermal_img)

        if self.augment:
            thermal_img, mask = self.augment_data(thermal_img, mask)

        thermal_img = cv2.resize(
            thermal_img, self.config.IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, self.config.IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        thermal_img = torch.from_numpy(thermal_img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()

        return thermal_img, mask, sample_id


# ============================================================================
# SWIN TRANSFORMER BLOCKS
# ============================================================================

def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention"""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size //
                                   patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size


# ============================================================================
# NESTED DENSE SKIP CONNECTIONS (UNet++ style)
# ============================================================================

class NestedConvBlock(nn.Module):
    """Nested convolution block for dense skip connections"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# SWIN-UNET++ MODEL
# ============================================================================

class SwinUNetPlusPlus(nn.Module):
    """Swin-UNet++ architecture with nested dense skip connections"""

    def __init__(self, img_size=256, in_channels=1, num_classes=10,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_channels, embed_dim=embed_dim
        )

        # Swin Transformer encoder stages
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=8,
                    shift_size=0 if (i % 2 == 0) else 8 // 2
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)

            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(layer_dim)
                self.downsample_layers.append(downsample)

        # Projection layers to convert from transformer features to CNN features
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** i), 1)
            for i in range(self.num_layers)
        ])

        # Nested dense skip connections (UNet++ style)
        # x^0_0, x^1_0, x^2_0, x^3_0 are encoder outputs
        # x^0_1, x^1_1, x^2_1, x^3_1 are nested blocks
        self.conv0_1 = NestedConvBlock(embed_dim * 3, embed_dim)  # 288 -> 96
        self.conv1_1 = NestedConvBlock(
            embed_dim * 6, embed_dim * 2)  # 576 -> 192
        self.conv2_1 = NestedConvBlock(
            embed_dim * 12, embed_dim * 4)  # 1152 -> 384

        self.conv0_2 = NestedConvBlock(embed_dim * 4, embed_dim)  # 384 -> 96
        self.conv1_2 = NestedConvBlock(
            embed_dim * 8, embed_dim * 2)  # 768 -> 192

        self.conv0_3 = NestedConvBlock(embed_dim * 5, embed_dim)  # 480 -> 96

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(
            embed_dim * 2, embed_dim * 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(
            embed_dim * 4, embed_dim * 4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(
            embed_dim * 8, embed_dim * 8, 2, stride=2)

        # Final output layer
        self.final = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x, H_enc, W_enc = self.patch_embed(x)

        # Encoder - Swin Transformer stages
        enc_features = []
        for i_layer in range(self.num_layers):
            # Set spatial dimensions for blocks
            for blk in self.layers[i_layer]:
                blk.H, blk.W = H_enc, W_enc
                x = blk(x)

            # Convert to CNN format and store
            x_cnn = x.view(B, H_enc, W_enc, -1).permute(0, 3, 1, 2)
            x_cnn = self.proj_layers[i_layer](x_cnn)
            enc_features.append(x_cnn)

            # Downsample
            if i_layer < self.num_layers - 1:
                x = self.downsample_layers[i_layer](x, H_enc, W_enc)
                H_enc = H_enc // 2
                W_enc = W_enc // 2

        # Decoder with nested dense skip connections (UNet++ style)
        x0_0, x1_0, x2_0, x3_0 = enc_features

        # First column of nested blocks
        x2_1 = self.up3(x3_0)
        x2_1 = torch.cat([x2_1, x2_0], dim=1)
        x2_1 = self.conv2_1(x2_1)

        x1_1 = self.up2(x2_1)
        x1_1 = torch.cat([x1_1, x1_0], dim=1)
        x1_1 = self.conv1_1(x1_1)

        x0_1 = self.up1(x1_1)
        x0_1 = torch.cat([x0_1, x0_0], dim=1)
        x0_1 = self.conv0_1(x0_1)

        # Second column
        x1_2 = self.up2(x2_1)
        x1_2 = torch.cat([x1_2, x1_0, x1_1], dim=1)
        x1_2 = self.conv1_2(x1_2)

        x0_2 = self.up1(x1_2)
        x0_2 = torch.cat([x0_2, x0_0, x0_1], dim=1)
        x0_2 = self.conv0_2(x0_2)

        # Third column
        x0_3 = self.up1(x1_2)
        x0_3 = torch.cat([x0_3, x0_0, x0_1, x0_2], dim=1)
        x0_3 = self.conv0_3(x0_3)

        # Final upsampling to original resolution
        x_out = F.interpolate(x0_3, size=(
            H, W), mode='bilinear', align_corners=False)

        # Output
        output = self.final(x_out)

        return output


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
# METRICS
# ============================================================================

def calculate_iou(pred, target, num_classes):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return ious


# ============================================================================
# ENHANCED TRAINER WITH METRICS
# ============================================================================

class Trainer:
    """Enhanced training pipeline with comprehensive metrics"""

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
        self.val_ious = []
        self.inference_times = []
        self.best_val_loss = float('inf')

        # Calculate model size
        self.model_params = sum(p.numel()
                                for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {self.model_params:,}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for images, masks, _ in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.config.DEVICE)
            masks = masks.to(self.config.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate the model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_ious = []
        inference_times = []

        with torch.no_grad():
            for images, masks, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)

                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = (time.time() - start_time) * \
                    1000 / images.size(0)
                inference_times.append(inference_time)

                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                # Calculate IoU
                preds = torch.argmax(outputs, dim=1)
                ious = calculate_iou(preds, masks, self.config.NUM_CLASSES)
                all_ious.append(ious)

        avg_loss = total_loss / len(self.val_loader)
        avg_inference_time = np.mean(inference_times)

        # Calculate mean IoU (ignoring NaN values)
        mean_iou_per_class = np.nanmean(all_ious, axis=0)
        mean_iou = np.nanmean(mean_iou_per_class)

        return avg_loss, mean_iou, avg_inference_time

    def train(self):
        """Full training loop with metrics logging"""
        print(f"Training on {self.config.DEVICE}")
        print(f"Total Parameters: {self.model_params:,}")

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_iou, inference_time = self.validate()
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.inference_times.append(inference_time)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val IoU: {val_iou:.4f}")
            print(f"Inference Time: {inference_time:.2f} ms/image")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    'best_swin_unet_plusplus_model.pth'
                )
                print("âœ“ Saved best model")

    def plot_training_history(self):
        """Plot comprehensive training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training & Validation Loss')

        # IoU
        axes[0, 1].plot(self.val_ious, label='Val IoU', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].set_title('Validation IoU')

        # Inference Time
        axes[1, 0].plot(self.inference_times,
                        label='Inference Time', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].legend()
        axes[1, 0].set_title('Inference Time per Image')

        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Model: Swin-UNet++
        
        Parameters: {self.model_params:,}
        
        Best Val Loss: {self.best_val_loss:.4f}
        Best Val IoU: {max(self.val_ious):.4f}
        
        Avg Inference Time: {np.mean(self.inference_times):.2f} ms
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                        verticalalignment='center')

        plt.tight_layout()
        plt.savefig('swin_unet_plusplus_training_history.png', dpi=150)
        plt.close()


# ============================================================================
# INFERENCE
# ============================================================================

class ThermalFaceDetector:
    """Inference class for detecting facial regions in thermal images"""

    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model = SwinUNetPlusPlus(
            img_size=config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=config.NUM_CLASSES
        )
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
        """Predict facial regions in a thermal image"""
        original_shape = thermal_image.shape

        thermal_image_norm = self.normalize_thermal(thermal_image)

        thermal_image_resized = cv2.resize(
            thermal_image_norm, self.config.IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        image_tensor = torch.from_numpy(
            thermal_image_resized).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.float().to(self.config.DEVICE)

        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        regions = {}
        for idx, region_name in enumerate(self.config.REGION_NAMES):
            region_mask = (pred_mask == idx).astype(np.uint8)
            regions[region_name] = region_mask

        return regions, pred_mask

    def get_stats_info(self, thermal_image: np.ndarray, regions: Dict[str, np.ndarray]):
        """Calculate statistics for each region"""
        stats_info = {}

        for region_name, mask in regions.items():
            region_temps = thermal_image[mask == 1]

            if len(region_temps) == 0:
                stats_info[region_name] = {
                    'mean': None, 'median': None, 'mode': None,
                    'std': None, 'min': None, 'max': None, 'pixel_count': 0
                }
                continue

            mean_temp = np.mean(region_temps)
            median_temp = np.median(region_temps)
            std_temp = np.std(region_temps)
            min_temp = np.min(region_temps)
            max_temp = np.max(region_temps)

            bins = np.arange(region_temps.min(), region_temps.max() + 0.1, 0.1)
            hist, bin_edges = np.histogram(region_temps, bins=bins)
            mode_idx = np.argmax(hist)
            mode_temp = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

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

        axes[0].imshow(thermal_image, cmap='hot')
        axes[0].set_title('Original Thermal Image')
        axes[0].axis('off')

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
        """Print thermal statistics report"""
        print("\n" + "="*80)
        print("THERMAL STATISTICS REPORT (Â°C) - Swin-UNet++")
        print("="*80)

        for region_name, stats in stats_info.items():
            print(f"\n{region_name}:")
            print("-" * 80)

            if stats['pixel_count'] == 0:
                print("  No pixels detected in this region")
                continue

            print(f"  Mean:        {stats['mean']:.2f} Â°C")
            print(f"  Median:      {stats['median']:.2f} Â°C")
            print(f"  Mode:        {stats['mode']:.2f} Â°C")
            print(f"  Std Dev:     {stats['std']:.2f} Â°C")
            print(f"  Min:         {stats['min']:.2f} Â°C")
            print(f"  Max:         {stats['max']:.2f} Â°C")
            print(f"  Pixel Count: {stats['pixel_count']}")

        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    config = Config()

    print("Loading annotations...")
    data_loader = ThermalDataLoader(config)
    data_loader.load_annotations()

    if data_loader.polygons is None:
        print("Error: Polygons not loaded. Check if annotation files exist.")
        return

    sample_ids = list(data_loader.polygons.keys())

    train_ids, val_ids = train_test_split(
        sample_ids, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")

    train_dataset = ThermalFaceDataset(
        train_ids, data_loader, config, augment=True)
    val_dataset = ThermalFaceDataset(
        val_ids, data_loader, config, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print("\nInitializing Swin-UNet++ model...")
    model = SwinUNetPlusPlus(
        img_size=config.IMAGE_SIZE[0],
        in_channels=1,
        num_classes=config.NUM_CLASSES
    )

    print("\nStarting training...")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    trainer.plot_training_history()

    print("\nTesting inference...")
    detector = ThermalFaceDetector('best_swin_unet_plusplus_model.pth', config)

    test_id = val_ids[0]
    test_image = data_loader.load_thermal_image(test_id)
    test_image = data_loader.crop_to_bbox(test_image, test_id)

    regions, pred_mask = detector.predict(test_image)
    detector.visualize_predictions(
        test_image, regions,
        save_path='swin_unet_plusplus_prediction_example.png'
    )

    print("\nâœ“ Training completed successfully!")
    print(f"Best model saved to: best_swin_unet_plusplus_model.pth")


def execution():
    """Execution function for inference/demo"""
    config = Config()

    print("Loading annotations...")
    data_loader = ThermalDataLoader(config)
    data_loader.load_annotations()

    if data_loader.polygons is None:
        print("Error: Polygons not loaded. Check if annotation files exist.")
        return

    sample_ids = list(data_loader.polygons.keys())

    train_ids, val_ids = train_test_split(
        sample_ids, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")

    print("\nTesting inference...")
    detector = ThermalFaceDetector('best_swin_unet_plusplus_model.pth', config)

    test_id = val_ids[1]
    test_image = data_loader.load_thermal_image(test_id)
    test_image = data_loader.crop_to_bbox(test_image, test_id)

    print("Min and Max values of test_image")
    print(test_image.min(), test_image.max())

    regions, pred_mask = detector.predict(test_image)
    stats_info = detector.get_stats_info(test_image, regions)
    detector.print_stats_report(stats_info)

    background_mean = stats_info['background']['mean']
    nose_temp_std = stats_info['Nariz']['std']


if __name__ == "__main__":
    execution()
