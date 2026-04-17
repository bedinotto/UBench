"""
Unified Training Module
=======================
Consistent training loops, loss functions, and metrics for all models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    from codes.unified_data import Config
except ImportError:
    from unified_data import Config


def _safe_filename(name: str) -> str:
    """
    Convert a model name to a filesystem-safe filename stem.

    Rules (cross-platform — Windows is the most restrictive):
    - Spaces        → underscore
    - Hyphens       → underscore
    - Plus signs    → 'plus'
    - Any remaining non-alphanumeric/underscore chars are dropped
    - Result is lowercased

    Examples
    --------
    'U-Net'         → 'u_net'
    'TransUNet'     → 'transunet'
    'Swin-UNet++'   → 'swin_unetplusplus'
    """
    import re
    s = name.lower()
    s = s.replace(' ', '_')
    s = s.replace('-', '_')
    s = s.replace('+', 'plus')
    # Drop any character that is not alphanumeric or underscore
    s = re.sub(r'[^\w]', '', s)
    # Collapse consecutive underscores
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
        
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


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> List[float]:
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


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Calculate Dice score"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()


class UnifiedTrainer:
    """
    Unified training pipeline for all models
    Ensures consistent training, validation, and metric tracking
    """
    
    def __init__(self, model: nn.Module, model_name: str,
                 train_loader: DataLoader, val_loader: DataLoader,
                 config: Config, learning_rate: float = 1e-4,
                 num_epochs: int = 100):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            model_name: Name of the model (for logging)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.model = model.to(config.DEVICE)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.num_epochs = num_epochs
        
        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dice_scores = []
        self.inference_times = []
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        
        # Model parameters
        self.model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training time
        self.training_start_time = None
        self.training_end_time = None
        
        print(f"\n{'='*70}")
        print(f"Initializing {model_name} Trainer")
        print(f"{'='*70}")
        print(f"Model Parameters: {self.model_params:,}")
        print(f"Training Device: {config.DEVICE}")
        print(f"Train Batches: {len(train_loader)}")
        print(f"Val Batches: {len(val_loader)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*70}\n")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training {self.model_name}")
        for images, masks, _ in pbar:
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float, float, float]:
        """
        Validate the model with comprehensive metrics
        
        Returns:
            avg_loss, mean_iou, mean_dice, avg_inference_time
        """
        self.model.eval()
        total_loss = 0
        all_ious = []
        all_dice_scores = []
        inference_times = []
        
        pbar = tqdm(self.val_loader, desc=f"Validating {self.model_name}")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = (time.time() - start_time) * 1000 / images.size(0)
                inference_times.append(inference_time)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate IoU
                preds = torch.argmax(outputs, dim=1)
                ious = calculate_iou(preds, masks, self.config.NUM_CLASSES)
                all_ious.append(ious)
                
                # Calculate Dice score
                dice = calculate_dice_score(outputs, masks, self.config.NUM_CLASSES)
                all_dice_scores.append(dice)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_inference_time = np.mean(inference_times)
        
        # Calculate mean IoU (ignoring NaN values)
        mean_iou_per_class = np.nanmean(all_ious, axis=0)
        mean_iou = np.nanmean(mean_iou_per_class)
        
        # Calculate mean Dice score
        mean_dice = np.mean(all_dice_scores)
        
        return avg_loss, mean_iou, mean_dice, avg_inference_time
    
    def train(self):
        """Full training loop with metrics logging"""
        print(f"\n{'='*70}")
        print(f"Starting Training: {self.model_name}")
        print(f"{'='*70}\n")
        
        self.training_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_iou, val_dice, inference_time = self.validate()
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dice_scores.append(val_dice)
            self.inference_times.append(inference_time)
            
            # Print metrics
            print(f"\nMetrics:")
            print(f"  Train Loss:     {train_loss:.4f}")
            print(f"  Val Loss:       {val_loss:.4f}")
            print(f"  Val mIoU:       {val_iou:.4f}")
            print(f"  Val Dice:       {val_dice:.4f}")
            print(f"  Inference Time: {inference_time:.2f} ms/image")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate:  {current_lr:.6f}")
            
            # Save best model (based on validation loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                models_dir = self.config.OUTPUT_DIR / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                model_path = models_dir / f"best_{_safe_filename(self.model_name)}_model.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"  ✅ Saved best model to: {model_path}")
            
            # Track best IoU
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
            
            print("-" * 70)
        
        self.training_end_time = time.time()
        training_duration = (self.training_end_time - self.training_start_time) / 60  # minutes
        
        print(f"\n{'='*70}")
        print(f"✅ Training Completed: {self.model_name}")
        print(f"{'='*70}")
        print(f"Training Duration: {training_duration:.1f} minutes")
        print(f"Best Val Loss:     {self.best_val_loss:.4f}")
        print(f"Best Val mIoU:     {self.best_val_iou:.4f}")
        print(f"{'='*70}\n")
    
    def plot_training_history(self, save_dir: Optional[Path] = None):
        """Plot comprehensive training metrics"""
        if save_dir is None:
            save_dir = self.config.OUTPUT_DIR / "plots"
        save_dir.mkdir(exist_ok=True, parents=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU
        axes[0, 1].plot(self.val_ious, label='Val mIoU', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('mIoU', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].set_title('Validation mIoU', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice Score
        axes[1, 0].plot(self.val_dice_scores, label='Val Dice', color='blue', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Dice Score', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inference Time
        axes[1, 1].plot(self.inference_times, label='Inference Time', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Time (ms)', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].set_title('Inference Time per Image', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_name} Training History', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = save_dir / f"{_safe_filename(self.model_name)}_training_history.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training history plot saved to: {save_path}")
    
    def save_metrics(self, save_dir: Optional[Path] = None):
        """Save training metrics to JSON"""
        if save_dir is None:
            save_dir = self.config.LOG_DIR
        save_dir.mkdir(exist_ok=True, parents=True)
        
        training_duration = None
        if self.training_start_time and self.training_end_time:
            training_duration = (self.training_end_time - self.training_start_time) / 60
        
        metrics = {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "num_epochs": self.num_epochs,
            "training_duration_minutes": training_duration,
            "best_val_loss": self.best_val_loss,
            "best_val_iou": self.best_val_iou,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "final_val_iou": self.val_ious[-1] if self.val_ious else None,
            "final_val_dice": self.val_dice_scores[-1] if self.val_dice_scores else None,
            "avg_inference_time_ms": np.mean(self.inference_times) if self.inference_times else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ious": self.val_ious,
            "val_dice_scores": self.val_dice_scores,
            "inference_times": self.inference_times
        }
        
        save_path = save_dir / f"{_safe_filename(self.model_name)}_metrics.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Training metrics saved to: {save_path}")
        
        return metrics


if __name__ == "__main__":
    print("Unified Training Module - Ready")
    print("Import this module to use UnifiedTrainer for model training")
