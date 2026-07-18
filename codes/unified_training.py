"""
Unified Training Module
=======================
Consistent training loops, loss functions, and metrics for all models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
try:
    from codes.unified_data import Config
    from codes.naming import checkpoint_path, epoch_checkpoint_glob
    from codes.metrics import SegmentationMetrics
except ImportError:
    from unified_data import Config
    from naming import checkpoint_path, epoch_checkpoint_glob
    from metrics import SegmentationMetrics


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
    """Dice Loss for segmentation.

    NOTE: softmax on fp16 logits overflows (exp of large values → inf → NaN).
    We explicitly cast to float32 before softmax to stay numerically stable
    under AMP autocast.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Cast to fp32 to prevent softmax overflow under AMP
        pred = F.softmax(pred.float(), dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss.

    The forward pass is wrapped with autocast(enabled=False) so that all
    loss arithmetic runs in float32 regardless of the outer AMP context.
    This prevents fp16 overflow in softmax (Dice) and log-sum-exp (CE).
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        # Disable autocast so that all loss math runs in fp32
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Ensure inputs are fp32 for numerically stable loss computation
            pred = pred.float()
            ce = self.ce_loss(pred, target)
            dice = self.dice_loss(pred, target)
            return self.ce_weight * ce + self.dice_weight * dice


class UnifiedTrainer:
    """
    Unified training pipeline for all models
    Ensures consistent training, validation, and metric tracking
    """

    # How many epoch checkpoints to keep on disk (older ones are pruned).
    # Resume only needs the most recent (UB-06 restores from the latest epoch),
    # so keeping 1 halves the full-state checkpoint footprint that near-full
    # disks turned into spurious ENOSPC test failures (UB-25). Re-verified
    # against test_resume.py.
    _CHECKPOINT_KEEP_LAST = 1

    def __init__(self, model: nn.Module, model_name: str,
                 train_loader: DataLoader, val_loader: DataLoader,
                 config: Config, learning_rate: float = 1e-4,
                 num_epochs: int = 100, grad_clip_norm: float = 1.0,
                 max_nan_tolerance: int = 50, *,
                 model_key: str, fold: int):
        """
        Initialize trainer

        Args:
            model: PyTorch model to train
            model_name: Name of the model (for logging, plots, metrics files)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            grad_clip_norm: Maximum gradient norm for clipping (stabilises AMP)
            max_nan_tolerance: Abort training after this many consecutive NaN batches
            model_key: Registry key (``unet``/``transunet``/``swin_unet_plus_plus``)
                — the sole source for checkpoint filenames (UB-02/R5)
            fold: 1-based fold number, embedded in checkpoint filenames
        """
        self.model = model.to(config.DEVICE)
        self.model_name = model_name
        self.model_key = model_key
        self.fold = fold
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.num_epochs = num_epochs
        self.grad_clip_norm = grad_clip_norm
        self.max_nan_tolerance = max_nan_tolerance

        # Compute class weights if they are present in config (otherwise None)
        class_weights = None
        if hasattr(config, 'CLASS_WEIGHTS') and config.CLASS_WEIGHTS is not None:
            class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float).to(config.DEVICE)
            
        # Loss and optimizer
        self.criterion = CombinedLoss(class_weights=class_weights)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Validation metrics go through the single shared authority (UB-11/R5):
        # hard IoU + hard Dice on argmax, macro excluding classes absent from
        # the target, background reported separately.  The benchmark uses the
        # same SegmentationMetrics so the two toolchains agree by construction.
        self.val_metrics = SegmentationMetrics(config.NUM_CLASSES, device=config.DEVICE)

        # Automatic Mixed Precision (AMP) — enabled on GPUs with reliable fp16
        # Tensor Core support (RTX 20xx / 30xx / 40xx, A-series, etc.).
        # GTX cards (e.g. GTX 1660 Ti, sm_75) are excluded: they lack
        # dedicated fp16 Tensor Cores so AMP would give no speed gain and
        # risks numerical instability in the softmax / loss path.
        self.scaler = self._build_scaler(config)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dice_scores = []
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0

        # Model parameters
        self.model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Training time
        self.training_start_time = None
        self.training_end_time = None

        # Checkpoint directory — lives inside the run-specific output directory
        self._checkpoint_dir = config.OUTPUT_DIR / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Initializing {model_name} Trainer")
        print(f"{'='*70}")
        print(f"Model Parameters: {self.model_params:,}")
        print(f"Training Device: {config.DEVICE}")
        print(f"Train Batches: {len(train_loader)}")
        print(f"Val Batches: {len(val_loader)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Gradient Clip Norm: {grad_clip_norm}")
        print(f"Checkpoint Dir: {self._checkpoint_dir}")
        amp_status = "enabled" if self.scaler is not None else "disabled (GTX/CPU)"
        print(f"AMP (fp16):      {amp_status}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # AMP helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scaler(config) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Return a GradScaler when the active GPU can benefit from AMP, else None.

        Rules
        -----
        - CPU or no CUDA  → None (AMP only works on CUDA).
        - GTX-class GPU   → None (no fp16 Tensor Cores; AMP = overhead with no gain).
        - RTX / A-series  → GradScaler (sm ≥ 7.5 with true Tensor Core support).
        """
        if not torch.cuda.is_available() or str(config.DEVICE) == 'cpu':
            return None

        gpu_name = torch.cuda.get_device_name(0).upper()
        major, _ = torch.cuda.get_device_capability(0)

        # GTX Turing cards (sm_75) have fp16 hardware but no dedicated Tensor
        # Core training pipelines — skip AMP to keep training numerically stable.
        is_gtx = 'GTX' in gpu_name
        if is_gtx or major < 7:
            return None

        return torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, epoch: int) -> Path:
        """Return the expected path for a given epoch checkpoint (UB-02/R5)."""
        return checkpoint_path(self.config.OUTPUT_DIR, self.model_key,
                               self.fold, "epoch", epoch=epoch)

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Scan the checkpoint directory for the most recent epoch file."""
        import glob
        pattern = epoch_checkpoint_glob(self.config.OUTPUT_DIR,
                                        self.model_key, self.fold)
        candidates = sorted(glob.glob(pattern))
        return Path(candidates[-1]) if candidates else None

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float) -> Path:
        """
        Save a full training checkpoint at the end of *epoch* (0-indexed).

        The checkpoint contains everything needed to resume exactly where
        training was interrupted:
          - epoch index (the epoch just finished)
          - model weights
          - optimizer state
          - scheduler state
          - scaler state (AMP, may be None)
          - all metric history up to this point
          - best-so-far tracking scalars

        Older checkpoints beyond `_CHECKPOINT_KEEP_LAST` are pruned to
        avoid filling the disk with .pth files.

        Returns
        -------
        Path to the saved checkpoint file.
        """
        ckpt_path = self._checkpoint_path(epoch)
        checkpoint = {
            # ── identity ──────────────────────────────────────────────
            'epoch': epoch,
            'model_name': self.model_name,
            # ── learnable state ───────────────────────────────────────
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            # ── loss snapshot ─────────────────────────────────────────
            'train_loss': train_loss,
            'val_loss': val_loss,
            # ── metric history ────────────────────────────────────────
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'val_dice_scores': self.val_dice_scores,
            # ── best-so-far trackers ──────────────────────────────────
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"  💾 Checkpoint saved: {ckpt_path.name}")

        # Prune old checkpoints, keep the last N
        import glob
        pattern = epoch_checkpoint_glob(self.config.OUTPUT_DIR,
                                        self.model_key, self.fold)
        all_ckpts = sorted(glob.glob(pattern))
        for old_ckpt in all_ckpts[: -self._CHECKPOINT_KEEP_LAST]:
            try:
                Path(old_ckpt).unlink()
                print(f"  🗑️  Pruned old checkpoint: {Path(old_ckpt).name}")
            except OSError:
                pass  # Non-fatal if the file is already gone

        return ckpt_path

    def load_checkpoint(self, ckpt_path: Path) -> int:
        """
        Restore trainer state from *ckpt_path*.

        Loads model weights, optimizer, scheduler, scaler, and all metric
        history so that `train()` can continue seamlessly from where it
        left off.

        Returns
        -------
        The epoch index that was *last completed* (i.e. training should
        resume from ``returned_epoch + 1``).
        """
        print(f"  ♻️  Resuming from checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=self.config.DEVICE,
                                weights_only=True)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore metric histories
        self.train_losses      = checkpoint.get('train_losses', [])
        self.val_losses        = checkpoint.get('val_losses', [])
        self.val_ious          = checkpoint.get('val_ious', [])
        self.val_dice_scores   = checkpoint.get('val_dice_scores', [])
        # 'inference_times' (present in pre-UB-09 checkpoints) is intentionally
        # ignored — torch.load still reads the whole dict, so old checkpoints
        # load fine; the key is simply no longer restored (T2.1).
        self.best_val_loss     = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_iou      = checkpoint.get('best_val_iou', 0.0)

        last_epoch: int = checkpoint['epoch']
        print(f"  ✅ Resumed at epoch {last_epoch + 1} "
              f"(best val loss so far: {self.best_val_loss:.4f})")
        return last_epoch
    
    def train_epoch(self) -> float:
        """Train for one epoch with gradient clipping and NaN protection"""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        consecutive_nan = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training {self.model_name}")
        for images, masks, _ in pbar:
            images = images.to(self.config.DEVICE, non_blocking=True)
            masks = masks.to(self.config.DEVICE, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with Automatic Mixed Precision (AMP)
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # NaN protection: skip backward pass if loss is NaN/Inf
                if not math.isfinite(loss.item()):
                    consecutive_nan += 1
                    if consecutive_nan == 1:
                        print("\n=== DEBUG NaN DETECTED ===")
                        print(f"Model: {self.model_name}")
                        print(f"Images: min={images.min().item():.4f}, max={images.max().item():.4f}, has_nan={torch.isnan(images).any().item()}")
                        print(f"Masks: min={masks.min().item()}, max={masks.max().item()}, unique={torch.unique(masks).tolist()}, has_nan={torch.isnan(masks).any().item()}")
                        print(f"Outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, has_nan={torch.isnan(outputs).any().item()}")
                        try:
                            with torch.amp.autocast('cuda', enabled=False):
                                pred_float = outputs.float()
                                ce_val = self.criterion.ce_loss(pred_float, masks).item()
                                dice_val = self.criterion.dice_loss(pred_float, masks).item()
                                print(f"Loss components (FP32): CE={ce_val:.4f}, Dice={dice_val:.4f}")
                        except Exception as e:
                            print(f"Error computing loss components: {e}")
                        print("==========================\n")
                    pbar.set_postfix({'loss': 'NaN', 'nan_streak': consecutive_nan})
                    if consecutive_nan >= self.max_nan_tolerance:
                        raise RuntimeError(
                            f"{self.model_name}: {consecutive_nan} consecutive NaN losses. "
                            f"Training is numerically unstable — aborting."
                        )
                    continue
                consecutive_nan = 0
                
                # Backward pass with gradient clipping (AMP-compatible)
                self.scaler.scale(loss).backward()
                # Unscale gradients before clipping so the threshold is in fp32 scale
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                if not math.isfinite(loss.item()):
                    consecutive_nan += 1
                    if consecutive_nan == 1:
                        print("\n=== DEBUG NaN DETECTED ===")
                        print(f"Model: {self.model_name}")
                        print(f"Images: min={images.min().item():.4f}, max={images.max().item():.4f}, has_nan={torch.isnan(images).any().item()}")
                        print(f"Masks: min={masks.min().item()}, max={masks.max().item()}, unique={torch.unique(masks).tolist()}, has_nan={torch.isnan(masks).any().item()}")
                        print(f"Outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, has_nan={torch.isnan(outputs).any().item()}")
                        try:
                            pred_float = outputs.float()
                            ce_val = self.criterion.ce_loss(pred_float, masks).item()
                            dice_val = self.criterion.dice_loss(pred_float, masks).item()
                            print(f"Loss components (FP32): CE={ce_val:.4f}, Dice={dice_val:.4f}")
                        except Exception as e:
                            print(f"Error computing loss components: {e}")
                        print("==========================\n")
                    pbar.set_postfix({'loss': 'NaN', 'nan_streak': consecutive_nan})
                    if consecutive_nan >= self.max_nan_tolerance:
                        raise RuntimeError(
                            f"{self.model_name}: {consecutive_nan} consecutive NaN losses. "
                            f"Training is numerically unstable — aborting."
                        )
                    continue
                consecutive_nan = 0
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if valid_batches == 0:
            return float('nan')
        return total_loss / valid_batches
    
    def validate(self) -> Tuple[float, float, float]:
        """
        Validate the model with comprehensive metrics.

        Returns:
            avg_loss, mean_iou, mean_dice

        Per-epoch inference timing was removed (UB-09, T2.1): it was measured
        without ``torch.cuda.synchronize()`` — on GPU that captures kernel
        *launch* time, not compute — and the timed span also included the loss.
        Trustworthy latency is measured once in the benchmark via
        ``benchmark_models.timed_inference`` with warm-up discard (M3).
        """
        self.model.eval()
        total_loss = 0

        self.val_metrics.reset()

        pbar = tqdm(self.val_loader, desc=f"Validating {self.model_name}")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.config.DEVICE, non_blocking=True)
                masks = masks.to(self.config.DEVICE, non_blocking=True)

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                total_loss += loss.item()
                self.val_metrics.update(outputs, masks)   # argmax taken inside

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.val_loader)

        # Hard IoU / Dice from the single shared authority: macro over classes
        # present in the target, background excluded (UB-11/M2).
        seg = self.val_metrics.compute()
        mean_iou = seg["mean_iou"]
        mean_dice = seg["mean_dice"]

        return avg_loss, mean_iou, mean_dice
    
    def train(self):
        """Full training loop with per-epoch checkpointing and auto-resume."""
        print(f"\n{'='*70}")
        print(f"Starting Training: {self.model_name}")
        print(f"{'='*70}\n")

        self.training_start_time = time.time()

        # ── Auto-resume: detect the latest checkpoint for this model ──
        start_epoch = 0
        latest_ckpt = self._find_latest_checkpoint()
        if latest_ckpt is not None:
            start_epoch = self.load_checkpoint(latest_ckpt) + 1
            if start_epoch >= self.num_epochs:
                print(f"  ⚠️  All {self.num_epochs} epochs already completed "
                      f"— nothing to do. Call plot_training_history() or "
                      f"save_metrics() to export results.")
                self.training_end_time = time.time()
                return
        # ─────────────────────────────────────────────────────────────

        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 70)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_iou, val_dice = self.validate()
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dice_scores.append(val_dice)

            # Print metrics
            print(f"\nMetrics:")
            print(f"  Train Loss:     {train_loss:.4f}")
            print(f"  Val Loss:       {val_loss:.4f}")
            print(f"  Val mIoU:       {val_iou:.4f}")
            print(f"  Val Dice:       {val_dice:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate:  {current_lr:.6f}")

            # Save best model (weights-only, based on validation loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = checkpoint_path(self.config.OUTPUT_DIR,
                                             self.model_key, self.fold, "best")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f"  ✅ Saved best model to: {model_path}")

            # Track best IoU
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou

            # ── Per-epoch checkpoint (full state — enables crash recovery) ──
            self.save_checkpoint(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            # ────────────────────────────────────────────────────────────────

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
        
        # Per-epoch inference timing was removed (UB-09, T2.1) — it was an
        # unsynced, loss-polluted measurement.  The 4th panel is left blank;
        # trustworthy latency lives in the benchmark report.
        axes[1, 1].axis('off')

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
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ious": self.val_ious,
            "val_dice_scores": self.val_dice_scores,
        }
        
        save_path = save_dir / f"{_safe_filename(self.model_name)}_metrics.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Training metrics saved to: {save_path}")
        
        return metrics


class ThermalFaceDetector:
    """Inference class for detecting facial regions in thermal images"""

    def __init__(self, model: nn.Module, model_path: str, config: Config):
        self.config = config
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        self.model.to(config.DEVICE)
        self.model.eval()

    def normalize_thermal(self, thermal_img: np.ndarray) -> np.ndarray:
        """Normalize thermal image to [0, 1] range.

        A flat image (min == max) normalizes to all-zeros, not the raw values —
        returning the un-normalized image would leak out-of-range magnitudes
        into a supposedly [0, 1] tensor (UB-15).
        """
        min_val = thermal_img.min()
        max_val = thermal_img.max()
        if max_val - min_val > 0:
            return (thermal_img - min_val) / (max_val - min_val)
        return np.zeros_like(thermal_img, dtype=np.float32)

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
            # For continuous data, we bin the temperatures using 0.1°C bins
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
        # Grid sized to fit the original + every region. The old 2x5 (10 axes)
        # silently dropped the last of 11 panels (1 original + 10 regions) — UB-15.
        n_panels = 1 + len(regions)
        ncols = 4
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.asarray(axes).flatten()

        # Show original image
        axes[0].imshow(thermal_image, cmap='hot')
        axes[0].set_title('Original Thermal Image')
        axes[0].axis('off')

        # Show each region (the grid now has room for all of them)
        for idx, (region_name, mask) in enumerate(regions.items(), 1):
            axes[idx].imshow(mask, cmap='gray')
            axes[idx].set_title(region_name, fontsize=8)
            axes[idx].axis('off')

        # Hide any spare axes
        for spare in range(n_panels, len(axes)):
            axes[spare].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

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


if __name__ == "__main__":
    print("Unified Training Module - Ready")
    print("Import this module to use UnifiedTrainer for model training")
