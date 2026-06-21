"""
Main Training Pipeline Orchestrator
===================================
Manages the complete training and benchmarking pipeline
"""

import sys
import os
import gc
import traceback
import multiprocessing
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to sys.path *before* importing codes modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from codes.logger import TeeLogger



# Import modules
from codes.hardware_detector import detect_and_optimize, HardwareProfile
from codes.unified_data import (
    Config, create_kfold_data_loaders, seed_everything,
    MultiDirectoryDataLoader, shutdown_data_loaders,
)
from codes.unified_training import UnifiedTrainer

# Import model architectures
import codes.unet_v2 as unet_v2
import codes.transunet as transunet
import codes.swin_unet_plus_plus as swin_unet_plus_plus
import codes.benchmark_models as benchmark_models


class Pipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, models_to_train=None, skip_benchmark=False):
        """
        Initialize pipeline
        
        Args:
            models_to_train: List of model names to train. 
                           None = all models. Options: ['unet', 'transunet', 'swin']
            skip_benchmark: If True, skip benchmarking after training
        """
        self.models_to_train = models_to_train or ['unet', 'transunet', 'swin']
        self.skip_benchmark = skip_benchmark
        
        print("\n" + "="*80)
        print("THERMAL FACE DETECTION - AUTOMATED TRAINING PIPELINE")
        print("="*80 + "\n")
        
        # Create timestamped run directories
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_output_dir = Path("outputs") / self.run_timestamp
        self.run_log_dir = Path("logs") / self.run_timestamp
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        # ── Start full-run console logger ──────────────────────────────────
        # Expose log dir via env var so child scripts (setup, extract) can
        # append to the same log directory.
        os.environ["UBENCH_LOG_DIR"] = str(self.run_log_dir)
        self._tee = TeeLogger(self.run_log_dir / "pipeline.log")
        self._tee.start()
        # ──────────────────────────────────────────────────────────────────

        print(f"Run ID:     {self.run_timestamp}")
        print(f"Output dir: {self.run_output_dir}")
        print(f"Log dir:    {self.run_log_dir}")
        print(f"Console log:{self.run_log_dir / 'pipeline.log'}")
        
        # Detect hardware and optimize
        print("\nStep 1: Hardware Detection & Optimization")
        print("-"*80)
        self.hardware_profile = detect_and_optimize(
            log_dir=str(self.run_log_dir)
        )
        
        # Initialize configuration with timestamped dirs
        print("\nStep 2: Configuration & Data Validation")
        print("-"*80)
        self.config = Config(
            output_dir=str(self.run_output_dir),
            log_dir=str(self.run_log_dir)
        )
        print("✅ Configuration validated")
        print("✅ Data directories verified")
        
        # Set global seed
        seed_everything(self.config.RANDOM_SEED)
        print(f"✅ Global seeds fixed to {self.config.RANDOM_SEED}")
        
        # Store training results (metrics only — models are saved to disk)
        self.training_results = {}
    
    def load_shared_data(self):
        """Load annotations once (lightweight) — DataLoaders are created lazily."""
        print("\nStep 3: Data Loading (annotations)")
        print("-"*80)

        # Load annotations from all Sx directories once — this is cheap.
        self._shared_data_loader = MultiDirectoryDataLoader(self.config)
        self._shared_data_loader.load_annotations()
        print(f"\u2705 Annotations loaded (DataLoaders will be created per-fold)\n")

    def _get_fold_loaders(self, model_name: str, fold_idx: int):
        """Create DataLoaders for a single model + fold (lazy, on-demand).

        This avoids the previous approach of pre-creating all 30 loaders
        (3 models × 5 folds × train/val) up front, which exhausted Windows
        shared-memory file mappings.
        """
        batch_size = self.hardware_profile.batch_sizes.get(model_name, 8)
        num_workers = self.hardware_profile.num_workers

        folds_data, _ = create_kfold_data_loaders(
            self.config,
            batch_size=batch_size,
            num_workers=num_workers,
            shared_data_loader=self._shared_data_loader,
        )
        return folds_data[fold_idx]

    @staticmethod
    def _cleanup_fold_resources(*loaders):
        """Shut down DataLoader workers and free GPU memory between runs."""
        shutdown_data_loaders(*loaders)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_unet(self, fold_idx: int):
        """Train U-Net model on a specific fold"""
        print("\n" + "="*80)
        print(f"TRAINING: U-Net (Fold {fold_idx + 1}/{self.config.K_FOLDS})")
        print("="*80)
        
        # Create loaders lazily for this fold
        loaders = self._get_fold_loaders('unet', fold_idx)
        
        # Create model
        model = unet_v2.UNet(in_channels=1, num_classes=self.config.NUM_CLASSES)
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name=f"U-Net_Fold-{fold_idx + 1}",
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        if 'U-Net' not in self.training_results:
            self.training_results['U-Net'] = []
            
        self.training_results['U-Net'].append(metrics)
        
        # Cleanup: release loaders + GPU memory before next model
        del model, trainer
        self._cleanup_fold_resources(loaders['train_loader'], loaders['val_loader'])
        
        return metrics
    
    def train_transunet(self, fold_idx: int):
        """Train TransUNet model on a specific fold"""
        print("\n" + "="*80)
        print(f"TRAINING: TransUNet (Fold {fold_idx + 1}/{self.config.K_FOLDS})")
        print("="*80)
        
        # Clear GPU cache before heavy model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create loaders lazily for this fold
        loaders = self._get_fold_loaders('transunet', fold_idx)
        
        # Create model
        model = transunet.TransUNet(
            img_size=self.config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=self.config.NUM_CLASSES
        )
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name=f"TransUNet_Fold-{fold_idx + 1}",
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        if 'TransUNet' not in self.training_results:
            self.training_results['TransUNet'] = []
            
        self.training_results['TransUNet'].append(metrics)
        
        # Cleanup: release loaders + GPU memory before next model
        del model, trainer
        self._cleanup_fold_resources(loaders['train_loader'], loaders['val_loader'])
        
        return metrics
    
    def train_swin_unet(self, fold_idx: int):
        """Train Swin-UNet++ model on a specific fold"""
        print("\n" + "="*80)
        print(f"TRAINING: Swin-UNet++ (Fold {fold_idx + 1}/{self.config.K_FOLDS})")
        print("="*80)
        
        # Clear GPU cache before heavy model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create loaders lazily for this fold
        loaders = self._get_fold_loaders('swin', fold_idx)
        
        # Create model
        model = swin_unet_plus_plus.SwinUNetPlusPlus(
            img_size=self.config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=self.config.NUM_CLASSES
        )
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name=f"Swin-UNet++_Fold-{fold_idx + 1}",
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        if 'Swin-UNet++' not in self.training_results:
            self.training_results['Swin-UNet++'] = []
            
        self.training_results['Swin-UNet++'].append(metrics)
        
        # Cleanup: release loaders + GPU memory before next model
        del model, trainer
        self._cleanup_fold_resources(loaders['train_loader'], loaders['val_loader'])
        
        return metrics
    
    def train_all_models(self):
        """Train all selected models sequentially over all folds"""
        print("\nStep 4: Model Training (K-Folds)")
        print("-"*80)
        
        start_time = datetime.now()
        
        for fold_idx in range(self.config.K_FOLDS):
            print(f"\n>>>> STARTING FOLD {fold_idx + 1}/{self.config.K_FOLDS} <<<<")
            
            # Train models based on selection
            if 'unet' in self.models_to_train:
                try:
                    self.train_unet(fold_idx)
                except Exception as e:
                    print(f"❌ U-Net training failed on Fold {fold_idx + 1}: {e}")
            
            if 'transunet' in self.models_to_train:
                try:
                    self.train_transunet(fold_idx)
                except Exception as e:
                    print(f"❌ TransUNet training failed on Fold {fold_idx + 1}: {e}")
            
            if 'swin' in self.models_to_train:
                try:
                    self.train_swin_unet(fold_idx)
                except Exception as e:
                    print(f"❌ Swin-UNet++ training failed on Fold {fold_idx + 1}: {e}")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print(f"✅ All training completed in {total_duration:.1f} minutes")
        print("="*80)
    
    def _create_benchmark_loaders(self):
        """Create a single validation loader for benchmarking (using first fold)."""
        first_model = self.models_to_train[0]
        loaders = self._get_fold_loaders(first_model, 0)
        return loaders['val_loader']

    def run_benchmark(self):
        """Run comprehensive benchmark on all trained models"""
        if self.skip_benchmark:
            print("\n⚠️  Skipping benchmark (--skip-benchmark flag set)")
            return
        
        print("\nStep 5: Comprehensive Benchmarking")
        print("-"*80)
        
        # Prepare models dictionary
        models_dict = {}
        
        if 'unet' in self.models_to_train:
            models_dict['U-Net'] = unet_v2.UNet(
                in_channels=1, num_classes=self.config.NUM_CLASSES
            )
        
        if 'transunet' in self.models_to_train:
            models_dict['TransUNet'] = transunet.TransUNet(
                img_size=self.config.IMAGE_SIZE[0],
                in_channels=1,
                num_classes=self.config.NUM_CLASSES
            )
        
        if 'swin' in self.models_to_train:
            models_dict['Swin-UNet++'] = swin_unet_plus_plus.SwinUNetPlusPlus(
                img_size=self.config.IMAGE_SIZE[0],
                in_channels=1,
                num_classes=self.config.NUM_CLASSES
            )
        
        # Run benchmark
        # We'll evaluate the fold 1 model on fold 1 validation data as a proxy for now, 
        # as benchmark_models expects a single model and loader.
        # Future improvement: modify benchmark_models to aggregate across all folds.
        print("Note: Benchmarking is currently evaluating Fold 1 models.")
        val_loader = self._create_benchmark_loaders()
        
        # Run benchmark
        comparison_df = benchmark_models.run_benchmark(
            models_dict, self.config, val_loader
        )
        
        print("\n✅ Benchmark completed successfully")
        
        return comparison_df
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            # Load shared annotation data (lightweight)
            self.load_shared_data()
            
            # Train models
            self.train_all_models()
            
            # Run benchmark
            self.run_benchmark()
            
            # Final summary
            self.print_summary()
            
            print("\n" + "="*80)
            print("\u2705\u2705\u2705 PIPELINE COMPLETED SUCCESSFULLY \u2705\u2705\u2705")
            print("="*80)
            print(f"\nRun ID:   {self.run_timestamp}")
            print(f"Results saved to:")
            print(f"  Models:  {self.config.OUTPUT_DIR / 'models'}")
            print(f"  Plots:   {self.config.OUTPUT_DIR / 'plots'}")
            print(f"  Logs:    {self.config.LOG_DIR}")
            print(f"  Console: {self.run_log_dir / 'pipeline.log'}")
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n\u274c Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Always close the log file cleanly, even on failure
            self._tee.stop()
    
    def print_summary(self):
        """Print training summary averaged across folds"""
        print("\n" + "="*80)
        print(f"TRAINING SUMMARY (Averaged over {self.config.K_FOLDS} Folds)")
        print("="*80)
        
        for model_name, metrics_list in self.training_results.items():
            if metrics_list:
                import numpy as np
                avg_time = np.mean([m.get('training_duration_minutes', 0) for m in metrics_list if m.get('training_duration_minutes')])
                avg_val_loss = np.mean([m.get('best_val_loss', 0) for m in metrics_list])
                avg_val_iou = np.mean([m.get('best_val_iou', 0) for m in metrics_list])
                avg_val_dice = np.mean([m.get('final_val_dice', 0) for m in metrics_list if m.get('final_val_dice')])
                model_params = metrics_list[0].get('model_params', 0)
                
                print(f"\n{model_name}:")
                print(f"  Avg Training Time: {avg_time:.1f} minutes / fold")
                print(f"  Avg Best Val Loss: {avg_val_loss:.4f}")
                print(f"  Avg Best Val mIoU: {avg_val_iou:.4f}")
                print(f"  Avg Final Val Dice:{avg_val_dice:.4f}")
                print(f"  Model Params:      {model_params:,}")
        
        print("\n" + "="*80)


def main():
    """Main entry point with global error catcher for crash-safe operation."""
    parser = argparse.ArgumentParser(
        description='Automated Training Pipeline for Thermal Face Detection'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['unet', 'transunet', 'swin'],
        default=None,
        help='Models to train (default: all models)'
    )
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Skip benchmarking after training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    args = parser.parse_args()

    # Update epoch count if specified
    if args.epochs != 100:
        # This will be used by Config class
        os.environ['NUM_EPOCHS'] = str(args.epochs)

    # Create and run pipeline
    pipeline = Pipeline(
        models_to_train=args.models,
        skip_benchmark=args.skip_benchmark
    )

    # ── Global error catcher ─────────────────────────────────────────────
    # Any unhandled exception (OOM, dimension mismatch, disk full, etc.) is
    # written to a dedicated error log file so the exact traceback is never
    # lost, even if stdout/stderr are redirected or the console window closes.
    try:
        success = pipeline.run()
    except Exception as e:  # noqa: BLE001
        tb_str = traceback.format_exc()

        # Prefer the run-specific log dir; fall back to the repo root.
        log_dir = Path(os.environ.get("UBENCH_LOG_DIR", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        error_log_path = log_dir / f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"UBench Pipeline Fatal Error\n")
            f.write(f"Occurred at: {datetime.now().isoformat()}\n")
            f.write(f"Exception:   {type(e).__name__}: {e}\n")
            f.write("=" * 70 + "\n\n")
            f.write(tb_str)

        # Echo to stderr so it is still visible in the console / log file.
        print(f"\n\u274c FATAL ERROR — full traceback written to: {error_log_path}",
              file=sys.stderr)
        print(tb_str, file=sys.stderr)

        # Re-raise so the process exits with a non-zero code, which lets any
        # orchestrator / AI agent detect the failure automatically.
        raise
    # ─────────────────────────────────────────────────────────────────────

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # freeze_support() is required on Windows when the pipeline is packaged
    # into a frozen executable (PyInstaller / cx_Freeze). It is a documented
    # no-op on Linux/Mac and when running from a normal Python interpreter,
    # so it is safe to always call it here.
    multiprocessing.freeze_support()
    main()
