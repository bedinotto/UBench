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
    Config, create_kfold_data_loaders, create_single_fold_loader,
    seed_everything, MultiDirectoryDataLoader, shutdown_data_loaders,
)
from codes.unified_training import UnifiedTrainer
from codes.preprocess_data import preprocess_all_data

# Import model architectures and registry
from codes.model_registry import create_model
# Import models to ensure they are registered
import codes.unet_v2
import codes.transunet
import codes.swin_unet_plus_plus
import codes.benchmark_models as benchmark_models


class Pipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, models_to_train=None, skip_benchmark=False,
                 force_preprocess: bool = False):
        """
        Initialize pipeline

        Args:
            models_to_train: List of model names to train.
                           None = all models. Options: ['unet', 'transunet', 'swin']
            skip_benchmark: If True, skip benchmarking after training
            force_preprocess: If True, rebuild data/processed even when
                           metadata.csv already exists
        """
        self.models_to_train = models_to_train or ['unet', 'transunet', 'swin']
        self.skip_benchmark = skip_benchmark
        self.force_preprocess = force_preprocess
        
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
    
    def ensure_preprocessed_data(self):
        """Run offline preprocessing when its outputs are missing (UB-01, T1.1).

        Training and benchmarking load ``data/processed/{images,masks}`` via
        ``metadata.csv``; a fresh clone has neither, so this stage creates
        them.  ``--force-preprocess`` rebuilds unconditionally.
        """
        print("\nStep 3: Offline Preprocessing")
        print("-"*80)
        metadata_path = self.config.PROCESSED_DIR / "metadata.csv"
        if metadata_path.exists() and not self.force_preprocess:
            print(f"✅ Preprocessed data found at {metadata_path} — skipping "
                  f"(use --force-preprocess to rebuild)")
            return
        if self.force_preprocess:
            print("--force-preprocess set — rebuilding preprocessed data")
        else:
            print(f"Preprocessed data not found at {metadata_path} — "
                  f"running offline preprocessing")
        preprocess_all_data()

    def load_shared_data(self):
        """Load annotations once (lightweight) — DataLoaders are created lazily."""
        print("\nStep 4: Data Loading (annotations)")
        print("-"*80)

        # Load annotations from        # (Shared data loader discovery removed as we use offline preprocessed arrays)
        self._shared_data_loader = None
        print(f"\u2705 Annotations loaded (DataLoaders will be created per-fold)\n")

    def _get_fold_loaders(self, model_name: str, fold_idx: int):
        """Create DataLoaders for a single model + fold (lazy, on-demand).

        Uses ``create_single_fold_loader`` so that only the requested
        fold's workers are spawned, preventing semaphore / shared-memory
        leaks from discarded loaders.
        """
        # Hard lookup (R4/UB-05): model_name must be a registry key with a
        # per-tier batch size; an unknown key raises instead of defaulting.
        batch_size = self.hardware_profile.batch_sizes[model_name]
        num_workers = self.hardware_profile.num_workers
        if 'NUM_WORKERS' in os.environ:
            num_workers = int(os.environ['NUM_WORKERS'])

        return create_single_fold_loader(
            self.config,
            fold_idx=fold_idx,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @staticmethod
    def _cleanup_fold_resources(*loaders):
        """Shut down DataLoader workers and free GPU memory between runs."""
        shutdown_data_loaders(*loaders)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_model(self, model_name: str, fold_idx: int):
        """Train a dynamic model from the registry on a specific fold"""
        print("\n" + "="*80)
        print(f"TRAINING: {model_name.upper()} (Fold {fold_idx + 1}/{self.config.K_FOLDS})")
        print("="*80)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create loaders lazily for this fold
        loaders = self._get_fold_loaders(model_name, fold_idx)
        
        # Create model dynamically
        kwargs = {
            'in_channels': 1,
            'num_classes': self.config.NUM_CLASSES
        }
        if model_name in ['transunet', 'swin_unet_plus_plus', 'swin']:
            kwargs['img_size'] = self.config.IMAGE_SIZE[0]
            
        model = create_model(model_name, **kwargs)
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name=f"{model_name}_Fold-{fold_idx + 1}",
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS,
            model_key=model_name,   # registry key (train_all_models resolves it)
            fold=fold_idx + 1,
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        if model_name not in self.training_results:
            self.training_results[model_name] = []
            
        self.training_results[model_name].append(metrics)
        
        # Cleanup
        del model, trainer
        self._cleanup_fold_resources(loaders['train_loader'], loaders['val_loader'])
        
        return metrics
    
    def train_all_models(self):
        """Train all selected models sequentially over all folds"""
        print("\nStep 5: Model Training (K-Folds)")
        print("-"*80)
        
        start_time = datetime.now()
        
        for fold_idx in range(self.config.K_FOLDS):
            print(f"\n>>>> STARTING FOLD {fold_idx + 1}/{self.config.K_FOLDS} <<<<")
            
            # Train models based on selection
            for model_name in self.models_to_train:
                if model_name == 'swin':
                    registry_name = 'swin_unet_plus_plus'
                else:
                    registry_name = model_name
                    
                try:
                    self.train_model(registry_name, fold_idx)
                except Exception as e:
                    print(f"❌ {model_name} training failed on Fold {fold_idx + 1}: {e}")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print(f"✅ All training completed in {total_duration:.1f} minutes")
        print("="*80)
    
    def run_benchmark(self):
        """Run comprehensive benchmark on all trained models, aggregating across all folds"""
        if self.skip_benchmark:
            print("\n⚠️  Skipping benchmark (--skip-benchmark flag set)")
            return
        
        print("\nStep 6: Comprehensive Benchmarking")
        print("-"*80)
        
        # Prepare models dictionary (display name → instance) and the
        # display→registry key map the benchmark needs for file I/O (UB-02/R5)
        models_dict = {}
        model_keys = {}
        for model_name in self.models_to_train:
            if model_name == 'swin':
                registry_name = 'swin_unet_plus_plus'
                display_name = 'Swin-UNet++'
            elif model_name == 'unet':
                registry_name = 'unet'
                display_name = 'U-Net'
            elif model_name == 'transunet':
                registry_name = 'transunet'
                display_name = 'TransUNet'
            else:
                registry_name = model_name
                display_name = model_name

            kwargs = {
                'in_channels': 1,
                'num_classes': self.config.NUM_CLASSES
            }
            if registry_name in ['transunet', 'swin_unet_plus_plus']:
                kwargs['img_size'] = self.config.IMAGE_SIZE[0]

            models_dict[display_name] = create_model(registry_name, **kwargs)
            model_keys[display_name] = registry_name
            
        # Get loaders for all models and folds (lazy creation)
        val_loaders_dict = {}
        for display_name in models_dict.keys():
            model_key = model_keys[display_name]
            loaders = []
            for fold_idx in range(self.config.K_FOLDS):
                loaders_data = self._get_fold_loaders(model_key, fold_idx)
                loaders.append(loaders_data['val_loader'])
            val_loaders_dict[display_name] = loaders
            
        print(f"Aggregating benchmark results across all {self.config.K_FOLDS} folds.")
        comparison_df = benchmark_models.run_benchmark(
            models_dict, self.config, val_loaders_dict=val_loaders_dict,
            model_keys=model_keys
        )
        
        print("\n✅ Benchmark completed successfully")
        
        return comparison_df
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            # Ensure offline-preprocessed arrays exist (UB-01)
            self.ensure_preprocessed_data()

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
    parser.add_argument(
        '--detect-anomaly',
        action='store_true',
        help='Enable PyTorch anomaly detection to debug NaN/Inf gradients'
    )
    parser.add_argument(
        '--force-preprocess',
        action='store_true',
        help='Rebuild data/processed even if metadata.csv already exists'
    )

    args = parser.parse_args()

    # Enable anomaly detection if requested
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("⚠️  WARNING: PyTorch anomaly detection is enabled. This will heavily degrade training performance.")

    # Update epoch count if specified
    if args.epochs != 100:
        # This will be used by Config class
        os.environ['NUM_EPOCHS'] = str(args.epochs)

    # Create and run pipeline
    pipeline = Pipeline(
        models_to_train=args.models,
        skip_benchmark=args.skip_benchmark,
        force_preprocess=args.force_preprocess
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
