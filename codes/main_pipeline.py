"""
Main Training Pipeline Orchestrator
===================================
Manages the complete training and benchmarking pipeline
"""

import sys
import os
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
from codes.unified_data import Config, create_data_loaders
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
        
        # Store training results
        self.training_results = {}
        self.models = {}
    
    def create_data_loaders(self):
        """Create data loaders with hardware-optimized settings"""
        print("\nStep 3: Data Loading")
        print("-"*80)
        
        # Use hardware-optimized workers
        num_workers = self.hardware_profile.num_workers
        
        # Create data loaders for each model
        self.data_loaders = {}
        
        shared_loader = None
        for model_name in ['unet', 'transunet', 'swin']:
            batch_size = self.hardware_profile.batch_sizes.get(model_name, 8)
            
            train_loader, val_loader, train_ids, val_ids, shared_loader = create_data_loaders(
                self.config,
                batch_size=batch_size,
                num_workers=num_workers,
                shared_data_loader=shared_loader
            )
            
            self.data_loaders[model_name] = {
                'train': train_loader,
                'val': val_loader,
                'train_ids': train_ids,
                'val_ids': val_ids,
                'batch_size': batch_size
            }
        
        print("✅ Data loaders created successfully\n")
    
    def train_unet(self):
        """Train U-Net model"""
        print("\n" + "="*80)
        print("TRAINING: U-Net")
        print("="*80)
        
        # Get data loaders
        loaders = self.data_loaders['unet']
        
        # Create model
        model = unet_v2.UNet(in_channels=1, num_classes=self.config.NUM_CLASSES)
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name="U-Net",
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        self.training_results['U-Net'] = metrics
        self.models['U-Net'] = model
        
        return metrics
    
    def train_transunet(self):
        """Train TransUNet model"""
        print("\n" + "="*80)
        print("TRAINING: TransUNet")
        print("="*80)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get data loaders
        loaders = self.data_loaders['transunet']
        
        # Create model
        model = transunet.TransUNet(
            img_size=self.config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=self.config.NUM_CLASSES
        )
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name="TransUNet",
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        self.training_results['TransUNet'] = metrics
        self.models['TransUNet'] = model
        
        return metrics
    
    def train_swin_unet(self):
        """Train Swin-UNet++ model"""
        print("\n" + "="*80)
        print("TRAINING: Swin-UNet++")
        print("="*80)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get data loaders
        loaders = self.data_loaders['swin']
        
        # Create model
        model = swin_unet_plus_plus.SwinUNetPlusPlus(
            img_size=self.config.IMAGE_SIZE[0],
            in_channels=1,
            num_classes=self.config.NUM_CLASSES
        )
        
        # Train
        trainer = UnifiedTrainer(
            model=model,
            model_name="Swin-UNet++",
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            config=self.config,
            learning_rate=self.config.LEARNING_RATE,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        trainer.train()
        trainer.plot_training_history()
        metrics = trainer.save_metrics()
        
        self.training_results['Swin-UNet++'] = metrics
        self.models['Swin-UNet++'] = model
        
        return metrics
    
    def train_all_models(self):
        """Train all selected models sequentially"""
        print("\nStep 4: Model Training")
        print("-"*80)
        
        start_time = datetime.now()
        
        # Train models based on selection
        if 'unet' in self.models_to_train:
            try:
                self.train_unet()
            except Exception as e:
                print(f"❌ U-Net training failed: {e}")
        
        if 'transunet' in self.models_to_train:
            try:
                self.train_transunet()
            except Exception as e:
                print(f"❌ TransUNet training failed: {e}")
        
        if 'swin' in self.models_to_train:
            try:
                self.train_swin_unet()
            except Exception as e:
                print(f"❌ Swin-UNet++ training failed: {e}")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print(f"✅ All training completed in {total_duration:.1f} minutes")
        print("="*80)
    
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
        
        # Use validation loader from any model (they're all the same data)
        val_loader = self.data_loaders[self.models_to_train[0]]['val']
        
        # Run benchmark
        comparison_df = benchmark_models.run_benchmark(
            models_dict, self.config, val_loader
        )
        
        print("\n✅ Benchmark completed successfully")
        
        return comparison_df
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            # Create data loaders
            self.create_data_loaders()
            
            # Train models
            self.train_all_models()
            
            # Run benchmark
            self.run_benchmark()
            
            # Final summary
            self.print_summary()
            
            print("\n" + "="*80)
            print("✅✅✅ PIPELINE COMPLETED SUCCESSFULLY ✅✅✅")
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
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Always close the log file cleanly, even on failure
            self._tee.stop()
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        for model_name, metrics in self.training_results.items():
            if metrics:
                print(f"\n{model_name}:")
                print(f"  Training Time:   {metrics.get('training_duration_minutes', 0):.1f} minutes")
                print(f"  Best Val Loss:   {metrics.get('best_val_loss', 0):.4f}")
                print(f"  Best Val mIoU:   {metrics.get('best_val_iou', 0):.4f}")
                print(f"  Final Val Dice:  {metrics.get('final_val_dice', 0):.4f}")
                print(f"  Model Params:    {metrics.get('model_params', 0):,}")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
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
    
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # freeze_support() is required on Windows when the pipeline is packaged
    # into a frozen executable (PyInstaller / cx_Freeze). It is a documented
    # no-op on Linux/Mac and when running from a normal Python interpreter,
    # so it is safe to always call it here.
    multiprocessing.freeze_support()
    main()
