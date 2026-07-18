"""
Main Training Pipeline Orchestrator
===================================
Manages the complete training and benchmarking pipeline
"""

import sys
import os
import gc
import json
import subprocess
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
    create_test_loader, seed_everything, MultiDirectoryDataLoader,
    shutdown_data_loaders, load_split_metadata, resolve_fold_count,
)
from codes.unified_training import UnifiedTrainer
from codes.preprocess_data import preprocess_all_data

# Import model architectures and registry
from codes.model_registry import create_model
# Import models to ensure they are registered
import codes.unet_v2
import codes.transunet
import codes.swin_unet_plus_plus
import codes.swin_pretrained          # T3.2: pretrained SwinV2 encoder (M5)
import codes.transunet_pretrained     # T3.2: pretrained R50+ViT-B/16 hybrid (M5)
import codes.benchmark_models as benchmark_models


def _git(args: list[str], default: str = "unknown") -> str:
    """Best-effort ``git`` invocation for run provenance; never raises."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True,
            cwd=str(Path(__file__).parent), timeout=5,
        )
        return (out.stdout or "").strip() or default
    except Exception:
        return default


def collect_run_metadata(config, run_id: str, models_to_train: list[str]) -> dict:
    """Collect per-run provenance (M6): git state, env, seed, and effective config.

    Pure function (no I/O beyond a best-effort ``git`` read) so it is unit
    testable. ``Pipeline`` writes the result to ``logs/<run_id>/run_metadata.json``.
    """
    cuda = torch.cuda.is_available()
    return {
        "run_id": run_id,
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"], default="")),
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "random_seed": config.RANDOM_SEED,
        "deterministic": config.DETERMINISTIC,
        "models_to_train": list(models_to_train),
        "num_epochs": config.NUM_EPOCHS,
        "k_folds": config.K_FOLDS,
        "test_subjects": list(config.TEST_SUBJECTS),
        "image_size": list(config.IMAGE_SIZE),
        "num_classes": config.NUM_CLASSES,
        # TODO(T3.5): record the resolved lockfile hash once lockfiles land (UB-20b).
        "lockfile_hash": None,
    }


def apply_epochs_override(epochs: int | None) -> None:
    """Export ``--epochs`` to the ``NUM_EPOCHS`` env var Config reads (UB-13).

    Exported whenever the flag was given — including exactly 100, the value
    the old ``if args.epochs != 100`` guard silently dropped.  ``None``
    (flag absent) exports nothing, leaving ``codes/config.yaml`` (or a
    pre-set ``NUM_EPOCHS``) as the authority.
    """
    if epochs is not None:
        os.environ['NUM_EPOCHS'] = str(epochs)


def write_error_log(text: str) -> Path:
    """Persist *text* to a timestamped error log and return its path.

    Single authority for error-log placement: the run's log dir when the
    pipeline got far enough to export ``UBENCH_LOG_DIR``, the current
    directory otherwise.
    """
    log_dir = Path(os.environ.get("UBENCH_LOG_DIR", "."))
    log_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = log_dir / f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write(text)
    return error_log_path


class Pipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, models_to_train=None, skip_benchmark=False,
                 force_preprocess: bool = False, fail_fast: bool = False,
                 resume: str | None = None):
        """
        Initialize pipeline

        Args:
            models_to_train: List of model names to train.
                           None = all models. Options: ['unet', 'transunet', 'swin']
            skip_benchmark: If True, skip benchmarking after training
            force_preprocess: If True, rebuild data/processed even when
                           metadata.csv already exists
            fail_fast: If True, abort the run on the first model/fold
                           training failure instead of continuing
            resume: Run ID of a previous run whose ``outputs/<run_id>`` and
                           ``logs/<run_id>`` directories are reused so the
                           trainers pick up their epoch checkpoints (UB-06).
                           None = start a fresh timestamped run.
        """
        self.models_to_train = models_to_train or ['unet', 'transunet', 'swin']
        self.skip_benchmark = skip_benchmark
        self.force_preprocess = force_preprocess
        self.fail_fast = fail_fast

        print("\n" + "="*80)
        print("THERMAL FACE DETECTION - AUTOMATED TRAINING PIPELINE")
        print("="*80 + "\n")

        # Resolve the run identity (precedence: --resume > UBENCH_RUN_ID >
        # fresh timestamp): --resume reuses an existing run's dirs so
        # checkpoint discovery finds prior epochs (UB-06); UBENCH_RUN_ID is
        # exported by run.sh/run.bat so shell and Python share one run id
        # and one logs/<ts> dir per invocation (UB-14).
        if resume is not None:
            resume_dir = Path("outputs") / resume
            if not resume_dir.is_dir():
                available = sorted(
                    p.name for p in Path("outputs").glob("*")
                    if p.is_dir() and not p.is_symlink()
                )
                raise FileNotFoundError(
                    f"--resume {resume!r}: run directory {resume_dir} does not "
                    f"exist. Available run IDs under outputs/: {available or 'none'}"
                )
            self.run_timestamp = resume
            print(f"♻️  Resuming previous run: {resume}")
        elif os.environ.get("UBENCH_RUN_ID"):
            self.run_timestamp = os.environ["UBENCH_RUN_ID"]
        else:
            self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_output_dir = Path("outputs") / self.run_timestamp
        self.run_log_dir = Path("logs") / self.run_timestamp
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        self._update_latest_symlink()

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
        
        # Set global seed + own the cuDNN determinism decision (M6)
        seed_everything(self.config.RANDOM_SEED, deterministic=self.config.DETERMINISTIC)
        print(f"✅ Global seeds fixed to {self.config.RANDOM_SEED} "
              f"(deterministic={self.config.DETERMINISTIC})")

        # Per-run provenance (M6): git SHA/dirty, torch/CUDA, GPU, seed, config
        metadata = collect_run_metadata(
            self.config, self.run_timestamp, self.models_to_train
        )
        metadata_path = self.run_log_dir / "run_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Run metadata written to: {metadata_path}")
        
        # Store training results (metrics only — models are saved to disk)
        self.training_results = {}

        # Failure registry (UB-07): every caught per-model/fold exception is
        # recorded here; the SUCCESS banner and exit code depend on it.
        self.failures: list[dict] = []

    def _update_latest_symlink(self) -> None:
        """Point ``outputs/latest`` at this run's output dir (UB-06).

        The replacement is atomic (create-then-rename) so a concurrent
        reader never sees a missing link.  Platforms or filesystems without
        symlink support (e.g. Windows without developer mode) get a warning,
        not a crash — the link is a convenience, not a load-bearing path.
        """
        latest = self.run_output_dir.parent / "latest"
        tmp = self.run_output_dir.parent / ".latest.tmp"
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            # Relative target: the link stays valid if outputs/ is moved.
            tmp.symlink_to(self.run_output_dir.name, target_is_directory=True)
            tmp.replace(latest)
        except OSError as exc:
            print(f"⚠️  Could not update {latest} symlink: {exc}")

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

        # Test-only failure injection (UB-07 tests): inert unless the env
        # var names this exact model key.
        if os.environ.get("UBENCH_INJECT_FAIL") == model_name:
            raise RuntimeError(
                f"Injected failure for {model_name} (UBENCH_INJECT_FAIL)"
            )
        
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
        if model_name in ['transunet', 'swin_unet_plus_plus', 'swin',
                           'swin_pretrained', 'transunet_pretrained']:
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
                    self.failures.append({
                        'model': registry_name,
                        'fold': fold_idx + 1,
                        'error': repr(e),
                        'traceback': traceback.format_exc(),
                    })
                    print(f"❌ {model_name} training failed on Fold {fold_idx + 1}: {e}")
                    if self.fail_fast:
                        print("--fail-fast set — aborting on first failure")
                        raise

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60

        if self.failures:
            print(f"\n⚠️  Training loop finished in {total_duration:.1f} minutes "
                  f"with {len(self.failures)} failure(s)")
        else:
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
            elif model_name == 'swin_pretrained':
                registry_name = 'swin_pretrained'
                display_name = 'SwinV2-UNet (ImageNet)'
            elif model_name == 'transunet_pretrained':
                registry_name = 'transunet_pretrained'
                display_name = 'TransUNet (ImageNet)'
            else:
                registry_name = model_name
                display_name = model_name

            kwargs = {
                'in_channels': 1,
                'num_classes': self.config.NUM_CLASSES
            }
            if registry_name in ['transunet', 'swin_unet_plus_plus',
                                  'swin_pretrained', 'transunet_pretrained']:
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

        # Held-out TEST loader (M1): one shared set, scored per fold-model.
        # Batch size is irrelevant to accuracy (confusion-matrix metrics are
        # batch-invariant); reuse a representative model's batch size.
        num_workers = self.hardware_profile.num_workers
        if 'NUM_WORKERS' in os.environ:
            num_workers = int(os.environ['NUM_WORKERS'])
        probe_key = model_keys[next(iter(models_dict))]
        test_loader = create_test_loader(
            self.config, self.hardware_profile.batch_sizes[probe_key], num_workers
        )
        if test_loader is not None:
            print(f"Held-out TEST subjects {self.config.TEST_SUBJECTS}: "
                  f"{len(test_loader.dataset)} samples scored per fold-model.")

        print(f"Aggregating benchmark results across all {self.config.K_FOLDS} folds.")
        comparison_df = benchmark_models.run_benchmark(
            models_dict, self.config, val_loaders_dict=val_loaders_dict,
            model_keys=model_keys, test_loader=test_loader
        )
        if test_loader is not None:
            shutdown_data_loaders(test_loader)
        
        print("\n✅ Benchmark completed successfully")
        
        return comparison_df
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            # Ensure offline-preprocessed arrays exist (UB-01)
            self.ensure_preprocessed_data()

            # Resolve the effective leave-subjects-out fold count once
            # (UB-03): the training loop, benchmark loop and the loaders'
            # GroupKFold must all agree on the same K.
            metadata_df = load_split_metadata(self.config)
            self.config.K_FOLDS = resolve_fold_count(
                self.config.K_FOLDS, metadata_df['dataset']
            )

            # Train models
            self.train_all_models()
            
            # Run benchmark
            self.run_benchmark()

            # Final summary
            self.print_summary()

            # Honest exit (UB-07): the SUCCESS banner is earned only when
            # every model \u00d7 fold trained; otherwise summarize and fail.
            if self.failures:
                self._report_failures()
                return False

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
            traceback.print_exc()
            if self.failures:
                self._report_failures()
            return False
        finally:
            # Always close the log file cleanly, even on failure
            self._tee.stop()
    
    def _report_failures(self):
        """Print the end-of-run failure summary and persist tracebacks (UB-07)."""
        print("\n" + "="*80)
        print(f"❌ PIPELINE FINISHED WITH {len(self.failures)} TRAINING FAILURE(S)")
        print("="*80)
        sections = []
        for failure in self.failures:
            print(f"  ❌ {failure['model']} fold {failure['fold']}: {failure['error']}")
            sections.append(
                f"{failure['model']} fold {failure['fold']}: {failure['error']}\n"
                f"{failure['traceback']}"
            )
        error_log_path = write_error_log(
            "UBench Pipeline Training Failures\n"
            f"Occurred at: {datetime.now().isoformat()}\n"
            + "=" * 70 + "\n\n"
            + ("\n" + "=" * 70 + "\n\n").join(sections)
        )
        print(f"Full tracebacks written to: {error_log_path}")
        print("="*80)

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
        default=None,
        help='Number of training epochs (default: value from codes/config.yaml)'
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
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Abort the run on the first model/fold training failure'
    )
    parser.add_argument(
        '--resume',
        metavar='RUN_ID',
        default=None,
        help='Reuse outputs/<RUN_ID> and logs/<RUN_ID> from a previous run '
             'and continue training from its epoch checkpoints'
    )

    args = parser.parse_args()

    # Enable anomaly detection if requested
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("⚠️  WARNING: PyTorch anomaly detection is enabled. This will heavily degrade training performance.")

    # Export --epochs to the env var Config reads, whenever given (UB-13)
    apply_epochs_override(args.epochs)

    # ── Global error catcher ─────────────────────────────────────────────
    # Any unhandled exception — including a Pipeline constructor crash
    # (missing data dir, hardware detection) — is written to a dedicated
    # error log file so the exact traceback is never lost, even if
    # stdout/stderr are redirected or the console window closes (UB-07).
    try:
        pipeline = Pipeline(
            models_to_train=args.models,
            skip_benchmark=args.skip_benchmark,
            force_preprocess=args.force_preprocess,
            fail_fast=args.fail_fast,
            resume=args.resume
        )
        success = pipeline.run()
    except Exception as e:  # noqa: BLE001
        tb_str = traceback.format_exc()

        error_log_path = write_error_log(
            "UBench Pipeline Fatal Error\n"
            f"Occurred at: {datetime.now().isoformat()}\n"
            f"Exception:   {type(e).__name__}: {e}\n"
            + "=" * 70 + "\n\n"
            + tb_str
        )

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
