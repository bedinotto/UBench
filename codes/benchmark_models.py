"""
Comprehensive Model Benchmarking Suite
======================================
Compare trained models on multiple metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    from codes.unified_data import Config, create_kfold_data_loaders, shutdown_data_loaders
    from codes.unified_training import _safe_filename, CombinedLoss
    from codes.naming import checkpoint_path
    from codes.metrics import SegmentationMetrics
except ImportError:
    from unified_data import Config, create_kfold_data_loaders, shutdown_data_loaders
    from unified_training import _safe_filename, CombinedLoss
    from naming import checkpoint_path
    from metrics import SegmentationMetrics


def timed_inference(model: nn.Module, val_loader, device, warmup: int = 5) -> Dict:
    """Measure per-image inference latency honestly (UB-09, M3).

    Two correctness properties the old in-loop timing lacked:

    * **Synchronization** — ``torch.cuda.synchronize()`` is called immediately
      before and after each timed forward pass **iff** ``device`` is CUDA.
      Without it, ``time.time()`` on GPU captures kernel *launch* time, not
      compute.  On CPU the calls are skipped (nothing to synchronize).
    * **Warm-up discard** — the first
      ``n_warmup = min(warmup, max(0, n_batches - 1))`` batches are run but not
      timed, so one-time cuDNN autotune / allocator cost does not pollute the
      mean.  The ``max(0, n_batches - 1)`` guard keeps at least one *measured*
      batch even for the tiny (<6-batch) validation loaders in the smoke suite.

    Only the forward pass is timed — the loss is not computed here (it belongs
    to the metrics pass), so latency reflects inference alone.

    Args:
        model: model to benchmark (moved to ``eval`` mode here).
        val_loader: iterable of ``(images, masks, _)`` batches with a ``len()``.
        device: ``torch.device`` or str; decides whether to synchronize.
        warmup: desired number of warm-up batches to discard.

    Returns:
        Dict with per-image latency and the measurement conditions (M9)::

            {"mean_inference_time_ms", "std_inference_time_ms", "fps",
             "n_warmup", "n_measured", "batch_size", "dtype", "device"}
    """
    device = torch.device(device)
    use_sync = device.type == "cuda"

    n_batches = len(val_loader)
    n_warmup = min(warmup, max(0, n_batches - 1))

    per_image_ms: List[float] = []
    batch_size = None
    dtype = None

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch[0]
            images = images.to(device, non_blocking=True)
            if batch_size is None:
                batch_size = int(images.size(0))
                dtype = str(getattr(images, "dtype", "unknown"))

            if i < n_warmup:
                model(images)          # warm-up: executed but not recorded
                continue

            if use_sync:
                torch.cuda.synchronize()
            start = time.time()
            model(images)
            if use_sync:
                torch.cuda.synchronize()
            per_image_ms.append((time.time() - start) * 1000.0 / images.size(0))

    n_measured = len(per_image_ms)
    mean_ms = float(np.mean(per_image_ms)) if n_measured else 0.0
    std_ms = float(np.std(per_image_ms)) if n_measured else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_inference_time_ms": mean_ms,
        "std_inference_time_ms": std_ms,
        "fps": fps,
        "n_warmup": n_warmup,
        "n_measured": n_measured,
        "batch_size": batch_size,
        "dtype": dtype,
        "device": str(device),
    }


# Fixed batch size for the VRAM probe — the SAME for every model so peak memory
# is comparable (UB-10, M3). Changing this changes the reported column label.
MEMORY_PROBE_BATCH_SIZE = 4

# The one memory column in the comparison table/report (M9).
_VRAM_COL = f"VRAM @ batch={MEMORY_PROBE_BATCH_SIZE} (fixed, inference)"


def probe_peak_memory(model: nn.Module, device, image_size,
                      in_channels: int = 1,
                      batch_size: int = MEMORY_PROBE_BATCH_SIZE) -> Optional[float]:
    """Peak GPU memory (MB) for one inference forward on a FIXED synthetic batch.

    Comparability fix (UB-10, M3): the old code read peak VRAM *during each
    model's evaluation pass*, at that model's hardware-selected batch size, so
    the figures were not comparable across models. This probe instead runs a
    single forward on a synthetic ``(batch_size, in_channels, H, W)`` tensor —
    the same shape for every model, independent of the loader or the model's
    training batch — so the number reflects the model, deterministically.

    Inference-mode and isolated: ``model.eval()`` + ``torch.no_grad()`` (the
    same mode as :func:`timed_inference`), ``reset_peak_memory_stats()``
    immediately before the single forward and ``max_memory_allocated()``
    immediately after. Run this as its own step so it neither perturbs nor is
    perturbed by the timing / metrics passes.

    Args:
        model: model to probe (put into eval mode here).
        device: ``torch.device`` or str.
        image_size: ``(H, W)`` for the synthetic input (e.g. ``config.IMAGE_SIZE``).
        in_channels: input channels (1 for thermal).
        batch_size: fixed probe batch size (shared across models).

    Returns:
        Peak allocated MB on CUDA, or ``None`` on CPU — there is nothing to
        measure without a GPU, and reporting 0 would fabricate a number (R10).
        Callers render ``None`` as "n/a (CPU)".
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None

    h, w = int(image_size[0]), int(image_size[1])
    probe_input = torch.zeros((batch_size, in_channels, h, w), device=device)

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(probe_input)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _format_vram(mb: Optional[float]) -> str:
    """Render a VRAM figure, or 'n/a (CPU)' when it was not measured."""
    return "n/a (CPU)" if mb is None else f"{mb:.2f} MB"


def evaluate_accuracy(model: nn.Module, loader: DataLoader,
                      num_classes: int, device) -> Dict:
    """Accuracy of one model on one loader via the shared authority (UB-11/R5).

    Returns the loss (CE+Dice, the training criterion) and hard macro mIoU/Dice
    — no timing, no VRAM. Used to score each fold-model on the held-out TEST set
    (M1); the same authority the CV pass uses, so CV and TEST numbers compare.
    """
    device = torch.device(device)
    model.eval()
    seg = SegmentationMetrics(num_classes, device=device)
    criterion = CombinedLoss()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            seg.update(outputs, masks)
            n_batches += 1
    m = seg.compute()
    return {
        "avg_loss": total_loss / n_batches if n_batches else 0.0,
        "mean_iou": m["mean_iou"],
        "mean_dice": m["mean_dice"],
    }


class ModelBenchmark:
    """Comprehensive benchmark for model comparison"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def load_model(self, model: nn.Module, model_name: str, 
                   model_path: Path) -> nn.Module:
        """Load trained model weights"""
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE,
                                             weights_only=True))
            model.to(self.config.DEVICE)
            model.eval()
            print(f"✅ Loaded {model_name} from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
    
    def benchmark_model(self, model: nn.Module, model_name: str,
                       val_loader: DataLoader) -> Dict:
        """
        Comprehensive benchmark of a single model
        
        Returns dictionary with all metrics
        """
        if model is None:
            return None
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        model.eval()
        
        # Metrics via the single shared authority (UB-11/R5): hard IoU + hard
        # Dice on argmax, macro over classes present in the target, background
        # reported separately — identical definitions to the trainer.
        seg_metrics = SegmentationMetrics(self.config.NUM_CLASSES, device=self.config.DEVICE)

        # Loss = the training criterion (CE + Dice) from the shared CombinedLoss
        # class, so "loss" means the same thing here as during training (was a
        # benchmark-only CrossEntropyLoss — UB-11).
        criterion = CombinedLoss()
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Testing {model_name}")
            for images, masks, _ in pbar:
                images = images.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)

                outputs = model(images)

                # Loss (CE + Dice — the training criterion)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                # Accumulate hard IoU / Dice (argmax taken inside)
                seg_metrics.update(outputs, masks)

        # Calculate statistics
        avg_loss = total_loss / len(val_loader)

        seg = seg_metrics.compute()
        mean_iou = seg["mean_iou"]
        mean_dice = seg["mean_dice"]
        # Dispersion across the present, non-background classes (a within-model
        # spread; cross-fold variance is computed separately in run_benchmark).
        present_fg = [c for c in seg["present_classes"] if c != 0]
        iou_fg = [seg["per_class_iou"][c] for c in present_fg]
        dice_fg = [seg["per_class_dice"][c] for c in present_fg]
        std_iou = float(np.std(iou_fg)) if len(iou_fg) > 1 else 0.0
        std_dice = float(np.std(dice_fg)) if len(dice_fg) > 1 else 0.0

        # Model size
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = model_params * 4 / (1024 ** 2)  # Assuming float32

        # ── VRAM: an isolated probe at a FIXED batch size shared by every model,
        # so peak memory is comparable (UB-10/M3).  Returns None on CPU.
        vram_probe_mb = probe_peak_memory(model, self.config.DEVICE,
                                          self.config.IMAGE_SIZE)

        # ── Latency: a separate warm-up-discarded, correctly-synced pass so the
        # measurement is polluted neither by first-batch cost nor by the
        # loss/metrics work above (UB-09/M3).
        timing = timed_inference(model, val_loader, self.config.DEVICE)
        mean_inference_time = timing["mean_inference_time_ms"]
        std_inference_time = timing["std_inference_time_ms"]
        fps = timing["fps"]

        # Per-class IoU statistics (from the shared authority; NaN for classes
        # absent from both prediction and target)
        class_iou_means = seg["per_class_iou"]
        class_iou_stds = [0.0] * self.config.NUM_CLASSES # Without batch-level info, we can't compute variance here easily
        
        results = {
            "model_name": model_name,
            "model_params": model_params,
            "model_size_mb": model_size_mb,
            "avg_loss": avg_loss,
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "mean_inference_time_ms": mean_inference_time,
            "std_inference_time_ms": std_inference_time,
            "fps": fps,
            # Timing conditions (M9) — attached so the report can disclose them.
            "timing_n_warmup": timing["n_warmup"],
            "timing_n_measured": timing["n_measured"],
            "timing_batch_size": timing["batch_size"],
            "timing_dtype": timing["dtype"],
            "timing_device": timing["device"],
            "vram_probe_mb": vram_probe_mb,
            "vram_probe_batch": MEMORY_PROBE_BATCH_SIZE,
            "class_iou_means": class_iou_means,
            "class_iou_stds": class_iou_stds,
        }

        # Print results
        print(f"\nResults:")
        print(f"  Model Params:    {model_params:,} ({model_size_mb:.2f} MB)")
        print(f"  mIoU (hard,macro):  {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"  Dice (hard,macro):  {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"  Inference Time:  {mean_inference_time:.2f} ± {std_inference_time:.2f} ms/image "
              f"(warm-up={timing['n_warmup']}, measured={timing['n_measured']} batches, "
              f"batch={timing['batch_size']}, {timing['dtype']}, {timing['device']})")
        print(f"  FPS:             {fps:.2f}")
        print(f"  Loss (CE+Dice):  {avg_loss:.4f}")
        print(f"  VRAM @ batch={MEMORY_PROBE_BATCH_SIZE} (fixed): {_format_vram(vram_probe_mb)}")
        print(f"{'='*70}")
        
        return results
    
    def compare_models(self, results_dict: Dict[str, Dict]):
        """Generate comparison visualizations and reports"""
        if not results_dict:
            print("No results to compare")
            return
        
        # Create comparison dataframe. "mIoU"/"Dice Score" are the CV headline;
        # TEST columns are added only when a held-out test set was scored (M1),
        # so a CV-only run is unchanged (backward-compatible).
        comparison_data = []
        for model_name, results in results_dict.items():
            if results:
                row = {
                    "Model": model_name,
                    "mIoU": results["mean_iou"],
                    "Dice Score": results["mean_dice"],
                    "Inference (ms)": results["mean_inference_time_ms"],
                    "FPS": results["fps"],
                    "Params (M)": results["model_params"] / 1e6,
                    "Size (MB)": results["model_size_mb"],
                    # One comparable memory column (fixed-batch probe). None (CPU)
                    # becomes NaN so the column stays numeric for the CSV/plots.
                    _VRAM_COL: results["vram_probe_mb"]
                    if results["vram_probe_mb"] is not None else float("nan"),
                }
                if results.get("test_mean_iou") is not None:
                    row["mIoU (TEST)"] = results["test_mean_iou"]
                    row["Dice (TEST)"] = results["test_mean_dice"]
                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        
        if df.empty:
            print("\n⚠️  No models produced valid results — skipping comparison.")
            return df
        
        # Save comparison table
        output_path = self.config.OUTPUT_DIR / "benchmark_comparison.csv"
        df.to_csv(str(output_path), index=False)
        print(f"\n✅ Benchmark comparison saved to: {output_path}")
        
        # Print comparison table
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Create visualizations
        self._create_comparison_plots(df, results_dict)
        
        # Generate detailed report
        self._generate_report(df, results_dict)
        
        return df
    
    def _create_comparison_plots(self, df: pd.DataFrame, results_dict: Dict):
        """Create comprehensive comparison plots"""
        output_dir = self.config.OUTPUT_DIR / "plots"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Accuracy Metrics Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # IoU comparison
        axes[0].bar(df["Model"], df["mIoU"], color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0].set_ylabel('Mean IoU', fontsize=12)
        axes[0].set_title('Model Accuracy: Mean IoU', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1])
        for i, v in enumerate(df["mIoU"]):
            axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
        
        # Dice comparison
        axes[1].bar(df["Model"], df["Dice Score"], color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[1].set_ylabel('Dice Score', fontsize=12)
        axes[1].set_title('Model Accuracy: Dice Score', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        for i, v in enumerate(df["Dice Score"]):
            axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "accuracy_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Speed Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inference time
        axes[0].bar(df["Model"], df["Inference (ms)"], color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0].set_ylabel('Inference Time (ms)', fontsize=12)
        axes[0].set_title('Inference Speed: Time per Image', fontsize=14, fontweight='bold')
        for i, v in enumerate(df["Inference (ms)"]):
            axes[0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=10)
        
        # FPS
        axes[1].bar(df["Model"], df["FPS"], color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[1].set_ylabel('FPS', fontsize=12)
        axes[1].set_title('Throughput: Frames Per Second', fontsize=14, fontweight='bold')
        for i, v in enumerate(df["FPS"]):
            axes[1].text(i, v + 1, f'{v:.2f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "speed_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Model Complexity
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Parameters
        axes[0].bar(df["Model"], df["Params (M)"], color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
        axes[0].set_title('Model Complexity: Parameters', fontsize=14, fontweight='bold')
        for i, v in enumerate(df["Params (M)"]):
            axes[0].text(i, v + 1, f'{v:.1f}M', ha='center', fontsize=10)
        
        # GPU Memory (fixed-batch probe; blank with an n/a note on CPU runs,
        # where every value is NaN and bar() cannot plot).
        axes[1].set_ylabel('GPU Memory (MB)', fontsize=12)
        axes[1].set_title(f'Memory Usage: {_VRAM_COL}', fontsize=13, fontweight='bold')
        if df[_VRAM_COL].notna().any():
            axes[1].bar(df["Model"], df[_VRAM_COL], color=['#3498db', '#e74c3c', '#2ecc71'])
            for i, v in enumerate(df[_VRAM_COL]):
                if pd.notna(v):
                    axes[1].text(i, v + 5, f'{v:.0f}', ha='center', fontsize=10)
        else:
            axes[1].text(0.5, 0.5, 'n/a (CPU)', ha='center', va='center',
                         transform=axes[1].transAxes, fontsize=12, color='gray')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "complexity_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Per-class IoU heatmap
        class_ious = []
        model_names = []
        for model_name, results in results_dict.items():
            if results:
                class_ious.append(results["class_iou_means"])
                model_names.append(model_name)
        
        if class_ious:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(np.array(class_ious), annot=True, fmt='.3f', 
                       xticklabels=[name[:15] for name in self.config.REGION_NAMES],
                       yticklabels=model_names, cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': 'IoU'})
            ax.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(str(output_dir / "per_class_iou_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Comparison plots saved to: {output_dir}")
    
    def _generate_report(self, df: pd.DataFrame, results_dict: Dict):
        """Generate detailed text report"""
        # Save report to LOG_DIR (alongside other per-run log files)
        report_path = self.config.LOG_DIR / "benchmark_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MODEL BENCHMARK REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("SUMMARY TABLE\n")
            f.write("-"*80 + "\n")
            f.write(df.to_string(index=False) + "\n\n")
            
            # Best model analysis
            f.write("BEST MODELS BY METRIC\n")
            f.write("-"*80 + "\n")
            f.write(f"Highest mIoU:     {df.loc[df['mIoU'].idxmax(), 'Model']} "
                   f"({df['mIoU'].max():.4f})\n")
            f.write(f"Highest Dice:     {df.loc[df['Dice Score'].idxmax(), 'Model']} "
                   f"({df['Dice Score'].max():.4f})\n")
            f.write(f"Fastest:          {df.loc[df['Inference (ms)'].idxmin(), 'Model']} "
                   f"({df['Inference (ms)'].min():.2f} ms)\n")
            f.write(f"Highest FPS:      {df.loc[df['FPS'].idxmax(), 'Model']} "
                   f"({df['FPS'].max():.2f} FPS)\n")
            f.write(f"Smallest:         {df.loc[df['Params (M)'].idxmin(), 'Model']} "
                   f"({df['Params (M)'].min():.1f}M params)\n")
            if df[_VRAM_COL].notna().any():
                f.write(f"Lowest VRAM:      {df.loc[df[_VRAM_COL].idxmin(), 'Model']} "
                       f"({df[_VRAM_COL].min():.0f} MB, {_VRAM_COL})\n\n")
            else:
                f.write(f"Lowest VRAM:      n/a (CPU — {_VRAM_COL} needs a GPU)\n\n")
            
            # Detailed per-model analysis
            f.write("DETAILED PER-MODEL ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for model_name, results in results_dict.items():
                if results:
                    f.write(f"{model_name}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Architecture Complexity:\n")
                    f.write(f"  Parameters:       {results['model_params']:,}\n")
                    f.write(f"  Model Size:       {results['model_size_mb']:.2f} MB\n")
                    f.write(f"  VRAM @ batch={results.get('vram_probe_batch', MEMORY_PROBE_BATCH_SIZE)} "
                           f"(fixed, inference): {_format_vram(results.get('vram_probe_mb'))}\n\n")
                    
                    f.write(f"Accuracy — CROSS-VALIDATION (hard, argmax, macro excl. absent; ± cross-fold — UB-11/M2):\n")
                    f.write(f"  mIoU (macro):     {results['mean_iou']:.4f} ± {results['std_iou']:.4f}\n")
                    f.write(f"  Dice (macro):     {results['mean_dice']:.4f} ± {results['std_dice']:.4f}\n")
                    f.write(f"  Loss (CE+Dice):   {results['avg_loss']:.4f}\n\n")

                    if results.get('test_mean_iou') is not None:
                        f.write(f"Accuracy — HELD-OUT TEST SUBJECTS "
                               f"({results.get('test_n_folds')} fold-models scored; ± cross-fold — M1):\n")
                        f.write(f"  mIoU (macro):     {results['test_mean_iou']:.4f} ± {results['test_std_iou']:.4f}\n")
                        f.write(f"  Dice (macro):     {results['test_mean_dice']:.4f} ± {results['test_std_dice']:.4f}\n")
                        f.write(f"  Loss (CE+Dice):   {results['test_avg_loss']:.4f}\n\n")

                    f.write(f"Speed Metrics (warm-up discarded, synced on CUDA — UB-09/M3):\n")
                    f.write(f"  Inference Time:   {results['mean_inference_time_ms']:.2f} ± "
                           f"{results['std_inference_time_ms']:.2f} ms/image\n")
                    f.write(f"  Throughput (FPS): {results['fps']:.2f}\n")
                    f.write(f"  Conditions:       warm-up={results.get('timing_n_warmup')}, "
                           f"measured={results.get('timing_n_measured')} batches, "
                           f"batch_size={results.get('timing_batch_size')}, "
                           f"dtype={results.get('timing_dtype')}, "
                           f"device={results.get('timing_device')}, "
                           f"torch={torch.__version__}\n\n")
                    
                    f.write(f"Per-Class IoU:\n")
                    for idx, (region_name, iou_mean, iou_std) in enumerate(
                        zip(self.config.REGION_NAMES, 
                            results['class_iou_means'],
                            results['class_iou_stds'])):
                        f.write(f"  {region_name:30s}: {iou_mean:.4f} ± {iou_std:.4f}\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            best_accuracy = df.loc[df['mIoU'].idxmax(), 'Model']
            best_speed = df.loc[df['Inference (ms)'].idxmin(), 'Model']
            
            f.write(f"For Highest Accuracy:     Use {best_accuracy}\n")
            f.write(f"For Real-time Processing: Use {best_speed}\n")
            
            # Efficiency analysis
            df['Efficiency'] = df['mIoU'] / (df['Inference (ms)'] / 100)
            best_efficiency = df.loc[df['Efficiency'].idxmax(), 'Model']
            f.write(f"For Best Efficiency:      Use {best_efficiency}\n")
            
            f.write("="*80 + "\n")
        
        print(f"✅ Detailed report saved to: {report_path}")


def run_benchmark(models_dict: Dict[str, nn.Module], config: Config,
                 val_loaders_dict: Dict[str, List[DataLoader]] = None,
                 val_loader: DataLoader = None, *,
                 model_keys: Dict[str, str],
                 test_loader: DataLoader = None) -> pd.DataFrame:
    """
    Run complete benchmark suite, aggregating metrics across all folds if multiple loaders are provided.

    Args:
        models_dict: Dictionary of {display_name: model_instance}
        config: Configuration object
        val_loaders_dict: Dict mapping display_name to list of DataLoaders (one per fold)
        val_loader: Fallback validation data loader (used if val_loaders_dict is not provided)
        model_keys: Dict mapping display_name to registry key — checkpoint
            paths are derived from registry keys only (UB-02/R5); display
            names stay in logs, plots, and the CSV's Model column
        test_loader: Optional held-out TEST-subject loader (M1). When provided,
            every fold-model is *also* scored on it and the results are reported
            as a TEST section (mean ± std across folds), parallel to CV. No model
            is selected on the test set — it is held out, so there is nothing to
            select on; this is the most honest default (M9).

    Returns:
        Comparison dataframe
    """
    benchmark = ModelBenchmark(config)
    results_dict = {}

    for model_name, model in models_dict.items():
        fold_results = []
        test_fold_results = []   # each fold-model scored on the held-out TEST set (M1)
        num_folds = config.K_FOLDS
        model_key = model_keys[model_name]  # hard lookup — a typo must raise (R4)

        # Determine loaders for this model
        loaders = []
        if val_loaders_dict and model_name in val_loaders_dict:
            loaders = val_loaders_dict[model_name]

        # If no fold loaders are provided, evaluate fold 1 using val_loader
        if not loaders:
            model_path = checkpoint_path(config.OUTPUT_DIR, model_key, 1, "best")
            if not model_path.exists():
                print(f"⚠️  Model weights not found for {model_name} at {model_path}")
                results_dict[model_name] = None
                continue

            loaded_model = benchmark.load_model(model, model_name, model_path)
            if loaded_model is not None and val_loader is not None:
                results = benchmark.benchmark_model(loaded_model, model_name, val_loader)
                results_dict[model_name] = results
            else:
                results_dict[model_name] = None
            continue

        # Iterate over all folds
        for fold_idx in range(num_folds):
            model_path = checkpoint_path(config.OUTPUT_DIR, model_key,
                                         fold_idx + 1, "best")
            if not model_path.exists():
                print(f"⚠️  Model weights not found for {model_name} Fold {fold_idx + 1} at {model_path}")
                continue
                
            loaded_model = benchmark.load_model(model, f"{model_name} (Fold {fold_idx + 1})", model_path)
            if loaded_model is None:
                continue
                
            fold_loader = loaders[fold_idx]
            results = benchmark.benchmark_model(loaded_model, f"{model_name}_Fold_{fold_idx + 1}", fold_loader)
            if results:
                fold_results.append(results)

            # Also score this fold-model on the held-out TEST subjects (M1) —
            # same held-out set for every fold, evaluated, never selected on.
            if test_loader is not None:
                test_fold_results.append(
                    evaluate_accuracy(loaded_model, test_loader,
                                      config.NUM_CLASSES, config.DEVICE)
                )

            # Clean up fold loader resources and GPU memory after this evaluation
            shutdown_data_loaders(fold_loader)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if not fold_results:
            print(f"⚠️  No valid fold results for model {model_name}")
            results_dict[model_name] = None
            continue
            
        # Aggregate results across folds
        avg_loss = np.mean([r["avg_loss"] for r in fold_results])
        mean_iou = np.mean([r["mean_iou"] for r in fold_results])
        std_iou = np.std([r["mean_iou"] for r in fold_results])  # cross-fold variance of mIoU
        mean_dice = np.mean([r["mean_dice"] for r in fold_results])
        std_dice = np.std([r["mean_dice"] for r in fold_results])  # cross-fold variance of Dice
        mean_inference_time = np.mean([r["mean_inference_time_ms"] for r in fold_results])
        std_inference_time = np.std([r["mean_inference_time_ms"] for r in fold_results])
        fps = 1000.0 / mean_inference_time if mean_inference_time > 0 else 0.0
        # VRAM probe is model-deterministic (fixed batch, same device) — take a
        # representative fold rather than averaging (avoids np.mean over None on CPU).
        vram_probe_mb = fold_results[0].get("vram_probe_mb")

        # Aggregate per-class IoU means and stds across folds
        class_iou_means = []
        class_iou_stds = []
        for cls_idx in range(config.NUM_CLASSES):
            cls_vals = [r["class_iou_means"][cls_idx] for r in fold_results]
            class_iou_means.append(float(np.mean(cls_vals)))
            class_iou_stds.append(float(np.std(cls_vals)))
            
        aggregated_results = {
            "model_name": model_name,
            "model_params": fold_results[0]["model_params"],
            "model_size_mb": fold_results[0]["model_size_mb"],
            "avg_loss": float(avg_loss),
            "mean_iou": float(mean_iou),
            "std_iou": float(std_iou),
            "mean_dice": float(mean_dice),
            "std_dice": float(std_dice),
            "mean_inference_time_ms": float(mean_inference_time),
            "std_inference_time_ms": float(std_inference_time),
            "fps": float(fps),
            # Timing conditions are constant across folds (same device/dtype and
            # warm-up policy) — carry a representative fold's values (M9).
            "timing_n_warmup": fold_results[0].get("timing_n_warmup"),
            "timing_n_measured": fold_results[0].get("timing_n_measured"),
            "timing_batch_size": fold_results[0].get("timing_batch_size"),
            "timing_dtype": fold_results[0].get("timing_dtype"),
            "timing_device": fold_results[0].get("timing_device"),
            "vram_probe_mb": vram_probe_mb,
            "vram_probe_batch": fold_results[0].get("vram_probe_batch"),
            "class_iou_means": class_iou_means,
            "class_iou_stds": class_iou_stds,
        }

        # Held-out TEST metrics: each fold-model scored on the same test set,
        # aggregated mean ± std across folds (M1). Absent when no test subjects.
        if test_fold_results:
            aggregated_results.update({
                "test_mean_iou": float(np.mean([r["mean_iou"] for r in test_fold_results])),
                "test_std_iou": float(np.std([r["mean_iou"] for r in test_fold_results])),
                "test_mean_dice": float(np.mean([r["mean_dice"] for r in test_fold_results])),
                "test_std_dice": float(np.std([r["mean_dice"] for r in test_fold_results])),
                "test_avg_loss": float(np.mean([r["avg_loss"] for r in test_fold_results])),
                "test_n_folds": len(test_fold_results),
            })

        results_dict[model_name] = aggregated_results
        
        # Print aggregated results
        print(f"\n==============================================================")
        print(f"Aggregated Cross-Fold Results for {model_name} ({len(fold_results)}/{num_folds} Folds):")
        print(f"==============================================================")
        print(f"  Model Params:    {aggregated_results['model_params']:,} ({aggregated_results['model_size_mb']:.2f} MB)")
        print(f"  CV   mIoU (hard,macro):  {aggregated_results['mean_iou']:.4f} ± {aggregated_results['std_iou']:.4f}  (± cross-fold)")
        print(f"  CV   Dice (hard,macro):  {aggregated_results['mean_dice']:.4f} ± {aggregated_results['std_dice']:.4f}  (± cross-fold)")
        if aggregated_results.get('test_mean_iou') is not None:
            print(f"  TEST mIoU (hard,macro):  {aggregated_results['test_mean_iou']:.4f} ± {aggregated_results['test_std_iou']:.4f}  (held-out, ± cross-fold)")
            print(f"  TEST Dice (hard,macro):  {aggregated_results['test_mean_dice']:.4f} ± {aggregated_results['test_std_dice']:.4f}  (held-out, ± cross-fold)")
        print(f"  Inference Time:  {aggregated_results['mean_inference_time_ms']:.2f} ± {aggregated_results['std_inference_time_ms']:.2f} ms")
        print(f"  FPS:             {aggregated_results['fps']:.2f}")
        print(f"  Loss (CE+Dice):  {aggregated_results['avg_loss']:.4f}")
        print(f"  VRAM @ batch={aggregated_results.get('vram_probe_batch', MEMORY_PROBE_BATCH_SIZE)} (fixed): "
              f"{_format_vram(aggregated_results.get('vram_probe_mb'))}")
        print(f"==============================================================\n")
        
        # Save individual aggregated results to LOG_DIR
        log_dir = config.LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        results_path = log_dir / f"{_safe_filename(model_name)}_benchmark.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_results, f, indent=2)
            
    # Generate comparison
    comparison_df = benchmark.compare_models(results_dict)
    
    return comparison_df


if __name__ == "__main__":
    print("Comprehensive Benchmark Module - Ready")
    print("Use run_benchmark() to compare trained models")
