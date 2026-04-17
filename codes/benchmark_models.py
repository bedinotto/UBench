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
from typing import Dict, List, Tuple
try:
    from codes.unified_data import Config, create_data_loaders
    from codes.unified_training import calculate_iou, calculate_dice_score, _safe_filename
except ImportError:
    from unified_data import Config, create_data_loaders
    from unified_training import calculate_iou, calculate_dice_score, _safe_filename


class ModelBenchmark:
    """Comprehensive benchmark for model comparison"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def load_model(self, model: nn.Module, model_name: str, 
                   model_path: Path) -> nn.Module:
        """Load trained model weights"""
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
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
        
        # Metrics containers
        all_ious = []
        all_dice_scores = []
        inference_times = []
        class_ious = [[] for _ in range(self.config.NUM_CLASSES)]
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        
        # GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Testing {model_name}")
            for images, masks, _ in pbar:
                images = images.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)
                
                # Measure inference time (per image)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                outputs = model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = (time.time() - start_time) * 1000 / images.size(0)
                inference_times.append(inference_time)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate IoU per class
                preds = torch.argmax(outputs, dim=1)
                ious = calculate_iou(preds, masks, self.config.NUM_CLASSES)
                all_ious.append(ious)
                
                # Store per-class IoUs
                for cls_idx, iou in enumerate(ious):
                    if not np.isnan(iou):
                        class_ious[cls_idx].append(iou)
                
                # Calculate Dice score
                dice = calculate_dice_score(outputs, masks, self.config.NUM_CLASSES)
                all_dice_scores.append(dice)
        
        # Calculate statistics
        avg_loss = total_loss / len(val_loader)
        mean_iou_per_class = np.nanmean(all_ious, axis=0)
        mean_iou = np.nanmean(mean_iou_per_class)
        std_iou = np.nanstd(all_ious)
        
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        fps = 1000.0 / mean_inference_time
        
        # Model size
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = model_params * 4 / (1024 ** 2)  # Assuming float32
        
        # GPU memory
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Per-class IoU statistics
        class_iou_means = [np.mean(cls_ious) if cls_ious else 0.0 
                          for cls_ious in class_ious]
        class_iou_stds = [np.std(cls_ious) if cls_ious else 0.0 
                         for cls_ious in class_ious]
        
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
            "peak_memory_mb": peak_memory_mb,
            "class_iou_means": class_iou_means,
            "class_iou_stds": class_iou_stds,
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  Model Params:    {model_params:,} ({model_size_mb:.2f} MB)")
        print(f"  Mean IoU:        {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"  Mean Dice:       {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"  Inference Time:  {mean_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"  FPS:             {fps:.2f}")
        print(f"  Avg Loss:        {avg_loss:.4f}")
        if torch.cuda.is_available():
            print(f"  Peak GPU Memory: {peak_memory_mb:.2f} MB")
        print(f"{'='*70}")
        
        return results
    
    def compare_models(self, results_dict: Dict[str, Dict]):
        """Generate comparison visualizations and reports"""
        if not results_dict:
            print("No results to compare")
            return
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in results_dict.items():
            if results:
                comparison_data.append({
                    "Model": model_name,
                    "mIoU": results["mean_iou"],
                    "Dice Score": results["mean_dice"],
                    "Inference (ms)": results["mean_inference_time_ms"],
                    "FPS": results["fps"],
                    "Params (M)": results["model_params"] / 1e6,
                    "Size (MB)": results["model_size_mb"],
                    "GPU Mem (MB)": results["peak_memory_mb"],
                })
        
        df = pd.DataFrame(comparison_data)
        
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
        
        # GPU Memory
        axes[1].bar(df["Model"], df["GPU Mem (MB)"], color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[1].set_ylabel('GPU Memory (MB)', fontsize=12)
        axes[1].set_title('Memory Usage: Peak GPU Memory', fontsize=14, fontweight='bold')
        for i, v in enumerate(df["GPU Mem (MB)"]):
            axes[1].text(i, v + 5, f'{v:.0f}', ha='center', fontsize=10)
        
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
            f.write(f"Lowest GPU Mem:   {df.loc[df['GPU Mem (MB)'].idxmin(), 'Model']} "
                   f"({df['GPU Mem (MB)'].min():.0f} MB)\n\n")
            
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
                    f.write(f"  Peak GPU Memory:  {results['peak_memory_mb']:.2f} MB\n\n")
                    
                    f.write(f"Accuracy Metrics:\n")
                    f.write(f"  Mean IoU:         {results['mean_iou']:.4f} ± {results['std_iou']:.4f}\n")
                    f.write(f"  Mean Dice:        {results['mean_dice']:.4f} ± {results['std_dice']:.4f}\n")
                    f.write(f"  Avg Loss:         {results['avg_loss']:.4f}\n\n")
                    
                    f.write(f"Speed Metrics:\n")
                    f.write(f"  Inference Time:   {results['mean_inference_time_ms']:.2f} ± "
                           f"{results['std_inference_time_ms']:.2f} ms\n")
                    f.write(f"  Throughput (FPS): {results['fps']:.2f}\n\n")
                    
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
                 val_loader: DataLoader) -> pd.DataFrame:
    """
    Run complete benchmark suite
    
    Args:
        models_dict: Dictionary of {model_name: model_instance}
        config: Configuration object
        val_loader: Validation data loader
    
    Returns:
        Comparison dataframe
    """
    benchmark = ModelBenchmark(config)
    results_dict = {}
    
    # Benchmark each model
    for model_name, model in models_dict.items():
        # Use the same safe-name logic as unified_training so the path matches
        model_path = config.OUTPUT_DIR / "models" / f"best_{_safe_filename(model_name)}_model.pth"

        if not model_path.exists():
            print(f"⚠️  Model weights not found: {model_path}")
            results_dict[model_name] = None
            continue

        # Load model
        loaded_model = benchmark.load_model(model, model_name, model_path)

        # Run benchmark
        results = benchmark.benchmark_model(loaded_model, model_name, val_loader)
        results_dict[model_name] = results

        # Save individual results to LOG_DIR
        if results:
            log_dir = config.LOG_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            results_path = log_dir / f"{_safe_filename(model_name)}_benchmark.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                # Convert numpy types to Python types for JSON serialization
                serializable_results = {
                    k: (v.tolist() if isinstance(v, np.ndarray)
                        else float(v) if isinstance(v, (np.float32, np.float64))
                        else v)
                    for k, v in results.items()
                }
                json.dump(serializable_results, f, indent=2)
    
    # Generate comparison
    comparison_df = benchmark.compare_models(results_dict)
    
    return comparison_df


if __name__ == "__main__":
    print("Comprehensive Benchmark Module - Ready")
    print("Use run_benchmark() to compare trained models")
