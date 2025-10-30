#!/usr/bin/env python3
"""
Evaluation script for aggregating results across all folds.
Generates confusion matrices and training plots.
"""

import subprocess
import sys
import os

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "pandas", "numpy", "scikit-learn"])

# Now import the libraries
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_fold_metrics(model_dir):
    """Load metrics from all folds"""
    metrics_file = os.path.join(model_dir, 'all_folds_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            fold_metrics = json.load(f)
        return fold_metrics
    
    return []

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - All Folds Aggregated', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def plot_training_curves(fold_metrics, output_path):
    """Plot training and validation loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    for i, metrics in enumerate(fold_metrics):
        if 'train_loss' in metrics and 'val_loss' in metrics:
            epochs = range(1, len(metrics['train_loss']) + 1)
            axes[0].plot(epochs, metrics['train_loss'], alpha=0.5, label=f'Fold {i+1} Train')
            axes[0].plot(epochs, metrics['val_loss'], alpha=0.5, label=f'Fold {i+1} Val', linestyle='--')
    
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot F1 scores
    for i, metrics in enumerate(fold_metrics):
        if 'val_f1' in metrics:
            epochs = range(1, len(metrics['val_f1']) + 1)
            axes[1].plot(epochs, metrics['val_f1'], alpha=0.7, label=f'Fold {i+1}')
    
    axes[1].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_path}")

def aggregate_results(fold_metrics, output_dir):
    """Aggregate and save results"""
    
    # Aggregate final metrics
    summary = fold_metrics
    all_tn, all_fp, all_fn, all_tp = 0, 0, 0, 0
    
    for metrics in fold_metrics:
        all_tn += metrics.get('true_negatives', 0)
        all_fp += metrics.get('false_positives', 0)
        all_fn += metrics.get('false_negatives', 0)
        all_tp += metrics.get('true_positives', 0)
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, 'fold_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*70}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*70}")
    print(f"Mean F1:         {summary_df['f1_score'].mean():.4f} ± {summary_df['f1_score'].std():.4f}")
    print(f"Mean Accuracy:   {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std():.4f}")
    print(f"Mean Precision:  {summary_df['precision'].mean():.4f} ± {summary_df['precision'].std():.4f}")
    print(f"Mean Recall:     {summary_df['recall'].mean():.4f} ± {summary_df['recall'].std():.4f}")
    print(f"{'='*70}\n")
    
    # Generate aggregate confusion matrix
    cm = np.array([[all_tn, all_fp], [all_fn, all_tp]])
    cm_path = os.path.join(output_dir, 'confusion_matrix_aggregate.png')
    plot_confusion_matrix(cm, ['Non-Defect', 'Defect'], cm_path)
    
    print(f"Summary saved to: {summary_path}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("EVALUATION - AGGREGATING ALL FOLDS")
    print(f"{'='*70}\n")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics from all folds
    fold_metrics = load_fold_metrics(args.model_dir)
    
    if not fold_metrics:
        print("\nNo fold metrics found. Ensure training completed successfully.")
        return
    
    print(f"\nFound metrics for {len(fold_metrics)} folds")
    
    # Aggregate and visualize
    summary_df = aggregate_results(fold_metrics, args.output_dir)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
