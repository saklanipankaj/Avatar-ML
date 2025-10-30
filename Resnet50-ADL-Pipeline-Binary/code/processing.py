#!/usr/bin/env python3
"""
Data Preprocessing Script for SageMaker Pipeline
Saves image paths and fold splits instead of loading all images
"""

import os
import json
import argparse
from glob import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np

def load_image_paths_and_labels(input_dir: str, class_name: str):
    """Load image paths and labels from input directory with train types"""
    image_paths = []
    labels = []
    
    train_types = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Loading paths for class: {class_name}")
    for train_type in train_types:
        train_type_dir = os.path.join(input_dir, train_type)
        
        for label_name, label_value in [('non-defect', 0), ('defect', 1)]:
            folder_path = os.path.join(train_type_dir, label_name, class_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            paths = glob(os.path.join(folder_path, '*.jpg'))
            print(f"Found {len(paths)} images for '{train_type}/{label_name}'")
            
            image_paths.extend(paths)
            labels.extend([label_value] * len(paths))
    
    return image_paths, np.array(labels)

def save_fold_metadata(image_paths, labels, output_path: str, filename: str):
    """Save fold metadata as JSON"""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    
    data = {
        'image_paths': image_paths,
        'labels': labels.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved to: {filepath}")

def process_class_data(input_dir: str, train_output_dir: str, val_output_dir: str, class_name: str, n_folds: int = 3):
    """Create k-fold splits and save metadata"""
    print(f"\nProcessing class: {class_name}")
    
    image_paths, labels = load_image_paths_and_labels(input_dir, class_name)
    
    if len(image_paths) == 0:
        print(f"ERROR: No images found for {class_name}")
        return
    
    print(f"Total images: {len(image_paths)}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # StratifiedKFold maintains class distribution across folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(image_paths, labels)):
        fold_num = fold_idx + 1
        print(f"\nProcessing Fold {fold_num}/{n_folds}")
        
        train_paths = [image_paths[i] for i in train_indices]
        val_paths = [image_paths[i] for i in val_indices]
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        
        print(f"Train: {len(train_paths)} images")
        print(f"Val: {len(val_paths)} images")
        
        train_fold_dir = os.path.join(train_output_dir, class_name, f'fold_{fold_num}')
        val_fold_dir = os.path.join(val_output_dir, class_name, f'fold_{fold_num}')
        save_fold_metadata(train_paths, train_labels, train_fold_dir, 'train_metadata.json')
        save_fold_metadata(val_paths, val_labels, val_fold_dir, 'val_metadata.json')

def discover_defect_classes(input_dir: str):
    """Discover all defect classes from train_type folders"""
    defect_classes = set()
    
    train_types = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for train_type in train_types:
        for label_name in ['defect', 'non-defect']:
            label_dir = os.path.join(input_dir, train_type, label_name)
            if os.path.isdir(label_dir):
                classes = [d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))]
                defect_classes.update(classes)
    
    return sorted(list(defect_classes))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    
    args = parser.parse_args()
    
    print("Starting data preprocessing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Discover all defect classes from the dataset
    class_names = discover_defect_classes(args.input_dir)
    print(f"\nDiscovered defect classes: {class_names}")
    
    train_output_dir = os.path.join(args.output_dir, 'train')
    val_output_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    for class_name in class_names:
        process_class_data(args.input_dir, train_output_dir, val_output_dir, class_name)
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
