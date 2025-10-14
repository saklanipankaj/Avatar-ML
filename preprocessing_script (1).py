
#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
import logging
import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_dir, image_size=(224, 224), val_split=0.2):
    """Load and preprocess AR dataset"""
    class_names = ['non-defect', 'defect']
    class_map = {name: i for i, name in enumerate(class_names)}
    
    search_pattern = os.path.join(data_dir, 'AR_Train', '*', 'shifted_grab_handle', '*.jpg')
    image_paths = glob(search_pattern)
    
    if not image_paths:
        logger.warning(f"No images found with pattern: {search_pattern}")
        return (None, None), (None, None)
    
    images = []
    labels = []
    
    logger.info(f"Found {len(image_paths)} images. Preprocessing...")
    
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            
            img_resized = cv2.resize(img, (image_size[1], image_size[0]))
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            class_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
            
            if class_name in class_map:
                images.append(img_normalized)
                labels.append(class_map[class_name])
                
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
    
    if not images:
        return (None, None), (None, None)
    
    X = np.array(images)
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )
    
    logger.info(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val)

def save_data(data, output_path, filename):
    """Save data using pickle"""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-height", default=224)
    parser.add_argument("--image-width", default=224)
    parser.add_argument("--val-split", default=0.2)
    args = parser.parse_args()
    
    input_dir = "/opt/ml/processing/input"
    output_train_dir = "/opt/ml/processing/output/train"
    output_val_dir = "/opt/ml/processing/output/val"
    
    (X_train, y_train), (X_val, y_val) = load_and_preprocess_data(
        data_dir=input_dir,
        image_size=(args.image_height, args.image_width),
        val_split=args.val_split
    )
    
    if X_train is not None:
        save_data({'images': X_train, 'labels': y_train}, output_train_dir, 'train_data.pkl')
        save_data({'images': X_val, 'labels': y_val}, output_val_dir, 'val_data.pkl')
        save_data({
            'image_size': (args.image_height, args.image_width),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }, output_train_dir, 'metadata.pkl')
        logger.info("Preprocessing completed successfully!")
    else:
        logger.error("No data processed")
