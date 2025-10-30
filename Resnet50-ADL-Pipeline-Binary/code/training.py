#!/usr/bin/env python3
"""
Model Training Script for SageMaker Pipeline
Runs in PyTorch container for model training step
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

class ImageDataset(Dataset):
    """Dataset that loads images on-demand from paths"""
    
    def __init__(self, metadata_path: str, images_dir: str):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Remap paths from preprocessing container to training container
        self.image_paths = []
        for path in data['image_paths']:
            # Extract relative path after '/opt/ml/processing/input/'
            rel_path = path.split('/opt/ml/processing/input/')[-1]
            new_path = os.path.join(images_dir, rel_path)
            self.image_paths.append(new_path)
        
        self.labels = data['labels']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Loaded {len(self.labels)} samples")
        print(f"Class distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

class ADL(nn.Module):
    """Attention-based Dropout Layer"""
    
    def __init__(self, in_channels: int, drop_rate: float, drop_threshold: float):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_threshold = drop_threshold
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, feature_maps: torch.Tensor):
        attention_map = self.attention_conv(feature_maps)
        attention_map = torch.sigmoid(attention_map)
        
        if self.training:
            B, _, H, W = attention_map.shape
            attention_flat = attention_map.view(B, -1)
            M = int(H * W * self.drop_rate)
            if M <= 0:
                return feature_maps, attention_map
            _, topk_indices = torch.topk(attention_flat, k=M, dim=1)
            drop_mask = torch.ones_like(attention_flat)
            drop_mask.scatter_(dim=1, index=topk_indices, value=0)
            drop_mask = drop_mask.view(B, 1, H, W)
        else:
            drop_mask = (attention_map < self.drop_threshold).float()
        
        dropped_feature_maps = feature_maps * drop_mask
        return dropped_feature_maps, attention_map

class ADLModel(nn.Module):
    """ResNet50-based model with Attention-based Dropout Layer"""
    
    def __init__(self, num_classes: int, drop_rate: float, drop_threshold: float):
        super().__init__()
        
        resnet = resnet50(
            weights=ResNet50_Weights.DEFAULT,
            zero_init_residual=True,
            replace_stride_with_dilation=[True, True, True]
        )
        
        # Extract feature layers
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        in_channels = 2048  # ResNet50 final feature map channels
        self.adl_layer = ADL(in_channels, drop_rate, drop_threshold)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        feature_maps = self.backbone(x)
        dropped_maps, attention_map = self.adl_layer(feature_maps)
        logits = self.classifier(dropped_maps)
        logits = F.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
        return logits, attention_map

def train_one_epoch(model, dataloader, criterion_cls, optimizer, device, adl_alpha):
    """Train for one epoch with progress bar"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits, attention_map = model(inputs)
        
        # Classification loss
        loss_cls = criterion_cls(logits, labels)
        
        # ADL regularization loss
        loss_adl = torch.mean(attention_map)
        
        # Total loss
        total_loss = loss_cls + adl_alpha * loss_adl
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += total_loss.item() * inputs.size(0)
        _, preds = torch.max(logits, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss.item(), 'acc': (correct_predictions.double() / total_samples).item()})
    
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    return {"loss": epoch_loss, "accuracy": epoch_acc.item()}

def validate(model, dataloader, criterion_cls, device, verbose=False):
    """Validate the model with comprehensive metrics"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = model(inputs)
            loss = criterion_cls(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    epoch_loss = running_loss / len(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0, labels=[0, 1]
    )
    
    # Binary metrics for defect detection
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0, pos_label=1
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity and Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall
    
    # Prediction distribution
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    pred_distribution = dict(zip(unique_preds.tolist(), pred_counts.tolist()))
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    label_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))
    
    if verbose:
        print(f"\n{'='*60}")
        print("DETAILED VALIDATION METRICS")
        print(f"{'='*60}")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"\nPer-Class Metrics:")
        print(f"  Class 0 (Non-Defect): Prec: {precision_per_class[0]:.4f}, Rec: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
        print(f"  Class 1 (Defect):     Prec: {precision_per_class[1]:.4f}, Rec: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
        print(f"\nBinary Metrics (Defect Detection):")
        print(f"  Sensitivity (TPR): {sensitivity:.4f}")
        print(f"  Specificity (TNR): {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        print(f"\nPrediction Distribution: {pred_distribution}")
        print(f"Ground Truth Distribution: {label_distribution}")
        print(f"{'='*60}\n")
    
    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "pred_distribution": pred_distribution,
        "label_distribution": label_distribution
    }

def print_epoch_summary(epoch, total_epochs, train_metrics, val_metrics):
    """Print clean epoch summary with warnings"""
    print(f"\n{'─'*70}")
    print(f"Epoch {epoch+1}/{total_epochs}")
    print(f"{'─'*70}")
    print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")
    print(f"      | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | BalAcc: {val_metrics['balanced_accuracy']:.4f}")
    print(f"      | Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f}")
    
    if val_metrics['f1_score'] == 0.0:
        print(f"\n⚠️  WARNING: F1 = 0! Model may be predicting only one class.")
        print(f"   Predictions: {val_metrics['pred_distribution']}")
        print(f"   Ground Truth: {val_metrics['label_distribution']}")
    
    if val_metrics['true_positives'] == 0:
        print(f"⚠️  WARNING: No defects detected (TP = 0)")
    
    if val_metrics['true_negatives'] == 0:
        print(f"⚠️  WARNING: All samples predicted as defects (TN = 0)")
    
    print(f"{'─'*70}")

def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--images', type=str, default=os.environ.get('SM_CHANNEL_IMAGES'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--defect_classes', type=str, default='lighting_panel,shifted_grab_handle,frosted_window,Diffuser_cover', 
                        help='Comma-separated list of defect classes or "all" to train on all classes')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.4)
    parser.add_argument('--drop_threshold', type=float, default=0.8)
    parser.add_argument('--adl_alpha', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Parse defect classes - if 'all', discover all classes from data
    if args.defect_classes.lower() == 'all':
        # Discover all classes from train directory
        class_names = [d for d in os.listdir(args.train) if os.path.isdir(os.path.join(args.train, d))]
        print(f"Training on all discovered defect classes: {class_names}")
    else:
        class_names = [c.strip() for c in args.defect_classes.split(',')]
        print(f"Training on specified defect classes: {class_names}")
    
    # Find training data files for specified classes
    train_files = []
    val_files = []
    
    for class_name in class_names:
        for root, dirs, files in os.walk(os.path.join(args.train, class_name)):
            for file in files:
                if file.endswith('train_metadata.json'):
                    train_files.append(os.path.join(root, file))
        
        for root, dirs, files in os.walk(os.path.join(args.val, class_name)):
            for file in files:
                if file.endswith('val_metadata.json'):
                    val_files.append(os.path.join(root, file))
    
    print(f"Found {len(train_files)} training files")
    print(f"Found {len(val_files)} validation files")
    
    # Train on the first fold of the first class
    if train_files and val_files:
        train_dataset = ImageDataset(train_files[0], args.images)
        val_dataset = ImageDataset(val_files[0], args.images)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Initialize model
        model = ADLModel(args.num_classes, args.drop_rate, args.drop_threshold).to(device)
        
        # Initialize optimizer and loss
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
        
        # Training loop
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total epochs: {args.epochs}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        
        for epoch in range(args.epochs):
            # Training
            train_metrics = train_one_epoch(
                model, train_loader, criterion_cls, optimizer, device, args.adl_alpha
            )
            
            # Validation (verbose every 10 epochs)
            val_metrics = validate(
                model, val_loader, criterion_cls, device, 
                verbose=(epoch % 10 == 0 or epoch == args.epochs - 1)
            )
            
            # Print epoch summary
            print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics)
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                epochs_without_improvement = 0
                
                # Save model
                torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
                print(f"✓ New best model saved (F1: {best_val_f1:.4f})")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= args.patience:
                print(f"\n⏹️  Early stopping after {args.patience} epochs without improvement")
                print(f"   Best F1 score: {best_val_f1:.4f}")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best F1 score: {best_val_f1:.4f}")
        print(f"Model saved to: {os.path.join(args.model_dir, 'model.pth')}")
        print(f"{'='*70}\n")
        
        # Final evaluation with best model
        print(f"\n{'='*70}")
        print(f"FINAL EVALUATION ON BEST MODEL")
        print(f"{'='*70}")
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth'), map_location=device))
        
        # Run final evaluation
        final_metrics = validate(model, val_loader, criterion_cls, device, verbose=True)
        
        print(f"\nFinal Best Model Performance:")
        print(f"  F1 Score:         {final_metrics['f1_score']:.4f}")
        print(f"  Accuracy:         {final_metrics['accuracy']:.4f}")
        print(f"  Balanced Acc:     {final_metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:        {final_metrics['precision']:.4f}")
        print(f"  Recall:           {final_metrics['recall']:.4f}")
        print(f"  Sensitivity:      {final_metrics['sensitivity']:.4f}")
        print(f"  Specificity:      {final_metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {final_metrics['true_negatives']:4d}  FP: {final_metrics['false_positives']:4d}")
        print(f"  FN: {final_metrics['false_negatives']:4d}  TP: {final_metrics['true_positives']:4d}")
        print(f"{'='*70}\n")
        
        # Save metrics as JSON for evaluation step
        metrics_output = {
            'final_f1': final_metrics['f1_score'],
            'final_accuracy': final_metrics['accuracy'],
            'final_balanced_accuracy': final_metrics['balanced_accuracy'],
            'final_precision': final_metrics['precision'],
            'final_recall': final_metrics['recall'],
            'final_sensitivity': final_metrics['sensitivity'],
            'final_specificity': final_metrics['specificity'],
            'labels': final_metrics['label_distribution'],
            'predictions': final_metrics['pred_distribution']
        }
        
        metrics_path = os.path.join(args.model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_output, f, indent=2)
        print(f"Metrics saved to: {metrics_path}\n")
    
    else:
        print("No training data found!")

if __name__ == "__main__":
    main()