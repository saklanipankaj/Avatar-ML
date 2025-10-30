import os
import json
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
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
        print(f"\nDetailed Metrics | F1: {f1:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | CM: TN={tn} FP={fp} FN={fn} TP={tp}")
    
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
    """Print compact epoch summary"""
    warnings = []
    if val_metrics['f1_score'] == 0.0:
        warnings.append("F1=0")
    if val_metrics['true_positives'] == 0:
        warnings.append("TP=0")
    if val_metrics['true_negatives'] == 0:
        warnings.append("TN=0")
    
    warn_str = f" ⚠️ {','.join(warnings)}" if warnings else ""
    print(f"E{epoch+1:2d}/{total_epochs} | T: {train_metrics['loss']:.3f}/{train_metrics['accuracy']:.3f} | V: {val_metrics['loss']:.3f}/{val_metrics['accuracy']:.3f}/{val_metrics['f1_score']:.3f}{warn_str}")

def save_confusion_matrix(cm, fold_key, fold_dir):
    """Save confusion matrix as PNG"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Defect'], 
                yticklabels=['Normal', 'Defect'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {fold_key}')
    plt.tight_layout()
    cm_path = os.path.join(fold_dir, f'{fold_key}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm_path

def train_single_fold(fold_key, train_file, val_file, images_dir, args, device):
    """Train a single fold"""
    start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    class_name = fold_key.split('_fold_')[0]
    class_dir = os.path.join(args.model_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    fold_dir = os.path.join(class_dir, fold_key)
    os.makedirs(fold_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Training {class_name} - {fold_key}")
    print(f"Start time: {start_datetime}")
    print(f"Saving to: {fold_dir}")
    print(f"{'='*50}")
    
    # Load data
    train_dataset = ImageDataset(train_file, images_dir)
    val_dataset = ImageDataset(val_file, images_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = ADLModel(args.num_classes, args.drop_rate, args.drop_threshold).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
    
    # Training loop with epoch progress bar
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    best_metrics = None
    
    epoch_pbar = tqdm(range(args.epochs), desc=f'Training {class_name}', unit='epoch')
    for epoch in epoch_pbar:
        train_metrics = train_one_epoch(model, train_loader, criterion_cls, optimizer, device, args.adl_alpha)
        val_metrics = validate(model, val_loader, criterion_cls, device, verbose=(epoch % 10 == 0))
        print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics)
        scheduler.step()
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            epochs_without_improvement = 0
            best_metrics = val_metrics
            # Save fold model - FIXED INDENTATION
            fold_model_path = os.path.join(fold_dir, 'best_model.pth')
            torch.save(model.state_dict(), fold_model_path)
            print(f"✓ New best F1: {best_val_f1:.4f} - Model saved to {fold_model_path}")
        else:
            epochs_without_improvement += 1
        
        epoch_pbar.set_postfix({'F1': f'{val_metrics["f1_score"]:.4f}', 'Best': f'{best_val_f1:.4f}', 'Patience': f'{epochs_without_improvement}/{args.patience}'})
        
        if epochs_without_improvement >= args.patience:
            print(f"\nℹ️  Early stopping after {args.patience} epochs without improvement")
            break
    
    epoch_pbar.close()
    
    end_time = time.time()
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_duration = end_time - start_time
    
    print(f"\n{class_name} completed | Best F1: {best_val_f1:.4f}")
    print(f"Training duration: {training_duration/60:.2f} minutes ({training_duration:.2f} seconds)\n")
    
    # Save confusion matrix
    if best_metrics is not None:
        cm_path = save_confusion_matrix(best_metrics['confusion_matrix'], fold_key, fold_dir)
        print(f"Confusion matrix saved to: {cm_path}")
    
        # Save fold metrics
        fold_metrics_path = os.path.join(fold_dir, 'metrics.json')
        fold_metrics_data = {
            'fold': fold_key,
            'class_name': class_name,
            'fold_directory': fold_dir,
            'start_time': start_datetime,
            'end_time': end_datetime,
            'training_duration_seconds': training_duration,
            'final_f1': best_val_f1,
            'final_accuracy': best_metrics['accuracy'],
            'final_balanced_accuracy': best_metrics['balanced_accuracy'],
            'final_precision': best_metrics['precision'],
            'final_recall': best_metrics['recall'],
            'final_sensitivity': best_metrics['sensitivity'],
            'final_specificity': best_metrics['specificity'],
            'true_negatives': best_metrics['true_negatives'],
            'false_positives': best_metrics['false_positives'],
            'false_negatives': best_metrics['false_negatives'],
            'true_positives': best_metrics['true_positives'],
            'confusion_matrix': best_metrics['confusion_matrix'].tolist()
        }
        with open(fold_metrics_path, 'w') as f:
            json.dump(fold_metrics_data, f, indent=2)
        print(f"Metrics saved to: {fold_metrics_path}")
    else:
        print("Warning: No best metrics found (training may have failed)")
        fold_metrics_data = {
            'fold': fold_key,
            'class_name': class_name,
            'error': 'Training completed but no best metrics recorded'
        }
    
    return fold_metrics_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--images', type=str, default=os.environ.get('SM_CHANNEL_IMAGES'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--defect_classes', type=str, default=None, help='Comma-separated list of classes to train (e.g., "Blowhole,Crack"). If not specified, trains all classes.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.4)
    parser.add_argument('--drop_threshold', type=float, default=0.8)
    parser.add_argument('--adl_alpha', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*50}")
    print("Training Configuration")
    print(f"{'='*50}")
    print(f"Model directory: {args.model_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Patience: {args.patience}")
    print(f"{'='*50}\n")
    
    # Parse defect_classes
    defect_classes_list = None
    if args.defect_classes:
        defect_classes_list = [c.strip() for c in args.defect_classes.split(',')]
    
    # Discover all folds
    fold_data = {}
    for class_name in os.listdir(args.train):
        class_train_dir = os.path.join(args.train, class_name)
        if not os.path.isdir(class_train_dir):
            continue
        
        # Filter by defect_classes if specified
        if defect_classes_list and class_name not in defect_classes_list:
            continue
        
        for fold_dir in os.listdir(class_train_dir):
            if fold_dir.startswith('fold_'):
                fold_num = fold_dir.split('_')[1]
                fold_key = f"{class_name}_fold_{fold_num}"
                train_file = os.path.join(class_train_dir, fold_dir, 'train_metadata.json')
                val_file = os.path.join(args.val, class_name, fold_dir, 'val_metadata.json')
                
                if os.path.exists(train_file) and os.path.exists(val_file):
                    fold_data[fold_key] = {'train': train_file, 'val': val_file}
    
    if defect_classes_list:
        print(f"\nTraining only classes: {', '.join(