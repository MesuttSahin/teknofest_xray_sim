import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import os
import sys

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import config
from src.models.model import ChestXRayResNet
from src.data.dataset import ChestXRayDataset
from src.data.transforms import get_transforms

def load_model(model_path, device):
    print(f"[INFO] Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    model = ChestXRayResNet(num_classes=config.NUM_CLASSES)
    
    # Check if the checkpoint corresponds to V1 (Simple Linear) or V2 (Sequential with Dropout)
    # V2 has 'model.fc.1.weight' because it uses nn.Sequential(Dropout, Linear)
    # V1 (Old) likely has 'model.fc.weight' because it used just nn.Linear
    
    keys = list(state_dict.keys())
    has_sequential = any('model.fc.1.weight' in k for k in keys)
    
    if not has_sequential:
        print("[INFO] Detected V1 architecture (No Dropout in FC). Adjusting model...")
        # Get input features from the current Linear layer (which is at index 1 of Sequential)
        num_ftrs = model.model.fc[1].in_features
        # Replace Sequential with simple Linear
        model.model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
        
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[ERROR] Failed to load state_dict: {e}")
        # Final attempt: maybe keys don't have 'model.' prefix? (unlikely given inspection)
        # But if they don't, we can try removing/adding prefix.
        # For now, just re-raise.
        raise e
        
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    print("[INFO] Evaluating model...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return all_preds, all_labels

def calculate_f1_scores(preds, labels, class_names):
    f1_scores = {}
    for i, class_name in enumerate(class_names):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        f1_scores[class_name] = f1
    return f1_scores

def plot_comparison(f1_v1, f1_v2, class_names, output_path):
    print(f"[INFO] Plotting comparison to {output_path}...")
    
    # Data preparation
    indices = np.arange(len(class_names))
    width = 0.35
    
    v1_scores = [f1_v1[name] for name in class_names]
    v2_scores = [f1_v2[name] for name in class_names]
    
    plt.figure(figsize=(15, 8))
    plt.bar(indices - width/2, v1_scores, width, label='V1 (Old) F1 Score', color='skyblue')
    plt.bar(indices + width/2, v2_scores, width, label='V2 (New) F1 Score', color='lightgreen')
    
    plt.xlabel('Diseases')
    plt.ylabel('F1 Score')
    plt.title('Weighted Loss & Augmentation Etkisi (V1 vs V2)')
    plt.xticks(indices, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("[INFO] Plot saved.")

def main():
    device = config.DEVICE
    print(f"[INFO] Using device: {device}")

    # Paths
    val_csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'val_list.csv')
    v1_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'best_model.pth')
    v2_model_path = config.BEST_MODEL_PATH_V2
    output_plot_path = os.path.join(config.LOGS_DIR, 'comparison.png')

    # Dataset & DataLoader
    val_transforms = get_transforms('val')
    val_dataset = ChestXRayDataset(val_csv_path, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # 1. Evaluate V1
    if os.path.exists(v1_model_path):
        model_v1 = load_model(v1_model_path, device)
        preds_v1, labels = evaluate_model(model_v1, val_loader, device)
        f1_v1 = calculate_f1_scores(preds_v1, labels, config.CLASS_NAMES)
    else:
        print(f"[WARNING] V1 model not found at {v1_model_path}. Using zeros for comparison.")
        f1_v1 = {name: 0.0 for name in config.CLASS_NAMES}

    # 2. Evaluate V2
    if os.path.exists(v2_model_path):
        model_v2 = load_model(v2_model_path, device)
        preds_v2, _ = evaluate_model(model_v2, val_loader, device) # Labels are same
        f1_v2 = calculate_f1_scores(preds_v2, labels, config.CLASS_NAMES)
    else:
        print(f"[WARNING] V2 model not found at {v2_model_path}. Using zeros for comparison.")
        f1_v2 = {name: 0.0 for name in config.CLASS_NAMES}

    # 3. Plot
    plot_comparison(f1_v1, f1_v2, config.CLASS_NAMES, output_plot_path)

    # 4. Report
    print("\n" + "="*40)
    print("      MODEL COMPARISON REPORT      ")
    print("="*40)
    print(f"{'Disease':<20} | {'V1 F1':<10} | {'V2 F1':<10} | {'Diff':<10}")
    print("-" * 58)
    
    for name in config.CLASS_NAMES:
        s1 = f1_v1.get(name, 0)
        s2 = f1_v2.get(name, 0)
        diff = s2 - s1
        diff_str = f"{diff:+.4f}"
        
        # Highlight significant improvements
        marker = " << IMPROVED" if diff > 0.05 else ""
        if name in ['Hernia', 'Nodule'] and diff > 0:
            marker += " [TARGET]"
            
        print(f"{name:<20} | {s1:.4f}     | {s2:.4f}     | {diff_str:<10}{marker}")
        
    print("="*40)
    print(f"Comparison chart saved to: {output_plot_path}")

if __name__ == "__main__":
    main()
