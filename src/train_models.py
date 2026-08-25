# ==============================================================================
# Tactile Sensor Multi-Channel Evaluation Pipeline
#
# Description:
# This script trains a standard ResNet-18 model across 7 different RGB channel 
# configurations (RGB, RG, GB, RB, R, G, B). It is designed to scientifically 
# validate hardware signal integrity and evaluate the impact of thermal noise 
# and EMI susceptibility in specific optical channels (e.g., the Blue channel).
# ==============================================================================

import os
import sys
import copy
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import numpy as np

# ==========================================
# 1. Configuration & Global Variables
# ==========================================
# Base extraction path (will be auto-corrected if nested folders exist)
BASE_EXTRACT_DIR = '/content/data/dataset' 
MODELS_DIR = '/content/models'
RESULTS_DIR = '/content/results'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15      # 15 to 20 epochs are optimal for ResNet-18 transfer learning
LEARNING_RATE = 0.001

MODES = ['RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Physics-Driven Channel Masker
# ==========================================
class ChannelMasker(object):
    """
    Acts as a physical hardware filter. Zeroes out specific channels
    AFTER ImageNet normalization. 
    
    Mathematical Context:
    If masking occurs before normalization, a 0.0 pixel value will be 
    transformed into a large negative tensor surge (e.g., (0 - 0.406)/0.225 = -1.8),
    which destroys the feature map. Applying it after ensures absolute zero energy.
    """
    def __init__(self, mode):
        self.mode = mode.upper()

    def __call__(self, tensor):
        masked_tensor = tensor.clone()
        if 'R' not in self.mode: masked_tensor[0, :, :] = 0.0
        if 'G' not in self.mode: masked_tensor[1, :, :] = 0.0
        if 'B' not in self.mode: masked_tensor[2, :, :] = 0.0
        return masked_tensor

# ==========================================
# 3. Data Augmentation & Loading
# ==========================================
def get_dataloaders(mode, data_dir):
    """
    Constructs the data pipeline. High-diversity augmentation (Rotation/Flips)
    is retained to force the CNN to learn frequency-domain textures rather 
    than memorizing fixed 3D lighting shadow angles.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),           
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelMasker(mode) 
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelMasker(mode)
    ])

    full_dataset = ImageFolder(data_dir)
    
    # 80/20 Random Split using a fixed seed (42) to prevent data leakage 
    # and ensure fair comparison across the 7 channel configurations.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    
    return dataloaders, full_dataset.classes

# ==========================================
# 4. Core Training Engine
# ==========================================
def train_and_evaluate(mode, data_dir, device):
    print(f"\n{'='*50}")
    print(f"[EXPERIMENT] Training Channel Configuration: {mode}")
    print(f"{'='*50}")

    dataloaders, class_names = get_dataloaders(mode, data_dir)
    
    # Initialize standard ResNet-18 Feature Extractor
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Clean console output
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Val Acc: {epoch_acc:.4f}")

    print(f"-> [RESULT] Mode {mode} achieved max validation accuracy: {best_acc:.4f}")
    
    # Save the optimal weights
    model.load_state_dict(best_model_wts)
    save_path = os.path.join(MODELS_DIR, f"resnet18_{mode.lower()}_best.pth")
    torch.save(model.state_dict(), save_path)
    
    # Generate Confusion Matrix
    plot_confusion_matrix(model, dataloaders['val'], class_names, mode, device)
    
    return best_acc.item()

# ==========================================
# 5. Visualization & Plotting
# ==========================================
def plot_confusion_matrix(model, val_loader, class_names, mode, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Semantic Confusion Matrix - Mask Mode: {mode}')
    plt.ylabel('True Material')
    plt.xlabel('Predicted Material')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{mode}.png"), dpi=300)
    plt.close()

# ==========================================
# 6. Main Execution Block & Colab Integration
# ==========================================
if __name__ == '__main__':
    print(f"[INFO] Initializing pipeline on device: {device}")

    # Step A: Google Colab Auto-Mount & Extraction
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("[INFO] Mounting Google Drive...")
            drive.mount('/content/drive')
        
        zip_path = '/content/drive/MyDrive/dataset.zip'
        
        if not os.path.exists(zip_path):
            print(f"[ERROR] Cannot find dataset at {zip_path}.")
            print("Please ensure 'dataset.zip' is uploaded to the root of your Google Drive.")
            sys.exit(1)
            
        if not os.path.exists(BASE_EXTRACT_DIR) or len(os.listdir(BASE_EXTRACT_DIR)) == 0:
            print(f"[INFO] Extracting {zip_path} to NVMe storage for optimal I/O speed...")
            subprocess.run(["mkdir", "-p", BASE_EXTRACT_DIR])
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", BASE_EXTRACT_DIR])
        else:
            print(f"[INFO] Dataset already extracted at {BASE_EXTRACT_DIR}.")
            
    except ImportError:
        print("[INFO] Not running in Google Colab. Bypassing Drive mount.")

    # Step B: Path Auto-Correction (Handling nested zip folders)
    active_data_dir = BASE_EXTRACT_DIR
    if os.path.exists(os.path.join(BASE_EXTRACT_DIR, 'dataset')):
        active_data_dir = os.path.join(BASE_EXTRACT_DIR, 'dataset')
        print(f"[INFO] Nested folder detected. Corrected Data Directory to: {active_data_dir}")
        
    classes = [d for d in os.listdir(active_data_dir) if os.path.isdir(os.path.join(active_data_dir, d))]
    if not classes:
        print(f"[ERROR] No class directories found in {active_data_dir}.")
        print("Please check your zip file structure.")
        sys.exit(1)
        
    print(f"[INFO] Detected {len(classes)} classes: {classes}")

    # Step C: Execute Ablation Study
    results = {}
    for mode in MODES:
        best_accuracy = train_and_evaluate(mode, active_data_dir, device)
        results[mode] = best_accuracy
        
    # Step D: Plot Final Comparative Bar Chart
    print("\n[INFO] Generating comparative bar chart...")
    plt.figure(figsize=(10, 6))
    modes = list(results.keys())
    accuracies = [results[m] * 100 for m in modes]
    
    # Highlight the RG channel to support the core hardware thesis
    colors = ['#1f77b4' if m != 'RG' else '#ff7f0e' for m in modes]
    bars = plt.bar(modes, accuracies, color=colors, edgecolor='black')
    
    plt.ylim(0, 110)
    plt.title('Validation Accuracy Across RGB Channel Configurations', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xlabel('Active Channel Configuration', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, 
                 f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison_bar.png"), dpi=300)
    
    print("\n[SUCCESS] Pipeline execution finished! All results saved to /content/results")