"""
train_efficientnet.py
EfficientNet-B0 transfer learning for Pakistani politician classification.

Phases:
  Phase 1 — Freeze all pretrained layers, train only the new classifier head.
  Phase 2 — Unfreeze everything, fine-tune with a lower learning rate.

Usage (local, after downloading results from Colab):
    python src/models/train_efficientnet.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

from src.models.model_config import MODEL_CONFIG


# ---------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------

def build_efficientnet_b0(num_classes, freeze_layers=True):
    """
    Build EfficientNet-B0 with ImageNet pretrained weights.
    Replaces the classifier head for num_classes.
    If freeze_layers=True, only the new head is trainable initially.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ---------------------------------------------------------------
# Data transforms  (identical pipeline to ResNet for fair comparison)
# ---------------------------------------------------------------

def get_transforms(image_size, mean, std):
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------

def load_datasets(dataset_path, train_tf, val_tf, batch_size):
    train_ds = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'val'), transform=val_tf)
    test_ds = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'test'), transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes


# ---------------------------------------------------------------
# Training loop (Phase 1 + Phase 2)
# ---------------------------------------------------------------

def train_model(model, model_name, train_loader, val_loader, config, device):
    """
    Two-phase training:
      Phase 1: Frozen backbone, train head only.
      Phase 2: Unfreeze all, fine-tune with lower LR.

    Returns: model, history, best_val_acc, best_model_path
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs(config['model_save_path'], exist_ok=True)
    best_model_path = os.path.join(config['model_save_path'], f'best_{model_name}.pth')

    print(f"\n{'='*60}\nPhase 1 Training: {model_name}\n{'='*60}")

    for epoch in range(config['num_epochs']):
        t0 = time.time()

        # --- train ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_correct += out.argmax(1).eq(labels).sum().item()
            t_total += labels.size(0)

        avg_tl = t_loss / t_total
        avg_ta = t_correct / t_total

        # --- validate ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += out.argmax(1).eq(labels).sum().item()
                v_total += labels.size(0)

        avg_vl = v_loss / v_total
        avg_va = v_correct / v_total
        scheduler.step(avg_vl)

        history['train_loss'].append(avg_tl)
        history['train_acc'].append(avg_ta)
        history['val_loss'].append(avg_vl)
        history['val_acc'].append(avg_va)

        print(f"Epoch [{epoch+1:3d}/{config['num_epochs']}] "
              f"TLoss:{avg_tl:.4f} TAcc:{avg_ta:.4f} "
              f"VLoss:{avg_vl:.4f} VAcc:{avg_va:.4f} "
              f"({time.time()-t0:.1f}s)")

        if avg_va > best_val_acc:
            best_val_acc = avg_va
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, best_model_path)
            print(f"  ✅ Saved best model  Val Acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{config['patience']}")

        if epochs_no_improve >= config['patience']:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

    # --- Phase 2: fine-tune all layers ---
    print(f"\n{'='*40}\nPhase 2: Fine-tuning all layers\n{'='*40}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = True

    ft_optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'] * config['finetune_lr_factor'],
        weight_decay=config['weight_decay'],
    )

    for epoch in range(config['finetune_epochs']):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{config['finetune_epochs']}"):
            imgs, labels = imgs.to(device), labels.to(device)
            ft_optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            ft_optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_correct += out.argmax(1).eq(labels).sum().item()
            t_total += labels.size(0)

        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += out.argmax(1).eq(labels).sum().item()
                v_total += labels.size(0)

        ft_va = v_correct / v_total
        print(f"FT [{epoch+1}/{config['finetune_epochs']}] Val Acc: {ft_va:.4f}")

        if ft_va > best_val_acc:
            best_val_acc = ft_va
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': ft_optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, best_model_path)
            print(f"  ✅ Fine-tuned best saved  Val Acc: {best_val_acc:.4f}")

    print(f"\n🏁 Done: {model_name}  Best Val Acc: {best_val_acc:.4f}")
    print(f"   Saved: {best_model_path}")
    return model, history, best_val_acc, best_model_path


# ---------------------------------------------------------------
# Entry point (local run)
# ---------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = MODEL_CONFIG
    train_tf, val_tf = get_transforms(
        cfg['image_size'], cfg['imagenet_mean'], cfg['imagenet_std'])
    train_loader, val_loader, test_loader, class_names = load_datasets(
        cfg['dataset_path'], train_tf, val_tf, cfg['batch_size'])

    cfg['num_classes'] = len(class_names)
    cfg['class_names'] = class_names

    model = build_efficientnet_b0(cfg['num_classes'], freeze_layers=True)
    train_model(model, 'EfficientNetB0', train_loader, val_loader, cfg, device)