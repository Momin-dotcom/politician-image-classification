"""
Standalone evaluation script.
Run this to evaluate a saved model on the test set.
Usage: python src/evaluation/evaluate.py --model results/best_ResNet50.pth --model-type resnet50
"""

import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def load_model(model_path, model_type, num_classes):
    """Load a saved model checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint.get('class_names', None)
    return model, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .pth file')
    parser.add_argument('--model-type', required=True, choices=['resnet50', 'efficientnet_b0'])
    parser.add_argument('--dataset', default='dataset/test', help='Test dataset path')
    parser.add_argument('--output', default='results', help='Where to save results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(args.dataset, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes
    
    model, saved_class_names = load_model(args.model, args.model_type, len(class_names))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {args.model_type}')
    plt.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    plt.savefig(os.path.join(args.output, f'cm_{args.model_type}.png'), dpi=150)
    plt.show()


if __name__ == '__main__':
    main()