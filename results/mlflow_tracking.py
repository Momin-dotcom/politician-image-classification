"""
MLflow Experiment Tracking for Politician Classification
Run this AFTER downloading model results from Google Drive.
This script re-logs results into your local MLflow dashboard for screenshots.
"""

import mlflow
import mlflow.pytorch
import json
import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "Pakistani_Politician_Classification"
RESULTS_PATH = "results"  # Local results folder (copy from Drive)

# Results from your Colab training (fill these in after training)
RESULTS = {
    "ResNet50": {
        "test_accuracy": 0.0,     # Fill in after training
        "test_precision": 0.0,
        "test_recall": 0.0,
        "test_f1": 0.0,
        "best_val_accuracy": 0.0,
        "num_epochs_trained": 0,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "pretrained": True,
        "image_size": 224,
        "num_classes": 16,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
    },
    "EfficientNetB0": {
        "test_accuracy": 0.0,     # Fill in after training
        "test_precision": 0.0,
        "test_recall": 0.0,
        "test_f1": 0.0,
        "best_val_accuracy": 0.0,
        "num_epochs_trained": 0,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "pretrained": True,
        "image_size": 224,
        "num_classes": 16,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
    }
}


def log_experiment(model_name, params_metrics):
    """Log a complete experiment run to MLflow."""
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    params = {k: v for k, v in params_metrics.items() 
              if k not in ['test_accuracy', 'test_precision', 'test_recall', 
                           'test_f1', 'best_val_accuracy', 'num_epochs_trained']}
    
    metrics = {
        'test_accuracy': params_metrics['test_accuracy'],
        'test_precision': params_metrics['test_precision'],
        'test_recall': params_metrics['test_recall'],
        'test_f1': params_metrics['test_f1'],
        'best_val_accuracy': params_metrics['best_val_accuracy'],
        'num_epochs_trained': params_metrics['num_epochs_trained'],
    }
    
    with mlflow.start_run(run_name=f"{model_name}_Final"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('model', model_name)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log result artifacts if they exist
        artifacts = [
            f'confusion_matrix_{model_name}.png',
            f'training_curves_{model_name}.png',
            'model_comparison_chart.png',
            'model_comparison.csv',
        ]
        
        for artifact in artifacts:
            path = os.path.join(RESULTS_PATH, artifact)
            if os.path.exists(path):
                mlflow.log_artifact(path)
                print(f"  Logged artifact: {artifact}")
            else:
                print(f"  Artifact not found (skip): {artifact}")
        
        print(f"✅ {model_name} logged to MLflow")
        print(f"   Test Accuracy: {metrics['test_accuracy']*100:.2f}%")


if __name__ == "__main__":
    print("="*50)
    print("MLflow Experiment Logging")
    print("="*50)
    
    for model_name, data in RESULTS.items():
        log_experiment(model_name, data)
    
    print("\n" + "="*50)
    print("All experiments logged!")
    print("Run this command to open the MLflow dashboard:")
    print("  mlflow ui")
    print("Then open: http://localhost:5000")
    print("="*50)