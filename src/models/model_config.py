"""
model_config.py
Central configuration for all model training.
"""

MODEL_CONFIG = {
    # Dataset
    'dataset_path': 'dataset',
    'num_classes': 16,
    'image_size': 224,

    # Training
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 7,

    # Fine-tuning
    'finetune_epochs': 10,
    'finetune_lr_factor': 0.1,

    # Augmentation
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std':  [0.229, 0.224, 0.225],

    # Paths
    'model_save_path': 'saved_models',
    'results_path': 'results',

    # MLflow
    'mlflow_experiment': 'Pakistani_Politician_Classification',
}