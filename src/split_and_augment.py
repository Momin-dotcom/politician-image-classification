import os
import shutil
import random
from PIL import Image, ImageEnhance
import numpy as np

RAW_DIR = "dataset/raw"
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

def split_dataset():
    politicians = os.listdir(RAW_DIR)
    
    for politician in politicians:
        src = os.path.join(RAW_DIR, politician)
        if not os.path.isdir(src):
            continue
            
        images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            dest = os.path.join(f"dataset/{split}", politician)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(src, img), os.path.join(dest, img))
        
        print(f"{politician}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

def augment_image(img_path, dest_dir, base_name):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    
    augmentations = []
    
    # Rotation
    augmentations.append(img.rotate(15))
    augmentations.append(img.rotate(-15))
    
    # Flipping
    augmentations.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Brightness
    augmentations.append(ImageEnhance.Brightness(img).enhance(1.3))
    augmentations.append(ImageEnhance.Brightness(img).enhance(0.7))
    
    # Cropping
    w, h = img.size
    crop = img.crop((20, 20, w-20, h-20)).resize((224, 224))
    augmentations.append(crop)
    
    # Zoom
    zoom = img.crop((30, 30, w-30, h-30)).resize((224, 224))
    augmentations.append(zoom)
    
    for i, aug in enumerate(augmentations):
        aug.save(os.path.join(dest_dir, f"aug_{i}_{base_name}"))

def augment_train():
    politicians = os.listdir(TRAIN_DIR)
    
    for politician in politicians:
        train_path = os.path.join(TRAIN_DIR, politician)
        if not os.path.isdir(train_path):
            continue
            
        images = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(train_path, img_name)
            try:
                augment_image(img_path, train_path, img_name)
            except Exception as e:
                print(f"Error augmenting {img_name}: {e}")
        
        total = len(os.listdir(train_path))
        print(f"{politician}: {total} images after augmentation")

if __name__ == "__main__":
    print("Splitting dataset...")
    split_dataset()
    print("\nAugmenting training data...")
    augment_train()
    print("\nDone!")