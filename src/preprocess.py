import os
import hashlib
from PIL import Image

RAW_DIR = "dataset/raw"
SIZE = (224, 224)

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def process():
    politicians = os.listdir(RAW_DIR)
    
    for politician in politicians:
        folder = os.path.join(RAW_DIR, politician)
        if not os.path.isdir(folder):
            continue
        
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        hashes = {}
        removed_corrupt = 0
        removed_duplicate = 0
        resized = 0
        
        for img_name in images:
            img_path = os.path.join(folder, img_name)
            
            # Corrupted check
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                os.remove(img_path)
                removed_corrupt += 1
                continue
            
            # Duplicate check
            h = get_hash(img_path)
            if h in hashes:
                os.remove(img_path)
                removed_duplicate += 1
                continue
            hashes[h] = img_name
            
            # Resize
            if img.size != SIZE:
                img = img.resize(SIZE)
                img.save(img_path)
                resized += 1
        
        remaining = len(os.listdir(folder))
        print(f"{politician}: removed {removed_corrupt} corrupt, {removed_duplicate} duplicates, resized {resized}, remaining {remaining}")

if __name__ == "__main__":
    print("Processing raw dataset...")
    process()
    print("Done!")