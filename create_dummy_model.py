import os
import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = [
    "imran_khan", "nawaz_sharif", "asif_ali_zardari", "shehbaz_sharif",
    "bilawal_bhutto", "maryam_nawaz", "ch_nisar", "aitzaz_ahsan",
    "fazlur_rehman", "sirajul_haq", "pervez_musharraf", "raheel_sharif",
    "asim_munir", "shah_mahmood_qureshi", "khurshid_shah", "ahsan_iqbal"
]

MODEL_PATH = "models/best_model.pth"

if os.path.exists(MODEL_PATH):
    print(f"Real model already exists at {MODEL_PATH}, skipping dummy creation.")
else:
    os.makedirs("models", exist_ok=True)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Dummy model created at {MODEL_PATH}")