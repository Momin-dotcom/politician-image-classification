import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = [
    "imran_khan", "nawaz_sharif", "asif_ali_zardari", "shehbaz_sharif",
    "bilawal_bhutto", "maryam_nawaz", "ch_nisar", "aitzaz_ahsan",
    "fazlur_rehman", "sirajul_haq", "pervez_musharraf", "raheel_sharif",
    "asim_munir", "shah_mahmood_qureshi", "khurshid_shah", "ahsan_iqbal"
]

# Using ResNet50 — swap this to whatever Member 1 used
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

# Save dummy weights
torch.save(model.state_dict(), "models/best_model.pth")
print(f"Dummy model saved with {len(CLASS_NAMES)} classes")
print("Classes:", CLASS_NAMES)