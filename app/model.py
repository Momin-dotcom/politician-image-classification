import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from app.classes import CLASS_NAMES

DEVICE = torch.device("cpu")
MODEL_PATH = "models/best_model.pth"


def load_model():
    """
    IMPORTANT FOR MEMBER 1:
    If you used a different architecture (EfficientNet, VGG, etc),
    replace models.resnet50 here with the correct model.
    The fc layer output size must match len(CLASS_NAMES) = 16
    """
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully on {DEVICE}")
    print(f"Classes: {len(CLASS_NAMES)}")
    return model


# Load once at startup
model = load_model()

# Must match whatever transforms Member 1 used during training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image_bytes: bytes):
    """
    Takes raw image bytes, returns top 3 predictions with confidence scores.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top3_probs, top3_indices = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top3_probs, top3_indices):
        results.append({
            "class": CLASS_NAMES[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results
