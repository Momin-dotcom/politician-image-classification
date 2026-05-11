from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.model import predict
import time

app = FastAPI(
    title="Pakistani Politician Classifier",
    description="CNN-based facial recognition for 16 Pakistani public figures",
    version="1.0.0"
)

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time
START_TIME = time.time()


@app.get("/health")
def health():
    """Health check endpoint — used by CI/CD and monitoring"""
    return {
        "status": "ok",
        "message": "API is running",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a face image and get top 3 politician predictions.
    Accepts: image/jpeg, image/png
    Returns: filename + top3 predictions with confidence scores
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG/PNG accepted."
        )

    # Read file
    image_bytes = await file.read()

    # Validate not empty
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty."
        )

    # Run prediction
    predictions = predict(image_bytes)

    return {
        "filename": file.filename,
        "top3_predictions": predictions
    }


@app.get("/classes")
def get_classes():
    """Returns all 16 class names the model can predict"""
    from app.classes import CLASS_NAMES
    return {"total": len(CLASS_NAMES), "classes": CLASS_NAMES}
