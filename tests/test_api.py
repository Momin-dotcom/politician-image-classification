import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────

def make_fake_image(format="JPEG"):
    """Creates a valid in-memory image for testing"""
    img = Image.new("RGB", (224, 224), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


# ── Health Tests ──────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_has_uptime():
    response = client.get("/health")
    assert "uptime_seconds" in response.json()


# ── Classes Tests ─────────────────────────────────────────

def test_classes_returns_200():
    response = client.get("/classes")
    assert response.status_code == 200


def test_classes_returns_16():
    response = client.get("/classes")
    assert response.json()["total"] == 16


def test_classes_contains_imran_khan():
    response = client.get("/classes")
    assert "imran_khan" in response.json()["classes"]


# ── Predict Tests ─────────────────────────────────────────

def test_predict_returns_200():
    img = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200


def test_predict_returns_top3():
    img = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    data = response.json()
    assert "top3_predictions" in data
    assert len(data["top3_predictions"]) == 3


def test_predict_has_class_and_confidence():
    img = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    prediction = response.json()["top3_predictions"][0]
    assert "class" in prediction
    assert "confidence" in prediction


def test_predict_confidence_is_percentage():
    img = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    for pred in response.json()["top3_predictions"]:
        assert 0.0 <= pred["confidence"] <= 100.0


def test_predict_class_is_valid():
    """Predicted class must be one of our 16 known classes"""
    from app.classes import CLASS_NAMES
    img = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    for pred in response.json()["top3_predictions"]:
        assert pred["class"] in CLASS_NAMES


def test_predict_returns_filename():
    img = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.json()["filename"] == "test.jpg"


def test_predict_png_also_works():
    img = make_fake_image(format="PNG")
    response = client.post(
        "/predict",
        files={"file": ("test.png", img, "image/png")}
    )
    assert response.status_code == 200


# ── Invalid Input Tests ───────────────────────────────────

def test_invalid_file_type_returns_422():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 422


def test_empty_file_returns_422():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 422


def test_predict_no_file_returns_422():
    response = client.post("/predict")
    assert response.status_code == 422
