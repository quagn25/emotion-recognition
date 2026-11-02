import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "cnn_loaded" in data

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "Custom CNN + CLAHE & YOLOv8-Face"
    assert len(data["labels"]) == 8

def test_predict_frame_valid(client):
    # Synthesize dummy black frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    response = client.post("/predict/frame", json={
        "image_base64": b64_str,
        "model_type": "cnn",
        "session_id": "test_session_123"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "face_count" in data
    assert "data" in data
    assert "latency_ms" in data

def test_predict_frame_invalid_base64(client):
    response = client.post("/predict/frame", json={"image_base64": "invalid_data_string", "model_type": "cnn"})
    assert response.status_code == 400

def test_session_reset(client):
    response = client.post("/session/reset", json={"session_id": "test_session_123"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_predict_image_no_face(client):
    # Black image with no faces
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    response = client.post(
        "/predict/image",
        files={"file": ("blank.jpg", buf.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["face_count"] == 0
    assert len(data["data"]) == 0
    assert "annotated_image" in data

def test_predict_image_invalid_type(client):
    response = client.post(
        "/predict/image",
        files={"file": ("test.txt", b"Hello world text file", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_video_invalid_format(client):
    response = client.post(
        "/predict/video",
        files={"file": ("fake_video.txt", b"not a video content", "text/plain")}
    )
    assert response.status_code == 400
