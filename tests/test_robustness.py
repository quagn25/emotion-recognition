import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _process_single_frame, smoother_manager
from app.config import settings
from app.services.preprocessing import UnifiedPreprocessor
from app.services.detector import FaceDetector
from app.services.emotion_classifier import EmotionClassifier
from app.services.video_service import VideoProcessor


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_tiny_face_rejection():
    """Khuôn mặt quá nhỏ (< min_face_size = 20) phải bị loại bỏ để tránh lỗi tỷ lệ."""
    preprocessor = UnifiedPreprocessor()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Box 10x10 < min_face_size
    roi, box = preprocessor.safe_crop_face(img, 50, 50, 60, 60)
    assert roi is None
    assert box[2] < settings.min_face_size or box[3] < settings.min_face_size


def test_corrupt_video_upload(client):
    """Tệp video chứa byte rác hoặc không thể giải mã phải trả về HTTP 400 Bad Request."""
    corrupt_bytes = b"CORRUPTED_NOT_A_REAL_VIDEO_HEADER_1234567890" * 50
    response = client.post(
        "/predict/video",
        files={"file": ("corrupt.mp4", corrupt_bytes, "video/mp4")}
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "lỗi" in detail.lower() or "giải mã" in detail.lower() or "không hợp lệ" in detail.lower()


def test_empty_video_zero_frames(client, tmp_path):
    """Tệp video rỗng (0 frames) phải trả về HTTP 400 Bad Request."""
    empty_video_path = tmp_path / "empty.mp4"
    # Create empty 0-byte file with .mp4 extension
    empty_video_path.write_bytes(b"")

    with open(empty_video_path, "rb") as f:
        response = client.post(
            "/predict/video",
            files={"file": ("empty.mp4", f.read(), "video/mp4")}
        )
    assert response.status_code == 400


def test_concurrent_sessions_smoother_isolation():
    """Hai session webcam hoạt động đồng thời phải có bộ đệm làm mịn hoàn toàn độc lập."""
    smoother_manager.reset_all()

    smoother_a = smoother_manager.get_smoother("session_alpha")
    smoother_b = smoother_manager.get_smoother("session_beta")

    assert smoother_a is not smoother_b

    # Update session A with predominantly Angry prob vector
    boxes = [[50, 50, 100, 100]]
    labels = {0: "angry", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}

    probs_angry = [np.array([0.9, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0])]
    res_a = smoother_a.update(boxes, probs_angry, labels)

    # Update session B with predominantly Happy prob vector
    probs_happy = [np.array([0.01, 0.01, 0.01, 0.01, 0.92, 0.02, 0.01, 0.01])]
    res_b = smoother_b.update(boxes, probs_happy, labels)

    emotion_a = res_a[0][0]
    emotion_b = res_b[0][0]

    assert emotion_a == "angry"
    assert emotion_b == "happy"


def test_session_manager_lru_capacity_capping():
    """Kiểm tra SessionSmootherManager không bị leak bộ nhớ khi có nhiều session (LRU limit)."""
    smoother_manager.reset_all()

    # Create 110 sessions
    for i in range(110):
        smoother_manager.get_smoother(f"client_session_{i}")

    # Maximum allowed sessions is 100
    assert len(smoother_manager._sessions) <= 100


def test_video_path_traversal_protection(client):
    """Kiểm tra ngăn chặn tấn công Path Traversal tại /videos/{filename}."""
    # Attempt directory traversal
    response = client.get("/videos/../../etc/passwd")
    assert response.status_code in (400, 404)

    response2 = client.get("/videos/..%2F..%2Fmain.py")
    assert response2.status_code in (400, 404)


def test_processed_video_integrity(tmp_path):
    """Kiểm tra tệp video xuất ra sau khi xử lý có thể mở và đọc khung hình bình thường."""
    input_video = str(tmp_path / "valid_input.mp4")
    output_video = str(tmp_path / "valid_output.mp4")

    w, h, fps = 320, 240, 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(input_video, fourcc, fps, (w, h))

    # 4 frames with a circle
    for i in range(4):
        f = np.full((h, w, 3), 100 + i * 20, dtype=np.uint8)
        cv2.circle(f, (160, 120), 35, (255, 255, 255), -1)
        writer.write(f)
    writer.release()

    detector = FaceDetector()
    classifier = EmotionClassifier()
    preprocessor = UnifiedPreprocessor()
    video_proc = VideoProcessor(detector, classifier, preprocessor)

    summary = video_proc.process_video_file(input_video, output_video, frame_skip=1)

    assert os.path.exists(output_video)
    assert os.path.getsize(output_video) > 500
    assert summary["processed_frames"] == 4

    # Verify that the output video can be read by cv2.VideoCapture
    cap = cv2.VideoCapture(output_video)
    assert cap.isOpened()
    ret, frame = cap.read()
    cap.release()
    assert ret is True
    assert frame is not None
