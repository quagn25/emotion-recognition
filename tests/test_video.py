import os
import cv2
import numpy as np
import pytest
from app.services.detector import FaceDetector
from app.services.emotion_classifier import EmotionClassifier
from app.services.preprocessing import UnifiedPreprocessor
from app.services.video_service import VideoProcessor
from app.config import settings

def test_video_processing_synthetic(tmp_path):
    # Create a 5-frame dummy video
    input_video_path = str(tmp_path / "test_input.mp4")
    output_video_path = str(tmp_path / "test_output.mp4")

    w, h = 320, 240
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(input_video_path, fourcc, fps, (w, h))

    for i in range(5):
        frame = np.full((h, w, 3), i * 40, dtype=np.uint8)
        # Draw a synthetic face-like circle
        cv2.circle(frame, (160, 120), 40, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()

    assert os.path.exists(input_video_path)

    detector = FaceDetector(settings.model_yolo_path)
    classifier = EmotionClassifier(settings.model_cnn_path, settings.labels_path)
    preprocessor = UnifiedPreprocessor()
    processor = VideoProcessor(detector, classifier, preprocessor)

    summary = processor.process_video_file(input_video_path, output_video_path, frame_skip=2)

    assert os.path.exists(output_video_path)
    assert summary["processed_frames"] == 5
    assert summary["output_path"] == output_video_path
