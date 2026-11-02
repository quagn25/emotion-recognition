import os
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
from app.config import settings

class FaceDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.model_yolo_path
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"[FaceDetector] Loading YOLOv8 Face from {self.model_path}...")
                self.model = YOLO(self.model_path)
                self.is_loaded = True
                print("[FaceDetector] YOLOv8 Face loaded successfully.")
            except Exception as e:
                print(f"[FaceDetector] Failed to load YOLOv8 Face: {e}")
                self.model = None
                self.is_loaded = False
        else:
            print(f"[FaceDetector] Model file not found at {self.model_path}")
            self.model = None
            self.is_loaded = False

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image.
        Returns list of (x1, y1, x2, y2, confidence).
        """
        if not self.is_loaded or self.model is None or image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        detected = []

        try:
            results = self.model(
                image,
                conf=settings.yolo_confidence,
                iou=settings.yolo_iou,
                verbose=False
            )

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 1.0

                    x1 = max(0, int(coords[0]))
                    y1 = max(0, int(coords[1]))
                    x2 = min(img_w, int(coords[2]))
                    y2 = min(img_h, int(coords[3]))

                    w = x2 - x1
                    h = y2 - y1

                    if w >= settings.min_face_size and h >= settings.min_face_size:
                        detected.append((x1, y1, x2, y2, conf))

        except Exception as e:
            print(f"[FaceDetector] Detection error: {e}")

        return detected
