import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from app.config import settings
from app.services.detector import FaceDetector
from app.services.emotion_classifier import EmotionClassifier
from app.services.preprocessing import UnifiedPreprocessor
from app.services.smoothing import TemporalSmoother

EMOTION_COLORS = {
    "angry": (0, 0, 255),      # Red
    "contempt": (180, 105, 255),# Pink/Magenta (BGR)
    "disgust": (0, 140, 255),  # Orange
    "fear": (128, 0, 128),     # Purple
    "happy": (0, 255, 0),      # Green
    "neutral": (200, 200, 200),# Gray
    "sad": (255, 0, 0),        # Blue
    "surprise": (0, 255, 255), # Yellow
    "Unknown": (120, 120, 120) # Dark Gray
}

class VideoProcessor:
    def __init__(
        self,
        detector: FaceDetector,
        classifier: EmotionClassifier,
        preprocessor: UnifiedPreprocessor
    ):
        self.detector = detector
        self.classifier = classifier
        self.preprocessor = preprocessor

    def process_video_file(
        self,
        input_path: str,
        output_path: str,
        frame_skip: int = None
    ) -> Dict:
        """
        Processes a video file with face detection, emotion classification,
        temporal smoothing, and annotations.
        """
        skip = frame_skip or settings.video_frame_skip
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

        smoother = TemporalSmoother()
        last_faces_data = [] # Stores last detected face info for skipped frames
        frame_idx = 0
        total_faces_counted = 0
        emotion_summary = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run full detection & classification every `skip` frames
                if frame_idx % skip == 0 or len(last_faces_data) == 0:
                    raw_boxes = self.detector.detect_faces(frame)
                    face_rois = []
                    boxes_for_smoothing = []

                    for (x1, y1, x2, y2, det_conf) in raw_boxes:
                        roi, adjusted_box = self.preprocessor.safe_crop_face(frame, x1, y1, x2, y2)
                        if roi is not None:
                            try:
                                tensor = self.preprocessor.preprocess_face(roi)
                                face_rois.append(tensor)
                                boxes_for_smoothing.append(adjusted_box)
                            except Exception:
                                continue

                    if face_rois:
                        preds = self.classifier.predict_batch(face_rois)
                        prob_vectors = [p[3] for p in preds]

                        # Apply temporal smoothing
                        smoothed = smoother.update(
                            boxes_for_smoothing,
                            prob_vectors,
                            self.classifier.labels
                        )

                        last_faces_data = []
                        for box, (emotion, conf, _, track_id) in zip(boxes_for_smoothing, smoothed):
                            # Ensure box is bounded within video frame
                            clipped_x = max(0, min(width - 1, box[0]))
                            clipped_y = max(0, min(height - 1, box[1]))
                            clipped_w = max(10, min(width - clipped_x, box[2]))
                            clipped_h = max(10, min(height - clipped_y, box[3]))
                            last_faces_data.append({
                                "box": [clipped_x, clipped_y, clipped_w, clipped_h],
                                "emotion": emotion,
                                "confidence": conf,
                                "track_id": track_id
                            })
                            total_faces_counted += 1
                            emotion_summary[emotion] = emotion_summary.get(emotion, 0) + 1
                    else:
                        last_faces_data = []

                # Draw annotations on the frame (using current or last tracked faces)
                for item in last_faces_data:
                    bx, by, bw, bh = item["box"]
                    emotion = item["emotion"]
                    conf = item["confidence"]
                    track_id = item.get("track_id")
                    color = EMOTION_COLORS.get(emotion, (120, 120, 120))

                    # Bounding box
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

                    # Text label with track id and background
                    track_str = f"#{track_id + 1} " if track_id is not None else ""
                    label_text = f"{track_str}{emotion} ({int(conf * 100)}%)"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(
                        frame,
                        (bx, max(0, by - th - 10)),
                        (bx + tw + 6, by),
                        color,
                        -1
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        (bx + 3, max(15, by - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                out.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            out.release()

        # Validate output video file creation
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("Lỗi: Không thể xuất tệp video sau khi xử lý (kích thước tệp rỗng).")

        return {
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "fps": orig_fps,
            "width": width,
            "height": height,
            "total_face_instances": total_faces_counted,
            "emotion_summary": emotion_summary,
            "output_path": output_path
        }
