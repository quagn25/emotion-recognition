import numpy as np
from typing import Dict, List, Tuple, Optional
from app.config import settings

class TemporalSmoother:
    def __init__(self, alpha: float = None, max_age: int = 15, iou_threshold: float = 0.3):
        self.alpha = alpha if alpha is not None else settings.temporal_smoothing_alpha
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        # Stores tracked faces: {track_id: {"box": [x, y, w, h], "probs": np.ndarray, "age": int}}
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0

    def _compute_iou(self, boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        unionArea = boxAArea + boxBArea - interArea

        if unionArea <= 0:
            return 0.0
        return interArea / unionArea

    def update(
        self,
        boxes: List[List[int]],
        prob_vectors: List[np.ndarray],
        labels_map: Dict[int, str]
    ) -> List[Tuple[str, float, np.ndarray, int]]:
        """
        Match detected boxes with existing tracks, apply Exponential Moving Average (EMA)
        smoothing to probability vectors, and update state.
        Returns list of (smoothed_emotion, smoothed_confidence, smoothed_probs, track_id).
        """
        # Increment age for existing tracks
        for track in self.tracks.values():
            track["age"] += 1

        results = []
        matched_tracks = set()

        for box, probs in zip(boxes, prob_vectors):
            best_iou = 0.0
            best_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self._compute_iou(box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None and best_iou >= self.iou_threshold:
                # Update existing track with EMA
                prev_probs = self.tracks[best_id]["probs"]
                smoothed_probs = self.alpha * probs + (1.0 - self.alpha) * prev_probs
                self.tracks[best_id]["probs"] = smoothed_probs
                self.tracks[best_id]["box"] = box
                self.tracks[best_id]["age"] = 0
                matched_tracks.add(best_id)
            else:
                # Create new track
                best_id = self.next_id
                self.next_id += 1
                smoothed_probs = probs.copy()
                self.tracks[best_id] = {
                    "box": box,
                    "probs": smoothed_probs,
                    "age": 0
                }
                matched_tracks.add(best_id)

            max_idx = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[max_idx])

            if confidence < settings.confidence_threshold:
                emotion = "Unknown"
            else:
                emotion = labels_map.get(max_idx, "Unknown")

            results.append((emotion, confidence, smoothed_probs, best_id))

        # Remove dead tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}

        return results

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
