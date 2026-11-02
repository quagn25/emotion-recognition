import base64
import json
import os
import cv2
import numpy as np
from typing import Tuple, List, Optional
from app.config import settings

class UnifiedPreprocessor:
    def __init__(self, config_path: Optional[str] = None):
        self.target_size = (48, 48)
        self.color_mode = "grayscale"
        self.use_clahe = True
        self.clip_limit = 2.0
        self.tile_grid_size = (8, 8)
        self.crop_padding = 0.15
        self.square_crop = True

        config_file = config_path or settings.preprocessing_config_path
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.target_size = tuple(cfg.get("target_size", [48, 48]))
                    self.color_mode = cfg.get("color_mode", "grayscale")
                    self.use_clahe = cfg.get("use_clahe", True)
                    self.clip_limit = float(cfg.get("clahe_clip_limit", 2.0))
                    grid = cfg.get("clahe_tile_grid_size") or cfg.get("clahe_grid_size") or [8, 8]
                    self.tile_grid_size = tuple(grid)
                    self.crop_padding = float(cfg.get("crop_padding", 0.15))
                    self.square_crop = cfg.get("square_crop", True)
            except Exception as e:
                print(f"[UnifiedPreprocessor] Error loading config: {e}. Using defaults.")

        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def decode_base64_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 string (with or without data URL prefix) to BGR OpenCV image."""
        if not image_base64 or not isinstance(image_base64, str):
            raise ValueError("Empty or invalid base64 string provided.")

        if "," in image_base64:
            encoded_data = image_base64.split(",", 1)[1]
        else:
            encoded_data = image_base64

        try:
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError("Decoded image buffer is empty or corrupted.")
            return img
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    def safe_crop_face(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        padding: Optional[float] = None,
        square: Optional[bool] = None
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Crop face with proportional padding and square expansion to prevent aspect ratio distortion.
        Returns (face_roi, [x, y, w, h]).
        """
        img_h, img_w = image.shape[:2]
        pad = padding if padding is not None else self.crop_padding
        use_square = square if square is not None else self.square_crop

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return None, [0, 0, 0, 0]

        # Calculate padding
        pad_w = int(w * pad)
        pad_h = int(h * pad)

        new_x1 = x1 - pad_w
        new_x2 = x2 + pad_w
        new_y1 = y1 - pad_h
        new_y2 = y2 + pad_h

        # Square expansion if configured
        if use_square:
            box_w = new_x2 - new_x1
            box_h = new_y2 - new_y1
            max_side = max(box_w, box_h)
            center_x = (new_x1 + new_x2) // 2
            center_y = (new_y1 + new_y2) // 2

            new_x1 = center_x - max_side // 2
            new_x2 = new_x1 + max_side
            new_y1 = center_y - max_side // 2
            new_y2 = new_y1 + max_side

        # Clip strictly within image boundaries
        crop_x1 = max(0, new_x1)
        crop_y1 = max(0, new_y1)
        crop_x2 = min(img_w, new_x2)
        crop_y2 = min(img_h, new_y2)

        final_w = crop_x2 - crop_x1
        final_h = crop_y2 - crop_y1

        if final_w < settings.min_face_size or final_h < settings.min_face_size:
            return None, [int(crop_x1), int(crop_y1), int(final_w), int(final_h)]

        roi = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if roi.size == 0:
            return None, [int(crop_x1), int(crop_y1), int(final_w), int(final_h)]

        return roi, [int(crop_x1), int(crop_y1), int(final_w), int(final_h)]

    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Preprocess face ROI: Grayscale -> CLAHE -> Resize -> Normalize [0, 1].
        Returns tensor ready for model input with shape (target_h, target_w, channels).
        """
        if face_roi is None or face_roi.size == 0:
            raise ValueError("Invalid face ROI passed to preprocess_face.")

        if self.color_mode == "grayscale":
            if len(face_roi.shape) == 3 and face_roi.shape[-1] == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            elif len(face_roi.shape) == 3 and face_roi.shape[-1] == 1:
                gray = face_roi.squeeze(-1)
            else:
                gray = face_roi

            if self.use_clahe:
                processed = self.clahe.apply(gray)
            else:
                processed = gray

            resized = cv2.resize(processed, self.target_size)
            normalized = resized.astype(np.float32) / 255.0
            tensor = np.expand_dims(normalized, axis=-1) # (48, 48, 1)
        else:
            # RGB mode
            if len(face_roi.shape) == 2:
                rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            elif len(face_roi.shape) == 3 and face_roi.shape[-1] == 3:
                rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                rgb = face_roi

            resized = cv2.resize(rgb, self.target_size)
            tensor = resized.astype(np.float32) / 255.0

        return tensor

preprocessor = UnifiedPreprocessor()
