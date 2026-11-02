import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch  # Preload torch C10 DLL before scipy/sklearn to prevent WinError 1114
except ImportError:
    pass
from pydantic import BaseModel

class AppConfig(BaseModel):
    # GPU Configuration (Default: False -> Uses CPU unless USE_GPU=true)
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")

    # Locked Model Package Paths (All models permanently located in models/ directory)
    models_dir: str = os.getenv("MODELS_DIR", "models")
    model_cnn_path: str = os.getenv("MODEL_CNN_PATH", os.path.join("models", "emotion_model.keras"))
    model_yolo_path: str = os.getenv("MODEL_YOLO_PATH", os.path.join("models", "yolov8n-face.pt"))
    labels_path: str = os.getenv("LABELS_PATH", os.path.join("models", "labels.json"))
    preprocessing_config_path: str = os.getenv("PREPROCESSING_CONFIG_PATH", os.path.join("models", "preprocessing_config.json"))
    temp_dir: str = os.getenv("TEMP_DIR", "temp_files")

    # Detection & Inference Thresholds
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    yolo_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.40"))
    yolo_iou: float = float(os.getenv("YOLO_IOU", "0.45"))
    min_face_size: int = int(os.getenv("MIN_FACE_SIZE", "20"))
    crop_padding: float = float(os.getenv("CROP_PADDING", "0.15"))

    # Smoothing & Batching
    temporal_smoothing_alpha: float = float(os.getenv("TEMPORAL_SMOOTHING_ALPHA", "0.65"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    video_frame_skip: int = int(os.getenv("VIDEO_FRAME_SKIP", "2"))
    session_timeout_seconds: int = int(os.getenv("SESSION_TIMEOUT_SECONDS", "600"))

settings = AppConfig()

# Configure CUDA device visibility based on use_gpu
if not settings.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # If USE_GPU=true, ensure CUDA_VISIBLE_DEVICES is not hiding GPUs
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        del os.environ["CUDA_VISIBLE_DEVICES"]
