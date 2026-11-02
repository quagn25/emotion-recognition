from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ImagePayload(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string (with or without data URL prefix)")
    model_type: Optional[str] = Field("cnn", description="Model architecture type (default: cnn)")
    session_id: Optional[str] = Field(None, description="Optional client session identifier for tracking & temporal smoothing")

class TopEmotion(BaseModel):
    emotion: str
    confidence: float

class FaceDetectionResult(BaseModel):
    box: List[int] = Field(..., description="[x, y, width, height] coordinates")
    emotion: str
    confidence: float
    top_3: List[TopEmotion] = Field(default_factory=list)
    track_id: Optional[int] = Field(None, description="Stable face track ID across frames")

class PredictResponse(BaseModel):
    status: str
    message: str = "success"
    face_count: int = 0
    data: List[FaceDetectionResult] = Field(default_factory=list)
    latency_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str
    cnn_loaded: bool
    yolo_loaded: bool
    device: str
    labels_count: int
    labels: Dict[int, str]

class ModelInfoResponse(BaseModel):
    model_type: str
    input_shape: List[int]
    labels: List[str]
    confidence_threshold: float
    yolo_confidence: float
    clahe_enabled: bool
