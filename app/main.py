import os
import time
import uuid
import shutil
import cv2
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.schemas import (
    ImagePayload,
    PredictResponse,
    FaceDetectionResult,
    HealthResponse,
    ModelInfoResponse,
    TopEmotion
)
from app.services.detector import FaceDetector
from app.services.emotion_classifier import EmotionClassifier
from app.services.preprocessing import UnifiedPreprocessor
from app.services.smoothing import TemporalSmoother
from app.services.video_service import VideoProcessor, EMOTION_COLORS
import base64

# Global service instances
detector: Optional[FaceDetector] = None
classifier: Optional[EmotionClassifier] = None
preprocessor = UnifiedPreprocessor()
video_processor: Optional[VideoProcessor] = None

class SessionSmootherManager:
    """
    Manages isolated temporal smoothers per client session to avoid cross-talk.
    Includes timeout cleanup and LRU capacity capping to prevent memory leaks.
    """
    def __init__(self, timeout_seconds: int = 600, max_sessions: int = 100):
        self._sessions: dict = {}
        self._timeout = timeout_seconds
        self._max_sessions = max_sessions

    def get_smoother(self, session_id: Optional[str]) -> TemporalSmoother:
        now = time.time()
        # Periodic cleanup of idle sessions (> timeout)
        expired = [s for s, (_, t) in self._sessions.items() if now - t > self._timeout]
        for s in expired:
            self._sessions.pop(s, None)

        sid = session_id or "default_session"
        if sid not in self._sessions:
            # If at capacity, evict oldest session (LRU)
            if len(self._sessions) >= self._max_sessions:
                oldest_sid = min(self._sessions.keys(), key=lambda k: self._sessions[k][1])
                self._sessions.pop(oldest_sid, None)
            self._sessions[sid] = (TemporalSmoother(), now)
        else:
            smoother, _ = self._sessions[sid]
            self._sessions[sid] = (smoother, now)
        return self._sessions[sid][0]

    def reset_session(self, session_id: Optional[str]):
        sid = session_id or "default_session"
        if sid in self._sessions:
            self._sessions[sid][0].reset()
            self._sessions.pop(sid, None)

    def reset_all(self):
        self._sessions.clear()

smoother_manager = SessionSmootherManager()

def _cleanup_old_temp_files(max_age_seconds: int = 3600):
    """Clean up temp files older than max_age_seconds."""
    try:
        now = time.time()
        for root, _, files in os.walk(settings.temp_dir):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    if now - os.path.getmtime(fp) > max_age_seconds:
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
    except Exception as e:
        print(f"[TempCleanup] Error cleaning files: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, classifier, video_processor
    print("--- [Lifespan] Initializing Emotion Recognition Services ---")
    print(f"-> Target Device: {'GPU' if settings.use_gpu else 'CPU'}")

    # Load Models from locked models/ directory
    detector = FaceDetector(settings.model_yolo_path)
    classifier = EmotionClassifier(settings.model_cnn_path, settings.labels_path)
    video_processor = VideoProcessor(detector, classifier, preprocessor)

    # Ensure temp directory exists
    os.makedirs(settings.temp_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.temp_dir, "outputs"), exist_ok=True)
    _cleanup_old_temp_files(max_age_seconds=3600)

    print("--- [Lifespan] All Services Initialized Successfully ---")
    yield
    print("--- [Lifespan] Shutting Down Services ---")
    smoother_manager.reset_all()

app = FastAPI(
    title="Emotion Recognition System API",
    description="Real-time Facial Emotion Recognition combining YOLOv8-Face & Custom CNN with CLAHE",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration: explicit origins + regex to avoid wildcard credentials rejection
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the single-page application frontend."""
    index_file = "index.html"
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    return HTMLResponse("<h2>index.html not found</h2>", status_code=404)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint providing subsystem statuses."""
    device_info = "GPU" if settings.use_gpu else "CPU"
    cnn_ok = classifier.is_loaded if classifier else False
    yolo_ok = detector.is_loaded if detector else False

    status_str = "healthy" if (cnn_ok and yolo_ok) else "degraded"

    return HealthResponse(
        status=status_str,
        cnn_loaded=cnn_ok,
        yolo_loaded=yolo_ok,
        device=device_info,
        labels_count=len(classifier.labels) if classifier else 0,
        labels=classifier.labels if classifier else {}
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Return model architecture and configuration metadata."""
    return ModelInfoResponse(
        model_type="Custom CNN + CLAHE & YOLOv8-Face",
        input_shape=list(preprocessor.target_size) + [1 if preprocessor.color_mode == "grayscale" else 3],
        labels=list(classifier.labels.values()) if classifier else [],
        confidence_threshold=settings.confidence_threshold,
        yolo_confidence=settings.yolo_confidence,
        clahe_enabled=preprocessor.use_clahe
    )

def _process_single_frame(
    img: np.ndarray,
    apply_smoothing: bool = False,
    smoother: Optional[TemporalSmoother] = None
) -> List[FaceDetectionResult]:
    """Internal synchronous function to detect and classify emotions in a frame."""
    if detector is None or classifier is None:
        return []

    # 1. Detect faces using YOLO (with downscale optimization for speed on large frames)
    img_h, img_w = img.shape[:2]
    max_det_dim = 640
    if max(img_h, img_w) > max_det_dim:
        scale = max_det_dim / float(max(img_h, img_w))
        det_w, det_h = int(img_w * scale), int(img_h * scale)
        det_img = cv2.resize(img, (det_w, det_h), interpolation=cv2.INTER_AREA)
        raw_boxes_scaled = detector.detect_faces(det_img)
        raw_boxes = []
        inv_scale = 1.0 / scale
        for (x1, y1, x2, y2, det_conf) in raw_boxes_scaled:
            raw_boxes.append((
                int(x1 * inv_scale),
                int(y1 * inv_scale),
                int(x2 * inv_scale),
                int(y2 * inv_scale),
                det_conf
            ))
    else:
        raw_boxes = detector.detect_faces(img)

    if not raw_boxes:
        return []

    # 2. Crop faces with padding and square expansion
    face_rois = []
    valid_boxes = []

    for (x1, y1, x2, y2, det_conf) in raw_boxes:
        roi, adjusted_box = preprocessor.safe_crop_face(img, x1, y1, x2, y2)
        if roi is not None:
            try:
                tensor = preprocessor.preprocess_face(roi)
                face_rois.append(tensor)
                valid_boxes.append(adjusted_box)
            except Exception:
                continue

    if not face_rois:
        return []

    # 3. Batch prediction
    preds = classifier.predict_batch(face_rois)

    results = []
    if preds and len(preds) == len(valid_boxes):
        if apply_smoothing and smoother is not None:
            prob_vectors = [p[3] for p in preds]
            smoothed = smoother.update(valid_boxes, prob_vectors, classifier.labels)
            for (box, (emotion, conf, _, track_id), (_, _, top_3, _)) in zip(valid_boxes, smoothed, preds):
                results.append(FaceDetectionResult(
                    box=box,
                    emotion=emotion,
                    confidence=conf,
                    top_3=top_3,
                    track_id=track_id
                ))
        else:
            for idx, (box, (emotion, conf, top_3, _)) in enumerate(zip(valid_boxes, preds)):
                results.append(FaceDetectionResult(
                    box=box,
                    emotion=emotion,
                    confidence=conf,
                    top_3=top_3,
                    track_id=idx
                ))
    else:
        # Fallback: if classifier failed or is offline, report Unknown with 0.0 confidence (NO FAKE CONFIDENCE)
        for idx, box in enumerate(valid_boxes):
            results.append(FaceDetectionResult(
                box=box,
                emotion="Unknown",
                confidence=0.0,
                top_3=[],
                track_id=idx
            ))

    return results

@app.post("/predict", response_model=PredictResponse)
@app.post("/predict/frame", response_model=PredictResponse)
async def predict_frame(payload: ImagePayload):
    """
    Real-time webcam frame inference endpoint with safe base64 decoding
    and session-isolated temporal smoothing with track IDs.
    """
    start_time = time.time()

    if not payload.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_base64 field is empty or missing."
        )

    # Decode image in threadpool to avoid blocking
    try:
        img = await run_in_threadpool(preprocessor.decode_base64_image, payload.image_base64)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Base64 image decoding failed: {str(e)}"
        )

    # Fetch session-isolated smoother
    smoother = smoother_manager.get_smoother(payload.session_id)

    # Process frame with session smoother
    results = await run_in_threadpool(_process_single_frame, img, True, smoother)
    latency = (time.time() - start_time) * 1000.0

    return PredictResponse(
        status="success",
        message="Face detection and emotion prediction complete",
        face_count=len(results),
        data=results,
        latency_ms=round(latency, 2)
    )

@app.post("/session/reset")
async def reset_session(payload: Optional[dict] = None):
    """Reset temporal smoothing state for a specific client session or all."""
    sid = payload.get("session_id") if payload else None
    smoother_manager.reset_session(sid)
    return {
        "status": "success",
        "message": f"Session '{sid or 'default'}' smoother reset successfully."
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Multipart image upload endpoint with file size limit, mime verification,
    and single-pass base64 encoding.
    """
    start_time = time.time()

    # 1. Validate file extension and content type
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    is_valid_type = (file.content_type in ALLOWED_IMAGE_TYPES) or (file.filename and file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Định dạng tệp không hợp lệ. Hệ thống chỉ hỗ trợ JPEG, PNG, WEBP, BMP."
        )

    # 2. Enforce file size limit
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi đọc tệp tải lên: {str(e)}"
        )

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Kích thước tệp ({len(contents) / (1024 * 1024):.1f}MB) vượt quá giới hạn cho phép {settings.max_upload_size_mb}MB."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp ảnh tải lên rỗng (0 bytes)."
        )

    # 3. Decode image and validate dimensions
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0 or img.shape[0] < 20 or img.shape[1] < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể giải mã dữ liệu ảnh hoặc kích thước ảnh quá nhỏ (tối thiểu 20x20 pixel)."
        )

    # 4. Run detection and classification (no temporal smoothing for static image)
    results = await run_in_threadpool(_process_single_frame, img, False, None)

    # 5. Annotate image directly (single encoding pass)
    annotated = img.copy()
    for res in results:
        bx, by, bw, bh = res.box
        color = EMOTION_COLORS.get(res.emotion, (120, 120, 120))

        # Box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 3)

        # Label
        text = f"{res.emotion} ({int(res.confidence * 100)}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (bx, max(0, by - th - 10)), (bx + tw + 6, by), color, -1)
        cv2.putText(annotated, text, (bx + 3, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Single base64 encode
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_data_url = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    latency = (time.time() - start_time) * 1000.0

    return {
        "status": "success",
        "message": "Phân tích ảnh thành công" if len(results) > 0 else "Không tìm thấy khuôn mặt nào trong ảnh.",
        "face_count": len(results),
        "data": [r.dict() for r in results],
        "annotated_image": annotated_data_url,
        "latency_ms": round(latency, 2)
    }

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Multipart video upload endpoint with size check and auto-cleanup.
    """
    if not file.content_type.startswith("video/") and not (file.filename and file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp video không hợp lệ (hỗ trợ .mp4, .avi, .mov, .mkv)."
        )

    # Enforce video upload size limit
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    unique_id = uuid.uuid4().hex
    input_ext = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    if not input_ext:
        input_ext = ".mp4"

    temp_input_path = os.path.join(settings.temp_dir, f"input_{unique_id}{input_ext}")
    output_filename = f"annotated_{unique_id}.mp4"
    temp_output_path = os.path.join(settings.temp_dir, "outputs", output_filename)

    # Periodic cleanup of old temp files (> 1 hour)
    _cleanup_old_temp_files(max_age_seconds=3600)

    # Save uploaded video with chunked size check
    bytes_written = 0
    try:
        with open(temp_input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Dung lượng video ({bytes_written / (1024 * 1024):.1f}MB) vượt quá giới hạn {settings.max_upload_size_mb}MB."
                    )
                buffer.write(chunk)
    except HTTPException:
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except Exception: pass
        raise
    except Exception as e:
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except Exception: pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi lưu trữ tệp video: {str(e)}"
        )

    # Validate actual video content (decodability and non-zero frames)
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        cap.release()
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except Exception: pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp video bị lỗi hoặc định dạng không thể giải mã (codec không hợp lệ)."
        )
    ret, first_frame = cap.read()
    cap.release()
    if not ret or first_frame is None or first_frame.size == 0:
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except Exception: pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp video rỗng (không chứa khung hình hợp lệ nào)."
        )

    # Process video in threadpool
    try:
        summary = await run_in_threadpool(
            video_processor.process_video_file,
            temp_input_path,
            temp_output_path
        )
    except Exception as e:
        if os.path.exists(temp_output_path):
            try: os.remove(temp_output_path)
            except Exception: pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Xử lý video thất bại: {str(e)}"
        )
    finally:
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except Exception: pass

    return {
        "status": "success",
        "message": "Video processed successfully",
        "video_url": f"/videos/{output_filename}",
        "summary": summary
    }

@app.get("/videos/{filename}")
async def get_processed_video(filename: str):
    """Download or stream processed video with strict path traversal protection."""
    safe_filename = os.path.basename(filename)
    if safe_filename != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên tệp không hợp lệ."
        )

    outputs_dir = os.path.realpath(os.path.join(settings.temp_dir, "outputs"))
    video_path = os.path.realpath(os.path.join(outputs_dir, safe_filename))

    if not video_path.startswith(outputs_dir) or not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested video file does not exist."
        )
    return FileResponse(video_path, media_type="video/mp4", filename=safe_filename)

# Local testing function for webcam opencv loop
def process_webcam_frame(frame):
    """Direct OpenCV frame processing helper for local testing."""
    if detector is None or classifier is None:
        return frame
    results = _process_single_frame(frame, apply_smoothing=True)
    for res in results:
        bx, by, bw, bh = res.box
        color = EMOTION_COLORS.get(res.emotion, (0, 255, 0))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
        text = f"{res.emotion} ({int(res.confidence * 100)}%)"
        cv2.putText(frame, text, (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
