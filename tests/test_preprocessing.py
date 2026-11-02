import cv2
import numpy as np
import pytest
import base64
from app.services.preprocessing import UnifiedPreprocessor

@pytest.fixture
def preprocessor():
    return UnifiedPreprocessor()

def test_safe_crop_face_boundaries(preprocessor):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Face box near edge
    x1, y1, x2, y2 = 10, 10, 100, 100
    roi, box = preprocessor.safe_crop_face(img, x1, y1, x2, y2, padding=0.2, square=True)

    assert roi is not None
    assert roi.size > 0
    # Cropped coordinates must be non-negative and within image boundaries
    bx, by, bw, bh = box
    assert bx >= 0 and by >= 0
    assert bx + bw <= 640
    assert by + bh <= 480

def test_safe_crop_face_invalid_box(preprocessor):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    roi, box = preprocessor.safe_crop_face(img, 100, 100, 100, 100) # w=0, h=0
    assert roi is None

def test_preprocess_face_shape_and_range(preprocessor):
    face_roi = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess_face(face_roi)

    # Grayscale shape (48, 48, 1)
    assert tensor.shape == (48, 48, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

def test_decode_base64_valid_and_invalid(preprocessor):
    # Create a small valid test image
    test_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", test_img)
    valid_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    decoded = preprocessor.decode_base64_image(valid_b64)
    assert decoded is not None
    assert decoded.shape == (100, 100, 3)

    # Corrupt string
    with pytest.raises(ValueError):
        preprocessor.decode_base64_image("invalid_not_base64_garbage!@#$")

    with pytest.raises(ValueError):
        preprocessor.decode_base64_image("")
