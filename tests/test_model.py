import os
import json
import numpy as np
import pytest
from app.services.emotion_classifier import EmotionClassifier
from app.config import settings

def test_labels_loading():
    classifier = EmotionClassifier()
    assert len(classifier.labels) == 8
    assert 0 in classifier.labels
    assert "happy" in classifier.labels.values()
    assert "contempt" in classifier.labels.values()
    expected_labels = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for idx, name in enumerate(expected_labels):
        assert classifier.labels[idx] == name

def test_classifier_contract():
    if not os.path.exists(settings.model_cnn_path):
        pytest.skip(f"Model file {settings.model_cnn_path} not found for contract testing.")

    classifier = EmotionClassifier()
    assert classifier.is_loaded

    # Generate dummy batch of 2 faces: shape (2, 48, 48, 1)
    dummy_faces = [np.random.rand(48, 48, 1).astype(np.float32) for _ in range(2)]
    results = classifier.predict_batch(dummy_faces)

    assert len(results) == 2
    for emotion, conf, top_3, probs in results:
        assert isinstance(emotion, str)
        assert 0.0 <= conf <= 1.0
        assert len(top_3) <= 3
        assert len(probs) == 8
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
