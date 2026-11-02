import os
import io
import json
import zipfile
import h5py
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from app.config import settings
from app.schemas import TopEmotion

def build_improved_cnn_architecture(num_classes: int = 8) -> tf.keras.Model:
    inputs = tf.keras.layers.Input((48, 48, 1))
    l2_reg = 1e-4

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 4
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    # GAP & Dense
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Improved_CNN_GAP")

class EmotionClassifier:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.model_path = model_path or settings.model_cnn_path
        self.labels_path = labels_path or settings.labels_path
        self.model = None
        self.labels = {}
        self.is_loaded = False
        self._load_labels()
        self._load_model()

    def _load_labels(self):
        default_labels = {
            0: "angry", 1: "contempt", 2: "disgust", 3: "fear",
            4: "happy", 5: "neutral", 6: "sad", 7: "surprise"
        }
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.labels = {i: str(v) for i, v in enumerate(data)}
                    elif isinstance(data, dict):
                        if all(isinstance(k, str) and k.isdigit() for k in data.keys()):
                            self.labels = {int(k): str(v) for k, v in data.items()}
                        elif all(isinstance(v, int) for v in data.values()):
                            self.labels = {int(v): str(k) for k, v in data.items()}
                        else:
                            self.labels = default_labels
                    else:
                        self.labels = default_labels
            except Exception as e:
                print(f"[EmotionClassifier] Error loading labels from {self.labels_path}: {e}")
                self.labels = default_labels
        else:
            self.labels = default_labels

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[EmotionClassifier] Model file not found at {self.model_path}")
            self.model = None
            self.is_loaded = False
            return

        print(f"[EmotionClassifier] Loading CNN model from {self.model_path}...")
        loaded = False

        # 1. Thử load chuẩn qua tf.keras.models.load_model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            loaded = True
            print("[EmotionClassifier] Loaded model via standard Keras loader.")
        except Exception as e_std:
            # 2. Xử lý trường hợp file .keras từ Keras 3 (dạng zip archive) trên môi trường Keras 2
            if zipfile.is_zipfile(self.model_path):
                try:
                    z = zipfile.ZipFile(self.model_path)
                    if "model.weights.h5" in z.namelist():
                        print("[EmotionClassifier] Detected Keras 3 archive. Reconstructing Improved_CNN_GAP architecture...")
                        num_classes = len(self.labels) if len(self.labels) > 0 else 8
                        self.model = build_improved_cnn_architecture(num_classes=num_classes)

                        # Load trọng số trực tiếp từ model.weights.h5
                        h5_bytes = io.BytesIO(z.read("model.weights.h5"))
                        with h5py.File(h5_bytes, "r") as f:
                            if "layers" in f:
                                for layer in self.model.layers:
                                    if layer.name in f["layers"] and "vars" in f["layers"][layer.name]:
                                        var_group = f["layers"][layer.name]["vars"]
                                        weights = [var_group[k][()] for k in sorted(var_group.keys(), key=int)]
                                        if weights:
                                            layer.set_weights(weights)
                        loaded = True
                        print("[EmotionClassifier] Successfully loaded Keras 3 weights into local TensorFlow model.")
                except Exception as e_zip:
                    print(f"[EmotionClassifier] Failed to reconstruct Keras 3 model: {e_zip}")

        if loaded and self.model is not None:
            self.is_loaded = True
            num_outputs = self.model.output_shape[-1]
            if num_outputs != len(self.labels) or num_outputs != 8:
                print(f"[EmotionClassifier] CRITICAL MISMATCH: Model has {num_outputs} outputs but FERPlus requires 8 labels ({list(self.labels.values())}).")
            print(f"[EmotionClassifier] Model locked and ready with {num_outputs} outputs and {len(self.labels)} labels.")
        else:
            self.model = None
            self.is_loaded = False
            print(f"[EmotionClassifier] Could not load model from {self.model_path}")

    def predict_batch(
        self,
        face_tensors: List[np.ndarray],
        confidence_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, List[TopEmotion], np.ndarray]]:
        """
        Predict emotions for a batch of face tensors.
        Returns list of (predicted_emotion, confidence, top_3_list, raw_probabilities_vector).
        """
        if not self.is_loaded or self.model is None or len(face_tensors) == 0:
            return []

        threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
        batch = np.array(face_tensors) # Shape: (N, 48, 48, 1)

        try:
            predictions = self.model.predict(batch, verbose=0) # Shape: (N, 7)
        except Exception as e:
            print(f"[EmotionClassifier] Prediction error: {e}")
            return []

        results = []
        for probs in predictions:
            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])

            # Extract Top-3 emotions and calculate margin
            sorted_indices = np.argsort(probs)[::-1][:3]
            top_prob = float(probs[sorted_indices[0]])
            second_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = top_prob - second_prob

            # Apply calibrated rejection threshold & margin entropy check for "Unknown"
            if confidence < threshold or (confidence < 0.45 and margin < 0.03):
                predicted_emotion = "Unknown"
            else:
                predicted_emotion = self.labels.get(max_idx, "Unknown")

            top_3 = [
                TopEmotion(
                    emotion=self.labels.get(int(idx), f"class_{idx}"),
                    confidence=float(probs[idx])
                )
                for idx in sorted_indices
            ]

            results.append((predicted_emotion, confidence, top_3, probs))

        return results
