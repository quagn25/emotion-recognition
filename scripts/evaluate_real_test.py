"""
Script đánh giá mô hình nhận diện cảm xúc trên tập dữ liệu thực tế (real_test).
Hỗ trợ kiểm tra domain gap giữa tập dữ liệu huấn luyện FERPlus và ảnh thực tế từ webcam.
Bao gồm: Accuracy, Balanced Accuracy, Precision/Recall/F1 từng lớp, Confusion Matrix,
và Tỷ lệ không phát hiện được khuôn mặt (Undetected Face Rate).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from pathlib import Path

# Thiết lập đường dẫn gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Preload detector and torch before sklearn/matplotlib to avoid WinError 1114
from app.config import settings
from app.services.detector import FaceDetector
from app.services.preprocessing import UnifiedPreprocessor
from app.services.emotion_classifier import EmotionClassifier

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, ConfusionMatrixDisplay


def evaluate_real_test():
    real_test_dir = BASE_DIR / "real_test"
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    detector = FaceDetector(settings.model_yolo_path)
    classifier = EmotionClassifier(settings.model_cnn_path, settings.labels_path)
    preprocessor = UnifiedPreprocessor(settings.preprocessing_config_path)

    # Lấy danh sách 8 nhãn theo đúng thứ tự
    labels_dict = classifier.labels  # {0: 'angry', 1: 'contempt', ...}
    label_to_idx = {name: idx for idx, name in labels_dict.items()}
    labels_list = [labels_dict[i] for i in range(len(labels_dict))]

    image_paths = []
    y_true = []
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label_idx, label_name in labels_dict.items():
        class_dir = real_test_dir / label_name
        if not class_dir.exists():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in valid_exts:
                image_paths.append(p)
                y_true.append(label_idx)

    total_images = len(image_paths)
    if total_images == 0:
        print(f"[INFO] Thư mục '{real_test_dir}' hiện chưa có ảnh kiểm thử nào.")
        print(f"[INFO] Vui lòng thêm các ảnh khuôn mặt thu thập từ webcam vào các thư mục tương ứng:")
        for name in labels_list:
            print(f"       - real_test/{name}/")
        print("[INFO] Bạn cũng có thể dùng công cụ: python scripts/capture_real_test.py để chụp trực tiếp từ webcam.")
        return

    print(f"[*] Tìm thấy tổng cộng {total_images} ảnh kiểm thử thực tế trên {len(labels_list)} lớp cảm xúc.")
    print("[*] Đang tiến hành phát hiện khuôn mặt và suy luận...")

    y_pred = []
    detected_count = 0
    undetected_count = 0

    for idx, (img_path, true_label_idx) in enumerate(zip(image_paths, y_true)):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[-] Không thể đọc ảnh: {img_path}")
            continue

        raw_faces = detector.detect_faces(img)
        
        if len(raw_faces) == 0:
            # Fallback nếu ảnh đã là crop khuôn mặt sẵn
            roi = img
            undetected_count += 1
        else:
            detected_count += 1
            # Chọn khuôn mặt có diện tích lớn nhất
            best_face = max(raw_faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            x1, y1, x2, y2, _ = best_face
            roi, _ = preprocessor.safe_crop_face(img, x1, y1, x2, y2)
            if roi is None:
                roi = img

        processed_tensor = preprocessor.preprocess_face(roi)
        results = classifier.predict_batch([processed_tensor])
        
        if results:
            pred_label, conf, top_3, _ = results[0]
            pred_idx = label_to_idx.get(pred_label, -1)
        else:
            pred_idx = -1
            conf = 0.0

        y_pred.append((true_label_idx, pred_idx, conf))

    # Đánh giá kết quả
    correct = sum(1 for t, p, _ in y_pred if t == p)
    acc = correct / len(y_pred) if y_pred else 0.0
    undetected_rate = (undetected_count / len(y_pred) * 100.0) if y_pred else 0.0

    y_true_valid = [t for t, p, _ in y_pred if p >= 0]
    y_pred_valid = [p for t, p, _ in y_pred if p >= 0]

    balanced_acc = balanced_accuracy_score(y_true_valid, y_pred_valid) if len(y_true_valid) > 0 else 0.0
    macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0) if len(y_true_valid) > 0 else 0.0
    report_text = classification_report(
        y_true_valid, y_pred_valid,
        labels=list(range(len(labels_list))),
        target_names=labels_list,
        digits=4,
        zero_division=0
    )

    print("=" * 65)
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN DỮ LIỆU THỰC TẾ (REAL-WORLD EVALUATION):")
    print("=" * 65)
    print(f" - Tổng số ảnh xử lý        : {len(y_pred)}")
    print(f" - Khuôn mặt phát hiện      : {detected_count}")
    print(f" - Không detect được mặt    : {undetected_count} ({undetected_rate:.2f}%)")
    print(f" - Overall Accuracy         : {acc * 100:.2f}% ({correct}/{len(y_pred)})")
    print(f" - Balanced Accuracy        : {balanced_acc * 100:.2f}%")
    print(f" - Macro-F1 Score           : {macro_f1 * 100:.2f}%")
    print("=" * 65)
    print("\nChi tiết Classification Report:\n", report_text)

    # Xuất báo cáo văn bản
    report_file = reports_dir / "real_test_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO ĐÁNH GIÁ TRÊN DỮ LIỆU WEBCAM THỰC TẾ (REAL_TEST)\n")
        f.write("=" * 65 + "\n")
        f.write(f"Tổng số ảnh kiểm thử        : {len(y_pred)}\n")
        f.write(f"Khuôn mặt phát hiện được    : {detected_count}\n")
        f.write(f"Tỷ lệ không phát hiện mặt   : {undetected_rate:.2f}% ({undetected_count}/{len(y_pred)})\n")
        f.write(f"Độ chính xác (Accuracy)     : {acc * 100:.2f}%\n")
        f.write(f"Balanced Accuracy           : {balanced_acc * 100:.2f}%\n")
        f.write(f"Macro-F1 Score              : {macro_f1 * 100:.2f}%\n\n")
        f.write("Chi tiết Classification Report:\n")
        f.write(report_text + "\n")

    # Xuất JSON số liệu
    metrics_file = reports_dir / "real_test_metrics.json"
    metrics_data = {
        "total_samples": len(y_pred),
        "detected_faces": detected_count,
        "undetected_faces": undetected_count,
        "undetected_rate_percent": round(undetected_rate, 2),
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4)
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Xuất Confusion Matrix Image
    if len(y_true_valid) > 0:
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(len(labels_list))))
        cm_norm = np.zeros_like(cm, dtype=float)
        for r_idx in range(len(cm)):
            row_sum = np.sum(cm[r_idx])
            if row_sum > 0:
                cm_norm[r_idx] = cm[r_idx] / float(row_sum)

        fig, ax = plt.subplots(figsize=(9, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels_list)
        disp.plot(cmap='Blues', ax=ax, values_format='.2f', colorbar=True)
        plt.title(f"Real-World Test Confusion Matrix (Acc: {acc * 100:.2f}%)")
        plt.xlabel("Predicted Emotion")
        plt.ylabel("True Emotion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        cm_file = reports_dir / "real_test_confusion_matrix.png"
        plt.savefig(cm_file, dpi=200)
        plt.close()
        print(f"[+] Confusion matrix đã lưu vào: {cm_file}")

    print(f"[+] Báo cáo chi tiết đã lưu vào: {report_file}")
    print(f"[+] Dữ liệu số liệu JSON đã lưu vào: {metrics_file}")


if __name__ == "__main__":
    evaluate_real_test()
