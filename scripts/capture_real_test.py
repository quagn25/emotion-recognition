"""
Công cụ thu thập mẫu ảnh webcam thực tế cho 8 lớp cảm xúc.
Giúp người dùng nhanh chóng tạo tập dữ liệu kiểm thử real_test từ webcam máy tính.
"""

import os
import sys
import time
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings
from app.services.detector import FaceDetector


def main():
    detector = FaceDetector()
    labels = settings.EMOTION_LABELS  # ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    real_test_dir = BASE_DIR / "real_test"
    for label in labels:
        (real_test_dir / label).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" CÔNG CỤ THU THẬP ẢNH WEBCAM THỰC TẾ (REAL_TEST CAPTURE)")
    print("=" * 65)
    print(" Hướng dẫn phím bấm:")
    for idx, name in enumerate(labels):
        print(f"  [{idx}] : Lưu ảnh cho cảm xúc '{name.upper()}'")
    print("  [q] : Thoát chương trình")
    print("=" * 65)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Không thể mở Webcam. Vui lòng kiểm tra quyền truy cập camera.")
        return

    saved_counts = {name: len(list((real_test_dir / name).glob("*.jpg"))) for name in labels}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        raw_faces = detector.detect_faces(frame)

        # Draw detected face boxes
        for (x1, y1, x2, y2, conf) in raw_faces:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Face: {int(conf*100)}%", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Status text
        cv2.putText(display_frame, "Nhan phim [0-7] de luu cam xuc, [q] de thoat", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset = 55
        for idx, name in enumerate(labels):
            cnt = saved_counts[name]
            cv2.putText(display_frame, f"[{idx}] {name}: {cnt}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            y_offset += 20

        cv2.imshow("Real-Test Webcam Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if ord('0') <= key <= ord(str(len(labels) - 1)):
            cls_idx = key - ord('0')
            cls_name = labels[cls_idx]
            
            timestamp = int(time.time() * 1000)
            filename = f"webcam_{cls_name}_{timestamp}.jpg"
            save_path = real_test_dir / cls_name / filename
            
            # Save original frame
            cv2.imwrite(str(save_path), frame)
            saved_counts[cls_name] += 1
            print(f"[+] Đã lưu: {save_path.name} -> real_test/{cls_name}/ (Tổng: {saved_counts[cls_name]})")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Đã đóng công cụ thu thập ảnh.")


if __name__ == "__main__":
    main()
