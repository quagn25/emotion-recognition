# Real-World Webcam Evaluation Protocol (Tập Kiểm Thử Thực Tế)

Thư mục này được thiết kế để thu thập và kiểm thử chất lượng mô hình nhận diện cảm xúc khuôn mặt trên dữ liệu ảnh thực tế thu được từ webcam / môi trường thực tế (domain gap assessment).

## 1. Cấu trúc thư mục

```text
real_test/
├── README.md
├── angry/      # Khuôn mặt biểu cảm tức giận
├── contempt/   # Khuôn mặt biểu cảm khinh khi / nhếch mép
├── disgust/    # Khuôn mặt biểu cảm ghê tởm / nhăn mũi
├── fear/       # Khuôn mặt biểu cảm sợ hãi
├── happy/      # Khuôn mặt biểu cảm vui vẻ / cười
├── neutral/    # Khuôn mặt trạng thái trung tính / bình thường
├── sad/        # Khuôn mặt biểu cảm buồn rầu
└── surprise/   # Khuôn mặt biểu cảm ngạc nhiên
```

## 2. Tiêu chuẩn thu thập ảnh (300 - 1.000 ảnh)

Để kiểm tra chính xác khả năng tổng quát hóa và đo lường khoảng cách miền dữ liệu (Domain Gap) giữa dữ liệu FERPlus (ảnh xám 48x48) và webcam thực tế:

1. **Điều kiện ánh sáng:**
   - Đủ sáng tự nhiên hoặc đèn huỳnh quang phòng học / làm việc.
   - Thử nghiệm cả trường hợp ngược sáng nhẹ hoặc ánh sáng vàng / yếu.
2. **Góc chụp & Khoảng cách:**
   - Chụp trực diện (Frontal, yaw/pitch < 15°).
   - Chụp nghiêng nhẹ (yaw 15° - 30°).
   - Khoảng cách gần (mặt chiếm 50% khung hình) và xa (ngồi cách webcam 1m - 1.5m).
3. **Đối tượng thực tế:**
   - Khuôn mặt người Việt Nam / Châu Á (giúp đánh giá bias chủng tộc so với tập dữ liệu FERPlus gốc).
   - Có thể có kính mắt, tóc mái che một phần trán.
4. **Định dạng file:**
   - Hỗ trợ `.jpg`, `.jpeg`, `.png`.
   - Ảnh gốc chụp từ webcam (VD: 640x480, 1280x720) — pipeline hệ thống sẽ tự động detect khuôn mặt bằng YOLOv8-Face, square crop + 15% padding, CLAHE, và resize về 48x48.

## 3. Chạy đánh giá tự động

Sử dụng script đánh giá tự động:

```bash
python scripts/evaluate_real_test.py
```

Kết quả đánh giá (Accuracy, Confusion Matrix, Classification Report trên tập thực tế) sẽ được tự động xuất ra `reports/real_test_report.txt` và `reports/real_test_confusion_matrix.png`.
