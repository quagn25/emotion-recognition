# Đề Tài: Ứng Dụng Mạng Nơ Ron Tích Chập Trong Nhận Dạng Biểu Cảm (Facial Emotion Recognition)

Dự án Deep Learning phân loại cảm xúc khuôn mặt thời gian thực từ Webcam, Ảnh tĩnh và Video. Hệ thống kết hợp mạng phát hiện khuôn mặt **YOLOv8-Face**, pipeline tiền xử lý cân bằng độ tương phản cục bộ thích ứng **CLAHE (Contrast Limited Adaptive Histogram Equalization)**, mô hình phân loại **Improved CNN (GlobalAveragePooling2D + L2 Regularization)** tối ưu trên bộ dữ liệu **FERPlus (8 cảm xúc)**, và bộ làm mịn temporal đa phiên (`SessionSmootherManager`) với cơ chế gán định danh khuôn mặt (`track_id`).

---

## 1. Kết Quả Huấn Luyện & Đánh Giá Mô Hình

Mô hình được huấn luyện và đánh giá độc lập trên tập kiểm thử **FERPlus Test Set** gồm **8.341 ảnh** (bộ nhãn được gán lại bởi Microsoft Research từ FER2013 với 10 chuyên gia gán nhãn cho mỗi ảnh).

### Chỉ số tổng quan (Overall Performance)

| Chỉ số (Metric) | Kết quả (%) | Ghi chú |
| :--- | :---: | :--- |
| **Test Accuracy** | **70.76%** | Tỷ lệ dự đoán chính xác toàn bộ 8.341 ảnh test |
| **Balanced Accuracy** | **70.14%** | Đánh giá công bằng qua các lớp mất cân bằng dữ liệu |
| **Macro-F1 Score** | **69.47%** | Trung bình F1 không phụ thuộc vào kích thước lớp |
| **Weighted-F1 Score** | **69.84%** | F1 có trọng số theo phân phối mẫu |
| **Macro-Precision** | **73.16%** | Độ chính xác dự đoán trung bình |
| **Macro-Recall** | **70.14%** | Độ bao phủ trung bình |

### Đánh giá chi tiết từng lớp biểu cảm (Per-class Metrics)

| Lớp cảm xúc (Emotion) | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Angry** (Tức giận) | 58.84% | 77.90% | 67.04% | 1,000 |
| **Contempt** (Khinh khi) | **85.38%** | 33.30% | 47.91% | 1,000 |
| **Disgust** (Ghê tởm) | 76.55% | 55.50% | 64.35% | 1,000 |
| **Fear** (Sợ hãi) | 82.89% | 67.80% | 74.59% | 1,000 |
| **Happy** (Vui vẻ) | 84.89% | **86.50%** | **85.69%** | 1,000 |
| **Neutral** (Trung tính) | 72.68% | 85.31% | 78.49% | 1,341 |
| **Sad** (Buồn rầu) | 56.29% | 68.50% | 61.80% | 1,000 |
| **Surprise** (Ngạc nhiên) | 67.74% | **86.30%** | 75.90% | 1,000 |

### Phân tích các lớp yếu & Domain Gap
- **Contempt (Khinh khi):** Đạt Precision rất cao (85.38%) nhưng Recall thấp (33.30%). Biểu cảm khinh khi đặc trưng bởi các biến dạng bất đối xứng vi mô (nhếch một bên mép), rất dễ bị suy giảm khi giảm độ phân giải xuống 48×48 pixel.
- **Disgust (Ghê tởm):** Recall đạt 55.50%. Các biểu cảm chun mũi, nhăn trán dễ bị nhầm lẫn sang `Angry`.
- **Sad (Buồn bã):** F1-score đạt 61.80%, dễ bị nhầm lẫn với trạng thái `Neutral` khi cường độ biểu cảm ở mức độ nhẹ.
- **Happy & Surprise:** Hai lớp có đặc trưng hình thái nổi bật nhất (khẩu hình mở rộng, mắt mở to, nụ cười rõ nét) đạt hiệu năng cao nhất (> 85% F1 và Recall).

*Các biểu đồ và báo cáo chi tiết được lưu trữ trong thư mục `reports/`:*
- `reports/metrics.json`: Dữ liệu số liệu phân tích định lượng đầy đủ.
- `reports/classification_report.txt`: Báo cáo phân loại text chi tiết.
- `reports/training_curves.png`: Đồ thị Loss và Accuracy trong quá trình huấn luyện.
- `reports/confusion_matrix.png`: Ma trận nhầm lẫn (Confusion Matrix).
- `reports/error_samples.png`: Phân tích các mẫu dự đoán sai tiêu biểu.

---

## 2. Gói Mô Hình Chuẩn Hóa (Locked Model Package)

Ứng dụng khóa cố định bộ tệp mô hình và cấu hình trong thư mục `models/`:

```text
models/
├── emotion_model.keras          # Mô hình Improved CNN (8 output logits)
├── labels.json                  # Ánh xạ chuẩn 8 lớp FERPlus (0: angry ... 7: surprise)
├── preprocessing_config.json    # Cấu hình tiền xử lý đồng bộ (CLAHE tile 8x8, padding 15%)
└── yolov8n-face.pt              # Mô hình YOLOv8 Nano Face Detection
```

---

## 3. Cấu Trúc Thư Mục Dự Án

```text
ProjectDL/
├── app/
│   ├── config.py                 # Cấu hình hệ thống (DLL preloading, AppConfig)
│   ├── schemas.py                # Pydantic Schemas cho API
│   └── services/
│       ├── detector.py           # YOLOv8-Face Detection (downscale optimization)
│       ├── emotion_classifier.py # Improved CNN (8 lớp) + Margin-based Unknown filter
│       ├── losses.py             # Categorical Focal Loss & Class Weight calculation
│       ├── model_factory.py      # Baseline CNN, Improved CNN, MobileNetV3 architectures
│       ├── preprocessing.py      # Pipeline Safe Square Crop + CLAHE
│       ├── smoothing.py          # Temporal Smoothing & Session Manager (LRU max 100)
│       └── video_service.py      # Video processor & track assignment
├── configs/
│   └── train.yaml                # Cấu hình siêu tham số tập trung & Ablation study
├── models/                       # Gói mô hình đã khóa
├── reports/                      # Báo cáo đánh giá và biểu đồ hiệu năng
├── real_test/                    # Tập dữ liệu kiểm thử webcam thực tế (8 lớp cảm xúc)
├── scripts/
│   ├── capture_real_test.py     # Công cụ chụp mẫu trực tiếp từ webcam
│   └── evaluate_real_test.py    # Script đánh giá đa chiều trên real_test
├── tests/                        # Bộ kiểm thử tự động pytest (22 test cases, 100% Pass)
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_robustness.py        # Edge-cases: multi-face, occluded, tiny, corrupt video
│   └── test_video.py
├── index.html                    # Giao diện web đề tài "Ứng dụng mạng nơ ron tích chập..."
├── main.py                       # FastAPI application & REST endpoints
├── train.py                      # CLI training & ablation script
├── evaluate.py                   # CLI model evaluation script
├── inference.py                  # CLI image/webcam inference script
├── run.ps1                       # Script khởi động 1-click cho Windows
├── run.sh                        # Script khởi động 1-click cho Linux/macOS
├── project_emotions.ipynb        # Jupyter Notebook nghiên cứu & minh họa
└── requirements.txt              # Danh sách thư viện phụ thuộc
```

---

## 4. Hướng Dẫn Khởi Chạy 1-Click

### Khởi động nhanh
```powershell
# Windows PowerShell
.\run.ps1
```
```bash
# Linux / macOS
chmod +x run.sh && ./run.sh
```

Hoặc chạy thủ công qua lệnh:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Truy cập giao diện: [http://localhost:8000](http://localhost:8000).

---

## 5. Pipeline CLI (Huấn Luyện, Đánh Giá, Suy Luận)

### 1. Huấn luyện mô hình & Thực nghiệm bóc tách (Ablation Study)
```powershell
# Chạy với cấu hình mặc định trong configs/train.yaml
python train.py

# Huấn luyện thử nghiệm Baseline CNN không dùng CLAHE
python train.py --model baseline_cnn --no-use-clahe --epochs 30

# Huấn luyện MobileNetV3 với Focal Loss
python train.py --model mobilenetv3 --loss focal_loss --epochs 40
```

### 2. Đánh giá mô hình CLI
```powershell
python evaluate.py --model models/emotion_model.keras --test-dir test/
```

### 3. Suy luận trực tiếp CLI
```powershell
# Nhận diện trên ảnh tĩnh
python inference.py --source test_image.jpg --output output.jpg

# Mở webcam trực tiếp qua OpenCV
python inference.py --webcam 0
```

---

## 6. Đánh Giá Trên Dữ Liệu Thực Tế (`real_test/`)

1. **Thu thập ảnh từ webcam:**
   ```powershell
   python scripts/capture_real_test.py
   ```
   Nhấn phím `0`–`7` tương ứng với 8 cảm xúc để chụp nhanh khuôn mặt từ webcam vào `real_test/<emotion>/`.

2. **Chạy đánh giá toàn diện:**
   ```powershell
   python scripts/evaluate_real_test.py
   ```
   Hệ thống sẽ tự động tính toán:
   - Overall Accuracy & Balanced Accuracy.
   - Precision, Recall, F1-Score từng lớp.
   - Tỷ lệ không phát hiện được khuôn mặt (Undetected Face Rate).
   - Xuất biểu đồ ma trận nhầm lẫn: `reports/real_test_confusion_matrix.png`.
   - Xuất báo cáo văn bản: `reports/real_test_report.txt` và `reports/real_test_metrics.json`.

---

## 7. Chạy Kiểm Thử Tự Động (Pytest Suite)

Chạy toàn bộ 22 bài kiểm thử (bao gồm kiểm thử biên và kiểm thử bảo mật):
```powershell
pytest tests/ -v
```

Kết quả: **`22 passed (100% GREEN)`**:
- Kiểm tra kết nối Health check & cấu hình 8 nhãn `/model-info`.
- Kiểm tra giải mã an toàn Base64 và xử lý lỗi chuỗi độc hại.
- Kiểm tra tính toán Bounding Box tại sát các mép viền ảnh.
- Kiểm tra định dạng đầu ra tensor và dải giá trị sau CLAHE.
- Kiểm tra hợp đồng đầu ra của mô hình (8 logits softmax).
- Kiểm tra loại bỏ khuôn mặt quá nhỏ (< 20px).
- Kiểm tra từ chối video hỏng và video 0-frame.
- Kiểm tra cách ly hoàn toàn giữa 2 session webcam đồng thời.
- Kiểm tra giới hạn dung lượng LRU (tối đa 100 sessions, chống rò rỉ bộ nhớ).
- Kiểm tra bảo vệ chống tấn công Path Traversal tại `/videos/{filename}`.
- Kiểm tra tính toàn vẹn của tệp video xuất ra sau khi xử lý.

---

## 8. Tài Liệu REST API

| Method | Endpoint | Mô tả | Tham số chính |
| :---: | :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra tình trạng backend và thiết bị (CPU/GPU) | Không |
| `GET` | `/model-info` | Lấy siêu dữ liệu mô hình, danh sách 8 nhãn và kích thước input | Không |
| `POST` | `/predict/frame` | Nhận diện cảm xúc theo từng frame webcam thời gian thực | `image_base64`, `session_id`, `model_type` |
| `POST` | `/session/reset` | Xóa bộ nhớ đệm làm mịn khi kết thúc hoặc chuyển phiên webcam | `session_id` |
| `POST` | `/predict/image` | Tải lên ảnh tĩnh (JPG/PNG), phân tích và trả về ảnh đã vẽ box | `file` (Multipart form-data) |
| `POST` | `/predict/video` | Tải lên video (MP4/AVI), xử lý offline và trả về video đã gán nhãn | `file` (Multipart form-data) |
| `GET` | `/videos/{filename}` | Tải hoặc phát trực tiếp video đã xử lý (được bảo vệ Path Traversal) | `filename` (URL path parameter) |
